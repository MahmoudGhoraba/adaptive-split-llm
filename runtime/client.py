"""
Client side of the split runtime: embedding, head layers 0..H-1, tail layers N-T..N-1, output layer (tied
embedding), optional secret adapters. Keeps its own KV caches (head and tail); the server keeps the middle one.

Greedy generation, one round trip per step: prefill sends P x d values up and back, each decode step 1 x d.
Every step is timed (time.perf_counter_ns, after device synchronisation):
  head_s     embedding + head layers (+ device sync)
  ser_s      cast to the wire dtype (fp16) + device -> host copy + serialisation of the hidden states
  rtt_s      send start -> reply fully received
  server_compute_s / server_total_s   reported by the server (layers / request handling)
  net_s      rtt_s - server_total_s   (transport + network + both kernels' socket handling; no shared clock)
  tail_s     host -> device copy, tail layers, output layer, argmax (.item() synchronises)
             = tail_layers_s (host -> device copy + tail layers, synchronised) + out_s (final norm, output layer, argmax)
  step_s     whole step on the client
"""
import socket
import time
import types

import torch

import partial_model as pm
import protocol as pr
from netem import precise_sleep


class SplitClient:
    def __init__(self, host, port, H, T, device="mps", dtype=torch.float16, mdir=None, load="cpu", netem=None,
                 wire_dtype=torch.float16):
        self.mdir = mdir or pm.model_dir()
        self.n_layers = pm.load_config(self.mdir).num_hidden_layers
        self.H, self.T, self.device, self.wire_dtype = H, T, device, wire_dtype     # hidden states sent as fp16
        self.m = pm.PartialLlama(self.mdir, pm.split_segments(H, T, self.n_layers)[0], True, device, dtype, load=load)
        self.conn = pr.Conn(socket.create_connection((host, port)), netem)     # netem: emulated UPLINK
        self.conn.send(pr.json_msg(pr.HELLO, dict(H=H, T=T, model_id=pm.MODEL_ID, dtype=str(dtype))))
        rep = self.conn.recv()
        if rep.type != pr.HELLO:
            raise RuntimeError(f"server refused: {bytes(rep.payload).decode()}")
        self.server_info = rep.json()
        self.session = 0

    @classmethod
    def from_parts(cls, conn, m, H, T, device="cpu", wire_dtype=torch.float16):
        """A SplitClient over an existing connection and an already loaded client PartialLlama (reconfigurable runtime)."""
        c = cls.__new__(cls)
        c.conn, c.m, c.H, c.T, c.device, c.wire_dtype = conn, m, H, T, device, wire_dtype
        c.n_layers, c.session, c.server_info = m.cfg.num_hidden_layers, 0, {}
        return c

    # -------------------------------------------------------------------- adapters (client only)
    def add_secret_adapters(self, head_seed, tail_seed):
        mods = self.m.add_secret_adapters(range(0, self.H), head_seed)
        if self.T:
            mods += self.m.add_secret_adapters(range(self.n_layers - self.T, self.n_layers), tail_seed)
        return mods

    def remove_adapters(self):
        pm.su.remove_all_adapters(types.SimpleNamespace(model=types.SimpleNamespace(
            layers=list(self.m.global_layers.values()))))

    # -------------------------------------------------------------------- control messages
    def _call(self, msg, expect):
        self.conn.send(msg); rep = self.conn.recv()
        if rep.type != expect:
            raise RuntimeError(f"expected {pr.TYPE_NAME[expect]}, got {pr.TYPE_NAME[rep.type]}: {bytes(rep.payload)[:300]}")
        return rep

    def reset(self):
        self.session += 1
        self._call(pr.Msg(pr.RESET, session=self.session), pr.RESET)

    def ping(self, n_bytes=0):
        t0 = time.perf_counter_ns(); self._call(pr.Msg(pr.PING, bytes(n_bytes)), pr.PING)
        return (time.perf_counter_ns() - t0) / 1e9

    def server_stats(self):
        return self._call(pr.Msg(pr.STATS), pr.STATS).json()

    def close(self):
        try:
            self.conn.send(pr.Msg(pr.BYE))
        finally:
            self.conn.close()

    # -------------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, ids, max_new_tokens, eos=(), capture=False, gap_s=0.0):
        """Greedy decoding of ids [1, P]. Returns tokens, logit margins (top-1 - top-2) and one timing row per step.
        eos=() -> exactly max_new_tokens steps (fixed answer length A, used for timing runs).
        capture=True also returns the boundary states (outputs of layers H, N-T, N) on the CPU, for correctness.
        gap_s: idle time before every DECODE message (not counted in any phase; downclock check)."""
        self.reset()
        bH, bM, bN = [], [], []
        caches, x, pos, toks, margins, steps = self.m.new_caches(), ids, 0, [], [], []
        ns = time.perf_counter_ns
        for step in range(max_new_tokens):
            t0 = ns()
            h = self.m.run("head", self.m.embed_ids(x), pos, caches)
            pm.sync(self.device); t1 = ns()
            req = pr.tensor_msg(pr.PREFILL if step == 0 else pr.DECODE, h.to(self.wire_dtype), session=self.session, step=step,
                                pos_start=pos)
            t2 = ns()
            if gap_s and step > 0:
                precise_sleep(gap_s); t2g = ns()
            else:
                t2g = t2
            up = self.conn.send(req); rep = self.conn.recv(); t3 = ns()
            gap = t2g - t2                               # rtt starts after the gap; step_s excludes it
            if rep.type == pr.ERROR:
                raise RuntimeError(f"server error at step {step}: {bytes(rep.payload).decode()}")
            h = rep.tensor()
            if capture:
                bH.append(req.tensor()[0].clone()); bM.append(h[0].clone())
            if self.T:
                h = self.m.run("tail", h, pos, caches)
            if capture:
                bN.append(h[0].to("cpu"))
            pm.sync(self.device); t3b = ns()
            lg = self.m.logits(h[:, -1:])[0, -1]
            top = lg.float().topk(2)
            nxt = int(top.indices[0]); t4 = ns()
            toks.append(nxt); margins.append(float(top.values[0] - top.values[1]))
            steps.append(dict(step=step, kind="prefill" if step == 0 else "decode", n_tokens=x.shape[1], pos_start=pos,
                              head_s=(t1 - t0) / 1e9, ser_s=(t2 - t1) / 1e9, rtt_s=(t3 - t2g) / 1e9,
                              server_compute_s=rep.server_compute_ns / 1e9, server_total_s=rep.server_total_ns / 1e9,
                              net_s=(t3 - t2g - rep.server_total_ns) / 1e9, tail_s=(t4 - t3) / 1e9,
                              tail_layers_s=(t3b - t3) / 1e9, out_s=(t4 - t3b) / 1e9,
                              step_s=(t4 - t0 - gap) / 1e9, bytes_up=up, bytes_down=pr.HEADER.size + len(rep.payload)))
            pos += x.shape[1]
            if nxt in eos:
                break
            x = torch.tensor([[nxt]])
        out = dict(tokens=toks, margins=margins, steps=steps)
        if capture:
            out.update(bH=torch.cat(bH), bM=torch.cat(bM), bN=torch.cat(bN))
        return out
