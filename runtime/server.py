"""
Server side of the split runtime: the U = N - H - T middle layers (public weights, no adapters, no embedding,
no output layer). Keeps its own KV cache; every PREFILL/DECODE must start at the position its cache holds.

Run:  python server.py --H 4 --T 2 [--port 0] [--device mps]
      python server.py --reconfigurable [--H .. --T ..]     (task 4: the client chooses the split per request)
Prints "READY <port>" once the layers are loaded and the socket listens. Serves one client connection at a time.

Reconfigurable mode (one configuration in memory at a time, so client + server fit in 8 GB):
  HELLO {H, T}        H + T = N: all on client -> the server frees its layers; H = T = 0: all on server -> loads
                      embedding + all layers + output layer; otherwise loads layers H .. N-T-1. Unchanged (H, T):
                      nothing is reloaded. Reply carries load_s (reload time, reported separately as switching cost).
  HELLO {..., netem}  {rate_mbit, delay_ms} or null: sets the emulated DOWNLINK of this connection (emulation
                      control for experiments, not part of a real protocol)
  GENERATE {ids, A}   all on server: greedy A tokens on the full model; reply {tokens}, server_compute_ns
  PING b"probe"       server-speed probe: a fixed small kernel ((64x2048)@(2048x2048) fp16, x8; 8 MiB weights, below
                      the 32 MiB size that costs a 1 GiB MPS heap) timed on the server; reply JSON {probe_s, ref_s},
                      ref_s = the same probe at server start (idle, warm). The client estimates share = ref_s / probe_s.
                      NOT RELIABLE on Apple MPS (measured 14 Nov): after idle gaps the GPU power state makes the probe
                      2-7x slower with no load (share estimates 0.14-0.62 for an idle server); kept for documentation.
  PING b"load"        server-REPORTED load: reply JSON {share} (the server's current free share; here the emulated
                      value set by HELLO {load_share}, default 1.0). Default estimator input (task 4 follow-up).
"""
import argparse
import gc
import socket
import sys
import time

import torch

import memstat
import partial_model as pm
import protocol as pr
from netem import NetEm

DTYPE = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


class State:
    def __init__(self, a, mdir):
        self.a, self.mdir, self.m, self.HT, self.load_share = a, mdir, None, None, 1.0
        self.n = pm.load_config(mdir).num_hidden_layers
        g = torch.Generator().manual_seed(0)
        self.pw = torch.randn(2048, 2048, generator=g).half().to(a.device) * 0.02
        self.px = torch.randn(64, 2048, generator=g).half().to(a.device)
        for _ in range(3):
            self.probe()
        self.probe_ref = min(self.probe() for _ in range(5))

    def probe(self):
        pm.sync(self.a.device); t0 = time.perf_counter()
        y = self.px
        for _ in range(8):
            y = y @ self.pw
        pm.sync(self.a.device)
        return time.perf_counter() - t0

    def load(self, H, T):
        """Load the layers for (H, T); returns the load time [s] (0 if unchanged)."""
        if (H, T) == self.HT:
            return 0.0
        t0 = time.perf_counter()
        self.m = None; gc.collect()
        if self.a.device == "mps":
            torch.mps.empty_cache()
        if H == 0 and T == 0:
            self.m = pm.PartialLlama(self.mdir, dict(all=(0, self.n)), True, self.a.device, DTYPE[self.a.dtype], load=self.a.load)
        elif H + T < self.n:
            self.m = pm.PartialLlama(self.mdir, pm.split_segments(H, T, self.n)[1], False, self.a.device,
                                     DTYPE[self.a.dtype], load=self.a.load)
        pm.sync(self.a.device)
        self.HT = (H, T)
        return time.perf_counter() - t0

    def info(self):
        if self.m is None:
            return dict(U=0, layers=None)
        seg = next(iter(self.m.segs.values()))
        return dict(U=seg.stop - seg.start, layers=[seg.start, seg.stop], device=str(self.m.device), dtype=str(self.m.dtype))


def serve_conn(conn, st, reconfigurable):
    caches = st.m.new_caches() if st.m else None
    while True:
        try:
            req = conn.recv()
        except (ConnectionError, OSError):
            return
        t0 = time.perf_counter_ns()
        if req.type in (pr.PREFILL, pr.DECODE):
            try:
                tc0 = time.perf_counter_ns()
                out = st.m.run("mid", req.tensor(), req.pos_start, caches)
                out = out.to("cpu")                                      # includes the device synchronisation
                tc1 = time.perf_counter_ns()
                rep = pr.tensor_msg(req.type, out, session=req.session, step=req.step, pos_start=req.pos_start)
                rep.server_compute_ns, rep.server_total_ns = tc1 - tc0, time.perf_counter_ns() - t0
            except AssertionError as e:                                  # position mismatch: report, keep serving
                rep = pr.Msg(pr.ERROR, str(e).encode(), session=req.session, step=req.step)
            conn.send(rep)
        elif req.type == pr.HELLO:
            j = req.json()
            HT = (j.get("H"), j.get("T"))
            if "load_share" in j:
                st.load_share = float(j["load_share"])
            if "netem" in j:
                nm = j["netem"]
                conn.netem = NetEm(nm["rate_mbit"] * 1e6, nm["delay_ms"] / 1e3, st.a.burst) if nm else None
            if HT != st.HT and not reconfigurable:
                conn.send(pr.Msg(pr.ERROR, f"server holds layers for H={st.HT[0]}, T={st.HT[1]}, client asked {j}".encode()))
                continue
            load_s = st.load(*HT)
            caches = st.m.new_caches() if st.m else None
            conn.send(pr.json_msg(pr.HELLO, dict(st.info(), load_s=load_s)))
        elif req.type == pr.GENERATE:
            j = req.json()
            tc0 = time.perf_counter_ns()
            g = pm.generate_local(st.m, torch.tensor([j["ids"]]), j["A"], st.a.device)
            tc1 = time.perf_counter_ns()
            rep = pr.json_msg(pr.GENERATE, dict(tokens=g["tokens"]))
            rep.server_compute_ns, rep.server_total_ns = tc1 - tc0, time.perf_counter_ns() - t0
            conn.send(rep)
        elif req.type == pr.RESET:
            caches = st.m.new_caches() if st.m else None
            conn.send(pr.Msg(pr.RESET, session=req.session))
        elif req.type == pr.PING:
            if bytes(req.payload) == b"load":
                conn.send(pr.json_msg(pr.PING, dict(share=st.load_share), step=req.step))
            elif bytes(req.payload) == b"probe":
                conn.send(pr.json_msg(pr.PING, dict(probe_s=st.probe(), ref_s=st.probe_ref), step=req.step))
            else:
                conn.send(pr.Msg(pr.PING, req.payload, step=req.step))
        elif req.type == pr.STATS:
            conn.send(pr.json_msg(pr.STATS, memstat.snapshot()))
        elif req.type == pr.BYE:
            return
        else:
            conn.send(pr.Msg(pr.ERROR, f"unexpected message type {req.type}".encode()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int); ap.add_argument("--T", type=int)
    ap.add_argument("--reconfigurable", action="store_true")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--device", default="mps"); ap.add_argument("--dtype", default="fp16", choices=DTYPE)
    ap.add_argument("--load", default="cpu"); ap.add_argument("--model-dir")
    ap.add_argument("--rate-mbit", type=float, help="emulated DOWNLINK rate (application level); off if not given")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="emulated one-way delay = RTT / 2")
    ap.add_argument("--burst", type=int, default=0)
    a = ap.parse_args()
    if not a.reconfigurable and (a.H is None or a.T is None):
        ap.error("--H and --T are required unless --reconfigurable")
    st = State(a, a.model_dir or pm.model_dir())
    if a.H is not None:
        st.load(a.H, a.T)
    srv = socket.create_server((a.host, a.port))
    print(f"READY {srv.getsockname()[1]}", flush=True)
    while True:
        sock, _ = srv.accept()
        net = NetEm(a.rate_mbit * 1e6, a.delay_ms / 1e3, a.burst) if a.rate_mbit else None
        conn = pr.Conn(sock, net)
        serve_conn(conn, st, a.reconfigurable)
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
