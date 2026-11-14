"""
Task 4, item 5: a short mobile trace END-TO-END through the real runtime (MEASURED), for the adaptive policy
(calibrated model + online estimates) and the fixed-per-class policy; measured vs predicted (calibrated model).

Trace: make_stream(30, mobile) of step 5 (Wi-Fi -> 4G -> poor -> 4G -> Wi-Fi, 6 requests each, +-20% jitter,
downlink = 2.5 x uplink), prompt lengths mapped to the nearest bucket in {32, 128, 256, 512} (timing protocol),
A = 64 fixed (EOS ignored), synthetic prompt ids. Network: application-level emulation, set per request on both
sides (client uplink, server downlink via HELLO). Client = M2-4T profile (CPU fp16, 4 threads), server = MPS fp16.
Reconfigurable runtime: only the chosen configuration is in memory; reload time (client and server) is measured
and reported SEPARATELY as switching cost (predicted by calibrated_model.transition). Control messages (HELLO, RESET)
are excluded from the request latency (the PING of the estimating policies is counted). Every request is tagged cold (first use of this configuration and
prompt length in the run: includes MPS kernel compilation) or warm.

Policies: adaptive (calibrated model + online estimates), switch_aware (also counts the predicted reload cost
from the resident configuration, setting a = this testbed; myopic), fixed_per_class (chosen once at nominal 4G).
Default estimator (adaptive, switch_aware): one PING b"load" per request -> RTT + server-REPORTED load (emulated
server share, 1.0 here); effective rate from the prefill message (net time - estimated RTT - fitted overhead). The
PING time is COUNTED in the request latency of these policies (fixed_per_class sends no PING).
--repeats R runs every policy R times, interleaved (A, S, F, A, S, F, ...), each run with a fresh server.
Secret adapters are the untrained ones (seeds 2000/2001): latency only; leakage/quality come from the tables.

Run:  python run_trace_real.py [--n 30]   ->  check_out/task4_real_trace.csv, task4_real_trace.json
"""
import argparse
import csv
import gc
import json
import os
import socket
import statistics as st
import subprocess
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.join(HERE, "..", "step5_per_request")
sys.path.insert(0, HERE); sys.path.insert(0, STEP5)
import memstat
import partial_model as pm
import protocol as pr
from check_partial import env_info
from client import SplitClient
from netem import NetEm
import per_request_controller as prc
prc.load_multisplit(os.path.join(STEP5, "measured_multisplit.txt"))
import calibrated_model as cm                                                   # noqa: E402

OUT = os.path.join(HERE, "check_out")
PROF, A, BUCKETS = "M2-4T", 64, [32, 128, 256, 512]
OPTS = [o for o in prc.options() if o["kind"] != "noise"]        # noise: never feasible for the classes; not in runtime
CL = {c["key"]: c for c in prc.CLASSES}
NOMINAL = dict(rate_up=prc.PHASES["lte"][1], rate_down=2.5 * prc.PHASES["lte"][1], rtt=prc.PHASES["lte"][2], server_share=1.0)
N = 16


def key_of(o):
    k = o["kind"]
    if k == "local":
        return ("local",)
    if k == "cloud":
        return ("cloud",)
    return ("split", o["H"], o["T"], k)


class ReconfigClient:
    def __init__(self, port):
        self.conn = pr.Conn(socket.create_connection(("127.0.0.1", port)))
        self.mdir = pm.model_dir(); self.m = None; self.layers = None; self.key = None; self.used = set()

    def _call(self, msg, expect):
        self.conn.send(msg); rep = self.conn.recv()
        if rep.type != expect:
            raise RuntimeError(f"expected {pr.TYPE_NAME[expect]}, got {pr.TYPE_NAME[rep.type]}: {bytes(rep.payload)[:300]}")
        return rep

    def set_network(self, cond):
        self.conn.netem = NetEm(cond["rate_up"], cond["rtt"] / 2)
        return dict(rate_mbit=cond["rate_down"] / 1e6, delay_ms=cond["rtt"] / 2 * 1e3)

    def configure(self, o, cond):
        k = key_of(o)
        layers = ("all",) if k[0] == "local" else None if k[0] == "cloud" else ("split", k[1], k[2])
        HT = (N, 0) if k[0] == "local" else (0, 0) if k[0] == "cloud" else (k[1], k[2])
        c_load = 0.0
        if layers != self.layers:                     # free first (memory), then the server, then load
            self.m = None; gc.collect()
        srv = self._call(pr.json_msg(pr.HELLO, dict(H=HT[0], T=HT[1], netem=self.set_network(cond))), pr.HELLO).json()
        if layers != self.layers:
            t0 = time.perf_counter()
            if layers == ("all",):
                self.m = pm.PartialLlama(self.mdir, dict(all=(0, N)), True, "cpu", torch.float16)
            elif layers is not None:
                self.m = pm.PartialLlama(self.mdir, pm.split_segments(layers[1], layers[2], N)[0], True, "cpu", torch.float16)
            c_load = time.perf_counter() - t0; self.layers = layers
        if self.m is not None and k[0] == "split":                 # adapters for this option (untrained secret)
            pm.su.remove_all_adapters(types.SimpleNamespace(model=types.SimpleNamespace(layers=list(self.m.global_layers.values()))))
            if k[3] in ("secret", "secret_u"):
                self.m.add_secret_adapters(range(0, k[1]), 2000)
            if k[3] == "secret_u" and k[2]:
                self.m.add_secret_adapters(range(N - k[2], N), 2001)
        self.key = k
        return c_load, srv["load_s"]

    def ping(self):
        t0 = time.perf_counter(); self._call(pr.Msg(pr.PING), pr.PING); return time.perf_counter() - t0

    def serve(self, o, ids):
        k = key_of(o)
        if k[0] == "local":
            g = pm.generate_local(self.m, ids, A, "cpu")
            return dict(total=sum(s["step_s"] for s in g["steps"]), ttft=g["steps"][0]["step_s"], server_compute=0.0, steps=g["steps"])
        if k[0] == "cloud":
            t0 = time.perf_counter()
            rep = self._call(pr.json_msg(pr.GENERATE, dict(ids=ids[0].tolist(), A=A)), pr.GENERATE)
            return dict(total=time.perf_counter() - t0, ttft=None, server_compute=rep.server_compute_ns / 1e9, steps=[])
        c = SplitClient.from_parts(self.conn, self.m, k[1], k[2], "cpu")
        g = c.generate(ids, A, eos=())
        s = g["steps"]
        return dict(total=sum(x["step_s"] for x in s), ttft=s[0]["step_s"], server_compute=sum(x["server_compute_s"] for x in s),
                    steps=s)

    def close(self):
        try:
            self.conn.send(pr.Msg(pr.BYE))
        finally:
            self.conn.close()


def run_policy(policy, trace, log, rep=0):
    srv = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "server.py"), "--reconfigurable"],
                           stdout=subprocess.PIPE, stderr=log, text=True)
    port = int(srv.stdout.readline().split()[1])
    rc, est, rows, state = ReconfigClient(port), cm.Estimator(), [], cm.EMPTY
    fixed = {k: cm.decide_cal(256, A, c, NOMINAL, PROF, OPTS)[0] for k, c in CL.items()}
    for r in trace:
        cls, true = CL[r["cls"]], {k: r[k] for k in ("rate_up", "rate_down", "rtt", "server_share")}
        P = r["Pb"]; ids = torch.randint(0, 128000, (1, P), generator=torch.Generator().manual_seed(r["i"]))
        estimating = policy in ("adaptive", "switch_aware")
        cond_used = est.cond() if estimating else NOMINAL
        if policy == "adaptive":
            o = cm.decide_cal(P, A, cls, est.cond(), PROF, OPTS)[0]
        elif policy == "switch_aware":
            o = cm.decide_switch(P, A, cls, est.cond(), PROF, OPTS, state, "a")[0]
        else:
            o = fixed[r["cls"]]
        pred_reload, state = cm.transition(state, o, "a")
        c_load, s_load = rc.configure(o, true)                    # emulation follows the TRUE trace
        rtt_obs = share_rep = None; ping_s = 0.0
        if estimating:                                            # PING with server-reported load (counted)
            t0 = time.perf_counter(); rep_ = rc._call(pr.Msg(pr.PING, b"load"), pr.PING); ping_s = time.perf_counter() - t0
            rtt_obs, share_rep = ping_s, rep_.json()["share"]
        cold = (key_of(o), P) not in rc.used; rc.used.add((key_of(o), P))
        s0 = memstat.snapshot()
        res = rc.serve(o, ids)
        s1 = memstat.snapshot()
        pred_true, det = cm.latency_cal(o, P, A, true, PROF, detail=True)
        if estimating:
            pred_true += cm.ping_cost(true)
        pred_used = cm.latency_cal(o, P, A, cond_used, PROF)
        # observations (estimating policies use them for the next request)
        rate_obs = share_obs = None
        if estimating:
            est.observe(rtt=rtt_obs, share=share_rep)
        if res["steps"] and "net_s" in res["steps"][0]:
            p0 = res["steps"][0]
            ov = cm.CAL["overhead"]["m0_s"] + cm.CAL["overhead"]["m1_s_per_byte"] * (p0["bytes_up"] + p0["bytes_down"])
            rate_obs = cm.Estimator.rate_from_message(p0["bytes_up"], p0["net_s"] - ov, est.rtt)
        if res["server_compute"] > 0:                             # logged only (server-reported load is used)
            _, d1 = cm.latency_cal(o, P, A, {**est.cond(), "server_share": 1.0}, PROF, detail=True)
            share_obs = d1["server"] / res["server_compute"]
        if estimating:
            est.observe(rate=rate_obs)
        meas = res["total"] + ping_s
        row = dict(policy=policy, rep=rep, i=r["i"], phase=r["phase"], cls=r["cls"], P=P, choice=prc.short2(o), cold=cold,
                   rate_up_mbit=true["rate_up"] / 1e6, rtt_ms=true["rtt"] * 1e3, measured_s=meas, ping_s=ping_s,
                   predicted_reload_s=pred_reload,
                   measured_ttft_s=res["ttft"], predicted_true_s=pred_true, err_pct=100 * (pred_true - meas) / meas,
                   predicted_used_s=pred_used, predicted_ttft_s=det.get("ttft"), client_load_s=c_load, server_load_s=s_load,
                   ping_rtt_ms=rtt_obs * 1e3 if rtt_obs else None, rate_obs_mbit=rate_obs / 1e6 if rate_obs else None,
                   share_timing_obs=share_obs, share_reported=share_rep,
                   est_rtt_ms=est.rtt * 1e3, est_rate_mbit=est.rate / 1e6, est_share=est.share,
                   swap_before_gb=s0["swap_used"] / 1e9, swap_after_gb=s1["swap_used"] / 1e9,
                   swap_grew=s1["swap_used"] > s0["swap_used"] + 10**7, client_footprint_gb=s1["footprint"] / 1e9,
                   sys_available_gb=s1["sys_available"] / 1e9)
        rows.append(row)
        print(f"  {policy:15s} r{rep} {r['i']:2d} {r['phase']:5s} {r['cls']:6s} P={P:3d} {row['choice']:32s}: "
              f"measured {meas:6.2f} s, predicted {pred_true:6.2f} s ({row['err_pct']:+5.1f}%) | reload client "
              f"{c_load:4.1f} s server {s_load:4.1f} s (predicted {pred_reload:4.1f}) | swap grew {row['swap_grew']}", flush=True)
    rc.close(); srv.terminate(); srv.wait(timeout=30)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    trace = prc.make_stream(a.n, prc.NETWORKS[1])
    phases = prc.NETWORKS[1]["phases"]
    for r in trace:
        r["Pb"] = min(BUCKETS, key=lambda b: abs(b - r["P"])); r["server_share"] = 1.0
        r["phase"] = phases[min(r["i"] * len(phases) // a.n, len(phases) - 1)]
    s0 = memstat.snapshot()
    print(f"{env_info()}\nsystem available {s0['sys_available']/1e9:.2f} GB, swap {s0['swap_used']/1e9:.2f} GB")
    log = open(os.path.join(OUT, "task4_real_server.log"), "w")
    rows = []
    POLS = ("adaptive", "switch_aware", "fixed_per_class")
    for rep in range(a.repeats):
        for pol in POLS:
            print(f"\n{pol}, repeat {rep}:", flush=True)
            rows += run_policy(pol, trace, log, rep)
    with open(os.path.join(OUT, "task4_real_trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for pol in POLS:
        per = []
        for rep in range(a.repeats):
            rr = [r for r in rows if r["policy"] == pol and r["rep"] == rep]
            meas, pred = sum(r["measured_s"] for r in rr), sum(r["predicted_true_s"] for r in rr)
            rel = sum(r["client_load_s"] + r["server_load_s"] for r in rr); prel = sum(r["predicted_reload_s"] for r in rr)
            per.append(dict(measured=meas, predicted=pred, reload=rel, predicted_reload=prel, total=meas + rel,
                            predicted_total=pred + prel, ping=sum(r["ping_s"] for r in rr),
                            changes=sum(rr[i]["choice"] != rr[i - 1]["choice"] for i in range(1, len(rr))),
                            mape=st.mean(abs(r["err_pct"]) for r in rr), swap=sum(r["swap_grew"] for r in rr)))
        m = lambda k: st.mean(x[k] for x in per)
        sd = lambda k: st.stdev(x[k] for x in per) if len(per) > 1 else 0.0
        summ[pol] = dict(per_repeat=per, **{f"{k}_mean": m(k) for k in per[0]}, **{f"{k}_sd": sd(k) for k in per[0]})
        s = summ[pol]
        print(f"\n{pol} ({a.repeats} repeats): measured {s['measured_mean']:.1f} ± {s['measured_sd']:.1f} s (incl. PING "
              f"{s['ping_mean']:.1f} s), predicted {s['predicted_mean']:.1f} s ({100*(s['predicted_mean']/s['measured_mean']-1):+.1f}%) | "
              f"reload {s['reload_mean']:.1f} ± {s['reload_sd']:.1f} s (predicted {s['predicted_reload_mean']:.1f}) | total "
              f"{s['total_mean']:.1f} ± {s['total_sd']:.1f} s (predicted {s['predicted_total_mean']:.1f}) | changes {s['changes_mean']:.1f} | "
              f"per-request MAPE {s['mape_mean']:.1f}% | requests with swap growth {s['swap_mean']:.1f}")
    s1 = memstat.snapshot()
    json.dump(dict(env=env_info(), profile=PROF, A=A, n=a.n, summary=summ, system_start=s0, system_end=s1),
              open(os.path.join(OUT, "task4_real_trace.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
