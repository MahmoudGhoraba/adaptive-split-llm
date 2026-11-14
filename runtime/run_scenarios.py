"""
One end-to-end split request per network scenario (application-level emulation), timing protocol of timing.py:
server.py (middle layers, MPS fp16, emulated downlink) + SplitClient (CPU, chosen dtype, emulated uplink).
Warm-up of the prompt-length bucket first; fixed answer length A (EOS ignored); memory/swap logged per request.

Expected network time per message pair (from the configured link, what the emulator should add):
  RTT + (bytes_up + bytes_down) x 8 / rate          (rate symmetric)

Run:  python run_scenarios.py [--H 4 --T 2 --P 128 --A 32 --scenarios none wifi 4g poor]
Writes check_out/scenarios_steps.jsonl, scenarios_summary.csv, scenarios.json
"""
import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import partial_model as pm
import timing
from check_partial import env_info
from client import SplitClient
from netem import NetEm, SCENARIOS

OUT = os.path.join(HERE, "check_out")
DT = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)
PH = ["head_s", "ser_s", "net_s", "server_compute_s", "tail_s", "step_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--P", type=int, default=128, choices=timing.BUCKETS); ap.add_argument("--A", type=int, default=32)
    ap.add_argument("--dtype", choices=DT)
    ap.add_argument("--scenarios", nargs="+", default=["none"] + list(SCENARIOS))
    a = ap.parse_args()
    dt = a.dtype or json.load(open(os.path.join(OUT, "cpu_dtype.json")))["choice"]
    mdir = pm.model_dir(); env = env_info(); s0 = memstat.snapshot()
    print(f"{env}\n(H, T) = ({a.H}, {a.T}), client CPU {dt}, server MPS fp16, P = {a.P}, A = {a.A} (fixed); system "
          f"available {s0['sys_available']/1e9:.2f} GB, swap {s0['swap_used']/1e9:.2f} GB")
    rows, steps_f, log = [], open(os.path.join(OUT, "scenarios_steps.jsonl"), "w"), open(os.path.join(OUT, "scenarios_server.log"), "w")
    for name in a.scenarios:
        sc = SCENARIOS.get(name)
        args = [sys.executable, "-u", os.path.join(HERE, "server.py"), "--H", str(a.H), "--T", str(a.T)]
        if sc:
            args += ["--rate-mbit", str(sc["rate"] / 1e6), "--delay-ms", str(sc["rtt"] / 2 * 1e3)]
        srv = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=log, text=True)
        port = int(srv.stdout.readline().split()[1])
        c = SplitClient("127.0.0.1", port, a.H, a.T, device="cpu", dtype=DT[dt], mdir=mdir,
                        netem=NetEm.scenario(name) if sc else None)
        timing.warmup(c, [a.P])
        g = timing.timed_generate(c, timing.synthetic_ids(a.P, seed=1), a.A)
        sv = c.server_stats(); c.close(); srv.terminate(); srv.wait(timeout=30)
        pre, dec = g["steps"][0], g["steps"][1:]
        exp = lambda s: (sc["rtt"] + (s["bytes_up"] + s["bytes_down"]) * 8 / sc["rate"]) if sc else 0.0
        r = dict(scenario=name, rate_mbit=sc["rate"] / 1e6 if sc else None, rtt_ms=sc["rtt"] * 1e3 if sc else None,
                 H=a.H, T=a.T, P=a.P, A=a.A, client_dtype=dt, n_steps=len(g["steps"]), ttft_s=pre["step_s"],
                 total_s=sum(s["step_s"] for s in g["steps"]), bytes_up=sum(s["bytes_up"] for s in g["steps"]),
                 bytes_down=sum(s["bytes_down"] for s in g["steps"]),
                 prefill_expected_net_s=exp(pre), decode_expected_net_s=exp(dec[0]),
                 server_footprint_gb=sv["footprint"] / 1e9, **{k: v for k, v in g["mem"].items()})
        for k in PH:
            r[f"prefill_{k}"] = pre[k]; r[f"decode_median_{k}"] = st.median(s[k] for s in dec)
        rows.append(r)
        for s in g["steps"]:
            steps_f.write(json.dumps(dict(scenario=name, **s)) + "\n")
        print(f"\n{name}" + (f" ({r['rate_mbit']:g} Mbit/s, RTT {r['rtt_ms']:.0f} ms)" if sc else " (plain loopback)")
              + f": TTFT {r['ttft_s']:.3f} s, total {r['total_s']:.2f} s for {r['n_steps']} steps, "
                f"{r['bytes_up']/1e3:.0f} kB up / {r['bytes_down']/1e3:.0f} kB down; swap grew: {r['swap_grew']}")
        for ph in ("prefill", "decode"):
            key = "prefill_" if ph == "prefill" else "decode_median_"
            print(f"  {ph:7s} [ms]: " + "  ".join(f"{k[:-2]} {r[key + k]*1e3:.2f}" for k in PH)
                  + f"  | net expected {r[ph + '_expected_net_s']*1e3:.2f}", flush=True)
    s1 = memstat.snapshot()
    json.dump(dict(env=env, rows=rows, system_start=s0, system_end=s1), open(os.path.join(OUT, "scenarios.json"), "w"), indent=1)
    with open(os.path.join(OUT, "scenarios_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nsystem swap {s0['swap_used']/1e9:.2f} -> {s1['swap_used']/1e9:.2f} GB. Saved to {OUT}/")


if __name__ == "__main__":
    main()
