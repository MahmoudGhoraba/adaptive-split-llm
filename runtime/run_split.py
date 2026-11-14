"""
First end-to-end split run on loopback (step 5): server.py in its own process, SplitClient in this one, TCP on
127.0.0.1, NO network emulation. Compares greedy tokens with the full-model reference of check_partial.py
(check_out/reference.pt: same prompt ids, 48 tokens; public model and untrained secret adapters, seeds 2000/2001)
and logs per-phase timings (MEASURED; first run, not the task 3 protocol: no repeats, Chrome open).

Run:  python run_split.py [--splits "4,2 1,1"]
Writes check_out/e2e_steps.jsonl (one row per step), e2e_summary.csv (one row per generation), e2e.json
"""
import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import partial_model as pm
from check_partial import CFG as CHECK_CFG, env_info, eos_ids
from client import SplitClient

OUT = os.path.join(HERE, "check_out")
PHASES = ["head_s", "ser_s", "net_s", "server_compute_s", "server_total_s", "rtt_s", "tail_s", "step_s"]


def start_server(H, T, log):
    p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "server.py"), "--H", str(H), "--T", str(T)],
                         stdout=subprocess.PIPE, stderr=log, text=True)
    line = p.stdout.readline().strip()
    if not line.startswith("READY"):
        p.kill(); raise RuntimeError(f"server did not start (got {line!r}); see {log.name}")
    return p, int(line.split()[1])


def compare(got, ref):
    first = next((i for i, (u, v) in enumerate(zip(got["tokens"], ref["tokens"])) if u != v), None)
    r = dict(tokens_equal=first is None and len(got["tokens"]) == len(ref["tokens"]), n_tokens=len(ref["tokens"]),
             n_tokens_split=len(got["tokens"]), first_divergence=first,
             max_margin_diff=max(abs(a - b) for a, b in zip(got["margins"], ref["margins"])))
    if first is not None:
        r.update(margin_ref_at_div=ref["margins"][first], margin_split_at_div=got["margins"][first])
    return r


def summarise(steps):
    pre, dec = steps[0], steps[1:]
    s = dict(P=pre["n_tokens"], n_steps=len(steps), ttft_s=pre["step_s"], total_s=sum(x["step_s"] for x in steps),
             bytes_up=sum(x["bytes_up"] for x in steps), bytes_down=sum(x["bytes_down"] for x in steps))
    for k in PHASES:
        s[f"prefill_{k}"] = pre[k]
        if dec:
            s[f"decode_median_{k}"] = st.median(x[k] for x in dec)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="4,2 1,1")
    a = ap.parse_args()
    splits = [tuple(int(v) for v in s.split(",")) for s in a.splits.split()]
    ref = torch.load(os.path.join(OUT, "reference.pt"), weights_only=False)
    mdir = pm.model_dir(); eos = eos_ids(mdir); n_new = CHECK_CFG["max_new_tokens"]
    env, sys0 = env_info(), memstat.snapshot()
    print(f"{env}\nsystem at start: available {sys0['sys_available']/1e9:.2f} GB, swap {sys0['swap_used']/1e9:.2f} GB")
    rows, mem, steps_f = [], [], open(os.path.join(OUT, "e2e_steps.jsonl"), "w")
    log = open(os.path.join(OUT, "e2e_server.log"), "w")
    for H, T in splits:
        s_before = memstat.snapshot()
        srv, port = start_server(H, T, log)
        c = SplitClient("127.0.0.1", port, H, T, mdir=mdir)
        print(f"\nH={H} T={T} U={c.server_info['U']} (server layers {c.server_info['layers']}), port {port}", flush=True)
        rtt0 = [c.ping() for _ in range(20)]
        c.generate(ref["prompts"][0], 8, eos)                       # warm-up (MPS kernels), discarded
        variants = [("public", ref["public"])]
        if (H, T) in ref["adapters"]:
            variants.append(("adapters", ref["adapters"][(H, T)]))
        for var, refs in variants:
            if var == "adapters":
                c.add_secret_adapters(CHECK_CFG["head_seed"], CHECK_CFG["tail_seed"])
            for i, ids in enumerate(ref["prompts"]):
                g = c.generate(ids, n_new, eos)
                r = dict(H=H, T=T, U=16 - H - T, variant=var, prompt=i, **compare(g, refs[i]), **summarise(g["steps"]))
                if var == "adapters":
                    r["differs_from_public"] = g["tokens"] != ref["public"][i]["tokens"]
                rows.append(r)
                for x in g["steps"]:
                    steps_f.write(json.dumps(dict(H=H, T=T, variant=var, prompt=i, **x)) + "\n")
                print(f"  {var:8s} prompt {i} (P={r['P']}): tokens {'EQUAL' if r['tokens_equal'] else 'DIFFER'} "
                      f"{r['n_tokens_split']}/{r['n_tokens']}"
                      + (f" first divergence {r['first_divergence']} (margins ref {r['margin_ref_at_div']:.3f}, split "
                         f"{r['margin_split_at_div']:.3f})" if r["first_divergence"] is not None else "")
                      + f", max |margin diff| {r['max_margin_diff']:.1e}"
                      + (f", differs from public: {r['differs_from_public']}" if var == "adapters" else "")
                      + f" | TTFT {r['ttft_s']*1e3:.0f} ms, decode {r['decode_median_step_s']*1e3:.1f} ms/token, "
                        f"total {r['total_s']:.2f} s", flush=True)
            if var == "adapters":
                c.remove_adapters()
        m = dict(H=H, T=T, ping_rtt_median_ms=st.median(rtt0) * 1e3, client=memstat.snapshot(), server=c.server_stats(),
                 system_before=s_before)
        mem.append(m)
        c.close(); srv.terminate(); srv.wait(timeout=30)
        del c; torch.mps.empty_cache()
        print(f"  memory: client footprint {m['client']['footprint']/1e9:.2f} GB (peak {m['client']['peak_footprint']/1e9:.2f}), "
              f"server {m['server']['footprint']/1e9:.2f} GB (peak {m['server']['peak_footprint']/1e9:.2f}); system available "
              f"{s_before['sys_available']/1e9:.2f} -> {m['client']['sys_available']/1e9:.2f} GB, swap "
              f"{s_before['swap_used']/1e9:.2f} -> {m['client']['swap_used']/1e9:.2f} GB; empty PING RTT median "
              f"{m['ping_rtt_median_ms']:.3f} ms", flush=True)
    sys1 = memstat.snapshot()
    json.dump(dict(env=env, system_start=sys0, system_end=sys1, swap_grew=sys1["swap_used"] > sys0["swap_used"] + 10**7,
                   rows=rows, memory=mem), open(os.path.join(OUT, "e2e.json"), "w"), indent=1)
    with open(os.path.join(OUT, "e2e_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    ok = all(r["tokens_equal"] for r in rows)
    print(f"\n{'ALL TOKENS EQUAL TO THE FULL MODEL' if ok else 'TOKEN MISMATCH (see above)'}; "
          f"system swap {sys0['swap_used']/1e9:.2f} -> {sys1['swap_used']/1e9:.2f} GB. Saved to {OUT}/")


if __name__ == "__main__":
    main()
