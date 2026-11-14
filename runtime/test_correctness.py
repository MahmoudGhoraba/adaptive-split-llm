"""
Step 6 correctness test of the REAL runtime: client on CPU (chosen dtype), server on MPS (fp16), two processes,
TCP loopback, hidden states sent as fp16. Same splits, prompts and reference as check_partial.py (full model,
fp16 on MPS, check_out/reference.pt); public model and untrained secret adapters (seeds 2000/2001).

Different devices/dtypes give different rounding, so divergence is ALLOWED but always reported:
  tokens_equal, first divergence step, top-1 - top-2 logit margin there (reference and split),
  boundary states (layers H, N-T, N) relative L2 error, max over the positions whose inputs are still identical
  (prompt + tokens before the first divergence).

Run:  python test_correctness.py [--dtype bf16]   (default: the choice in check_out/cpu_dtype.json)
Writes check_out/correctness_cpu_client.json, .csv
"""
import argparse
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import partial_model as pm
from check_partial import CFG as CHECK_CFG, env_info, eos_ids, rel_err
from client import SplitClient
from run_split import start_server

OUT = os.path.join(HERE, "check_out")
DT = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def compare(got, ref, H, T, n):
    first = next((i for i, (u, v) in enumerate(zip(got["tokens"], ref["tokens"])) if u != v), None)
    P = got["steps"][0]["n_tokens"]
    k = P + first if first is not None else None          # positions with identical inputs on both runs
    r = dict(tokens_equal=first is None and len(got["tokens"]) == len(ref["tokens"]), n_tokens=len(ref["tokens"]),
             n_tokens_split=len(got["tokens"]), first_divergence=first,
             n_matching_tokens=first if first is not None else len(ref["tokens"]))
    for nm, L in (("bH", H), ("bM", n - T), ("bN", n)):
        r[f"rel_err_{nm}"], r[f"abs_err_{nm}"] = rel_err(got[nm][:k], ref["layers"][L - 1][:k])
    if first is not None:
        r.update(margin_ref_at_div=ref["margins"][first], margin_split_at_div=got["margins"][first],
                 min_margin_ref_before_div=min(ref["margins"][:first + 1]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=DT)
    a = ap.parse_args()
    dt = a.dtype or json.load(open(os.path.join(OUT, "cpu_dtype.json")))["choice"]
    ref = torch.load(os.path.join(OUT, "reference.pt"), weights_only=False)
    mdir = pm.model_dir(); eos = eos_ids(mdir); n = pm.load_config(mdir).num_hidden_layers
    env, s0 = env_info(), memstat.snapshot()
    print(f"{env}\nclient CPU {dt}, server MPS fp16, wire fp16; system available {s0['sys_available']/1e9:.2f} GB, "
          f"swap {s0['swap_used']/1e9:.2f} GB")
    rows, log = [], open(os.path.join(OUT, "correctness_server.log"), "w")
    for H, T in CHECK_CFG["splits"]:
        sb = memstat.snapshot()
        srv, port = start_server(H, T, log)
        c = SplitClient("127.0.0.1", port, H, T, device="cpu", dtype=DT[dt], mdir=mdir)
        variants = [("public", ref["public"])] + ([("adapters", ref["adapters"][(H, T)])] if (H, T) in ref["adapters"] else [])
        for var, refs in variants:
            if var == "adapters":
                c.add_secret_adapters(CHECK_CFG["head_seed"], CHECK_CFG["tail_seed"])
            for i, ids in enumerate(ref["prompts"]):
                g = c.generate(ids, CHECK_CFG["max_new_tokens"], eos, capture=True)
                r = dict(H=H, T=T, U=n - H - T, variant=var, prompt=i, client_dtype=dt, **compare(g, refs[i], H, T, n))
                if var == "adapters":
                    r["differs_from_public"] = g["tokens"] != ref["public"][i]["tokens"]
                rows.append(r)
                print(f"  H={H:2d} T={T:2d} {var:8s} prompt {i}: tokens {'EQUAL' if r['tokens_equal'] else 'DIFFER'} "
                      f"({r['n_matching_tokens']}/{r['n_tokens']} before divergence)"
                      + (f", first divergence step {r['first_divergence']}: margin ref {r['margin_ref_at_div']:.3f}, "
                         f"split {r['margin_split_at_div']:.3f}" if r["first_divergence"] is not None else "")
                      + f" | rel err H/N-T/N {r['rel_err_bH']:.1e}/{r['rel_err_bM']:.1e}/{r['rel_err_bN']:.1e}", flush=True)
            if var == "adapters":
                c.remove_adapters()
        se = memstat.snapshot()
        for r in rows[-len(variants) * len(ref["prompts"]):]:
            r.update(swap_before_gb=sb["swap_used"] / 1e9, swap_after_gb=se["swap_used"] / 1e9,
                     swap_grew=se["swap_used"] > sb["swap_used"] + 10**7)
        c.close(); srv.terminate(); srv.wait(timeout=30)
    s1 = memstat.snapshot()
    eq = sum(r["tokens_equal"] for r in rows)
    print(f"\n{eq}/{len(rows)} generations token-identical to the full model; system swap "
          f"{s0['swap_used']/1e9:.2f} -> {s1['swap_used']/1e9:.2f} GB")
    json.dump(dict(env=env, client_dtype=dt, system_start=s0, system_end=s1, rows=rows),
              open(os.path.join(OUT, "correctness_cpu_client.json"), "w"), indent=1)
    with open(os.path.join(OUT, "correctness_cpu_client.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
