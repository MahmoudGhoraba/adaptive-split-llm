"""
Device profiles from MEASUREMENTS (task 4): the Mac client (CPU fp16) with torch.set_num_threads(n), n = 2 and 1
(n = 4 = the task 3 measurements). Client (4,2) = embedding + 4 head + 2 tail layers + output layer, no server.

Per thread count, in a fresh process: 1 warm-up per prompt length, then 3 repeats of
  prefill at P = 32, 128, 512 (fit) and 256 (test);  decode A = 32 steps after P = 128 and P = 512
Timed per phase: head (4 layers), tail (2 layers), output layer. Footprint logged.
Fitted with the same models as fit_task3.py (prefill: L*(FLOPs/f + c) + o; decode: L*(d0 + d1*k) + o).

Run:  python measure_profiles.py [2 1]        ->  check_out/profiles_steps.jsonl, profiles.json
"""
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "check_out")
CFG = dict(H=4, T=2, fit_P=[32, 128, 512], test_P=[256], decode_from=[128, 512], A=32, repeats=3)


def worker(n_threads):
    import torch
    torch.set_num_threads(n_threads)
    import memstat
    import partial_model as pm
    m = pm.PartialLlama(pm.model_dir(), pm.split_segments(CFG["H"], CFG["T"], 16)[0], True, "cpu", torch.float16)
    g = torch.Generator().manual_seed(0); ns = time.perf_counter_ns; rows = []

    @torch.no_grad()
    def run(P, A, rep, record=True):
        caches, x, pos = m.new_caches(), torch.randint(0, 128000, (1, P), generator=g), 0
        for step in range(1 + A):
            t0 = ns(); h = m.run("head", m.embed_ids(x), pos, caches)
            t1 = ns(); h = m.run("tail", h, pos, caches)
            t2 = ns(); nxt = int(m.logits(h[:, -1:])[0, -1].argmax()); t3 = ns()
            if record:
                rows.append(dict(threads=n_threads, P=P, rep=rep, kind="prefill" if step == 0 else "decode",
                                 pos_start=pos, head_s=(t1 - t0) / 1e9, tail_s=(t2 - t1) / 1e9, out_s=(t3 - t2) / 1e9))
            pos += x.shape[1]; x = torch.tensor([[nxt]])

    Ps = sorted(set(CFG["fit_P"] + CFG["test_P"]))
    for P in Ps:
        run(P, CFG["A"] if P in CFG["decode_from"] else 0, -1, record=False)          # warm-up
    for rep in range(CFG["repeats"]):
        for P in Ps:
            run(P, CFG["A"] if P in CFG["decode_from"] else 0, rep)
    s = memstat.snapshot()
    print(json.dumps(dict(threads=n_threads, torch_threads=torch.get_num_threads(), footprint=s["footprint"],
                          rss=s["rss"], swap=s["swap_used"], rows=rows)))


def main(threads):
    import numpy as np
    import fit_task3 as ft
    out, steps_f = {}, open(os.path.join(OUT, "profiles_steps.jsonl"), "w")
    for n in threads:
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--worker", str(n)], capture_output=True, text=True, timeout=7200)
        if r.returncode:
            print(r.stderr[-2000:]); raise SystemExit(f"FAILED threads={n}")
        d = json.loads(r.stdout.strip().splitlines()[-1])
        for x in d["rows"]:
            steps_f.write(json.dumps(x) + "\n")
        obs = []
        for x in d["rows"]:
            k = x["pos_start"] + 1
            obs.append(("client", x["kind"], CFG["H"], x["P"], k, x["head_s"], f"t{n}"))
            obs.append(("client", x["kind"], CFG["T"], x["P"], k, x["tail_s"], f"t{n}"))
        prof = dict(threads=n, torch_threads=d["torch_threads"], footprint_gb=d["footprint"] / 1e9,
                    output_layer_ms=1e3 * st.mean(x["out_s"] for x in d["rows"]), wall_s=time.time() - t0)
        for kind in ("prefill", "decode"):
            tr = [o for o in obs if o[1] == kind and o[3] in CFG["fit_P"]]
            te = [o for o in obs if o[1] == kind and o[3] in CFG["test_P"]]
            ft.PREFILL_MODEL = "flops"
            beta = ft.fit(tr)
            prof[kind] = dict(params=beta.tolist(), train=ft.summary(ft.errors(beta, tr)),
                              test=ft.summary(ft.errors(beta, te)) if te else None)
            if kind == "prefill":
                prof[kind].update(f_eff_flops=1 / beta[0])
        out[str(n)] = prof
        print(f"threads {n}: prefill f_eff {prof['prefill']['f_eff_flops']/1e9:.1f} GFLOP/s (train MAPE "
              f"{prof['prefill']['train']['mape_pct']:.1f}%, test P=256 {prof['prefill']['test']['mape_pct']:.1f}%), decode d0 "
              f"{prof['decode']['params'][0]*1e3:.2f} ms/layer d1 {prof['decode']['params'][1]*1e6:.3f} us/token (train MAPE "
              f"{prof['decode']['train']['mape_pct']:.1f}%), output layer {prof['output_layer_ms']:.2f} ms, footprint "
              f"{prof['footprint_gb']:.2f} GB [{prof['wall_s']:.0f} s]", flush=True)
    json.dump(dict(cfg=CFG, label="FITTED from MEASURED steps", profiles=out), open(os.path.join(OUT, "profiles.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main([int(a) for a in sys.argv[1:]] or [2, 1])
