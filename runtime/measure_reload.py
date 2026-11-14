"""
Switching cost (task 4 follow-up): reload time per side vs number of layers (MEASURED), and a linear fit (FITTED).

  client (CPU fp16): embedding/output + L layers, L = 2, 4, 6, 8, 10, 16  (L = 16: all on client)
  server (MPS fp16): U middle layers, U = 4, 6, 8, 10, 12, 14; and all on server (16 layers + embedding/output)
One process per side, configurations in shuffled order, 3 repeats; before every load the previous weights are
freed (gc + torch.mps.empty_cache), as in the reconfigurable runtime. The safetensors file is in the OS page cache
after the first load (first load reported separately: cold).
Fit: t = a + b * L + c * [embedding/output loaded].

Run:  python measure_reload.py      ->  check_out/reload.json
"""
import gc
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "check_out")
CLIENT = [(1, 1), (2, 2), (4, 2), (6, 2), (8, 2), "all"]        # H + T = 2, 4, 6, 8, 10; all = 16
SERVER = [(1, 1), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), "all"]   # U = 14, 12, 10, 8, 6, 4; all = 16 + ends
REPEATS = 3


def worker(side):
    import torch
    import partial_model as pm
    dev = "cpu" if side == "client" else "mps"
    mdir = pm.model_dir(); rows = []; m = None
    jobs = [(c, r) for c in (CLIENT if side == "client" else SERVER) for r in range(REPEATS)]
    random.Random(0).shuffle(jobs)
    first = True
    for cfg, rep in jobs:
        m = None; gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        if cfg == "all":
            m = pm.PartialLlama(mdir, dict(all=(0, 16)), True, dev, torch.float16); L, ends = 16, True
        else:
            c, s = pm.split_segments(cfg[0], cfg[1], 16)
            m = pm.PartialLlama(mdir, c if side == "client" else s, side == "client", dev, torch.float16)
            L, ends = (cfg[0] + cfg[1], True) if side == "client" else (16 - cfg[0] - cfg[1], False)
        pm.sync(dev)
        rows.append(dict(side=side, config=str(cfg), L=L, ends=ends, rep=rep, cold=first, load_s=time.perf_counter() - t0))
        first = False
    print(json.dumps(rows))


def main():
    import numpy as np
    rows = []
    for side in ("client", "server"):
        r = subprocess.run([sys.executable, __file__, "--worker", side], capture_output=True, text=True, timeout=3600)
        if r.returncode:
            print(r.stderr[-2000:]); raise SystemExit(f"FAILED {side}")
        rows += json.loads(r.stdout.strip().splitlines()[-1])
    fits = {}
    for side in ("client", "server"):
        rr = [x for x in rows if x["side"] == side and not x["cold"]]
        X = np.array([[1.0, x["L"], float(x["ends"])] for x in rr]); y = np.array([x["load_s"] for x in rr])
        if side == "client":                                   # client always loads the embedding: no separate term
            X = X[:, :2]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        fits[side] = dict(params=beta.tolist(), mape_pct=float(np.mean(np.abs(pred - y) / y) * 100),
                          max_abs_err_s=float(np.max(np.abs(pred - y))))
        print(f"{side}: t = {beta[0]:.3f} s + {beta[1]*1e3:.1f} ms/layer" + (f" + {beta[2]:.3f} s [ends]" if len(beta) > 2 else "")
              + f"  (MAPE {fits[side]['mape_pct']:.1f}%, max err {fits[side]['max_abs_err_s']:.2f} s)")
        for cfg in sorted({x["config"] for x in rr}, key=lambda c: next(x["L"] for x in rr if x["config"] == c)):
            v = [x["load_s"] for x in rr if x["config"] == cfg]
            print(f"   {cfg:8s} L={next(x['L'] for x in rr if x['config'] == cfg):2d}: {np.mean(v):.2f} ± {np.std(v):.2f} s")
        cold = [x for x in rows if x["side"] == side and x["cold"]]
        print(f"   cold (first load in the process): {cold[0]['config']} {cold[0]['load_s']:.2f} s")
    json.dump(dict(rows=rows, fits=fits, label="MEASURED rows, FITTED params [a, b(, c)]"),
              open(os.path.join(OUT, "reload.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
