"""
SECRET HEAD + TAIL AT MANY SPLIT POINTS - gives the adaptive controller real split choices
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

WHY
  Until now the secret adapters were measured only at H = 4, T = 2, so every protected decision of the
  optimiser (Step 3c) and the adaptive controller (Step 4) was forced to that one split. This script
  repeats the head + tail measurement (secret_ushape.py, same attack, same metrics) at several (H, T)
  points, and repeats H = 4, T = 2 with new training seeds to show the run-to-run spread.

WHAT IT RUNS
  For every (H, T, seed) job it calls   secret_ushape.py --H .. --T .. --seed .. --configs ..
  Each job trains the attackers without the secret once, then for every (lambda_prompt, lambda_answer):
  trains secret adapters on the H head and T tail layers, measures quality, prompt leakage (attacker A and B)
  and answer leakage (server runs the public tail). Finished jobs are skipped, so a disconnected Colab
  session can simply be restarted.

BATCHES (each fits in one Colab T4 session; run them in any order)
  A : (H, T) = (2, 1), (2, 2)
  B : (H, T) = (1, 1), (4, 1)
  C : (H, T) = (6, 2), (8, 2)
  D : (H, T) = (4, 2) with training seeds 1 and 2 (seed 0 = the earlier run)
  default configs: (lambda_prompt, lambda_answer) = (0.1, 0), (0.1, 0.01), (0.1, 0.03), (0.1, 0.1)
  (lambda_answer = 0.3 is left out: it failed the quality check at H = 4)

OUTPUT (in --save_dir)
  runs/H{H}_T{T}_seed{s}/          one folder per job (results.json, csv, plot)
  multisplit_results.csv           every row of every job
  measured_multisplit.txt          SECRET_U dict for the optimiser: mean over seeds, with the spread
  multisplit_tradeoff.png          answer and prompt leakage vs quality, one line per (H, T)

HOW TO RUN ON GOOGLE COLAB (T4 GPU)
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload secret_multisplit.py AND secret_ushape.py (same folder)
  3. from huggingface_hub import notebook_login; notebook_login()
  4. !pip install -q datasets
  5. Recommended, so nothing is lost if Colab disconnects:
        from google.colab import drive; drive.mount('/content/drive')
  6. !python secret_multisplit.py --batch A --save_dir /content/drive/MyDrive/multisplit
     (next session: --batch B, then C, then D, all with the same --save_dir)
  7. After the last batch, or at any time:
        !python secret_multisplit.py --collect --save_dir /content/drive/MyDrive/multisplit
     and download multisplit_results.csv, measured_multisplit.txt, multisplit_tradeoff.png

  Quick code test without the real model (CPU, a few minutes):
        python secret_multisplit.py --smoke
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "secret_ushape.py")

BATCHES = {
    "A": dict(points=[(2, 1), (2, 2)], seeds=[0]),
    "B": dict(points=[(1, 1), (4, 1)], seeds=[0]),
    "C": dict(points=[(6, 2), (8, 2)], seeds=[0]),
    "D": dict(points=[(4, 2)], seeds=[1, 2]),
}
CONFIGS = "0.1,0 0.1,0.01 0.1,0.03 0.1,0.1"
SMOKE = dict(points=[(1, 1), (2, 1)], seeds=[0, 1], configs="0.1,0 0.1,0.1")


def job_dir(save_dir, H, T, s):
    return os.path.join(save_dir, "runs", f"H{H}_T{T}_seed{s}")


def run_jobs(points, seeds, configs, steps, save_dir, smoke):
    jobs = [(H, T, s) for (H, T) in points for s in seeds]
    print(f"{len(jobs)} job(s): {jobs}\nconfigs (lambda_prompt,lambda_answer): {configs}\n", flush=True)
    for i, (H, T, s) in enumerate(jobs, 1):
        out = job_dir(save_dir, H, T, s)
        if os.path.exists(os.path.join(out, "results.json")):
            print(f"[{i}/{len(jobs)}] H={H}, T={T}, seed={s}: already done, skipped", flush=True)
            continue
        cmd = [sys.executable, WORKER, "--H", str(H), "--T", str(T), "--seed", str(s),
               "--configs", configs, "--out", out]
        if steps:
            cmd += ["--steps", str(steps)]
        if smoke:
            cmd += ["--smoke"]
        print(f"[{i}/{len(jobs)}] H={H}, T={T}, seed={s}: running ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"Job H={H}, T={T}, seed={s} failed (exit code {r.returncode}). Fix the error and run the "
                     f"same command again: finished jobs are skipped.")
        print(f"[{i}/{len(jobs)}] done in {(time.time() - t0) / 60:.1f} min\n", flush=True)


def collect(save_dir):
    files = sorted(glob.glob(os.path.join(save_dir, "runs", "*", "results.json")))
    if not files:
        sys.exit(f"No finished jobs in {save_dir}/runs/")
    rows = []
    for f in files:
        d = json.load(open(f))
        c = d["config"]; H, T, s = c["H"], c["T"], c.get("train_seed", 0)
        failed = [x["name"] for x in d["sanity_checks"] if not x["passed"]]
        for r in d["rows"]:
            g = lambda x, k: None if x is None else x[k]
            rows.append(dict(H=H, T=T, seed=s, lam_p=r["lam_p"], lam_a=r["lam_a"],
                             agreement=r["agreement"], ppl_ratio=r["ppl_ratio"],
                             leak=g(r["prompt_A"], "content_leak"), private_exact=g(r["prompt_A"], "private_exact"),
                             leak_B=g(r["prompt_B"], "content_leak"), private_exact_B=g(r["prompt_B"], "private_exact"),
                             answer_leak=g(r["answer_server_tail"], "content_leak"),
                             smoke=d["smoke"], failed_checks="; ".join(failed)))
    keys = list(rows[0].keys())
    with open(os.path.join(save_dir, "multisplit_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # mean (and spread) over training seeds for every (H, T, lam_p, lam_a)
    groups = {}
    for r in rows:
        if r["lam_p"] is None:
            continue
        groups.setdefault((r["H"], r["T"], r["lam_p"], r["lam_a"]), []).append(r)
    fields = ["leak", "private_exact", "leak_B", "private_exact_B", "answer_leak", "agreement", "ppl_ratio"]
    lines = ["# secret head + secret tail at several splits (secret_multisplit.py); mean over training seeds",
             "# (H, T, lambda_prompt, lambda_answer): leak/answer_leak = attacker without the secret "
             "(answer: server runs the public tail); *_B = attacker with the secret",
             "# agreement > 0.5 is required by the sanity check; entries below it are kept but marked", "SECRET_U = {"]
    spread = ["", "# spread over seeds (half of max - min), only where more than one seed was run:"]
    for k in sorted(groups):
        g = groups[k]
        m = {f: float(np.mean([r[f] for r in g])) for f in fields}
        body = ", ".join(f"{f}={round(m[f], 4 if f != 'ppl_ratio' else 3)}" for f in fields)
        flag = "   # agreement < 0.5" if m["agreement"] < 0.5 else ""
        lines.append(f"    {k}: dict({body}),   # n_seeds={len(g)}{flag}")
        if len(g) > 1:
            sp = {f: max(r[f] for r in g) - min(r[f] for r in g) for f in ("leak", "answer_leak", "agreement")}
            spread.append(f"#   {k}: leak +-{sp['leak']/2:.3f}, answer_leak +-{sp['answer_leak']/2:.3f}, "
                          f"agreement +-{sp['agreement']/2:.3f}  (half range, {len(g)} seeds)")
    lines.append("}")
    with open(os.path.join(save_dir, "measured_multisplit.txt"), "w") as f:
        f.write("\n".join(lines + spread) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    splits = sorted({(k[0], k[1]) for k in groups})
    cmap = plt.get_cmap("viridis")
    for i, (H, T) in enumerate(splits):
        ks = sorted([k for k in groups if k[:2] == (H, T)], key=lambda k: k[3])
        q = [100 * np.mean([r["agreement"] for r in groups[k]]) for k in ks]
        col = cmap(i / max(1, len(splits) - 1))
        for a_, fld in zip(ax, ("answer_leak", "leak")):
            y = [100 * np.mean([r[fld] for r in groups[k]]) for k in ks]
            a_.plot(q, y, "-o", color=col, lw=2, ms=5, label=f"H={H}, T={T}")
    ax[0].set_title("ANSWER: server runs the public tail", fontsize=10.5)
    ax[1].set_title("PROMPT: attacker without the secret", fontsize=10.5)
    for a_ in ax:
        a_.set_xlabel("Answer quality: next token same as public model [%]")
        a_.set_ylim(0, 105); a_.grid(alpha=.3); a_.invert_xaxis()
    ax[0].set_ylabel("Content tokens recovered by the server [%]")
    ax[1].legend(frameon=False, fontsize=9, loc="lower left")
    smoke = any(r["smoke"] for r in rows)
    fig.suptitle("Secret head + tail at several splits: each point = one λ_answer (λ_prompt = 0.1), mean over seeds"
                 + ("  [TINY RANDOM MODEL, smoke test]" if smoke else ""), fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, "multisplit_tradeoff.png"), dpi=200)

    print(f"\n{len(files)} job(s), {len(rows)} rows collected into {save_dir}/")
    print("   multisplit_results.csv, measured_multisplit.txt, multisplit_tradeoff.png")
    bad = [r for r in rows if r["failed_checks"]]
    if bad:
        print("   NOTE: some jobs failed a sanity check (column failed_checks); report them, do not drop them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=sorted(BATCHES), help="which group of splits to run")
    ap.add_argument("--points", type=str, help='custom splits, e.g. "2,1 6,2" (H,T)')
    ap.add_argument("--seeds", type=str, help='e.g. "0 1 2"')
    ap.add_argument("--configs", type=str, default=CONFIGS)
    ap.add_argument("--steps", type=int, help="defender training steps (default 1200)")
    ap.add_argument("--save_dir", type=str, default="multisplit_results")
    ap.add_argument("--collect", action="store_true", help="only merge finished jobs")
    ap.add_argument("--smoke", action="store_true", help="tiny random model on CPU, to test the code")
    a = ap.parse_args()

    if a.smoke:
        save = a.save_dir if a.save_dir != "multisplit_results" else "multisplit_results_smoke"
        run_jobs(SMOKE["points"], SMOKE["seeds"], SMOKE["configs"], 60, save, smoke=True)
        collect(save)
        return
    if not a.collect:
        if a.points:
            points = [tuple(int(v) for v in p.split(",")) for p in a.points.split()]
            seeds = [int(s) for s in a.seeds.split()] if a.seeds else [0]
        elif a.batch:
            points, seeds = BATCHES[a.batch]["points"], BATCHES[a.batch]["seeds"]
            if a.seeds:
                seeds = [int(s) for s in a.seeds.split()]
        else:
            sys.exit("Give --batch A|B|C|D, or --points, or --collect (see the top of this file).")
        run_jobs(points, seeds, a.configs, a.steps, a.save_dir, smoke=False)
    collect(a.save_dir)


if __name__ == "__main__":
    main()
