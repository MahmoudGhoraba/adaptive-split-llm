"""
STEP 3b - Optimisation with TWO decisions: split point AND noise level
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script (own copy of the Step 1 latency model). Leakage and answer quality come from
the MEASURED noise sweep on Llama-3.2-1B-Instruct (noise_sweep.py, Colab, 2026-11-02):
an adaptive MLP attacker that knows the noise level, 40k held-out tokens, 400 prompts with
planted private details.

Decision for each task i:  scheme s = (H, T, sigma)
    H in {1, 2, 4, 6, 8, 10, 12}   (the splits measured with noise), T = tail layers on the client,
    sigma = noise strength added to the hidden state the client sends (0 = no protection)
    plus the two baselines: all-local (no leakage, but needs ~2.5 GB) and all-cloud (100% leakage).

    minimise   sum_i L(i, s_i)
    subject to C1 privacy : prompt_leak(H, sigma) <= eps          (content tokens, adaptive attacker)
               C2 quality : agreement(H, sigma)   >= q_min        (next token same as the clean model)
               C3 memory  : client memory(H + T)  <= M_client
               C4 deadline: L(i, s) <= deadline   (optional)

Adding noise costs no measurable time, so latency depends only on H + T (Step 1). With public tail
weights the server can compute the answer itself, so answer leakage is 100% in every split and is
reported separately; the privacy limit applies to the PROMPT.
Solved exactly by checking every option for every task (like ILLOO).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- measured tables (noise sweep)
MEASURED = {
    (1, 0.0): dict(leak=0.9996, private_exact=0.9363, agreement=1.0, ppl_ratio=1.0),
    (1, 0.25): dict(leak=0.9994, private_exact=0.9346, agreement=0.9162, ppl_ratio=1.014),
    (1, 0.5): dict(leak=0.9996, private_exact=0.938, agreement=0.8324, ppl_ratio=1.081),
    (1, 1.0): dict(leak=0.9983, private_exact=0.9321, agreement=0.6235, ppl_ratio=1.784),
    (1, 2.0): dict(leak=0.9836, private_exact=0.8567, agreement=0.1826, ppl_ratio=40.878),
    (1, 4.0): dict(leak=0.7531, private_exact=0.2565, agreement=0.0752, ppl_ratio=173.087),
    (2, 0.0): dict(leak=0.9974, private_exact=0.9296, agreement=1.0, ppl_ratio=1.0),
    (2, 0.25): dict(leak=0.9969, private_exact=0.9338, agreement=0.7592, ppl_ratio=1.221),
    (2, 0.5): dict(leak=0.9941, private_exact=0.8977, agreement=0.4205, ppl_ratio=4.749),
    (2, 1.0): dict(leak=0.9776, private_exact=0.7603, agreement=0.1329, ppl_ratio=61.874),
    (2, 2.0): dict(leak=0.864, private_exact=0.3906, agreement=0.039, ppl_ratio=3596.575),
    (2, 4.0): dict(leak=0.3225, private_exact=0.0453, agreement=0.0145, ppl_ratio=8519.692),
    (4, 0.0): dict(leak=0.987, private_exact=0.8927, agreement=1.0, ppl_ratio=1.0),
    (4, 0.25): dict(leak=0.9834, private_exact=0.865, agreement=0.8513, ppl_ratio=1.058),
    (4, 0.5): dict(leak=0.9752, private_exact=0.7804, agreement=0.6666, ppl_ratio=1.495),
    (4, 1.0): dict(leak=0.9379, private_exact=0.6127, agreement=0.2445, ppl_ratio=17.071),
    (4, 2.0): dict(leak=0.7862, private_exact=0.3026, agreement=0.0475, ppl_ratio=2914.194),
    (4, 4.0): dict(leak=0.2619, private_exact=0.0469, agreement=0.0148, ppl_ratio=5798.971),
    (6, 0.0): dict(leak=0.9778, private_exact=0.8122, agreement=1.0, ppl_ratio=1.0),
    (6, 0.25): dict(leak=0.9729, private_exact=0.7997, agreement=0.8671, ppl_ratio=1.05),
    (6, 0.5): dict(leak=0.9668, private_exact=0.7469, agreement=0.7115, ppl_ratio=1.355),
    (6, 1.0): dict(leak=0.9186, private_exact=0.5817, agreement=0.3045, ppl_ratio=10.965),
    (6, 2.0): dict(leak=0.7111, private_exact=0.2565, agreement=0.0551, ppl_ratio=2249.372),
    (6, 4.0): dict(leak=0.1881, private_exact=0.0344, agreement=0.0134, ppl_ratio=4859.391),
    (8, 0.0): dict(leak=0.9752, private_exact=0.8282, agreement=1.0, ppl_ratio=1.0),
    (8, 0.25): dict(leak=0.9713, private_exact=0.7804, agreement=0.8615, ppl_ratio=1.075),
    (8, 0.5): dict(leak=0.961, private_exact=0.7192, agreement=0.6995, ppl_ratio=1.492),
    (8, 1.0): dict(leak=0.9173, private_exact=0.565, agreement=0.2937, ppl_ratio=15.453),
    (8, 2.0): dict(leak=0.7105, private_exact=0.2498, agreement=0.0607, ppl_ratio=1566.679),
    (8, 4.0): dict(leak=0.1782, private_exact=0.0344, agreement=0.0144, ppl_ratio=4344.564),
    (10, 0.0): dict(leak=0.9785, private_exact=0.7972, agreement=1.0, ppl_ratio=1.0),
    (10, 0.25): dict(leak=0.9714, private_exact=0.7904, agreement=0.8529, ppl_ratio=1.088),
    (10, 0.5): dict(leak=0.9634, private_exact=0.7309, agreement=0.6968, ppl_ratio=1.511),
    (10, 1.0): dict(leak=0.9283, private_exact=0.6144, agreement=0.344, ppl_ratio=9.812),
    (10, 2.0): dict(leak=0.7724, private_exact=0.3026, agreement=0.0947, ppl_ratio=265.194),
    (10, 4.0): dict(leak=0.2566, private_exact=0.0369, agreement=0.0144, ppl_ratio=2419.65),
    (12, 0.0): dict(leak=0.9721, private_exact=0.7963, agreement=1.0, ppl_ratio=1.0),
    (12, 0.25): dict(leak=0.967, private_exact=0.7594, agreement=0.8656, ppl_ratio=1.072),
    (12, 0.5): dict(leak=0.9575, private_exact=0.7343, agreement=0.7286, ppl_ratio=1.422),
    (12, 1.0): dict(leak=0.9248, private_exact=0.57, agreement=0.4469, ppl_ratio=4.9),
    (12, 2.0): dict(leak=0.7622, private_exact=0.2615, agreement=0.1816, ppl_ratio=51.511),
    (12, 4.0): dict(leak=0.278, private_exact=0.0427, agreement=0.0193, ppl_ratio=1285.178),
}
H_MEASURED = sorted({k[0] for k in MEASURED})
SIGMAS = sorted({k[1] for k in MEASURED})

# ---------------------------------------------------------------- model + system (same as Steps 1-3)
N_LAYERS, D_MODEL, N_HEADS, N_KV_HEADS, D_FF, VOCAB = 16, 2048, 32, 8, 8192, 128_256
HEAD_DIM = D_MODEL // N_HEADS
P_LAYER = D_MODEL * D_MODEL * 2 + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2 + 3 * D_MODEL * D_FF
P_HEAD = D_MODEL * VOCAB

PARAMS = dict(n_tasks=20, prompt_min=32, prompt_max=512, answer_len=128,
              f_client=10e9, f_server=1e12, server_share=1.0,
              rate_up=20e6, rate_down=50e6, rtt=0.050,
              bytes_per_token_text=4, bytes_per_value=2, seed=7)

EPS_MAIN    = 0.975    # prompt privacy limit for the main plot (lowest feasible at quality >= 85% is ~97.25%)
Q_MAIN      = 0.85     # minimum answer quality (agreement with the clean model)
M_CLIENT_GB = 2.0
DEADLINE_S  = None
Q_LEVELS    = [0.95, 0.85, 0.70, 0.30]           # quality limits compared in the plots
EPS_SWEEP   = np.round(np.arange(0.15, 1.0001, 0.0025), 4)


def layer_flops(n_new, ctx_start):
    pos = np.arange(ctx_start + 1, ctx_start + n_new + 1)
    return 2 * P_LAYER * n_new + 4 * D_MODEL * pos.sum()


_W = {}
def work_one_layer(P, A):
    if (P, A) not in _W:
        _W[(P, A)] = layer_flops(P, 0) + sum(layer_flops(1, P + k) for k in range(A))
    return _W[(P, A)]


def make_tasks(p):
    return np.random.default_rng(p["seed"]).integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


def schemes():
    out = [("local",), ("cloud",)]
    for H in H_MEASURED:
        for T in range(0, N_LAYERS - H):
            for s in SIGMAS:
                out.append(("split", H, T, s))
    return out


def name(s):
    if s[0] == "local":
        return "All on client"
    if s[0] == "cloud":
        return "All on server"
    return f"H={s[1]}, T={s[2]}, σ={s[3]:g}"


def latency(P, s, p):
    P = int(P); A = p["answer_len"]; w = work_one_layer(P, A); head = 2 * P_HEAD * (1 + A)
    if s[0] == "local":
        return (N_LAYERS * w + head) / p["f_client"]
    if s[0] == "cloud":
        tr = P * p["bytes_per_token_text"] * 8 / p["rate_up"] + A * p["bytes_per_token_text"] * 8 / p["rate_down"]
        return (N_LAYERS * w + head) / (p["f_server"] * p["server_share"]) + tr + p["rtt"]
    H, T = s[1], s[2]; U = N_LAYERS - H - T
    client = ((H + T) * w + head) / p["f_client"]
    server = U * w / (p["f_server"] * p["server_share"])
    transfer = (P + A) * D_MODEL * p["bytes_per_value"] * 8 * (1 / p["rate_up"] + 1 / p["rate_down"])
    return client + server + transfer + (1 + A) * p["rtt"]


def prompt_leak(s):
    return 0.0 if s[0] == "local" else 1.0 if s[0] == "cloud" else MEASURED[(s[1], s[3])]["leak"]


def private_exact(s):
    return 0.0 if s[0] == "local" else 1.0 if s[0] == "cloud" else MEASURED[(s[1], s[3])]["private_exact"]


def quality(s):
    return 1.0 if s[0] != "split" else MEASURED[(s[1], s[3])]["agreement"]


def client_memory_gb(P, s, p):
    if s[0] == "cloud":
        return 0.0
    layers = N_LAYERS if s[0] == "local" else s[1] + s[2]
    kv = 2 * N_KV_HEADS * HEAD_DIM * p["bytes_per_value"] * (int(P) + p["answer_len"])
    return ((layers * P_LAYER + P_HEAD) * p["bytes_per_value"] + layers * kv) / 1e9


def solve(tasks, eps, q_min, p, m=M_CLIENT_GB, deadline=DEADLINE_S):
    S = schemes(); out = []
    for P in tasks:
        best = None
        for s in S:
            if prompt_leak(s) > eps + 1e-12 or quality(s) < q_min - 1e-12:
                continue
            if client_memory_gb(P, s, p) > m:
                continue
            L = latency(P, s, p)
            if deadline is not None and L > deadline:
                continue
            if best is None or L < best[1]:
                best = (s, L)
        out.append(best)
    return out


def run(p=PARAMS, out="."):
    tasks = make_tasks(p)
    tag = "Measured on Llama-3.2-1B (adaptive attacker); tail weights public"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    loc_L = sum(latency(P, ("local",), p) for P in tasks)
    cl_L = sum(latency(P, ("cloud",), p) for P in tasks)

    # ---- main solution ----
    sol = solve(tasks, EPS_MAIN, Q_MAIN, p)
    feasible = all(sol)
    with open(f"{out}/split_noise_per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "prompt_tokens", "chosen", "H", "T", "sigma", "latency_s", "prompt_leak_percent",
                    "private_details_exact_percent", "agreement_percent", "client_memory_GB"])
        for i, (P, c) in enumerate(zip(tasks, sol), 1):
            if c is None:
                w.writerow([i, P, "INFEASIBLE"] + [""] * 8); continue
            s, L = c
            w.writerow([i, P, name(s), *(s[1:4] if s[0] == "split" else ("", "", "")), round(L, 2),
                        round(100 * prompt_leak(s), 2), round(100 * private_exact(s), 2),
                        round(100 * quality(s), 2), round(client_memory_gb(P, s, p), 3)])

    # ---- sweep eps x q_min ----
    sweep = {}
    for q in Q_LEVELS:
        sweep[q] = []
        for e in EPS_SWEEP:
            so = solve(tasks, e, q, p)
            if all(so):
                sweep[q].append((e, sum(c[1] for c in so), so))
            else:
                sweep[q].append((e, None, None))
    with open(f"{out}/split_noise_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_min_percent", "eps_percent", "total_latency_s", "most_used_choice"])
        for q in Q_LEVELS:
            for e, L, so in sweep[q]:
                if L is None:
                    w.writerow([round(100 * q), round(100 * e, 2), "", "INFEASIBLE"]); continue
                names = [name(c[0]) for c in so]
                w.writerow([round(100 * q), round(100 * e, 2), round(L, 2), max(set(names), key=names.count)])

    cols = ["#1a9e55", "#2a6fdb", "#d98a1c", "#7a3cc9"]

    # ---- Figure 1: price of privacy for several quality limits ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for q, c in zip(Q_LEVELS, cols):
        pts = [(100 * e, L / 60) for e, L, _ in sweep[q] if L is not None and e < 1.0]
        if pts:
            ax.step(*zip(*pts), where="post", color=c, lw=2, label=f"quality ≥ {100*q:.0f}%  (feasible from ε = {pts[0][0]:.1f}%)")
    ax.axhline(loc_L / 60, color="#2a6fdb", ls=":", lw=1.2)
    ax.text(90.1, loc_L / 60 + 0.4, "All on client (does not fit in 2 GB)", fontsize=8.5, color="#2a6fdb")
    ax.axhline(cl_L / 60, color="#e0533d", ls="--", lw=1.2)
    ax.text(90.1, cl_L / 60 + 0.4, "All on server (leaks 100%)", fontsize=8.5, color="#e0533d")
    ax.set_xlabel("Prompt privacy limit ε: max. share of content tokens the server may recover [%]")
    ax.set_ylabel(f"Total latency for {len(tasks)} tasks [min]")
    ax.set_title(f"Price of privacy with split + noise\n{tag}", fontsize=11)
    ax.set_xlim(90, 100.3); ax.grid(alpha=.3)
    ax.text(90.1, 2.2, "Below ε ≈ 92% nothing is feasible at any quality ≥ 30%", fontsize=8.5, color="#a33")
    ax.legend(frameon=True, framealpha=.95, edgecolor="none", fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(); fig.savefig(f"{out}/step3b_price_of_privacy.png", dpi=200)

    # ---- Figure 2: what is achievable: lowest leakage for each quality limit ----
    qs = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    best_leak, best_exact = [], []
    for q in qs:
        cands = [k for k, v in MEASURED.items() if v["agreement"] >= q - 1e-12]
        k = min(cands, key=lambda k: MEASURED[k]["leak"])
        best_leak.append(100 * MEASURED[k]["leak"]); best_exact.append(100 * MEASURED[k]["private_exact"])
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.step(100 * qs, best_leak, where="post", color="#1a9e55", lw=2.2, label="Content tokens recovered")
    ax.step(100 * qs, best_exact, where="post", color="#e0533d", lw=2.2, label="Private details recovered completely")
    for (H, s), v in MEASURED.items():
        ax.scatter(100 * v["agreement"], 100 * v["leak"], s=10, color="#9aa3ad", zorder=1)
    ax.axvspan(85, 100, color="#e5f4e4", alpha=.6, lw=0)
    ax.text(92.5, 8, "usable\nanswers", ha="center", fontsize=9, color="#1a7a3e")
    ax.set_xlabel("Minimum answer quality: next token same as clean model [%]")
    ax.set_ylabel("Lowest achievable leakage [%]")
    ax.set_title(f"Best privacy reachable at each quality level (grey = every split × noise)\n{tag}", fontsize=10.5)
    ax.set_ylim(0, 102); ax.grid(alpha=.3)
    ax.legend(frameon=True, framealpha=.95, edgecolor="none", fontsize=9, loc="center", bbox_to_anchor=(0.6, 0.5))
    fig.tight_layout(); fig.savefig(f"{out}/step3b_privacy_vs_quality.png", dpi=200)

    # ---- Figure 3: the sketch with the optimum ----
    if feasible:
        n = np.arange(1, len(tasks) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
        ax[0].plot(n, np.cumsum([latency(P, ("local",), p) for P in tasks]) / 60, "-o", ms=4, color="#2a6fdb", label="All on client")
        ax[0].plot(n, np.cumsum([c[1] for c in sol]) / 60, "-D", ms=4, lw=2.2, color="#1a9e55",
                   label=f"Optimal split + noise (ε={100*EPS_MAIN:.1f}%, quality ≥ {100*Q_MAIN:.0f}%)")
        ax[0].plot(n, np.cumsum([latency(P, ("cloud",), p) for P in tasks]) / 60, "-s", ms=4, color="#e0533d", label="All on server")
        ax[0].set_ylabel("Total latency [min]"); ax[0].set_title("Latency\n ", fontsize=11)
        ax[1].plot(n, np.cumsum(tasks), "-s", ms=4, color="#e0533d", label="All on server")
        ax[1].plot(n, np.cumsum([prompt_leak(c[0]) * P for P, c in zip(tasks, sol)]), "-D", ms=4, lw=2.2,
                   color="#1a9e55", label="Optimal split + noise")
        ax[1].plot(n, np.zeros(len(n)), "-o", ms=4, color="#2a6fdb", label="All on client")
        ax[1].set_ylabel("Prompt content tokens recovered"); ax[1].set_title(f"Prompt leakage\n{tag}", fontsize=11)
        for a in ax:
            a.set_xlabel("Number of tasks"); a.set_xticks(n[::2]); a.grid(alpha=.3)
            a.legend(frameon=False, fontsize=9, loc="upper left")
        fig.tight_layout(); fig.savefig(f"{out}/step3b_optimal_vs_baselines.png", dpi=200)

    return tasks, sol, sweep, loc_L


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    tasks, sol, sweep, loc_L = run(out=out)
    p = PARAMS
    print("Leakage and quality: MEASURED (noise sweep, Llama-3.2-1B, adaptive attacker); tail public.\n")
    print(f"Limits: prompt leakage <= {100*EPS_MAIN:.1f}%, quality >= {100*Q_MAIN:.0f}%, client memory <= {M_CLIENT_GB} GB\n")
    if not all(sol):
        print("INFEASIBLE for some tasks: lower Q_MAIN or raise EPS_MAIN.")
    else:
        print("Task  prompt  chosen                 latency   prompt leak  private exact  quality")
        for i, (P, c) in enumerate(zip(tasks, sol), 1):
            s, L = c
            print(f"{i:4d}  {P:6d}  {name(s):22s} {L:7.1f} s  {100*prompt_leak(s):8.1f}%  "
                  f"{100*private_exact(s):10.1f}%  {100*quality(s):6.1f}%")
        L = sum(c[1] for c in sol)
        print(f"\nTotal latency: {L/60:.1f} min (all-local {loc_L/60:.1f} min)")
    print("\nLowest feasible prompt-privacy limit for each quality limit:")
    for q in Q_LEVELS:
        f = [x for x in sweep[q] if x[1] is not None]
        print(f"  quality >= {100*q:3.0f}%:  eps >= {100*f[0][0]:.2f}%  ({f[0][1]/60:.1f} min)" if f else
              f"  quality >= {100*q:3.0f}%:  never feasible")
