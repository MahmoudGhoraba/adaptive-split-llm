"""
STEP 3 - Optimisation: the best split under a privacy limit
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script (contains its own copy of the Step 1 latency model and the Step 2 leakage
model, with the same parameters), so it runs without the other files.

PROBLEM (written like Section IV of Zamzam, Elshabrawy & Ashour, IEEE Access 2021)

  Decision variable  x(i, s) in {0, 1}: task i runs with scheme s, where s is
                     all-local, all-cloud, or a U-shaped split (H, U, T)  (122 options)

  minimise   L = sum_i sum_s x(i, s) * L(i, s)                 total latency of all tasks

  subject to
    C1 privacy  : sum_s x(i, s) * leak(i, s) <= eps             for every task i
                  (leak = share of the task's content tokens the server recovers)
    C2 deadline : sum_s x(i, s) * L(i, s)    <= deadline        for every task i  (optional)
    C3 memory   : memory needed on the client by scheme s <= M_client
                  (weights of the client layers + embedding/LM head + their KV cache)
    C4 one choice: sum_s x(i, s) = 1                            for every task i

  The constraints do not link different tasks, so each task is solved on its own by
  checking all 122 options and keeping the fastest feasible one (exhaustive search, the
  same idea as the paper's ILLOO benchmark; it is exact and cheap with only 16 layers).

!!! The leakage values are the Step 2 PLACEHOLDERS until the attack results are entered
    in MEASURED_ASR_IN / MEASURED_ASR_OUT. !!!
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# 1. Model: Llama-3.2-1B-Instruct
# ----------------------------------------------------------------------------------
N_LAYERS   = 16
D_MODEL    = 2048
N_HEADS    = 32
N_KV_HEADS = 8
HEAD_DIM   = D_MODEL // N_HEADS
D_FF       = 8192
VOCAB      = 128_256

P_LAYER = (D_MODEL * D_MODEL * 2 + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2 + 3 * D_MODEL * D_FF)
P_HEAD  = D_MODEL * VOCAB          # LM head (tied with the embedding table in Llama-3.2-1B)


def layer_flops(n_new, ctx_start):
    pos = np.arange(ctx_start + 1, ctx_start + n_new + 1)
    return 2 * P_LAYER * n_new + 4 * D_MODEL * pos.sum()


# ----------------------------------------------------------------------------------
# 2. Parameters (same as Steps 1 and 2) + optimisation settings
# ----------------------------------------------------------------------------------
PARAMS = dict(
    n_tasks          = 20,
    prompt_min       = 32,
    prompt_max       = 512,
    answer_len       = 128,
    f_client         = 10e9,     # [FLOP/s]
    f_server         = 1e12,     # [FLOP/s]
    server_share     = 1.0,
    rate_up          = 20e6,     # [bit/s]
    rate_down        = 50e6,     # [bit/s]
    rtt              = 0.050,    # [s]
    bytes_per_token_text   = 4,
    bytes_per_value        = 2,  # fp16 weights, hidden states and KV cache
    seed             = 7,
)

EPS_MAIN    = 0.995     # privacy limit for the main plots. With measured leakage nothing below ~99.5% is
                        # feasible (splitting alone leaks 96-100%), so 99.5% is the strictest workable limit
EPS_SWEEP   = np.round(np.arange(0.20, 1.0001, 0.005), 3)
M_CLIENT_GB = 2.0       # client memory available for the model [GB]
DEADLINE_S  = None      # per-task deadline in seconds, or None for no deadline

# ---- leakage model (identical to Step 2) ---------------------------------------------
TAIL_WEIGHTS_PRIVATE = False   # public tail weights: server can run the tail itself
ASR_IN_START,  ASR_IN_FLOOR,  ASR_IN_DECAY  = 1.00, 0.15, 0.30
ASR_OUT_START, ASR_OUT_FLOOR, ASR_OUT_DECAY = 1.00, 0.10, 0.35
# MEASURED on Llama-3.2-1B-Instruct (inversion_attack.py, Colab, 2026-10-30):
# content-token recovery above a blind guess, MLP attacker, 40k held-out tokens (CI about +-0.3 points).
# The out-side attacker failed its sanity check (T=0 gave 30%, expected ~100%), so the tail is treated
# as PUBLIC weights: the server can run it itself -> ASR_out = 1 for every T.
MEASURED_ASR_IN  = {1: 0.9997, 2: 0.9978, 3: 0.9906, 4: 0.9827, 5: 0.978, 6: 0.9729, 7: 0.9754, 8: 0.9757,
                    9: 0.979, 10: 0.9743, 11: 0.9747, 12: 0.974, 13: 0.9723, 14: 0.9657, 15: 0.9643}
MEASURED_ASR_OUT = {}
USING_PLACEHOLDER = not MEASURED_ASR_IN


def asr_in(H):
    if H in MEASURED_ASR_IN:
        return MEASURED_ASR_IN[H]
    return ASR_IN_FLOOR + (ASR_IN_START - ASR_IN_FLOOR) * np.exp(-ASR_IN_DECAY * H)


def asr_out(T):
    if not TAIL_WEIGHTS_PRIVATE:
        return 1.0
    if T in MEASURED_ASR_OUT:
        return MEASURED_ASR_OUT[T]
    return ASR_OUT_FLOOR + (ASR_OUT_START - ASR_OUT_FLOOR) * np.exp(-ASR_OUT_DECAY * T)


# ----------------------------------------------------------------------------------
# 3. Schemes: latency, leakage and client memory
# ----------------------------------------------------------------------------------
def make_tasks(p):
    rng = np.random.default_rng(p["seed"])
    return rng.integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


def schemes():
    """All options: ('local',), ('cloud',), ('split', H, T)."""
    out = [("local",), ("cloud",)]
    out += [("split", H, T) for H in range(1, N_LAYERS) for T in range(0, N_LAYERS - H)]
    return out


def name(s):
    if s[0] != "split":
        return "All on client" if s[0] == "local" else "All on server"
    H, T = s[1], s[2]
    return f"H={H}, U={N_LAYERS-H-T}, T={T}"


_W_CACHE = {}


def work_one_layer(P, A):
    """FLOPs of one decoder layer for one task (prefill + A decode steps); cached."""
    key = (int(P), int(A))
    if key not in _W_CACHE:
        _W_CACHE[key] = layer_flops(P, 0) + sum(layer_flops(1, P + k) for k in range(A))
    return _W_CACHE[key]


def latency(P, s, p):
    A = p["answer_len"]
    w = work_one_layer(P, A)
    head = 2 * P_HEAD * (1 + A)
    if s[0] == "local":
        return (N_LAYERS * w + head) / p["f_client"]
    if s[0] == "cloud":
        tr = (P * p["bytes_per_token_text"] * 8 / p["rate_up"]
              + A * p["bytes_per_token_text"] * 8 / p["rate_down"])
        return (N_LAYERS * w + head) / (p["f_server"] * p["server_share"]) + tr + p["rtt"]
    H, T = s[1], s[2]
    U = N_LAYERS - H - T
    client = ((H + T) * w + head) / p["f_client"]
    server = U * w / (p["f_server"] * p["server_share"])
    transfer = (P + A) * D_MODEL * p["bytes_per_value"] * 8 * (1 / p["rate_up"] + 1 / p["rate_down"])
    return client + server + transfer + (1 + A) * p["rtt"]


def leak_rate(P, s, p):
    """Share of the task's content tokens the server recovers."""
    A = p["answer_len"]
    if s[0] == "local":
        return 0.0
    if s[0] == "cloud":
        return 1.0
    a_in, a_out = asr_in(s[1]), asr_out(s[2])
    leaked = a_in * P + (1 - (1 - a_in) * (1 - a_out)) * A
    return leaked / (P + A)


def client_memory_gb(P, s, p):
    """Weights of the client's layers + embedding/LM head + KV cache of the client's layers."""
    if s[0] == "cloud":
        return 0.0
    layers = N_LAYERS if s[0] == "local" else s[1] + s[2]
    kv_per_token_layer = 2 * N_KV_HEADS * HEAD_DIM * p["bytes_per_value"]
    ctx = P + p["answer_len"]
    bytes_ = (layers * P_LAYER + P_HEAD) * p["bytes_per_value"] + layers * kv_per_token_layer * ctx
    return bytes_ / 1e9


# ----------------------------------------------------------------------------------
# 4. Solver: exhaustive search per task (exact)
# ----------------------------------------------------------------------------------
def solve(tasks, eps, p, m_client=M_CLIENT_GB, deadline=DEADLINE_S):
    choice = []
    for P in tasks:
        best = None
        for s in schemes():
            L = latency(P, s, p)
            if leak_rate(P, s, p) > eps + 1e-12:                     # C1 privacy
                continue
            if deadline is not None and L > deadline:                # C2 deadline
                continue
            if client_memory_gb(P, s, p) > m_client:                 # C3 memory
                continue
            if best is None or L < best[1]:
                best = (s, L)
        choice.append(best)                                          # None = infeasible
    return choice


# ----------------------------------------------------------------------------------
# 5. Run, save, plot
# ----------------------------------------------------------------------------------
def run(p=PARAMS, out="."):
    tasks = make_tasks(p)
    S = schemes()
    tag = ("PLACEHOLDER leakage (not measured)" if USING_PLACEHOLDER else
           "Measured leakage, Llama-3.2-1B (MLP inversion); tail weights public")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    C_LOCAL, C_CLOUD, C_OPT = "#2a6fdb", "#e0533d", "#1a9e55"

    # totals over 20 tasks for every scheme (used by the trade-off plot)
    tot = []
    for s in S:
        L = sum(latency(P, s, p) for P in tasks)
        lk = sum(leak_rate(P, s, p) * (P + p["answer_len"]) for P in tasks) / sum(P + p["answer_len"] for P in tasks)
        mem = max(client_memory_gb(P, s, p) for P in tasks)
        tot.append(dict(s=s, L=L, leak=lk, mem=mem, fits=mem <= M_CLIENT_GB))

    # main solution
    sol = solve(tasks, EPS_MAIN, p)
    bad = [i + 1 for i, c in enumerate(sol) if c is None]
    if bad:
        raise SystemExit(f"eps = {100*EPS_MAIN:.1f}% is infeasible for tasks {bad}: no split meets the "
                         f"privacy limit within {M_CLIENT_GB} GB of client memory. Raise EPS_MAIN or M_CLIENT_GB.")
    with open(f"{out}/optimal_split_per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "prompt_tokens", "chosen_scheme", "H", "U", "T", "latency_s",
                    "leak_rate_percent", "client_memory_GB", "local_latency_s", "cloud_latency_s"])
        for i, (P, c) in enumerate(zip(tasks, sol), 1):
            if c is None:
                w.writerow([i, P, "INFEASIBLE"] + [""] * 8); continue
            s, L = c
            H, T = (s[1], s[2]) if s[0] == "split" else ("", "")
            U = N_LAYERS - s[1] - s[2] if s[0] == "split" else ""
            w.writerow([i, P, name(s), H, U, T, round(L, 3), round(100 * leak_rate(P, s, p), 2),
                        round(client_memory_gb(P, s, p), 3),
                        round(latency(P, ("local",), p), 3), round(latency(P, ("cloud",), p), 3)])

    # sweep over eps
    sweep = []
    for e in EPS_SWEEP:
        so = solve(tasks, e, p)
        if any(c is None for c in so):
            sweep.append((e, None, None, None)); continue
        L = sum(c[1] for c in so)
        lk = sum(leak_rate(P, c[0], p) * (P + p["answer_len"]) for P, c in zip(tasks, so)) / \
             sum(P + p["answer_len"] for P in tasks)
        common = max(set(name(c[0]) for c in so), key=lambda n: sum(name(c[0]) == n for c in so))
        sweep.append((e, L, lk, common))
    with open(f"{out}/optimal_vs_privacy_limit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps_percent", "total_latency_s", "achieved_leak_percent", "most_used_split"])
        for e, L, lk, c in sweep:
            w.writerow([round(100 * e), "" if L is None else round(L, 2),
                        "" if lk is None else round(100 * lk, 2), c or "INFEASIBLE"])

    # --- Figure 1: trade-off (every scheme), Pareto front, eps line, chosen point ----------
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sp = [t for t in tot if t["s"][0] == "split"]
    ax.scatter([t["L"] / 60 for t in sp if t["fits"]], [100 * t["leak"] for t in sp if t["fits"]],
               s=22, color="#9aa3ad", label="U-shaped splits (fit in client memory)", zorder=2)
    ax.scatter([t["L"] / 60 for t in sp if not t["fits"]], [100 * t["leak"] for t in sp if not t["fits"]],
               s=22, facecolors="none", edgecolors="#9aa3ad",
               label=f"Splits that need > {M_CLIENT_GB:g} GB on the client", zorder=2)
    loc = next(t for t in tot if t["s"][0] == "local"); cl = next(t for t in tot if t["s"][0] == "cloud")
    ax.scatter([loc["L"] / 60], [0], s=90, marker="o", color=C_LOCAL, zorder=4,
               label=f"All on client ({loc['mem']:.1f} GB: does not fit)" if not loc["fits"] else "All on client")
    ax.scatter([cl["L"] / 60], [100], s=90, marker="s", color=C_CLOUD, zorder=4, label="All on server")
    feas = sorted([t for t in tot if t["fits"]], key=lambda t: t["L"])
    front, best_leak = [], 1e9
    for t in feas:
        if t["leak"] < best_leak - 1e-12:
            front.append(t); best_leak = t["leak"]
    ax.plot([t["L"] / 60 for t in front], [100 * t["leak"] for t in front], "-", color=C_OPT, lw=1.6,
            zorder=3, label="Best trade-off (Pareto front)")
    ax.axhline(100 * EPS_MAIN, color=C_OPT, ls="--", lw=1)
    ax.text(0.3, 100 * EPS_MAIN + (1.5 if USING_PLACEHOLDER else -5.5),
            f"privacy limit ε = {100*EPS_MAIN:.1f}%", color=C_OPT, ha="left", fontsize=9)
    chosen = min([t for t in feas if t["leak"] <= EPS_MAIN], key=lambda t: t["L"])
    ax.scatter([chosen["L"] / 60], [100 * chosen["leak"]], s=420, facecolors="none",
               edgecolors=C_OPT, lw=2.2, zorder=5)
    ax.annotate(f"best single split for all tasks:\n{name(chosen['s'])}, "
                f"{chosen['L']/60:.1f} min, {100*chosen['leak']:.0f}% leaked",
                (chosen["L"] / 60, 100 * chosen["leak"]), xytext=(60, 95) if USING_PLACEHOLDER else (50, -110),
                textcoords="offset points",
                fontsize=9.5, color=C_OPT, arrowprops=dict(arrowstyle="->", color=C_OPT))
    ax.set_xlabel(f"Total latency for {len(tasks)} tasks [min]")
    ax.set_ylabel("Content tokens recovered by the server [%]")
    ax.set_title(f"Latency vs privacy for every scheme\n{tag}", fontsize=11)
    ax.set_ylim(-4, 106)
    ax.grid(alpha=.3)
    ax.legend(frameon=True, framealpha=.95, edgecolor="none", fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/step3_tradeoff.png", dpi=200)

    # --- Figure 2: optimal latency vs privacy limit -----------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xs = [100 * e for e, L, _, _ in sweep if L is not None]
    ys = [L / 60 for e, L, _, _ in sweep if L is not None]
    ax.step(xs, ys, where="post", color=C_OPT, lw=2, label="Optimal split (this work)")
    ax.axhline(loc["L"] / 60, color=C_LOCAL, ls="--", label="All on client")
    ax.axhline(cl["L"] / 60, color=C_CLOUD, ls="--", label="All on server (allowed only at ε = 100%)")
    infeas = [100 * e for e, L, _, _ in sweep if L is None]
    if infeas:
        ax.axvspan(min(infeas) - 0.5, max(infeas) + 0.5, color="#f3d7d7", alpha=.6, lw=0)
        ax.text((min(infeas) + max(infeas)) / 2, ax.get_ylim()[1] * 0.5, "no feasible\nsplit",
                ha="center", color="#a33", fontsize=9)
    feas_e = [x[0] for x in sweep if x[1] is not None]
    marks = (0.5, 0.7, 0.9) if USING_PLACEHOLDER else (feas_e[0],) if feas_e else ()
    for e_mark in marks:
        hit = next((x for x in sweep if abs(x[0] - e_mark) < 1e-9 and x[1] is not None), None)
        if hit:
            ax.annotate(f"ε={100*e_mark:.1f}%: mostly\n{hit[3]}", (100 * e_mark, hit[1] / 60),
                        xytext=(-60, 30), textcoords="offset points", ha="center", fontsize=8,
                        color="#333", arrowprops=dict(arrowstyle="-", color="#999", lw=.8))
    ax.set_xlabel("Privacy limit ε: max. share of content tokens the server may recover [%]")
    ax.set_ylabel(f"Total latency for {len(tasks)} tasks [min]")
    ax.set_title(f"Price of privacy: optimal latency for each privacy limit\n{tag}", fontsize=11)
    ax.grid(alpha=.3)
    ax.legend(frameon=True, framealpha=.95, edgecolor="none", fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.92))
    fig.tight_layout()
    fig.savefig(f"{out}/step3_latency_vs_privacy_limit.png", dpi=200)

    # --- Figure 3: the sketch, with the optimum added ---------------------------------------
    n = np.arange(1, len(tasks) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
    opt_L = np.cumsum([c[1] for c in sol]) / 60
    opt_K = np.cumsum([leak_rate(P, c[0], p) * (P + p["answer_len"]) for P, c in zip(tasks, sol)])
    loc_L = np.cumsum([latency(P, ("local",), p) for P in tasks]) / 60
    cl_L = np.cumsum([latency(P, ("cloud",), p) for P in tasks]) / 60
    cl_K = np.cumsum([P + p["answer_len"] for P in tasks])
    ax[0].plot(n, loc_L, "-o", ms=4, color=C_LOCAL, label="All on client")
    ax[0].plot(n, opt_L, "-D", ms=4, color=C_OPT, lw=2.2, label=f"Optimal split, ε = {100*EPS_MAIN:.1f}%")
    ax[0].plot(n, cl_L, "-s", ms=4, color=C_CLOUD, label="All on server")
    ax[0].set_ylabel("Total latency [min]"); ax[0].set_title("Latency\n ", fontsize=11)
    ax[1].plot(n, cl_K, "-s", ms=4, color=C_CLOUD, label="All on server")
    ax[1].plot(n, opt_K, "-D", ms=4, color=C_OPT, lw=2.2, label=f"Optimal split, ε = {100*EPS_MAIN:.1f}%")
    ax[1].plot(n, np.zeros_like(n), "-o", ms=4, color=C_LOCAL, label="All on client")
    ax[1].set_ylabel("Content tokens recovered by the server"); ax[1].set_title(f"Privacy leakage\n{tag}", fontsize=11)
    for a in ax:
        a.set_xlabel("Number of tasks"); a.set_xticks(n[::2]); a.grid(alpha=.3)
        a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/step3_optimal_vs_baselines.png", dpi=200)

    return tasks, sol, sweep, tot, chosen


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    tasks, sol, sweep, tot, chosen = run(out=out)
    p = PARAMS
    if USING_PLACEHOLDER:
        print("NOTE: leakage values are PLACEHOLDERS, not measured attack results.\n")
    else:
        print("Leakage: MEASURED prompt side (Llama-3.2-1B), tail treated as public (ASR_out = 1).\n")
    loc = next(t for t in tot if t["s"][0] == "local")
    print(f"Client memory limit: {M_CLIENT_GB} GB  (whole model needs {loc['mem']:.2f} GB -> "
          f"{'fits' if loc['fits'] else 'all-local NOT allowed'})")
    print(f"Privacy limit eps = {100*EPS_MAIN:.1f}%\n")
    print("Task  prompt  chosen split           latency   leaked")
    for i, (P, c) in enumerate(zip(tasks, sol), 1):
        if c is None:
            print(f"{i:4d}  {P:6d}  INFEASIBLE"); continue
        print(f"{i:4d}  {P:6d}  {name(c[0]):22s} {c[1]:7.1f} s  {100*leak_rate(P, c[0], p):5.1f}%")
    L = sum(c[1] for c in sol if c)
    print(f"\nTotal latency, optimal : {L:8.1f} s ({L/60:.1f} min)")
    print(f"Total latency, all-local: {loc['L']:8.1f} s ({loc['L']/60:.1f} min)")
    print(f"Optimal is {100*(1 - L/loc['L']):.0f}% faster than all-local while leaking at most "
          f"{100*EPS_MAIN:.1f}% (all-cloud leaks 100%).")
    feas = [s for s in sweep if s[1] is not None]
    if len(feas) < len(sweep):
        print(f"No feasible split below eps = {100*feas[0][0]:.1f}% (memory limit + leakage floor).")
