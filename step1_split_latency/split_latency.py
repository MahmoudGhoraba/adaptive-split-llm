"""
STEP 1 - Latency of the U-shaped split
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script (does not need baselines.py). Same model and same parameters as the
baselines, so the two baselines are the end points of every plot here.

U-shaped split of Llama-3.2-1B (16 decoder layers):

    client:  embedding + H first layers          (head, H)
    server:  U = 16 - H - T middle layers         (U)
    client:  T last layers + final norm + LM head (tail, T)

The LM head always stays on the client, so the server never sees the prompt text or the
answer text, only hidden states. H >= 1 and U >= 1; T can be 0 (then only the LM head is
on the client after the server). That gives 120 possible splits.

Latency of one task (prompt of P tokens, answer of A tokens), Zamzam et al. style:
    processing   = FLOPs / capability
    transmission = bytes * 8 / data rate
    every exchange with the server also costs one RTT

  Prefill (whole prompt at once):
    client runs H layers on P tokens
    -> send P hidden states up -> server runs U layers -> send P hidden states down
    -> client runs T layers + LM head            (1 round trip)
  Decode (one round trip per generated token):
    client runs H layers on 1 token -> up -> server U layers -> down -> client T + LM head

  L_split = client compute + server compute + transfer time + (1 + A) * RTT
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# 1. Model: Llama-3.2-1B-Instruct (from its config.json)
# ----------------------------------------------------------------------------------
N_LAYERS   = 16
D_MODEL    = 2048
N_HEADS    = 32
N_KV_HEADS = 8
HEAD_DIM   = D_MODEL // N_HEADS
D_FF       = 8192
VOCAB      = 128_256

P_LAYER = (D_MODEL * D_MODEL * 2 + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2   # attention q,o,k,v
           + 3 * D_MODEL * D_FF)                                            # MLP
P_HEAD  = D_MODEL * VOCAB                                                   # LM head


def layer_flops(n_new, ctx_start):
    """FLOPs for one decoder layer processing n_new tokens starting at position ctx_start."""
    pos = np.arange(ctx_start + 1, ctx_start + n_new + 1)
    return 2 * P_LAYER * n_new + 4 * D_MODEL * pos.sum()


# ----------------------------------------------------------------------------------
# 2. System parameters (same as the baselines)
# ----------------------------------------------------------------------------------
PARAMS = dict(
    n_tasks          = 20,
    prompt_min       = 32,
    prompt_max       = 512,
    answer_len       = 128,
    f_client         = 10e9,      # client capability [FLOP/s]
    f_server         = 1e12,      # server capability [FLOP/s]
    server_share     = 1.0,       # fraction of the server available to this client (1 = idle server)
    rate_up          = 20e6,      # [bit/s]
    rate_down        = 50e6,      # [bit/s]
    rtt              = 0.050,     # [s]
    bytes_per_token_text   = 4,   # token id (all-cloud baseline)
    bytes_per_hidden_value = 2,   # fp16 hidden states (split)
    seed             = 7,
)

# splits drawn as curves in the main plot: (H, T)
SHOW_SPLITS = [(1, 0), (2, 1), (4, 2), (6, 4), (8, 6)]


def make_tasks(p):
    rng = np.random.default_rng(p["seed"])
    return rng.integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


def per_layer_work(P, A):
    """FLOPs of ONE layer for a task: prefill of P tokens + A decode steps."""
    prefill = layer_flops(P, 0)
    decode = sum(layer_flops(1, P + k) for k in range(A))
    return prefill + decode


# ----------------------------------------------------------------------------------
# 3. Latency models
# ----------------------------------------------------------------------------------
def latency_local(P, p):
    A = p["answer_len"]
    W = N_LAYERS * per_layer_work(P, A) + 2 * P_HEAD * (1 + A)
    t = W / p["f_client"]
    return dict(total=t, client=t, server=0.0, transfer=0.0, rtt=0.0)


def latency_cloud(P, p):
    A = p["answer_len"]
    W = N_LAYERS * per_layer_work(P, A) + 2 * P_HEAD * (1 + A)
    srv = W / (p["f_server"] * p["server_share"])
    tr = (P * p["bytes_per_token_text"] * 8 / p["rate_up"]
          + A * p["bytes_per_token_text"] * 8 / p["rate_down"])
    return dict(total=srv + tr + p["rtt"], client=0.0, server=srv, transfer=tr, rtt=p["rtt"])


def latency_split(P, H, T, p):
    """U-shaped split: H layers + T layers + LM head on the client, U layers on the server."""
    A = p["answer_len"]
    U = N_LAYERS - H - T
    assert H >= 1 and T >= 0 and U >= 1, (H, T)
    w = per_layer_work(P, A)

    client = ((H + T) * w + 2 * P_HEAD * (1 + A)) / p["f_client"]
    server = U * w / (p["f_server"] * p["server_share"])

    hidden_bytes = D_MODEL * p["bytes_per_hidden_value"]       # one token's hidden state
    tokens_sent = P + A                                          # prompt once, then 1 per decode step
    transfer = tokens_sent * hidden_bytes * 8 * (1 / p["rate_up"] + 1 / p["rate_down"])
    rtt = (1 + A) * p["rtt"]                                     # prefill + one per decode step

    return dict(total=client + server + transfer + rtt,
                client=client, server=server, transfer=transfer, rtt=rtt)


def all_splits():
    return [(H, T) for H in range(1, N_LAYERS) for T in range(0, N_LAYERS - H)]


# ----------------------------------------------------------------------------------
# 4. Run, save, plot
# ----------------------------------------------------------------------------------
def total(fn, tasks, *args):
    parts = [fn(P, *args) for P in tasks]
    return {k: sum(d[k] for d in parts) for k in parts[0]}


def run(p=PARAMS, out="."):
    tasks = make_tasks(p)
    base_local = total(latency_local, tasks, p)
    base_cloud = total(latency_cloud, tasks, p)

    # every split, 20 tasks
    results = []
    for H, T in all_splits():
        r = total(latency_split, tasks, H, T, p)
        results.append(dict(H=H, U=N_LAYERS - H - T, T=T, client_layers=H + T, **r))
    results.sort(key=lambda r: r["total"])

    with open(f"{out}/split_latency_results.csv", "w", newline="") as f:
        cols = ["scheme", "H", "U", "T", "client_layers",
                "total_s", "client_compute_s", "server_compute_s", "transfer_s", "rtt_s"]
        w = csv.writer(f)
        w.writerow(cols)
        for name, b in [("all-local", base_local), ("all-cloud", base_cloud)]:
            w.writerow([name, "", "", "", "", *(round(b[k], 3) for k in
                        ["total", "client", "server", "transfer", "rtt"])])
        for r in results:
            w.writerow(["split", r["H"], r["U"], r["T"], r["client_layers"],
                        *(round(r[k], 3) for k in ["total", "client", "server", "transfer", "rtt"])])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    C_LOCAL, C_CLOUD = "#2a6fdb", "#e0533d"
    greys = plt.cm.viridis(np.linspace(0.15, 0.85, len(SHOW_SPLITS)))
    n = np.arange(1, len(tasks) + 1)

    # --- Figure 1: cumulative latency vs number of tasks (the sketch) -----------------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(n, np.cumsum([latency_local(P, p)["total"] for P in tasks]) / 60, "-o", ms=4,
            color=C_LOCAL, lw=2.2, label="All on client (local)")
    for (H, T), c in zip(SHOW_SPLITS, greys):
        y = np.cumsum([latency_split(P, H, T, p)["total"] for P in tasks]) / 60
        ax.plot(n, y, "-", color=c, lw=1.8,
                label=f"Split H={H}, U={N_LAYERS-H-T}, T={T}  ({H+T} layer{'s' if H+T > 1 else ''} on client)")
    ax.plot(n, np.cumsum([latency_cloud(P, p)["total"] for P in tasks]) / 60, "-s", ms=4,
            color=C_CLOUD, lw=2.2, label="All on server (cloud)")
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Total latency [min]")
    ax.set_title("Latency: U-shaped splits between the two baselines")
    ax.set_xticks(n[::2])
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/step1_latency_vs_tasks.png", dpi=200)

    # --- Figure 2: heatmap of total latency over (H, T) --------------------------------
    grid = np.full((N_LAYERS, N_LAYERS), np.nan)          # rows T, cols H
    for r in results:
        grid[r["T"], r["H"]] = r["total"] / 60
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(grid, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xlim(0.5, N_LAYERS - 0.5); ax.set_ylim(-0.5, N_LAYERS - 1.5)
    ax.set_xlabel("H = layers at the start on the client")
    ax.set_ylabel("T = layers at the end on the client")
    ax.set_title(f"Total latency for {len(tasks)} tasks [min] (U = 16 − H − T on server)")
    ax.set_xticks(range(1, N_LAYERS)); ax.set_yticks(range(0, N_LAYERS - 1))
    for r in results:
        if r["H"] + r["T"] <= 15:
            ax.text(r["H"], r["T"], f"{r['total']/60:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white" if r["total"] / 60 > np.nanmean(grid) else "black")
    best = results[0]
    ax.scatter([best["H"]], [best["T"]], s=260, facecolors="none", edgecolors="red", lw=2)
    fig.colorbar(im, ax=ax, label="min")
    fig.tight_layout()
    fig.savefig(f"{out}/step1_latency_heatmap.png", dpi=200)

    # --- Figure 3: where the time goes ---------------------------------------------------
    labels = ["All on\nclient"] + [f"H={H},T={T}" for H, T in SHOW_SPLITS] + ["All on\nserver"]
    rows = [base_local] + [total(latency_split, tasks, H, T, p) for H, T in SHOW_SPLITS] + [base_cloud]
    parts = [("client", "Client compute", "#2a6fdb"), ("server", "Server compute", "#e0533d"),
             ("transfer", "Hidden-state / text transfer", "#f2b134"), ("rtt", "Round trips (RTT)", "#7d7d7d")]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bottom = np.zeros(len(rows))
    for key, name, col in parts:
        v = np.array([r[key] for r in rows]) / 60
        ax.bar(labels, v, bottom=bottom, color=col, label=name, width=0.65)
        bottom += v
    for i, b in enumerate(bottom):
        ax.text(i, b, f"{b:.1f}" if b >= 1 else f"{b*60:.0f} s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"Total latency for {len(tasks)} tasks [min]")
    ax.set_title("Where the time goes")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out}/step1_latency_breakdown.png", dpi=200)

    return tasks, base_local, base_cloud, results


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    tasks, bl, bc, res = run(out=out)
    print(f"All on client : {bl['total']:8.1f} s  ({bl['total']/60:.1f} min)")
    print(f"All on server : {bc['total']:8.1f} s")
    print("\nFastest 5 splits (20 tasks):")
    for r in res[:5]:
        print(f"  H={r['H']:2d} U={r['U']:2d} T={r['T']:2d}  {r['total']:7.1f} s  "
              f"(client {r['client']:.0f} s, server {r['server']:.1f} s, "
              f"transfer {r['transfer']:.1f} s, RTT {r['rtt']:.1f} s)")
    print("\nSlowest split:")
    r = res[-1]
    print(f"  H={r['H']:2d} U={r['U']:2d} T={r['T']:2d}  {r['total']:7.1f} s")
    print("\nCurves in the plot:")
    for H, T in SHOW_SPLITS:
        r = next(x for x in res if x["H"] == H and x["T"] == T)
        print(f"  H={H:2d} U={r['U']:2d} T={T:2d}  {r['total']:7.1f} s ({r['total']/60:.1f} min)")
