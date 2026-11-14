"""
Baselines for: Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Week-1 analytic model, in the style of Zamzam, Elshabrawy & Ashour (IEEE Access 2021):
    processing time = (work) / (device capability)
    transmission time = (data size) / (data rate)
Work is counted in FLOPs from the real architecture of Llama-3.2-1B-Instruct.
Later the FLOP-based times can be replaced by per-layer times measured on real hardware.

Two baselines (the paper's "local computing" / "full offloading" schemes):
    B1  all-local : the whole model runs on the client            -> slow, nothing leaks
    B2  all-cloud : the client sends the prompt text to the server -> fast, everything leaks
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
D_MODEL    = 2048          # hidden size
N_HEADS    = 32
N_KV_HEADS = 8
HEAD_DIM   = D_MODEL // N_HEADS   # 64
D_FF       = 8192          # MLP intermediate size
VOCAB      = 128_256

# parameters (weights) in one decoder layer
P_ATTN  = D_MODEL * D_MODEL * 2 + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2   # q,o + k,v
P_MLP   = 3 * D_MODEL * D_FF                                              # gate, up, down
P_LAYER = P_ATTN + P_MLP
P_HEAD  = D_MODEL * VOCAB                                                  # LM head


def layer_flops(n_new, ctx_start):
    """FLOPs for ONE decoder layer processing n_new tokens whose positions start at ctx_start.
    Linear layers: 2 FLOPs per weight per token. Attention: QK^T and AV, 2*2*ctx*d per token."""
    linear = 2 * P_LAYER * n_new
    pos = np.arange(ctx_start + 1, ctx_start + n_new + 1)      # context length seen by each token
    attn = 4 * D_MODEL * pos.sum()
    return linear + attn


def task_flops(prompt_len, answer_len, layers=N_LAYERS, with_head=True):
    """Total FLOPs to run `layers` decoder layers (+ LM head) for one task:
    prefill of the prompt, then answer_len decode steps (one token at a time)."""
    prefill = layers * layer_flops(prompt_len, 0)
    decode = sum(layers * layer_flops(1, prompt_len + k) for k in range(answer_len))
    head = 2 * P_HEAD * (1 + answer_len) if with_head else 0   # one logit vector per generated token
    return prefill + decode + head


# ----------------------------------------------------------------------------------
# 2. System parameters (Table 1). Change here, everything else follows.
# ----------------------------------------------------------------------------------
PARAMS = dict(
    n_tasks          = 20,        # tasks per client
    prompt_min       = 32,        # prompt length range in tokens (tasks differ in prompt length)
    prompt_max       = 512,
    answer_len       = 128,       # generated tokens per task
    f_client         = 10e9,      # client effective capability [FLOP/s]  (laptop/phone CPU)
    f_server         = 1e12,      # server effective capability [FLOP/s]  (GPU)
    rate_up          = 20e6,      # client -> server data rate [bit/s]
    rate_down        = 50e6,      # server -> client data rate [bit/s]
    rtt              = 0.050,     # round-trip time [s]
    bytes_per_token_text = 4,     # token id as int32 (text sent in all-cloud)
    seed             = 7,
)


def make_tasks(p):
    rng = np.random.default_rng(p["seed"])
    return rng.integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


# ----------------------------------------------------------------------------------
# 3. Latency of each baseline for one task
# ----------------------------------------------------------------------------------
def latency_local(P, p):
    """B1: everything on the client. L = W_total / f_client (no network)."""
    W = task_flops(P, p["answer_len"])
    return dict(total=W / p["f_client"], compute_client=W / p["f_client"],
                compute_server=0.0, network=0.0)


def latency_cloud(P, p):
    """B2: client sends the prompt, server runs all layers and the LM head, returns the answer.
    L = RTT + send(prompt) + W_total / f_server + receive(answer)."""
    W = task_flops(P, p["answer_len"])
    t_up = P * p["bytes_per_token_text"] * 8 / p["rate_up"]
    t_down = p["answer_len"] * p["bytes_per_token_text"] * 8 / p["rate_down"]
    t_srv = W / p["f_server"]
    net = p["rtt"] + t_up + t_down
    return dict(total=net + t_srv, compute_client=0.0, compute_server=t_srv, network=net)


# ----------------------------------------------------------------------------------
# 4. Privacy: what the server can recover
#    attack success rate (ASR) = fraction of the client's tokens the server recovers.
#    all-local -> server sees nothing -> ASR = 0
#    all-cloud -> server sees the prompt and the answer in plain text -> ASR = 1
#    exposed tokens = ASR x (prompt + answer tokens), summed over tasks
# ----------------------------------------------------------------------------------
def leaked_tokens(P, p, asr):
    return asr * (P + p["answer_len"])


# ----------------------------------------------------------------------------------
# 5. Run, save, plot
# ----------------------------------------------------------------------------------
def run(p=PARAMS, out="."):
    tasks = make_tasks(p)
    rows = []
    for i, P in enumerate(tasks, 1):
        lo, cl = latency_local(P, p), latency_cloud(P, p)
        rows.append(dict(task=i, prompt_tokens=int(P), answer_tokens=p["answer_len"],
                         gflop=task_flops(P, p["answer_len"]) / 1e9,
                         local_s=lo["total"], cloud_s=cl["total"],
                         cloud_network_s=cl["network"], cloud_server_s=cl["compute_server"],
                         local_leaked=leaked_tokens(P, p, 0.0),
                         cloud_leaked=leaked_tokens(P, p, 1.0)))

    with open(f"{out}/baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    n = np.arange(1, len(rows) + 1)
    cum_local = np.cumsum([r["local_s"] for r in rows])
    cum_cloud = np.cumsum([r["cloud_s"] for r in rows])
    cum_leak_local = np.cumsum([r["local_leaked"] for r in rows])
    cum_leak_cloud = np.cumsum([r["cloud_leaked"] for r in rows])

    C_LOCAL, C_CLOUD = "#2a6fdb", "#e0533d"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].plot(n, cum_local / 60, "-o", ms=4, color=C_LOCAL, label="All on client (local)")
    ax[0].plot(n, cum_cloud / 60, "-s", ms=4, color=C_CLOUD, label="All on server (cloud)")
    ax[0].set_xlabel("Number of tasks")
    ax[0].set_ylabel("Total latency [min]")
    ax[0].set_title("Latency")
    ax[0].set_xticks(n[::2] if len(n) > 10 else n)
    ax[0].grid(alpha=.3); ax[0].legend(frameon=False)
    ax[0].annotate(f"{cum_local[-1]/60:.1f} min", (n[-1], cum_local[-1]/60),
                   textcoords="offset points", xytext=(-10, 8), ha="right", color=C_LOCAL)
    ax[0].annotate(f"{cum_cloud[-1]:.1f} s", (n[-1], cum_cloud[-1]/60),
                   textcoords="offset points", xytext=(-10, 8), ha="right", color=C_CLOUD)

    ax[1].plot(n, cum_leak_cloud, "-s", ms=4, color=C_CLOUD, label="All on server (cloud), ASR = 100%")
    ax[1].plot(n, cum_leak_local, "-o", ms=4, color=C_LOCAL, label="All on client (local), ASR = 0%")
    ax[1].set_xlabel("Number of tasks")
    ax[1].set_ylabel("Tokens recovered by the server")
    ax[1].set_title("Privacy leakage")
    ax[1].set_xticks(n[::2] if len(n) > 10 else n)
    ax[1].grid(alpha=.3); ax[1].legend(frameon=False, loc="upper left")

    fig.suptitle(f"Baselines: Llama-3.2-1B, {p['n_tasks']} tasks, prompts {p['prompt_min']}–{p['prompt_max']} "
                 f"tokens, {p['answer_len']}-token answers", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/baselines_latency_privacy.png", dpi=200)

    # client capability sweep (the second ticked metric)
    caps = np.logspace(9, 12, 40)   # 1 GFLOP/s .. 1 TFLOP/s
    tot_local = [sum(latency_local(P, {**p, "f_client": c})["total"] for P in tasks) for c in caps]
    tot_cloud = sum(latency_cloud(P, p)["total"] for P in tasks)
    fig2, a = plt.subplots(figsize=(6.4, 4.4))
    a.loglog(caps / 1e9, tot_local, color=C_LOCAL, label="All on client (local)")
    a.axhline(tot_cloud, color=C_CLOUD, ls="--", label="All on server (cloud)")
    a.axvline(p["f_client"] / 1e9, color="grey", ls=":", lw=1)
    a.text(p["f_client"] / 1e9 * 1.08, a.get_ylim()[1] * 0.5 if False else max(tot_local) * 0.5,
           "default client", color="grey", fontsize=9)
    a.set_xlabel("Client capability [GFLOP/s]")
    a.set_ylabel(f"Total latency for {p['n_tasks']} tasks [s]")
    a.set_title("Effect of client capability")
    a.grid(alpha=.3, which="both"); a.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(f"{out}/baselines_client_capability.png", dpi=200)

    return rows, cum_local[-1], cum_cloud[-1]


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    rows, L, C = run(out=out)
    print(f"params/layer = {P_LAYER/1e6:.2f} M, LM head = {P_HEAD/1e6:.1f} M, "
          f"total = {(N_LAYERS*P_LAYER + P_HEAD)/1e9:.3f} B")
    for r in rows:
        print(f"task {r['task']:2d}  P={r['prompt_tokens']:3d}  {r['gflop']:7.1f} GFLOP  "
              f"local={r['local_s']:6.1f}s  cloud={r['cloud_s']:.3f}s")
    print(f"TOTAL 20 tasks: local = {L:.1f} s ({L/60:.1f} min), cloud = {C:.2f} s")
