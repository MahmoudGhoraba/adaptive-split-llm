"""
STEP 2 - Privacy leakage of the U-shaped split
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script (does not need baselines.py or split_latency.py). Same tasks and same
U-shaped split as Step 1:

    client:  embedding + H first layers          (head, H)
    server:  U = 16 - H - T middle layers         (U)
    client:  T last layers + final norm + LM head (tail, T)

WHAT THE SERVER SEES
  (a) Going INTO the server: the hidden state after layer H, for every prompt token AND for
      every generated token (each answer token is fed back into the model at the next step).
      -> leaks the prompt and the answer, depending on H.
  (b) Coming OUT of the server: the hidden state after layer 16 - T. It is used to predict the
      next token, so it leaks the answer, depending on T.

PRIVACY METRIC (attack success rate, ASR)
  ASR = fraction of the user's CONTENT tokens (names, numbers, nouns, verbs; not "the", "is",
  "a" or punctuation) that the attacker recovers, ABOVE what a blind attacker that only knows
  word frequencies would guess. All-local -> 0, all-cloud -> 1 (the server reads the text).

      prompt token leaks if it is recovered from (a):     ASR_prompt = ASR_in(H)
      answer token leaks if recovered from (a) OR (b):    ASR_answer = 1 - (1 - ASR_in(H)) (1 - ASR_out(T))
      tokens recovered for one task = ASR_prompt * P + ASR_answer * A

!!! PLACEHOLDER !!!
  ASR_in(H) and ASR_out(T) below are ASSUMED curves with the shape reported in the literature
  (leakage falls as more layers stay on the client but does not reach zero; Cunningham 2026,
  Dong et al. 2025). They are NOT measurements. Replace MEASURED_ASR_IN / MEASURED_ASR_OUT with
  your own attack results (one value per H and per T) when you have them.

!!! ASSUMPTION ON THE TAIL !!!
  If the T tail layers use the public Llama weights, the server can run them itself on what it
  sends back, so T protects nothing (ASR_out = 1 for every T). The tail only helps if its
  weights are private to the client (e.g. private LoRA adapters, see PrivateLoRA / FL-LLaMA).
  Set TAIL_WEIGHTS_PRIVATE below to choose.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_LAYERS = 16   # Llama-3.2-1B

# ----------------------------------------------------------------------------------
# 1. Parameters (tasks are identical to the baselines and Step 1)
# ----------------------------------------------------------------------------------
PARAMS = dict(
    n_tasks    = 20,
    prompt_min = 32,
    prompt_max = 512,
    answer_len = 128,
    seed       = 7,
)

TAIL_WEIGHTS_PRIVATE = False    # False -> server can recompute the tail, ASR_out = 1 always

# Placeholder curve settings: ASR = floor + (start - floor) * exp(-decay * layers)
ASR_IN_START,  ASR_IN_FLOOR,  ASR_IN_DECAY  = 1.00, 0.15, 0.30   # H = 0 would be raw embeddings
ASR_OUT_START, ASR_OUT_FLOOR, ASR_OUT_DECAY = 1.00, 0.10, 0.35   # T = 0: server output = final state

# MEASURED on Llama-3.2-1B-Instruct (inversion_attack.py, Colab, 2026-10-30):
# content-token recovery above a blind guess, MLP attacker, 40k held-out tokens (CI about +-0.3 points).
# The out-side attacker failed its sanity check (T=0 gave 30%, expected ~100%), so the tail is treated
# as PUBLIC weights: the server can run it itself -> ASR_out = 1 for every T.
MEASURED_ASR_IN  = {1: 0.9997, 2: 0.9978, 3: 0.9906, 4: 0.9827, 5: 0.978, 6: 0.9729, 7: 0.9754, 8: 0.9757,
                    9: 0.979, 10: 0.9743, 11: 0.9747, 12: 0.974, 13: 0.9723, 14: 0.9657, 15: 0.9643}
MEASURED_ASR_OUT = {}
USING_PLACEHOLDER = not MEASURED_ASR_IN

SHOW_SPLITS = [(1, 0), (2, 1), (4, 2), (6, 4), (8, 6)]   # same curves as Step 1


def asr_in(H):
    """Content-token recovery from the hidden state the client sends after H layers."""
    if H in MEASURED_ASR_IN:
        return MEASURED_ASR_IN[H]
    return ASR_IN_FLOOR + (ASR_IN_START - ASR_IN_FLOOR) * np.exp(-ASR_IN_DECAY * H)


def asr_out(T):
    """Answer-token recovery from the hidden state the server sends back (T layers left)."""
    if not TAIL_WEIGHTS_PRIVATE:
        return 1.0
    if T in MEASURED_ASR_OUT:
        return MEASURED_ASR_OUT[T]
    return ASR_OUT_FLOOR + (ASR_OUT_START - ASR_OUT_FLOOR) * np.exp(-ASR_OUT_DECAY * T)


def make_tasks(p):
    rng = np.random.default_rng(p["seed"])
    return rng.integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


# ----------------------------------------------------------------------------------
# 2. Leakage per task
# ----------------------------------------------------------------------------------
def leak_local(P, p):
    return dict(prompt=0.0, answer=0.0, total=0.0)


def leak_cloud(P, p):
    A = p["answer_len"]
    return dict(prompt=float(P), answer=float(A), total=float(P + A))


def leak_split(P, H, T, p):
    A = p["answer_len"]
    a_in, a_out = asr_in(H), asr_out(T)
    prompt = a_in * P
    answer = (1 - (1 - a_in) * (1 - a_out)) * A
    return dict(prompt=prompt, answer=answer, total=prompt + answer)


def all_splits():
    return [(H, T) for H in range(1, N_LAYERS) for T in range(0, N_LAYERS - H)]


def total(fn, tasks, *args):
    parts = [fn(P, *args) for P in tasks]
    return {k: sum(d[k] for d in parts) for k in parts[0]}


# ----------------------------------------------------------------------------------
# 3. Run, save, plot
# ----------------------------------------------------------------------------------
def run(p=PARAMS, out="."):
    tasks = make_tasks(p)
    all_tokens = float(sum(tasks) + len(tasks) * p["answer_len"])

    base_local = total(leak_local, tasks, p)
    base_cloud = total(leak_cloud, tasks, p)

    results = []
    for H, T in all_splits():
        r = total(leak_split, tasks, H, T, p)
        results.append(dict(H=H, U=N_LAYERS - H - T, T=T, client_layers=H + T,
                            asr_in=asr_in(H), asr_out=asr_out(T),
                            leak_rate=r["total"] / all_tokens, **r))
    results.sort(key=lambda r: r["total"])

    with open(f"{out}/split_privacy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "H", "U", "T", "client_layers", "asr_in_H", "asr_out_T",
                    "prompt_tokens_leaked", "answer_tokens_leaked", "total_tokens_leaked",
                    "leak_rate_percent"])
        for name, b in [("all-local", base_local), ("all-cloud", base_cloud)]:
            w.writerow([name, "", "", "", "", "", "", round(b["prompt"], 1), round(b["answer"], 1),
                        round(b["total"], 1), round(100 * b["total"] / all_tokens, 2)])
        for r in results:
            w.writerow(["split", r["H"], r["U"], r["T"], r["client_layers"],
                        round(r["asr_in"], 4), round(r["asr_out"], 4),
                        round(r["prompt"], 1), round(r["answer"], 1), round(r["total"], 1),
                        round(100 * r["leak_rate"], 2)])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    C_LOCAL, C_CLOUD = "#2a6fdb", "#e0533d"
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(SHOW_SPLITS)))
    n = np.arange(1, len(tasks) + 1)
    tag = ("PLACEHOLDER leakage curves (not measured)" if USING_PLACEHOLDER else
           "Measured on Llama-3.2-1B (MLP inversion); tail weights public" if not TAIL_WEIGHTS_PRIVATE else
           "Measured on Llama-3.2-1B (MLP inversion)")

    # --- Figure 1: cumulative tokens recovered vs number of tasks (the sketch) ----------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(n, np.cumsum([leak_cloud(P, p)["total"] for P in tasks]), "-s", ms=4,
            color=C_CLOUD, lw=2.2, label="All on server (cloud), ASR = 100%")
    for (H, T), c in zip(SHOW_SPLITS, cols):
        y = np.cumsum([leak_split(P, H, T, p)["total"] for P in tasks])
        ax.plot(n, y, "-", color=c, lw=1.8,
                label=f"Split H={H}, U={N_LAYERS-H-T}, T={T}  ({H+T} layer{'s' if H+T > 1 else ''} on client)")
    ax.plot(n, np.zeros_like(n), "-o", ms=4, color=C_LOCAL, lw=2.2,
            label="All on client (local), ASR = 0%")
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Content tokens recovered by the server")
    ax.set_title(f"Privacy leakage: U-shaped splits between the two baselines\n{tag}", fontsize=11)
    ax.set_xticks(n[::2])
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/step2_privacy_vs_tasks.png", dpi=200)

    # --- Figure 2: heatmap of leak rate over (H, T) --------------------------------------
    grid = np.full((N_LAYERS, N_LAYERS), np.nan)
    for r in results:
        grid[r["T"], r["H"]] = 100 * r["leak_rate"]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    vmin = 0 if USING_PLACEHOLDER else np.floor(np.nanmin(grid)) - 1     # zoom in when all values are high
    im = ax.imshow(grid, origin="lower", cmap="Reds", aspect="auto", vmin=vmin, vmax=100)
    ax.set_xlim(0.5, N_LAYERS - 0.5); ax.set_ylim(-0.5, N_LAYERS - 1.5)
    ax.set_xlabel("H = layers at the start on the client")
    ax.set_ylabel("T = layers at the end on the client")
    ax.set_title(f"Share of the user's tokens recovered [%]\n{tag}", fontsize=11)
    ax.set_xticks(range(1, N_LAYERS)); ax.set_yticks(range(0, N_LAYERS - 1))
    for r in results:
        v = 100 * r["leak_rate"]
        ax.text(r["H"], r["T"], f"{v:.1f}" if not USING_PLACEHOLDER else f"{v:.0f}",
                ha="center", va="center", fontsize=6 if not USING_PLACEHOLDER else 7,
                color="white" if v > vmin + 0.55 * (100 - vmin) else "black")
    fig.colorbar(im, ax=ax, label="% of tokens recovered" + ("" if USING_PLACEHOLDER else f" (scale {vmin:.0f}-100)"))
    fig.tight_layout()
    fig.savefig(f"{out}/step2_privacy_heatmap.png", dpi=200)

    # --- Figure 3: the two leakage curves used -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    Hs = np.arange(1, N_LAYERS)
    Ts = np.arange(0, N_LAYERS - 1)
    ax.plot(Hs, [100 * asr_in(h) for h in Hs], "-o", ms=4, color="#7a3cc9",
            label="ASR_in(H): what the client sends (prompt + answer)")
    ax.plot(Ts, [100 * asr_out(t) for t in Ts], "-s", ms=4, color="#d98a1c",
            label="ASR_out(T): what the server sends back (answer)"
                  + (" = 100%, public tail" if not TAIL_WEIGHTS_PRIVATE else ""))
    ax.set_xlabel("Layers kept on the client (H for ASR_in, T for ASR_out)")
    ax.set_ylabel("Content-token recovery [%]")
    ax.set_ylim(0, 105)
    ax.set_title(f"Leakage per boundary\n{tag}", fontsize=11)
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/step2_asr_curves.png", dpi=200)

    return tasks, all_tokens, base_cloud, results


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    tasks, all_tokens, bc, res = run(out=out)
    print("NOTE: leakage curves are PLACEHOLDERS, not measured attack results." if USING_PLACEHOLDER else
          "Leakage: MEASURED prompt side (Llama-3.2-1B), tail treated as public (ASR_out = 1).")
    print(f"Tail weights private: {TAIL_WEIGHTS_PRIVATE}\n")
    print(f"User tokens in {len(tasks)} tasks: {all_tokens:.0f}")
    print(f"All on client : 0 tokens recovered (0%)")
    print(f"All on server : {bc['total']:.0f} tokens recovered (100%)")
    print("\nMost private 5 splits:")
    for r in res[:5]:
        print(f"  H={r['H']:2d} U={r['U']:2d} T={r['T']:2d}  {r['total']:6.0f} tokens "
              f"({100*r['leak_rate']:.1f}%)  prompt {r['prompt']:.0f}, answer {r['answer']:.0f}")
    print("\nCurves in the plot:")
    for H, T in SHOW_SPLITS:
        r = next(x for x in res if x["H"] == H and x["T"] == T)
        print(f"  H={H:2d} U={r['U']:2d} T={T:2d}  {r['total']:6.0f} tokens ({100*r['leak_rate']:.1f}%)")
