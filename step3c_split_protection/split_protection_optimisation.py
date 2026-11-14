"""
STEP 3c - Optimisation over split AND protection, for the prompt AND the answer   (version 2)
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script. All leakage and quality values are MEASURED on Llama-3.2-1B-Instruct:
  split only               : inversion_attack.py   (H = 1..15)
  split + noise            : noise_sweep.py        (H = 1,2,4,6,8,10,12 x sigma = 0.25..4)
  split + secret head      : secret_head.py        (H = 4 x lambda = 0.05..0.3)
  split + secret head+tail : secret_ushape.py      (H = 4, T = 2, lambda_prompt = 0.1, lambda_answer = 0..0.3)
Latency comes from the formula model of Steps 1-3 (assumed speeds); adapters add ~0.5% compute per layer.

Decision: one option o for all tasks (521 options incl. all-local and all-cloud).
    minimise   total latency L(o)
    subject to prompt leakage(o) <= eps_prompt        (content words, attacker without the secret)
               answer leakage(o) <= eps_answer        (server runs the public tail on what it sends back)
               answer quality(o) ok                   (QUALITY_METRIC: agreement >= q_min, or perplexity ratio <= q_min)
               client memory(o) <= 2 GB
With a public tail the server can compute every answer, so only the secret head+tail options give
answer leakage below 100%.
"""
# ================================================================================== core model
# (identical in split_protection_optimisation.py and adaptive_controller.py, version 2: answer privacy)
import numpy as np

# split alone (first attack): leakage per H, quality unchanged
SPLIT_ONLY = {
    1: dict(leak=0.9997, private_exact=0.9573),
    2: dict(leak=0.9978, private_exact=0.9606),
    3: dict(leak=0.9906, private_exact=0.9271),
    4: dict(leak=0.9827, private_exact=0.922),
    5: dict(leak=0.978, private_exact=0.912),
    6: dict(leak=0.9729, private_exact=0.8751),
    7: dict(leak=0.9754, private_exact=0.8843),
    8: dict(leak=0.9757, private_exact=0.8935),
    9: dict(leak=0.979, private_exact=0.8843),
    10: dict(leak=0.9743, private_exact=0.886),
    11: dict(leak=0.9747, private_exact=0.8894),
    12: dict(leak=0.974, private_exact=0.9078),
    13: dict(leak=0.9723, private_exact=0.8684),
    14: dict(leak=0.9657, private_exact=0.8374),
    15: dict(leak=0.9643, private_exact=0.7787),
}
# noise sweep: (H, sigma)
NOISE = {
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
# secret adapters (second run, 1200 steps): (H, lambda); leak = attacker WITHOUT the secret, leak_B = WITH
SECRET = {
    (4, 0.05): dict(leak=0.9507, private_exact=0.7821, leak_B=0.9806, private_exact_B=0.8994, agreement=0.9016, ppl_ratio=0.995),
    (4, 0.1): dict(leak=0.8771, private_exact=0.6966, leak_B=0.9497, private_exact_B=0.8424, agreement=0.873, ppl_ratio=1.015),
    (4, 0.2): dict(leak=0.6486, private_exact=0.5608, leak_B=0.8927, private_exact_B=0.7829, agreement=0.7858, ppl_ratio=1.096),
    (4, 0.3): dict(leak=0.3425, private_exact=0.4367, leak_B=0.8673, private_exact_B=0.5482, agreement=0.6591, ppl_ratio=1.384),
}
# secret head + secret tail (secret_ushape.py, two runs, 1200 steps): (H, T, lambda_prompt, lambda_answer)
# leak/answer_leak = attacker without the secret (answer: server runs the public tail); *_B = attacker with the secret
SECRET_U = {
    (4, 2, 0.1, 0.0): dict(leak=0.869, private_exact=0.6421, leak_B=0.9467, private_exact_B=0.803, answer_leak=0.9281, agreement=0.8704, ppl_ratio=1.012),
    (4, 2, 0.1, 0.01): dict(leak=0.8894, private_exact=0.6781, leak_B=0.9504, private_exact_B=0.8013, answer_leak=0.7293, agreement=0.8356, ppl_ratio=1.045),
    (4, 2, 0.1, 0.03): dict(leak=0.7907, private_exact=0.5809, leak_B=0.9449, private_exact_B=0.8064, answer_leak=0.4104, agreement=0.7677, ppl_ratio=1.103),
    (4, 2, 0.1, 0.1): dict(leak=0.6547, private_exact=0.4811, leak_B=0.9363, private_exact_B=0.7284, answer_leak=0.1223, agreement=0.6111, ppl_ratio=1.472),
    (4, 2, 0.1, 0.3): dict(leak=0.7081, private_exact=0.4526, leak_B=0.9673, private_exact_B=0.8097, answer_leak=0.0079, agreement=0.3966, ppl_ratio=3.502),
    (4, 2, 0.2, 0.3): dict(leak=0.0, private_exact=0.0, leak_B=0.3418, private_exact_B=0.0344, answer_leak=0.0, agreement=0.1668, ppl_ratio=25.297),
}

N_LAYERS, D_MODEL, N_HEADS, N_KV_HEADS, D_FF, VOCAB = 16, 2048, 32, 8, 8192, 128_256
HEAD_DIM = D_MODEL // N_HEADS
P_LAYER = D_MODEL * D_MODEL * 2 + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2 + 3 * D_MODEL * D_FF
P_HEAD = D_MODEL * VOCAB
LORA_RANK = 32
# secret LoRA parameters per layer: r * (in + out) for q, k, v, o, gate, up, down
P_LORA = LORA_RANK * ((D_MODEL + D_MODEL) * 2 + (D_MODEL + N_KV_HEADS * HEAD_DIM) * 2 + (D_MODEL + D_FF) * 3)

PARAMS = dict(n_tasks=20, prompt_min=32, prompt_max=512, answer_len=128,
              f_client=10e9, f_server=1e12, server_share=1.0,
              rate_up=20e6, rate_down=50e6, rtt=0.050,
              bytes_per_token_text=4, bytes_per_value=2, m_client_gb=2.0, seed=7)


def layer_flops(n_new, ctx_start):
    pos = np.arange(ctx_start + 1, ctx_start + n_new + 1)
    return 2 * P_LAYER * n_new + 4 * D_MODEL * pos.sum()


def make_tasks(p):
    return np.random.default_rng(p["seed"]).integers(p["prompt_min"], p["prompt_max"] + 1, size=p["n_tasks"])


def workload(p):
    """Sums over all tasks, so the latency of any option is a closed formula (fast for sweeps)."""
    tasks = make_tasks(p); A = p["answer_len"]
    w = [layer_flops(int(P), 0) + sum(layer_flops(1, int(P) + k) for k in range(A)) for P in tasks]
    return dict(tasks=tasks, W=float(sum(w)),                         # FLOPs of ONE layer, all tasks
                HEAD=float(2 * P_HEAD * (1 + A) * len(tasks)),         # output layer, all tasks
                TOK=float(sum(int(P) + A for P in tasks)),             # tokens through the head layers
                TXT_UP=float(sum(tasks) * 4 * 8), TXT_DOWN=float(len(tasks) * A * 4 * 8),
                ROUND_TRIPS=float(len(tasks) * (1 + A)), P_MAX=int(max(tasks)))


def options():
    """Every scheme the controller can choose. kind: local, cloud, split, noise, secret."""
    out = [dict(kind="local"), dict(kind="cloud")]
    for H in range(1, N_LAYERS):
        for T in range(0, N_LAYERS - H):
            out.append(dict(kind="split", H=H, T=T, **SPLIT_ONLY[H], agreement=1.0))
    for (H, s), v in NOISE.items():
        if s == 0:
            continue
        for T in range(0, N_LAYERS - H):
            out.append(dict(kind="noise", H=H, T=T, sigma=s, **v))
    for (H, lam), v in SECRET.items():
        for T in range(0, N_LAYERS - H):
            out.append(dict(kind="secret", H=H, T=T, lam=lam, **v))
    for (H, T, lp, la), v in SECRET_U.items():                   # secret head AND secret tail (T fixed by training)
        out.append(dict(kind="secret_u", H=H, T=T, lam=lp, lam_a=la, **v))
    for o in out:
        if o["kind"] == "local":
            o.update(leak=0.0, private_exact=0.0, agreement=1.0, answer_leak=0.0)
        if o["kind"] == "cloud":
            o.update(leak=1.0, private_exact=1.0, agreement=1.0, answer_leak=1.0)
        o.setdefault("ppl_ratio", 1.0)
        o.setdefault("answer_leak", 1.0)       # public tail: the server can compute every answer word
    return out


def latency(o, wl, p):
    """Total latency of all tasks [s] for option o."""
    fc, fs = p["f_client"], p["f_server"] * p["server_share"]
    if o["kind"] == "local":
        return (N_LAYERS * wl["W"] + wl["HEAD"]) / fc
    if o["kind"] == "cloud":
        return ((N_LAYERS * wl["W"] + wl["HEAD"]) / fs + wl["TXT_UP"] / p["rate_up"]
                + wl["TXT_DOWN"] / p["rate_down"] + p["n_tasks"] * p["rtt"])
    H, T = o["H"], o["T"]; U = N_LAYERS - H - T
    client = (H + T) * wl["W"] + wl["HEAD"]
    if o["kind"] == "secret":
        client += H * 2 * P_LORA * wl["TOK"]                     # adapters: ~0.5% extra work per layer
    if o["kind"] == "secret_u":
        client += (H + T) * 2 * P_LORA * wl["TOK"]               # adapters in head and tail
    hidden_bits = wl["TOK"] * D_MODEL * p["bytes_per_value"] * 8
    return (client / fc + U * wl["W"] / fs + hidden_bits * (1 / p["rate_up"] + 1 / p["rate_down"])
            + wl["ROUND_TRIPS"] * p["rtt"])


def memory_gb(o, wl, p):
    if o["kind"] == "cloud":
        return 0.0
    layers = N_LAYERS if o["kind"] == "local" else o["H"] + o["T"]
    kv = 2 * N_KV_HEADS * HEAD_DIM * p["bytes_per_value"] * (wl["P_MAX"] + p["answer_len"])
    extra = (o["H"] * P_LORA * 4 if o["kind"] == "secret" else
             (o["H"] + o["T"]) * P_LORA * 4 if o["kind"] == "secret_u" else 0)   # fp32 adapters
    return ((layers * P_LAYER + P_HEAD) * p["bytes_per_value"] + layers * kv + extra) / 1e9


QUALITY_METRIC = "agreement"   # "agreement": next word same as public model >= q_min
                               # "ppl":       perplexity ratio <= q_min (e.g. 1.10)


def quality_ok(o, q_min):
    if QUALITY_METRIC == "ppl":
        return o["ppl_ratio"] <= q_min + 1e-9
    return o["agreement"] >= q_min - 1e-12


def best(eps, q_min, wl, p, opts=None, eps_answer=1.0):
    """Fastest option with prompt leakage <= eps, answer leakage <= eps_answer, quality ok, memory <= limit."""
    b = None
    for o in (opts or OPTIONS):
        if (o["leak"] > eps + 1e-12 or o["answer_leak"] > eps_answer + 1e-12 or not quality_ok(o, q_min)
                or memory_gb(o, wl, p) > p["m_client_gb"]):
            continue
        L = latency(o, wl, p)
        if b is None or L < b[1]:
            b = (o, L)
    return b


def name(o):
    k = o["kind"]
    if k == "local": return "All on client"
    if k == "cloud": return "All on server"
    s = f"H={o['H']}, T={o['T']}"
    if k == "noise": s += f", noise σ={o['sigma']:g}"
    if k == "secret": s += f", secret head λ={o['lam']:g}"
    if k == "secret_u": s += f", secret head+tail λp={o['lam']:g} λa={o['lam_a']:g}"
    return s


OPTIONS = options()
FAMILY_COLOR = dict(local="#2a6fdb", cloud="#e0533d", split="#9aa3ad", noise="#d98a1c", secret="#1a9e55",
                    secret_u="#0e6e6e")
FAMILY_NAME = dict(local="All on client", cloud="All on server", split="Split only", noise="Split + noise",
                   secret="Split + secret head", secret_u="Split + secret head and tail")
# ================================================================================== end core


# ================================================================================== Step 3c (v2)
EPS_MAIN, EPS_ANSWER_MAIN, Q_MAIN = 0.80, 0.50, 0.75   # prompt <= 80%, answer <= 50%, quality >= 75%
Q_FLOOR = 0.50                                         # requirements map: models below 50% agreement count as broken


def frontier(fits, key, fam_filter, qs):
    ys = []
    for q in qs:
        c = [o[key] for o in fits if fam_filter(o) and quality_ok(o, q)]
        ys.append(100 * min(c) if c else np.nan)
    return ys


def run(out="."):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = PARAMS; wl = workload(p); tasks = wl["tasks"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fits = [o for o in OPTIONS if memory_gb(o, wl, p) <= p["m_client_gb"]]
    qs = np.round(np.arange(0.35, 1.0001, 0.005), 3)

    # ---- Figure 1: best prompt and answer privacy at each quality level ----
    fams = [("Split only", lambda o: o["kind"] == "split", FAMILY_COLOR["split"], 4),
            ("Split + noise", lambda o: o["kind"] in ("split", "noise"), FAMILY_COLOR["noise"], 2),
            ("Split + secret head", lambda o: o["kind"] == "secret", FAMILY_COLOR["secret"], 2.2),
            ("Split + secret head and tail", lambda o: o["kind"] == "secret_u", FAMILY_COLOR["secret_u"], 2.6)]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for lab, f, col, lw in fams:
        ax[0].step(100 * qs, frontier(fits, "leak", f, qs), where="post", color=col, lw=lw, label=lab)
        ax[1].step(100 * qs, frontier(fits, "answer_leak", f, qs), where="post", color=col, lw=lw, label=lab)
    for a, t in zip(ax, ["PROMPT: lowest leakage reachable", "ANSWER: lowest leakage reachable"]):
        a.axvspan(85, 100.5, color="#e5f4e4", alpha=.6, lw=0)
        a.text(92.5, 4, "usable\nanswers", ha="center", fontsize=9, color="#1a7a3e")
        a.set_xlim(35, 100.5); a.set_ylim(0, 103); a.grid(alpha=.3)
        a.set_xlabel("Minimum answer quality: next token same as public model [%]")
        a.set_title(t, fontsize=11, loc="left")
    ax[0].set_ylabel("Recovered by the server (attacker without the secret) [%]")
    ax[1].legend(frameon=False, fontsize=9, loc="center left")
    ax[1].text(36, 96, "public tail: server runs it and\nreads 100% of answers", fontsize=8.5, color="#5b636b", va="top")
    fig.suptitle("Best privacy reachable at each quality level, prompt and answer (memory <= 2 GB; measured on Llama-3.2-1B)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{out}/step3c_privacy_vs_quality.png", dpi=200)

    # ---- Figure 2: which quality can be kept for each (prompt, answer) privacy requirement ----
    ep = np.round(np.arange(0.40, 1.0001, 0.05), 2); ea = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    grid = np.full((len(ea), len(ep)), np.nan); lab = [["" for _ in ep] for _ in ea]
    short = dict(local="L", cloud="C", split="S", noise="N", secret="h", secret_u="U")
    for j, a_ in enumerate(ea):
        for i, e_ in enumerate(ep):
            c = [o for o in fits if o["leak"] <= e_ + 1e-12 and o["answer_leak"] <= a_ + 1e-12
                 and o["agreement"] >= Q_FLOOR]
            if c:
                o = max(c, key=lambda o: (o["agreement"], -latency(o, wl, p)))
                grid[j, i] = 100 * o["agreement"]
                lab[j][i] = short[o["kind"]] + (f"{o['lam_a']:g}" if o["kind"] == "secret_u" else
                                                 f"{o['lam']:g}" if o["kind"] == "secret" else "")
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    im = ax.pcolormesh(np.arange(len(ep) + 1) - 0.5, np.arange(len(ea) + 1) - 0.5, grid, cmap="Greens", vmin=0, vmax=100)
    for j in range(len(ea)):
        for i in range(len(ep)):
            if np.isnan(grid[j, i]):
                ax.text(i, j, "×", ha="center", va="center", color="#b0b5ba", fontsize=9)
            else:
                ax.text(i, j, f"{grid[j,i]:.0f}\n{lab[j][i]}", ha="center", va="center", fontsize=6.5,
                        color="white" if grid[j, i] > 60 else "#222")
    ax.set_xticks(range(len(ep))); ax.set_xticklabels([f"{100*v:.0f}" for v in ep])
    ax.set_yticks(range(len(ea))); ax.set_yticklabels([f"{100*v:.0f}" for v in ea])
    ax.set_xlabel("Prompt privacy limit ε_prompt [%]"); ax.set_ylabel("Answer privacy limit ε_answer [%]")
    ax.set_title("Best answer quality (agreement %) that meets both privacy limits\n"
                 "C cloud · h secret head (λ) · U secret head+tail (λa) · × infeasible (or quality < 50%)", fontsize=10.5)
    fig.colorbar(im, ax=ax, label="Answer quality [%]")
    fig.tight_layout(); fig.savefig(f"{out}/step3c_privacy_requirements_map.png", dpi=200)
    with open(f"{out}/step3c_requirements_map.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["eps_prompt_percent", "eps_answer_percent", "best_agreement_percent", "choice"])
        for j, a_ in enumerate(ea):
            for i, e_ in enumerate(ep):
                w.writerow([round(100 * e_), round(100 * a_), "" if np.isnan(grid[j, i]) else round(grid[j, i], 1), lab[j][i] or "INFEASIBLE"])

    # ---- Figure 3: the sketch with the optimum (prompt and answer) ----
    b = best(EPS_MAIN, Q_MAIN, wl, p, eps_answer=EPS_ANSWER_MAIN)
    if b is not None:
        o, L = b
        one = lambda P: ({**p, "n_tasks": 1, "prompt_min": int(P), "prompt_max": int(P)})
        t_opt = [latency(o, workload(one(P)), one(P)) for P in tasks]
        t_loc = [latency(dict(kind="local"), workload(one(P)), one(P)) for P in tasks]
        t_cl = [latency(dict(kind="cloud"), workload(one(P)), one(P)) for P in tasks]
        A = p["answer_len"]; n = np.arange(1, len(tasks) + 1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        ax[0].plot(n, np.cumsum(t_loc) / 60, "-o", ms=4, color="#2a6fdb", label="All on client")
        ax[0].plot(n, np.cumsum(t_opt) / 60, "-D", ms=4, lw=2.2, color=FAMILY_COLOR[o["kind"]], label="Optimal")
        ax[0].plot(n, np.cumsum(t_cl) / 60, "-s", ms=4, color="#e0533d", label="All on server")
        ax[0].set_ylabel("Total latency [min]"); ax[0].set_title("Latency", fontsize=11)
        for k, (key, tot, ttl) in enumerate([("leak", tasks, "Prompt content words recovered"),
                                             ("answer_leak", np.full(len(tasks), A), "Answer words recovered")], 1):
            ax[k].plot(n, np.cumsum(tot), "-s", ms=4, color="#e0533d", label="All on server")
            ax[k].plot(n, np.cumsum(o[key] * tot), "-D", ms=4, lw=2.2, color=FAMILY_COLOR[o["kind"]], label="Optimal")
            ax[k].plot(n, np.zeros(len(n)), "-o", ms=4, color="#2a6fdb", label="All on client")
            ax[k].set_title(ttl, fontsize=11)
        for a in ax:
            a.set_xlabel("Number of tasks"); a.set_xticks(n[::2]); a.grid(alpha=.3); a.legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle(f"Optimum for prompt <= {100*EPS_MAIN:.0f}%, answer <= {100*EPS_ANSWER_MAIN:.0f}%, quality >= {100*Q_MAIN:.0f}%, "
                     f"memory <= 2 GB:  {name(o)}  ({L/60:.1f} min)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{out}/step3c_optimal_vs_baselines.png", dpi=200)
    return wl, b


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    wl, b = run(out)
    p = PARAMS
    print("Leakage and quality: MEASURED on Llama-3.2-1B (attacker without the secret); latency: formula model.")
    print(f"Quality metric: {QUALITY_METRIC}\n")
    print(f"{'prompt<=':>9} {'answer<=':>9} {'quality':>8}  chosen                                          latency   prompt  answer  private  quality")
    for e, ea, q in [(1.0, 1.0, 0.95), (0.90, 1.0, 0.85), (0.90, 0.75, 0.80), (0.80, 0.50, 0.75), (0.70, 0.15, 0.60), (0.50, 1.0, 0.65), (0.50, 0.50, 0.60)]:
        bb = best(e, q, wl, p, eps_answer=ea)
        if bb is None:
            print(f"{100*e:8.0f}% {100*ea:8.0f}% {100*q:7.0f}%  INFEASIBLE"); continue
        o, L = bb
        print(f"{100*e:8.0f}% {100*ea:8.0f}% {100*q:7.0f}%  {name(o):46s} {L/60:6.1f} min  {100*o['leak']:5.1f}%  "
              f"{100*o['answer_leak']:5.1f}%  {100*o['private_exact']:6.1f}%  {100*o['agreement']:5.1f}%")
