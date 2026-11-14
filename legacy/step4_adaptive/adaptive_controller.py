"""
STEP 4 - The adaptive controller: how the best choice changes with network, device and server load   (version 2)
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script. Same measured leakage/quality tables and latency model as Step 3c (version 2). For each
combination of conditions the controller re-solves
    minimise latency  s.t.  prompt leakage <= eps_prompt,  answer leakage <= eps_answer,
                            answer quality >= q_min,  client memory <= M
choosing among all-local, all-cloud, splits, split + noise, split + secret head, split + secret head and tail.

Three requirements: no privacy; private prompt; private prompt AND answer.
Client memory is 4 GB here (a laptop), so running everything locally is an allowed option.
Conditions swept: uplink rate and round-trip time (network); client capability and the free share of
the server (device and server load).

Notes: leakage and quality are measured on Llama-3.2-1B; latency uses the formula model with these
swept speeds. Leakage assumes an attacker without the secret adapters.
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


# ================================================================================== Step 4
SCENARIOS = [
    dict(key="open", title="No privacy (quality >= 95%)", eps=1.00, eps_a=1.00, q=0.95),
    dict(key="private", title="Private prompt (prompt <= 70%, quality >= 75%)", eps=0.70, eps_a=1.00, q=0.75),
    dict(key="both", title="Private prompt + answer (prompt <= 80%, answer <= 50%, quality >= 75%)",
         eps=0.80, eps_a=0.50, q=0.75),
]
M_CLIENT_STEP4 = 4.0                                    # a laptop with 4 GB free, so all-local is possible
UP = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200]) * 1e6        # uplink; downlink = 2.5 x uplink
RTT = np.array([5, 10, 20, 50, 100, 150, 200, 300, 500]) * 1e-3
FC = np.array([2, 5, 10, 20, 35, 50, 75, 100, 200]) * 1e9           # client capability
SHARE = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])  # share of the server free (1 = idle)


def short(o):
    k = o["kind"]
    if k == "local": return "client"
    if k == "cloud": return "cloud"
    s = f"H{o['H']}" + (f"T{o['T']}" if o["T"] else "")
    if k == "secret": s += f"\nλ{o['lam']:g}"
    if k == "secret_u": s += f"\nλa{o['lam_a']:g}"
    if k == "noise": s += f"\nσ{o['sigma']:g}"
    return s


def decide(sc, p):
    return best(sc["eps"], sc["q"], WL, p, eps_answer=sc["eps_a"])


def run(out="."):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    base = {**PARAMS, "m_client_gb": M_CLIENT_STEP4}
    rows = []

    # ---- Figure 1: decision maps ----
    fig, axes = plt.subplots(3, 2, figsize=(13, 15.5))
    for r, sc in enumerate(SCENARIOS):
        for c, (xs, ys, xk, yk, xl, yl, xf, yf) in enumerate([
                (UP, RTT, "rate_up", "rtt", "Uplink [Mbit/s] (downlink = 2.5x)", "Round-trip time [ms]",
                 lambda v: f"{v/1e6:g}", lambda v: f"{v*1e3:g}"),
                (FC, SHARE, "f_client", "server_share", "Client capability [GFLOP/s]", "Server share available (1 = idle)",
                 lambda v: f"{v/1e9:g}", lambda v: f"{v:g}")]):
            ax = axes[r, c]
            img = np.zeros((len(ys), len(xs), 3))
            for j, yv in enumerate(ys):
                for i, xv in enumerate(xs):
                    p = {**base, xk: xv, yk: yv}
                    if xk == "rate_up":
                        p["rate_down"] = 2.5 * xv
                    b = decide(sc, p)
                    if b is None:
                        img[j, i] = (1, 1, 1); txt = "none"; fam = "none"
                    else:
                        o, L = b
                        img[j, i] = matplotlib.colors.to_rgb(FAMILY_COLOR[o["kind"]])
                        txt = short(o); fam = o["kind"]
                    ax.text(i, j, txt, ha="center", va="center", fontsize=6.4,
                            color="white" if fam in ("local", "cloud", "secret", "secret_u") else "#222")
                    rows.append([sc["key"], ["network", "device"][c], xk, xv, yk, yv,
                                 "INFEASIBLE" if b is None else name(b[0]), "" if b is None else round(b[1], 2)])
            ax.imshow(img, origin="lower", aspect="auto", alpha=0.85)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels([xf(v) for v in xs], fontsize=8)
            ax.set_yticks(range(len(ys))); ax.set_yticklabels([yf(v) for v in ys], fontsize=8)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.spines[:].set_visible(False)
            # mark the default operating point
            dx = int(np.argmin(np.abs(np.log(xs / base[xk])))); dy = int(np.argmin(np.abs(np.log(ys / base[yk]))))
            ax.scatter([dx], [dy], s=420, facecolors="none", edgecolors="black", lw=1.6)
            ax.set_title(f"{sc['title']}\n{'Network conditions' if c == 0 else 'Device and server load'}", fontsize=10.5)
    fams = ["local", "cloud", "split", "noise", "secret", "secret_u"]
    fig.legend([Patch(color=FAMILY_COLOR[f]) for f in fams], [FAMILY_NAME[f] for f in fams], loc="lower center",
               ncol=6, frameon=False, fontsize=9.5)
    fig.suptitle("What the adaptive controller chooses as conditions change (20 tasks, client memory 4 GB; ○ = default conditions)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(f"{out}/step4_decision_maps.png", dpi=170)
    with open(f"{out}/step4_decisions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario", "map", "x_param", "x_value", "y_param", "y_value", "chosen", "total_latency_s"])
        w.writerows(rows)

    # ---- Figure 2: adaptive vs fixed strategies ----
    fixed = {
        "open": [("All on client", next(o for o in OPTIONS if o["kind"] == "local")),
                 ("All on server", next(o for o in OPTIONS if o["kind"] == "cloud")),
                 ("Fixed split H=1, T=0", next(o for o in OPTIONS if o["kind"] == "split" and o["H"] == 1 and o["T"] == 0))],
        "both": [("All on client", next(o for o in OPTIONS if o["kind"] == "local")),
                 ("Fixed secret head+tail λa=0.03", next(o for o in OPTIONS if o["kind"] == "secret_u" and o["lam_a"] == 0.03))],
    }
    sweeps = [("open", "rate_up", np.geomspace(0.2e6, 200e6, 60), "Uplink [Mbit/s]", 1e-6),
              ("open", "server_share", np.geomspace(1.0, 0.005, 60), "Server share available", 1),
              ("both", "rtt", np.geomspace(0.002, 0.5, 60), "Round-trip time [ms]", 1e3),
              ("both", "f_client", np.geomspace(1e9, 300e9, 60), "Client capability [GFLOP/s]", 1e-9)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    gains = []
    for ax, (sk, k, vals, xl, sc_) in zip(axes.flat, sweeps):
        sc = next(s for s in SCENARIOS if s["key"] == sk)
        ad = []
        fx = {lab: [] for lab, _ in fixed[sk]}
        for v in vals:
            p = {**base, k: v}
            if k == "rate_up":
                p["rate_down"] = 2.5 * v
            b = decide(sc, p); ad.append(b[1] / 60 if b else np.nan)
            for lab, o in fixed[sk]:
                ok = (memory_gb(o, WL, p) <= p["m_client_gb"] and o["leak"] <= sc["eps"] + 1e-12
                      and o["answer_leak"] <= sc["eps_a"] + 1e-12 and quality_ok(o, sc["q"]))
                fx[lab].append(latency(o, WL, p) / 60 if ok else np.nan)
        cols = ["#2a6fdb", "#e0533d", "#9aa3ad"] if sk == "open" else ["#2a6fdb", "#6fc2c2"]
        for (lab, _), col in zip(fixed[sk], cols):
            ax.plot(vals * sc_, fx[lab], "--", color=col, lw=1.8, label=lab)
        ax.plot(vals * sc_, ad, "-", color="black", lw=2.6, label="Adaptive controller")
        worst = np.nanmax(np.vstack([fx[l] for l in fx]), axis=0)
        gains.append((sk, k, float(np.nanmax(worst / np.array(ad)))))
        ax.set_xscale("log"); ax.set_yscale("log")
        if k == "server_share":
            ax.invert_xaxis()
        ax.set_xlabel(xl); ax.set_ylabel("Total latency for 20 tasks [min]")
        ax.set_title(f"{sc['title']}", fontsize=9.5)
        ax.grid(alpha=.3, which="both"); ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Adaptive controller vs fixed strategies (fixed strategies shown only where they meet the requirement)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out}/step4_adaptive_vs_fixed.png", dpi=180)
    return gains


WL = workload(PARAMS)

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    gains = run(out)
    base = {**PARAMS, "m_client_gb": M_CLIENT_STEP4}
    print(f"Client memory {M_CLIENT_STEP4} GB. Default conditions: uplink 20 Mbit/s, RTT 50 ms, client 10 GFLOP/s, idle server.\n")
    for sc in SCENARIOS:
        b = decide(sc, base)
        print(f"{sc['title']}: default choice = {name(b[0])}, {b[1]/60:.1f} min")
    print("\nLargest slow-down of the WORST fixed strategy vs the adaptive controller along each sweep:")
    for sk, k, g in gains:
        print(f"  {sk:8s} sweep over {k:12s}: up to {g:.1f}x")
