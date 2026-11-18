"""
STEP 5 - Per-request adaptive controller: chooses the split and protection for EACH request
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script. Same latency/memory model and measured tables as Steps 3c and 4, plus the new
secret head+tail measurements at several splits (measured_multisplit.txt from secret_multisplit.py,
placed in this folder; every batch you add is picked up automatically).

For every request the controller solves
    minimise latency  s.t.  prompt leakage <= eps,  answer leakage <= eps_answer,
                            quality >= q,  client memory <= M
using that request's prompt length, its privacy class, the device, and the network/server state
at the moment it arrives.

Privacy classes: Open | Private answer | Private prompt | Private prompt+answer
Devices (ASSUMED speeds until the timing measurements): phone 2 GB 5 GFLOP/s, laptop 4 GB 20 GFLOP/s,
edge box 8 GB 100 GFLOP/s.

Outputs
  step5_decision_table.csv   choice per device x privacy class (one 256-token request)
  step5_decision_maps.png    choice per privacy class over uplink x round-trip time (laptop)
  step5_request_stream.png   200 requests on a mobile network (laptop): what is chosen, cumulative latency
  step5_devices.png          total latency per device and network scenario: adaptive vs fixed policies
  step5_stream.csv, step5_summary.csv

Run:  python per_request_controller.py            (reads ./measured_multisplit.txt if present)
      python per_request_controller.py path/to/measured_multisplit.txt
Leakage values assume an attacker WITHOUT the secret adapters. Latency is modelled, not measured.
"""
# ================================================================================== core model
# (identical in split_protection_optimisation.py and adaptive_controller.py, version 2: answer privacy)
import os
import re
import sys

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
# OLDER SCRIPT VERSION (pre-multisplit secret_ushape.py, results/secret_ushape_run1/run2). Kept for documentation and
# as a fallback only: load_multisplit() REPLACES this table with the current-script runs (task 1). Not comparable
# with the current script: at lambda_a = 0.1 its prompt leakage 65.5% lies outside the new seeds' 74.3-78.5%.
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


SECRET_U_SEEDVALS = {}                   # key -> {field: [value per seed]} (multisplit_results.csv next to the table)

# ---- privacy check robust to training variance (decided 15 Nov 2026) -----------------------------------------------
# LEAK_CHECK = "robust": an option passes a leakage limit only if its ROBUST leakage <= limit:
#   >= 2 training seeds: max over seeds;  otherwise: mean + ROBUST_MARGIN[field] (the largest half range over seeds
#   observed for that metric over all multi-seed entries; prompt and answer separately), capped at 1.0.
#   The margin is also applied to the single-run split / noise / secret-head options (borrowed from the head+tail
#   seeds). Values that hold BY CONSTRUCTION get no margin: all on client (0), all on server (1), public tail answer (1).
# LEAK_CHECK = "mean": the earlier check on the seed mean (kept for comparison).
LEAK_CHECK = "robust"
LEAK_FIELDS = ("leak", "answer_leak")
SEED_FIELDS = LEAK_FIELDS + ("leak_B",)  # leak_B (attacker B) only for the sensitivity check of the feasibility map
ROBUST_MARGIN = {f: 0.0 for f in SEED_FIELDS}

# ---- answer leakage on GENERATED answers (decided 18 Nov 2026) -----------------------------------------------------
# ANSWER_CHECK = "generated": the robust answer leakage (above) + the gap between generated-answer and teacher-forced
#   answer leakage MEASURED in task 5a (results/task5/5a_*/answer_generation.csv), applied like the seed check:
#   an entry that 5a measured gets its own gap (generated - teacher-forced, same adapters); every other head+tail entry
#   gets the largest gap observed over all 5a entries. Capped at 1.0; no gap on values that hold by construction.
# ANSWER_CHECK = "proxy": the teacher-forced proxy alone (earlier behaviour). Only used when LEAK_CHECK = "robust".
ANSWER_CHECK = "generated"
GEN_GAP = {}                             # (H, T, lam_p, lam_a) -> generated - teacher-forced (5a, MEASURED)
GEN_GAP_MAX = 0.0
GEN_GAP_FROM = None                      # key of the entry with the largest gap

# ---- candidate set (decided 18 Nov 2026) ---------------------------------------------------------------------------
# All on client is NOT a candidate for any privacy class (the thesis is protected split inference); it stays in
# options() only as a reference baseline (all_on_client policy). LOCAL_AS_CANDIDATE = True restores the earlier set.
LOCAL_AS_CANDIDATE = False


def candidates(opts):
    """The options the controller may choose from (all on client excluded unless LOCAL_AS_CANDIDATE)."""
    return [o for o in opts if LOCAL_AS_CANDIDATE or o["kind"] != "local"]


def load_generated_gap(pattern):
    """Gap generated - teacher-forced answer leakage per adapter entry from the task-5a csv files (glob pattern)."""
    import csv, glob
    global GEN_GAP_MAX, GEN_GAP_FROM
    GEN_GAP.clear(); GEN_GAP_MAX, GEN_GAP_FROM = 0.0, None
    for path in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(path)):
            k = (int(r["H"]), int(r["T"]), float(r["lam_p"]), float(r["lam_a"]))
            GEN_GAP[k] = float(r["generated_answer_leak"]) - float(r["teacher_forced_answer_leak"])
    if not GEN_GAP:
        print(f"WARNING: no task-5a files {pattern}: answer leakage = teacher-forced proxy")
        return 0
    GEN_GAP_FROM = max(GEN_GAP, key=GEN_GAP.get); GEN_GAP_MAX = GEN_GAP[GEN_GAP_FROM]
    return len(GEN_GAP)


def load_seed_values(path):
    """Per-seed leak/answer_leak/leak_B of the current-script runs (secret_multisplit.py --collect csv); sets
    ROBUST_MARGIN."""
    import csv
    SECRET_U_SEEDVALS.clear()
    if not path or not os.path.exists(path):
        print(f"WARNING: no per-seed file {path}: robust check = mean + 0 for every option")
        return 0
    for r in csv.DictReader(open(path)):
        if r["lam_a"] == "" or r.get("smoke", "False") == "True":
            continue
        k = (int(r["H"]), int(r["T"]), float(r["lam_p"]), float(r["lam_a"]))
        d = SECRET_U_SEEDVALS.setdefault(k, {f: [] for f in SEED_FIELDS})
        for f in SEED_FIELDS:
            d[f].append(float(r[f]))
    for f in SEED_FIELDS:
        ROBUST_MARGIN[f] = max([(max(v[f]) - min(v[f])) / 2 for v in SECRET_U_SEEDVALS.values() if len(v[f]) > 1],
                               default=0.0)
    return len(SECRET_U_SEEDVALS)


def robust_value(o, f):
    """Robust leakage of option o for field f (see LEAK_CHECK; answer leakage also ANSWER_CHECK)."""
    if f == "leak_B" and "leak_B" not in o:          # no secret (local, cloud, split, noise): attacker B = attacker A
        return robust_value(o, "leak")
    if o["kind"] in ("local", "cloud") or (f == "answer_leak" and o.get("answer_by_construction")):
        return o[f]
    v = []
    if o["kind"] == "secret_u":
        v = SECRET_U_SEEDVALS.get((o["H"], o["T"], o["lam"], o["lam_a"]), {}).get(f, [])
    r = max(v) if len(v) >= 2 else o[f] + ROBUST_MARGIN[f]
    if f == "answer_leak" and ANSWER_CHECK == "generated" and o["kind"] == "secret_u":
        r += GEN_GAP.get((o["H"], o["T"], o["lam"], o["lam_a"]), GEN_GAP_MAX)
    return min(1.0, r)


def leak_ok(o, eps, eps_a, mode=None):
    """Prompt leakage <= eps and answer leakage <= eps_a under LEAK_CHECK ("robust" or "mean")."""
    if (mode or LEAK_CHECK) == "mean":
        return o["leak"] <= eps + 1e-12 and o["answer_leak"] <= eps_a + 1e-12
    return o["leak_r"] <= eps + 1e-12 and o["answer_leak_r"] <= eps_a + 1e-12


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
        if "answer_leak" not in o:
            o["answer_leak"] = 1.0             # public tail: the server can compute every answer word
            o["answer_by_construction"] = True
    for o in out:
        o["leak_r"], o["answer_leak_r"] = robust_value(o, "leak"), robust_value(o, "answer_leak")
        o["leak_B_r"] = robust_value(o, "leak_B")      # attacker B, sensitivity check only
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
    for o in candidates(opts or OPTIONS):                       # all on client is not a candidate (18 Nov 2026)
        if (not leak_ok(o, eps, eps_answer) or not quality_ok(o, q_min)
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


# ================================================================================== Step 5
# --------------------------------------------------------------------------- new measurements (multi-split)
SECRET_U_OLD_SCRIPT = dict(SECRET_U)   # the older-script table above, untouched (documentation)
SECRET_U_SEEDS = {}                      # key -> number of training seeds (current script), set by load_multisplit()
SECRET_U_SPREAD = {}                     # key -> {field: half range over seeds}, only where n_seeds > 1
def load_multisplit(path):
    """REPLACE SECRET_U by the table in measured_multisplit.txt (secret_multisplit.py --collect; current script only).
    The older-script entries above are NOT merged (task 1: different script version, not comparable).
    Also fills SECRET_U_SEEDS and SECRET_U_SPREAD. Without the file the older table stays (with a warning)."""
    if not path or not os.path.exists(path):
        print(f"WARNING: no {os.path.basename(path or 'measured_multisplit.txt')} found: using the OLDER-SCRIPT "
              f"H=4, T=2 table (not comparable with the current script)")
        return 0
    ns = {}
    src = open(path).read()
    exec(src.split("\n# spread")[0], {}, ns)
    new = ns["SECRET_U"]
    SECRET_U.clear(); SECRET_U_SEEDS.clear(); SECRET_U_SPREAD.clear()
    for line in src.splitlines():
        m = re.match(r"\s*(\(.*?\)):.*n_seeds=(\d+)", line)
        if m:
            SECRET_U_SEEDS[eval(m.group(1))] = int(m.group(2))
        m = re.match(r"#\s*(\(.*?\)):\s*(.*?)\s*\(half range", line)
        if m:
            SECRET_U_SPREAD[eval(m.group(1))] = {f: float(v) for f, v in re.findall(r"(\w+) \+-([0-9.]+)", m.group(2))}
    for k, v in new.items():
        SECRET_U[k] = dict(v)
    load_seed_values(os.path.join(os.path.dirname(path), "multisplit_results.csv"))
    load_generated_gap(os.path.join(os.path.dirname(os.path.abspath(path)), "..", "results", "task5", "5a_*",
                                    "answer_generation.csv"))
    return len(new)


# --------------------------------------------------------------------------- one request
def request_workload(P, A):
    """Same formulas as workload(), for ONE request with prompt length P and answer length A."""
    w = layer_flops(int(P), 0) + sum(layer_flops(1, int(P) + k) for k in range(A))
    return dict(W=float(w), HEAD=float(2 * P_HEAD * (1 + A)), TOK=float(P + A),
                TXT_UP=float(P * 4 * 8), TXT_DOWN=float(A * 4 * 8), ROUND_TRIPS=float(1 + A), P_MAX=int(P))


# privacy classes a request can carry (limits on what the server may recover)
CLASSES = [
    dict(key="open",    title="Open",                  eps=1.00, eps_a=1.00, q=0.95),
    dict(key="answer",  title="Private answer",        eps=1.00, eps_a=0.40, q=0.75),
    dict(key="prompt",  title="Private prompt",        eps=0.80, eps_a=1.00, q=0.75),
    dict(key="both",    title="Private prompt+answer", eps=0.80, eps_a=0.50, q=0.75),
]
CLASS_MIX = dict(open=0.4, answer=0.2, prompt=0.2, both=0.2)

# device profiles (ASSUMED values, to be replaced by the timing measurements)
DEVICES = [
    dict(key="phone",  title="Phone (2 GB, 5 GFLOP/s)",    f_client=5e9,   m_client_gb=2.0),
    dict(key="laptop", title="Laptop (4 GB, 20 GFLOP/s)",  f_client=20e9,  m_client_gb=4.0),
    dict(key="edge",   title="Edge box (8 GB, 100 GFLOP/s)", f_client=100e9, m_client_gb=8.0),
]


def decide_request(P, cls, p, opts):
    """Fastest option for ONE request under its privacy class and the current conditions p."""
    wl = request_workload(P, p["answer_len"])
    pp = {**p, "n_tasks": 1}
    b = None
    for o in candidates(opts):
        if (not leak_ok(o, cls["eps"], cls["eps_a"])
                or not quality_ok(o, cls["q"]) or memory_gb(o, wl, pp) > p["m_client_gb"]):
            continue
        L = latency(o, wl, pp)
        if b is None or L < b[1]:
            b = (o, L)
    return b


def fixed_latency(o, P, cls, p):
    """Latency of a FIXED option for one request, or None if it breaks the request's limits or memory."""
    wl = request_workload(P, p["answer_len"]); pp = {**p, "n_tasks": 1}
    if (not leak_ok(o, cls["eps"], cls["eps_a"])
            or not quality_ok(o, cls["q"]) or memory_gb(o, wl, pp) > p["m_client_gb"]):
        return None
    return latency(o, wl, pp)


def short2(o):
    k = o["kind"]
    if k == "local": return "All on client"
    if k == "cloud": return "All on server"
    s = f"H={o['H']}, T={o['T']}"
    if k == "secret": s += f", secret head λp={o['lam']:g}"
    if k == "secret_u": s += f", head+tail λa={o['lam_a']:g}"
    if k == "noise": s += f", noise σ={o['sigma']:g}"
    return s


# --------------------------------------------------------------------------- request stream with a changing network
# network phases: (name, uplink [bit/s], round-trip time [s]); downlink = 2.5 x uplink   (ASSUMED values)
PHASES = dict(wifi=("good Wi-Fi", 50e6, 0.020), lte=("4G", 10e6, 0.060), poor=("poor coverage", 1e6, 0.250))
NETWORKS = [
    dict(key="stable", title="Stable network (4G all the time)", phases=["lte"] * 5),
    dict(key="mobile", title="Mobile network (Wi-Fi -> 4G -> poor coverage -> 4G -> Wi-Fi)",
         phases=["wifi", "lte", "poor", "lte", "wifi"]),
]


def make_stream(n, net, seed=11):
    """n requests: prompt length, privacy class, and the network/server state when each arrives.
    The network follows the scenario's phases (equal number of requests each) with +-20% random variation.
    The same seed gives the same prompts and classes in every scenario, so only the network differs."""
    rng = np.random.default_rng(seed)
    P = rng.integers(PARAMS["prompt_min"], PARAMS["prompt_max"] + 1, size=n)
    keys = list(CLASS_MIX); cls = rng.choice(keys, size=n, p=[CLASS_MIX[k] for k in keys])
    ph = [net["phases"][min(i * len(net["phases"]) // n, len(net["phases"]) - 1)] for i in range(n)]
    jit = np.random.default_rng(seed + 1).uniform(0.8, 1.2, size=(2, n))
    up = np.array([PHASES[k][1] for k in ph]) * jit[0]
    rtt = np.array([PHASES[k][2] for k in ph]) * jit[1]
    share = np.clip(0.6 + 0.4 * np.sin(np.arange(n) / n * 4 * np.pi) + rng.normal(0, 0.05, n), 0.02, 1.0)
    return [dict(i=i, P=int(P[i]), cls=str(cls[i]), rate_up=float(up[i]), rate_down=2.5 * float(up[i]),
                 rtt=float(rtt[i]), server_share=float(share[i])) for i in range(n)]


def run(out=".", multisplit_path=None):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    n_new = load_multisplit(multisplit_path)
    opts = options()
    splits_u = sorted({(o["H"], o["T"]) for o in opts if o["kind"] == "secret_u"})
    print(f"Secret head+tail measured at {len(splits_u)} split(s): {splits_u}  ({n_new} new entries loaded)")
    CL = {c["key"]: c for c in CLASSES}
    base = {**PARAMS}

    # ---------------------------------------------------------------- 1) decision table: device x privacy class
    P_typ = 256
    table = []
    for dv in DEVICES:
        p = {**base, **{k: dv[k] for k in ("f_client", "m_client_gb")}}
        for c in CLASSES:
            b = decide_request(P_typ, c, p, opts)
            table.append(dict(device=dv["title"], cls=c["title"], choice=short2(b[0]) if b else "infeasible",
                              latency_s=round(b[1], 2) if b else None,
                              prompt_leak=round(b[0]["leak"], 3) if b else None,
                              answer_leak=round(b[0]["answer_leak"], 3) if b else None,
                              agreement=round(b[0]["agreement"], 3) if b else None))
    with open(os.path.join(out, "step5_decision_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(f"\nDecision for one request (prompt {P_typ} tokens, answer {PARAMS['answer_len']}) at "
          f"{base['rate_up']/1e6:g}/{base['rate_down']/1e6:g} Mbit/s, RTT {base['rtt']*1e3:g} ms:")
    for r in table:
        lat = f"{r['latency_s']:.1f} s" if r["latency_s"] is not None else "-"
        print(f"   {r['device']:<30} {r['cls']:<22} -> {r['choice']:<34} {lat}")

    # ---------------------------------------------------------------- 2) decision maps over the network (laptop)
    dv = DEVICES[1]
    p0 = {**base, **{k: dv[k] for k in ("f_client", "m_client_gb")}}
    UPS = np.geomspace(0.5e6, 200e6, 25); RTTS = np.geomspace(0.005, 0.5, 25)
    labels = {}
    maps = {}
    for c in CLASSES:
        M = np.full((len(RTTS), len(UPS)), -1)
        for i, rt in enumerate(RTTS):
            for j, up in enumerate(UPS):
                b = decide_request(P_typ, c, {**p0, "rate_up": up, "rate_down": 2.5 * up, "rtt": rt}, opts)
                if b:
                    nm = short2(b[0]); labels.setdefault(nm, len(labels)); M[i, j] = labels[nm]
        maps[c["key"]] = M
    names = sorted(labels, key=labels.get)
    palette = ["#2a6fdb", "#e0533d", "#1a9e55", "#0e6e6e", "#d98a1c", "#7a3cc9", "#9aa3ad", "#b5446e",
               "#3aa6b9", "#8c6d1f", "#5b8e2f", "#c2185b"]
    cols = {nm: palette[k % len(palette)] for k, nm in enumerate(names)}
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff"] + [cols[nm] for nm in names])
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for a_, c in zip(axs, CLASSES):
        a_.pcolormesh(UPS / 1e6, RTTS * 1e3, maps[c["key"]] + 1, cmap=cmap, vmin=0, vmax=len(names), shading="nearest")
        lim = ([f"prompt≤{c['eps']:.0%}"] if c["eps"] < 1 else []) + ([f"answer≤{c['eps_a']:.0%}"] if c["eps_a"] < 1 else [])
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_title(c["title"] + "\n" + ", ".join(lim + [f"quality≥{c['q']:.0%}"]), fontsize=9.5)
        a_.set_xlabel("Uplink [Mbit/s]")
    axs[0].set_ylabel("Round-trip time [ms]")
    used = [nm for nm in names if any((maps[k] == labels[nm]).any() for k in maps)]
    fig.legend(handles=[Patch(color=cols[nm], label=nm) for nm in used] + [Patch(color="#ffffff", ec="#999", label="no option meets the limits")],
               loc="lower center", ncol=min(5, len(used) + 1), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Choice for one request ({P_typ}-token prompt) on the laptop profile, by privacy class and network", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95)); fig.savefig(os.path.join(out, "step5_decision_maps.png"), dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- 3) request stream: adaptive vs fixed policies
    summary = []
    rows = []
    local = next(o for o in opts if o["kind"] == "local"); cloud = next(o for o in opts if o["kind"] == "cloud")
    for net in NETWORKS:
        stream = make_stream(200, net)
        for dv in DEVICES:
            pdv = {**base, **{k: dv[k] for k in ("f_client", "m_client_gb")}}
            # fixed per class: chosen once at the nominal network, then kept for every request of that class
            per_class_fixed = {c["key"]: decide_request(P_typ, c, pdv, opts) for c in CLASSES}
            tot = dict(adaptive=0.0, fixed_per_class=0.0, all_local=0.0, all_cloud=0.0)
            bad = dict(adaptive=0, fixed_per_class=0, all_local=0, all_cloud=0)
            for r in stream:
                p = {**pdv, "rate_up": r["rate_up"], "rate_down": r["rate_down"], "rtt": r["rtt"],
                     "server_share": r["server_share"]}
                c = CL[r["cls"]]
                b = decide_request(r["P"], c, p, opts)
                fx = per_class_fixed[r["cls"]]
                lat = dict(adaptive=b[1] if b else None,
                           fixed_per_class=fixed_latency(fx[0], r["P"], c, p) if fx else None,
                           all_local=fixed_latency(local, r["P"], c, p),
                           all_cloud=fixed_latency(cloud, r["P"], c, p))
                for k, v in lat.items():
                    if v is None:
                        bad[k] += 1
                    else:
                        tot[k] += v
                rows.append(dict(network=net["key"], device=dv["key"], **r, choice=short2(b[0]) if b else "infeasible",
                                 **{f"lat_{k}": (round(v, 3) if v is not None else "") for k, v in lat.items()}))
            for k in tot:
                summary.append(dict(network=net["key"], device=dv["title"], policy=k,
                                    total_latency_min=round(tot[k] / 60, 2), requests_not_served=bad[k]))
    with open(os.path.join(out, "step5_stream.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "step5_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"\nStreams of 200 requests (mix {CLASS_MIX}):")
    print("   'not served' = the policy breaks the request's privacy/quality limit or does not fit in memory")
    for s_ in summary:
        print(f"   {s_['network']:<7} {s_['device']:<30} {s_['policy']:<16} total {s_['total_latency_min']:7.2f} min, "
              f"not served {s_['requests_not_served']:3d}")

    # timeline figure for the laptop
    lap = [r for r in rows if r["device"] == "laptop" and r["network"] == "mobile"]
    chosen = sorted({r["choice"] for r in lap})
    ccol = {nm: (cols[nm] if nm in cols else palette[(len(cols) + k) % len(palette)]) for k, nm in enumerate(chosen)}
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6, 1.3]))
    x = np.arange(len(lap))
    ax[0].plot(x, [r["rate_up"] / 1e6 for r in lap], color="#444", lw=1.5, label="uplink [Mbit/s]")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Uplink [Mbit/s]")
    a2 = ax[0].twinx(); a2.plot(x, [r["server_share"] for r in lap], color="#d98a1c", lw=1.2)
    a2.set_ylabel("Free server share", color="#d98a1c"); a2.spines["top"].set_visible(False)
    ax[0].set_title("Conditions when each request arrives", fontsize=10)
    ykeys = [c["key"] for c in CLASSES]
    for r in lap:
        ax[1].scatter(r["i"], ykeys.index(r["cls"]), color=ccol[r["choice"]], s=22)
    ax[1].set_yticks(range(len(CLASSES))); ax[1].set_yticklabels([c["title"] for c in CLASSES])
    ax[1].set_title("Option chosen for each request (colour)", fontsize=10)
    ax[1].legend(handles=[Patch(color=ccol[nm], label=nm) for nm in chosen], frameon=False, fontsize=8.5,
                 loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for k, col, lab in [("adaptive", "#111", "Adaptive (per request)"), ("fixed_per_class", "#0e6e6e", "Fixed per class"),
                        ("all_local", "#2a6fdb", "All on client")]:
        y = np.cumsum([r[f"lat_{k}"] if r[f"lat_{k}"] != "" else np.nan for r in lap]) / 60
        ax[2].plot(x, y, color=col, lw=2.2 if k == "adaptive" else 1.6, ls="-" if k == "adaptive" else "--", label=lab)
    ax[2].set_ylabel("Cumulative latency [min]"); ax[2].set_xlabel("Request number")
    ax[2].legend(frameon=False, fontsize=9); ax[2].set_title("Cumulative latency (only requests the policy can serve)", fontsize=10)
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.suptitle("Per-request adaptive controller: laptop profile, mobile network (assumed device speed, modelled latency)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "step5_request_stream.png"), dpi=200); plt.close(fig)

    # summary bars: one panel per network scenario
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    pols = [("adaptive", "Adaptive (per request)", "#111"), ("fixed_per_class", "Fixed per class", "#0e6e6e"),
            ("all_local", "All on client", "#2a6fdb")]
    wbar = 0.26
    for ax, net in zip(axs, NETWORKS):
        for j, (k, lab, col) in enumerate(pols):
            vals = [next(x for x in summary if x["network"] == net["key"] and x["device"] == dv["title"]
                         and x["policy"] == k) for dv in DEVICES]
            xs = np.arange(len(DEVICES)) + (j - 1) * wbar
            ax.bar(xs, [v["total_latency_min"] for v in vals], wbar, color=col, label=lab)
            for xx, v in zip(xs, vals):
                if v["requests_not_served"]:
                    ax.text(xx, v["total_latency_min"] + 1, f"{v['requests_not_served']}\nnot\nserved", ha="center",
                            va="bottom", fontsize=7.5, color="#b00")
        ax.set_xticks(range(len(DEVICES))); ax.set_xticklabels([d["title"] for d in DEVICES], fontsize=8.5)
        ax.set_title(net["title"], fontsize=10); ax.grid(alpha=.3, axis="y")
    axs[0].set_ylabel("Total latency, 200 requests [min]"); axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Same 200 requests on three device profiles (all on server breaks every private request; see csv)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "step5_devices.png"), dpi=200); plt.close(fig)
    print(f"\nSaved to {out}/: step5_decision_table.csv, step5_decision_maps.png, step5_request_stream.png, "
          f"step5_devices.png, step5_stream.csv, step5_summary.csv")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ms = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "measured_multisplit.txt")
    run(here, ms)
