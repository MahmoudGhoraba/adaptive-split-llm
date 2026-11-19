# Results summary (as of 19 Nov 2026)

All leakage/quality numbers are MEASURED on Llama-3.2-1B-Instruct (Colab T4). All latency numbers are
MODELLED (assumed speeds). Leakage = content leakage (see README.md, Definitions). Raw files are in `results/`.

RULE VERSIONS OF THE CONTROLLER (to avoid mixing them in the paper). CURRENT rule (18 Nov 2026, §9l; use this in the
paper): all on client is not a candidate (reference baseline only); prompt leakage robust to training variance;
answer leakage = robust teacher-forced value + generated-answer gap (task 5a). SUPERSEDED (earlier rules; kept as
the record, NOT for the paper): §6 "earlier table", §9c–§9k (all on client a candidate; teacher-forced proxy without gap;
mean check up to 9g, robust check from 9h), `step5_per_request/task4_summary_v1.csv` (14 Nov), and the MEASURED
real-runtime trace of §9d / §9f (`runtime/check_out/task4_real_trace.csv`, decisions taken under the 14–15 Nov
rule; rerun under the current rule: §9m, Figure 5). Files `*_meancheck*` (step5_per_request): EARLIER RULE (proxy without gap, seed
means), NOT USED IN THE PAPER.

## 1. Baselines and split latency (modelled, 20 tasks, prompts 32–512, answers 128)

| Configuration | Latency | Prompt leak | Answer leak |
|---|---|---|---|
| All on client | 1797 s (30.0 min) | 0 | 0 |
| All on server | 19.0 s | 100% | 100% |
| Fastest split H=1, T=0 | 403 s (6.7 min) | 99.97% | 100% |
| Slowest split H=15, U=1 | 30.7 min | 96.4% | 100% |

Split latency depends only on H + T in the model (same bytes and round trips for every split).

## 2. Prompt leakage without protection (results/attack)

| H | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| Content leakage % | 99.97 | 99.78 | 99.06 | 98.27 | 97.29 | 97.57 | 97.43 | 97.40 | 96.57 | 96.43 |
| Full private details % | 95.7 | 96.1 | 92.7 | 92.2 | 87.5 | 89.4 | 88.6 | 90.8 | 83.7 | 77.9 |

## 3. Gaussian noise (results/noise) — content leakage % / next-token agreement %

| H | σ=0.25 | σ=0.5 | σ=1 | σ=2 | σ=4 |
|---|---|---|---|---|---|
| 1 | 99.9 / 91.6 | 99.96 / 83.2 | 99.8 / 62.4 | 98.4 / 18.3 | 75.3 / 7.5 |
| 4 | 98.3 / 85.1 | 97.5 / 66.7 | 93.8 / 24.5 | 78.6 / 4.8 | 26.2 / 1.5 |
| 6 | 97.3 / 86.7 | 96.7 / 71.2 | 91.9 / 30.5 | 71.1 / 5.5 | 18.8 / 1.3 |
| 8 | 97.1 / 86.2 | 96.1 / 70.0 | 91.7 / 29.4 | 71.1 / 6.1 | 17.8 / 1.4 |
| 12 | 96.7 / 86.6 | 95.8 / 72.9 | 92.5 / 44.7 | 76.2 / 18.2 | 27.8 / 1.9 |

Negative result: leakage < 90% only when agreement < 19%.

## 4. Secret head only, H = 4 (results/secret_head_*)

Run 1 (600 steps): λp=0 → 98.4% leak at 89.6% agreement (secret init alone hides nothing); λp=0.3 → 41.8% at 66.6%;
λp=1 and 3 collapse the model (agreement 3.2% / 6.0%) — failed quality check.

Run 2 (1200 steps):

| λp | Agreement | PPL ratio | Leak A | Full details A | Leak B | Full details B |
|---|---|---|---|---|---|---|
| 0.05 | 90.2% | 0.995 | 95.1% | 78.2% | 98.1% | 89.9% |
| 0.1 | 87.3% | 1.015 | 87.7% | 69.7% | 95.0% | 84.2% |
| 0.2 | 78.6% | 1.10 | 64.9% | 56.1% | 89.3% | 78.3% |
| 0.3 | 65.9% | 1.38 | 34.3% | 43.7% | 86.7% | 54.8% |

## 5. Secret head + tail, λp = 0.1

### 5a. Older single-split runs at H=4, T=2 (results/secret_ushape_run1, run2; older script version)

OLDER SCRIPT VERSION (pre-multisplit secret_ushape.py). Kept for documentation only; since task 1 (14 Nov 2026) these
values are NOT used by the controller (`load_multisplit()` replaces them; table kept in per_request_controller.py as
`SECRET_U_OLD_SCRIPT`). Not comparable with the current script: at λa = 0.1 the old prompt leakage 65.5% lies outside
the three current-script seeds at (4,2) (74.3, 75.7, 78.5%). The λa = 0.3 and λp = 0.2 entries exist only here and
are therefore no longer controller options (both failed the quality limit of every class, so no decision changes).

| λa | Agreement | Answer leak | Prompt leak A | Prompt leak B |
|---|---|---|---|---|
| 0 | 87.0% | 92.8% | 86.9% | 94.7% |
| 0.01 | 83.6% | 72.9% | 88.9% | 95.0% |
| 0.03 | 76.8% | 41.0% | 79.1% | 94.5% |
| 0.1 | 61.1% | 12.2% | 65.5% | 93.6% |
| 0.3 | 39.7% | 0.8% | 70.8% | 96.7% (failed quality check) |

λp=0.2, λa=0.3 collapsed the model (agreement 16.7%).

### 5b. Multi-split runs (results/multisplit; current script only) — prompt leak A / answer leak / agreement (%)

MEASURED; mean over training seeds; λp = 0.1. This is the complete SECRET_U table of the controller (28 entries,
7 splits; task 1, 14 Nov 2026; (1,1) seed 1 added 18 Nov 2026).

| Split | λa=0 | λa=0.01 | λa=0.03 | λa=0.1 | seeds |
|---|---|---|---|---|---|
| H=1,T=1 | 99.97 / 96.1 / 93.3 | 99.97 / 96.2 / 93.1 | 99.88 / 70.3 / 80.3 | 99.86 / 38.3 / 74.2 | 2 (seeds 0, 1) |
| H=2,T=1 | 98.7 / 95.2 / 91.6 | 98.9 / 84.5 / 88.7 | 96.7 / 55.4 / 80.6 | 72.0 / 19.8 / 61.6 | 1 (seed 0) |
| H=2,T=2 | 98.4 / 94.5 / 91.5 | 98.4 / 72.4 / 86.6 | 96.2 / 39.4 / 79.1 | 81.0 / 15.2 / 63.3 | 2 (seeds 0, 1) |
| H=4,T=1 | 88.1 / 95.0 / 86.8 | 86.2 / 84.6 / 84.5 | 81.0 / 55.8 / 78.1 | 63.6 / 27.4 / 61.7 | 1 (seed 0) |
| H=4,T=2 | 89.3 / 93.2 / 86.7 | 82.3 / 73.6 / 83.1 | 78.9 / 40.8 / 76.7 | 76.2 / 13.3 / 61.3 | 3 (seeds 0, 1, 2) |
| H=6,T=2 | 68.2 / 92.0 / 81.4 | 70.9 / 74.3 / 79.4 | 55.7 / 39.4 / 72.4 | 64.4 / 17.8 / 61.4 | 1 (seed 0) |
| H=8,T=2 | 67.2 / 91.7 / 80.2 | 57.6 / 74.2 / 77.3 | 55.5 / 38.5 / 72.8 | 58.9 / 17.0 / 61.4 | 1 (seed 0) |

Seed spread (half of max − min over seeds; prompt leak A / answer leak / agreement, percentage points):

| Split | λa=0 | λa=0.01 | λa=0.03 | λa=0.1 |
|---|---|---|---|---|
| H=1,T=1 (2 seeds) | ±0.0 / ±0.2 / ±0.2 | ±0.0 / ±0.0 / ±0.2 | ±0.0 / ±2.1 / ±0.4 | ±0.1 / ±2.8 / ±4.7 |
| H=2,T=2 (2 seeds) | ±0.4 / ±0.0 / ±0.5 | ±0.3 / ±0.4 / ±0.6 | ±0.6 / ±1.8 / ±0.1 | ±6.3 / ±1.5 / ±1.2 |
| H=4,T=2 (3 seeds) | ±2.6 / ±0.5 / ±0.2 | ±1.5 / ±1.0 / ±0.3 | ±4.9 / ±0.1 / ±0.6 | ±2.1 / ±3.0 / ±1.1 |

Prompt-leak spread reaches ±4.9–6.3 points while answer-leak spread is ≤ 3.0 points; agreement spread is ≤ 1.2 points
except (1,1) λa = 0.1 (±4.7: seeds 0/1 78.9 / 69.6%; answer leak 35.5 / 41.2%).
Prompt-leakage differences below ~10 points between splits are therefore NOT reliable (the other four splits have one
seed and no spread estimate). Example: (4,2) λa = 0.03, seeds 0/1/2: 83.4 / 73.6 / 79.9%.
New runs (14 Nov 2026, current script, adapters saved on Drive): (4,2) seed 0, (2,2) seed 1, (8,2) seed 0 — all three
pass every sanity check; (1,1) seed 1 (18 Nov 2026, adapters on Drive) also passes every sanity check. The runs (2,2) seed 0, (4,2) seeds 1–2 and (6,2) seed 0 fail "B ≥ A" on the ANSWER side only
(learned answer inverter, known issue); the prompt side B ≥ A holds in every run.
Attacker B (with secret): 99.6–100% at H ≤ 2; 93–95% at H = 4; 83–90% at H = 6; 79–85% at H = 8.
Unprotected prompt leakage (no adapters): 99.98 / 99.79 / 98.7 / 97.8 / 97.6% at H = 1 / 2 / 4 / 6 / 8.

Findings: prompt hiding needs H ≥ 4 (H = 1 hides nothing); H = 6 and H = 8 reach 55–71% (attacker A) but at most
81% agreement; λa raises answer hiding at every split while prompt leakage does not fall monotonically with λa
(within the seed spread). Answer leakage at equal λa is similar across splits (λa = 0.03: 38.5–40.8% for T = 2,
55–72% for T = 1), so T = 2 hides the answer better than T = 1 at similar quality.

## 6. Controllers (modelled latency)

Step 3c (20 tasks, 2 GB): prompt ≤ 80%, answer ≤ 50%, quality ≥ 75% → H=4, T=2, λa=0.03, 15.5 min (old (4,2) data).

Step 5 per request (256-token prompt, 4G), ASSUMED model (MODELLED), CURRENT rule (18 Nov 2026; regenerated with
`python per_request_controller.py` -> step5_decision_table.csv, step5_summary.csv, step5_*.png, step5_run_log.txt):

| Device (assumed) | Open | Private answer | Private prompt | Prompt + answer |
|---|---|---|---|---|
| Phone 2 GB | server 0.9 s | infeasible | H=4 secret head λp=0.2, 60.2 s | infeasible |
| Laptop 4 GB | server 0.9 s | infeasible | H=4 secret head λp=0.2, 21.0 s | infeasible |
| Edge 8 GB | server 0.9 s | infeasible | H=4 secret head λp=0.2, 10.5 s | infeasible |

SUPERSEDED earlier table (all on client a candidate, teacher-forced proxy; rechecked after task 1, not for the paper):
Phone 2 GB: server 0.9 s | H=1,T=1 λa=0.1, 40.9 s | H=4 secret head λp=0.2, 60.2 s | H=4,T=2 λa=0.03, 79.4 s;
Laptop 4 GB: server 0.9 s | H=1,T=1 λa=0.1, 16.2 s | H=4 secret head λp=0.2, 21.0 s | H=4,T=2 λa=0.03, 25.7 s;
Edge 8 GB: server 0.9 s | all on client 8.2 s (private answer, private prompt, prompt + answer).
Step 3c line above: SUPERSEDED (older optimiser, older (4,2) data, all on client a candidate).

Earlier stream result (batch A data only): laptop on a mobile trace 55.8 min (per request) vs 66.9 min
(fixed per class) for 200 requests; equal on a stable network; phone gains nothing. Superseded by §9 (calibrated model) and §9g (rerun after task 1).

## 7. Split runtime: memory per side (MEASURED, 14 Nov 2026; C3 calibration)

MacBook Pro, Apple M2, 8 GB, macOS 26.6.2, torch 2.15.0 MPS, fp16 weights, each side alone in a fresh process,
prompt P = 512 + A = 128 decode steps (runtime/check_partial.py, runtime/check_out/memory.csv).
Footprint = process physical footprint (includes MPS buffers). Modelled = `memory_gb()` formulas (fp16).

| Side | (H, T) | Footprint after load | Footprint after P+A | Tensor bytes (MPS) | Modelled |
|---|---|---|---|---|---|
| client | (1, 1) | 1.93 GB | 1.94 GB | 0.774 GB | 0.771 GB |
| server | (1, 1), U=14 | 2.51 GB | 2.56 GB | 1.721 GB | 1.721 GB |
| client | (4, 2) | 3.00 GB | 3.03 GB | 1.266 GB | 1.263 GB |
| server | (4, 2), U=10 | 2.51 GB | 2.55 GB | 1.230 GB | 1.229 GB |
| client | (8, 2) | 3.00 GB | 3.04 GB | 1.758 GB | 1.755 GB |
| server | (8, 2), U=6 | 1.40 GB | 1.44 GB | 0.738 GB | 0.738 GB |

Findings: tensor bytes match `memory_gb()` within 0.4%, but the process footprint is 1.5–2.4x the modelled value
and grows in steps (client (4,2) and (8,2) both 3.0 GB although their tensors differ by 0.5 GB). The analytical
memory constraint therefore underestimates real client memory; in task 4 the controller's memory constraint
should use measured footprints. System swap grew 1.13 -> 2.93 GB during the full-model reference and the
one-process split check (Chrome open, ~2.3 GB available at start); it did not grow during the per-side runs.

Correctness of the partial models (same process, no sockets, fp16 MPS, 3 chat prompts, 48 greedy tokens):
greedy tokens and boundary states (layers H, N−T, N) are bitwise identical to the full model at 10 splits without
adapters and 4 splits with (untrained) secret adapters; the adapters change the tokens in all 12 adapter runs.

### 7a. Where the extra footprint comes from (MEASURED, runtime/check_memory.py, check_out/memory_check.csv)

- Empty process (python + torch + transformers imported, MPS initialised, no weights): footprint 0.29 GB.
- Loader variants at (4,2), footprint after loading: client 3.005 / 3.005 / 3.696 GB and server 2.515 / 2.513 /
  3.293 GB for cpu_keep (first version) / cpu + gc + empty_cache / direct-to-MPS. Freeing the CPU temporaries
  changes nothing; direct loading raises the load peak (bf16 and fp16 copies on MPS at once) to 3.69 GB.
- Cause: the MPS driver holds 2.16–2.22 GB for 1.23–1.27 GB of tensors on each side. Probe: every MPS buffer of
  >= 32 MiB is placed in a 1 GiB heap (one 32 MiB tensor -> 1024 MiB driver memory; 18 x 32 MiB -> 1024 MiB;
  60 x 8 MiB -> 480 MiB, no overhead). Llama-3.2-1B's MLP matrices are 32 MiB and the embedding 501 MiB, so a side
  occupies whole 1 GiB heaps -> step-wise footprint. empty_cache() cannot release heaps holding live weights.
- Footprints after P + A are noisy under memory pressure (e.g. client cpu_keep 3.005 -> 2.499 GB while swap
  grew): macOS compresses/swaps pages of the process, so footprint alone understates the cost then.
- Client (4,2) + server (4,2) as two processes at the same time (Chrome open, 1.98 GB available before): available
  memory fell to 1.18 GB and swap grew 2.38 -> 3.30 GB (max while sampling). They do NOT fit together without swap
  growth under these conditions. To be repeated with Chrome closed before the task 3 timing runs.

### 7b. First loopback split run (MEASURED, SINGLE RUN — not task-3 quality)

runtime/run_split.py, 14 Nov: server.py (middle layers, MPS) and client.py (MPS) as two processes, TCP on
127.0.0.1, no network emulation, 3 chat prompts (P = 46–57), 48 greedy tokens, 1 repeat, 8-token warm-up on one
prompt only, Chrome open, system swap grew 2.12 -> 3.00 GB during (4,2). Both sides on the same GPU.
Tokens: 12/12 generations equal to the full model (public and untrained secret adapters at (4,2) and (1,1)).

Decode step, median over 47 steps [ms]:

| Split | Head | Serialise | Network (RTT − server) | Server compute | Tail + output layer | Step |
|---|---|---|---|---|---|---|
| (4,2) public | 6.9–7.1 | 0.21 | 0.16–0.17 | 16.7–17.1 (U=10) | 10.2–10.4 | 34.3–35.2 |
| (4,2) adapters | 8.5–8.9 | 0.21–0.23 | 0.18–0.24 | 17.1–18.3 | 10.9–11.9 | 37.0–39.7 |
| (1,1) public | 2.1 | 0.22 | 0.15–0.16 | 22.4–22.7 (U=14) | 8.3–8.5 | 33.3–33.8 |
| (1,1) adapters | 2.4–2.5 | 0.21–0.22 | 0.15–0.18 | 22.5–23.1 | 8.6–8.8 | 34.0–35.0 |

Prefill (time to first token): 65–80 ms, with outliers of 283 ms and 1298 ms in the head phase (likely first-time
MPS kernel compilation for a new prompt length / the adapter path; not verified). Bytes per decode message:
4,148 each way (4,096 payload + 52 header). Empty PING RTT median 0.06–0.08 ms.

### 7c. Client on CPU: dtype choice (MEASURED, 14 Nov, Chrome closed; runtime/check_cpu_dtype.py, check_out/cpu_dtype.*)

Client (4,2) on CPU (4 threads), 1 warm-up + 5 repeats; mean ± sd over repeats. Server not involved (part 1).

| Client dtype | Footprint | Prefill P=128 [ms/layer] | Prefill P=512 total [ms] | Decode [ms/layer] | Output layer, decode [ms] | Decode step [ms] |
|---|---|---|---|---|---|---|
| fp16 | 1.59 GB | 101.5 | 2696 ± 19 | 2.39 | 8.22 ± 0.08 | 22.5 |
| bf16 | 1.59 GB | 133.8 | 3421 ± 12 | 2.42 | 8.34 ± 0.15 | 22.8 |
| fp32 | 2.89 GB | 27.0 | 444 ± 1 | 5.32 | 20.98 ± 0.13 | 52.9 |

fp32 prefill is 4–6x faster (BLAS path) but decode 2.3x slower and memory 1.8x larger; for a P=512, A=128 request
the client compute is ~5.6 s (fp16), ~6.4 s (bf16), ~7.2 s (fp32). Failed first run reported: bf16 showed a 0.28 GB
footprint because bf16 -> bf16 loading kept the weights as views of the mmapped file (fixed: forced copy).
Joint client (CPU) + server (MPS) at (4,2), P=512, A=128: fp16 swap growth +0.06 GB in run 1, none in runs 2–3;
bf16 none (2 runs). Choice: fp16 (fastest, no swap growth in 2 of 3 runs, and more accurate, see 7d).
Joint decode step 41.5–42.8 ms (MPS client: 34 ms in 7b); TTFT P=512: 2.9 s fp16, 3.7–3.8 s bf16.

### 7d. Correctness, client CPU + server MPS (MEASURED; runtime/test_correctness.py)

Same 10 splits, 3 prompts, 48 greedy tokens as the partial check; reference = full model fp16 on MPS.

| Client | Public (30) | Untrained adapters (12) | Boundary rel. error (layers H / N−T / N) |
|---|---|---|---|
| fp16 | 30/30 identical | 9/12 identical | max 6.6e-3 / 8.2e-3 / 8.6e-3 (within 1e-2 at every split) |
| bf16 | 30/30 identical | 5/12 identical | max 4.4e-2 / 5.2e-2 / 5.2e-2 |

All divergences occur at near-ties: fp16 divergences at steps 0, 22, 8 with reference margins 0.000, 0.000, 0.016
(the reference itself is tied; CPU and MPS break the tie differently). bf16 split margins at divergence are exactly 0
(bf16 logits are quantised in steps of ~0.06 at magnitude 16). System swap grew during both runs (0.92 -> 1.71 GB bf16,
1.54 -> 3.45 GB fp16; 10 server restarts and client loads).

### 7e. Network emulation (application level) and one request per scenario (MEASURED, single request each)

Emulator check (runtime/check_netem.py --mode app; 30 empty PINGs, 5 bulk transfers):

| Scenario | RTT set / measured [ms] | Throughput set / measured [Mbit/s] |
|---|---|---|
| Wi-Fi | 20 / 20.29 (+1.4%) | 50 / 49.68 ± 0.03 (−0.65%) |
| 4G | 60 / 60.55 (+0.8%) | 10 / 9.984 ± 0.004 (−0.16%) |
| Poor | 250 / 251.39 (+0.2%) | 1 / 1.000 (+0.05%) |

Failed first check reported: RTT +6.5 to +13 ms because macOS oversleeps time.sleep by ~50% (cap ~10 ms); fixed
with 1 ms sleep slices + spin.

dummynet validation (kernel-level shaping, macOS dnctl/pf on lo0, run by the student, 4G only; MEASURED):
RTT configured 60 ms, measured 63.9 ms (p10–p90 61.6–64.0, +6.4%); throughput 9.981 Mbit/s (−0.19%).
Application-level on the same scenario: 60.55 ms, 9.984 Mbit/s. Throughput is validated; the application-level
RTT is ~3.4 ms lower than kernel-level shaping. Wi-Fi and poor coverage not validated with dummynet.

(4,2), client CPU fp16, server MPS fp16, P = 128 (warmed up), A = 32 fixed, one request per scenario; no swap
growth during any request (system swap 1.41 -> 1.85 GB between requests):

| Scenario | TTFT | Total (32 steps) | Prefill net measured / expected | Decode step (median) | Decode net / expected | Decode server compute |
|---|---|---|---|---|---|---|
| loopback | 0.71 s | 1.97 s | 1.2 ms / – | 40.2 ms | 0.21 ms / – | 16.7 ms |
| Wi-Fi | 0.89 s | 2.89 s | 189.6 / 187.8 ms | 64.2 ms | 21.6 / 21.3 ms | 18.9 ms |
| 4G | 1.70 s | 5.20 s | 900.9 / 898.9 ms | 112.8 ms | 67.0 / 66.6 ms | 19.9 ms |
| Poor | 9.45 s | 21.16 s | 8641.7 / 8639.4 ms | 377.4 ms | 317.2 / 316.4 ms | 26.7 ms |

Expected net = RTT + (bytes up + down) x 8 / rate. Client prefill (P=128): head 368–433 ms, tail 208–267 ms.
Observation (cause not verified): server decode compute rises from 16.7 ms (loopback) to 26.7 ms (poor) as the GPU
idles longer between messages — server speed depends on the network; to be checked in task 3.

## 8. Task 3: measurement campaign and calibrated latency/memory model (14 Nov 2026)

Setup (MEASURED): MacBook Pro Apple M2 8 GB, macOS 26.6.2, torch 2.15.0, Chrome closed. Client = CPU fp16 (4 threads),
server = MPS fp16, wire fp16, TCP loopback (network added from RTT + bytes/rate, validated in 7e). Timing protocol
of runtime/timing.py: one full-length warm-up per prompt bucket, fixed A = 64 (EOS ignored), 5 repeats.
Scripts: runtime/measure_task3.py, runtime/fit_task3.py. Data: runtime/check_out/task3_steps.jsonl (every step),
task3_gens.csv (280 requests), task3_memory.json, task3_fit.json, task3_log.txt.

Failed / perturbed runs (reported, kept): the first (4,1) run was unstable (client prefill P=512: 3505–4637 ms over
repeats, P=1024: 10099 -> 6756 ms; other splits within ±3%) and contained the only request with swap growth
(P=512, repeat 1); it ran with the least free memory (~1.2 GB). Set aside in task3_*_split41_run1_perturbed.* and
rerun; the rerun is consistent (P=512 TTFT 2525 ± 31 ms vs 5738 ms). After the rerun: 0 of 280 requests with swap
growth; system swap rose 1.27 -> 1.91 GB between requests (process restarts).

### 8a. Measured end-to-end on loopback (mean ± sd over 5 repeats; A = 64)

| Config | TTFT P=32 | P=128 | P=256 | P=512 | P=1024 | Decode step (P=256) |
|---|---|---|---|---|---|---|
| (1,1) | 92 ± 1 ms | 324 ± 13 | 617 ± 18 | 1246 ± 43 | 2857 ± 13 | 36.3 ms |
| (2,2) | 132 ± 6 | 563 ± 6 | 1134 ± 12 | 2477 ± 49 | 5288 ± 52 | 37.7 |
| (4,1) (rerun) | 160 ± 6 | 613 ± 13 | 1192 ± 24 | 2525 ± 31 | 5529 ± 75 | 39.0 |
| (4,2) | 180 ± 2 | 766 ± 11 | 1566 ± 23 | 3346 ± 110 | 7015 ± 28 | 40.0 |
| (4,2) adapters | 189 ± 5 | 779 ± 8 | 1604 ± 25 | 3397 ± 38 | 7091 ± 126 | 42.4 |
| (6,2) | 220 ± 6 | 980 ± 9 | 2012 ± 31 | 4200 ± 35 | 9100 ± 71 | 43.0 |
| (8,2) | 265 ± 10 | 1189 ± 12 | 2469 ± 14 | 5232 ± 26 | 11188 ± 77 | 44.3 |
| (4,0) | 138 ± 9 | 560 ± 4 | 1125 ± 11 | 2334 ± 9 | 4912 ± 31 | 37.7 |
| all on client (CPU) | 364 ± 10 | 1825 ± 5 | 3852 ± 8 | 8219 ± 31 | 17972 ± 150 | 46.3 |
| all on server (MPS) | 49 ± 0 | 117 ± 0 | 205 ± 0 | 388 ± 0 | 769 ± 1 | 30.6 |

Secret adapters at (4,2) (untrained; overhead only): TTFT +1 to +5%, decode +2.4 ms/step (+6%).
Client prefill dominates TTFT (CPU); decode is 36–48 ms/step at every split (client output layer alone 8.4 ms).

### 8b. Fitted latency model per side (FITTED on P = 32, 128, 512; tested on P = 256, 1024)

Model: prefill of L layers t = L·(FLOPs_layer(P)/f_eff + c) + o; decode t = L·(d0 + d1·k) + o, k = context.
Least squares on relative error. Variant with separate linear/attention FLOP rates also fitted; the single-rate
variant had the lower TRAINING error on both sides and is used. Errors = mean over repeats per (config, L, P).

| Side / phase | Parameters (FITTED) | Train MAPE (max) | Test MAPE (max) |
|---|---|---|---|
| Client CPU prefill | f_eff = 131 GFLOP/s, c = −7.7 ms/layer, o = −0.6 ms | 3.3% (11.4%) | 6.7% (15.7%) |
| Client CPU decode | d0 = 2.28 ms/layer, d1 = 0.35 µs/token/layer, o = 0.14 ms | 0.7% (–) | 0.6% (2.3%) |
| Server MPS prefill | f_eff = 2.53 TFLOP/s, c = −0.9 ms/layer, o = 30.9 ms | 5.1% (18.1%) | 4.0% (9.1%) |
| Server MPS decode | d0 = 1.28 ms/layer, d1 = 0.14 µs/token/layer, o = 3.7 ms | 1.0% (–) | 2.1% (6.2%) |
| Output layer (MEASURED) | client CPU 8.39 ± 1.01 ms; server MPS 5.71 ± 0.07 ms | – | – |
| Per-message overhead, loopback (FITTED) | 0.23 ms + 0.58 ns/byte (ser + transport + server handling) | – | – |

Limitations: the negative per-layer constant c is not physical (the prefill cost grows faster than the FLOP count);
client prefill at P = 1024 is under-predicted by 8–16% (extrapolation beyond the fit range).

Effective speed vs the ASSUMED values in latency() (client 10 GFLOP/s, server 1 TFLOP/s; laptop profile 20 GFLOP/s):

| | Prefill (FITTED) | Decode, effective at context 64 / 1088 (FITTED) | Assumed |
|---|---|---|---|
| Client (CPU fp16) | 131 GFLOP/s | 53 / 49 GFLOP/s | 10 GFLOP/s (20 laptop) |
| Server (MPS fp16) | 2.53 TFLOP/s | 95 / 91 GFLOP/s | 1 TFLOP/s |

The assumed client is 5–13x too slow; the assumed server is 2.5x too slow for prefill but ~10x too fast for decode
(decode is memory-bound). One speed per side cannot describe both phases.

### 8c. Server downclock check (MEASURED; (4,2), P = 128, A = 64, 3 repeats, no network emulation)

Idle gap before each decode message: median (p10–p90) per decode step [ms]:

| Gap | Server compute (10 layers) | Client head (4 layers) | Client tail layers (2) | Client output layer |
|---|---|---|---|---|
| 0 ms | 16.18 (16.07–18.33) | 9.52 | 4.78 | 8.34 |
| 20 ms | 18.96 (17.41–19.17) | 9.51 | 5.06 | 8.40 |
| 60 ms | 20.83 (19.19–21.57) | 10.14 | 5.87 | 8.59 |
| 250 ms | 27.00 (25.76–30.22) | 10.29 | 10.39 (6.75–11.85) | 9.83 |

Server compute grows by 67% between no gap and a 250 ms gap, so server speed depends on the network RTT. The client
slows down too (tail layers 2.2x, the first work after the reply arrives). The cause is not identified (GPU/CPU
power states are the likely explanation).

### 8d. Composition: fitted model + network formula vs measured end-to-end (P = 256, A = 64, 3 repeats)

| Split | Scenario | Measured total | Fitted total (error) | Measured TTFT | Fitted TTFT (error) | latency() ASSUMED (error) |
|---|---|---|---|---|---|---|
| (4,2) | Wi-Fi | 5.99 s | 5.78 s (−3.5%) | 1.975 s | 1.905 s (−3.5%) | 29.1 s (+387%) |
| (4,2) | 4G | 10.54 s | 10.02 s (−5.0%) | 3.396 s | 3.287 s (−3.2%) | 33.4 s (+217%) |
| (4,2) | Poor | 42.61 s | 41.04 s (−3.7%) | 18.691 s | 18.577 s (−0.6%) | 64.6 s (+52%) |
| (2,2) | Wi-Fi | 5.41 s | 5.20 s (−3.8%) | 1.474 s | 1.463 s (−0.8%) | 21.3 s (+295%) |
| (2,2) | 4G | 10.18 s | 9.44 s (−7.2%) | 2.930 s | 2.845 s (−2.9%) | 25.6 s (+152%) |
| (2,2) | Poor | 42.08 s | 40.46 s (−3.8%) | 18.252 s | 18.136 s (−0.6%) | 56.9 s (+35%) |

Held-out loopback requests (P = 256 and 1024, all 7 splits): total error −9.7% to +3.5%, TTFT −13.9% to +11.5%.
The fitted model always under-predicts under emulated networks (−3.5% to −7.2%); this agrees in sign and size with
the downclock effect in 8c (the model was fitted with no idle gaps) but this is not verified. latency() with the
assumed speeds over-predicts by +35% to +893%.

### 8e. Memory per side (MEASURED footprint after P = 1024 + A = 64 vs memory_gb())

| Config | Client CPU footprint / modelled | Server MPS footprint / modelled |
|---|---|---|
| (1,1) | 1.17 / 0.77 GB (1.52x) | 2.57 / 1.73 GB (1.49x) |
| (2,2) | 1.42 / 1.02 GB (1.39x) | 2.57 / 1.49 GB (1.73x) |
| (4,1) | 1.55 / 1.14 GB (1.35x) | 2.57 / 1.36 GB (1.89x) |
| (4,2) | 1.79 / 1.27 GB (1.41x) | 2.57 / 1.24 GB (2.08x) |
| (6,2) | 1.92 / 1.52 GB (1.27x) | 1.48 / 0.99 GB (1.50x) |
| (8,2) | 2.17 / 1.76 GB (1.23x) | 1.45 / 0.74 GB (1.96x) |
| (4,0) | 1.42 / 1.02 GB (1.39x) | 2.57 / 1.49 GB (1.74x) |
| all on client (CPU) | 2.92 / 2.51 GB (1.16x) | – |
| all on server (MPS) | – | 3.60 (peak 4.08) / 2.51 GB (1.63x) |

Client (CPU): footprint − modelled = 0.40–0.41 GB in 7 of 8 rows (fixed python/torch overhead; 0.29 GB of it is the
empty-process baseline, 7a) and 0.52 GB at (4,2), the only client that also loaded and removed the adapters. Server (MPS): footprint steps with the 1 GiB heaps (2.57 GB for U ≥ 10,
1.45–1.48 GB for U ≤ 8), independent of the exact layer count.

## 9. Task 4: adaptive optimisation with the calibrated model (14 Nov 2026)

Data merge first: results/multisplit/runs/ collected (secret_multisplit.py --collect); new entries (6,2) at
λa = 0, 0.01, 0.03, 0.1 (n_seeds = 1; MEASURED): prompt leak A 68.2 / 70.9 / 55.7 / 64.4%, answer leak 92.0 / 74.3
/ 39.4 / 17.8%, agreement 81.4 / 79.4 / 72.4 / 61.4%. The (6,2) job FAILS the "B >= A" check on the answer side
(learned answer inverter, λa = 0.01: B 33.1% < A 35.1%; known issue); prompt side B >= A holds. (8,2) pending.
step5 now loads 24 SECRET_U entries at 6 splits. The previous tables were read-only; backed up before replacing.
(Superseded by task 1, §9g: since 14 Nov 2026 step5 loads ONLY the current-script table, 28 entries at 7 splits.)

Code (latency()/memory_gb() of step 5 unchanged, for comparison):
step5_per_request/calibrated_model.py (latency_cal, memory_cal, Estimator), calibration.json (built by
runtime/build_calibration.py), adaptive_eval.py (items 4 and 6), runtime/run_trace_real.py (item 5),
runtime/measure_profiles.py (item 2); reconfigurable runtime (server.py --reconfigurable, GENERATE message).

### 9a. Calibrated model (MODELLED from FITTED / MEASURED parameters)

Separate prefill and decode models per side (8b), output-layer cost (client 8.39 ms, server 5.71 ms), per-message
overhead, network = RTT + bytes/rate per direction, idle-gap slowdown (decode steps; gap = network time of a
message, interpolated in the measured table 8c: server x1.00/1.17/1.29/1.67, client x1.00/1.02/1.09/1.35 at
0/20/60/250 ms), secret-adapter overhead (MEASURED at (4,2): 0.36 ms per layer per decode step, ~18 µs per layer
per prompt token), A round trips per request (round-trip fix; latency() still uses 1 + A). Memory: client =
memory_gb() + 0.40 GB; all on server: 0.29 GB.
Check against the task 3 measurements (A = 64): emulated-network requests (2,2)/(4,2) x Wi-Fi/4G/poor −2.5% to
+0.6% (the gap factor was measured at (4,2), P = 128, so (4,2) is not fully independent); held-out loopback requests
−10.9% to +3.5% (largest errors at P = 1024).

### 9b. Device profiles (FITTED from MEASURED, client (4,2) on CPU fp16, 3 repeats)

| Profile | Prefill f_eff | Decode d0 / d1 | Output layer | Memory budget |
|---|---|---|---|---|
| M2-4T (task 3 fit; re-measured here: 146 GFLOP/s) | 131 GFLOP/s | 2.28 ms/layer, 0.35 µs/token | 8.39 ms | 4 GB (ASSUMED) |
| M2-2T (torch.set_num_threads(2)) | 80.4 GFLOP/s | 2.28 ms/layer, 0.56 µs/token | 8.22 ms | 2 GB (ASSUMED) |
| M2-1T (torch.set_num_threads(1)) | 47.6 GFLOP/s | 2.55 ms/layer, 1.14 µs/token | 10.05 ms | 2 GB (ASSUMED) |
| assumed-phone (sensitivity, MODELLED) | 5 GFLOP/s (M2-4T x 26.2) | 59.8 ms/layer | 219 ms | 2 GB |
| assumed-edge (sensitivity, MODELLED) | 100 GFLOP/s (M2-4T x 1.31) | 2.99 ms/layer | 11.0 ms | 8 GB |

Fit errors: prefill train MAPE 1.2 / 2.3 / 4.1%, test (P = 256) 5.4 / 6.7 / 2.3% for 4 / 2 / 1 threads; decode
train MAPE 0.1%. Fewer threads slow prefill (compute-bound) but hardly decode (memory-bound). Footprint 1.59 GB in
all three. Memory budgets cannot be measured on one 8 GB machine and are assumptions.

### 9c. Policies in the calibrated model (MODELLED; 200 requests, A = 128; 20 noise seeds for estimating policies)

Mean-based privacy check (before 15 Nov 2026; files *_meancheck: earlier rule (proxy without gap), not used in the paper). Robust check: see 9h.

Total latency [min] (gap to the oracle). Estimator: EWMA α = 0.3, ±3% observation noise, prior = nominal 4G.

| Trace | Profile | Adaptive | Adaptive + probe | Oracle | Fixed per class | Best single fixed | All on client | Adaptive, assumed model |
|---|---|---|---|---|---|---|---|---|
| Stable 4G | M2-4T | 25.3 (0.0%) | 25.3 (0.0%) | 25.3 | 25.3 (0.0%) | 33.4 (+32.0%) | 33.4 (+32.0%) | 39.0 (+54.1%) |
| Mobile | M2-4T | 24.8 (+0.5%) | 24.8 (+0.5%) | 24.7 | 25.3 (+2.4%) | 33.4 (+35.1%) | 33.4 (+35.1%) | 37.0 (+49.7%) |
| Server load | M2-4T | 29.0 (+6.6%; 2.4–13.2) | 27.6 (+1.3%; 1.25–1.53) | 27.2 | 30.0 (+10.0%) | 33.4 (+22.5%) | 33.4 (+22.5%) | 51.3 (+88.4%) |
| Mobile | M2-2T | 53.7 (0.0%) | 53.7 (0.0%) | 53.7 | 53.7 (0.0%) | infeasible | infeasible | 53.7 (0.0%) |
| Server load | M2-2T | 52.3 (+0.1%) | 52.3 (+0.1%) | 52.3 | 52.6 (+0.6%) | infeasible | infeasible | 52.6 (+0.6%) |
| Mobile | M2-1T | 56.8 (0.0%) | 56.8 (0.0%) | 56.8 | 56.8 (0.0%) | infeasible | infeasible | 56.8 (0.0%) |
| Server load | M2-1T | 55.6 (0.0%) | 55.6 (0.0%) | 55.6 | 55.7 (+0.3%) | infeasible | infeasible | 55.7 (+0.3%) |
| Mobile | assumed-edge | 29.8 (+1.0%) | 29.8 (+1.1%) | 29.5 | 31.3 (+6.2%) | 43.7 (+48.1%) | 43.7 (+48.1%) | 29.8 (+1.0%) |
| Server load | assumed-edge | 34.9 (+1.6%) | 34.5 (+0.6%) | 34.3 | 36.0 (+4.8%) | 43.7 (+27.2%) | 43.7 (+27.2%) | 36.0 (+4.8%) |

(Full table incl. stable trace for all profiles and assumed-phone: step5_per_request/task4_summary.csv.)
Violations (privacy, quality, memory): 0 for every adaptive, oracle and fixed-per-class run; "infeasible" = all on
client does not fit the 2 GB budget (and the best single fixed is then infeasible: no single option satisfies all
classes). The best single fixed option is always all on client where it fits.
Decision changes within a class (mobile, M2-4T): adaptive 15, oracle 18, fixed 0; (server load) adaptive 7.1,
probe 2.1, oracle 2.0. Total changes are dominated by the class mix (94 on the stable trace, all between classes).

Findings:
- With the measured M2 client, all on client is the fastest option for private requests on most networks;
  adaptation pays when it can switch between all on client, all on server and splits (M2-4T, edge). On the 2 GB
  profiles all on client does not fit, every split pays the same per-token network cost, so the network does not
  change the choice and adaptation gains ≤ 0.6%.
- FAILURE MODE (reported, not hidden): estimating server speed only from requests that use the server gets stuck.
  After the load phase, all requests go to the client, the server is never observed again, and the estimate stays
  at ~0.32 after the server has recovered (seed-dependent: +2.4% to +13.2% vs the oracle). A server probe in the
  per-request PING reply removes it (+1.25% to +1.53%).
- A controller driven by the ASSUMED model loses 50–88% on the M2-4T profile in the calibrated world.

### 9d. Real-runtime check (MEASURED; 30-request mobile trace, A = 64, M2-4T, application-level emulation)

| Policy | Measured total | Predicted total (error) | Per-request MAPE | Reload time (not in model) | Changes | Swap growth |
|---|---|---|---|---|---|---|
| Adaptive (estimated) | 137.7 s | 133.7 s (−2.9%) | 5.4% | 95.7 s | 21 | 0 requests |
| Fixed per class | 149.6 s | 145.1 s (−3.1%) | 4.7% | 63.4 s | 16 | 0 requests |

Measured gain of the adaptive policy: −8.0% (predicted −7.9%). The policies differ on 3 of 30 requests (private
answer on Wi-Fi: (1,1) secret head+tail λa = 0.1 instead of all on client): 6.56 vs 11.00 s, 4.99 vs 6.70 s,
6.55 vs 11.12 s. Rate estimates from the prefill messages: 63.8 / 69.0 / 69.9 Mbit/s vs true r_eff 62.6 / 67.1 /
68.8 (+1.6 to +2.8%).
Prediction error by request type (post-hoc grouping, first request after a reload or with a new prompt length vs
steady): all on server −11.0 / −11.6% on the first request after the server reloaded its weights, −0.1 to +0.4%
otherwise; all on client −1.3 to +4.2%; splits (all first after reload) −7.9%. The cold/warm tag written by the
script (first use of configuration x prompt length) missed reloads and is superseded by this grouping.
Reload cost: one configuration in memory at a time (needed to fit client + server in 8 GB on one machine), so each
switch between all on client / all on server / split reloads weights (1.2–7.1 s per switch). Including reloads the
adaptive policy is slower (233.4 s vs 213.0 s): on this testbed switching cost outweighs the gain. On separate
client and server machines the server would not need to reload; switching cost is not in the model or controller.

### 9e. Calibration impact: assumed vs calibrated decisions (MODELLED, A = 128)

Mean-based privacy check (before 15 Nov 2026; files *_meancheck: earlier rule (proxy without gap), not used in the paper). Robust check: see 9h.

Grid: 5 prompt lengths x 3 networks (Wi-Fi/4G/poor) x server share {1, 0.25} = 30 points per class; traces = the
three 200-request traces (true conditions). Assumed counterpart by memory budget.

| Profile (assumed counterpart) | Open | Private answer | Private prompt | Prompt + answer | Trace requests changed |
|---|---|---|---|---|---|
| M2-4T (laptop 20 GFLOP/s) | 12/30 | 18/30 | 17/30 | 18/30 | 327/600 (54.5%) |
| M2-2T (phone 5 GFLOP/s) | 1/30 | 9/30 | 10/30 | 0/30 | 21/600 (3.5%) |
| M2-1T (phone) | 0/30 | 6/30 | 7/30 | 0/30 | 14/600 (2.3%) |
| assumed-phone (phone) | 0/30 | 0/30 | 0/30 | 0/30 | 0/600 |
| assumed-edge (edge) | 9/30 | 2/30 | 2/30 | 1/30 | 23/600 (3.8%) |

Main changes: M2-4T private classes split/secret options -> all on client (the assumed 20 GFLOP/s client is 6.5x
too slow for prefill); open class all on server -> all on client when the server is slow (share 0.25); on the 2 GB
profiles (1,1) λa = 0.1 -> (2,2) λa = 0.03 and secret head T = 0 -> T = 4 when the server is slow (the calibrated
server decode per layer is slower than assumed, so moving layers to the client pays off sooner).

### 9f. Switching cost, default estimator, memory budgets (task 4 follow-up, 14 Nov 2026)

Reload time (MEASURED). Isolated per side (runtime/measure_reload.py; one process per side, shuffled, 3 repeats):
client 0.15–0.50 s for 2–6 layers, 0.95–1.82 s for 8–10, 4.53 ± 0.99 s for all 16 layers; server 0.19–0.54 s for
4–12 layers, 1.69 s for 14, 5.13 ± 0.75 s for all on server. NOT linear in the layer count (linear fit MAPE 208% client,
78% server): reload time depends on memory pressure (what the other side just freed). In situ (the real-runtime
trace, other side resident): client 1.61 s (split, n = 3) / 3.02 s (all on client, n = 20); server 0.08 s (free only,
n = 19) / 3.01 s (U = 14, n = 3) / 5.24 s (all on server, n = 16). Switching-cost model (FITTED, few samples):
client 1.41 s + 0.099 s/layer; server by target (free / U-proportional / all). Setting a = this testbed (both sides
reload; the client frees its weights for all-on-server requests); setting b = dedicated server with every layer
resident and a client that keeps its weights (MODELLED).

Server-speed probe (MEASURED, FAILED as a load estimator): a fixed (64x2048)·(2048x2048) fp16 kernel x8 timed on the
server takes 2.0 ms back-to-back, but 4.3 / 6.1 / 6.8 / 13.4 ms after idle gaps of 20 / 60 / 250 / 2000 ms with no
load (share estimates 0.49 / 0.35 / 0.31 / 0.20 for an idle server); repeating it over PING round trips or as a
64-iteration burst on the server still gives 0.45–0.62. On Apple MPS the GPU power state dominates short probes.
Default estimator instead: the per-request PING reply carries the server-REPORTED load (in the runtime the emulated
share; a real server would report from its scheduler); its cost (RTT + 0.2 ms) is counted. The stuck-estimate
failure mode (share observed only when the server is used; 9c) is kept as policy adaptive_no_load.
LABEL: with the default estimator the server load is REPORTED by the server (assumed to be a truthful scheduler value,
±3% noise in the simulation), NOT estimated by the client. On the server-load trace the default adaptive policy is
therefore near-oracle by construction; only adaptive_no_load shows what estimation from observed timings achieves.

Modelled evaluation (calibrated model, 200 requests, A = 128, 20 seeds; totals in min, gap to the oracle in the same
accounting; the oracle is not switching-aware and sends no PING). M2-4T:

| Trace | Policy | Compute + network + PING | + reload, setting a | + reload, setting b |
|---|---|---|---|---|
| Stable | adaptive | 25.49 (+0.8%) | 32.09 (+0.6%) | 25.54 (+0.8%) |
| Stable | switching-aware (a) / (b) | 26.82 / 25.49 | 32.01 (+0.4%) / 32.09 | 26.87 / 25.54 (+0.8%) |
| Stable | fixed per class | 25.29 (0.0%) | 31.89 (0.0%) | 25.34 (0.0%) |
| Mobile | adaptive | 25.10 (+1.6%) | 32.68 (−0.1%) | 25.96 (+0.4%) |
| Mobile | switching-aware (a) / (b) | 26.93 / 25.59 | 32.03 (−2.1%) / 32.18 | 26.98 / 25.64 (−0.9%) |
| Mobile | fixed per class | 25.31 (+2.4%) | 31.91 (−2.4%) | 25.36 (−1.9%) |
| Server load | adaptive | 27.79 (+2.0%) | 32.46 (+2.6%) | 27.84 (+2.0%) |
| Server load | adaptive_no_load (failure mode) | 29.25 (+7.4%) | 33.69 (+6.5%) | 29.30 (+7.3%) |
| Server load | switching-aware (a) / (b) | 28.36 / 27.79 | 32.31 (+2.2%) / 32.46 | 28.41 / 27.84 (+2.0%) |
| Server load | fixed per class | 29.96 (+10.0%) | 36.56 (+15.6%) | 30.01 (+10.0%) |
| any | all on client | 33.37 | 33.42 | 33.42 |

Findings: on this testbed (setting a) reloads add 4.7–7.6 min to the policies that alternate between all on client
and all on server; totals then lie between 31.9 and 36.6 min (all on client 33.4). Mobile, setting a: fixed per class
31.91, switching-aware 32.03, adaptive 32.68; server load, setting a: switching-aware 32.31, adaptive 32.46, fixed per
class 36.56. Counting the PING, fixed per class beats adaptive on the stable trace in every accounting and on the
mobile trace with reloads (a and b); adaptive is better on mobile only without reloads (25.10 vs 25.31). Adaptation
pays clearly when the server load changes (fixed per class +10.0% / +15.6% / +10.0%). The switching-aware rule is
myopic (one request ahead) and keeps slower configurations to avoid reloads.

Memory-budget sensitivity (MODELLED, 5 seeds, mobile and server-load traces; full table task4_budget_sensitivity.csv):
3 GB and 4 GB give identical results for every measured profile (all on client, 2.91 GB at P = 512 + A = 128, fits);
at 2 GB all on client does not fit and every policy is within 2.2% of the oracle. M2-1T at 3–4 GB, mobile: adaptive
+1.5%, fixed per class +36.9%, all on client +74.2% vs the oracle (compute accounting).

Real-runtime check, 3 repeats (MEASURED; 30-request mobile trace, A = 64, M2-4T, application-level emulation; the
decisions were identical in all three repeats, the spread is timing only; no request with swap growth):

| Policy | Requests (incl. PING) | Reloads (predicted) | Total (predicted, error) | Changes |
|---|---|---|---|---|
| Adaptive | 140.3 ± 1.2 s (PING 2.6 s) | 97.8 ± 15.9 s (89.5) | 238.1 ± 16.3 s (225.8, −5.2%) | 21 |
| Switching-aware (a) | 169.5 ± 0.4 s (PING 2.6 s) | 38.8 ± 2.7 s (40.9) | 208.4 ± 2.3 s (206.7, −0.8%) | 9 |
| Fixed per class | 148.9 ± 0.9 s | 67.7 ± 10.1 s (69.5) | 216.6 ± 10.9 s (214.5, −1.0%) | 16 |

Including reloads the switching-aware policy is fastest (−3.8% vs fixed per class, −12.5% vs adaptive), but the
difference to fixed per class (208.4 vs 216.6 s, 8.2 s) is within fixed per class's run-to-run spread (± 10.9 s, reload
time varies between repeats), so it is not a demonstrated improvement with 3 repeats; the gap to adaptive (29.7 s) is
larger than both spreads (± 2.3 / ± 16.3 s). Without
reloads it is slowest, because it keeps open requests on the client (2.2–6.9 s instead of 2.2–3.1 s on the server)
to avoid 5.2 s server reloads (8 of 30 requests differ from adaptive). Predicted request time −2.2 to −2.8%.

### 9g. Rerun after task 1 (14 Nov 2026; MODELLED decisions, same calibration and traces)

SECRET_U now comes ONLY from the current multisplit script (5b: 28 entries, 7 splits, incl. (8,2); the older-script
(4,2) entries of 5a are dropped). New table copied to step5_per_request/measured_multisplit.txt; adaptive_eval.py
rerun (policies x 3 traces x 5 profiles, 20 seeds; calibration impact; memory-budget sensitivity 2/3/4 GB).
Result: task4_summary.csv, task4_decisions.csv, task4_calibration_impact.csv and task4_budget_sensitivity.csv are
BYTE-IDENTICAL to the previous run; every number in 9c, 9e and 9f stands. No decision uses (8,2).
Reason: the controller only tests each option against the class limits and then picks the fastest, and every
option's pass/fail is unchanged. Per class, the set of secret head + tail options that pass the privacy and quality
limits is identical, except that (8,2) λa = 0 and 0.01 newly pass for "private prompt". (8,2) needs 2.21 GB (calibrated
memory, P = 256), so it is infeasible under the 2 GB budgets; under 4 GB (M2-4T) and 8 GB (assumed edge) all on client
is faster (M2-4T, 4G, P = 256: 9.7 s vs 18.9 s). The dropped older-script λa = 0.3 / λp = 0.2 entries failed every
class's quality limit. The step-5 assumed-model decision table (§6) is also unchanged. The real-runtime trace (9d)
was not rerun; because the passing option sets are unchanged and (8,2) is never fastest in the modelled traces, its
decisions are not expected to change (not verified on the runtime).
Fragile margins (relevant for the seed-spread rule in 5b): the selected entries pass their limits by less than their
seed spread. (4,2) λa = 0.03 (chosen for "private prompt+answer") has prompt leak 78.9% vs the 80% limit (±4.9; seed 0
alone, 83.4%, would fail); (2,2) λa = 0.03 (chosen for "private answer" on 2 GB profiles when the server is slow) has
answer leak 39.4% vs 40% (±1.8; seed 1 alone, 41.2%, would fail). The decisions therefore rest on seed means, not on
a margin that the seed variation clears.

### 9h. Robust privacy check (15 Nov 2026; MODELLED decisions; now the controller default)

NUMBERS SUPERSEDED by 9k (18 Nov 2026: second (1,1) seed); kept as the record of the 15 Nov run.

Rule (decided after the fragile margins of 9g): an option passes a leakage limit only if its ROBUST leakage ≤ limit.
Robust leakage = max over training seeds if the entry has ≥ 2 seeds ((2,2), (4,2); since 18 Nov also (1,1)); otherwise mean + the largest
half range over seeds observed for that metric over all multi-seed entries: prompt +6.34 points ((2,2) λa = 0.1),
answer +3.03 points ((4,2) λa = 0.1); capped at 100%. The margin is also applied to the single-run split, noise and
secret-head options (borrowed from the head + tail seeds; no seeds exist for them). No margin on values that hold
by construction (all on client 0; all on server and the public-tail answer leakage 1). Quality is still checked on
the seed mean. Per-seed values: step5_per_request/multisplit_results.csv (copy of results/multisplit). The old check
remains available (`LEAK_CHECK = "mean"`, `adaptive_eval.py --leak-check mean`; it reproduces the previous outputs
byte for byte). Comparison: step5_per_request/compare_leak_check.py -> task4_leak_check_comparison.csv,
task4_leak_check_transitions.csv.

Entries that change pass/fail for the class they were chosen for:
(4,2) λa = 0.03 robust prompt leak 83.4% (seed 0) > 80% -> fails private prompt+answer; (2,2) λa = 0.03 robust answer
leak 41.2% (seed 1) > 40% -> fails private answer. (1,1) λa = 0.1 stays feasible for private answer (35.5 + 3.0 =
38.5% ≤ 40%); H = 4 secret head λp = 0.2 stays feasible for private prompt (64.9 + 6.3 = 71.2% ≤ 80%). No secret
head + tail entry passes private prompt+answer (prompt ≤ 80%, answer ≤ 50%, agreement ≥ 75%) any more; only all on
client does.

Decision changes (oracle policy, three 200-request traces, true conditions):

| Profile | Class | Mean check -> robust check | Requests |
|---|---|---|---|
| M2-2T, M2-1T, assumed phone (2 GB) | Private prompt+answer | (4,2) λa = 0.03 -> INFEASIBLE (all on client needs > 2 GB) | 108/108 each |
| M2-2T (2 GB) | Private answer | (2,2) λa = 0.03 -> (1,1) λa = 0.1 (slow-server phase) | 11/126 |
| M2-1T (2 GB) | Private answer | (2,2) λa = 0.03 -> (1,1) λa = 0.1 | 8/126 |
| M2-4T (4 GB) | Private prompt+answer | (4,2) λa = 0.03 -> all on client | 4/108 |
| assumed edge (8 GB) | Private prompt+answer | (4,2) λa = 0.03 -> all on client | 14/108 |

Infeasibility: on every 2 GB profile, private prompt+answer requests (36 of 200 per trace, 18%) cannot be served
under the robust check; all other classes stay feasible on all profiles. With a 3 or 4 GB budget every class is
feasible on every M2 profile (all on client for prompt+answer).
Latency cost of the robust check on the requests served under both checks (oracle): M2-4T +0.32% (mobile), 0.00%
(stable, server load); assumed edge +1.19% (mobile); 2 GB profiles +0.00 to +0.28% (the (1,1) λa = 0.1 replacement
for private answer). Memory sensitivity (oracle, mobile, 3 GB): M2-4T +0.3%, M2-2T +1.3%, M2-1T +3.7% (28.56 -> 28.94
and 35.46 -> 36.78 min). Totals on 2 GB profiles fall (e.g. M2-2T mobile oracle 53.67 -> 39.55 min) only because
36 requests are no longer served; they are not a speed-up.
Policy ranking unchanged: M2-4T mobile, gap to the oracle adaptive +1.49% (mean check +1.60%), fixed per class
+2.11% (+2.43%); the assumed-model policy improves on M2-4T (stable +54.9% -> +36.7%, server load +89.2% -> +65.7%)
because the robust check removes the (4,2) option it misjudged, forcing all on client.
Calibration impact under the robust check (Figure 6): M2-4T 231/600 trace decisions change (38.5%; was 327/600)
— the private prompt+answer class no longer changes (both models now choose all on client or are infeasible);
2 GB profiles 6–10/600 (private-answer changes vanish: both models now choose (1,1) λa = 0.1); assumed edge 22/600.
Real-runtime trace (9d): replaying its logged estimator states (runtime/replay_trace_decisions.py) reproduces all 270
logged decisions with the mean check and gives the same 270 decisions with the robust check (M2-4T, A = 64, prompts
≤ 512), so the trace was NOT rerun; its MEASURED results stand.

### 9i. Feasibility analysis of the privacy limits (15 Nov 2026; MODELLED; robust check; no new runs)

NUMBERS SUPERSEDED by 9k (18 Nov 2026: second (1,1) seed); kept as the record of the 15 Nov run.

step5_per_request/feasibility_analysis.py -> feasibility_grid.csv, feasibility_min_prompt_leak.csv,
feasibility_relax.json; Figure 8. Grid: ε_p ∈ {60, 70, 80, 90, 100}%, ε_a ∈ {20, 30, 40, 50, 100}%,
Q ∈ {60, 65, 70, 75, 80, 85}% (150 combinations) x memory budgets 2 / 3 / 4 / 8 GB x profiles M2-4T / M2-2T / M2-1T
(FITTED from MEASURED; budget overridden). Fastest option and calibrated latency: 256-token prompt, nominal 4G,
A = 128, server share 1. Feasibility depends only on the budget (calibrated memory does not depend on the profile).

Feasible combinations: 2 GB 48/150; 3, 4 and 8 GB 150/150 (all on client, 2.88 GB, fits and leaks nothing). Every
server-using option needs ≤ 2.21 GB ((8,2)), so the server-option map is identical for 3, 4 and 8 GB. At 2 GB a
combination is feasible iff ε_p ≥ the minimum robust prompt leakage of Figure 8a for its (ε_a, Q); no server-using
option reaches agreement ≥ 80% with ε_a ≤ 50%, or agreement ≥ 65% with ε_a ≤ 30%.

Class limits (fastest option, latency in s; M2-4T / M2-2T / M2-1T):

| Class | 2 GB | 3, 4 and 8 GB |
|---|---|---|
| Open (Q 95%) | all on server 4.2 / 4.2 / 4.2 | all on server 4.2 / 4.2 / 4.2 |
| Private answer (ε_a 40%) | (1,1) λa = 0.1: 15.8 / 16.1 / 17.0 | all on client 9.7 / 12.2; M2-1T (1,1) λa = 0.1 17.0 |
| Private prompt (ε_p 80%) | H=4 secret head λp = 0.2: 16.6 / 17.2 / 18.7 | all on client 9.7 / 12.2 / 17.6 |
| Private prompt+answer (80 / 50%) | INFEASIBLE | all on client 9.7 / 12.2 / 17.6 |

Why private prompt+answer is infeasible on 2 GB (SENSITIVITY ANALYSIS — the class definitions are NOT changed):
memory excludes the only option that meets all limits trivially (all on client, 2.88 GB > 2 GB), and among the
server-using options that fit, the leakage and quality limits cannot be met together. Every option with robust
prompt leakage ≤ 80% and robust answer leakage ≤ 50% has agreement ≤ 72.4%; every option with agreement ≥ 75% and
answer ≤ 50% leaks ≥ 83.4% of the prompt. The binding constraint is therefore the joint prompt-leakage / quality
trade-off of the secret head + tail at λp = 0.1, given the 2 GB memory limit; neither quality nor leakage alone
explains it. Smallest single-limit relaxation that makes the class feasible (others unchanged):
- prompt limit ε_p 80% -> 83.4% (+3.4 points): (4,2) λa = 0.03 (robust = seed 0; answer 40.9%, agreement 76.7%);
- quality Q 75% -> 72.4% (−2.6 points): (6,2) λa = 0.03 (1 seed; robust prompt 62.0%, answer 42.4%);
- answer limit ε_a 50% -> 77.3% (+27.3 points): (6,2) λa = 0.01 (robust prompt 77.3%, agreement 79.4%);
- memory 2 GB -> 2.88 GB: all on client.
The two smallest relaxations (+3.4 points prompt, −2.6 points quality) are smaller than the measured seed spread
of prompt leakage (±6.3 points), so whether the class is feasible near these limits is not resolved by the current
seeds; with the earlier mean-based check the class was feasible on 2 GB ((4,2) λa = 0.03, mean 78.9%).

### 9j. Decision diversity: when the controller adapts and why (15 Nov 2026; MODELLED; robust check; no new runs)

NUMBERS SUPERSEDED by 9k (18 Nov 2026: second (1,1) seed); kept as the record of the 15 Nov run.

step5_per_request/decision_diversity.py -> task4_diversity.csv, task4_switch_causes.csv; Figure 9. Input: the
decisions of 9h (three 200-request traces x five profiles; adaptive = noise seed 0, oracle). Cause of a switch:
memory limit if the previous configuration no longer fits for the new prompt; otherwise counterfactual from the
previous request's inputs, one factor at a time (privacy class, prompt length, network = rate + RTT, server load);
the replay reproduces all 6000 logged adaptive/oracle decisions.

| Profile (budget) | Distinct configs | Switches / 199 | Split / local / server / not served | Requests adapted to conditions (stable / mobile / server load) |
|---|---|---|---|---|
| M2-4T (4 GB) | 2–4 | 66–104 | 0–9% / 51–69% / 31–41% / 0% | 0 / 17 / 20 |
| M2-2T (2 GB) | 3–4 | 143–144 | 41% / 0% / 41% / 18% | 0 / 0 / 10 |
| M2-1T (2 GB) | 3–4 | 143–145 | 41% / 0% / 41% / 18% | 0 / 0 / 8 |
| assumed phone (2 GB) | 3 | 143 | 41% / 0% / 41% / 18% | 0 / 0 / 0 |
| assumed edge (8 GB) | 2–4 | 70–108 | 0–13% / 46–68% / 32–41% / 0% | 0 / 26 / 18 |

(adaptive policy; "adapted" = the choice differs from the one for the same class and prompt length at nominal 4G
and an unloaded server.) Causes of the 1826 switches (adaptive, all traces and profiles): privacy class change
97.9%, prompt length 0.4%, several factors 1.5% (mostly class + prompt length), interaction 0.2%; network and server
load alone never cause a switch between consecutive requests, and the memory limit causes none (memory hardly
depends on the prompt length). Oracle: the same within 0.4 points.
Within one class (consecutive requests of the same class) only 43 switches occur over all 15 trace/profile pairs:
prompt length 58%, interaction 14%, network 12%, server load 9%, several 7%.

When and why it adapts: the decision is driven almost entirely by the privacy class, which fixes the configuration
family (open -> all on server; private answer -> (1,1) λa = 0.1 on 2 GB; private prompt -> H = 4 secret head on
2 GB or all on client from 3 GB; prompt+answer -> all on client or not served). The conditions change the choice
for 0–13% of requests: on the 4 GB / 8 GB profiles open requests leave the server during poor coverage (mobile) or
the slow-server phase (server load), and private requests move between all on client and a secret split; on 2 GB,
where all on client does not fit, the only condition-driven change is the private-prompt secret head moving from
T = 0 to T = 4 during the slow-server phase, depending on the prompt length (Figure 9b); in the mobile trace the
2 GB decisions never change with the network (Figure 9a). On the stable trace nothing adapts. The adaptive policy
follows the oracle closely (within-class adaptation 43 vs 42 switches, adapted requests within 1–6 per pair);
the estimator lags the true conditions by a few requests after each phase change (Figure 9). These are MODELLED
results on synthetic traces. In the MEASURED real-runtime trace (9d, M2-4T, 30 requests x 3 repeats) the class also
fixed most choices, except private answer, where adaptive chose (1,1) λa = 0.1 for 9 and all on client for 15 of 24
request-repeats depending on the estimated network (counted from task4_real_trace.csv; not cause-attributed).

### 9k. Rerun after the second (1,1) seed (18 Nov 2026; MODELLED decisions; robust check; same calibration and traces)

New run: (1,1) seed 1 (current script, Colab T4, all sanity checks pass); (1,1) now has 2 seeds, so its robust
leakage is the max over seeds instead of mean + margin. The largest half ranges are unchanged (prompt +6.34 points,
answer +3.03 points), so no other entry's robust value changes. Rerun: adaptive_eval.py (robust and --leak-check
mean), compare_leak_check.py, feasibility_analysis.py, decision_diversity.py, runtime/replay_trace_decisions.py,
Figures 6–9.

What changed: (1,1) λa = 0.1, the only private-answer option on 2 GB, now FAILS private answer twice: robust answer
leakage 41.2% (seed 1) > 40%, and mean agreement 74.2% (seeds 78.9 / 69.6%) < 75%. It fails the quality limit also
under the mean-based check. No other option fits 2 GB and meets ε_a 40% with Q 75%.

| Profile | Class | Before (9h) -> now | Requests (oracle, 3 traces) |
|---|---|---|---|
| M2-2T, M2-1T, assumed phone (2 GB) | Private answer | (1,1) λa = 0.1 -> INFEASIBLE | 126/126 each |
| M2-4T (4 GB) | Private answer | (1,1) λa = 0.1 -> all on client | 8/126 |
| assumed edge (8 GB) | Private answer | (1,1) λa = 0.1 -> all on client | 17/126 |
| M2-4T, adaptive only | Private prompt | H=4 secret head T=0 -> all on client | 1 |

Infeasibility under the robust check, every 2 GB profile: private prompt+answer (36 of 200 requests per trace) AND
private answer (42 of 200), i.e. 78 of 200 requests (39%) are not served; only open (all on server) and private
prompt (H = 4 secret head) remain. With the mean-based check private answer on 2 GB is served by (2,2) λa = 0.03
(mean answer 39.4%, seeds 37.6 / 41.2%). With a 3 GB budget every class is feasible (all on client); oracle, mobile
trace, 3 GB: M2-4T 25.01, M2-2T 29.62, M2-1T 39.11 min (mean check 24.81 / 28.79 / 36.16 min; +0.8 / +2.9 / +8.2%).
Latency cost of the robust check on requests served under both checks (oracle, mobile): M2-4T +0.77%, assumed edge
+2.60% (was +0.32 / +1.19%).
Policies (M2-4T, gap to the oracle, calibrated compute time): adaptive +0.80 / +1.30 / +2.01% (stable / mobile /
server load); fixed per class +0.00 / +1.21 / +9.96%. On the mobile trace fixed per class is now marginally better than
adaptive (0.09 points; was +2.11 vs +1.49%); adaptive keeps its advantage on the server-load trace. Assumed-model
policy +19.0 / +22.2 / +36.4%.
Calibration impact (Figure 6), robust check, trace decisions that change: M2-4T 125/600 (20.8%; was 231/600), now
only open (23) and private prompt (102), because private answer is served by all on client under both models;
M2-2T 10/600, M2-1T 6/600 (private prompt, T = 0 -> T = 4), assumed edge 21/600, assumed phone 0. Mean check: M2-4T
330/600 (55.0%).
Feasibility (Figure 8): 2 GB 45/150 limit combinations feasible (was 48); 3, 4 and 8 GB 150/150. Private answer at
2 GB (sensitivity analysis; class limits NOT changed): smallest single relaxation ε_a 40% -> 40.9% ((4,2) λa = 0.03,
max over 3 seeds; agreement 76.7%, 1.69 GB); or Q 75% -> 63.3% ((2,2) λa = 0.1); or memory -> 2.88 GB (all on client).
The +0.9-point relaxation is far below the answer-leak seed spread (±3.0) — and far below the gap between teacher-forced
and generated-answer leakage (§10a) — so the 40% boundary is not resolved by the data. Private prompt+answer: the
relaxations of 9i are unchanged (feasibility_relax.json identical).
Decision diversity (adaptive; Figure 9): 2 GB profiles now use 2–3 configurations (all on server, H = 4 secret head
T = 0 / T = 4) and 128–129 switches per trace; switch causes 1681: privacy class 98.4%, prompt length 0.2%, several
1.2%, interaction 0.2%; within-class switches 34 (oracle 24): prompt length 62%, server load 12%, network 9%,
several 9%, interaction 9%. Requests adapted to conditions: M2-4T 0 / 8 / 20, assumed edge 0 / 12 / 18, M2-2T
0 / 0 / 10, M2-1T 0 / 0 / 7, assumed phone 0 (stable / mobile / server load). The conclusions of 9j hold: the class
fixes the configuration family; on 2 GB the only condition-driven change is the secret head T = 0 -> T = 4 under a
slow server.
Real-runtime trace (9d, MEASURED, M2-4T, A = 64): replaying its logged estimator states with the new table changes
12 of 270 logged decisions under BOTH checks (private answer, P = 256 and 512: (1,1) λa = 0.1 -> all on client),
because (1,1) λa = 0.1 now fails the quality limit. The MEASURED trace was run with the earlier table; it was NOT
rerun. Its measured timings stay valid as measurements of the decisions it made, but 12 of those 270 decisions are
not what the current controller would choose.

### 9l. Protected split inference only + generated-answer leakage (18 Nov 2026; MODELLED decisions; no new runs)

Decisions (18 Nov 2026): (1) all on client is NOT a candidate for any privacy class; it is kept only as the
reference baseline policy all_on_client (the thesis is protected split inference, not avoiding the server);
(2) answer leakage in the controller = robust teacher-forced value (9h) + the gap generated − teacher-forced
MEASURED in task 5a (§10a), applied like the seed check: the 12 entries 5a measured ((4,2), (2,2), λa 0–0.1; since
19 Nov also (1,1) and (8,2), λa 0.03 / 0.1) get their own gap (+0.05 to +45.3 points); every other head + tail entry
gets the largest observed gap, +45.3 points ((8,2) λa = 0.03; was +40.8 from (2,2) λa = 0.1 before 19 Nov); capped
at 100%; no gap on by-construction values (public tail = 100%). Objective (latency) and constraints
C1–C4 unchanged; prompt leakage still attacker A. Code: per_request_controller.candidates(), ANSWER_CHECK = "generated",
GEN_GAP (read from results/task5/5a_*/answer_generation.csv). Measured tables unchanged.
Resulting smallest answer leakage of any candidate: 20.0% ((8,2) λa = 0.1, own gap +0.05, agreement 61.4%, 2.21 GB;
before 19 Nov: 46.2%, (4,2) λa = 0.1, agreement 61.3%); with agreement ≥ 75%: 68.1% ((2,2) λa = 0.03).
UPDATE 19 Nov 2026 (5a for (1,1) seed 1 and (8,2) seed 0, §10a): every decision, the servable classes, the policy
comparison, the calibration impact and the decision diversity below are UNCHANGED (task4_*.csv and step5_*.csv
byte-identical to commit b804293); only the relaxations at ≥ 3 GB and the grid counts changed (marked "19 Nov").

Servable privacy classes (fastest option at P = 256, nominal 4G, A = 128; latency M2-4T / M2-2T / M2-1T, MODELLED;
feasibility_classes.csv) — identical at 2, 3, 4 and 8 GB:

| Class (limits) | Attacker A (controller) | Attacker B (sensitivity) |
|---|---|---|
| Open (Q 95%) | all on server, 4.2 / 4.2 / 4.2 s | all on server |
| Private prompt (ε_p 80%, Q 75%) | H = 4 secret head λp = 0.2, T = 0: 16.6 / 17.2 / 18.7 s | INFEASIBLE |
| Private answer (ε_a 40%, Q 75%) | INFEASIBLE | INFEASIBLE |
| Private prompt+answer (80 / 50 / 75%) | INFEASIBLE | INFEASIBLE |

Smallest single-limit relaxation per infeasible class (SENSITIVITY ANALYSIS; class limits NOT changed; same at every
budget unless stated; memory never helps: every candidate needs ≤ 2.21 GB):
- private answer: ε_a 40% -> 68.1% ((2,2) λa = 0.03, agreement 79.1%); no relaxation of ε_p alone helps; 19 Nov: at
  ≥ 3 GB also Q 75% -> 61.4% ((8,2) λa = 0.1: prompt 65.2%, answer 20.0%; 2.21 GB), at 2 GB no Q relaxation helps.
- private prompt+answer: Q 75% -> 61.3% at 2 GB ((4,2) λa = 0.1: prompt 78.5% max over seeds, answer 46.2%) and
  (19 Nov) 61.4% at ≥ 3 GB ((8,2) λa = 0.1); or ε_a -> 100% (= the private-prompt option); no relaxation of ε_p alone
  helps.
- attacker B, private prompt: ε_p 80% -> 82.0% ((8,2) λa = 0.01, B 82.0%, agreement 77.3%; needs ≥ 3 GB) or 86.4% at
  2 GB ((6,2) λa = 0, 1 seed); a Q relaxation helps only down to 6.1% (noise σ = 2), i.e. not usefully.
- attacker B, private answer: as for A (ε_a -> 68.1%; 19 Nov: at ≥ 3 GB also Q -> 61.4%, (8,2) λa = 0.1, B 86.3%);
  private prompt+answer: no single relaxation helps.
Grid (Figure 8): attacker A 24/150 limit combinations feasible at 2 GB, 34/150 at 3–8 GB (25/150 before 19 Nov);
attacker B 12/150 at 2 GB, 17/150 at 3–8 GB (12/150 before 19 Nov); all added combinations have Q = 60% ((8,2)
λa = 0.1). (Was 45/150 at 2 GB and 150/150 at ≥ 3 GB with all on client and the proxy, 9k.)

Policy comparison (MODELLED, calibrated model, 3 traces x 5 profiles, adaptive policies 20 noise seeds): every profile
serves 122 of 200 requests per trace (open 82, private prompt 40); private answer (42) and private prompt+answer (36)
are not served on any profile. best_fixed is infeasible on every profile (no single candidate serves both open and
private prompt). Gap to the oracle (compute time, requests served by both):

| Profile | adaptive (stable / mobile / server load) | fixed per class | all on client (reference, 200 served) |
|---|---|---|---|
| M2-4T | +0.72 / +0.76 / +0.64% | +0.00 / +0.00 / +3.81% | +20.1 / +0.4 / −11.4% |
| M2-2T | +0.70 / +0.74 / +0.57% | +0.00 / +0.00 / +0.79% | does not fit (2 GB) |
| M2-1T | +0.65 / +0.70 / +0.56% | +0.00 / +0.00 / +0.37% | does not fit |
| assumed edge | +0.69 / +0.73 / +0.77% | +0.00 / +0.00 / +1.93% | +50.6 / +26.8 / +10.5% |

The adaptive gap on the stable and mobile traces is the PING cost (≈ 0.7%): the decisions equal the oracle's. Fixed
per class equals the oracle except under server load, where adaptive is better on M2-4T, M2-2T and the assumed edge
and fixed per class on M2-1T (+0.37 vs +0.56%). On M2-4T under server load all on client would be 11.4% faster than the
best protected split option on the same requests (reference only, not a candidate).
Decision diversity (adaptive): 2–3 configurations per profile and trace (all on server; H = 4 secret head T = 0, and
under the slow server T = 4 (2 GB) or T = 11 (4 / 8 GB)); switches 128–129 per trace, 99.1% caused by the privacy class;
the network changes no decision on any profile (mobile trace: 0 requests adapted); server load adapts 7–15 requests
(M2-4T 15, M2-2T 10, M2-1T 7, assumed edge 11, assumed phone 0).
Calibration impact (Figure 6): trace decisions that change between the assumed and the calibrated model: M2-4T 14/600
(2.3%; was 125/600), M2-2T 10, M2-1T 6, assumed edge 11, assumed phone 0 — all in private prompt (secret head T = 0 ->
T = 4 or T = 11). The robust and the mean-based check now give the same calibration impact.
Mean-based check (files *_meancheck: earlier rule (proxy without gap), not used in the paper; raw seed means; comparison only): private answer -> (2,2) λa = 0.03 and private
prompt+answer -> (4,2) λa = 0.03 on every profile (126 and 108 requests over 3 traces).
Real-runtime trace under these rules (9d setup: M2-4T, 30-request mobile trace, A = 64, 3 repeats x 3 policies): a
run started 18 Nov 2026 15:07 was ABORTED at 15:18 by decision (in repeat 3, adaptive policy, request 20; 201 of 270
request lines logged). Its partial logs are NOT used as data; they are kept only in
runtime/check_out/aborted_2026-11-18/ (console log and server log; no csv/json was written). Rerun: §9m.

### 9m. Real-runtime trace under the current rule (MEASURED, 19 Nov 2026)

Setup as 9d / 9f (M2-4T, 30-request mobile trace, A = 64, client CPU fp16 + server MPS fp16 on the same Mac, 3 repeats
x 3 policies, application-level emulation); controller = current rule (9l; 5a data of 19 Nov). Started after a reboot
with other apps closed; before the start: no leftover runtime processes, about 3.6 GB available (1.5 GB free +
2.0 GB inactive of 8 GB; memory_pressure 68% free). Run order: repeat-major (adaptive, switching-aware, fixed per
class, then the next repeat). Raw: runtime/check_out/task4_real_trace_protected.csv/.json, _console.log, server log.
Served: 15 of 30 requests per run (12 open -> all on server; 3 private prompt -> H = 4 secret head λp = 0.2, T = 0);
the 15 private answer (8) and private prompt+answer (7) requests are infeasible and not run (no PING, no time).
The decisions were identical in all three repeats of each policy. Adaptive and fixed per class made the SAME 30
decisions (they differ only by the PING); switching-aware served open requests 6 and 9 with the unprotected split
H = 4, T = 0 (8.4 s instead of 2.4 s on the server) to avoid reloads. No request with swap growth; available memory
during the run 0.96–2.11 GB.

| Policy | Requests (incl. PING), predicted (error) | Reloads (predicted) | Total (predicted, error) | Changes | MAPE |
|---|---|---|---|---|---|
| Adaptive | 48.7 ± 0.3 s (PING 1.3 s), 47.6 s (−2.2%) | 17.7 ± 4.5 s (19.3) | 66.4 ± 4.8 s (66.9, +0.7%) | 3 | 2.9 ± 0.3% |
| Switching-aware (a) | 61.6 ± 0.6 s (PING 1.3 s), 59.9 s (−2.8%) | 16.6 ± 1.1 s (19.3) | 78.2 ± 0.5 s (79.1, +1.2%) | 4 | 3.4 ± 1.0% |
| Fixed per class | 47.4 ± 0.3 s, 46.3 s (−2.2%) | 16.0 ± 2.2 s (19.3) | 63.4 ± 2.3 s (65.6, +3.4%) | 3 | 2.9 ± 0.4% |

(mean ± s.d. over 3 repeats; error = (predicted − measured) / measured; MAPE = mean |per-request error|.)
Disturbed repeats (flagged; kept in the table):
- adaptive repeat 0 (the FIRST run after the reboot): reloads 22.8 s vs 15.2 / 14.9 s in repeats 1 / 2, mostly the
  first request (secret head load 6.9 s vs 2.5 / 3.5 s) and request 6 (6.0 vs 4.6 / 3.8 s); its request time is normal
  (49.1 vs 48.5 / 48.6 s). Likely cold file cache after the reboot (NOT tested). Without it adaptive totals
  63.7 ± 0.2 s (2 repeats).
- switching-aware repeat 2, request 16 (open, all on server, no reload): 3.97 s vs 2.92 / 2.93 s in repeats 0 / 1
  (−26.3% error); one-request timing disturbance (+1.1 s on the total); cause unknown.
Comparison with the earlier run (9f, 14–15 Nov rule, 30 served requests; totals are NOT comparable because the
decisions and the number of served requests differ):
- Spread across repeats: request time coefficient of variation 0.6–1.0% (earlier 0.2–0.8%); total 0.6–7.2%
  (earlier 1.1–6.9%). As before the spread is almost entirely reload time (s.d. 1.1–4.5 s vs 0.3–0.6 s for requests).
- Measured vs predicted: request time −2.2 to −2.8% (earlier −2.2 to −2.8%: the calibrated model under-predicts by
  the same amount); per-request MAPE 2.9–3.4% (earlier 2.8–5.3%; fewer split/client requests, whose first request
  after a reload was the main error source). Largest per-request errors again on the first all-on-server request after
  a server reload (requests 6 / 11 / 22: −10 to −14%). Reloads are now over-predicted (16.0–17.7 vs 19.3 s; earlier
  under- and over-predicted, 38.8–97.8 vs 40.9–89.5 s), so the predicted totals are +0.7 to +3.4% (earlier −0.8 to
  −5.2%).
Policy result: fixed per class is fastest (63.4 s); adaptive is +3.0 s (+4.7%), of which 1.3 s is the PING, and the
rest is reload variation (repeat 0; without it +0.2 s); switching-aware is slowest (+14.8 s, +23%): it kept the
unprotected split for open requests 6 and 9 to avoid reloads (≈ +12 s of request time), yet made 4 instead of 3
configuration changes and its reloads were not shorter (16.6 vs 16.0 s), so its one-request-ahead rule did not pay
off on this trace. On this
trace (mobile network, no server-load change) the network did not change any decision (as predicted, 9l), so the
adaptive policy cannot gain over fixed per class; the gain of adaptation remains MODELLED (server-load trace, 9l).
(19 Nov: now also MEASURED on the server-load trace, §9n.)

### 9n. Real-runtime SERVER-LOAD trace (MEASURED, 19 Nov 2026)

Trace: the modelled server-load trace UNCHANGED (adaptive_eval.make_trace("server_load", 200, seed 11): stable 4G,
server share 0.25 ± 3% for requests 66–132, else 1.0 ± 3%); prompt lengths mapped to {32, 128, 256, 512} as in 9m.
ANSWER LENGTH A = 128, as in the modelled traces (adaptive_eval); the mobile run (9m, Figure 5) used A = 64. Reason
(model check, runtime/check_server_load_model.py -> check_out/server_load_model_check.csv, MODELLED): with the runtime
settings and A = 128 adaptive and fixed per class differ on exactly the 15 requests of task4_decisions.csv; with
A = 64 only on 10 of them (the long private prompts 73, 77, 94, 102, 130 stay at T = 0). Server load: EMULATED
(server.py: after every server computation the server waits compute x (1/share − 1), counted as server compute;
the PING reply reports the share to the adaptive estimator). Policies: adaptive and fixed per class, 3 repeats each,
interleaved (A, F, A, F, ...), fresh server per run; switching-aware NOT run on this trace (reported from 9m only).
Setup as 9m (M2-4T, client CPU fp16, server MPS fp16 on one Mac, one configuration in memory at a time). Started
after a reboot with other apps closed; before the start: no leftover runtime processes, about 4.2 GB available
(memory_pressure 79% free). Run 08:36–11:30 (2 h 54 min), exit 0; no request with swap growth; available memory
during the run 0.92–3.72 GB. Smoke run before (15-request version of the trace, 1 repeat, code check only, not data):
runtime/check_out/task4_real_trace_smoke_server_load.*. Raw: runtime/check_out/task4_real_trace_server_load.*.
Served: 122 of 200 requests per run (82 open, 40 private prompt; 78 private answer / private prompt+answer infeasible,
not run). Decisions identical in all 3 repeats of each policy; adaptive and fixed per class differ on exactly the
15 requests of the model check in every repeat (73, 74, 75, 77, 79, 81, 85, 86, 88, 94, 102, 107, 125, 130, 133:
private prompt, H = 4 secret head T = 11 instead of T = 0; 14 in the slow phase, 133 right after it); 58
configuration changes per run for both policies.

| Policy | Requests (incl. PING), predicted (error) | Reloads (predicted) | Total (predicted, error) | MAPE |
|---|---|---|---|---|
| Adaptive | 1440.2 ± 3.7 s (PING 7.5 s), 1419.6 s (−1.4%) | 237.1 ± 12.7 s (272.9) | 1677.2 ± 16.3 s (1692.5, +0.9%) | 3.3% |
| Fixed per class | 1522.5 ± 2.5 s, 1464.6 s (−3.8%) | 235.3 ± 8.7 s (284.4) | 1757.8 ± 11.0 s (1749.0, −0.5%) | 3.2% |

(mean ± s.d. over 3 repeats; error = (predicted − measured) / measured.)
Gain of adaptive over fixed per class: total −80.6 s (−4.6%; per repeat, paired by repeat index: 75.1 / 86.8 /
79.8 s); request time −82.3 s (−5.4%); reloads +1.7 s (no reduction, same number of changes). The gain is about 5x
the larger per-policy s.d. of the total (16.3 s) and every paired repeat shows a gain of ≥ 75 s, so it is larger than
the run-to-run spread (3 repeats; no formal test). On the 15 differing requests the adaptive choice saves 90.8 s of
request time (e.g. P = 32: 15.4 vs 23.6 s; P = 512: 26.2–28.0 vs 30.1–33.0 s; request 133, share back to 0.95: 15.9 vs
15.8 s, no gain); the PING costs 7.5 s over 122 requests. Predicted gain (MODELLED): 56.5 s (3.2%); the measured gain
is larger because the model under-predicts fixed per class's T = 0 split requests under the slow server (slow-phase
request time −4.5 to −5.0% error vs −0.6 to +0.3% for adaptive). Reloads are over-predicted for both (predicted 15% / 21% above measured).
Disturbed repeats: none clearly disturbed. Adaptive repeat 0 (the first run after the reboot, as in 9m) has the
longest reloads (251.2 vs 226.7 / 233.4 s, +21 s) with normal request time (1444.1 vs 1436.8 / 1439.5 s); likely cold
file cache (NOT tested); without it the paired gain is 83.3 s (repeats 1–2). No request has a prediction error above 15%.

## 10. Task 5: privacy on generated answers, known pairs and sessions (MEASURED, Colab T4, 16–18 Nov 2026)

Saved adapters of the current multisplit script: (4,2) seed 0 and (2,2) seed 1 (one training seed per split; no
seed spread for any number in this section). Raw outputs: results/task5/ (5a, 5b), results/task5d/ (5d; attacker
A checkpoints invA_H*.pt are not committed). Sanity check: the teacher-forced answer leakage recomputed by 5a and the
prompt leakage of attacker A recomputed by 5b reproduce the multisplit per-seed values exactly (e.g. (4,2) λa = 0.03:
40.88% and 83.38%; (2,2) λa = 0.03: 41.18%).

### 10a. Generated-answer leakage vs the teacher-forced proxy (step_task5_privacy/answer_generation.py)

100 chat-format prompts with planted details (name, hospital, disease, city, phone, amount); the PROTECTED client
generates the answer (greedy, ≤ 96 tokens); the server runs the PUBLIC tail + output layer on the states it returns.
Generated = content leakage on the generated tokens [95% CI]; teacher-forced = the proxy used by the controller
(WikiText). Details: planted details that appear in the answer, and the share of those the server recovers fully.

| Split | λa | Generated | Teacher-forced | Details appearing | Fully recovered |
|---|---|---|---|---|---|
| (4,2) | 0 | 94.6% [93.7, 95.6] | 93.4% | 190/323 | 94.2% |
| (4,2) | 0.01 | 87.7% [85.8, 89.4] | 74.5% | 152/323 | 92.8% |
| (4,2) | 0.03 | 70.8% [68.4, 73.3] | 40.9% | 203/323 | 78.3% |
| (4,2) | 0.1 | 40.1% [37.2, 43.2] | 10.9% | 221/280 | 24.9% |
| (2,2) | 0 | 95.9% [95.0, 96.7] | 94.5% | 176/323 | 95.5% |
| (2,2) | 0.01 | 86.5% [84.8, 88.0] | 72.0% | 188/323 | 95.2% |
| (2,2) | 0.03 | 68.1% [64.8, 70.9] | 41.2% | 175/323 | 69.1% |
| (2,2) | 0.1 | 54.4% [49.5, 59.8] | 13.7% | 262/323 | 30.9% |

Finding: on generated answers the server recovers much more than the teacher-forced proxy indicates: +1.2 to
+1.4 points without answer hiding, +27 to +30 points at λa = 0.03, +29 to +41 points at λa = 0.1. The
controller's answer limits (40% / 50%) are checked on the proxy; on generated answers no secret head + tail option
with agreement ≥ 75% (λa ≤ 0.03) is below 68%, so the feasibility of the private-answer and private prompt+answer
classes for server-using options rests on the proxy (NOT changed in the controller; a decision for the next step).
A possible cause (NOT tested): generated answers copy the prompt's details and follow the model's own distribution,
which the public tail predicts more easily than WikiText continuations.
Failed / limiting observations: at (4,2) λa = 0.1, 17 of 100 prompts produced an empty answer (EOS first) and are
excluded (n = 83); the public model refuses some prompts (e.g. money transfers), and in its own answers 57 of 99
planted details appear (first 30 prompts), so detail-level numbers cover only the details that appear.

Additional entries (19 Nov 2026; Colab T4; saved adapters (1,1) seed 1 and (8,2) seed 0, λa = 0.03 / 0.1; raw:
results/task5/5a_H1_T1/, 5a_H8_T2/; run with a temporary notebook, not committed). Sanity check: the recomputed
teacher-forced values reproduce the multisplit per-seed values ((1,1) seed 1: 68.20 / 41.17%; (8,2) seed 0: 38.52 /
16.95%; table: 68.20 / 41.17 / 38.52 / 16.95%). No empty answers (n = 100 in all four).

| Split (seed) | λa | Generated | Teacher-forced | Gap | Details appearing | Fully recovered |
|---|---|---|---|---|---|---|
| (1,1) (seed 1) | 0.03 | 87.6% [86.5, 89.0] | 68.2% | +19.4 | 186/323 | 97.8% |
| (1,1) (seed 1) | 0.1 | 85.9% [84.4, 87.5] | 41.2% | +44.7 | 167/323 | 92.2% |
| (8,2) (seed 0) | 0.03 | 83.8% [82.1, 85.4] | 38.5% | +45.3 | 185/323 | 81.1% |
| (8,2) (seed 0) | 0.1 | 17.0% [13.1, 21.0] | 17.0% | +0.05 | 210/323 | 14.3% |

Finding: gaps at λa ≥ 0.03 now span +0.05 to +45.3 points (12 entries); the largest, +45.3 ((8,2) λa = 0.03), replaces
+40.8 as the gap the controller adds to unmeasured entries. (8,2) λa = 0.1 is the only λa ≥ 0.03 entry without a gap;
its agreement is 61.4% (fails Q = 75%) and the example answers include refusals and repeated phrases, so the low
generated leakage may partly reflect degraded answers (observation, NOT tested; one seed). (1,1) has two training
seeds, but the gap is measured on seed 1 only and is added to the robust (max over seeds) teacher-forced value.

### 10b. Known-pairs attacker (step_task5_privacy/known_pairs_attack.py; λa = 0.03)

Attacker A fine-tuned on k known (text, protected state) sequences of 64 tokens (300 steps); "scratch" = an inverter
trained only on the pairs. Prompt side, content leakage (full private-detail recovery):

| Split | k = 0 (A) | k = 10 | k = 100 | k = 1000 | scratch k = 1000 | B (secret, full data) |
|---|---|---|---|---|---|---|
| (4,2) | 83.4% (68.4%) | 81.8% (56.9%) | 86.5% (66.9%) | 93.1% (84.0%) | 83.9% (37.0%) | 94.7% (80.3%) |
| (2,2) | 96.8% (95.4%) | 97.8% (92.4%) | 99.4% (93.9%) | 99.8% (94.5%) | 96.0% (63.2%) | 99.6% (93.4%) |

Finding: with 1000 known sequences (64k tokens) attacker A at (4,2) closes most of the gap to B (93.1 vs 94.7%) and
exceeds B in full detail recovery (84.0 vs 80.3%); 10 pairs slightly lower leakage (81.8%, CI 81.1–82.4 vs
82.8–83.9; over-fitting of 300 steps on 640 tokens). The secret adapters protect against attacker A only while the
attacker holds few known pairs. Answer side (learned answer inverter; known issue, NOT used for reported answer
leakage): (4,2) 20.0 -> 36.6% at k = 1000, B 31.7%; (2,2) 20.7 -> 35.9%, B 31.1%; in these runs B ≥ A holds on the
answer side, but A + 1000 pairs exceeds B.
LIMITATION (18 Nov 2026): the secret adapters protect the prompt against a server WITHOUT known data; a server that
collects known prompt–state pairs from the same user (same secret adapters) largely undoes the prompt protection
(1000 pairs: 83.4 -> 93.1% at (4,2), close to the 94.7% of attacker B, which holds the secret). Changing the secret
seed periodically (re-keying) would bound the number of pairs per secret; it is future work, NOT implemented. The
controller keeps attacker A; Figure 8c–d and §9l give the attacker-B sensitivity check.

### 10c. Session-level leakage, C2 (step_task5_privacy/session_attack.py; PRE-REGISTERED analysis, AUDIT §4)

600 templated sessions (sessions.py, seed 0; 5–10 turns, 190–389 tokens, mean 286.9; 55% of tokens beyond position
128), 400 train / 200 test sessions; 1000 test details (447 repeated ≥ 2 times, 553 mentioned once). KV-cached
capture matches the full forward pass (max relative difference 2.2e-3 at H = 2, 2.9e-3 at H = 4). 11–14 min per
configuration.

MAIN C2 RESULT (pre-registered): realistic session vote − realistic per-turn (first-mention turn), attacker A,
paired bootstrap (1000 resamples, 95% CI), percentage points; the confidence-weighted vote is the ONE secondary variant.

| Config | Per-turn | Session vote | Δ all | Δ repeated | Δ once | Linkage name+condition | Weighted Δ all (repeated) |
|---|---|---|---|---|---|---|---|
| (4,2) λa = 0 | 77.2% | 77.2% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 75.5 -> 75.5% | −0.4 [−0.8, −0.1] (−0.9 [−1.8, −0.2]) |
| (4,2) λa = 0.03 | 67.8% | 67.8% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 57.5 -> 57.5% | +0.1 [0.0, +0.3] (+0.2 [0.0, +0.7]) |
| (2,2) λa = 0 | 99.4% | 99.5% | +0.1 [0.0, +0.3] | +0.2 [0.0, +0.7] | +0.0 [0.0, 0.0] | 100 -> 100% | +0.1 [0.0, +0.4] (+0.2 [0.0, +0.7]) |
| (2,2) λa = 0.03 | 97.9% | 97.9% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 96.5 -> 96.5% | +0.0 [0.0, 0.0] (+0.0 [0.0, 0.0]) |

Answer to C2 in this setup: a server that observes whole sessions does NOT recover measurably more private details
than per-turn analysis, protected or unprotected; the upper CI bound of the gain is ≤ 0.3 points (all details) and
≤ 0.7 points (repeated details). The weighted vote does not change this (−0.4 to +0.1 points).
Reference points (not the C2 test): with the mention alignment GIVEN (upper bounds), union gains at most +0.4 points
for A ((4,2) λa = 0.03; repeated +0.7) and +1.9 [1.1, 2.8] points for attacker B ((4,2) λa = 0.03; repeated
+4.3 [2.5, 6.0]); the aligned plain vote is not better than per-turn (−0.2 to −5.1 points, repeated −0.4 to −11.4; CI excludes 0
except (2,2) λa = 0), wrong mentions outvoting right ones. Interpretation (not tested): repeated mentions of a detail
are decoded nearly identically, so a second mention adds little information for this attacker.
Protection effect by absolute position (attacker A, protected − public, per mention; 0–127 / 128–255 / 256+ tokens):
(4,2) λa = 0: −16.2 / −26.5 / −34.3 points; λa = 0.03: −28.1 / −33.8 / −38.3; (2,2) λa = 0: 0.0 / +0.1 / +0.4;
λa = 0.03: −2.1 / −1.4 / −1.6 (unprotected control at H = 4: 94.0 / 91.0 / 94.8%). The (4,2) protection does not
fade beyond the 128-token training length; it grows. (2,2) gives almost no prompt protection (as in §5b).
Consistency: attacker B ≥ attacker A in every configuration (realistic per-turn: (4,2) 81.4 vs 77.2% and 76.6 vs
67.8%; (2,2) 99.7 vs 99.4% and 98.2 vs 97.9%). Unprotected attacker A: (4,2) 94.7%, (2,2) 99.2%.
FAILED variant (reported, not fixed): the template-token attacker (A fine-tuned on public chat-template tokens of the
train sessions) is WORSE than plain attacker A everywhere: 25.6% / 16.0% at (4,2) λa = 0 / 0.03, 65.7% / 65.0% at
(2,2) (realistic per-turn). In the (4,2) λa = 0 run its fine-tuning loss reached 0.000 on 22 distinct target tokens
(Colab log): it collapses to the template vocabulary. Any fix would be EXPLORATORY under the pre-registration.
Limits: templated sessions with fixed assistant replies; one training seed per split; no learned cross-turn model;
attacker A decodes each token independently.
