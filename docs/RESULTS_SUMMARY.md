# Results summary (as of 14 Nov 2026)

All leakage/quality numbers are MEASURED on Llama-3.2-1B-Instruct (Colab T4). All latency numbers are
MODELLED (assumed speeds). Leakage = content leakage (see README.md, Definitions). Raw files are in `results/`.

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

| λa | Agreement | Answer leak | Prompt leak A | Prompt leak B |
|---|---|---|---|---|
| 0 | 87.0% | 92.8% | 86.9% | 94.7% |
| 0.01 | 83.6% | 72.9% | 88.9% | 95.0% |
| 0.03 | 76.8% | 41.0% | 79.1% | 94.5% |
| 0.1 | 61.1% | 12.2% | 65.5% | 93.6% |
| 0.3 | 39.7% | 0.8% | 70.8% | 96.7% (failed quality check) |

λp=0.2, λa=0.3 collapsed the model (agreement 16.7%).

### 5b. Multi-split runs (results/multisplit; current script) — prompt leak A / answer leak / agreement

| Split | λa=0 | λa=0.01 | λa=0.03 | λa=0.1 | seeds |
|---|---|---|---|---|---|
| H=1,T=1 | 99.97 / 96.0 / 93.1 | 99.98 / 96.3 / 92.9 | 99.89 / 72.4 / 80.7 | 99.97 / 35.5 / 78.9 | 1 |
| H=2,T=1 | 98.7 / 95.2 / 91.6 | 98.9 / 84.5 / 88.7 | 96.7 / 55.4 / 80.6 | 72.0 / 19.8 / 61.6 | 1 |
| H=2,T=2 | 98.8 / 94.5 / 92.0 | 98.6 / 72.8 / 87.2 | 95.6 / 37.6 / 79.2 | 74.7 / 16.7 / 62.1 | 1 |
| H=4,T=1 | 88.1 / 95.0 / 86.8 | 86.2 / 84.6 / 84.6 | 81.0 / 55.8 / 78.1 | 63.6 / 27.4 / 61.7 | 1 |
| H=4,T=2 | 88.0 / 93.0 / 86.8 | 83.3 / 73.1 / 83.1 | 76.7 / 40.7 / 76.5 | 77.1 / 14.4 / 61.8 | 2 (seeds 1, 2) |
| H=6,T=2 / H=8,T=2 | missing (batch C) | | | | |

Seed spread at (4,2), half-range: prompt leak ±0.1–3.1 points, answer leak ±0–2.5, agreement ±0.1–0.8.
Attacker B (with secret): 99.6–100% at H ≤ 2; 93–95% at H = 4.
Consistency check "B ≥ A" fails at (2,2) and (4,2) on the ANSWER side only (learned answer inverter).

Findings: prompt hiding needs H ≥ 4 (H=1 hides nothing); answer hiding works at every split; T=2 hides the
answer better than T=1 at equal quality. Differences below ~5 points are within seed variation.

## 6. Controllers (modelled latency)

Step 3c (20 tasks, 2 GB): prompt ≤ 80%, answer ≤ 50%, quality ≥ 75% → H=4, T=2, λa=0.03, 15.5 min (old (4,2) data).

Step 5 per request (256-token prompt, 4G), with all 5 measured splits loaded:

| Device (assumed) | Open | Private answer | Private prompt | Prompt + answer |
|---|---|---|---|---|
| Phone 2 GB | server 0.9 s | H=1,T=1 λa=0.1, 40.9 s | H=4 secret head λp=0.2, 60.2 s | H=4,T=2 λa=0.03, 79.4 s |
| Laptop 4 GB | server 0.9 s | H=1,T=1 λa=0.1, 16.2 s | H=4 secret head λp=0.2, 21.0 s | H=4,T=2 λa=0.03, 25.7 s |
| Edge 8 GB | server 0.9 s | all on client 8.2 s | all on client 8.2 s | all on client 8.2 s |

Earlier stream result (batch A data only): laptop on a mobile trace 55.8 min (per request) vs 66.9 min
(fixed per class) for 200 requests; equal on a stable network; phone gains nothing. Must be rerun after task 1.

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
