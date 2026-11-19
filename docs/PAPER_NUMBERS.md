# Paper numbers (single source for the write-up; as of 19 Nov 2026)

Every number the paper needs, CURRENT controller rule only (RESULTS_SUMMARY §9l: all on client is not a candidate,
only a reference baseline; prompt leakage robust to training variance; answer leakage = robust teacher-forced value +
generated-answer gap of task 5a; prompt attacker A in the controller, attacker B as a sensitivity check).
No superseded values (§6 "earlier table", §9c–§9k, the 9d/9f real-runtime runs and `*_meancheck*` files are NOT used).
Labels: MEASURED (experiment), MODELLED (calibrated model / controller rule applied to measured inputs), FITTED
(parameters fitted to measurements). "§" = section of docs/RESULTS_SUMMARY.md. Model: Llama-3.2-1B-Instruct (16 layers);
privacy experiments on a Colab T4; timing on a MacBook Pro (Apple M2, 8 GB), client CPU fp16 + server MPS fp16.

## 1. Leakage and quality of the secret head + tail at λa = 0.03 (all splits; λp = 0.1)

MEASURED (§5b; mean over training seeds; content leakage, %): prompt leakage of attacker A (no secret) and B (holds
the secret), full private-detail recovery of A, teacher-forced answer leakage, next-token agreement with the public
model. MODELLED (controller inputs, §9h/§9l): robust prompt leakage A (max over seeds if ≥ 2 seeds, else mean + 6.34
points), robust answer leakage (max over seeds, else mean + 3.03 points) + generated-answer gap (own gap if measured
in 5a, else +45.3 points), capped at 100%.

| Split (H,T) | Seeds | Prompt A | Prompt B | Detail recovery A | Answer (teacher-forced) | Agreement | Robust prompt A (MODELLED) | Robust answer + gap (MODELLED) | Gap used |
|---|---|---|---|---|---|---|---|---|---|
| (1,1) | 2 | 99.9 | 100.0 | 96.1 | 70.3 | 80.3 | 99.9 | 91.8 | own, +19.5 |
| (2,1) | 1 | 96.7 | 99.7 | 90.4 | 55.4 | 80.6 | 100.0 | 100.0 | max, +45.3 |
| (2,2) | 2 | 96.2 | 99.7 | 92.1 | 39.4 | 79.1 | 96.8 | 68.1 | own, +26.9 |
| (4,1) | 1 | 81.0 | 94.2 | 67.0 | 55.8 | 78.1 | 87.3 | 100.0 | max, +45.3 |
| (4,2) | 3 | 78.9 | 93.8 | 62.0 | 40.8 | 76.7 | 83.4 | 70.8 | own, +29.9 |
| (6,2) | 1 | 55.7 | 82.6 | 45.9 | 39.4 | 72.4 | 62.0 | 87.7 | max, +45.3 |
| (8,2) | 1 | 55.5 | 79.2 | 41.4 | 38.5 | 72.8 | 61.8 | 86.8 | own, +45.3 |

Seed spread (MEASURED, §5b, half range): prompt up to ±6.3 points, answer ≤ ±3.0, agreement ≤ ±1.2 (±4.7 at (1,1)
λa = 0.1); prompt-leakage differences below ~10 points between splits are not reliable. Unprotected prompt leakage
(MEASURED, §5b): 99.98 / 99.79 / 98.7 / 97.8 / 97.6% at H = 1 / 2 / 4 / 6 / 8.
Private-prompt option chosen by the controller (H = 4 secret head λp = 0.2, T = 0; MEASURED §4 run 2, one run):
prompt A 64.9%, B 89.3%, agreement 78.6%; robust (MODELLED) A 71.2%, B 90.1%; answer 100% (public tail by construction).

## 2. Task 5a: generated-answer vs teacher-forced answer leakage (MEASURED, §10a)

100 chat prompts with planted details, greedy answers ≤ 96 tokens; one training seed per split. 12 measured
configurations (the 8 of 16–18 Nov at (4,2) seed 0 and (2,2) seed 1, plus 4 of 19 Nov at (1,1) seed 1 and (8,2) seed 0).

| Split | λa | Generated [95% CI] | Teacher-forced | Gap (points) | Details appearing | Fully recovered |
|---|---|---|---|---|---|---|
| (4,2) | 0 | 94.6% [93.7, 95.6] | 93.4% | +1.2 | 190/323 | 94.2% |
| (4,2) | 0.01 | 87.7% [85.8, 89.4] | 74.5% | +13.2 | 152/323 | 92.8% |
| (4,2) | 0.03 | 70.8% [68.4, 73.3] | 40.9% | +29.9 | 203/323 | 78.3% |
| (4,2) | 0.1 | 40.1% [37.2, 43.2] | 10.9% | +29.2 | 221/280 | 24.9% |
| (2,2) | 0 | 95.9% [95.0, 96.7] | 94.5% | +1.4 | 176/323 | 95.5% |
| (2,2) | 0.01 | 86.5% [84.8, 88.0] | 72.0% | +14.5 | 188/323 | 95.2% |
| (2,2) | 0.03 | 68.1% [64.8, 70.9] | 41.2% | +26.9 | 175/323 | 69.1% |
| (2,2) | 0.1 | 54.4% [49.5, 59.8] | 13.7% | +40.8 | 262/323 | 30.9% |
| (1,1) | 0.03 | 87.6% [86.5, 89.0] | 68.2% | +19.5 | 186/323 | 97.8% |
| (1,1) | 0.1 | 85.9% [84.4, 87.5] | 41.2% | +44.7 | 167/323 | 92.2% |
| (8,2) | 0.03 | 83.8% [82.1, 85.4] | 38.5% | +45.3 | 185/323 | 81.1% |
| (8,2) | 0.1 | 17.0% [13.1, 21.0] | 17.0% | +0.05 | 210/323 | 14.3% |

Limits: at (4,2) λa = 0.1, 17 of 100 answers were empty (n = 83); (8,2) λa = 0.1 (agreement 61.4%) includes refusals
and repeated phrases (low leakage may partly reflect degraded answers; not tested).
Gap rule (MODELLED, §9l; `ANSWER_CHECK = "generated"`): answer leakage in the controller = robust teacher-forced value +
(generated − teacher-forced); measured entries use their own gap, every other head + tail entry the largest observed
gap, +45.3 points ((8,2) λa = 0.03); capped at 100%; no gap on by-construction values (public tail 100%).
Consequence (MODELLED): smallest answer leakage of any candidate 20.0% ((8,2) λa = 0.1, agreement 61.4%); with
agreement ≥ 75%: 68.1% ((2,2) λa = 0.03).

## 3. Task 5b: known-pairs attacker (MEASURED, §10b; λa = 0.03)

Prompt leakage of attacker A fine-tuned on k known (text, protected state) sequences of 64 tokens (full
private-detail recovery in brackets):

| Split | k = 0 (A) | k = 10 | k = 100 | k = 1000 | scratch, k = 1000 | Attacker B |
|---|---|---|---|---|---|---|
| (4,2) | 83.4% (68.4%) | 81.8% (56.9%) | 86.5% (66.9%) | 93.1% (84.0%) | 83.9% (37.0%) | 94.7% (80.3%) |
| (2,2) | 96.8% (95.4%) | 97.8% (92.4%) | 99.4% (93.9%) | 99.8% (94.5%) | 96.0% (63.2%) | 99.6% (93.4%) |

Limitation: a server that collects ~1000 known pairs from the same user largely undoes the prompt protection
(83.4 -> 93.1% at (4,2), B 94.7%); re-keying the secret is future work, not implemented.

## 4. C2: session-level leakage (MEASURED, §10c; pre-registered analysis)

600 templated sessions (5–10 turns, mean 286.9 tokens), 200 test sessions, 1000 test details (447 repeated, 553 once).
Main result: realistic session vote − realistic per-turn (first mention), attacker A, paired bootstrap 95% CI, points.

| Config | Per-turn | Session vote | Δ all | Δ repeated | Δ once | Linkage name + condition |
|---|---|---|---|---|---|---|
| (4,2) λa = 0 | 77.2% | 77.2% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 75.5 -> 75.5% |
| (4,2) λa = 0.03 | 67.8% | 67.8% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 57.5 -> 57.5% |
| (2,2) λa = 0 | 99.4% | 99.5% | +0.1 [0.0, +0.3] | +0.2 [0.0, +0.7] | +0.0 [0.0, 0.0] | 100 -> 100% |
| (2,2) λa = 0.03 | 97.9% | 97.9% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 96.5 -> 96.5% |

Upper CI bound of the session gain: ≤ 0.3 points (all details), ≤ 0.7 points (repeated).
Secondary (the one pre-registered variant), confidence-weighted vote, Δ all (repeated): (4,2) λa = 0: −0.4 [−0.8, −0.1]
(−0.9 [−1.8, −0.2]); (4,2) λa = 0.03: +0.1 [0.0, +0.3] (+0.2 [0.0, +0.7]); (2,2) λa = 0: +0.1 [0.0, +0.4]
(+0.2 [0.0, +0.7]); (2,2) λa = 0.03: +0.0 [0.0, 0.0] (+0.0 [0.0, 0.0]).
Upper bounds (not the C2 test; mention alignment given): union gain ≤ +0.4 points for A ((4,2) λa = 0.03; repeated
+0.7); attacker B +1.9 [1.1, 2.8] points ((4,2) λa = 0.03; repeated +4.3 [2.5, 6.0]). Aligned plain vote −0.2 to −5.1
points vs per-turn (repeated −0.4 to −11.4).
Position effect (attacker A, protected − public, per mention, at 0–127 / 128–255 / 256+ tokens): (4,2) λa = 0:
−16.2 / −26.5 / −34.3 points; (4,2) λa = 0.03: −28.1 / −33.8 / −38.3; (2,2) λa = 0: 0.0 / +0.1 / +0.4;
(2,2) λa = 0.03: −2.1 / −1.4 / −1.6 (unprotected H = 4: 94.0 / 91.0 / 94.8%). Protection grows beyond the
128-token training length.
Consistency: B ≥ A everywhere (per-turn (4,2) 81.4 vs 77.2% and 76.6 vs 67.8%; (2,2) 99.7 vs 99.4% and 98.2 vs 97.9%).
Failed variant: template-token attacker 25.6 / 16.0% at (4,2) λa = 0 / 0.03, 65.7 / 65.0% at (2,2) (worse than A).

## 5. Fitted speeds and model errors

Fitted per-side model (FITTED on P = 32 / 128 / 512, tested on P = 256 / 1024; §8b):

| Side / phase | Parameters (FITTED) | Train MAPE (max) | Test MAPE (max) |
|---|---|---|---|
| Client CPU prefill | f_eff = 131 GFLOP/s | 3.3% (11.4%) | 6.7% (15.7%) |
| Client CPU decode | 2.28 ms/layer + 0.35 µs/token/layer | 0.7% | 0.6% (2.3%) |
| Server MPS prefill | f_eff = 2.53 TFLOP/s | 5.1% (18.1%) | 4.0% (9.1%) |
| Server MPS decode | 1.28 ms/layer + 0.14 µs/token/layer | 1.0% | 2.1% (6.2%) |
| Output layer (MEASURED) | client 8.39 ms, server 5.71 ms | – | – |

Effective decode speed (FITTED): client 53 / 49 GFLOP/s, server 95 / 91 GFLOP/s (context 64 / 1088). ASSUMED speeds
in the analytical model: client 10 GFLOP/s (laptop 20), server 1 TFLOP/s: client 5–13x too slow, server 2.5x too
slow for prefill and ~10x too fast for decode.
Device profiles (FITTED from MEASURED, §9b): M2-4T prefill 131 GFLOP/s, M2-2T 80.4, M2-1T 47.6 GFLOP/s; decode
2.28 / 2.28 / 2.55 ms/layer; memory budgets 4 / 2 / 2 GB (ASSUMED).
Idle-gap slowdown (MEASURED, §8c): server decode compute +67% after a 250 ms gap (16.2 -> 27.0 ms).
Model errors:
- Fitted model + network formula vs MEASURED end-to-end (§8d, P = 256, A = 64): −0.6% to −7.2% (total and TTFT);
  ASSUMED latency(): +35% to +387% on the same requests.
- Calibrated model (MODELLED, §9a) vs task-3 measurements: emulated networks −2.5% to +0.6%; held-out loopback
  −10.9% to +3.5%.
- Real-runtime traces (MEASURED vs MODELLED, §9m/§9n): request time −1.4% to −3.8%, per-request MAPE 2.9–3.4%;
  reload model over-predicts by 9–21% (§9 below).
- Memory (MEASURED, §8e): client footprint = memory_gb() + 0.40 GB (7 of 8 configs; 0.52 GB at (4,2) with adapters).

## 6. Feasibility of the privacy classes (MODELLED, §9l; P = 256, A = 128, nominal 4G)

Class limits: open Q 95%; private prompt ε_p 80%, Q 75%; private answer ε_a 40%, Q 75%; private prompt+answer
ε_p 80%, ε_a 50%, Q 75%. All candidates need ≤ 2.21 GB ((8,2) is the only one above 2 GB).

| | 2 GB | ≥ 3 GB (3, 4, 8 GB identical) |
|---|---|---|
| Grid combinations feasible, attacker A (controller) | 24 / 150 | 34 / 150 |
| Grid combinations feasible, attacker B (sensitivity) | 12 / 150 | 17 / 150 |
| Servable classes, attacker A | open (all on server, 4.2 s), private prompt (H = 4 secret head T = 0, 16.6 / 17.2 / 18.7 s on M2-4T / 2T / 1T) | same |
| Servable classes, attacker B | open only | open only |

Grid: ε_p ∈ {60, 70, 80, 90, 100}%, ε_a ∈ {20, 30, 40, 50, 100}%, Q ∈ {60, …, 85}%. The 10 / 5 combinations added
at ≥ 3 GB all have Q = 60% ((8,2) λa = 0.1).

Smallest single-limit relaxation per infeasible class (sensitivity analysis; class limits NOT changed; "none" = no
relaxation of that limit alone helps; a memory relaxation never helps):

| Attacker | Budget | Class | ε_p | ε_a | Q |
|---|---|---|---|---|---|
| A | 2 GB | private answer | none | 40 -> 68.1% ((2,2) λa 0.03) | none |
| A | ≥ 3 GB | private answer | none | 40 -> 68.1% ((2,2) λa 0.03) | 75 -> 61.4% ((8,2) λa 0.1: prompt 65.2%, answer 20.0%) |
| A | 2 GB | private prompt+answer | none | 50 -> 100% (= private-prompt option) | 75 -> 61.3% ((4,2) λa 0.1: prompt 78.5%, answer 46.2%) |
| A | ≥ 3 GB | private prompt+answer | none | 50 -> 100% (= private-prompt option) | 75 -> 61.4% ((8,2) λa 0.1) |
| B | 2 GB | private prompt | 80 -> 86.3% ((6,2) λa 0, 1 seed) | none | 75 -> 6.1% (noise σ = 2; not useful) |
| B | ≥ 3 GB | private prompt | 80 -> 82.0% ((8,2) λa 0.01, agreement 77.3%) | none | 75 -> 6.1% (not useful) |
| B | 2 GB | private answer | none | 40 -> 68.1% ((2,2) λa 0.03) | none |
| B | ≥ 3 GB | private answer | none | 40 -> 68.1% | 75 -> 61.4% ((8,2) λa 0.1, B 86.3%) |
| B | all | private prompt+answer | none | none | none |

## 7. Calibration impact (MODELLED, §9l; Figure 6)

Decisions that change between the ASSUMED model (assumed device of equal memory) and the calibrated model.
Grid = 5 prompt lengths x 3 networks x server share {1, 0.25} = 30 points per class; traces = 3 x 200 requests.

| Profile (assumed counterpart) | Open (grid) | Private prompt (grid) | Private answer / P+A (grid) | Trace requests changed |
|---|---|---|---|---|
| M2-4T (laptop) | 3/30 | 12/30 | 0/30 / 0/30 (infeasible in both) | 14/600 (2.3%) |
| M2-2T (phone) | 1/30 | 10/30 | 0 / 0 | 10/600 (1.7%) |
| M2-1T (phone) | 0/30 | 7/30 | 0 / 0 | 6/600 (1.0%) |
| assumed edge (edge) | 2/30 | 10/30 | 0 / 0 | 11/600 (1.8%) |
| assumed phone (phone) | 0/30 | 0/30 | 0 / 0 | 0/600 |

All trace changes are private prompt (secret head T = 0 -> T = 11 on M2-4T / edge, T = 4 on M2-2T / M2-1T).

## 8. Policy comparison in the calibrated model (MODELLED, §9l)

200 requests per trace, A = 128; every policy serves 122 of 200 (open 82, private prompt 40); estimating policies 20
noise seeds (range ≤ 0.04 points). Gap to the oracle (%), compute time / including reloads (setting a, this testbed):

| Profile | Adaptive: stable | mobile | server load | Fixed per class: stable | mobile | server load |
|---|---|---|---|---|---|---|
| M2-4T | +0.72 / +0.56 | +0.76 / +0.62 | +0.64 / +0.28 | 0.00 / 0.00 | 0.00 / 0.00 | +3.81 / +3.61 |
| M2-2T | +0.70 / +0.55 | +0.74 / +0.61 | +0.57 / +0.95 | 0.00 / 0.00 | 0.00 / 0.00 | +0.79 / +0.58 |
| M2-1T | +0.65 / +0.52 | +0.70 / +0.58 | +0.56 / +0.82 | 0.00 / 0.00 | 0.00 / 0.00 | +0.37 / −0.27 |
| assumed edge | +0.69 / +0.54 | +0.73 / +0.60 | +0.77 / +1.09 | 0.00 / 0.00 | 0.00 / 0.00 | +1.93 / +1.88 |
| assumed phone | +0.16 / +0.15 | +0.19 / +0.18 | +0.15 / +0.14 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

On the stable and mobile traces adaptive makes the oracle's decisions; its gap is the per-request PING. best_fixed
(one option for all classes) is infeasible on every profile. On M2-4T, server-load trace: adaptive and fixed per class
differ on 15 requests (private prompt, T = 11 instead of T = 0). All on client (reference only, not a candidate,
200 served): M2-4T +20.1 / +0.4 / −11.4%, assumed edge +50.6 / +26.8 / +10.5% (compute, stable / mobile / server load);
does not fit on the 2 GB profiles.

## 9. Real-runtime traces (MEASURED; M2-4T, client CPU fp16 + server MPS fp16 on one Mac; 3 repeats; mean ± s.d.)

Error = (predicted − measured) / measured; predicted = calibrated model (MODELLED) + FITTED reload model. Reloads:
one configuration in memory at a time on this single-machine testbed. Decisions were identical in all repeats.
The two traces use DIFFERENT answer lengths (mobile A = 64, server-load A = 128 as in the modelled traces): do not pool.

Mobile trace (§9m, Figure 5): 30 requests (Wi-Fi -> 4G -> poor -> 4G -> Wi-Fi), A = 64, 15 served (12 open, 3 private
prompt); adaptive and fixed per class make the same decisions.

| Policy | Requests (predicted, error) | Reloads (predicted) | Total (predicted, error) | Changes | Per-request MAPE |
|---|---|---|---|---|---|
| Adaptive | 48.7 ± 0.3 s, PING 1.3 s (47.6, −2.2%) | 17.7 ± 4.5 s (19.3) | 66.4 ± 4.8 s (66.9, +0.7%) | 3 | 2.9% |
| Switching-aware | 61.6 ± 0.6 s, PING 1.3 s (59.9, −2.8%) | 16.6 ± 1.1 s (19.3) | 78.2 ± 0.5 s (79.1, +1.2%) | 4 | 3.4% |
| Fixed per class | 47.4 ± 0.3 s (46.3, −2.2%) | 16.0 ± 2.2 s (19.3) | 63.4 ± 2.3 s (65.6, +3.4%) | 3 | 2.9% |

Flagged: adaptive repeat 0 (first run after reboot) reloads 22.8 vs 15.2 / 14.9 s (without it 63.7 ± 0.2 s total).

Server-load trace (§9n, Figure 10): the modelled 200-request trace (stable 4G; emulated server share 0.25 for
requests 66–132), A = 128, 122 served (82 open, 40 private prompt); switching-aware not run. Adaptive and fixed per
class differ on exactly the 15 requests predicted by the model (73, 74, 75, 77, 79, 81, 85, 86, 88, 94, 102, 107, 125,
130, 133); 58 configuration changes each.

| Policy | Requests (predicted, error) | Reloads (predicted) | Total (predicted, error) | Per-request MAPE |
|---|---|---|---|---|
| Adaptive | 1440.2 ± 3.7 s, PING 7.5 s (1419.6, −1.4%) | 237.1 ± 12.7 s (272.9) | 1677.2 ± 16.3 s (1692.5, +0.9%) | 3.3% |
| Fixed per class | 1522.5 ± 2.5 s (1464.6, −3.8%) | 235.3 ± 8.7 s (284.4) | 1757.8 ± 11.0 s (1749.0, −0.5%) | 3.2% |

Gain of adaptive: −80.6 s total (−4.6%; paired per repeat 75.1 / 86.8 / 79.8 s), −82.3 s request time (−5.4%);
MODELLED gain 56.5 s (3.2%). Flagged: adaptive repeat 0 reloads 251.2 vs 226.7 / 233.4 s (paired gain without it
83.3 s).

## 10. Figures (docs/figures/, captions in CAPTIONS.md; all current)

1. `fig1_latency_model.png` — Calibrated vs assumed latency model: predicted vs MEASURED end-to-end time, (4,2) and (2,2), three emulated networks.
2. `fig2_decode_phases.png` — Per-phase breakdown of one MEASURED decode step at (4,2) (client compute, network, server compute).
3. `fig3_idle_gap.png` — MEASURED slowdown of each component after idle gaps before a decode message.
4. `fig4_memory.png` — MEASURED footprint per side vs MODELLED memory_gb() for every configuration.
5. `fig5_policies_runtime.png` — Real-runtime mobile trace (MEASURED, A = 64): request time + reloads per policy vs predicted.
6. `fig6_calibration_impact.png` — Share of decisions that change between the assumed and the calibrated model (MODELLED).
7. `fig7_multisplit_leakage.png` — MEASURED leakage vs quality of the secret head + tail at all seven splits.
8. `fig8_feasibility.png` — MODELLED minimum robust prompt leakage per (ε_a, Q), attacker A and B, 2 GB and 3–8 GB.
9. `fig9_decision_timeline.png` — MODELLED adaptive decisions over time on a 2 GB M2-1T client (mobile and server-load traces).
10. `fig10_server_load_runtime.png` — Real-runtime server-load trace (MEASURED, A = 128): adaptive vs fixed per class vs predicted.
