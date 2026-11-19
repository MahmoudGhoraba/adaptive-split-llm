# Audit of the existing implementation and roadmap (14 Nov 2026)

Statuses: IMPLEMENTED AND VERIFIED · IMPLEMENTED BUT NOT VALIDATED · PARTIALLY IMPLEMENTED · DOCUMENTED ONLY · NOT IMPLEMENTED

## 1. What the code actually does (inspected)

| Area | Finding |
|---|---|
| U-split execution | No real split. All GPU experiments run the whole model in one process and read hidden states via hooks. |
| Latency model | `latency()`: FLOPs / assumed speed + bytes / rate + RTT per round trip; `server_share` scales server speed. Never compared to a measurement. |
| Memory model | `memory_gb()`: layer params + KV cache + fp32 adapters. Analytical. |
| Prompt attack | Per-token MLP inverter on layer-H states; WikiText-2 + templated private-detail prompts; no chat template. Sanity checks pass. |
| Answer leakage | `server_public_tail()`: teacher-forced next-token comparison on WikiText, not generated answers. Learned answer inverter fails its check. |
| Quality | Teacher-forced next-token agreement + perplexity ratio. No generation/task benchmark. |
| Optimiser | `decide_request()`: exhaustive search over pre-measured options (529 after multisplit load) under 4 constraints. |
| Controller | Re-solves per request using conditions supplied by a synthetic trace; no measurement or online estimation. |

## 2. Status per contribution

### C1 Adaptive privacy-aware optimisation
| Requirement | Status |
|---|---|
| Layer placement (H, T) | IMPLEMENTED AND VERIFIED |
| Protection mechanism + strength | PARTIALLY IMPLEMENTED (measured grid only: λp=0.1, λa∈{0,0.01,0.03,0.1}; secret-head λp only at H=4) |
| Latency objective | IMPLEMENTED AND VALIDATED (calibrated model, RESULTS_SUMMARY §8–9); latency() kept, assumed |
| Memory constraint | IMPLEMENTED AND VALIDATED for the CPU client (measured footprint = memory_gb() + 0.40 GB); budgets ASSUMED |
| Quality constraint | IMPLEMENTED (proxy metric) |
| Prompt/answer leakage constraints | IMPLEMENTED with measured tables (not prompt-length dependent) |
| Reacting to bandwidth/RTT/compute/load | IMPLEMENTED AND PARTIALLY VALIDATED (14 Nov): calibrated model + online EWMA estimates;
  simulated traces (stable / mobile / server load) and one 30-request real-runtime trace (measured −8.0% vs fixed per class,
  prediction error −2.9%). Switching cost modelled (in-situ fit) + switching-aware policy; 3-repeat real check: switching-
  aware fastest incl. reloads (208.4 s vs 216.6 fixed, 238.1 adaptive; earlier rule). Rerun under the current rule
  (19 Nov, RESULTS_SUMMARY 9m; 15 of 30 requests servable): fixed per class 63.4 s, adaptive 66.4 s (same decisions,
  + PING), switching-aware 78.2 s; request-time prediction −2.2 to −2.8%. Server load: server-REPORTED in PING (timing probe
  fails on MPS). Open: reload model from few samples; myopic switching rule; one machine; budgets ASSUMED |

Controller verdict: per-request re-optimisation over a measured configuration table (like ILLCO in Zamzam et al.),
not a fixed rule — but inputs are given, not observed; no switching cost; no robustness-to-estimation-error test.
18 Nov 2026 (RESULTS_SUMMARY 9l): all on client removed from the candidates (reference baseline only); answer leakage
= robust proxy + generated-answer gap (5a). Result: only open and private prompt are servable at any budget (2–8 GB);
private answer and private prompt+answer are infeasible; decisions are class-driven, the network changes none, server
load changes 7–15 of 200 per trace. Attacker-B sensitivity: private prompt also infeasible.

### C2 Session-level leakage — IMPLEMENTED AND MEASURED (18 Nov 2026; RESULTS_SUMMARY §10c): pre-registered main result
(realistic session vote − realistic per-turn, attacker A) +0.0 to +0.1 points at (4,2) and (2,2), λa 0 / 0.03, upper CI ≤ +0.3
(repeated ≤ +0.7): no measurable session gain in this setup. Template-token attacker FAILED (collapse). Open: templated
sessions, one training seed per split, no learned cross-turn attacker.
### C3 Hardware calibration — IMPLEMENTED BUT PARTIALLY VALIDATED (14 Nov): real split runtime (runtime/), measured timing per
phase / memory per side on the Mac (client CPU fp16, server MPS fp16), fitted per-side latency model (test MAPE 0.6–6.7%),
composition validated under emulated Wi-Fi/4G/poor (−3.5% to −7.2%); see RESULTS_SUMMARY §7–8. Open: server/client slow down
after idle gaps (not in the model); only one machine (no real network link); device profiles still ASSUMED.
### C4 Prompt/answer leakage evaluation
| Item | Status |
|---|---|
| Prompt reconstruction | IMPLEMENTED AND VERIFIED (one attacker type) |
| Generated-answer reconstruction | IMPLEMENTED AND MEASURED (5a, 18 Nov; one seed per split): generated leakage 19–45 points above the teacher-forced proxy at λa ≥ 0.03 (12 entries incl. (1,1), (8,2) of 19 Nov; exception (8,2) λa 0.1, +0.05); since 18 Nov the controller adds the gap |
| Gaussian noise | IMPLEMENTED AND VERIFIED |
| Secret LoRA | IMPLEMENTED AND VERIFIED at 7 splits (H=1..8, current script); seeds: (4,2) 3, (2,2) 2, (1,1) 2, others 1 |
| Stronger attackers (known pairs, optimisation-based) | Known pairs IMPLEMENTED AND MEASURED (5b, λa 0.03): k = 1000 brings A at (4,2) from 83.4% to 93.1% (B 94.7%). Optimisation-based NOT IMPLEMENTED |
| Task-level quality | NOT IMPLEMENTED |

(Numbering above follows the current priority order in README.md, Research questions.)

## 3. Novelty notes (not exhaustive; two web searches + literature sheet)
- Secret-adapter defence overlaps with Gu et al., ICML 2026 (ADMI: adapters + mutual-information regularisation,
  dual-sided prompt/response leakage). Possible distinct parts: secret random init, server-runs-public-tail threat,
  multi-split measurement, content-token metric. Read the paper before claiming more than "empirical evaluation".
- Hardware calibration as a method is not new (Neurosurgeon profiles per-layer latency); value = validation.
- C1: plausible new combination (hard measured prompt+answer leakage limits in LLM split selection);
  related: Cheng et al. (CNN, SSIM budget), Koch et al. (privacy as weighted term).
- Session leakage: related KV-cache inversion (Luo et al., NDSS 2026); no cross-turn aggregation study in split
  inference found in two searches — do not claim "first".
- Session leakage is only meaningful under protection (unprotected per-turn leakage is already 96–100%).
  Three testable mechanisms: (a) repeated mentions give several partial chances; (b) adapters trained on 128-token
  sequences may not generalise to long sessions; (c) chat-template tokens repeated every turn give the server free
  known (token, protected state) pairs → known-plaintext adaptation toward attacker B.

## 4. Agreed sequence and tasks

### Task 1 — Consolidate experiments
Objective: one consistent, documented measurement table.
Files: results/multisplit/, step_secret_multisplit/, step5_per_request/per_request_controller.py (`load_multisplit`).
Work: obtain or run batch C (H=6,T=2; H=8,T=2); rerun (4,2) seed 0 with the CURRENT script and drop or label the old
run; add seeds 1, 2 at the splits the controller selects ((1,1), (2,2), (4,1)); record seed count + script per entry.
Done when (revised 15 Nov 2026; replaces "every SECRET_U entry has ≥ 3 seeds"): all entries come from the same script
with a reported spread and step5 loads only that table; selected entries have ≥2 seeds where GPU time allows; all
entries are checked with the robust rule (RESULTS_SUMMARY 9h: max over seeds if ≥ 2 seeds, else mean + the largest
observed seed half range of that metric).
**Status (15 Nov 2026): DONE.** Robust rule implemented and applied to every option (`LEAK_CHECK = "robust"`).
Selected entries under the robust rule: (1,1) λa = 0.1 for private answer (1 seed) and the H = 4 secret head λp = 0.2
for private prompt (1 run, older secret_head.py): both are checked with the margin, and a second seed of (1,1) λa = 0.1
is the first GPU item if time allows. History (14 Nov): Done: batch C complete ((6,2) and (8,2), seed 0); (4,2) seed 0
rerun with the current script, (2,2) seed 1 added; step5 `load_multisplit()` now REPLACES SECRET_U by the current-script
table (28 entries, 7 splits) — the older-script (4,2) runs are no longer merged (kept as `SECRET_U_OLD_SCRIPT` and in
RESULTS_SUMMARY 5a as an older script version; their 65.5% prompt leakage at λa = 0.1 lies outside the new seeds'
74.3–78.5%); seed count per entry recorded (n_seeds in measured_multisplit.txt, `SECRET_U_SEEDS`) and spread reported
(`SECRET_U_SPREAD`, RESULTS_SUMMARY 5b). Seeds: (4,2) 3, (2,2) 2, the other five splits 1 (the old ≥ 3-seed criterion is replaced, see above). Measured spread: prompt leak up to ±4.9–6.3 points, answer leak ≤ ±3.0, agreement ≤ ±1.2, so
prompt-leak differences under ~10 points between splits are not reliable. Task 4 rerun with the new table:
outputs unchanged, (8,2) never chosen (RESULTS_SUMMARY 9g). The fragile margins found there ((2,2) and (4,2) at λa = 0.03)
led to the robust rule (9h): both entries now fail their limits; private prompt+answer is infeasible on 2 GB.

### Task 2 — Real client–server split runtime (runtime/)
Objective: execute head+tail and middle layers in separate processes exchanging hidden states.
Work: see runtime/README.md. Done when split output tokens equal the full model's output for the same prompt
(greedy decoding), with and without adapters.

### Task 3 — Measure real performance
Objective: measured prefill/decode time, transfer time, RTT, throughput, peak memory per (H, T), P and context length.
Done when: ≥ 5 repeats per setting after warm-up, mean ± spread logged to CSV with device/OS/library versions.

### Task 4 — Complete adaptive optimisation
Objective: controller driven by estimated conditions; fitted latency model.
Work: fit latency model on held-out prompt lengths (report error); online estimates (moving averages) of RTT,
throughput, server speed; compare per-request vs best fixed vs oracle on traces; sensitivity analysis.
Deferred fixes to apply here (found 14 Nov, not yet changed in the code):
- [DONE in calibrated_model.py; latency() unchanged for comparison] Round trips: `request_workload()` / `workload()` use `ROUND_TRIPS = 1 + A` and `HEAD = 2·P_HEAD·(1 + A)`;
  a greedy A-token answer needs 1 prefill + (A − 1) decode steps = A round trips and A output-layer calls.
- [DONE in calibrated_model.py] Memory constraint: use MEASURED process footprints (RESULTS_SUMMARY §7), not `memory_gb()` alone.
Done when: figures/tables for each scenario; decisions under assumed vs calibrated parameters compared.

### Task 5 — Strengthen privacy, then sessions
Status 18 Nov: 5a, 5b and 5d RUN on the real model (Colab T4; (4,2) seed 0, (2,2) seed 1); results RESULTS_SUMMARY §10.
5a needed a fix for the Colab transformers version (apply_chat_template returned a tokenizers.Encoding; encode_chat now
tokenizes the template text). 5c and 5e NOT started.
Status 14 Nov: 5a (step_task5_privacy/answer_generation.py) and 5b (known_pairs_attack.py) IMPLEMENTED, smoke-tested on CPU
only (tiny random model); NOT run on the real model. Need saved adapters: secret_multisplit.py --points "4,2" --seeds 0 and
--points "2,2" --seeds 1 with the current secret_ushape.py, then run_task5_colab.ipynb.
5d (C2): sessions.py (templated 5-10-turn sessions, Llama chat template), capture_sessions.py (KV-cached layer-H capture),
session_attack.py (per-turn / union / vote / template-token / B; per-detail, by turn, by position, name+condition linkage)
and run_task5d_colab.ipynb IMPLEMENTED, smoke-tested on CPU only (metric self-test with a known-quality attacker passes);
NOT run on the real model. union/vote are upper bounds (mention alignment given); no learned cross-turn model.
Added: realistic session attacker WITHOUT alignment (decode every user turn, extract candidates with public name/city/
condition lists and phone/amount patterns, frequency vote across turns; extraction exact with a perfect decoder on 600
real-tokenizer sessions) and an unprotected control by position bucket (protected - public, paired by mention).
Notebook runs λa = 0 and 0.03 for (4,2) and (2,2); attacker A cached per split.
Added: realistic PER-TURN attacker (same decode + extraction, one turn at a time: first-mention turn; any single turn).
MAIN C2 RESULT = realistic session vote - realistic per-turn (first-mention turn), paired bootstrap CI; aligned per-turn,
union/vote and B kept as reference points.

#### PRE-REGISTERED analysis plan for C2 (fixed 14 Nov 2026, BEFORE any real-model session run)
Data/setup: step_task5_privacy/sessions.py as committed (600 sessions, seed 0, 400 train / 200 test by session),
(4,2) and (2,2), λa in {0, 0.03}, saved adapters from secret_multisplit.py (H4_T2_seed0, H2_T2_seed1), attacker A as in
secret_ushape.py (cached per split), session_attack.py as committed.
- Main attacker: realistic session attacker = decode every user turn with attacker A, extract candidates with the public
  vocabularies/patterns, PLAIN frequency vote across turns (ties: earliest).
- Baseline: realistic per-turn attacker = same decode + extraction, one turn at a time, guess of the detail's
  first-mention turn.
- MAIN C2 RESULT: realistic session vote - realistic per-turn, paired bootstrap (1000 resamples, 95% CI), reported for
  (i) all details, (ii) repeated details (>= 2 mentions), (iii) details mentioned once; plus name+condition linkage.
- ONE secondary variant, defined now: confidence-weighted vote = same extraction, each candidate's vote weighted by
  the attacker's mean top-1 token probability over the candidate's token span; same baseline, same bootstrap, same
  three groups.
- Reference points (not the C2 test): aligned per-turn (first mention), union and aligned vote (UPPER BOUNDS, alignment
  given), realistic any-single-turn (optimistic), template-token attacker, attacker B, unprotected control by position.
- No other attacker variant will be added after seeing real results; any later idea is reported as EXPLORATORY.
5a generated-answer leakage (chat-format prompts, real generation, answer-side private details);
5b known-pairs attacker (k = 10/100/1000 pairs); 5c task-level quality (small QA set);
5d session dataset (templated 5–10 turns), capture with KV cache, per-turn vs session vs template-token attacker,
metrics: per-detail recovery per session, by turn index, name+condition linkage; 5e (optional) session constraint.

## 5. Schedule agreed on 14 Nov 2026 — all four contributions by 30 Nov (writing afterwards)

| Dates | Mac | GPU (Colab, one account; Kaggle as extra quota) | Done |
|---|---|---|---|
| 15–17 Nov | Task 2: split runtime + correctness test | Batch C: (6,2), (8,2) | — |
| 18–20 Nov | Task 3: timing, network, memory measurements; fit + validate latency model | (4,2) seed 0 with the current script | C3 |
| 21–23 Nov | Task 4: controller with estimated conditions vs fixed vs oracle | 5a: generated-answer leakage at the chosen splits | C1 |
| 24–25 Nov | 5d prep: session dataset (templated 5–10 turns) + capture code | 5b: known-pairs attacker (k = 10/100/1000) | C4 |
| 26–29 Nov | 5d analysis + figures | 5d: sessions on (4,2) and (2,2) protected + unprotected; per-turn vs session vs template-token attacker | C2 |
| 30 Nov | Buffer | — | — |

Out of scope until after 30 Nov (future work unless time remains): extra seeds at (1,1), (2,2), (4,1);
task-level quality (5c); session constraint in the optimiser (5e); multi-device game.
Do not use multiple Colab accounts (disallowed by Colab's FAQ).
