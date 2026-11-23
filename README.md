# Privacy-Constrained Adaptive Split Inference of Large Language Models on Memory-Limited Devices

Code, raw results and figures for the paper and the bachelor thesis of Mahmoud Ghoraba
(supervisors: Dr Talal El-Shabrawy, Dr Marwa Zamzam).

Llama-3.2-1B-Instruct (16 decoder layers, hidden size 2048) is split in a U shape: the client runs the embedding,
the first H layers (head), the last T layers (tail) and the output layer; the server runs the middle
U = 16 − H − T layers. Secret LoRA adapters on the client layers protect the prompt (head) and the answer (tail)
against a server that inverts the hidden states. A per-request controller chooses (H, T, protection) to minimise
latency under memory, quality, prompt-leakage and answer-leakage limits, using a latency/memory model calibrated
on a real split runtime. The formulation follows Zamzam, Elshabrawy & Ashour, IEEE Access 2021 (latency =
processing + transmission + propagation; their battery constraint -> memory, their accuracy constraint -> privacy
limits). The U-shaped split itself is not a contribution (it is used by earlier work, e.g. Λ-Split).

- All final numbers of the paper, with their status (MEASURED / MODELLED / FITTED): `docs/PAPER_NUMBERS.md`
- Every result with its full context and history: `docs/RESULTS_SUMMARY.md`
- Status of every part of the project: `docs/AUDIT_AND_ROADMAP.md`
- Figure captions: `docs/figures/CAPTIONS.md`

**Model weights are not included.** Llama-3.2-1B-Instruct must be downloaded from Meta / Hugging Face
(`meta-llama/Llama-3.2-1B-Instruct`, gated) and is licensed under the Llama 3.2 Community License. Built with Llama.

Commit hashes quoted in raw result files and docs refer to the original history of this repository; translate them
to the published history with `COMMIT_MAP.txt` (old hash -> new hash).

## Research questions

1. **Adaptive privacy-aware optimisation** (main contribution): jointly choose layer placement, protection mechanism
   and strength under latency, memory, quality, prompt-leakage and answer-leakage constraints, and re-optimise as
   bandwidth, RTT, client compute and server load change.
2. **Session-level leakage in multi-turn conversations:** does a server that observes many turns recover more private
   information than per-turn analysis, especially against a protected client? (analysis plan fixed before the runs:
   `docs/AUDIT_AND_ROADMAP.md`, "PRE-REGISTERED analysis plan for C2")
3. **Hardware-calibrated performance modelling and validation:** replace assumed speeds with measurements on a
   MacBook Pro (Apple M2, 8 GB) and validate the latency and memory model.
4. **Empirical prompt and answer leakage:** prompt/answer reconstruction, Gaussian noise, secret LoRA adapters,
   privacy–utility trade-off.

## Definitions

- **Content leakage** l = max(0, (c − g)/(1 − g)): c = top-1 recovery of content tokens (stopwords, punctuation and
  tokens shorter than 3 characters removed), g = blind guess (most frequent content token). 0 = nothing learned.
- **Full private-detail recovery:** share of planted details (names, hospitals, diseases, phone numbers, amounts,
  cities) recovered completely. Prompts with planted details are templated.
- **Attacker A:** inverter trained on the public model (main threat). **Attacker B:** trained on the protected model
  (holds the secret; upper bound). Inverter = LayerNorm → Linear → GELU → Linear scored against the tied embeddings,
  400k WikiText-2 training tokens, 2 epochs; 40k test tokens; 95% bootstrap CIs (300 resamples).
- **Answer leakage (teacher-forced):** the server runs the public tail + output layer on the state it returns; share
  of the client's next content tokens it matches, computed teacher-forced on WikiText text.
- **Generated-answer leakage (task 5a):** the same attack on greedily generated answers to chat prompts.
- **Quality:** next-token agreement with the public model (teacher-forced) and perplexity ratio (a proxy).
- **Robust privacy check (controller):** an option passes a leakage limit only if its robust leakage ≤ limit; robust =
  max over training seeds if ≥ 2 seeds, else mean + the largest seed half range observed for that metric (prompt
  +6.3, answer +3.0 points), capped at 100%.
- **Answer leakage in the controller:** robust teacher-forced value + the generated − teacher-forced gap measured in
  task 5a (own gap for the 12 measured entries, otherwise the largest observed gap, +45.3 points), capped at 100%.
- **Candidate set:** all on client is not a candidate for any privacy class; it is only the reference baseline.
  Prompt leakage in the controller = attacker A; attacker B is a sensitivity check.
- **Secret adapters:** LoRA rank 32, α 32 on all 7 projections of the client layers, secret random initialisation
  (scale 0.02, client-held seed), trained with KL(public‖protected) + λp·NegEnt(prompt inverter) +
  λa·[NegEnt(answer inverter) + NegEnt(server public-tail output)]; 1200 steps, batch 4×128 tokens, AdamW 2e-4.
- **Latency model:** assumed model = FLOPs / assumed speed + bytes / rate + one RTT per generated token; calibrated
  model (`step5_per_request/calibrated_model.py`) = fitted per-side prefill/decode model, per-message overhead,
  network RTT + bytes/rate per direction, idle-gap slowdown, adapter overhead; memory = `memory_gb()` + 0.40 GB.

## Known limitations

- Only (4,2) (3 seeds), (2,2) and (1,1) (2 seeds each) have more than one training seed; prompt-leakage spread up to
  ±6.3 points, so prompt differences under ~10 points between splits are not reliable.
- Quality in the controller is a teacher-forced proxy. Generated-answer leakage (task 5a) is 19–45 points higher than
  the teacher-forced proxy at λa ≥ 0.03 (exception: (8,2) λa 0.1), so the controller adds the measured gap.
- Under the current rule, private answer and private prompt+answer are infeasible at every memory budget; only open
  and private prompt (H = 4 secret head) are served.
- Known-pairs limitation (task 5b): a server that collects known prompt–state pairs from the same user largely undoes
  the prompt protection. Periodic change of the secret seed (re-keying) is not implemented.
- The learned answer-side inverter fails its consistency check (attacker B < attacker A on the answer side); it is not
  used for reported answer leakage. The template-token session attacker failed (collapses to the template
  vocabulary) and is reported as a failed variant.
- Network and server conditions come from synthetic traces and are emulated in the runtime (application-level
  network emulation; server load emulated by the server); they are not observed on a real network or shared server.
- The runtime testbed is one 8 GB machine: one configuration is in memory at a time, so switching costs reloads.
  The mobile real-runtime trace used A = 64, the server-load trace A = 128 (as in the modelled traces): do not pool.

## Repository map

| Path | Contents | Runs on |
|---|---|---|
| `step_attack/` | prompt inversion attack at every split point H (unprotected) | Colab GPU |
| `step_noise/` | Gaussian-noise defence sweep with an adaptive attacker | Colab GPU |
| `step_secret/` | secret LoRA head only (H = 4) | Colab GPU |
| `step_secret_ushape/` | secret head + tail at one split (training, attackers A and B, quality) | Colab GPU |
| `step_secret_multisplit/` | driver running `secret_ushape.py` over many (H, T, seed); resumable; `--collect` merges | Colab GPU |
| `step_task5_privacy/` | task 5: generated-answer leakage (5a), known-pairs attacker (5b), multi-turn sessions (C2) | Colab GPU |
| `runtime/` | real client–server split runtime (partial models, wire protocol, server/client, network emulation), measurement and fitting scripts | Mac (CPU + MPS) |
| `step5_per_request/` | controller (`per_request_controller.py`), calibrated model, adaptive evaluation, feasibility analysis | laptop |
| `baselines/`, `step1_*` … `step3c_*` | earlier analytical models (assumed speeds) | laptop |
| `legacy/` | superseded whole-workload controller | — |
| `results/` | raw outputs of every Colab run (json / csv / png) | — |
| `runtime/check_out/` | raw outputs of every runtime measurement (jsonl / csv / json / logs) | — |
| `docs/figures/` | paper figures and `make_figures.py` | laptop |

## Setup

Laptop (runtime, analysis, figures); the MEASURED runtime results used Python 3.12.13 on a MacBook Pro (Apple M2, 8 GB):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-runtime.txt
hf auth login                                        # account must have accepted the Llama 3.2 license
hf download meta-llama/Llama-3.2-1B-Instruct         # the runtime loads the model with local_files_only=True
```

Colab (training, attacks, task 5): Python 3.13.16, T4 GPU; versions in `requirements-colab.txt`. Every notebook
starts with `notebook_login()` (paste your own Hugging Face token; never save it in the notebook).
Every Colab script supports `--smoke` (tiny random Llama on CPU) to test the code without a GPU or model access.

## 1. Privacy experiments (Colab)

Open the notebook, select a T4 GPU, run the cells. Outputs are written to Google Drive; copy them into `results/`.

| Notebook | What it runs | Results in |
|---|---|---|
| `step_attack/run_attack_colab.ipynb` | inversion attack at every H, no protection | `results/attack/` |
| `step_noise/run_noise_colab.ipynb` | Gaussian-noise sweep | `results/noise/` |
| `step_secret/run_secret_colab.ipynb` | secret head, H = 4 | `results/secret_head_run*/` |
| `step_secret_ushape/run_secret_ushape_colab.ipynb` | secret head + tail, one split | `results/secret_ushape_run*/` |
| `step_secret_multisplit/run_multisplit_colab.ipynb` | all splits and seeds (batches A–D), then `--collect` | `results/multisplit/` |
| `step_task5_privacy/run_task5_colab.ipynb` | 5a `answer_generation.py`, 5b `known_pairs_attack.py` | `results/task5/` |
| `step_task5_privacy/run_task5d_colab.ipynb` | C2: `sessions.py` → `capture_sessions.py` → `session_attack.py` | `results/task5d/` |

`secret_multisplit.py --collect` writes `measured_multisplit.txt`, the table the controller reads
(a copy is in `step5_per_request/`). Task 5 needs the SAVED trained adapters (`adapters_*.pt`, see below), e.g.

```bash
python answer_generation.py  --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
python known_pairs_attack.py --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
python sessions.py --n 600
python capture_sessions.py   --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
python session_attack.py     --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
```

The C2 analysis plan was fixed before any session run: `docs/AUDIT_AND_ROADMAP.md`, "PRE-REGISTERED analysis plan for C2".

## 2. Split runtime (Mac)

Client and server are separate processes on one machine (client CPU fp16, server MPS fp16), connected by TCP on
127.0.0.1. Network conditions are emulated at application level (`runtime/netem.py`: serialisation at the given rate
+ one-way delay on every message); `runtime/dummynet.sh` is the OS-level alternative used for a cross-check.
Server load is emulated by the server (it waits compute × (1/share − 1)). Run from `runtime/`:

```bash
python test_protocol.py                      # wire-format round trip, no model
python check_partial.py                      # split vs full model: identical greedy tokens
python test_correctness.py                   # same check across processes
python measure_task3.py                      # task 3 timing: check_out/task3_steps.jsonl, task3_gens.csv, task3_memory.json
python fit_task3.py                          # per-side speed model: check_out/task3_fit.json
python measure_profiles.py 2 1               # 2- and 1-thread client profiles: check_out/profiles.json
python measure_reload.py                     # reload times: check_out/reload.json
python build_calibration.py                  # -> step5_per_request/calibration.json
```

Hardware of all MEASURED runtime results: MacBook Pro (Mac14,7), Apple M2 (4 performance + 4 efficiency cores),
8 GB unified memory, macOS 26.6.2; client on CPU (fp16, 4 threads), server on MPS (fp16). Each process loads only its
own layers. Memory is tight: before a run, reboot or close other memory-heavy apps and check that no runtime process
is left over:

```bash
pgrep -fl "run_trace_real.py|server.py" || echo "no leftover processes"
```

To stop a run: Ctrl-C, then repeat the check and kill any leftover PID.

Real-runtime traces (each run starts and stops its own `server.py --reconfigurable`; outputs are written at the end;
reusing a `--tag` overwrites the earlier output). Close other memory-heavy apps first (8 GB machine):

```bash
python -u run_trace_real.py --n 30 --repeats 3 --tag _protected                  # mobile trace, A = 64 (~15 min)
python check_server_load_model.py                                                # model check before the next run
caffeinate -i python -u run_trace_real.py --trace server_load --repeats 3 \
    --policies adaptive,fixed_per_class --tag _server_load                       # server-load trace, A = 128 (~3 h)
```

## 3. Controller and analysis (laptop)

Run from `step5_per_request/`:

```bash
python per_request_controller.py             # controller on measured_multisplit.txt (robust leakage check)
python adaptive_eval.py                      # policies vs oracle on stable / mobile / server-load traces
python feasibility_analysis.py               # feasible privacy classes and smallest relaxations
python decision_diversity.py                 # what changes the decisions
python compare_leak_check.py                 # robust vs mean leakage check (needs adaptive_eval.py --leak-check mean)
```

`calibrated_model.py` (latency / memory model, estimator, switching cost) is a library used by these scripts.

## 4. Figures

```bash
python docs/figures/make_figures.py          # docs/figures/fig1 … fig10 from the existing results only (no new runs)
```

## Paper numbers → scripts → results

Section numbers refer to `docs/PAPER_NUMBERS.md`.

| Paper item | Script | Results file(s) |
|---|---|---|
| §1 leakage and quality per split (λa = 0.03) | `step_secret_multisplit/secret_multisplit.py` (+ `--collect`); robust values `step5_per_request/per_request_controller.py` | `results/multisplit/multisplit_results.csv`, `measured_multisplit.txt`, `runs/*/results.json` |
| §2 task 5a generated vs teacher-forced answer leakage | `step_task5_privacy/answer_generation.py` | `results/task5/5a_*/answer_generation.{json,csv}` |
| §3 task 5b known-pairs attacker | `step_task5_privacy/known_pairs_attack.py` | `results/task5/5b_*/known_pairs.{json,csv}` |
| §4 C2 session-level leakage | `step_task5_privacy/sessions.py`, `capture_sessions.py`, `session_attack.py` | `results/task5d/*/session_attack.{json,csv}`, `results/task5d/sessions/sessions.json` |
| §5 fitted speeds and model errors | `runtime/measure_task3.py`, `fit_task3.py`, `measure_profiles.py`, `measure_reload.py`, `build_calibration.py` | `runtime/check_out/task3_*`, `profiles.json`, `reload.json`, `step5_per_request/calibration.json` |
| §6 feasibility counts and relaxations | `step5_per_request/feasibility_analysis.py` | `step5_per_request/feasibility_grid.csv`, `feasibility_classes.csv`, `feasibility_min_prompt_leak.csv`, `feasibility_relax.json` |
| §7 calibration impact | `step5_per_request/adaptive_eval.py` | `step5_per_request/task4_calibration_impact.csv` |
| §8 policy comparison (gap to the oracle) | `step5_per_request/adaptive_eval.py` | `step5_per_request/task4_summary.csv`, `task4_decisions.csv` |
| §9 real-runtime mobile trace (A = 64) | `runtime/run_trace_real.py --tag _protected` | `runtime/check_out/task4_real_trace_protected.{csv,json}`, `_console.log` |
| §9 real-runtime server-load trace (A = 128) | `runtime/run_trace_real.py --trace server_load --tag _server_load`; `check_server_load_model.py` | `runtime/check_out/task4_real_trace_server_load.{csv,json}`, `server_load_model_check.csv` |
| Fig. 1 calibrated vs assumed latency model | `docs/figures/make_figures.py` | `runtime/check_out/task3_gens.csv` |
| Fig. 2 decode-step phases | `make_figures.py` | `runtime/check_out/task3_steps.jsonl` |
| Fig. 3 idle-gap slowdown | `make_figures.py` | `runtime/check_out/task3_steps.jsonl` |
| Fig. 4 memory per side | `make_figures.py` | `runtime/check_out/task3_fit.json` |
| Fig. 5 runtime policies, mobile trace | `make_figures.py` | `runtime/check_out/task4_real_trace_protected.csv` |
| Fig. 6 calibration impact | `make_figures.py` | `step5_per_request/task4_calibration_impact.csv` |
| Fig. 7 leakage vs quality, all splits | `make_figures.py` | `results/multisplit/measured_multisplit.txt` |
| Fig. 8 feasibility | `make_figures.py` | `step5_per_request/feasibility_min_prompt_leak.csv` |
| Fig. 9 decision timeline | `make_figures.py` | `step5_per_request/task4_decisions.csv` |
| Fig. 10 runtime, server-load trace | `make_figures.py` | `runtime/check_out/task4_real_trace_server_load.csv` |

## Trained adapters and inverters

The trained secret adapters (`adapters_lp*_la*.pt`) and the cached attacker-A inverters (`invA_H*.pt`) are not in git
(published separately: https://huggingface.co/Ghorabaaa/Llama-3.2-1B-Instruct-USplit-secret-adapters ; as published they are
no longer secret and give no protection). Without them, task 5 requires retraining with
`step_secret_multisplit/` first.

## License

Code: MIT (`LICENSE`). Llama-3.2-1B-Instruct is NOT included; it is licensed by Meta under the Llama 3.2 Community
License ("Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights
Reserved."). Documents in `docs/references/` keep their own licenses.
