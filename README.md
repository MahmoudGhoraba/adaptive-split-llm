# Adaptive Edge–Server LLM Inference with Privacy-Preserving U-Shaped Splitting

Bachelor thesis project (Mahmoud Ghoraba). Start with
**docs/AUDIT_AND_ROADMAP.md** (status + next tasks) and **docs/RESULTS_SUMMARY.md** (all numbers).

## Setup (MacBook)

```bash
cd adaptive-split-llm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login            # Llama-3.2-1B-Instruct is a gated model
```

## Run (analytical steps, laptop)

```bash
python baselines/baselines.py
python step1_split_latency/split_latency.py
python step3c_split_protection/split_protection_optimisation.py
python step5_per_request/per_request_controller.py      # reads step5_per_request/measured_multisplit.txt
```

## Run (GPU experiments, Google Colab T4)

Open the `run_*_colab.ipynb` in each `step_*` folder. Code test without a GPU: `python <script>.py --smoke`.
Multi-split adapters: `step_secret_multisplit/run_multisplit_colab.ipynb` (batches A–D, resumable).

