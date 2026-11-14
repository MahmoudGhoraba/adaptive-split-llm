# Runtime environment (set up 14 Nov 2026)

Machine (read from the system, not from the docs):
- MacBook Pro Mac14,7, **Apple M2** (8 cores: 4 performance + 4 efficiency), 8 GB unified memory.
  Earlier docs/HARDWARE_AND_TESTBED.md said "M2 Pro" (corrected); `sysctl machdep.cpu.brand_string` and
  `system_profiler` both report "Apple M2".
- macOS Tahoe 26.6.2 (build 25G83), arm64.
- `dnctl` (/usr/sbin/dnctl) and `pfctl` (/sbin/pfctl) present; both need sudo.

Python environment: `.venv/` in the repo root, created from Homebrew Python 3.12.13
(system `python3` is 3.14.5; 3.12 chosen for PyTorch wheel support).

```
python3.12 -m venv .venv
.venv/bin/pip install torch==2.15.0 transformers==5.23.0 psutil==7.2.2 numpy==2.5.4
```

Pinned versions (`pip freeze`, relevant packages):

| Package | Version |
|---|---|
| torch | 2.15.0 (MPS available and built: yes; fp16 and bf16 matmul on MPS checked) |
| transformers | 5.23.0 |
| safetensors | 0.8.0 |
| tokenizers | 0.23.3 |
| huggingface_hub | 2.4.0 |
| numpy | 2.5.4 |
| psutil | 7.2.2 |

Not installed in .venv: matplotlib, datasets (the Colab scripts need them; the runtime does not).

Model: meta-llama/Llama-3.2-1B-Instruct, local snapshot 9213176726f574b556790deb65791e0c5aa438b6
(one bf16 model.safetensors, 2.4 GB; config: 16 layers, hidden 2048, 32 heads, 8 KV heads, head_dim 64,
rope_type llama3 factor 32, rope_theta 500000, tied embeddings). The runtime loads it with
`local_files_only=True` and converts bf16 -> fp16 per tensor.

Memory measurement (runtime/memstat.py): process physical footprint via `proc_pid_rusage(RUSAGE_INFO_V4)`
(includes MPS buffers; RSS does not). Check run: a 0.5 GB fp16 MPS tensor raised the footprint by 1.08 GB
because the MPS allocator reserved a ~1.07 GB heap on the first allocation (a second 0.5 GB tensor added 0);
a 0.5 GB CPU tensor raised it by 0.50 GB. Footprint = real memory cost, mps_alloc = tensor bytes.
