# runtime/ — real client–server split runtime (PLANNED, not implemented)

Goal (task 2): run the client part (embedding, layers 1..H, layers N−T+1..N, output layer, optional secret
adapters) and the server part (layers H+1..N−T, public weights) in SEPARATE processes that exchange hidden
states, so time, bytes, RTT and memory can be measured (task 3).

## Design notes to discuss before coding

- Reuse model loading and the LoRA/adapter code from step_secret_multisplit/secret_ushape.py.
- Each side keeps its own KV cache for its own layers (client: head + tail layers; server: middle layers).
  Prefill sends P × d values up and back; each decode step sends 1 × d values up and back (one round trip per token).
- Rotary position ids must stay consistent on both sides (pass the position with each message).
- Wire format: fp16 tensor bytes + small header (step, position, shape). Transport: plain TCP socket first.
- Correctness test: greedy tokens of the split run must equal the full model's greedy tokens.
- Instrument every phase with time.perf_counter() (and torch.mps.synchronize() / torch.cuda.synchronize()
  before reading the clock), plus bytes sent/received and peak memory.

## Testbed options (see docs/HARDWARE_AND_TESTBED.md)
1. Default: both processes on the MacBook (client on CPU or MPS, server on MPS), with emulated network
   delay/bandwidth (macOS dummynet via dnctl/pfctl) — controlled, repeatable conditions.
2. LAN: client on the Mac, server on a second machine on the same network, if available.
3. Cloud GPU server only where the provider's terms allow an exposed service (not free Colab).
