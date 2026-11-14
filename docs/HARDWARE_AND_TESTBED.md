# Hardware and testbed decision (task 2–3)

## Question: MacBook (Apple M2) as client + Google Colab as server, simultaneously?

Technically possible only through a tunnel (Colab cannot accept incoming connections, and neither can a Mac
behind a home router). But Colab's FAQ lists, for free-tier runtimes, "file hosting, media serving, or other web
service offerings not related to interactive compute", "connecting to remote proxies" and "bypassing the notebook
UI" as disallowed; such runtimes may be terminated without warning. A tunnel that exposes a server process
from Colab falls under these. Paid users with a positive compute-unit balance have these restrictions relaxed —
check the current FAQ before relying on that.
Source: https://research.google.com/colaboratory/faq.html

A tunnel would also route traffic through the tunnel provider, so the measured network would be
"Mac → tunnel edge → Google data centre", not an edge-server link, and its conditions cannot be controlled.

## Recommended testbed

1. **Primary (controlled):** both processes on the MacBook (Apple M2).
   - Server process on the GPU (MPS backend), client process on CPU (or MPS for a "fast client" profile).
   - Network conditions emulated with macOS dummynet (`dnctl` + `pfctl`): set bandwidth and delay per scenario
     (Wi-Fi / 4G / poor coverage), so the adaptive controller can be tested under KNOWN, repeatable conditions.
   - Caveat to measure and report: client and server share the same unified memory and thermal budget, so their
     timings can interfere. Run each side alone as well, and compare.
2. **Real-network validation (optional):** server on a second machine on the same LAN, or a cloud GPU whose
   terms allow an exposed service. Use it to check the emulated results on at least one real link.
3. Colab stays the place for TRAINING/ATTACK experiments (unchanged), not for the live split.

## Known machine
- MacBook Pro (Mac14,7), Apple M2 (4P+4E cores; earlier docs said "M2 Pro", corrected 14 Nov after sysctl/system_profiler), 8 GB unified memory, macOS Tahoe 26.6.2.
- Memory budget (estimates, to be measured): full model fp16 ≈ 2.5 GB. Server part (≤ 12 middle layers, fp16)
  ≈ 1.5 GB; client part (H+T layers + embedding/output) ≈ 1.2 GB in fp16, ≈ 2.5 GB in fp32. Two processes plus
  macOS fit only if each loads its own layers; close other apps during measurements.
- Network emulation: first check `which dnctl pfctl` on Tahoe. If dummynet is unavailable or needs too much
  admin setup, emulate delay and bandwidth inside the runtime (sleep + rate limit around each send) — portable and
  deterministic, but label it as application-level emulation.

## Information still needed from the student
- PyTorch version with MPS.
- Whether a second machine (any laptop/PC with or without GPU) is available on the same network.
- Whether paid Colab compute units are available.
