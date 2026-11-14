"""
Timing protocol for every timed split run (from 15 Nov 2026; task 3 and later):

  1. Prompt lengths come from fixed BUCKETS; every bucket used in a run is warmed up first (MPS compiles kernels per
     tensor shape: the first prefill at a new length was 4-18x slower in the first loopback run), and the adapter
     path is warmed up separately when adapters are used.
  2. Fixed answer length A: EOS is ignored (client.generate(..., eos=())), so runs are comparable.
  3. Free memory and swap are logged before and after every timed request; rows where swap grew are flagged
     (swap_grew = True) and must be reported, not silently averaged.
"""
import torch

import memstat

BUCKETS = (32, 64, 128, 256, 512, 1024)


def synthetic_ids(P, seed=0, vocab=128000):
    """P token ids (content does not change the compute; real prompts are used for correctness runs)."""
    return torch.randint(0, vocab, (1, P), generator=torch.Generator().manual_seed(seed))


def warmup(client, P_values, adapter_seeds=None, A=4):
    """One short generation at every prompt length, public and (if adapter_seeds) with the adapters on.
    Leaves the client without adapters."""
    for P in sorted(set(P_values)):
        client.generate(synthetic_ids(P, seed=10_000 + P), A)
    if adapter_seeds is not None:
        client.add_secret_adapters(*adapter_seeds)
        for P in sorted(set(P_values)):
            client.generate(synthetic_ids(P, seed=20_000 + P), A)
        client.remove_adapters()


def timed_generate(client, ids, A, **kw):
    """Fixed-length generation with system memory/swap logged around it."""
    s0 = memstat.snapshot()
    g = client.generate(ids, A, eos=(), **kw)
    s1 = memstat.snapshot()
    g["mem"] = dict(sys_available_before=s0["sys_available"], sys_available_after=s1["sys_available"],
                    swap_before=s0["swap_used"], swap_after=s1["swap_used"], client_footprint=s1["footprint"],
                    client_rss=s1["rss"], swap_grew=s1["swap_used"] > s0["swap_used"] + 10**7)
    return g
