"""
Partial Llama for the split runtime: each process loads ONLY its own layers from model.safetensors.

  client: embedding (tied, also the output layer), head layers 0..H-1, tail layers N-T..N-1, final norm
  server: middle layers H..N-T-1 (public weights, no adapter code path)

A side holds one or more SEGMENTS (contiguous ranges of global layers). Each segment has its own KV cache
whose layer indices are local (0..len-1), and every call states pos_start: the cache must already hold exactly
pos_start positions, which keeps both sides' positions in sync (rotary ids = pos_start .. pos_start + n - 1).
Both sides build LlamaRotaryEmbedding from the model's config.json (Llama 3.2: rope_type llama3, factor 32).

The model is never loaded twice in one process: weights go tensor by tensor from the safetensors file to the
device (meta-device modules + load_state_dict(assign=True)), so no random fp32 copy is ever allocated.
"""
import copy
import gc
import json
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "step_secret_multisplit"))
import secret_ushape as su          # LoRALinear, PROJ, add_adapters, load_adapters (same code as the Colab runs)

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def model_dir(model_id=MODEL_ID):
    """Local snapshot of the model (already downloaded; never downloads)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True)


def load_config(mdir):
    cfg = LlamaConfig.from_pretrained(mdir)
    cfg._attn_implementation = "sdpa"           # same kernel as the full-model reference
    return cfg


def split_segments(H, T, n_layers):
    """Global layer ranges of each side for a U split."""
    U = n_layers - H - T
    assert H >= 1 and T >= 0 and U >= 1, (H, T, n_layers)
    client = dict(head=(0, H)) | (dict(tail=(n_layers - T, n_layers)) if T else {})
    server = dict(mid=(H, n_layers - T))
    return client, server


class Segment(nn.Module):
    def __init__(self, cfg, start, stop):
        super().__init__()
        self.start, self.stop = start, stop
        self.cfg = copy.deepcopy(cfg); self.cfg.num_hidden_layers = stop - start      # cache sized to the segment
        self.layers = nn.ModuleList(LlamaDecoderLayer(self.cfg, i) for i in range(stop - start))

    def new_cache(self):
        return DynamicCache(config=self.cfg)

    def forward(self, h, pos_start, cache, rotary):
        have = cache.get_seq_length()
        assert have == pos_start, f"layers {self.start}..{self.stop - 1}: cache holds {have}, message says {pos_start}"
        pos = torch.arange(pos_start, pos_start + h.shape[1], device=h.device)[None]
        mask = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=None,
                                  past_key_values=cache, position_ids=pos)
        pe = rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_embeddings=pe, position_ids=pos,
                      past_key_values=cache, use_cache=True)
        return h


class PartialLlama(nn.Module):
    """segments: {name: (start, stop)} global layer ranges; ends=True also loads embedding + final norm.
    load: "cpu"    read bf16 to CPU, convert, copy to the device (temporary CPU copies per tensor)
          "direct" read straight to the device, convert there
    Both end with gc + empty_cache, so freed temporaries are returned before memory is measured.
          "cpu_keep" as "cpu" without the gc + empty_cache (first version; kept to measure the difference)"""

    def __init__(self, mdir, segments, ends, device, dtype=torch.float16, load="cpu"):
        super().__init__()
        self.cfg = load_config(mdir)
        self.device, self.dtype, self.ends = torch.device(device), dtype, ends
        with torch.device("meta"):
            self.segs = nn.ModuleDict({k: Segment(self.cfg, a, b) for k, (a, b) in segments.items()})
            if ends:
                self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
                self.norm = LlamaRMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.rotary = LlamaRotaryEmbedding(self.cfg).to(self.device)        # not meta: inv_freq is computed
        self._load(mdir, load)
        self.eval().requires_grad_(False)

    def _wanted(self):
        """{file key: (module, local key)} for every tensor this side needs."""
        want = {}
        for seg in self.segs.values():
            for i, gl in enumerate(range(seg.start, seg.stop)):
                for k in seg.layers[i].state_dict():
                    want[f"model.layers.{gl}.{k}"] = (seg.layers[i], k)
        if self.ends:
            want["model.embed_tokens.weight"] = (self.embed, "weight")
            want["model.norm.weight"] = (self.norm, "weight")
        return want

    def _load(self, mdir, load):
        want, per_mod = self._wanted(), {}
        src = str(self.device) if load == "direct" else "cpu"
        files = [f for f in os.listdir(mdir) if f.endswith(".safetensors")]
        for fn in files:
            with safe_open(os.path.join(mdir, fn), framework="pt", device=src) as f:
                for key in f.keys():
                    if key in want:
                        mod, k = want.pop(key)
                        # copy=True: a bf16 -> bf16 CPU load would otherwise stay a view of the mmapped file
                        # (file-backed pages, not in the process footprint, evictable under memory pressure)
                        per_mod.setdefault(mod, {})[k] = f.get_tensor(key).to(self.device, self.dtype, copy=True)
        assert not want, f"missing in checkpoint: {sorted(want)[:5]}"
        for mod, sd in per_mod.items():
            mod.load_state_dict(sd, assign=True, strict=True)
        del per_mod
        if load == "cpu_keep":
            return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.synchronize(); torch.mps.empty_cache()

    @property
    def global_layers(self):
        """{0-based global index: decoder layer}, for adapters."""
        return {seg.start + i: l for seg in self.segs.values() for i, l in enumerate(seg.layers)}

    def add_secret_adapters(self, layer_idx, seed):
        """Same function, same seed order as the Colab runs (su.add_adapters); fp32 A/B on this device."""
        shim = types.SimpleNamespace(model=types.SimpleNamespace(layers=self.global_layers))
        return su.add_adapters(shim, layer_idx, seed, self.device)

    def load_secret_adapters(self, path):
        return su.load_adapters(self.global_layers, path, self.device)

    def new_caches(self):
        return {k: s.new_cache() for k, s in self.segs.items()}

    @torch.no_grad()
    def embed_ids(self, ids):
        return self.embed(ids.to(self.device))

    @torch.no_grad()
    def run(self, name, h, pos_start, caches):
        return self.segs[name](h.to(self.device, self.dtype), pos_start, caches[name], self.rotary)

    @torch.no_grad()
    def logits(self, h):
        return F.linear(self.norm(h.to(self.device, self.dtype)), self.embed.weight)


@torch.no_grad()
def generate_local(m, ids, A, device):
    """Greedy A tokens on a PartialLlama holding all layers ("all" segment) + embedding/output (one process)."""
    import time
    caches, x, pos, toks, steps, ns = m.new_caches(), ids, 0, [], [], time.perf_counter_ns
    for step in range(A):
        t0 = ns(); h = m.run("all", m.embed_ids(x), pos, caches); sync(device); t1 = ns()
        nxt = int(m.logits(h[:, -1:])[0, -1].argmax()); t2 = ns()
        toks.append(nxt)
        steps.append(dict(step=step, kind="prefill" if step == 0 else "decode", n_tokens=x.shape[1], pos_start=pos,
                          layers_s=(t1 - t0) / 1e9, out_s=(t2 - t1) / 1e9, step_s=(t2 - t0) / 1e9))
        pos += x.shape[1]; x = torch.tensor([[nxt]])
    return dict(tokens=toks, steps=steps)


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def write_smoke_model(out_dir):
    """Tiny random Llama (llama3 rope scaling, tied embeddings) saved like the real snapshot, for --smoke."""
    from transformers import LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=1024, rope_theta=500000.0,
                      rope_scaling=dict(rope_type="llama3", factor=32.0, low_freq_factor=1.0,
                                        high_freq_factor=4.0, original_max_position_embeddings=64))
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(out_dir)
    return out_dir
