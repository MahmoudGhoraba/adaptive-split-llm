"""
TASK 5d (C2) - capture the layer-H states of whole chat SESSIONS, run turn by turn with the KV cache (what the
server receives during a real session), with and without saved secret adapters.

Each session is fed message by message (chunks end after every end-of-turn token) into one DynamicCache, as in a
running conversation; a hook records the output of decoder layer H for every position. Sanity check: for the first
session the KV-cached capture is compared with one full forward pass (max |difference| reported).
To keep files small, TEST sessions are stored completely and TRAIN sessions only at template-token positions
(the free known pairs of the template-token attacker).

Run:  python capture_sessions.py --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt [--smoke]
      -> <out>/captures.pt  {public, protected: {session id: states (fp16)}, meta}
"""
import argparse
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "step_secret_multisplit")):
    sys.path.insert(0, p)
import secret_ushape as su
import sessions as S


def chunk_ends(sess, tok, smoke):
    """Token index after every end-of-turn token (message boundaries)."""
    if smoke:                                  # "<eot>" is 3 word tokens: the boundary is after the last one
        w = tok.encode(S.SMOKE_TPL["eot"]); ids = sess["ids"]
        ends = [i + len(w) for i in range(len(ids) - len(w) + 1) if ids[i:i + len(w)] == w]
    else:
        eot = tok.tok.convert_tokens_to_ids("<|eot_id|>")
        ends = [i + 1 for i, t in enumerate(sess["ids"]) if t == eot]
    if not ends or ends[-1] != len(sess["ids"]):
        ends.append(len(sess["ids"]))
    return ends


@torch.no_grad()
def capture_session(model, dev, sess, L, tok, smoke, kv=True):
    """States after layer L for every position of the session [n_tokens, d] (float16, CPU)."""
    from transformers.cache_utils import DynamicCache
    cat = su.Catcher(model, L)
    ids = torch.tensor([sess["ids"]], device=dev)
    if not kv:
        model(input_ids=ids); out = cat.state[0]
    else:
        cache, parts, start = DynamicCache(config=model.config), [], 0
        for end in chunk_ends(sess, tok, smoke):
            pos = torch.arange(start, end, device=dev)[None]
            model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True, position_ids=pos)
            parts.append(cat.state[0]); start = end
        out = torch.cat(parts)
    cat.remove()
    return out.half().cpu()


def capture_all(model, dev, sessions, L, tok, smoke, label):
    """{session id: states} - test sessions complete, train sessions only at template positions."""
    out = {}
    for s in sessions:
        st = capture_session(model, dev, s, L, tok, smoke)
        if s["split"] == "train":
            keep = [i for i, r in enumerate(s["role"]) if r == "template"]
            st = (st[keep], keep)
        out[s["id"]] = st
    print(f"   captured {len(out)} sessions ({label}), layer {L}", flush=True)
    return out


def kv_check(model, dev, sess, L, tok, smoke):
    a = capture_session(model, dev, sess, L, tok, smoke, kv=True).float()
    b = capture_session(model, dev, sess, L, tok, smoke, kv=False).float()
    return float((a - b).abs().max()), float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).max())


def adapters_by_side(model, H, T):
    n, head, tail = len(model.model.layers), [], []
    for li, layer in enumerate(model.model.layers):
        for parent, nm in su.PROJ:
            m = getattr(getattr(layer, parent), nm)
            if isinstance(m, su.LoRALinear):
                (head if li < H else tail if li >= n - T else []).append(m)
    return head, tail


def smoke_adapters(model, dev, out_dir, H=1, T=1):
    n = len(model.model.layers)
    mods = su.add_adapters(model, range(0, H), 2000, dev) + su.add_adapters(model, range(n - T, n), 2001, dev)
    for m in mods:
        m.B.data += 0.05 * torch.randn_like(m.B)
    path = os.path.join(out_dir, "adapters_smoke.pt")
    su.save_adapters(model, path, dict(model_id="smoke", smoke=True, H=H, T=T, lam_p=0.1, lam_a=0.0, train_seed=0,
                                       lora_rank=su.CFG["lora_rank"], lora_alpha=su.CFG["lora_alpha"],
                                       secret_init=su.CFG["secret_init"]))
    su.remove_all_adapters(model)
    return path


def setup(smoke, n_sessions=None):
    """Model, tokenizer, device, rendered sessions (the same in every script)."""
    if smoke:
        su.CFG.update(lora_rank=8, lora_alpha=8, train_tokens=40_000, test_tokens=4_000, n_sensitive=40, seq_len=64,
                      positions_per_step=128, epochs=1, bootstrap=30)
    su.set_seed(0); rng_py = random.Random(0)
    train_txt, test_txt = su.load_general_text(smoke, rng_py)
    raw = S.make_sessions(n_sessions)
    model, tok, dev = su.load_model(smoke, train_txt + test_txt + S.session_texts(raw))
    sessions = S.build(tok, smoke, sessions=raw)
    return model, tok, dev, sessions, train_txt, test_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=str); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int); ap.add_argument("--out", type=str, default="task5d_capture")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok, dev, sessions, _, _ = setup(a.smoke, a.n)
    path = a.adapters or (smoke_adapters(model, dev, a.out) if a.smoke else None)
    if not path:
        sys.exit("give --adapters <adapters_*.pt>")
    meta = torch.load(path, map_location="cpu")["meta"]; H = meta["H"]
    d0 = kv_check(model, dev, sessions[0], H, tok, a.smoke)
    print(f"KV-cache vs full forward, session 0: max |diff| {d0[0]:.2e}, max rel {d0[1]:.2e}")
    pub = capture_all(model, dev, sessions, H, tok, a.smoke, "public")
    su.load_adapters(dict(enumerate(model.model.layers)), path, dev)
    prot = capture_all(model, dev, sessions, H, tok, a.smoke, "protected")
    torch.save(dict(public=pub, protected=prot, meta=meta, kv_check=d0), os.path.join(a.out, "captures.pt"))
    print(f"Saved to {a.out}/captures.pt")


if __name__ == "__main__":
    main()
