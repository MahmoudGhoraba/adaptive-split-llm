"""
TASK 5b - Known-pairs attacker against the secret adapters (SAVED trained adapters).

Threat: the server cannot see the secret adapters, but may know k (text, protected state) pairs, e.g. texts it
knows the client sent (chat-template tokens, public documents) together with the hidden states it received.
  attacker A        inverter trained on the PUBLIC model (as in secret_ushape.py; 400k WikiText tokens, 2 epochs)
  A + k pairs       A fine-tuned on k known pairs from the PROTECTED client, k = 10, 100, 1000
                    (one pair = one known text of pair_len tokens and its protected states; k = 1000 x 64 tokens =
                    64k tokens, 16% of A's training data); fixed budget of ft_steps steps, lr ft_lr
  scratch k pairs   a new inverter trained only on the k pairs (same budget), to separate "A's prior" from "pairs"
  attacker B        inverter trained on the protected model with the full 400k tokens (upper bound; holds the secret)
Measured on the prompt side (state after layer H, target = input token): content leakage on WikiText test text and
full recovery of planted private details (templated prompts), as in secret_ushape.py. Also on the answer side
(state after layer N-T, target = the client's next token) with the LEARNED answer inverter - reported with the known
caveat that this inverter failed its consistency check (B < A) in the earlier runs.
Known pairs come from WikiText train text NOT used to train A (disjoint slice).

Run (Colab GPU):  python known_pairs_attack.py --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
Code test:        python known_pairs_attack.py --smoke
Writes <out>/known_pairs.json, known_pairs.csv
"""
import argparse
import copy
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "step_secret_multisplit")):
    sys.path.insert(0, p)
import secret_ushape as su

CFG = dict(ks=[10, 100, 1000], pair_len=64, ft_steps=300, ft_lr=1e-4, ft_batch=16, sides=["prompt", "answer"],
           with_B=True, seed=0, out_dir="task5b_results")
SMOKE = dict(ks=[2, 8], pair_len=16, ft_steps=20, ft_batch=4, out_dir="task5b_results_smoke")


def fine_tune(model, dev, inv0, pair_ids, E, L, target, label):
    """Continue training an inverter (or a fresh one if inv0 is None) on the known pairs only."""
    d = E.shape[1]
    inv = copy.deepcopy(inv0) if inv0 is not None else su.InvHead(d, su.CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
    inv.train()
    opt = torch.optim.AdamW(inv.parameters(), lr=CFG["ft_lr"] if inv0 is not None else su.CFG["inv_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"                                   # same mixed precision as su.train_inverter
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cat = su.Catcher(model, L); n = pair_ids.shape[0]; g = torch.Generator().manual_seed(CFG["seed"])
    for step in range(CFG["ft_steps"]):
        idx = torch.randint(0, n, (min(CFG["ft_batch"], n),), generator=g)
        ids = pair_ids[idx].to(dev)
        with torch.no_grad():
            out_tok = model(input_ids=ids).logits.argmax(-1)
        B, S = ids.shape
        x = cat.state[:, 1:].reshape(B * (S - 1), -1).float()            # position 0 = BOS, not a known token
        y = (ids if target == "input" else out_tok)[:, 1:].reshape(-1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = inv(x, E)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    cat.remove()
    print(f"   [{label}] {CFG['ft_steps']} steps on {n} pairs, final loss {loss.item():.3f}", flush=True)
    return inv.eval()


def smoke_adapters(model, dev, out_dir):
    n = len(model.model.layers)
    mods = su.add_adapters(model, [0], 2000, dev) + su.add_adapters(model, [n - 1], 2001, dev)
    for m in mods:
        m.B.data += 0.05 * torch.randn_like(m.B)
    path = os.path.join(out_dir, "adapters_smoke.pt")
    su.save_adapters(model, path, dict(model_id="smoke", smoke=True, H=1, T=1, lam_p=0.1, lam_a=0.0, train_seed=0,
                                       lora_rank=su.CFG["lora_rank"], lora_alpha=su.CFG["lora_alpha"],
                                       secret_init=su.CFG["secret_init"]))
    su.remove_all_adapters(model)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=str); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", type=str)
    ap.add_argument("--ks", type=str, help='e.g. "10 100 1000"'); ap.add_argument("--no_B", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        CFG.update(SMOKE)
        su.CFG.update(lora_rank=8, lora_alpha=8, train_tokens=40_000, test_tokens=4_000, n_sensitive=40, seq_len=64,
                      positions_per_step=128, epochs=1, bootstrap=30)
    if a.out: CFG["out_dir"] = a.out
    if a.ks: CFG["ks"] = [int(k) for k in a.ks.split()]
    if a.no_B: CFG["with_B"] = False
    os.makedirs(CFG["out_dir"], exist_ok=True)
    su.set_seed(CFG["seed"]); rng_py = random.Random(CFG["seed"]); rng_np = np.random.default_rng(CFG["seed"])
    train_txt, test_txt = su.load_general_text(a.smoke, rng_py)
    sens_prompts = su.make_sensitive_prompts(su.CFG["n_sensitive"], rng_py)
    model, tok, dev = su.load_model(a.smoke, train_txt + test_txt + [t for t, _ in sens_prompts])
    n = len(model.model.layers)
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    train_ids = su.chunk(tok, train_txt, su.CFG["seq_len"], su.CFG["train_tokens"])
    # known pairs: WikiText train text AFTER the tokens used to train A (disjoint)
    used = su.CFG["train_tokens"]; rest, count = [], 0
    for t in train_txt:
        count += len(tok.encode(" " + t))
        if count > used + 2000:
            rest.append(t)
    pool = su.chunk(tok, rest, CFG["pair_len"], max(CFG["ks"]) * (CFG["pair_len"] - 1))
    if pool.shape[0] < max(CFG["ks"]):
        sys.exit(f"only {pool.shape[0]} known-pair texts available, need {max(CFG['ks'])}")
    test_ids = su.chunk(tok, test_txt, su.CFG["seq_len"], su.CFG["test_tokens"])
    sens_ids, sens_mask, sens_valid = su.encode_sensitive(tok, sens_prompts, su.CFG["seq_len"])
    tables = su.token_tables(tok, train_ids)
    tv = np.ones(test_ids.shape, dtype=bool); tv[:, 0] = False

    path = a.adapters or (smoke_adapters(model, dev, CFG["out_dir"]) if a.smoke else None)
    if not path:
        sys.exit("give --adapters <adapters_*.pt> (written by secret_ushape.py with save_adapters)")
    meta = torch.load(path, map_location="cpu")["meta"]
    H, T = meta["H"], meta["T"]; LT = n - T
    side_cfg = dict(prompt=(H, "input"), answer=(LT, "output"))

    def leak(inv, side):
        L, target = side_cfg[side]
        pg, lg = su.inverter_predict(model, dev, inv, test_ids, E, L, target)
        g = su.leak_metrics(pg, lg, tv, tables, rng_np)
        r = dict(content_leak=g["content_leak"][0], ci=g["content_leak"][1:])
        if side == "prompt":
            ps, ls = su.inverter_predict(model, dev, inv, sens_ids, E, L, target)
            s = su.leak_metrics(ps, ls, sens_valid.numpy(), tables, rng_np, sens=sens_mask.numpy())
            r["private_exact"] = s["sensitive_exact"]
        return r

    rows, t0 = [], time.time()
    print(f"adapters {os.path.basename(path)}: H={H}, T={T}, λp={meta.get('lam_p')}, λa={meta.get('lam_a')}", flush=True)
    invA = {}
    for side in CFG["sides"]:                                 # attacker A on the PUBLIC model (adapters not loaded yet)
        L, target = side_cfg[side]
        invA[side] = su.train_inverter(model, dev, train_ids, E, L, target, f"A {side}")
    su.load_adapters(dict(enumerate(model.model.layers)), path, dev)          # the protected client
    for side in CFG["sides"]:
        L, target = side_cfg[side]
        rows.append(dict(side=side, attacker="A (public model)", k=0, **leak(invA[side], side)))
        for k in CFG["ks"]:
            pairs = pool[:k]
            ftA = fine_tune(model, dev, invA[side], pairs, E, L, target, f"A + {k} pairs, {side}")
            rows.append(dict(side=side, attacker="A + known pairs", k=k, **leak(ftA, side)))
            scr = fine_tune(model, dev, None, pairs, E, L, target, f"scratch {k} pairs, {side}")
            rows.append(dict(side=side, attacker="scratch on pairs", k=k, **leak(scr, side)))
            del ftA, scr
        if CFG["with_B"]:
            invB = su.train_inverter(model, dev, train_ids, E, L, target, f"B {side}")
            rows.append(dict(side=side, attacker="B (protected model, full data)", k=None, **leak(invB, side)))
            del invB
        for r in rows[-(2 * len(CFG["ks"]) + 1 + CFG["with_B"]):]:
            print(f"   {r['side']:6s} {r['attacker']:32s} k={str(r['k']):5s} content leak {r['content_leak']:.3f} "
                  f"(CI {r['ci'][0]:.3f}-{r['ci'][1]:.3f})" + (f", private exact {r['private_exact']:.3f}" if "private_exact" in r else ""),
                  flush=True)
    res = dict(config=CFG, smoke=a.smoke, adapters=os.path.basename(path), meta=meta, rows=rows,
               minutes=(time.time() - t0) / 60,
               caveat="answer side uses the learned answer inverter, which failed its B >= A check in earlier runs")
    json.dump(res, open(os.path.join(CFG["out_dir"], "known_pairs.json"), "w"), indent=1, default=str)
    with open(os.path.join(CFG["out_dir"], "known_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["side", "attacker", "k", "content_leak", "ci", "private_exact"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Saved to {CFG['out_dir']}/ ({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
