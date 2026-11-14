"""
TASK 5d (C2) - does a server that observes whole SESSIONS recover more private details than per-turn analysis?

Sessions: sessions.py (templated, 5-10 turns, details spread over turns, some repeated). States after layer H for
every token, captured turn by turn with the KV cache (capture_sessions.py), PUBLIC and PROTECTED (saved adapters).
Attackers (all decode tokens from the layer-H state with the existing InvHead, scored against tied embeddings):
  per_turn        attacker A (trained on the public model, WikiText, same budget as before) applied to each token;
                  the per-turn BASELINE uses only the FIRST mention of every detail (what one turn reveals)
  union           session attacker: a detail counts as recovered if ANY of its mentions is fully recovered
  vote            session attacker: per detail, log-probabilities summed over the aligned tokens of all mentions
                  with the same token sequence, then argmax (no extra training)
                  union and vote are given the mention alignment (which tokens are the same detail) = UPPER BOUNDS
                  on cross-turn aggregation; a learned model over earlier-turn states is NOT implemented
  realistic       session attacker WITHOUT alignment: decodes every user-content token of every turn with the
                  attacker (the server knows message boundaries from the per-message prefill chunks and the fixed
                  template positions inside them), extracts candidate details from the decoded text with PUBLIC
                  vocabularies/patterns (the generator's name, city and condition lists; phone 01 + 9 digits;
                  "<number> pounds"), and takes the most frequent candidate per detail type across turns (ties: earliest)
  realistic per-turn  the SAME decode + extraction pipeline on one turn at a time, no cross-turn vote (per turn: the
                  most frequent candidate of that turn, ties earliest):
                    turn_first  recovered if the guess from the turn of the detail's FIRST mention is right (uses which
                                turn holds the first mention)
                    turn_any    recovered if ANY single turn's own guess is right, turns evaluated independently
                                (optimistic: the attacker cannot tell which turn's guess is right)
                  MAIN C2 RESULT: realistic (session vote) - realistic per-turn (turn_first), paired bootstrap CI,
                  for all details, repeated details (>= 2 mentions) and details mentioned once (PRE-REGISTERED,
                  docs/AUDIT_AND_ROADMAP.md)
  realistic_weighted  the ONE pre-registered secondary variant: same extraction, each candidate's vote weighted by the
                  attacker's mean top-1 token probability over the candidate's token span; same baseline and bootstrap
  template        template-token attacker: A fine-tuned on (template token, protected state) pairs from the TRAIN
                  sessions (chat-template tokens repeat every turn and are public = free known pairs);
                  ft_steps steps (same routine and budget as 5b); applied first-mention / union / vote
  B               (optional) attacker trained on the protected model with the full data (holds the secret)
Attacker A is cached per split (invA_H<H>.pt next to the output folder): it depends only on the public model and H,
so the runs at different λa reuse it (same attacker, less GPU time).
Metrics on TEST sessions (95% bootstrap CI over sessions): full recovery per detail type; recovery of a mention
by turn index and by absolute position (0-127 / 128-255 / 256+; the adapters were trained on 128-token sequences),
also for an UNPROTECTED control (attacker A on public states at the same split), so the length effect can be
separated from the protection effect;
name + condition linkage per session (both recovered); every method minus the per-turn baseline (paired bootstrap).

Run (Colab GPU):  python session_attack.py --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt
Code test:        python session_attack.py --smoke
Writes <out>/session_attack.json, session_attack.csv
"""
import argparse
import collections
import copy
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "step_secret_multisplit")):
    sys.path.insert(0, p)
import secret_ushape as su
import sessions as S
import capture_sessions as C

CFG = dict(n_sessions=600, ft_steps=300, ft_lr=1e-4, ft_batch=512, with_B=True, bootstrap=1000, out_dir="task5d_results")
SMOKE = dict(n_sessions=60, ft_steps=20, ft_batch=64, bootstrap=100, out_dir="task5d_results_smoke")
POS_BUCKETS = [(0, 128), (128, 256), (256, 10**9)]


_alt = lambda xs: "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))
EXTRACT = dict(
    name=re.compile(r"\b(?:" + _alt(su.FIRST) + r")\s+(?:" + _alt(su.LAST) + r")\b"),
    city=re.compile(r"\b(?:" + _alt(su.CITIES) + r")\b"),
    condition=re.compile(r"\b(?:" + _alt(su.DISEASES) + r")\b"),
    phone=re.compile(r"01\d{9}"),
    amount=re.compile(r"\b(\d+)\s*pounds"),
)


def extract(text, with_spans=False):
    """Candidate details in a decoded turn (public vocabularies and patterns, no knowledge of the session);
    with_spans: (value, char start, char end) of the matched text."""
    out = {}
    for d, rx in EXTRACT.items():
        g = 1 if d == "amount" else 0
        out[d] = [(m.group(g), m.start(g), m.end(g)) if with_spans else m.group(g) for m in rx.finditer(text)]
    return out


def decode_text(tok, ids, smoke):
    return "".join(tok.decode_one(i) for i in ids) if smoke else tok.tok.decode(list(ids), skip_special_tokens=True)


def decode_with_spans(tok, ids, smoke):
    """Decoded text and the character span of every token in it (prefix decoding for the BPE tokenizer)."""
    if smoke:
        pieces = [tok.decode_one(i) for i in ids]
    else:
        pieces, prev = [], ""
        for k in range(1, len(ids) + 1):
            cur = tok.tok.decode(list(ids[:k]), skip_special_tokens=True); pieces.append(cur[len(prev):]); prev = cur
    spans, pos = [], 0
    for p_ in pieces:
        spans.append((pos, pos + len(p_))); pos += len(p_)
    return "".join(pieces), spans


def _most_frequent(cands):
    """Most frequent candidate, ties broken by the earliest occurrence; None if there is no candidate."""
    if not cands:
        return None
    cnt = collections.Counter(cands)
    return max(cnt, key=lambda c: (cnt[c], -cands.index(c)))


def _weighted_best(cands):
    """cands: list of (value, weight) in occurrence order -> value with the largest total weight (ties: earliest)."""
    if not cands:
        return None
    tot, first = collections.defaultdict(float), {}
    for k, (c, w) in enumerate(cands):
        tot[c] += w; first.setdefault(c, k)
    return max(tot, key=lambda c: (tot[c], -first[c]))


def realistic_guess(s, pred, tok, smoke, conf=None):
    """Session guess (plain frequency vote over the candidates of every decoded user turn, no mention alignment),
    confidence-weighted session guess (each candidate weighted by the mean top-1 probability over its tokens; conf =
    top-1 probability per position) and the per-turn guesses (each turn alone):
    ({type: guess}, {type: weighted guess}, {turn: {type: guess}})."""
    allc = {d: [] for d in S.DETAILS}; allw = {d: [] for d in S.DETAILS}; per_turn = {}
    for t in range(s["n_turns"]):
        pos = [i for i, (r, tt) in enumerate(zip(s["role"], s["turn"])) if r == "user" and tt == t]
        text, spans = decode_with_spans(tok, pred[pos], smoke)
        ex = extract(text, with_spans=True)
        per_turn[t] = {d: _most_frequent([c for c, _, _ in v]) for d, v in ex.items()}
        for d, v in ex.items():
            for c, a, b in v:                              # in turn order: ties -> earliest
                allc[d].append(c)
                ks = [pos[k] for k, (x, y) in enumerate(spans) if x < b and y > a]
                w = float(np.mean([conf[k] for k in ks])) if (conf is not None and ks) else 1.0
                allw[d].append((c, w))
    return ({d: _most_frequent(c) for d, c in allc.items()}, {d: _weighted_best(c) for d, c in allw.items()}, per_turn)


@torch.no_grad()
def decode(inv, states, E, dev, need_logp=None):
    """Argmax token and its probability for every position; log-probabilities at the positions in need_logp."""
    preds, conf, logp = [], [], {}
    for b in range(0, states.shape[0], 1024):
        x = states[b:b + 1024].to(dev).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = inv(x, E).float()
        preds.append(lg.argmax(-1).cpu()); conf.append(torch.softmax(lg, -1).max(-1).values.cpu())
        for p in (need_logp or []):
            if b <= p < b + 1024:
                logp[p] = F.log_softmax(lg[p - b], -1).cpu()
    return torch.cat(preds).numpy(), torch.cat(conf).numpy(), logp


def fine_tune_states(inv0, X, y, E, dev):
    """Continue training inverter inv0 on fixed (state, token) pairs (template positions)."""
    inv = copy.deepcopy(inv0).train()
    opt = torch.optim.AdamW(inv.parameters(), lr=CFG["ft_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(0); n = X.shape[0]
    for _ in range(CFG["ft_steps"]):
        idx = torch.randint(0, n, (min(CFG["ft_batch"], n),), generator=g)
        x, t = X[idx].to(dev).float(), y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lg = inv(x, E)
        loss = F.cross_entropy(lg.float(), t)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print(f"   [template attacker] {CFG['ft_steps']} steps on {n} template pairs ({len(set(y.tolist()))} distinct tokens), "
          f"final loss {loss.item():.3f}", flush=True)
    return inv.eval()


def session_results(inv, caps, sessions, E, dev, tok=None, smoke=False):
    """Per test session: per detail, first-mention / union / vote recovery; per mention: exact, turn, position."""
    out, mentions = [], []
    for s in sessions:
        if s["split"] != "test":
            continue
        st = caps[s["id"]]
        need = sorted({p for m in s["mentions"] for p in m["positions"]})
        pred, conf, logp = decode(inv, st, E, dev, need)
        ids = np.array(s["ids"])
        guess, wguess, turn_guess = realistic_guess(s, pred, tok, smoke, conf)
        det = {}
        by_type = collections.defaultdict(list)
        for m in s["mentions"]:
            ok = bool((pred[m["positions"]] == ids[m["positions"]]).all())
            by_type[m["type"]].append((m, ok))
            mentions.append(dict(session=s["id"], type=m["type"], turn=m["turn"], pos=m["positions"][0], exact=ok,
                                 n_mentions=len([x for x in s["mentions"] if x["type"] == m["type"]])))
        for d, lst in by_type.items():
            first = lst[0][1]; union = any(ok for _, ok in lst)
            groups = collections.defaultdict(list)                  # mentions with the same token sequence
            for m, _ in lst:
                groups[tuple(ids[m["positions"]])].append(m)
            vote = False
            for toks, ms in groups.items():
                L = len(toks)
                summed = [sum(logp[m["positions"][j]] for m in ms) for j in range(L)]
                vote |= all(int(summed[j].argmax()) == toks[j] for j in range(L))
            true = s["persona"][d]
            det[d] = dict(first=first, union=union, vote=vote, realistic=guess[d] == true,
                          realistic_weighted=wguess[d] == true,
                          realistic_turn_first=turn_guess[lst[0][0]["turn"]][d] == true,
                          realistic_turn_any=any(g[d] == true for g in turn_guess.values()), n_mentions=len(lst))
        out.append(dict(session=s["id"], details=det))
    return out, mentions


def boot(vals, n, rng):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarise(res, mentions, label, base=None, rng=None):
    """Recovery rates; base = the per-turn first-mention values of the same sessions for paired differences."""
    rows = []
    def sel(d, k, v):                              # detail-type filter, or the pre-registered groups
        return d == "all" or d == k or (d == "repeated" and v["n_mentions"] > 1) or (d == "once" and v["n_mentions"] == 1)
    for method in ("first", "union", "vote", "realistic", "realistic_weighted", "realistic_turn_first", "realistic_turn_any"):
        for d in S.DETAILS + ["all", "repeated", "once"]:
            vals = [r["details"][k][method] for r in res for k in r["details"] if sel(d, k, r["details"][k])]
            row = dict(attacker=label, method=method, detail=d, n=len(vals))
            row["recovery"], row["ci_lo"], row["ci_hi"] = boot(vals, CFG["bootstrap"], rng)
            if base is not None:
                b = [r["details"][k]["first"] for r in base for k in r["details"] if sel(d, k, r["details"][k])]
                row["minus_per_turn"], row["diff_lo"], row["diff_hi"] = boot(np.array(vals, float) - np.array(b, float),
                                                                             CFG["bootstrap"], rng)
            # same sessions, same attacker: minus its OWN realistic per-turn baseline (main C2 result for "realistic")
            rb = [r["details"][k]["realistic_turn_first"] for r in res for k in r["details"] if sel(d, k, r["details"][k])]
            row["minus_realistic_per_turn"], row["rdiff_lo"], row["rdiff_hi"] = boot(
                np.array(vals, float) - np.array(rb, float), CFG["bootstrap"], rng)
            rows.append(row)
        link = [r["details"]["name"][method] and r["details"]["condition"][method] for r in res
                if "name" in r["details"] and "condition" in r["details"]]
        row = dict(attacker=label, method=method, detail="name+condition linkage", n=len(link))
        row["recovery"], row["ci_lo"], row["ci_hi"] = boot(link, CFG["bootstrap"], rng)
        rlink = [r["details"]["name"]["realistic_turn_first"] and r["details"]["condition"]["realistic_turn_first"]
                 for r in res if "name" in r["details"] and "condition" in r["details"]]
        row["minus_realistic_per_turn"], row["rdiff_lo"], row["rdiff_hi"] = boot(
            np.array(link, float) - np.array(rlink, float), CFG["bootstrap"], rng)
        if base is not None:
            b = [r["details"]["name"]["first"] and r["details"]["condition"]["first"] for r in base
                 if "name" in r["details"] and "condition" in r["details"]]
            row["minus_per_turn"], row["diff_lo"], row["diff_hi"] = boot(np.array(link, float) - np.array(b, float),
                                                                         CFG["bootstrap"], rng)
        rows.append(row)
    by_turn = collections.defaultdict(list); by_pos = collections.defaultdict(list)
    for m in mentions:
        by_turn[m["turn"] + 1].append(m["exact"])
        for a, b in POS_BUCKETS:
            if a <= m["pos"] < b:
                by_pos[f"{a}-{b - 1 if b < 10**9 else 'end'}"].append(m["exact"])
    rows += [dict(attacker=label, method="mention", detail=f"turn {t}", n=len(v), **dict(zip(("recovery", "ci_lo", "ci_hi"),
                                                                                       boot(v, CFG["bootstrap"], rng))))
             for t, v in sorted(by_turn.items())]
    rows += [dict(attacker=label, method="mention", detail=f"position {k}", n=len(v), **dict(zip(("recovery", "ci_lo", "ci_hi"),
                                                                                           boot(v, CFG["bootstrap"], rng))))
             for k, v in by_pos.items()]
    return rows


class LookupInv(torch.nn.Module):
    """Self-test attacker: nearest token embedding (state = embedding + noise, so its accuracy is set by the noise)."""
    def forward(self, h, E):
        return 10.0 * (h @ E.t())


def selftest(sessions, E, dev, tok, smoke):
    """Code test of the session metrics with an attacker of known quality: union >= first mention for every detail,
    vote and linkage computed, and session aggregation helps when mentions repeat."""
    test = [s for s in sessions if s["split"] == "test"]
    g = torch.Generator().manual_seed(0); Ef = E.float().cpu()
    # zero noise: the attacker decodes every token correctly -> the realistic extraction must find every true detail
    caps0 = {s["id"]: Ef[torch.tensor(s["ids"])] for s in test}
    res0, _ = session_results(LookupInv(), caps0, sessions, E, dev, tok, smoke)
    r0 = np.mean([v["realistic"] for r in res0 for v in r["details"].values()])
    bad = [(r["session"], d) for r in res0 for d, v in r["details"].items() if not v["realistic"]][:5]
    assert r0 == 1.0, f"realistic extraction misses true details with a perfect decoder: {r0:.3f}, e.g. {bad}"
    w0 = np.mean([v["realistic_weighted"] for r in res0 for v in r["details"].values()])
    assert w0 == 1.0, f"confidence-weighted vote misses true details with a perfect decoder: {w0:.3f}"
    # weighting rule: one confident candidate beats two unconfident ones; the plain vote does the opposite
    cands = [("A", 0.9), ("B", 0.3), ("B", 0.3)]
    assert _weighted_best(cands) == "A" and _most_frequent([c for c, _ in cands]) == "B" and _weighted_best([]) is None
    rt0 = np.mean([v["realistic_turn_first"] for r in res0 for v in r["details"].values()])
    assert rt0 == 1.0, f"realistic per-turn (first-mention turn) misses true details with a perfect decoder: {rt0:.3f}"
    for sigma in (0.5, 1.0, 2.0, 4.0):
        scale = sigma * Ef.norm(dim=-1).mean() / Ef.shape[1] ** 0.5
        caps = {s["id"]: Ef[torch.tensor(s["ids"])] + scale * torch.randn(len(s["ids"]), Ef.shape[1], generator=g) for s in test}
        res, men = session_results(LookupInv(), caps, sessions, E, dev, tok, smoke)
        acc = np.mean([m["exact"] for m in men])
        if 0.2 < acc < 0.8:
            break
    first = np.mean([r["details"][d]["first"] for r in res for d in r["details"]])
    union = np.mean([r["details"][d]["union"] for r in res for d in r["details"]])
    vote = np.mean([r["details"][d]["vote"] for r in res for d in r["details"]])
    assert all(v["union"] >= v["first"] for r in res for v in r["details"].values()), "union < first mention"
    rep = [v for r in res for v in r["details"].values() if v["n_mentions"] > 1]
    assert rep and np.mean([v["union"] for v in rep]) > np.mean([v["first"] for v in rep]), "no gain from repeats"
    real = np.mean([r["details"][d]["realistic"] for r in res for d in r["details"]])
    rtf = np.mean([r["details"][d]["realistic_turn_first"] for r in res for d in r["details"]])
    rta = np.mean([r["details"][d]["realistic_turn_any"] for r in res for d in r["details"]])
    assert 0.0 < real <= 1.0
    assert all(v["realistic_turn_any"] >= v["realistic_turn_first"] for r in res for v in r["details"].values()), \
        "any single turn < first-mention turn"
    # on details mentioned once, the session vote sees the same single turn's candidates (plus decoding noise from
    # other turns), so it cannot beat the per-turn first-mention guess by much; on repeated details it should gain
    rws = summarise(res, men, "self-test", rng=np.random.default_rng(0))
    main = next(r for r in rws if r["method"] == "realistic" and r["detail"] == "all")
    sec = next(r for r in rws if r["method"] == "realistic_weighted" and r["detail"] == "all")
    wv = np.mean([r["details"][d]["realistic_weighted"] for r in res for d in r["details"]])
    assert abs(sec["minus_realistic_per_turn"] - (wv - rtf)) < 1e-9
    for g in ("repeated", "once"):
        rg = next(r for r in rws if r["method"] == "realistic" and r["detail"] == g)
        vals = [v for r in res for v in r["details"].values() if (v["n_mentions"] > 1) == (g == "repeated")]
        assert rg["n"] == len(vals) and abs(rg["minus_realistic_per_turn"] - (np.mean([v["realistic"] for v in vals])
                                                                              - np.mean([v["realistic_turn_first"] for v in vals]))) < 1e-9
    assert abs(main["minus_realistic_per_turn"] - (real - rtf)) < 1e-9 and main["rdiff_lo"] <= main["minus_realistic_per_turn"] <= main["rdiff_hi"], main
    once = [v for r in res for v in r["details"].values() if v["n_mentions"] == 1]
    rep2 = [v for r in res for v in r["details"].values() if v["n_mentions"] > 1]
    print(f"self-test: perfect decoder -> realistic {r0:.3f}, realistic per-turn {rt0:.3f}; noise {sigma}x (mention accuracy {acc:.2f}): first {first:.3f} "
          f"<= union {union:.3f}; vote {vote:.3f}; realistic session vote {real:.3f} vs realistic per-turn first-mention "
          f"turn {rtf:.3f} (any single turn {rta:.3f}); on repeated details {np.mean([v['realistic'] for v in rep2]):.3f} vs "
          f"{np.mean([v['realistic_turn_first'] for v in rep2]):.3f}, mentioned once {np.mean([v['realistic'] for v in once]):.3f} vs "
          f"{np.mean([v['realistic_turn_first'] for v in once]):.3f}; main C2 diff {main['minus_realistic_per_turn']:+.3f} "
          f"[{main['rdiff_lo']:+.3f}, {main['rdiff_hi']:+.3f}]; weighted vote {wv:.3f} (perfect decoder {w0:.3f}), diff "
          f"{sec['minus_realistic_per_turn']:+.3f} [{sec['rdiff_lo']:+.3f}, {sec['rdiff_hi']:+.3f}]; "
          f"repeated details first {np.mean([v['first'] for v in rep]):.3f} -> union {np.mean([v['union'] for v in rep]):.3f}: PASS",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=str); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", type=str)
    ap.add_argument("--n", type=int); ap.add_argument("--no_B", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        CFG.update(SMOKE)
    if a.out: CFG["out_dir"] = a.out
    if a.n: CFG["n_sessions"] = a.n
    if a.no_B: CFG["with_B"] = False
    os.makedirs(CFG["out_dir"], exist_ok=True)
    t0 = time.time(); rng = np.random.default_rng(0)
    model, tok, dev, sessions, train_txt, test_txt = C.setup(a.smoke, CFG["n_sessions"])
    print(json.dumps(S.stats(sessions)), flush=True)
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    train_ids = su.chunk(tok, train_txt, su.CFG["seq_len"], su.CFG["train_tokens"])
    path = a.adapters or (C.smoke_adapters(model, dev, CFG["out_dir"]) if a.smoke else None)
    if not path:
        sys.exit("give --adapters <adapters_*.pt>")
    if a.smoke:
        selftest(sessions, E, dev, tok, a.smoke)
    meta = torch.load(path, map_location="cpu")["meta"]; H = meta["H"]
    print(f"adapters {os.path.basename(path)}: H={H}, T={meta['T']}, λp={meta.get('lam_p')}, λa={meta.get('lam_a')}", flush=True)
    kv = C.kv_check(model, dev, sessions[0], H, tok, a.smoke)
    print(f"KV-cache vs full forward (session 0): max |diff| {kv[0]:.2e}, max rel {kv[1]:.2e}", flush=True)

    cache = os.path.join(os.path.dirname(os.path.abspath(CFG["out_dir"])), f"invA_H{H}{'_smoke' if a.smoke else ''}.pt")
    d = E.shape[1]
    if os.path.exists(cache):                     # A depends only on the public model and H: reuse across λ
        invA = su.InvHead(d, su.CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
        invA.load_state_dict(torch.load(cache, map_location=dev)); invA.eval()
        print(f"attacker A reused from {cache}", flush=True)
    else:
        invA = su.train_inverter(model, dev, train_ids, E, H, "input", "A prompt")       # public model
        torch.save(invA.state_dict(), cache)
    pub = C.capture_all(model, dev, sessions, H, tok, a.smoke, "public")
    su.load_adapters(dict(enumerate(model.model.layers)), path, dev)
    prot = C.capture_all(model, dev, sessions, H, tok, a.smoke, "protected")

    rows = []
    res_pub, men_pub = session_results(invA, pub, sessions, E, dev, tok, a.smoke)
    rows += summarise(res_pub, men_pub, "A on PUBLIC states (no defence)", rng=rng)
    res_A, men_A = session_results(invA, prot, sessions, E, dev, tok, a.smoke)
    rows += summarise(res_A, men_A, "A (per-turn / session)", base=res_A, rng=rng)
    # template-token attacker: known pairs at template positions of the TRAIN sessions (protected states)
    X, y = [], []
    for s in sessions:
        if s["split"] == "train":
            st, keep = prot[s["id"]]
            X.append(st); y.append(torch.tensor([s["ids"][i] for i in keep]))
    invT = fine_tune_states(invA, torch.cat(X), torch.cat(y), E, dev)
    res_T, men_T = session_results(invT, prot, sessions, E, dev, tok, a.smoke)
    rows += summarise(res_T, men_T, "template-token attacker", base=res_A, rng=rng)
    if CFG["with_B"]:
        invB = su.train_inverter(model, dev, train_ids, E, H, "input", "B prompt")
        res_B, men_B = session_results(invB, prot, sessions, E, dev, tok, a.smoke)
        rows += summarise(res_B, men_B, "B (protected model, full data)", base=res_A, rng=rng)

    # length vs protection: the same mentions, unprotected (public states) vs protected, by position bucket
    for (a_, b_) in POS_BUCKETS:
        key = f"{a_}-{b_ - 1 if b_ < 10**9 else 'end'}"
        pairs = [(p["exact"], q["exact"]) for p, q in zip(men_pub, men_A) if a_ <= p["pos"] < b_]
        if not pairs:
            continue
        pv, qv = np.array(pairs, dtype=float).T
        row = dict(attacker="protection effect (A: protected - public)", method="mention", detail=f"position {key}", n=len(pairs))
        row["recovery"], row["ci_lo"], row["ci_hi"] = boot(qv - pv, CFG["bootstrap"], rng)
        row["public"], row["protected"] = float(pv.mean()), float(qv.mean())
        rows.append(row)
    print("MAIN C2 RESULT (pre-registered): realistic session vote - realistic per-turn (first-mention turn), paired "
          "bootstrap; all / repeated / once / linkage; plain vote = main, weighted = the one secondary variant:")
    for r in rows:
        if r["method"] in ("realistic", "realistic_weighted") and r["detail"] in ("all", "repeated", "once", "name+condition linkage"):
            print(f"  {r['attacker']:34s} {'weighted (secondary)' if r['method'] == 'realistic_weighted' else 'plain vote (main)':21s} "
                  f"{r['detail']:24s} n={r['n']:4d}: {r['minus_realistic_per_turn']:+.3f} "
                  f"[{r['rdiff_lo']:+.3f}, {r['rdiff_hi']:+.3f}]", flush=True)
    for r in rows:
        if r["detail"] in ("all", "name+condition linkage") or r["method"] == "mention":
            print(f"  {r['attacker']:34s} {r['method']:7s} {r['detail']:24s} n={r['n']:4d}: {r['recovery']:.3f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" + (f" | vs per-turn {r['minus_per_turn']:+.3f} "
                  f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]" if "minus_per_turn" in r else ""), flush=True)
    res = dict(config=CFG, smoke=a.smoke, adapters=os.path.basename(path), meta=meta, kv_check=kv,
               session_stats=S.stats(sessions), rows=rows, minutes=(time.time() - t0) / 60,
               labels=dict(union_vote="UPPER BOUND: mention alignment given", sessions="TEMPLATED, assistant replies fixed",
                           realistic_turn_first="uses which turn holds the first mention",
                           realistic_turn_any="optimistic: any single turn's guess right",
                           main_c2="realistic (session vote) - realistic_turn_first, paired bootstrap"))
    json.dump(res, open(os.path.join(CFG["out_dir"], "session_attack.json"), "w"), indent=1, default=str)
    keys = ["attacker", "method", "detail", "n", "recovery", "ci_lo", "ci_hi", "minus_per_turn", "diff_lo", "diff_hi",
            "minus_realistic_per_turn", "rdiff_lo", "rdiff_hi", "public", "protected"]
    with open(os.path.join(CFG["out_dir"], "session_attack.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"Saved to {CFG['out_dir']}/ ({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
