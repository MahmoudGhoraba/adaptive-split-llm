"""
NOISE SWEEP - how much does noise protect the prompt, and what does it cost in answer quality?
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script (does not need inversion_attack.py). Model: Llama-3.2-1B-Instruct.

PROTECTION MECHANISM
  Before sending, the client adds Gaussian noise to the hidden state after layer H:

      h_sent = h + sigma * s_H * z,     z ~ N(0, I)

  s_H = per-dimension standard deviation of layer H's hidden states (measured on public text),
  so sigma is the noise strength relative to the natural spread of each dimension
  (sigma = 0 is the unprotected split measured before; sigma = 1 means noise as large as the signal's
  own variation). The server runs its U layers on the noisy state; the client's tail and LM head
  then produce the answer.

FOR EVERY (H, sigma) THE SCRIPT MEASURES
  1. Leakage: an ADAPTIVE attacker (knows the mechanism and sigma, trains on its own noisy samples)
     tries to recover each token from h_sent. Same metrics as inversion_attack.py:
       content_leak (MAIN: content tokens, above a blind guess), weighted_top1, sensitive_top1,
       sensitive_exact (planted names, hospitals, diseases, phone numbers, amounts, cities)
  2. Answer quality: run the full model with the noise injected at layer H on held-out text and
     compare with the clean model:
       agreement = share of positions where the noisy model predicts the same next token (1 = no harm)
       ppl / ppl_clean = perplexity on the true next tokens (1 = no harm, higher = worse)

  Only the prompt side is protected here. With public tail weights the server can still run the tail
  and read the answer, so answer leakage stays 100% whatever sigma is.

SANITY CHECKS (printed as PASS / FAIL)
  1. sigma = 0 gives agreement = 1.000 (noise hook is wired correctly)
  2. sigma = 0 leakage at the smallest H is high (> 0.9), as in the first attack
  3. leakage falls as sigma grows (for every H)
  4. quality falls as sigma grows (for every H)

HOW TO RUN ON GOOGLE COLAB (T4 GPU; roughly 40-60 min with the default grid)
  1. Runtime -> Change runtime type -> T4 GPU, upload this file
  2. from huggingface_hub import notebook_login; notebook_login()
  3. !pip install -q datasets
  4. !python noise_sweep.py
  5. Download noise_results/ (results.json, noise_results.csv, plots) and send it back.
  Smaller/faster grid:  !python noise_sweep.py --H 1 4 8 12 --sigma 0 0.5 1 2 4
  Quick code test without the real model (CPU, ~1 min):  python noise_sweep.py --smoke
"""
import argparse
import csv
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(
    model_id           = "meta-llama/Llama-3.2-1B-Instruct",
    H_list             = [1, 2, 4, 6, 8, 10, 12],        # splits to test (<= 12 fit in 2 GB on the client)
    sigma_list         = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    train_tokens       = 400_000,
    test_tokens        = 40_000,
    n_sensitive        = 400,
    seq_len            = 128,
    batch_seqs         = 8,
    positions_per_step = 384,
    epochs             = 2,
    lr                 = 1e-3,
    hidden_mlp         = 2048,
    std_batches        = 20,        # batches used to measure the per-dimension spread s_H
    bootstrap          = 300,
    seed               = 0,
    out_dir            = "noise_results",
)

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves s t d ll m re ve also will just may might must shall one two three would
""".split())


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ----------------------------------------------------------------------------------
# Tokenizers (real + smoke test)
# ----------------------------------------------------------------------------------
class HFTok:
    def __init__(self, tok):
        self.tok = tok; self.bos_id = tok.bos_token_id; self.vocab_size = len(tok)
    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]
    def encode_offsets(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]
    def decode_one(self, i):
        return self.tok.decode([i])


class WordTok:
    def __init__(self, texts):
        words = sorted(set(w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)))
        self.itos = ["<bos>"] + words; self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.bos_id = 0; self.vocab_size = len(self.itos)
    def encode_offsets(self, text):
        ids, offs = [], []
        for m in re.finditer(r"\w+|[^\w\s]", text):
            ids.append(self.stoi[m.group()]); offs.append((m.start(), m.end()))
        return ids, offs
    def encode(self, text):
        return self.encode_offsets(text)[0]
    def decode_one(self, i):
        return " " + self.itos[i]


# ----------------------------------------------------------------------------------
# Data (same as inversion_attack.py)
# ----------------------------------------------------------------------------------
FIRST = ["Ahmed", "Mona", "Youssef", "Nour", "Omar", "Salma", "Karim", "Laila", "Hassan", "Mariam",
         "John", "Emily", "Carlos", "Aisha", "Wei", "Sofia", "David", "Fatima", "Lucas", "Hana"]
LAST = ["Mahmoud", "Hassan", "Ibrahim", "Farouk", "Saleh", "Kamal", "Smith", "Garcia", "Chen",
        "Mansour", "Nasser", "Brown", "Rossi", "Haddad", "Khalil", "Morgan"]
HOSPITALS = ["Cleopatra Hospital", "Dar Al Fouad Hospital", "Kasr Al Ainy Hospital", "As-Salam Hospital",
             "Andalusia Hospital", "Saint Mary Hospital", "City General Hospital", "Nile Badrawi Hospital"]
DISEASES = ["diabetes", "breast cancer", "HIV", "epilepsy", "depression", "hepatitis C", "asthma",
            "kidney failure", "schizophrenia", "tuberculosis", "leukemia", "heart failure"]
CITIES = ["Cairo", "Alexandria", "Giza", "Mansoura", "Tanta", "Aswan", "Luxor", "London", "Berlin", "Dubai"]
TEMPLATES = [
    "My name is {name} and I was admitted to {hospital} for {disease}.",
    "Please write a sick note for {name}, who is being treated for {disease} at {hospital}.",
    "I live in {city} and my phone number is {phone}. Can you help me find a doctor for {disease}?",
    "Transfer {amount} pounds from the account of {name} to pay the bill at {hospital}.",
    "{name} from {city} was diagnosed with {disease} last month. What should the family know?",
    "Summarise the report: patient {name}, phone {phone}, condition {disease}, clinic {hospital}.",
    "I owe {amount} pounds to {name} in {city}. Write a polite message asking for more time.",
    "Can you explain the treatment options for {disease}? I am {name} and I am scared.",
]


def make_sensitive_prompts(n, rng):
    out = []
    for _ in range(n):
        tpl = rng.choice(TEMPLATES)
        fill = dict(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", hospital=rng.choice(HOSPITALS),
                    disease=rng.choice(DISEASES), city=rng.choice(CITIES),
                    phone="01" + "".join(str(rng.randint(0, 9)) for _ in range(9)),
                    amount=str(rng.randint(2, 99) * 500))
        text, spans = "", []
        for part in re.split(r"(\{\w+\})", tpl):
            m = re.fullmatch(r"\{(\w+)\}", part)
            if m:
                val = fill[m.group(1)]
                spans.append((len(text), len(text) + len(val), m.group(1))); text += val
            else:
                text += part
        out.append((text, spans))
    return out


def load_general_text(smoke, rng):
    if smoke:
        words = ("the patient doctor city river history music game people school water light time year "
                 "is was are and of to in a with for on by music king war army team song film road").split()
        para = lambda: " ".join(rng.choice(words) for _ in range(rng.randint(20, 60))) + " ."
        return [para() for _ in range(1500)], [para() for _ in range(200)]
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    clean = lambda split: [l.strip() for l in ds[split]["text"] if len(l.strip()) > 40 and not l.strip().startswith("=")]
    return clean("train"), clean("test")


def chunk(tok, texts, seq_len, max_tokens):
    ids = []
    for t in texts:
        ids.extend(tok.encode(" " + t))
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]; body = seq_len - 1; n = len(ids) // body
    return torch.tensor([[tok.bos_id] + ids[i * body:(i + 1) * body] for i in range(n)], dtype=torch.long)


def encode_sensitive(tok, prompts, seq_len):
    seqs, sens, valid = [], [], []
    for text, spans in prompts:
        ids, offs = tok.encode_offsets(text)
        ids, offs = ids[:seq_len - 1], offs[:seq_len - 1]
        s = [0] * len(ids)
        for k, (a, b) in enumerate(offs):
            for j, (sa, sb, _) in enumerate(spans):
                if a < sb and b > sa:
                    s[k] = j + 1
        pad = seq_len - 1 - len(ids)
        seqs.append([tok.bos_id] + ids + [tok.bos_id] * pad)
        sens.append([0] + s + [0] * pad)
        valid.append([0] + [1] * len(ids) + [0] * pad)
    return torch.tensor(seqs), torch.tensor(sens), torch.tensor(valid, dtype=torch.bool)


# ----------------------------------------------------------------------------------
# Model, hidden-state capture, noise injection
# ----------------------------------------------------------------------------------
def load_model(smoke, tok_texts=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if smoke:
        tok = WordTok(tok_texts)
        cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          tie_word_embeddings=True, max_position_embeddings=256)
        return LlamaForCausalLM(cfg).to(dev).eval(), tok, dev
    hf_tok = AutoTokenizer.from_pretrained(CFG["model_id"])
    dtype = torch.float16 if dev == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], torch_dtype=dtype)
    return model.to(dev).eval(), HFTok(hf_tok), dev


class Catcher:
    """states[L] = output of decoder layer L (L >= 1), only for the layers asked for."""
    def __init__(self, model, layers):
        self.states = {}
        self.h = [model.model.layers[L - 1].register_forward_hook(self._hook(L)) for L in layers]
    def _hook(self, L):
        def f(mod, inp, out):
            self.states[L] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return f
    def remove(self):
        for h in self.h:
            h.remove()


class NoiseInjector:
    """Adds sigma * s_H * z to the output of decoder layer H (what the client sends)."""
    def __init__(self, model, H, sigma, std_H, gen):
        self.sigma, self.std, self.gen = sigma, std_H, gen
        self.h = model.model.layers[H - 1].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        if self.sigma == 0:
            return out
        x = out[0] if isinstance(out, (tuple, list)) else out
        z = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=torch.float32)
        x2 = (x.float() + self.sigma * self.std * z).to(x.dtype)
        return (x2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else x2
    def remove(self):
        self.h.remove()


@torch.no_grad()
def layer_std(model, dev, ids, layers):
    """Per-dimension std of each layer's hidden states over non-BOS tokens."""
    c = Catcher(model, layers)
    acc = {L: [] for L in layers}
    for b in range(min(CFG["std_batches"], math.ceil(ids.shape[0] / CFG["batch_seqs"]))):
        x = ids[b * CFG["batch_seqs"]:(b + 1) * CFG["batch_seqs"]]
        if len(x) == 0:
            break
        c.states = {}
        model(input_ids=x.to(dev))
        for L in layers:
            acc[L].append(c.states[L][:, 1:].float().reshape(-1, c.states[L].shape[-1]))
    c.remove()
    return {L: torch.cat(v).std(0).clamp_min(1e-6) for L, v in acc.items()}


# ----------------------------------------------------------------------------------
# Attacker
# ----------------------------------------------------------------------------------
class InvHead(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, h, E):
        return self.scale * (self.net(h) @ E.t())


def noisy(h, sigma, std):
    h = h.float()
    return h if sigma == 0 else h + sigma * std * torch.randn_like(h)


def train_attackers(model, dev, train_ids, E, stds):
    d = E.shape[1]
    keys = [(H, s) for H in CFG["H_list"] for s in CFG["sigma_list"]]
    heads = {k: InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev) for k in keys}
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=CFG["lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    catcher = Catcher(model, CFG["H_list"])
    n_seq = train_ids.shape[0]
    spe = math.ceil(n_seq / CFG["batch_seqs"]); total = spe * CFG["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=total, pct_start=0.1)
    t0, step = time.time(), 0
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n_seq)
        for b in range(spe):
            idx = perm[b * CFG["batch_seqs"]:(b + 1) * CFG["batch_seqs"]]
            if len(idx) == 0:
                continue
            ids = train_ids[idx]
            with torch.no_grad():
                catcher.states = {}
                model(input_ids=ids.to(dev))
            B, S = ids.shape
            pos = torch.randperm(B * (S - 1), device=dev)[:CFG["positions_per_step"]]
            flat = (pos // (S - 1)) * S + (pos % (S - 1)) + 1
            y = ids.to(dev).reshape(-1)[flat]
            opt.zero_grad(set_to_none=True); losses = []
            for (H, s), head in heads.items():
                x = noisy(catcher.states[H].reshape(B * S, -1)[flat], s, stds[H])
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = head(x, E)
                loss = F.cross_entropy(logits.float(), y)
                scaler.scale(loss).backward(); losses.append(loss.item())
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                print(f"  step {step}/{total}  mean loss {np.mean(losses):.3f}  elapsed {el/60:.1f} min  "
                      f"eta {el/step*(total-step)/60:.1f} min", flush=True)
    catcher.remove()
    return heads


@torch.no_grad()
def attack_predict(model, dev, heads, ids_all, E, stds, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    catcher = Catcher(model, CFG["H_list"])
    preds = {k: [] for k in heads}
    for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
        ids = ids_all[b:b + CFG["batch_seqs"]]
        catcher.states = {}
        model(input_ids=ids.to(dev))
        B, S = ids.shape
        for (H, s), head in heads.items():
            h = catcher.states[H].reshape(B * S, -1).float()
            if s:
                h = h + s * stds[H] * torch.randn(h.shape, generator=g, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = head(h, E).argmax(-1)
            preds[(H, s)].append(p.reshape(B, S).cpu())
    catcher.remove()
    return {k: torch.cat(v).numpy() for k, v in preds.items()}


@torch.no_grad()
def quality(model, dev, ids_all, stds):
    """Agreement with the clean model and perplexity, for every (H, sigma)."""
    def run(H=None, s=0.0):
        inj = NoiseInjector(model, H, s, stds[H], torch.Generator(device=dev).manual_seed(123)) if H else None
        preds, nll, n = [], 0.0, 0
        for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
            ids = ids_all[b:b + CFG["batch_seqs"]].to(dev)
            logits = model(input_ids=ids).logits.float()
            preds.append(logits.argmax(-1).cpu())
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            nll += -lp.sum().item(); n += lp.numel()
        if inj:
            inj.remove()
        return torch.cat(preds), math.exp(nll / n)
    clean_pred, clean_ppl = run()
    out = {}
    for H in CFG["H_list"]:
        for s in CFG["sigma_list"]:
            p, ppl = run(H, s)
            out[(H, s)] = dict(agreement=float((p == clean_pred).float().mean()), ppl=ppl,
                               ppl_ratio=ppl / clean_ppl)
    return out, clean_ppl


# ----------------------------------------------------------------------------------
# Metrics (same definitions as inversion_attack.py)
# ----------------------------------------------------------------------------------
def token_tables(tok, train_ids):
    V = tok.vocab_size
    is_content = np.zeros(V, dtype=bool)
    for i in range(V):
        s = tok.decode_one(i).strip().lower()
        if re.search(r"[a-z0-9]", s) and s not in STOPWORDS and (len(s) >= 3 or s.isdigit()):
            is_content[i] = True
    counts = np.bincount(train_ids.reshape(-1).numpy(), minlength=V).astype(float)
    counts[tok.bos_id] = 0
    surprisal = -np.log2((counts + 1) / (counts.sum() + V))
    return is_content, surprisal, int(np.argmax(np.where(is_content, counts, -1)))


def ratio_ci(num, den, seq, rng):
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    S = seq.max() + 1
    nps = np.bincount(seq, weights=num, minlength=S); dps = np.bincount(seq, weights=den, minlength=S)
    bs = []
    for _ in range(CFG["bootstrap"]):
        r = rng.integers(0, S, S); d = dps[r].sum()
        if d > 0:
            bs.append(nps[r].sum() / d)
    return float(nps.sum() / dps.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def leak_metrics(pred, label, valid, tables, rng, sens=None):
    is_content, surprisal, blind_content = tables
    N, S = label.shape
    seq = np.repeat(np.arange(N), S).reshape(N, S)
    corr = pred == label; cm = valid & is_content[label]; w = surprisal[label] * valid
    out = {}
    c = ratio_ci(corr[cm].astype(float), np.ones(cm.sum()), seq[cm], rng)
    blind = float((label[cm] == blind_content).mean()) if cm.sum() else 0.0
    out["content_leak"] = tuple(float(np.clip((v - blind) / (1 - blind), 0, 1)) for v in c)
    out["weighted_top1"] = ratio_ci((corr * w)[valid], w[valid], seq[valid], rng)[0]
    out["all_top1"] = float(corr[valid].mean())
    if sens is not None:
        sm = valid & (sens > 0)
        out["sensitive_top1"] = ratio_ci(corr[sm].astype(float), np.ones(sm.sum()), seq[sm], rng)
        ok = tot = 0
        for i in range(N):
            for e in set(sens[i][sens[i] > 0].tolist()):
                p = (sens[i] == e) & valid[i]
                if p.any():
                    tot += 1; ok += int(corr[i][p].all())
        out["sensitive_exact"] = ok / tot if tot else float("nan")
    return out


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--H", type=int, nargs="+", help="splits to test, e.g. --H 1 4 8 12")
    ap.add_argument("--sigma", type=float, nargs="+", help="noise levels, e.g. --sigma 0 0.5 1 2")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    if a.smoke:
        CFG.update(H_list=[1, 2, 3], sigma_list=[0.0, 0.5, 2.0], train_tokens=60_000, test_tokens=6_000,
                   n_sensitive=60, seq_len=64, positions_per_step=256, epochs=3, bootstrap=50,
                   std_batches=10, out_dir="noise_results_smoke")
    if a.H: CFG["H_list"] = sorted(set(a.H))
    if a.sigma: CFG["sigma_list"] = sorted(set(a.sigma))
    if 0.0 not in CFG["sigma_list"]: CFG["sigma_list"] = [0.0] + CFG["sigma_list"]
    if a.epochs: CFG["epochs"] = a.epochs
    if a.out: CFG["out_dir"] = a.out
    set_seed(CFG["seed"]); rng_py = random.Random(CFG["seed"]); rng_np = np.random.default_rng(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)
    print(f"Grid: H = {CFG['H_list']}, sigma = {CFG['sigma_list']} "
          f"({len(CFG['H_list']) * len(CFG['sigma_list'])} attackers)", flush=True)

    print("1/6 data", flush=True)
    train_txt, test_txt = load_general_text(a.smoke, rng_py)
    sens_prompts = make_sensitive_prompts(CFG["n_sensitive"], rng_py)
    print("2/6 model", flush=True)
    model, tok, dev = load_model(a.smoke, train_txt + test_txt + [t for t, _ in sens_prompts])
    n_layers = len(model.model.layers)
    assert max(CFG["H_list"]) < n_layers, f"H must be < {n_layers}"
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    train_ids = chunk(tok, train_txt, CFG["seq_len"], CFG["train_tokens"])
    test_ids = chunk(tok, test_txt, CFG["seq_len"], CFG["test_tokens"])
    sens_ids, sens_mask, sens_valid = encode_sensitive(tok, sens_prompts, CFG["seq_len"])
    tables = token_tables(tok, train_ids)
    print(f"   device {dev}, {n_layers} layers; train {train_ids.numel()} tokens, test {test_ids.numel()}", flush=True)

    print("3/6 measuring the natural spread of each layer (noise scale)", flush=True)
    stds = layer_std(model, dev, train_ids, CFG["H_list"])

    print("4/6 answer quality under noise (clean model vs noisy model)", flush=True)
    qual, clean_ppl = quality(model, dev, test_ids, stds)
    print(f"   clean perplexity {clean_ppl:.2f}", flush=True)

    print("5/6 training adaptive attackers (one per split and noise level)", flush=True)
    heads = train_attackers(model, dev, train_ids, E, stds)
    for h in heads.values():
        h.eval()

    print("6/6 attacking held-out text and sensitive prompts", flush=True)
    p_test = attack_predict(model, dev, heads, test_ids, E, stds, seed=1)
    p_sens = attack_predict(model, dev, heads, sens_ids, E, stds, seed=2)
    tv = np.ones(test_ids.shape, dtype=bool); tv[:, 0] = False

    rows = []
    for H in CFG["H_list"]:
        for s in CFG["sigma_list"]:
            g = leak_metrics(p_test[(H, s)], test_ids.numpy(), tv, tables, rng_np)
            se = leak_metrics(p_sens[(H, s)], sens_ids.numpy(), sens_valid.numpy(), tables, rng_np,
                              sens=sens_mask.numpy())
            q = qual[(H, s)]
            rows.append(dict(H=H, sigma=s, content_leak=g["content_leak"], weighted_top1=g["weighted_top1"],
                             all_top1=g["all_top1"], sensitive_top1=se["sensitive_top1"],
                             sensitive_exact=se["sensitive_exact"], **q))
            print(f"   H={H:2d} sigma={s:4.2f}: leak {g['content_leak'][0]:.3f} "
                  f"[{g['content_leak'][1]:.3f},{g['content_leak'][2]:.3f}]  private {se['sensitive_top1'][0]:.3f} "
                  f"(exact {se['sensitive_exact']:.3f})  agreement {q['agreement']:.3f}  ppl x{q['ppl_ratio']:.2f}",
                  flush=True)

    # ---- sanity checks ----
    get = lambda H, s: next(r for r in rows if r["H"] == H and r["sigma"] == s)
    sig = CFG["sigma_list"]; H0 = CFG["H_list"][0]
    checks = [
        ("sigma = 0 leaves the model unchanged (agreement = 1)",
         all(abs(get(H, 0.0)["agreement"] - 1) < 1e-6 for H in CFG["H_list"]),
         ", ".join(f"H={H}: {get(H, 0.0)['agreement']:.4f}" for H in CFG["H_list"])),
        ("unprotected leakage is high at the smallest H", get(H0, 0.0)["content_leak"][0] > 0.9,
         f"H={H0}: {get(H0, 0.0)['content_leak'][0]:.3f}"),
        ("leakage falls as noise grows (largest vs no noise, every H)",
         all(get(H, sig[-1])["content_leak"][0] < get(H, 0.0)["content_leak"][0] for H in CFG["H_list"]), ""),
        ("quality falls as noise grows (largest vs no noise, every H)",
         all(get(H, sig[-1])["agreement"] < get(H, 0.0)["agreement"] for H in CFG["H_list"]), ""),
    ]
    print("\nSANITY CHECKS (a FAIL on check 1 or 2 means something is wrong in the setup)")
    for name, ok, info in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}  {('(' + info + ')') if info else ''}")

    # ---- save ----
    leak_d = {f"{r['H']},{r['sigma']}": round(r["content_leak"][0], 4) for r in rows}
    qual_d = {f"{r['H']},{r['sigma']}": round(r["agreement"], 4) for r in rows}
    with open(os.path.join(CFG["out_dir"], "results.json"), "w") as f:
        json.dump(dict(config=CFG, smoke=a.smoke, clean_ppl=clean_ppl,
                       sanity_checks=[dict(name=n, passed=bool(o), info=i) for n, o, i in checks],
                       MEASURED_LEAK=leak_d, MEASURED_AGREEMENT=qual_d, rows=rows), f, indent=1)
    with open(os.path.join(CFG["out_dir"], "noise_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["H", "sigma", "content_leak", "content_leak_low", "content_leak_high", "weighted_top1",
                    "all_top1", "sensitive_top1", "sensitive_exact", "agreement", "ppl", "ppl_ratio"])
        for r in rows:
            w.writerow([r["H"], r["sigma"], *[round(v, 4) for v in r["content_leak"]], round(r["weighted_top1"], 4),
                        round(r["all_top1"], 4), round(r["sensitive_top1"][0], 4), round(r["sensitive_exact"], 4),
                        round(r["agreement"], 4), round(r["ppl"], 3), round(r["ppl_ratio"], 3)])

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(CFG["H_list"])))
    xs = np.array(sig); xplot = np.where(xs == 0, xs[xs > 0].min() / 2 if (xs > 0).any() else 1, xs)
    title = "TINY RANDOM MODEL (smoke test)" if a.smoke else CFG["model_id"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for H, c in zip(CFG["H_list"], cols):
        rs = [get(H, s) for s in sig]
        lk = np.array([r["content_leak"] for r in rs])
        ax[0].fill_between(xplot, 100 * lk[:, 1], 100 * lk[:, 2], color=c, alpha=.15, lw=0)
        ax[0].plot(xplot, 100 * lk[:, 0], "-o", ms=4, color=c, label=f"H={H}")
        ax[1].plot(xplot, [100 * r["agreement"] for r in rs], "-o", ms=4, color=c, label=f"H={H}")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xticks(xplot); a_.set_xticklabels([f"{s:g}" for s in xs])
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a_.set_xlabel("Noise strength σ (relative to each dimension's spread)"); a_.grid(alpha=.3)
        a_.set_ylim(-2, 102)
    ax[0].set_ylabel("Content tokens recovered, above blind [%]"); ax[0].set_title("Leakage (adaptive attacker)")
    ax[1].set_ylabel("Next token same as clean model [%]"); ax[1].set_title("Answer quality")
    ax[1].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle(f"Noise on the hidden state sent to the server: {title}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(CFG["out_dir"], "noise_leak_quality.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for H, c in zip(CFG["H_list"], cols):
        rs = [get(H, s) for s in sig]
        ax.plot([100 * r["agreement"] for r in rs], [100 * r["content_leak"][0] for r in rs], "-o", ms=4,
                color=c, label=f"H={H}")
    ax.set_xlabel("Answer quality: next token same as clean model [%]")
    ax.set_ylabel("Leakage: content tokens recovered [%]")
    ax.set_title(f"Privacy-quality trade-off (each point = one noise level)\n{title}", fontsize=11)
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(CFG["out_dir"], "noise_tradeoff.png"), dpi=200)

    print("\nCOPY THESE INTO STEP 3 (keys are 'H,sigma'):")
    print(f"MEASURED_LEAK      = {leak_d}")
    print(f"MEASURED_AGREEMENT = {qual_d}")
    print(f"\nSaved to {CFG['out_dir']}/  (results.json, noise_results.csv, noise_leak_quality.png, noise_tradeoff.png)")


if __name__ == "__main__":
    main()
