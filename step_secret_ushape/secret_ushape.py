"""
SECRET U-SHAPE - protect the prompt AND the answer with secret adapters at both ends of the split
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script. Model: Llama-3.2-1B-Instruct, U-shaped split:
    client: embedding + H head layers   (+ SECRET adapters)
    server: U = 16 - H - T middle layers (public weights)
    client: T tail layers + output layer (+ SECRET adapters)

WHY
  With a public tail the server can run the tail itself on what it sends back and read every answer
  word. Those answer words are fed back through the client's head, so normal traffic would also give
  the server (word, hidden state) pairs to train an attacker on. Secret tail adapters stop both.

TRAINING (on the client; the server's layers never change)
  loss = KL(public model || defended model)                         keep the answers
       + lambda_prompt * (inverter on layer H becomes unsure)       hide the prompt
       + lambda_answer * (inverter on layer n-T becomes unsure      hide the answer
                          + the PUBLIC tail run on layer n-T becomes unsure)

MEASURED for every (lambda_prompt, lambda_answer)
  quality          next-token agreement with the public model, perplexity ratio
  prompt leakage   attacker without the secret (trained on the public model) / with the secret
                   (content words above a blind guess; plus full private details on planted prompts)
  answer leakage   (1) server runs the public tail on what it sends back -> does it get the real answer?
                   (2) attacker without the secret, (3) attacker with the secret
                   (share of the client's next content tokens the server recovers, above a blind guess)

SANITY CHECKS: no defence -> prompt attacker > 0.9 and server-public-tail answer recovery > 0.99;
defended models keep agreement > 0.5; attackers with the secret >= without.

HOW TO RUN ON GOOGLE COLAB (T4 GPU; roughly 1.5-2.5 h with the default 3 configurations)
  1. Runtime -> Change runtime type -> T4 GPU, upload this file
  2. from huggingface_hub import notebook_login; notebook_login()
  3. !pip install -q datasets
  4. !python secret_ushape.py                     (H = 4, T = 2, configs 0.1,0  0.1,0.3  0.2,0.3)
     faster:  !python secret_ushape.py --configs "0.1,0.3" --steps 800
  5. Download secret_ushape_results/ and send it back.
  Quick code test without the real model (CPU):  python secret_ushape.py --smoke
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
    H                  = 4,                    # head layers on the client
    T                  = 2,                    # tail layers on the client
    configs            = [(0.1, 0.0), (0.1, 0.3), (0.2, 0.3)],   # (lambda_prompt, lambda_answer)
    lora_rank          = 32,
    lora_alpha         = 32,
    secret_init        = 0.02,
    def_steps          = 1200,
    def_batch          = 4,
    def_lr             = 2e-4,
    inv_lr             = 1e-3,
    train_tokens       = 400_000,
    test_tokens        = 40_000,
    n_sensitive        = 400,
    seq_len            = 128,
    batch_seqs         = 8,
    positions_per_step = 384,
    epochs             = 2,
    hidden_mlp         = 2048,
    bootstrap          = 300,
    seed               = 0,
    out_dir            = "secret_ushape_results",
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
# Tokenizers and data (same as the other scripts)
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



# ----------------------------------------------------------------------------------
# Secret LoRA adapters
# ----------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, init_scale, gen):
        super().__init__()
        self.base = base
        fin, fout = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, fin, generator=gen) / math.sqrt(fin))
        self.B = nn.Parameter(torch.randn(fout, r, generator=gen) * init_scale)
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)
        return y


PROJ = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]


def add_adapters(model, layer_idx, seed, dev):
    """Secret LoRA adapters on the given decoder layers (0-based indices)."""
    gen = torch.Generator().manual_seed(seed)
    mods = []
    for li in layer_idx:
        layer = model.model.layers[li]
        for parent, nm in PROJ:
            par = getattr(layer, parent)
            lora = LoRALinear(getattr(par, nm), CFG["lora_rank"], CFG["lora_alpha"], CFG["secret_init"], gen)
            lora.A.data = lora.A.data.to(dev); lora.B.data = lora.B.data.to(dev)
            setattr(par, nm, lora); mods.append(lora)
    return mods


def remove_all_adapters(model):
    for layer in model.model.layers:
        for parent, nm in PROJ:
            par = getattr(layer, parent); m = getattr(par, nm)
            if isinstance(m, LoRALinear):
                setattr(par, nm, m.base)


def set_on(mods, on):
    for m in mods:
        m.enabled = on


class Catcher:
    """state = output of decoder layer L (1-based)."""
    def __init__(self, model, L, detach=True):
        self.state, self.detach = None, detach
        self.h = model.model.layers[L - 1].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        self.state = x.detach() if self.detach else x
    def remove(self):
        self.h.remove()


class InvHead(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, h, E):
        return self.scale * (self.net(h) @ E.t())


def sample_positions(B, S, n, dev):
    pos = torch.randperm(B * (S - 1), device=dev)[:n]
    return (pos // (S - 1)) * S + (pos % (S - 1)) + 1


def neg_entropy(logits):
    lp = F.log_softmax(logits.float(), -1)
    return (lp.exp() * lp).sum(-1).mean()


# ----------------------------------------------------------------------------------
# Attackers
#   target = "input":  recover the token the client sent (prompt side, at layer H)
#   target = "output": recover the token the client will output next (answer side, at layer n - T)
# ----------------------------------------------------------------------------------
def train_inverter(model, dev, ids_all, E, L, target, label):
    d = E.shape[1]
    inv = InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
    opt = torch.optim.AdamW(inv.parameters(), lr=CFG["inv_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cat = Catcher(model, L)
    n = ids_all.shape[0]; spe = math.ceil(n / CFG["batch_seqs"]); total = spe * CFG["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["inv_lr"], total_steps=total, pct_start=0.1)
    t0 = time.time()
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n)
        for b in range(spe):
            idx = perm[b * CFG["batch_seqs"]:(b + 1) * CFG["batch_seqs"]]
            if len(idx) == 0:
                continue
            ids = ids_all[idx].to(dev)
            with torch.no_grad():
                out_tok = model(input_ids=ids).logits.argmax(-1)
            B, S = ids.shape
            flat = sample_positions(B, S, CFG["positions_per_step"], dev)
            x = cat.state.reshape(B * S, -1)[flat].float()
            y = (ids if target == "input" else out_tok).reshape(-1)[flat]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = inv(x, E)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
        print(f"   [{label}] epoch {ep+1}/{CFG['epochs']}  loss {loss.item():.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
    cat.remove()
    return inv.eval()


@torch.no_grad()
def inverter_predict(model, dev, inv, ids_all, E, L, target):
    """Predictions of the inverter at layer L and the true labels, for the model as currently configured."""
    cat = Catcher(model, L); preds, labels = [], []
    for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
        ids = ids_all[b:b + CFG["batch_seqs"]].to(dev)
        out_tok = model(input_ids=ids).logits.argmax(-1)
        B, S = ids.shape
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            p = inv(cat.state.reshape(B * S, -1).float(), E).argmax(-1)
        preds.append(p.reshape(B, S).cpu()); labels.append((ids if target == "input" else out_tok).cpu())
    cat.remove()
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def argmax_tokens(model, dev, ids_all):
    out = []
    for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
        out.append(model(input_ids=ids_all[b:b + CFG["batch_seqs"]].to(dev)).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------------
# Defender: secret head + secret tail
# ----------------------------------------------------------------------------------
def train_defender(model, dev, head, tail, train_ids, E, H, LT, lam_p, lam_a):
    """KL to the public model + prompt adversary at layer H + answer adversaries:
       (a) an inverter learning the output token from what the server sends back (layer LT = n - T)
       (b) the server running the PUBLIC tail on what it sends back (head on, tail off)."""
    d = E.shape[1]
    params = [p for m in head + tail for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=CFG["def_lr"], weight_decay=0.0)
    mk = lambda: InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
    inv_p, inv_a = mk(), mk()
    opt_p = torch.optim.AdamW(inv_p.parameters(), lr=CFG["inv_lr"], weight_decay=0.01)
    opt_a = torch.optim.AdamW(inv_a.parameters(), lr=CFG["inv_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    sc, sc_p, sc_a = (torch.amp.GradScaler("cuda", enabled=use_amp) for _ in range(3))
    cat_p, cat_a = Catcher(model, H, detach=False), Catcher(model, LT, detach=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    n = train_ids.shape[0]; t0 = time.time()
    for step in range(1, CFG["def_steps"] + 1):
        ids = train_ids[torch.randint(0, n, (CFG["def_batch"],))].to(dev)
        B, S = ids.shape
        flat = sample_positions(B, S, min(CFG["positions_per_step"], B * (S - 1)), dev)
        with torch.no_grad():                                         # public model = teacher
            set_on(head + tail, False)
            t_logp = F.log_softmax(model(input_ids=ids).logits.float(), -1)
        set_on(head + tail, True)                                     # defended model
        s_logits = model(input_ids=ids).logits.float()
        kl = F.kl_div(F.log_softmax(s_logits, -1), t_logp, log_target=True, reduction="none").sum(-1)[:, 1:].mean()
        hp = cat_p.state.reshape(B * S, -1)[flat].float()
        ha = cat_a.state.reshape(B * S, -1)[flat].float()
        y_out = s_logits.argmax(-1).reshape(-1)[flat].detach()
        loss = kl
        for inv in (inv_p, inv_a):
            for p in inv.parameters():
                p.requires_grad_(False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if lam_p > 0:
                loss = loss + lam_p * neg_entropy(inv_p(hp, E))
            if lam_a > 0:
                loss = loss + lam_a * neg_entropy(inv_a(ha, E))
        opt.zero_grad(set_to_none=True)
        sc.scale(loss).backward()
        srv_ne = torch.tensor(0.0)
        if lam_a > 0:                                                 # server runs the public tail
            set_on(tail, False)
            srv_logits = model(input_ids=ids).logits.float()
            srv_ne = neg_entropy(srv_logits.reshape(B * S, -1)[flat])
            sc.scale(lam_a * srv_ne).backward()
            set_on(tail, True)
        sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); sc.step(opt); sc.update()
        for inv in (inv_p, inv_a):
            for p in inv.parameters():
                p.requires_grad_(True)
        # inverters keep adapting (min-max)
        for inv, o, s_, x, y in ((inv_p, opt_p, sc_p, hp.detach(), ids.reshape(-1)[flat]),
                                 (inv_a, opt_a, sc_a, ha.detach(), y_out)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                li = F.cross_entropy(inv(x, E).float(), y)
            o.zero_grad(set_to_none=True); s_.scale(li).backward(); s_.step(o); s_.update()
        if step % 50 == 0 or step == CFG["def_steps"]:
            print(f"   [defender λp={lam_p:g} λa={lam_a:g}] step {step}/{CFG['def_steps']}  KL {kl.item():.4f}  "
                  f"server-tail neg-entropy {srv_ne.item():.2f}  {(time.time()-t0)/60:.1f} min", flush=True)
    cat_p.remove(); cat_a.remove()
    for p in params:
        p.requires_grad_(False)


@torch.no_grad()
def quality(model, dev, mods, ids_all):
    def run(on):
        set_on(mods, on)
        preds, nll, n = [], 0.0, 0
        for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
            ids = ids_all[b:b + CFG["batch_seqs"]].to(dev)
            logits = model(input_ids=ids).logits.float()
            preds.append(logits.argmax(-1).cpu())
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            nll += -lp.sum().item(); n += lp.numel()
        return torch.cat(preds), math.exp(nll / n)
    p0, ppl0 = run(False); p1, ppl1 = run(True); set_on(mods, True)
    return dict(agreement=float((p0 == p1).float().mean()), ppl_ratio=ppl1 / ppl0)


# ----------------------------------------------------------------------------------
# Metrics (same definitions as before)
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
    c = ratio_ci(corr[cm].astype(float), np.ones(cm.sum()), seq[cm], rng)
    blind = float((label[cm] == blind_content).mean()) if cm.sum() else 0.0
    out = dict(content_leak=tuple(float(np.clip((v - blind) / (1 - blind), 0, 1)) for v in c),
               weighted_top1=ratio_ci((corr * w)[valid], w[valid], seq[valid], rng)[0])
    if sens is not None:
        sm = valid & (sens > 0)
        out["sensitive_top1"] = ratio_ci(corr[sm].astype(float), np.ones(sm.sum()), seq[sm], rng)[0]
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
    ap.add_argument("--H", type=int); ap.add_argument("--T", type=int)
    ap.add_argument("--configs", type=str, help='e.g. "0.1,0 0.1,0.3 0.2,0.3" (lambda_prompt,lambda_answer)')
    ap.add_argument("--steps", type=int); ap.add_argument("--out", type=str)
    a = ap.parse_args()
    if a.smoke:
        CFG.update(H=1, T=1, configs=[(0.1, 0.0), (0.1, 1.0)], lora_rank=8, lora_alpha=8, def_steps=60, def_batch=4,
                   train_tokens=40_000, test_tokens=4_000, n_sensitive=40, seq_len=64, positions_per_step=128,
                   epochs=2, bootstrap=30, out_dir="secret_ushape_results_smoke")
    if a.H: CFG["H"] = a.H
    if a.T: CFG["T"] = a.T
    if a.configs: CFG["configs"] = [tuple(float(v) for v in c.split(",")) for c in a.configs.split()]
    if a.steps: CFG["def_steps"] = a.steps
    if a.out: CFG["out_dir"] = a.out
    H, T = CFG["H"], CFG["T"]
    set_seed(CFG["seed"]); rng_py = random.Random(CFG["seed"]); rng_np = np.random.default_rng(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)

    train_txt, test_txt = load_general_text(a.smoke, rng_py)
    sens_prompts = make_sensitive_prompts(CFG["n_sensitive"], rng_py)
    model, tok, dev = load_model(a.smoke, train_txt + test_txt + [t for t, _ in sens_prompts])
    n_layers = len(model.model.layers); LT = n_layers - T
    assert H >= 1 and T >= 1 and H + T < n_layers
    print(f"U-split: client head H={H}, server U={n_layers-H-T}, client tail T={T}; configs {CFG['configs']}", flush=True)
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    train_ids = chunk(tok, train_txt, CFG["seq_len"], CFG["train_tokens"])
    test_ids = chunk(tok, test_txt, CFG["seq_len"], CFG["test_tokens"])
    sens_ids, sens_mask, sens_valid = encode_sensitive(tok, sens_prompts, CFG["seq_len"])
    tables = token_tables(tok, train_ids)
    tv = np.ones(test_ids.shape, dtype=bool); tv[:, 0] = False

    def prompt_leak(inv):
        pg, lg = inverter_predict(model, dev, inv, test_ids, E, H, "input")
        ps, ls = inverter_predict(model, dev, inv, sens_ids, E, H, "input")
        g = leak_metrics(pg, lg, tv, tables, rng_np)
        s = leak_metrics(ps, ls, sens_valid.numpy(), tables, rng_np, sens=sens_mask.numpy())
        return dict(content_leak=g["content_leak"][0], ci=g["content_leak"][1:], private_exact=s["sensitive_exact"])

    def answer_leak(inv):
        pg, lg = inverter_predict(model, dev, inv, test_ids, E, LT, "output")
        g = leak_metrics(pg, lg, tv, tables, rng_np)
        return dict(content_leak=g["content_leak"][0], ci=g["content_leak"][1:])

    def server_public_tail(head, tail):
        """Server runs the PUBLIC tail on what it sends back: does it get the client's real answer?"""
        set_on(head, True); set_on(tail, True); true_out = argmax_tokens(model, dev, test_ids)
        set_on(tail, False); srv_out = argmax_tokens(model, dev, test_ids); set_on(tail, True)
        g = leak_metrics(srv_out, true_out, tv, tables, rng_np)
        return dict(content_leak=g["content_leak"][0], ci=g["content_leak"][1:])

    # attackers trained on the PUBLIC model (they do not have the secret)
    print("\n1) attackers without the secret (trained on the public model)", flush=True)
    invA_p = train_inverter(model, dev, train_ids, E, H, "input", "A prompt")
    invA_a = train_inverter(model, dev, train_ids, E, LT, "output", "A answer")
    rows = [dict(config="none", lam_p=None, lam_a=None, agreement=1.0, ppl_ratio=1.0,
                 prompt_A=prompt_leak(invA_p), prompt_B=None,
                 answer_server_tail=server_public_tail([], []), answer_A=answer_leak(invA_a), answer_B=None)]
    r0 = rows[0]
    print(f"   no defence: prompt A {r0['prompt_A']['content_leak']:.3f}, answer via public tail "
          f"{r0['answer_server_tail']['content_leak']:.3f}, answer A {r0['answer_A']['content_leak']:.3f}", flush=True)

    for k, (lp, la) in enumerate(CFG["configs"]):
        print(f"\n2.{k+1}) secret head + secret tail, λ_prompt = {lp:g}, λ_answer = {la:g}", flush=True)
        head = add_adapters(model, range(0, H), seed=2000 + 2 * k, dev=dev)
        tail = add_adapters(model, range(n_layers - T, n_layers), seed=2001 + 2 * k, dev=dev)
        train_defender(model, dev, head, tail, train_ids, E, H, LT, lp, la)
        model.eval()
        q = quality(model, dev, head + tail, test_ids)
        set_on(head + tail, True)
        pA, aA = prompt_leak(invA_p), answer_leak(invA_a)
        aS = server_public_tail(head, tail)
        invB_p = train_inverter(model, dev, train_ids, E, H, "input", "B prompt")
        invB_a = train_inverter(model, dev, train_ids, E, LT, "output", "B answer")
        pB, aB = prompt_leak(invB_p), answer_leak(invB_a)
        rows.append(dict(config=f"λp={lp:g}, λa={la:g}", lam_p=lp, lam_a=la, **q, prompt_A=pA, prompt_B=pB,
                         answer_server_tail=aS, answer_A=aA, answer_B=aB))
        print(f"   quality {q['agreement']:.3f} (ppl x{q['ppl_ratio']:.2f}) | prompt: A {pA['content_leak']:.3f} "
              f"(private exact {pA['private_exact']:.3f}), B {pB['content_leak']:.3f} | answer: server public tail "
              f"{aS['content_leak']:.3f}, A {aA['content_leak']:.3f}, B {aB['content_leak']:.3f}", flush=True)
        remove_all_adapters(model); del invB_p, invB_a
        if dev == "cuda":
            torch.cuda.empty_cache()

    d = rows[1:]
    checks = [
        ("no defence: prompt attacker A > 0.9", r0["prompt_A"]["content_leak"] > 0.9, f"{r0['prompt_A']['content_leak']:.3f}"),
        ("no defence: server running the public tail recovers the answer (> 0.99)",
         r0["answer_server_tail"]["content_leak"] > 0.99, f"{r0['answer_server_tail']['content_leak']:.3f}"),
        ("every defended model keeps agreement > 0.5", all(r["agreement"] > 0.5 for r in d),
         ", ".join(f"{r['config']}: {r['agreement']:.3f}" for r in d)),
        ("attackers with the secret leak at least as much as without (prompt and answer)",
         all(r["prompt_B"]["content_leak"] >= r["prompt_A"]["content_leak"] - 0.02 and
             r["answer_B"]["content_leak"] >= r["answer_A"]["content_leak"] - 0.02 for r in d), ""),
    ]
    print("\nSANITY CHECKS")
    for nm, ok, info in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {nm}  {('(' + info + ')') if info else ''}")

    with open(os.path.join(CFG["out_dir"], "results.json"), "w") as f:
        json.dump(dict(config=CFG, smoke=a.smoke, rows=rows,
                       sanity_checks=[dict(name=n_, passed=bool(o), info=i) for n_, o, i in checks]), f, indent=1)
    with open(os.path.join(CFG["out_dir"], "secret_ushape_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "lambda_prompt", "lambda_answer", "H", "T", "agreement", "ppl_ratio",
                    "prompt_leak_no_secret", "prompt_private_exact_no_secret", "prompt_leak_with_secret",
                    "prompt_private_exact_with_secret", "answer_leak_server_public_tail",
                    "answer_leak_no_secret", "answer_leak_with_secret"])
        for r in rows:
            g = lambda x, k: "" if x is None else round(x[k], 4)
            w.writerow([r["config"], "" if r["lam_p"] is None else r["lam_p"], "" if r["lam_a"] is None else r["lam_a"],
                        H, T, round(r["agreement"], 4), round(r["ppl_ratio"], 3),
                        g(r["prompt_A"], "content_leak"), g(r["prompt_A"], "private_exact"),
                        g(r["prompt_B"], "content_leak"), g(r["prompt_B"], "private_exact"),
                        g(r["answer_server_tail"], "content_leak"), g(r["answer_A"], "content_leak"),
                        g(r["answer_B"], "content_leak")])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [("no defence" if r["lam_p"] is None else f"λp={r['lam_p']:g}\nλa={r['lam_a']:g}")
              + f"\nquality {100*r['agreement']:.0f}%" for r in rows]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    nanv = lambda r, k: np.nan if r[k] is None else 100 * r[k]["content_leak"]
    ax[0].bar(x - w / 2, [nanv(r, "prompt_A") for r in rows], w, color="#1a9e55", label="attacker without the secret")
    ax[0].bar(x + w / 2, [nanv(r, "prompt_B") for r in rows], w, color="#7a3cc9", label="attacker with the secret")
    ax[0].set_title(f"PROMPT: what the client sends after H={H} layers", fontsize=10.5)
    ax[1].bar(x - w, [nanv(r, "answer_server_tail") for r in rows], w, color="#e0533d", label="server runs the public tail")
    ax[1].bar(x, [nanv(r, "answer_A") for r in rows], w, color="#1a9e55", label="attacker without the secret")
    ax[1].bar(x + w, [nanv(r, "answer_B") for r in rows], w, color="#7a3cc9", label="attacker with the secret")
    ax[1].set_title(f"ANSWER: what the server sends back to the client's tail (T={T})", fontsize=10.5)
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(labels, fontsize=8.5); a_.set_ylim(0, 105); a_.grid(alpha=.3, axis="y")
        a_.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax[0].set_ylabel("Content tokens recovered by the server [%]")
    fig.suptitle(f"Secret head + secret tail (U-shaped split): "
                 f"{'TINY RANDOM MODEL (smoke test)' if a.smoke else CFG['model_id']}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(CFG["out_dir"], "secret_ushape.png"), dpi=200)
    print(f"\nSaved to {CFG['out_dir']}/  (results.json, secret_ushape_results.csv, secret_ushape.png)")


if __name__ == "__main__":
    main()
