"""
SECRET HEAD - can privately fine-tuned client layers hide the prompt while keeping the answers?
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Standalone script. Model: Llama-3.2-1B-Instruct, U-shaped split with H head layers on the client.

IDEA
  The client adds SECRET low-rank adapters (LoRA, random secret start) to its H head layers and
  trains them so that
    (a) the full model still gives the same answers as the public model   -> KL(public || defended)
    (b) an inverter watching what the client sends cannot tell the tokens  -> push the inverter's
        prediction towards uniform (adversarial term, weight lambda)
  The server keeps running the PUBLIC weights of its U layers; only the client's adapters are secret.
  lambda = 0 is "secret random fine-tune only" (no adversary during training).

TWO ATTACKERS (same MLP inverter and metrics as inversion_attack.py)
  A. no secret: trains on the PUBLIC model's hidden states (all it can generate itself) and is
     tested on what the defended client actually sends          -> realistic if the adapters stay secret
  B. secret leaked (worst case): trains on the DEFENDED model's hidden states -> upper bound on leakage

QUALITY: next-token agreement with the clean public model, and perplexity ratio, on held-out text.

SANITY CHECKS (PASS / FAIL)
  1. no defence: attacker A on the public model leaks > 0.9 at this H (matches the first attack)
  2. every defended model keeps agreement > 0.5 (the function is not destroyed)
  3. attacker B leaks at least as much as attacker A (knowing the weights cannot hurt the attacker)

HOW TO RUN ON GOOGLE COLAB (T4 GPU; about 40-70 min with the default grid)
  1. Runtime -> Change runtime type -> T4 GPU, upload this file
  2. from huggingface_hub import notebook_login; notebook_login()
  3. !pip install -q datasets
  4. !python secret_head.py                       (default: H = 4, lambda = 0, 0.3, 1, 3)
     smaller:  !python secret_head.py --lam 0 1 --steps 300
  5. Download secret_results/ and send it back.
  Quick code test without the real model (CPU):  python secret_head.py --smoke
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
    H                  = 4,                     # head layers on the client
    lam_list           = [0.0, 0.3, 1.0, 3.0],  # weight of the adversarial term
    lora_rank          = 32,
    lora_alpha         = 32,
    secret_init        = 0.02,                  # size of the secret random start of the adapters
    def_steps          = 600,                   # defender training steps per lambda
    def_batch          = 4,
    def_lr             = 2e-4,
    inv_lr             = 1e-3,
    train_tokens       = 400_000,               # attacker training text (as in the first attack)
    test_tokens        = 40_000,
    n_sensitive        = 400,
    seq_len            = 128,
    batch_seqs         = 8,
    positions_per_step = 384,
    epochs             = 2,
    hidden_mlp         = 2048,
    bootstrap          = 300,
    seed               = 0,
    out_dir            = "secret_results",
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
# Secret LoRA adapters on the client's head layers
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


def add_secret_adapters(model, H, seed, dev):
    gen = torch.Generator().manual_seed(seed)
    mods = []
    for layer in model.model.layers[:H]:
        for parent, nm in PROJ:
            par = getattr(layer, parent)
            lora = LoRALinear(getattr(par, nm), CFG["lora_rank"], CFG["lora_alpha"], CFG["secret_init"], gen)
            lora.A.data = lora.A.data.to(dev); lora.B.data = lora.B.data.to(dev)
            setattr(par, nm, lora); mods.append(lora)
    return mods


def remove_adapters(model, H):
    for layer in model.model.layers[:H]:
        for parent, nm in PROJ:
            par = getattr(layer, parent)
            m = getattr(par, nm)
            if isinstance(m, LoRALinear):
                setattr(par, nm, m.base)


def set_enabled(mods, on):
    for m in mods:
        m.enabled = on


class Catcher:
    def __init__(self, model, L, detach=True):
        self.state, self.detach = None, detach
        self.h = model.model.layers[L - 1].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        self.state = x.detach() if self.detach else x
    def remove(self):
        self.h.remove()


# ----------------------------------------------------------------------------------
# Inverter (attacker)
# ----------------------------------------------------------------------------------
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


def train_inverter(model, dev, ids_all, E, H, label):
    """Attacker training: inverter on whatever the model (as currently configured) sends at layer H."""
    d = E.shape[1]
    inv = InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
    opt = torch.optim.AdamW(inv.parameters(), lr=CFG["inv_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cat = Catcher(model, H)
    n = ids_all.shape[0]; spe = math.ceil(n / CFG["batch_seqs"]); total = spe * CFG["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["inv_lr"], total_steps=total, pct_start=0.1)
    t0, step = time.time(), 0
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n)
        for b in range(spe):
            idx = perm[b * CFG["batch_seqs"]:(b + 1) * CFG["batch_seqs"]]
            if len(idx) == 0:
                continue
            ids = ids_all[idx]
            with torch.no_grad():
                model(input_ids=ids.to(dev))
            B, S = ids.shape
            flat = sample_positions(B, S, CFG["positions_per_step"], dev)
            x = cat.state.reshape(B * S, -1)[flat].float(); y = ids.to(dev).reshape(-1)[flat]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = inv(x, E)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); step += 1
        print(f"   [{label}] epoch {ep+1}/{CFG['epochs']}  loss {loss.item():.3f}  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    cat.remove()
    return inv.eval()


@torch.no_grad()
def inverter_predict(model, dev, inv, ids_all, E, H):
    cat = Catcher(model, H); preds = []
    for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
        ids = ids_all[b:b + CFG["batch_seqs"]]
        model(input_ids=ids.to(dev))
        B, S = ids.shape
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            p = inv(cat.state.reshape(B * S, -1).float(), E).argmax(-1)
        preds.append(p.reshape(B, S).cpu())
    cat.remove()
    return torch.cat(preds).numpy()


# ----------------------------------------------------------------------------------
# Defender training (client's secret adapters)
# ----------------------------------------------------------------------------------
def train_defender(model, dev, mods, train_ids, E, H, lam):
    d = E.shape[1]
    params = [p for m in mods for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=CFG["def_lr"], weight_decay=0.0)
    inv = InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev)
    opt_inv = torch.optim.AdamW(inv.parameters(), lr=CFG["inv_lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_inv = torch.amp.GradScaler("cuda", enabled=use_amp)
    cat = Catcher(model, H, detach=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    n = train_ids.shape[0]; t0 = time.time()
    for step in range(1, CFG["def_steps"] + 1):
        ids = train_ids[torch.randint(0, n, (CFG["def_batch"],))].to(dev)
        B, S = ids.shape
        with torch.no_grad():                                   # public model = teacher
            set_enabled(mods, False)
            t_logp = F.log_softmax(model(input_ids=ids).logits.float(), -1)
            set_enabled(mods, True)
        s_logits = model(input_ids=ids).logits.float()          # defended model
        kl = F.kl_div(F.log_softmax(s_logits, -1), t_logp, log_target=True, reduction="none").sum(-1)[:, 1:].mean()
        loss = kl
        h = cat.state
        flat = sample_positions(B, S, min(CFG["positions_per_step"], B * (S - 1)), dev)
        hs = h.reshape(B * S, -1)[flat].float()
        if lam > 0:
            for p in inv.parameters():
                p.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                lg = inv(hs, E)
            pr = F.log_softmax(lg.float(), -1)
            neg_entropy = (pr.exp() * pr).sum(-1).mean()          # minimise -> inverter becomes unsure
            loss = kl + lam * neg_entropy
            for p in inv.parameters():
                p.requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        # inverter keeps adapting to the current defence (min-max)
        y = ids.reshape(-1)[flat]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            li = F.cross_entropy(inv(hs.detach(), E).float(), y)
        opt_inv.zero_grad(set_to_none=True); scaler_inv.scale(li).backward(); scaler_inv.step(opt_inv)
        scaler_inv.update()
        if step % 50 == 0 or step == CFG["def_steps"]:
            print(f"   [defender lambda={lam:g}] step {step}/{CFG['def_steps']}  KL {kl.item():.4f}  "
                  f"inverter CE {li.item():.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
    cat.remove()
    for p in params:
        p.requires_grad_(False)


@torch.no_grad()
def quality(model, dev, mods, ids_all):
    def run(on):
        set_enabled(mods, on)
        preds, nll, n = [], 0.0, 0
        for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
            ids = ids_all[b:b + CFG["batch_seqs"]].to(dev)
            logits = model(input_ids=ids).logits.float()
            preds.append(logits.argmax(-1).cpu())
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            nll += -lp.sum().item(); n += lp.numel()
        return torch.cat(preds), math.exp(nll / n)
    p0, ppl0 = run(False)
    p1, ppl1 = run(True)
    set_enabled(mods, True)
    return dict(agreement=float((p0 == p1).float().mean()), ppl=ppl1, ppl_clean=ppl0, ppl_ratio=ppl1 / ppl0)


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
    ap.add_argument("--H", type=int)
    ap.add_argument("--lam", type=float, nargs="+")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    if a.smoke:
        CFG.update(H=2, lam_list=[0.0, 1.0], lora_rank=8, lora_alpha=8, def_steps=60, def_batch=4,
                   train_tokens=40_000, test_tokens=4_000, n_sensitive=40, seq_len=64,
                   positions_per_step=128, epochs=2, bootstrap=30, out_dir="secret_results_smoke")
    if a.H: CFG["H"] = a.H
    if a.lam: CFG["lam_list"] = a.lam
    if a.steps: CFG["def_steps"] = a.steps
    if a.out: CFG["out_dir"] = a.out
    H = CFG["H"]
    set_seed(CFG["seed"]); rng_py = random.Random(CFG["seed"]); rng_np = np.random.default_rng(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)
    print(f"H = {H}, lambda = {CFG['lam_list']}, defender steps = {CFG['def_steps']}", flush=True)

    train_txt, test_txt = load_general_text(a.smoke, rng_py)
    sens_prompts = make_sensitive_prompts(CFG["n_sensitive"], rng_py)
    model, tok, dev = load_model(a.smoke, train_txt + test_txt + [t for t, _ in sens_prompts])
    n_layers = len(model.model.layers)
    assert 1 <= H < n_layers
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    train_ids = chunk(tok, train_txt, CFG["seq_len"], CFG["train_tokens"])
    test_ids = chunk(tok, test_txt, CFG["seq_len"], CFG["test_tokens"])
    sens_ids, sens_mask, sens_valid = encode_sensitive(tok, sens_prompts, CFG["seq_len"])
    tables = token_tables(tok, train_ids)
    tv = np.ones(test_ids.shape, dtype=bool); tv[:, 0] = False
    print(f"device {dev}, {n_layers} layers; train {train_ids.numel()} tokens, test {test_ids.numel()}", flush=True)

    def evaluate(inv):
        g = leak_metrics(inverter_predict(model, dev, inv, test_ids, E, H), test_ids.numpy(), tv, tables, rng_np)
        s = leak_metrics(inverter_predict(model, dev, inv, sens_ids, E, H), sens_ids.numpy(),
                         sens_valid.numpy(), tables, rng_np, sens=sens_mask.numpy())
        return dict(content_leak=g["content_leak"], weighted_top1=g["weighted_top1"],
                    sensitive_top1=s["sensitive_top1"], sensitive_exact=s["sensitive_exact"])

    # Attacker A is trained once on the PUBLIC model (it never sees the secret adapters)
    print("\n1) attacker A: trained on the public model", flush=True)
    inv_A = train_inverter(model, dev, train_ids, E, H, "attacker A")
    rows = [dict(defence="none (public model)", lam=None,
                 attacker_A=evaluate(inv_A), attacker_B=None,
                 agreement=1.0, ppl_ratio=1.0)]
    print(f"   no defence: attacker A leak {rows[0]['attacker_A']['content_leak'][0]:.3f}", flush=True)

    for k, lam in enumerate(CFG["lam_list"]):
        print(f"\n2.{k+1}) secret adapters, lambda = {lam:g}", flush=True)
        mods = add_secret_adapters(model, H, seed=1000 + k, dev=dev)
        train_defender(model, dev, mods, train_ids, E, H, lam)
        model.eval()
        q = quality(model, dev, mods, test_ids)
        rA = evaluate(inv_A)
        inv_B = train_inverter(model, dev, train_ids, E, H, f"attacker B, lambda={lam:g}")
        rB = evaluate(inv_B)
        rows.append(dict(defence="secret adapters", lam=lam, attacker_A=rA, attacker_B=rB, **q))
        print(f"   lambda={lam:g}: agreement {q['agreement']:.3f}, ppl x{q['ppl_ratio']:.2f} | "
              f"attacker A leak {rA['content_leak'][0]:.3f} (private exact {rA['sensitive_exact']:.3f}) | "
              f"attacker B leak {rB['content_leak'][0]:.3f} (private exact {rB['sensitive_exact']:.3f})", flush=True)
        remove_adapters(model, H)
        del inv_B
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- sanity checks ----
    d = rows[1:]
    checks = [
        ("no defence: attacker A leaks > 0.9 (matches the first attack)", rows[0]["attacker_A"]["content_leak"][0] > 0.9,
         f"{rows[0]['attacker_A']['content_leak'][0]:.3f}"),
        ("every defended model keeps agreement > 0.5", all(r["agreement"] > 0.5 for r in d),
         ", ".join(f"lambda={r['lam']:g}: {r['agreement']:.3f}" for r in d)),
        ("attacker B (knows weights) leaks at least as much as attacker A",
         all(r["attacker_B"]["content_leak"][0] >= r["attacker_A"]["content_leak"][0] - 0.02 for r in d), ""),
    ]
    print("\nSANITY CHECKS")
    for n_, ok, info in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {n_}  {('(' + info + ')') if info else ''}")

    # ---- save ----
    with open(os.path.join(CFG["out_dir"], "results.json"), "w") as f:
        json.dump(dict(config=CFG, smoke=a.smoke, rows=rows,
                       sanity_checks=[dict(name=n_, passed=bool(o), info=i) for n_, o, i in checks]), f, indent=1)
    with open(os.path.join(CFG["out_dir"], "secret_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defence", "lambda", "H", "agreement", "ppl_ratio",
                    "A_content_leak", "A_low", "A_high", "A_sensitive_top1", "A_sensitive_exact",
                    "B_content_leak", "B_low", "B_high", "B_sensitive_top1", "B_sensitive_exact"])
        for r in rows:
            A, B = r["attacker_A"], r["attacker_B"]
            w.writerow([r["defence"], "" if r["lam"] is None else r["lam"], H, round(r["agreement"], 4),
                        round(r["ppl_ratio"], 3), *[round(v, 4) for v in A["content_leak"]],
                        round(A["sensitive_top1"], 4), round(A["sensitive_exact"], 4),
                        *([round(v, 4) for v in B["content_leak"]] + [round(B["sensitive_top1"], 4),
                          round(B["sensitive_exact"], 4)] if B else [""] * 5)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.scatter([100], [100 * rows[0]["attacker_A"]["content_leak"][0]], s=90, color="#e0533d", zorder=4,
               label="No defence (public model)")
    xs = [100 * r["agreement"] for r in d]
    ax.plot(xs, [100 * r["attacker_A"]["content_leak"][0] for r in d], "-o", color="#1a9e55", lw=2,
            label="Secret adapters, attacker WITHOUT the secret")
    ax.plot(xs, [100 * r["attacker_B"]["content_leak"][0] for r in d], "--s", color="#7a3cc9", lw=2,
            label="Secret adapters, attacker WITH the secret (worst case)")
    for r, x in zip(d, xs):
        ax.annotate(f"λ={r['lam']:g}", (x, 100 * r["attacker_A"]["content_leak"][0]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8.5, color="#1a9e55")
    ax.set_xlabel("Answer quality: next token same as public model [%]")
    ax.set_ylabel("Content tokens recovered by the server [%]")
    ax.set_ylim(-2, 102); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(f"Secret client adapters at H = {H}: privacy vs quality\n"
                 f"{'TINY RANDOM MODEL (smoke test)' if a.smoke else CFG['model_id']}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(CFG["out_dir"], "secret_tradeoff.png"), dpi=200)
    print(f"\nSaved to {CFG['out_dir']}/  (results.json, secret_results.csv, secret_tradeoff.png)")


if __name__ == "__main__":
    main()
