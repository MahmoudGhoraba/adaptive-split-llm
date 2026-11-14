"""
INVERSION ATTACK - measures how much a curious server learns at every split
Adaptive Edge-Server Distributed LLM Inference with Privacy-Preserving Model Splitting

Replaces the PLACEHOLDER leakage curves of Step 2 / Step 3 with measured values for
Llama-3.2-1B-Instruct.

WHAT IS ATTACKED (U-shaped split, see Step 2)
  in-side  (H = 1..15): the server receives the output of layer H for every token and tries to
                        recover WHICH TOKEN it was                  -> ASR_in(H)
  out-side (T = 0..14): the server sends back the output of layer 16-T and tries to recover
                        WHICH TOKEN THE MODEL WILL OUTPUT next       -> ASR_out(T)
                        (assumes the tail weights are private; with public weights the server
                        could simply run the tail itself)

ATTACKER (MLP inversion, as in Cunningham 2026)
  For each layer, a small network  LayerNorm -> Linear -> GELU -> Linear  maps a hidden state to
  a vector that is scored against the public embedding table (Llama-3.2-1B ties input and output
  embeddings), so the attacker can name any of the 128,256 tokens. It is trained on public text
  (WikiText-2 train) and tested on text it never saw (WikiText-2 test) plus prompts with planted
  private details (names, hospitals, diseases, phone numbers, amounts, cities).

METRICS (all top-1, per side and layer)
  all_top1         every token; only for comparison with Cunningham (inflated by "the", "is", ...)
  content_top1     only content tokens (no stopwords, no punctuation)
  content_leak     MAIN METRIC = (content_top1 - blind) / (1 - blind), where "blind" is an attacker
                   that ignores the hidden state and always guesses the most frequent content token
                   -> 0 = learns nothing, 1 = recovers everything. This goes into Step 2 / Step 3.
  weighted_top1    every token weighted by its rarity (-log2 frequency): "hospital" >> "is"
  sensitive_top1   only the tokens of the planted private details
  sensitive_exact  share of planted details (e.g. a full phone number) recovered completely
  95% confidence intervals by bootstrap over sentences.

SANITY CHECKS (printed as PASS / FAIL)
  1. layer 0 (raw embeddings) must be almost fully recoverable   (content_top1 > 0.9)
  2. an attacker trained on random noise must not beat the blind guess
  3. (info) is leakage lower deep in the model than at layer 1? A "no" on the real model is a
     finding (deep layers still leak, Dong et al. 2025), not an error
  4. out-side at T = 0 should be close to 100% (the server holds the final hidden state)

HOW TO RUN ON GOOGLE COLAB (free T4 GPU is enough, about 25-35 min)
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload this file (folder icon on the left -> upload)
  3. In a cell:   from huggingface_hub import notebook_login; notebook_login()
     and paste your Hugging Face token (Settings -> Access Tokens on huggingface.co;
     your account must have accepted the Llama-3.2 licence).
  4. In a cell:   !pip install -q datasets
  5. In a cell:   !python inversion_attack.py
  6. Download the folder attack_results/ (results.json, attack_results.csv, plots) and send it back.

  Quick test without the real model (CPU, 1 min):   python inversion_attack.py --smoke
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

# ----------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------
CFG = dict(
    model_id          = "meta-llama/Llama-3.2-1B-Instruct",
    train_tokens      = 400_000,    # attacker training data (public text)
    test_tokens       = 40_000,     # held-out general text
    n_sensitive       = 400,        # prompts with planted private details
    seq_len           = 128,
    batch_seqs        = 8,          # sequences per LLM forward pass
    positions_per_step= 384,        # token positions used per training step (speed)
    epochs            = 2,
    lr                = 1e-3,
    hidden_mlp        = 2048,
    bootstrap         = 300,
    seed              = 0,
    out_dir           = "attack_results",
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
# Tokenizer adapter (same interface for the real Llama tokenizer and the smoke-test one)
# ----------------------------------------------------------------------------------
class HFTok:
    def __init__(self, tok):
        self.tok = tok
        self.bos_id = tok.bos_token_id
        self.vocab_size = len(tok)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode_offsets(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def decode_one(self, i):
        return self.tok.decode([i])


class WordTok:
    """Tiny word-level tokenizer for --smoke (no download needed)."""
    def __init__(self, texts):
        words = sorted(set(w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)))
        self.itos = ["<bos>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.bos_id = 0
        self.vocab_size = len(self.itos)

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
# Data
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
    """Returns list of (text, [(start, end, category), ...]) with planted private details."""
    out = []
    for _ in range(n):
        tpl = rng.choice(TEMPLATES)
        fill = dict(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", hospital=rng.choice(HOSPITALS),
                    disease=rng.choice(DISEASES), city=rng.choice(CITIES),
                    phone="01" + "".join(str(rng.randint(0, 9)) for _ in range(9)),
                    amount=str(rng.randint(2, 99) * 500))
        text, spans, pos = "", [], 0
        for part in re.split(r"(\{\w+\})", tpl):
            m = re.fullmatch(r"\{(\w+)\}", part)
            if m:
                val = fill[m.group(1)]
                spans.append((len(text), len(text) + len(val), m.group(1)))
                text += val
            else:
                text += part
        out.append((text, spans))
    return out


def load_general_text(smoke, rng):
    if smoke:
        words = ("the patient doctor city river history music game people school water light time year "
                 "is was are and of to in a with for on by music king war army team song film road").split()
        def para():
            return " ".join(rng.choice(words) for _ in range(rng.randint(20, 60))) + " ."
        return [para() for _ in range(1500)], [para() for _ in range(200)]
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    def clean(split):
        lines = [l.strip() for l in ds[split]["text"]]
        return [l for l in lines if len(l) > 40 and not l.startswith("=")]
    return clean("train"), clean("test")


def chunk(tok, texts, seq_len, max_tokens):
    """Concatenate texts into sequences of seq_len tokens, each starting with BOS."""
    ids = []
    for t in texts:
        ids.extend(tok.encode(" " + t))
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    body = seq_len - 1
    n = len(ids) // body
    seqs = [[tok.bos_id] + ids[i * body:(i + 1) * body] for i in range(n)]
    return torch.tensor(seqs, dtype=torch.long)


def encode_sensitive(tok, prompts, seq_len):
    """Each prompt as its own sequence (padded with BOS, padding masked out)."""
    seqs, sens, valid = [], [], []
    for text, spans in prompts:
        ids, offs = tok.encode_offsets(text)
        ids, offs = ids[:seq_len - 1], offs[:seq_len - 1]
        s = [0] * len(ids)
        for k, (a, b) in enumerate(offs):
            for j, (sa, sb, _) in enumerate(spans):
                if a < sb and b > sa:
                    s[k] = j + 1                      # entity number (1-based), 0 = not sensitive
        pad = seq_len - 1 - len(ids)
        seqs.append([tok.bos_id] + ids + [tok.bos_id] * pad)
        sens.append([0] + s + [0] * pad)
        valid.append([0] + [1] * len(ids) + [0] * pad)
    return (torch.tensor(seqs), torch.tensor(sens), torch.tensor(valid, dtype=torch.bool))


# ----------------------------------------------------------------------------------
# Model and hidden-state capture
# ----------------------------------------------------------------------------------
def load_model(smoke, tok_texts=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if smoke:
        tok = WordTok(tok_texts)
        cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          tie_word_embeddings=True, max_position_embeddings=256)
        model = LlamaForCausalLM(cfg).to(dev).eval()
        # give the toy model a little structure so the check "deeper = harder" is meaningful
        return model, tok, dev
    hf_tok = AutoTokenizer.from_pretrained(CFG["model_id"])
    dtype = torch.float16 if dev == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], dtype=dtype)
    except TypeError:                                   # older transformers
        model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], torch_dtype=dtype)
    model = model.to(dev).eval()
    return model, HFTok(hf_tok), dev


class Catcher:
    """Hooks: states[0] = embedding output, states[L] = output of decoder layer L (before final norm)."""
    def __init__(self, model):
        self.states = {}
        base = model.model
        self.n_layers = len(base.layers)
        self.h = [base.embed_tokens.register_forward_hook(self._hook(0))]
        for i, layer in enumerate(base.layers):
            self.h.append(layer.register_forward_hook(self._hook(i + 1)))

    def _hook(self, idx):
        def f(mod, inp, out):
            self.states[idx] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return f


@torch.no_grad()
def run_llm(model, catcher, ids, dev):
    """Returns hidden states {layer: [B,S,D]} and the model's greedy next-token prediction [B,S]."""
    catcher.states = {}
    logits = model(input_ids=ids.to(dev)).logits
    return dict(catcher.states), logits.argmax(-1)


# ----------------------------------------------------------------------------------
# Attacker
# ----------------------------------------------------------------------------------
class InvHead(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, E):
        z = self.net(h)
        return self.scale * (z @ E.t())          # scores against every vocabulary token


def head_list(n_layers):
    heads = [("in", L) for L in range(0, n_layers)]              # L=0 is the sanity check
    heads += [("out", L) for L in range(2, n_layers + 1)]        # T = n_layers - L = 0 .. n-2
    heads += [("noise", 1)]
    return heads


def features(states, side, L, noise_std):
    h = states[L]
    if side == "noise":
        return torch.randn_like(h, dtype=torch.float32) * noise_std
    return h.float()


def train_attackers(model, tok, dev, train_ids, n_layers, E):
    d = E.shape[1]
    heads = {k: InvHead(d, CFG["hidden_mlp"] if d >= 512 else 4 * d).to(dev) for k in head_list(n_layers)}
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    catcher = Catcher(model)
    n_seq = train_ids.shape[0]
    steps_per_epoch = math.ceil(n_seq / CFG["batch_seqs"])
    total = steps_per_epoch * CFG["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=total, pct_start=0.1)
    noise_std = None
    t0, step = time.time(), 0
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n_seq)
        for b in range(steps_per_epoch):
            idx = perm[b * CFG["batch_seqs"]:(b + 1) * CFG["batch_seqs"]]
            if len(idx) == 0:
                continue
            ids = train_ids[idx]
            states, pred = run_llm(model, catcher, ids, dev)
            if noise_std is None:
                noise_std = states[1].float().std().item()
            B, S = ids.shape
            pos = torch.randperm(B * (S - 1), device=dev)[:CFG["positions_per_step"]]
            # skip the BOS position 0
            flat = (pos // (S - 1)) * S + (pos % (S - 1)) + 1
            y_in = ids.to(dev).reshape(-1)[flat]
            y_out = pred.reshape(-1)[flat]
            opt.zero_grad(set_to_none=True)
            losses = []
            for (side, L), head in heads.items():
                x = features(states, side, L, noise_std).reshape(B * S, -1)[flat]
                y = y_out if side == "out" else y_in
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = head(x, E)
                loss = F.cross_entropy(logits.float(), y)
                scaler.scale(loss).backward()
                losses.append(loss.item())
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                print(f"  step {step}/{total}  mean loss {np.mean(losses):.3f}  "
                      f"elapsed {el/60:.1f} min  eta {el/step*(total-step)/60:.1f} min", flush=True)
    for h in catcher.h:
        h.remove()
    return heads, noise_std


@torch.no_grad()
def predict(model, dev, heads, ids_all, n_layers, E, noise_std):
    """Top-1 prediction of every head at every position. Returns dict head -> [N,S] and labels."""
    catcher = Catcher(model)
    preds = {k: [] for k in heads}
    y_out_all = []
    for b in range(0, ids_all.shape[0], CFG["batch_seqs"]):
        ids = ids_all[b:b + CFG["batch_seqs"]]
        states, pred = run_llm(model, catcher, ids, dev)
        y_out_all.append(pred.cpu())
        B, S = ids.shape
        for (side, L), head in heads.items():
            x = features(states, side, L, noise_std).reshape(B * S, -1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = head(x, E).argmax(-1)
            preds[(side, L)].append(p.reshape(B, S).cpu())
    for h in catcher.h:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in preds.items()}, torch.cat(y_out_all).numpy()


# ----------------------------------------------------------------------------------
# Metrics
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
    p = (counts + 1) / (counts.sum() + V)
    surprisal = -np.log2(p)
    blind_all = int(np.argmax(counts))
    blind_content = int(np.argmax(np.where(is_content, counts, -1)))
    return is_content, surprisal, blind_all, blind_content


def ratio_ci(num, den, seq, n_boot, rng):
    """Bootstrap over sentences for sum(num)/sum(den). Returns (value, low, high)."""
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    S = seq.max() + 1
    nps = np.bincount(seq, weights=num, minlength=S)
    dps = np.bincount(seq, weights=den, minlength=S)
    val = nps.sum() / dps.sum()
    bs = []
    for _ in range(n_boot):
        r = rng.integers(0, S, S)
        d = dps[r].sum()
        if d > 0:
            bs.append(nps[r].sum() / d)
    return float(val), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def metrics_for(pred, label, valid, tables, rng, sens=None):
    is_content, surprisal, blind_all, blind_content = tables
    N, S = label.shape
    seq = np.repeat(np.arange(N), S).reshape(N, S)
    m = valid
    corr = (pred == label)
    cm = m & is_content[label]
    w = surprisal[label] * m
    out = {}
    out["all_top1"] = ratio_ci(corr[m].astype(float), np.ones(m.sum()), seq[m], CFG["bootstrap"], rng)
    c = ratio_ci(corr[cm].astype(float), np.ones(cm.sum()), seq[cm], CFG["bootstrap"], rng)
    out["content_top1"] = c
    blind = float((label[cm] == blind_content).mean()) if cm.sum() else 0.0
    out["blind_content_top1"] = blind
    norm = lambda v: float(np.clip((v - blind) / (1 - blind), 0, 1)) if not math.isnan(v) else v
    out["content_leak"] = tuple(norm(v) for v in c)
    out["weighted_top1"] = ratio_ci((corr * w)[m], w[m], seq[m], CFG["bootstrap"], rng)
    out["n_tokens"] = int(m.sum()); out["n_content"] = int(cm.sum())
    if sens is not None:
        sm = m & (sens > 0)
        out["sensitive_top1"] = ratio_ci(corr[sm].astype(float), np.ones(sm.sum()), seq[sm],
                                         CFG["bootstrap"], rng)
        ok, tot = 0, 0
        for i in range(N):
            for e in set(sens[i][sens[i] > 0].tolist()):
                pos = (sens[i] == e) & m[i]
                if pos.any():
                    tot += 1; ok += int(corr[i][pos].all())
        out["sensitive_exact"] = ok / tot if tot else float("nan")
        out["n_sensitive"] = int(sm.sum())
    return out


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny random model, no download (tests the code)")
    ap.add_argument("--train-tokens", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    if a.train_tokens: CFG["train_tokens"] = a.train_tokens
    if a.epochs: CFG["epochs"] = a.epochs
    if a.out: CFG["out_dir"] = a.out
    if a.smoke:
        CFG.update(train_tokens=60_000, test_tokens=6_000, n_sensitive=60, seq_len=64,
                   positions_per_step=256, epochs=3, bootstrap=50, out_dir=a.out or "attack_results_smoke")
    set_seed(CFG["seed"])
    rng_py = random.Random(CFG["seed"])
    rng_np = np.random.default_rng(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)

    print("1/5 loading data", flush=True)
    train_txt, test_txt = load_general_text(a.smoke, rng_py)
    sens_prompts = make_sensitive_prompts(CFG["n_sensitive"], rng_py)

    print("2/5 loading model", flush=True)
    model, tok, dev = load_model(a.smoke, train_txt + test_txt + [t for t, _ in sens_prompts])
    n_layers = len(model.model.layers)
    E = model.get_input_embeddings().weight.detach()
    E = E.half() if dev == "cuda" else E.float()
    print(f"   device {dev}, {n_layers} layers, vocab {tok.vocab_size}", flush=True)

    train_ids = chunk(tok, train_txt, CFG["seq_len"], CFG["train_tokens"])
    test_ids = chunk(tok, test_txt, CFG["seq_len"], CFG["test_tokens"])
    sens_ids, sens_mask, sens_valid = encode_sensitive(tok, sens_prompts, CFG["seq_len"])
    print(f"   train {train_ids.numel()} tokens, test {test_ids.numel()}, "
          f"sensitive prompts {len(sens_prompts)}", flush=True)
    tables = token_tables(tok, train_ids)

    print("3/5 training attackers (one per layer and side, all at once)", flush=True)
    heads, noise_std = train_attackers(model, tok, dev, train_ids, n_layers, E)
    for h in heads.values():
        h.eval()

    print("4/5 attacking held-out text and sensitive prompts", flush=True)
    p_test, yout_test = predict(model, dev, heads, test_ids, n_layers, E, noise_std)
    p_sens, yout_sens = predict(model, dev, heads, sens_ids, n_layers, E, noise_std)
    test_lab_in = test_ids.numpy()
    test_valid = np.ones_like(test_lab_in, dtype=bool); test_valid[:, 0] = False
    sens_lab_in = sens_ids.numpy()
    sv = sens_valid.numpy(); sm = sens_mask.numpy()
    # out-side on sensitive prompts: positions whose NEXT token is a planted detail
    sm_next = np.zeros_like(sm); sm_next[:, :-1] = sm[:, 1:]
    sv_next = sv.copy(); sv_next[:, -1] = False; sv_next &= np.roll(sv, -1, axis=1)

    print("5/5 metrics", flush=True)
    rows = []
    for (side, L) in heads:
        if side == "out":
            g = metrics_for(p_test[(side, L)], yout_test, test_valid, tables, rng_np)
            s = metrics_for(p_sens[(side, L)], yout_sens, sv_next, tables, rng_np, sens=sm_next)
            split_k, split_v = "T", n_layers - L
        else:
            g = metrics_for(p_test[(side, L)], test_lab_in, test_valid, tables, rng_np)
            s = metrics_for(p_sens[(side, L)], sens_lab_in, sv, tables, rng_np, sens=sm)
            split_k, split_v = ("H", L) if side == "in" else ("noise", "-")
        rows.append(dict(side=side, layer=L, split=f"{split_k}={split_v}", general=g, sensitive=s))
        print(f"   {side:5s} layer {L:2d} ({split_k}={split_v}): content_leak {g['content_leak'][0]:.3f} "
              f"[{g['content_leak'][1]:.3f}, {g['content_leak'][2]:.3f}]  all_top1 {g['all_top1'][0]:.3f}  "
              f"weighted {g['weighted_top1'][0]:.3f}  sensitive {s['sensitive_top1'][0]:.3f}  "
              f"exact {s['sensitive_exact']:.3f}", flush=True)

    # ---- sanity checks ----
    get = lambda side, L: next(r for r in rows if r["side"] == side and r["layer"] == L)
    checks = []
    l0 = get("in", 0)["general"]["content_top1"][0]
    checks.append(("layer 0 (embeddings) almost fully recoverable", l0 > 0.9, f"content_top1 = {l0:.3f}"))
    nz = get("noise", 1)["general"]
    checks.append(("noise attacker does not beat blind guess", nz["content_leak"][0] < 0.02,
                   f"content_leak = {nz['content_leak'][0]:.3f}"))
    first, last = get("in", 1)["general"]["content_leak"][0], get("in", n_layers - 1)["general"]["content_leak"][0]
    checks.append(("leakage lower deep in the model than at layer 1", last < first,
                   f"H=1: {first:.3f}, H={n_layers-1}: {last:.3f}"))
    t0 = get("out", n_layers)["general"]["content_top1"][0]
    checks.append(("out-side at T=0 close to 100%", t0 > 0.8, f"content_top1 = {t0:.3f}"))
    print("\nSANITY CHECKS (a FAIL on checks 1, 2 or 4 means something is wrong in the setup)")
    for k, (name, ok, info) in enumerate(checks, 1):
        if k == 3:   # a trend, not a correctness test: if it fails on the real model it is a finding
            print(f"   [INFO] {name}: {'yes' if ok else 'NO - deep layers leak as much as shallow ones'}  ({info})")
        else:
            print(f"   [{'PASS' if ok else 'FAIL'}] {name}  ({info})")

    # ---- save ----
    meas_in = {r["layer"]: round(r["general"]["content_leak"][0], 4) for r in rows
               if r["side"] == "in" and 1 <= r["layer"] <= n_layers - 1}
    meas_out = {n_layers - r["layer"]: round(r["general"]["content_leak"][0], 4) for r in rows
                if r["side"] == "out"}
    meas_out = dict(sorted(meas_out.items()))
    with open(os.path.join(CFG["out_dir"], "results.json"), "w") as f:
        json.dump(dict(config=CFG, smoke=a.smoke, model=CFG["model_id"] if not a.smoke else "tiny-random",
                       sanity_checks=[dict(name=n, passed=bool(o), info=i) for n, o, i in checks],
                       MEASURED_ASR_IN=meas_in, MEASURED_ASR_OUT=meas_out, rows=rows), f, indent=1)
    with open(os.path.join(CFG["out_dir"], "attack_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "layer", "split", "content_leak", "content_leak_low", "content_leak_high",
                    "content_top1", "blind_content_top1", "all_top1", "weighted_top1",
                    "sensitive_top1", "sensitive_top1_low", "sensitive_top1_high", "sensitive_exact",
                    "n_test_tokens", "n_content_tokens", "n_sensitive_tokens"])
        for r in rows:
            g, s = r["general"], r["sensitive"]
            w.writerow([r["side"], r["layer"], r["split"], *[round(v, 4) for v in g["content_leak"]],
                        round(g["content_top1"][0], 4), round(g["blind_content_top1"], 4),
                        round(g["all_top1"][0], 4), round(g["weighted_top1"][0], 4),
                        *[round(v, 4) for v in s["sensitive_top1"]], round(s["sensitive_exact"], 4),
                        g["n_tokens"], g["n_content"], s["n_sensitive"]])

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a_, side, xs_key, title in [(ax[0], "in", "H", "What the client sends (after H layers)"),
                                    (ax[1], "out", "T", "What the server sends back (T layers left)")]:
        rs = [r for r in rows if r["side"] == side and (side == "out" or r["layer"] >= 1)]
        xs = [r["layer"] if side == "in" else n_layers - r["layer"] for r in rs]
        order = np.argsort(xs); xs = np.array(xs)[order]; rs = [rs[i] for i in order]
        cl = np.array([r["general"]["content_leak"] for r in rs])
        a_.fill_between(xs, 100 * cl[:, 1], 100 * cl[:, 2], color="#1a9e55", alpha=.2, lw=0)
        a_.plot(xs, 100 * cl[:, 0], "-o", ms=4, color="#1a9e55", lw=2, label="Content tokens, above blind (main)")
        a_.plot(xs, [100 * r["general"]["weighted_top1"][0] for r in rs], "-s", ms=3, color="#7a3cc9",
                label="Rarity-weighted")
        a_.plot(xs, [100 * r["sensitive"]["sensitive_top1"][0] for r in rs], "-^", ms=4, color="#e0533d",
                label="Planted private details")
        a_.plot(xs, [100 * r["general"]["all_top1"][0] for r in rs], "--", color="#888",
                label="All tokens (Cunningham-style)")
        a_.set_xlabel(f"{xs_key} = layers kept on the client"); a_.set_title(title, fontsize=11)
        a_.grid(alpha=.3); a_.set_ylim(0, 102)
        a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].set_ylabel("Recovered by the server [%]")
    ax[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Measured inversion attack: {'TINY RANDOM MODEL (smoke test)' if a.smoke else CFG['model_id']}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CFG["out_dir"], "attack_asr_vs_layer.png"), dpi=200)

    print("\nCOPY THESE INTO split_privacy.py AND split_optimisation.py:")
    print(f"MEASURED_ASR_IN  = {meas_in}")
    print(f"MEASURED_ASR_OUT = {meas_out}")
    print(f"\nSaved to {CFG['out_dir']}/  (results.json, attack_results.csv, attack_asr_vs_layer.png)")


if __name__ == "__main__":
    main()
