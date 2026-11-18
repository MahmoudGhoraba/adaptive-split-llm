"""
TASK 5a - Answer leakage on GENERATED answers (not teacher-forced), with SAVED trained secret adapters.

For every adapter file (adapters_lp*_la*.pt written by secret_ushape.py / save_adapters()):
  1. chat-format prompts whose expected answer repeats planted private details (name, hospital, disease, city,
     phone, amount; ANSWER_TEMPLATES below), n_prompts of them, same detail lists as the earlier attacks
  2. the PROTECTED client (secret head + secret tail on) generates the answer with real autoregressive greedy
     decoding (max_new tokens, stops at EOS)
  3. the server's view: it holds the state after layer N-T at every answer position (identical to the protected run,
     because layers H..N-T-1 are public) and runs the PUBLIC tail + output layer on it -> its guess of every next
     answer token (one forward pass over prompt + generated answer with the tail adapters off; a KV-cached decode
     gives the same states)
  4. answer leakage = content leakage (existing metric, leak_metrics(): stopwords/punctuation/short tokens removed,
     blind-guess corrected, 95% bootstrap CI) of the server's guesses vs the client's generated tokens
     + answer-side private details: every planted detail that APPEARS in the generated answer; full recovery = the
     server guessed all its tokens (sensitive_exact); also the share of planted details that appear at all
  5. teacher-forced number for the SAME adapters (as in secret_ushape.py: WikiText test text, server runs the public
     tail, compared with the protected model's argmax) for comparison
  Also the PUBLIC model's own answers (no adapters) for the share of details that appear (reference).

Run (Colab GPU):  python answer_generation.py --adapters runs/H4_T2_seed0/adapters_lp0.1_la0.03.pt [more ...]
Code test:        python answer_generation.py --smoke        (tiny random Llama on CPU, random "trained" adapters)
Writes <out>/answer_generation.json, answer_generation.csv
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "step_secret_multisplit")):
    sys.path.insert(0, p)
import secret_ushape as su                       # data, models, adapters, metrics: the same code as the earlier runs

CFG = dict(n_prompts=100, max_new=96, seed=0, out_dir="task5a_results")
SMOKE = dict(n_prompts=12, max_new=16, out_dir="task5a_results_smoke")

ANSWER_TEMPLATES = [
    "Write a short sick note for {name}, who is being treated for {disease} at {hospital}. Include the name, the condition and the hospital.",
    "Write a two-sentence letter to {hospital} confirming that {name} has an appointment about {disease}.",
    "Draft a text message to {name} asking them to call {phone} about their {disease} test results.",
    "Write a payment receipt: {amount} pounds paid by {name} to {hospital}. Repeat the amount and the name.",
    "Write a referral note for {name} from {city}, diagnosed with {disease}, to {hospital}.",
    "Write a reminder for {name} in {city}: the clinic number is {phone}. Repeat the number.",
    "Write an email to {name} confirming the transfer of {amount} pounds. Mention the amount.",
    "Write a one-paragraph summary of this case: {name}, {city}, {disease}, treated at {hospital}.",
]


def make_prompts(n, rng):
    out = []
    for _ in range(n):
        tpl = rng.choice(ANSWER_TEMPLATES)
        fill = dict(name=f"{rng.choice(su.FIRST)} {rng.choice(su.LAST)}", hospital=rng.choice(su.HOSPITALS),
                    disease=rng.choice(su.DISEASES), city=rng.choice(su.CITIES),
                    phone="01" + "".join(str(rng.randint(0, 9)) for _ in range(9)), amount=str(rng.randint(2, 99) * 500))
        used = re.findall(r"\{(\w+)\}", tpl)
        out.append(dict(text=tpl.format(**fill), details={k: fill[k] for k in dict.fromkeys(used)}))
    return out


def as_ids(x):
    """Token ids as a flat list of ints, whatever apply_chat_template / a tokenizer returned in this transformers
    version (list, dict / BatchEncoding, tokenizers.Encoding, tensor, or a batch of one of these)."""
    if hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "ids") and not isinstance(x, (list, tuple)):
        x = x.ids                                              # tokenizers.Encoding
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if x and not isinstance(x[0], int):                        # batch of one
        assert len(x) == 1, f"expected one sequence, got {len(x)}"
        return as_ids(x[0])
    return [int(i) for i in x]


def encode_chat(tok, text, smoke):
    if smoke:
        return [tok.bos_id] + tok.encode("User : " + text + " Assistant :")
    # render the template as TEXT, then tokenize (the text already holds <|begin_of_text|>): the same ids as
    # apply_chat_template(tokenize=True), but independent of what that call returns in a given transformers version
    hf = tok.tok
    text = hf.apply_chat_template([dict(role="user", content=text)], add_generation_prompt=True, tokenize=False)
    return as_ids(hf(text, add_special_tokens=False))


def token_char_spans(tok, ids, smoke):
    """Character span of every generated token inside the decoded answer text."""
    if smoke:
        pieces = [tok.decode_one(i) for i in ids]
    else:
        pieces, prev = [], ""
        for k in range(1, len(ids) + 1):
            cur = tok.tok.decode(ids[:k], skip_special_tokens=True); pieces.append(cur[len(prev):]); prev = cur
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p))); pos += len(p)
    return "".join(pieces), spans


def detail_mask(tok, ans, details, smoke):
    """Answer text, per-token detail id (0 = none, j = j-th planted detail) and how many details appear."""
    text, spans = token_char_spans(tok, ans, smoke)
    sens, found = np.zeros(len(ans), dtype=int), 0
    for j, v in enumerate(details.values(), 1):
        hit = False
        for m in re.finditer(re.escape(v.lower()), text.lower()):
            hit = True
            for t, (a, b) in enumerate(spans):
                if a < m.end() and b > m.start():
                    sens[t] = j
        found += hit
    return text, sens, found


def selftest_details(tok, smoke, tables, rng_np):
    """Code test of the answer-side detail path: an answer that contains two planted details; a server that
    guesses every token right (full recovery 1.0) and one that misses one token of the first detail (0.5)."""
    details = dict(name="Mona Hassan", disease="diabetes")
    ans = tok.encode(" note for Mona Hassan , treated for diabetes .")
    text, sens, found = detail_mask(tok, ans, details, smoke)
    assert found == 2 and (sens == 1).sum() >= 2 and (sens == 2).sum() >= 1, (text, sens)
    label = np.array([ans]); valid = np.ones_like(label, dtype=bool)
    ok = su.leak_metrics(label.copy(), label, valid, tables, rng_np, sens=sens[None])["sensitive_exact"]
    bad = label.copy(); bad[0, int(np.argmax(sens == 1))] = (bad[0, int(np.argmax(sens == 1))] + 1) % tok.vocab_size
    half = su.leak_metrics(bad, label, valid, tables, rng_np, sens=sens[None])["sensitive_exact"]
    assert ok == 1.0 and half == 0.5, (ok, half)
    print(f"self-test answer details: {found}/2 found in '{text.strip()}', full recovery {ok} (all right) / {half} (one token wrong): PASS")


@torch.no_grad()
def generate(model, dev, prompt_ids, max_new, eos):
    x = torch.tensor([prompt_ids], device=dev)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                         eos_token_id=list(eos) if eos else None, pad_token_id=list(eos)[0] if eos else 0)
    ans = out[0, len(prompt_ids):].tolist()
    if eos:
        ans = [t for t in ans[:next((i for i, t in enumerate(ans) if t in eos), len(ans))]]
    return ans


def eos_ids(model, smoke):
    if smoke:
        return set()
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


def server_guess(model, dev, seq, tail):
    """Server runs the PUBLIC tail on the states after layer N-T (head adapters on, tail adapters off)."""
    su.set_on(tail, False)
    with torch.no_grad():
        g = model(input_ids=torch.tensor([seq], device=dev)).logits.argmax(-1)[0].cpu().numpy()
    su.set_on(tail, True)
    return g


def evaluate_adapters(model, tok, dev, path, prompts, tables, test_ids, smoke, rng_np):
    _, meta = su.load_adapters(dict(enumerate(model.model.layers)), path, dev)
    H, T = meta["H"], meta["T"]
    head, tail = adapters_by_side(model, H, T)
    su.set_on(head + tail, True)
    eos = eos_ids(model, smoke)
    preds, labels, sens_rows, lens, appear, planted, answers = [], [], [], [], 0, 0, []
    L = max(1, CFG["max_new"])
    for p in prompts:
        pids = encode_chat(tok, p["text"], smoke)
        ans = generate(model, dev, pids, CFG["max_new"], eos)
        if not ans:
            continue
        seq = pids + ans
        g = server_guess(model, dev, seq, tail)
        guess = g[len(pids) - 1: len(seq) - 1]                   # server's guess of every generated token
        text, sens, found = detail_mask(tok, ans, p["details"], smoke)
        planted += len(p["details"]); appear += found
        pad = L - len(ans)
        preds.append(list(guess) + [0] * pad); labels.append(ans + [0] * pad)
        sens_rows.append(list(sens) + [0] * pad); lens.append(len(ans))
        answers.append(dict(prompt=p["text"], answer=text, details=p["details"]))
    pred, label = np.array(preds), np.array(labels)
    valid = np.zeros_like(label, dtype=bool)
    for i, n_ans in enumerate(lens):                              # only the generated positions count
        valid[i, :n_ans] = True
    m = su.leak_metrics(pred, label, valid, tables, rng_np, sens=np.array(sens_rows))
    # teacher-forced reference for the same adapters (secret_ushape.py definition, WikiText test text)
    su.set_on(head + tail, True); true_out = su.argmax_tokens(model, dev, test_ids)
    su.set_on(tail, False); srv_out = su.argmax_tokens(model, dev, test_ids); su.set_on(tail, True)
    tv = np.ones(test_ids.shape, dtype=bool); tv[:, 0] = False
    tf = su.leak_metrics(srv_out, true_out, tv, tables, rng_np)
    su.remove_all_adapters(model)
    return dict(adapters=os.path.basename(path), H=H, T=T, lam_p=meta.get("lam_p"), lam_a=meta.get("lam_a"),
                train_seed=meta.get("train_seed"), n_answers=len(answers),
                generated_answer_leak=m["content_leak"][0], generated_answer_leak_ci=m["content_leak"][1:],
                answer_details_planted=planted, answer_details_in_answer=appear,
                answer_detail_full_recovery=m.get("sensitive_exact"), answer_detail_token_recovery=m.get("sensitive_top1"),
                teacher_forced_answer_leak=tf["content_leak"][0], teacher_forced_answer_leak_ci=tf["content_leak"][1:],
                examples=answers[:3])


def adapters_by_side(model, H, T):
    """Secret LoRA modules on the head layers (< H) and on the tail layers (>= N - T)."""
    n, head, tail = len(model.model.layers), [], []
    for li, layer in enumerate(model.model.layers):
        for parent, nm in su.PROJ:
            m = getattr(getattr(layer, parent), nm)
            if isinstance(m, su.LoRALinear):
                (head if li < H else tail if li >= n - T else []).append(m)
    return head, tail


def smoke_adapters(model, dev, out_dir):
    """Random 'trained' adapters for the code test (H = 1, T = 1), written with save_adapters()."""
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
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", type=str)
    ap.add_argument("--n_prompts", type=int); ap.add_argument("--max_new", type=int)
    a = ap.parse_args()
    if a.smoke:
        CFG.update(SMOKE)
        su.CFG.update(lora_rank=8, lora_alpha=8, train_tokens=40_000, test_tokens=4_000, seq_len=64, bootstrap=30)
    for k in ("out", "n_prompts", "max_new"):
        if getattr(a, k):
            CFG["out_dir" if k == "out" else k] = getattr(a, k)
    os.makedirs(CFG["out_dir"], exist_ok=True)
    su.set_seed(CFG["seed"]); rng_py = random.Random(CFG["seed"]); rng_np = np.random.default_rng(CFG["seed"])
    train_txt, test_txt = su.load_general_text(a.smoke, rng_py)
    prompts = make_prompts(CFG["n_prompts"], rng_py)
    texts = train_txt + test_txt + [p["text"] + " User : Assistant : note for Mona Hassan , treated for diabetes ." for p in prompts]
    model, tok, dev = su.load_model(a.smoke, texts)
    train_ids = su.chunk(tok, train_txt, su.CFG["seq_len"], su.CFG["train_tokens"])
    test_ids = su.chunk(tok, test_txt, su.CFG["seq_len"], su.CFG["test_tokens"])
    tables = su.token_tables(tok, train_ids)
    if a.smoke:
        selftest_details(tok, a.smoke, tables, rng_np)
    paths = a.adapters or ([smoke_adapters(model, dev, CFG["out_dir"])] if a.smoke else [])
    if not paths:
        sys.exit("give --adapters <adapters_*.pt> (written by secret_ushape.py with save_adapters)")
    # reference: the PUBLIC model's own answers (how often the details appear without protection)
    eos = eos_ids(model, a.smoke); pub_appear = pub_planted = 0
    for p in prompts[:min(30, len(prompts))]:
        ans = generate(model, dev, encode_chat(tok, p["text"], a.smoke), CFG["max_new"], eos)
        text, _ = token_char_spans(tok, ans, a.smoke)
        for v in p["details"].values():
            pub_planted += 1; pub_appear += v.lower() in text.lower()
    rows = []
    for path in paths:
        t0 = time.time()
        r = evaluate_adapters(model, tok, dev, path, prompts, tables, test_ids, a.smoke, rng_np)
        r["minutes"] = (time.time() - t0) / 60
        rows.append(r)
        print(f"{r['adapters']} (H={r['H']}, T={r['T']}, λp={r['lam_p']}, λa={r['lam_a']}): GENERATED answer leak "
              f"{r['generated_answer_leak']:.3f} (CI {r['generated_answer_leak_ci'][0]:.3f}-{r['generated_answer_leak_ci'][1]:.3f}) | "
              f"TEACHER-FORCED {r['teacher_forced_answer_leak']:.3f} | answer details: {r['answer_details_in_answer']}/"
              f"{r['answer_details_planted']} appear, full recovery {r['answer_detail_full_recovery']} | {r['minutes']:.1f} min",
              flush=True)
    res = dict(config=CFG, smoke=a.smoke, public_model_details_in_answer=[pub_appear, pub_planted], rows=rows)
    json.dump(res, open(os.path.join(CFG["out_dir"], "answer_generation.json"), "w"), indent=1, default=str)
    keys = [k for k in rows[0] if k != "examples"]
    with open(os.path.join(CFG["out_dir"], "answer_generation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"public model (no adapters): details appear in {pub_appear}/{pub_planted} answers' text")
    print(f"Saved to {CFG['out_dir']}/")


if __name__ == "__main__":
    main()
