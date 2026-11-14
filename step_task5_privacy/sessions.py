"""
TASK 5d (C2) - templated multi-turn chat SESSIONS with private details spread over turns.

Every session (fixed seed, deterministic):
  persona: name, city, condition, phone, amount (same value lists as the earlier attacks, secret_ushape.py)
  5-10 user turns, each followed by a templated assistant reply (fixed neutral texts; NOT model-generated)
  every detail is introduced in one random turn; with probability p_repeat it is mentioned again in a later turn
  (several partial chances for a server); remaining turns are filler questions
Rendered with the Llama-3.2 chat template (fixed date_string) -> token ids, and for every token its role
  template (special/header tokens that repeat every turn, and the system block), user, assistant
and every detail MENTION: (type, value, turn index, token positions). Train/test split BY SESSION.
--smoke: word tokenizer (secret_ushape.WordTok) with marker words "<user>", "<assistant>", "<eot>" as template.

Run:  python sessions.py [--smoke] [--n 600]   -> prints statistics and writes <out>/sessions.json
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "step_secret_multisplit")):
    sys.path.insert(0, p)
import secret_ushape as su

CFG = dict(n_sessions=600, n_test=200, min_turns=5, max_turns=10, p_repeat=0.5, seed=0, date_string="14 Nov 2026")
DETAILS = ["name", "city", "condition", "phone", "amount"]
INTRO = dict(
    name=["Hi, my name is {name}.", "Hello, this is {name} writing.", "I am {name} and I need some advice."],
    city=["I live in {city}.", "I am writing from {city}.", "My family and I are based in {city}."],
    condition=["I was diagnosed with {condition} last month.", "My doctor says I have {condition}.",
               "I have been living with {condition} for a year."],
    phone=["You can reach me at {phone}.", "My phone number is {phone}.", "Please note my number, {phone}."],
    amount=["The treatment will cost {amount} pounds.", "I have to pay {amount} pounds next week.",
            "My insurance covers only {amount} pounds."],
)
REPEAT = dict(
    name=["As I said, I am {name}.", "{name} again, thank you for the help so far.", "This is still {name}."],
    city=["Is that also available in {city}?", "Back to {city}: are there clinics nearby?"],
    condition=["Does that change anything for {condition}?", "Is that safe with {condition}?"],
    phone=["If needed, call {phone} again.", "My number is still {phone}."],
    amount=["Can I split the {amount} pounds into two payments?", "Is {amount} pounds a normal price?"],
)
FILLER = ["What is a healthy breakfast?", "How much water should I drink every day?", "Can you explain how vaccines work?",
          "What are good exercises for back pain?", "How can I sleep better at night?", "Is coffee bad for the heart?",
          "How do I prepare for a doctor's appointment?", "What should I ask my pharmacist?",
          "How long does a flu usually last?", "What are the signs of dehydration?"]
REPLIES = ["Thank you for telling me. How can I help further?", "I understand. Here is some general advice.",
           "That is a good question. Let me explain briefly.", "Noted. Is there anything else you would like to know?",
           "Thanks. Please consult your doctor for personal advice."]
SMOKE_TPL = dict(user="<user>", assistant="<assistant>", eot="<eot>")


def make_session(rng, sid):
    persona = dict(name=f"{rng.choice(su.FIRST)} {rng.choice(su.LAST)}", city=rng.choice(su.CITIES),
                   condition=rng.choice(su.DISEASES), phone="01" + "".join(str(rng.randint(0, 9)) for _ in range(9)),
                   amount=str(rng.randint(2, 99) * 500))
    n = rng.randint(CFG["min_turns"], CFG["max_turns"])
    slots = [[] for _ in range(n)]
    for d in DETAILS:
        t = rng.randrange(n); slots[t].append(rng.choice(INTRO[d]))
        if rng.random() < CFG["p_repeat"] and t < n - 1:
            slots[rng.randrange(t + 1, n)].append(rng.choice(REPEAT[d]))
    msgs = []
    for t in range(n):
        text = " ".join(slots[t]) if slots[t] else rng.choice(FILLER)
        if slots[t] and rng.random() < 0.5:
            text += " " + rng.choice(FILLER)
        msgs.append(dict(role="user", content=text.format(**persona)))
        msgs.append(dict(role="assistant", content=rng.choice(REPLIES)))
    return dict(id=sid, persona=persona, messages=msgs, n_turns=n)


def render(sess, tok, smoke):
    """Token ids, role per token ('template' | 'user' | 'assistant'), turn index per token, and detail mentions."""
    msgs, persona = sess["messages"], sess["persona"]
    if smoke:
        ids, role, turn, text_tokens = [tok.bos_id], ["template"], [-1], []
        mentions, char_text = [], ""
        for k, m in enumerate(msgs):
            t = k // 2
            w = tok.encode(SMOKE_TPL[m["role"]])
            ids += w; role += ["template"] * len(w); turn += [t] * len(w)
            cids, offs = tok.encode_offsets(m["content"])
            base = len(ids)
            ids += cids; role += [m["role"]] * len(cids); turn += [t] * len(cids)
            if m["role"] == "user":
                for d in DETAILS:
                    for mt in re.finditer(re.escape(persona[d]), m["content"]):
                        pos = [base + j for j, (a, b) in enumerate(offs) if a < mt.end() and b > mt.start()]
                        mentions.append(dict(type=d, value=persona[d], turn=t, positions=pos))
            w = tok.encode(SMOKE_TPL["eot"])
            ids += w; role += ["template"] * len(w); turn += [t] * len(w)
        assert len(ids) == len(role) == len(turn)
        return dict(ids=ids, role=role, turn=turn, mentions=mentions)
    hf = tok.tok
    text = hf.apply_chat_template(msgs, date_string=CFG["date_string"], tokenize=False)
    enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    assert ids == hf.apply_chat_template(msgs, date_string=CFG["date_string"], return_dict=True)["input_ids"]
    # character ranges of every message content inside the rendered text (in order)
    spans, cur = [], 0
    for m in msgs:
        a = text.index(m["content"], cur); spans.append((a, a + len(m["content"]), m["role"])); cur = a + len(m["content"])
    role, turn = [], []
    for (a, b) in offs:
        r, t = "template", -1
        for k, (sa, sb, rr) in enumerate(spans):
            if a >= sa and b <= sb and b > a:
                r, t = rr, k // 2
                break
            if a < sa:
                t = k // 2 if k else -1
                break
        role.append(r); turn.append(t if r != "template" else max(t, -1))
    assert len(ids) == len(role) == len(turn)
    mentions = []
    for k, (sa, sb, rr) in enumerate(spans):
        if rr != "user":
            continue
        for d in DETAILS:
            for mt in re.finditer(re.escape(persona[d]), text[sa:sb]):
                a, b = sa + mt.start(), sa + mt.end()
                pos = [j for j, (x, y) in enumerate(offs) if x < b and y > a]
                mentions.append(dict(type=d, value=persona[d], turn=k // 2, positions=pos))
    return dict(ids=ids, role=role, turn=turn, mentions=mentions)


def make_sessions(n=None):
    rng = random.Random(CFG["seed"])
    return [make_session(rng, i) for i in range(n or CFG["n_sessions"])]


def session_texts(sessions):
    """All message texts + template markers (vocabulary of the smoke word tokenizer)."""
    return [m["content"] for s in sessions for m in s["messages"]] + [" ".join(SMOKE_TPL.values())]


def build(tok, smoke, n=None, sessions=None):
    sessions = sessions or make_sessions(n)
    n = len(sessions)
    for s in sessions:
        s.update(render(s, tok, smoke))
    n_test = min(CFG["n_test"], n // 3)
    order = list(range(n)); random.Random(CFG["seed"] + 1).shuffle(order)
    test = set(order[:n_test])
    for s in sessions:
        s["split"] = "test" if s["id"] in test else "train"
    return sessions


def stats(sessions):
    import collections
    c = collections.Counter(); rep = collections.Counter(); L = []
    for s in sessions:
        L.append(len(s["ids"]))
        per = collections.Counter(m["type"] for m in s["mentions"])
        for d in DETAILS:
            c[d] += per[d] > 0; rep[d] += per[d] > 1
    n = len(sessions)
    return dict(n_sessions=n, tokens_min=min(L), tokens_mean=sum(L) / n, tokens_max=max(L),
                share_tokens_beyond_128=sum(max(0, l - 128) for l in L) / sum(L),
                detail_present={d: c[d] / n for d in DETAILS}, detail_repeated={d: rep[d] / n for d in DETAILS},
                template_tokens_mean=sum(s["role"].count("template") for s in sessions) / n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--n", type=int)
    ap.add_argument("--out", type=str, default="task5d_sessions")
    a = ap.parse_args()
    sessions = make_sessions(a.n)
    if a.smoke:
        tok = su.WordTok(session_texts(sessions))
    else:
        from transformers import AutoTokenizer
        tok = su.HFTok(AutoTokenizer.from_pretrained(su.CFG["model_id"]))
    sessions = build(tok, a.smoke, sessions=sessions)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(config=CFG, smoke=a.smoke, sessions=sessions), open(os.path.join(a.out, "sessions.json"), "w"))
    print(json.dumps(stats(sessions), indent=1))
    s = sessions[0]
    print("example session 0:", s["persona"], "| turns", s["n_turns"], "| tokens", len(s["ids"]))
    for m in s["mentions"]:
        print("   mention", m["type"], m["value"], "turn", m["turn"], "tokens",
              [tok.decode_one(s["ids"][p]) for p in m["positions"]])


if __name__ == "__main__":
    main()
