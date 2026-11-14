"""
Step 2-3 check of the split runtime (no sockets yet): do partial models reproduce the full model?

Phases (each in its own process, so memory numbers are not mixed):
  reference  full HF model (Llama-3.2-1B-Instruct, fp16, sdpa) -> greedy tokens, top-1/top-2 logit margin and the
             output of every decoder layer, for 3 chat prompts; public model and, per split, the model with secret
             adapters (su.add_adapters, head seed 2000, tail seed 2001 = the Colab formula with k = 0, seed 0;
             UNTRAINED adapters: they test the code path, not protection)
  split      client (embedding + head + tail + output layer) and server (middle) as PartialLlama in ONE process,
             greedy loop head -> mid -> tail; compared with the reference
  memory     ONE side alone in a fresh process: load, prefill P tokens, A decode steps; footprint per stage

Pass criterion (fixed before the first run): identical greedy tokens over the test length AND boundary states
(outputs of layer H, N-T and N) within relative L2 error 1e-2 per position. On divergence the first divergent
step and both logit margins are logged.

Run:  python check_partial.py                 (all phases, real model, MPS)
      python check_partial.py --smoke         (tiny random Llama, CPU)
Writes check_out/ (or check_out_smoke/): reference.pt, check_partial.json, check_partial.csv, memory.csv
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import partial_model as pm
from partial_model import su

CFG = dict(
    splits          = [(1, 1), (2, 2), (4, 1), (4, 2), (6, 2), (8, 2), (1, 0), (4, 0), (15, 0), (1, 14)],
    adapter_splits  = [(1, 1), (2, 2), (4, 2), (8, 2)],
    head_seed       = 2000,
    tail_seed       = 2001,
    max_new_tokens  = 48,
    rel_tol         = 1e-2,
    memory_splits   = [(1, 1), (4, 2), (8, 2)],
    memory_P        = 512,                  # = prompt_max of the step3c/step5 workload
    memory_A        = 128,                  # = answer_len
    prompts         = [
        "Explain in two sentences why the sky is blue.",
        "My name is Mona Hassan and I was admitted to Cleopatra Hospital for diabetes. What should I eat?",
        "Write a short polite message asking my landlord for one more week to pay the rent.",
    ],
)
SMOKE = dict(splits=[(1, 1), (2, 2), (1, 0), (2, 3), (5, 0)], adapter_splits=[(1, 1), (2, 2)],
             max_new_tokens=12, memory_splits=[(2, 2)], memory_P=32, memory_A=8)


def setup(a):
    if a.smoke:
        CFG.update(SMOKE)
        out = os.path.join(HERE, "check_out_smoke")
        mdir = os.path.join(out, "tiny_llama")
        if not os.path.exists(os.path.join(mdir, "config.json")):
            pm.write_smoke_model(mdir)
        return out, mdir, "cpu", torch.float32
    return os.path.join(HERE, "check_out"), pm.model_dir(), "mps", torch.float16


def prompt_ids(a, mdir):
    if a.smoke:
        g = torch.Generator().manual_seed(1)
        return [torch.randint(1, 256, (1, n), generator=g) for n in (9, 17, 5)]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir)
    return [tok.apply_chat_template([dict(role="user", content=p)], add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True)["input_ids"] for p in CFG["prompts"]]


def eos_ids(mdir):
    e = json.load(open(os.path.join(mdir, "config.json"))).get("eos_token_id")
    return set(e if isinstance(e, list) else [e] if e is not None else [])


def margin(lg):
    t = lg.float().topk(2).values
    return float(t[0] - t[1])


# ---------------------------------------------------------------------------------------- reference
@torch.no_grad()
def greedy_full(model, ids, dev, eos):
    """Full model, KV cache, one token per step. Returns tokens, margins, outputs of every layer [N, pos, d]."""
    from transformers.cache_utils import DynamicCache
    outs = {i: [] for i in range(len(model.model.layers))}
    hooks = [l.register_forward_hook(lambda m, i, o, k=k: outs[k].append((o[0] if isinstance(o, tuple) else o)[0].cpu()))
             for k, l in enumerate(model.model.layers)]
    cache, x, toks, margins = DynamicCache(config=model.config), ids.to(dev), [], []
    for _ in range(CFG["max_new_tokens"]):
        lg = model(input_ids=x, past_key_values=cache, use_cache=True).logits[0, -1]
        nxt = int(lg.argmax()); toks.append(nxt); margins.append(margin(lg))
        if nxt in eos:
            break
        x = torch.tensor([[nxt]], device=dev)
    for h in hooks:
        h.remove()
    return dict(tokens=toks, margins=margins, layers=torch.stack([torch.cat(outs[k]) for k in sorted(outs)]))


def phase_reference(a, out, mdir, dev, dtype):
    from transformers import AutoModelForCausalLM
    m0 = memstat.snapshot()
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="sdpa").to(dev).eval()
    n = len(model.model.layers); eos = eos_ids(mdir)
    ref = dict(prompts=prompt_ids(a, mdir), public=[], adapters={})
    for ids in ref["prompts"]:
        ref["public"].append(greedy_full(model, ids, dev, eos))
    for H, T in CFG["adapter_splits"]:
        su.add_adapters(model, range(0, H), CFG["head_seed"], dev)
        if T:
            su.add_adapters(model, range(n - T, n), CFG["tail_seed"], dev)
        ref["adapters"][(H, T)] = [greedy_full(model, ids, dev, eos) for ids in ref["prompts"]]
        su.remove_all_adapters(model)
    ref["memory"] = dict(start=m0, end=memstat.snapshot())
    torch.save(ref, os.path.join(out, "reference.pt"))
    for i, r in enumerate(ref["public"]):
        print(f"  prompt {i}: {ref['prompts'][i].shape[1]} tokens -> {len(r['tokens'])} generated, "
              f"min margin {min(r['margins']):.3f}", flush=True)


# ---------------------------------------------------------------------------------------- split
@torch.no_grad()
def greedy_split(client, server, ids, eos, has_tail):
    cc, cs = client.new_caches(), server.new_caches()
    x, pos, toks, margins, bH, bM, bN = ids, 0, [], [], [], [], []
    for _ in range(CFG["max_new_tokens"]):
        h = client.run("head", client.embed_ids(x), pos, cc); bH.append(h[0].cpu())
        h = server.run("mid", h, pos, cs); bM.append(h[0].cpu())
        if has_tail:
            h = client.run("tail", h, pos, cc)
        bN.append(h[0].cpu())
        lg = client.logits(h)[0, -1]
        pos += x.shape[1]
        nxt = int(lg.argmax()); toks.append(nxt); margins.append(margin(lg))
        if nxt in eos:
            break
        x = torch.tensor([[nxt]])
    return dict(tokens=toks, margins=margins, bH=torch.cat(bH), bM=torch.cat(bM), bN=torch.cat(bN))


def rel_err(x, ref):
    """Max over positions of ||x - ref|| / ||ref|| (fp32), and max |x - ref|, on the common positions."""
    k = min(len(x), len(ref)); x, ref = x[:k].float(), ref[:k].float()
    return float(((x - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max()), float((x - ref).abs().max())


def compare(got, ref, H, T, n):
    first = next((i for i, (u, v) in enumerate(zip(got["tokens"], ref["tokens"])) if u != v), None)
    same = first is None and len(got["tokens"]) == len(ref["tokens"])
    errs = {}
    for nm, L in (("bH", H), ("bM", n - T), ("bN", n)):
        errs[nm] = rel_err(got[nm], ref["layers"][L - 1])
    ok = same and all(e[0] <= CFG["rel_tol"] for e in errs.values())
    row = dict(tokens_equal=same, n_tokens=len(ref["tokens"]), n_tokens_split=len(got["tokens"]),
               first_divergence=first, rel_err_H=errs["bH"][0], rel_err_NT=errs["bM"][0], rel_err_N=errs["bN"][0],
               abs_err_max=max(e[1] for e in errs.values()), passed=ok)
    if first is not None:
        row.update(margin_ref_at_div=ref["margins"][first], margin_split_at_div=got["margins"][first])
    return row


def phase_split(a, out, mdir, dev, dtype):
    ref = torch.load(os.path.join(out, "reference.pt"), weights_only=False)
    eos, rows = eos_ids(mdir), []
    n = pm.load_config(mdir).num_hidden_layers
    jobs = [(H, T, False) for H, T in CFG["splits"]] + [(H, T, True) for H, T in CFG["adapter_splits"]]
    for H, T, ad in jobs:
        cseg, sseg = pm.split_segments(H, T, n)
        t0 = time.perf_counter()
        client = pm.PartialLlama(mdir, cseg, True, dev, dtype)
        server = pm.PartialLlama(mdir, sseg, False, dev, dtype)
        load_s = time.perf_counter() - t0
        if ad:
            client.add_secret_adapters(range(0, H), CFG["head_seed"])
            if T:
                client.add_secret_adapters(range(n - T, n), CFG["tail_seed"])
        refs = ref["adapters"][(H, T)] if ad else ref["public"]
        for i, ids in enumerate(ref["prompts"]):
            r = compare(greedy_split(client, server, ids, eos, T > 0), refs[i], H, T, n)
            if ad:   # non-vacuous: the adapters must change the output of the public model
                r["differs_from_public"] = refs[i]["tokens"] != ref["public"][i]["tokens"]
                r["state_change_vs_public_H"] = rel_err(refs[i]["layers"][H - 1], ref["public"][i]["layers"][H - 1])[0]
            rows.append(dict(H=H, T=T, U=n - H - T, adapters=ad, prompt=i, load_s=round(load_s, 2), **r))
            print(f"  H={H:2d} T={T:2d} {'adapters' if ad else 'public  '} prompt {i}: "
                  f"{'PASS' if r['passed'] else 'FAIL'} tokens {r['n_tokens_split']}/{r['n_tokens']} "
                  f"equal={r['tokens_equal']} rel err H/N-T/N {r['rel_err_H']:.1e}/{r['rel_err_NT']:.1e}/"
                  f"{r['rel_err_N']:.1e}" + (f" | vs public: tokens differ={r['differs_from_public']}, "
                     f"layer-H state change {r['state_change_vs_public_H']:.2f}" if ad else "")
                  + (f" FIRST DIVERGENCE step {r['first_divergence']} margins ref {r['margin_ref_at_div']:.3f} "
                     f"split {r['margin_split_at_div']:.3f}" if r["first_divergence"] is not None else ""), flush=True)
        del client, server
        if dev == "mps":
            torch.mps.empty_cache()
    json.dump(rows, open(os.path.join(out, "split_rows.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------- memory
@torch.no_grad()
def phase_memory(a, out, mdir, dev, dtype, side, H, T):
    n = pm.load_config(mdir).num_hidden_layers
    cseg, sseg = pm.split_segments(H, T, n)
    P, A = CFG["memory_P"], CFG["memory_A"]
    snaps = dict(start=memstat.snapshot())
    m = pm.PartialLlama(mdir, cseg if side == "client" else sseg, side == "client", dev, dtype)
    pm.sync(dev); snaps["loaded"] = memstat.snapshot()
    caches = m.new_caches(); d = m.cfg.hidden_size
    g = torch.Generator().manual_seed(0)
    for step in range(1 + A):                     # prefill P tokens, then A - 0 decode steps of 1 token
        k, pos = (P, 0) if step == 0 else (1, P + step - 1)
        if side == "client":
            h = m.run("head", m.embed_ids(torch.randint(0, m.cfg.vocab_size, (1, k), generator=g)), pos, caches)
            if T:
                h = m.run("tail", h, pos, caches)
            m.logits(h[:, -1:])
        else:
            m.run("mid", torch.randn(1, k, d, generator=g), pos, caches)
        pm.sync(dev)
        if step == 0:
            snaps["prefill"] = memstat.snapshot()
    snaps["decode"] = memstat.snapshot()
    print(json.dumps(dict(side=side, H=H, T=T, P=P, A=A, snaps=snaps)))


def modelled_gb(side, H, T, P, A, smoke):
    """Same formulas as memory_gb() in step5 (fp16, KV for P + A positions); server: U layers + their KV."""
    if smoke:
        return None
    sys.path.insert(0, os.path.join(HERE, "..", "step5_per_request"))
    import per_request_controller as pr
    p = dict(bytes_per_value=2, answer_len=A)
    if side == "client":
        return pr.memory_gb(dict(kind="split", H=H, T=T), dict(P_MAX=P), p)
    U = pr.N_LAYERS - H - T
    kv = 2 * pr.N_KV_HEADS * pr.HEAD_DIM * 2 * (P + A)
    return U * (pr.P_LAYER * 2 + kv) / 1e9


# ---------------------------------------------------------------------------------------- driver
def child(args, log):
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    err = [l for l in r.stderr.splitlines() if "Redirects are currently not supported" not in l]
    if r.returncode != 0:
        print("\n".join(err[-30:]))
        raise SystemExit(f"FAILED: {' '.join(args)}")
    log.write(r.stdout); return r.stdout


def env_info():
    import transformers, safetensors
    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], capture_output=True, text=True)
    chip, mem = hw.stdout.split("\n")[:2]
    return dict(chip=chip, memory_gb=int(mem) / 2**30, macos=platform.mac_ver()[0], python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, safetensors=safetensors.__version__,
                git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                   cwd=HERE).stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phase", choices=["reference", "split", "memory"])
    ap.add_argument("--side"); ap.add_argument("--H", type=int); ap.add_argument("--T", type=int)
    ap.add_argument("--skip-reference", action="store_true", help="reuse check_out/reference.pt")
    a = ap.parse_args()
    out, mdir, dev, dtype = setup(a)
    os.makedirs(out, exist_ok=True)
    if a.phase == "reference":
        return phase_reference(a, out, mdir, dev, dtype)
    if a.phase == "split":
        return phase_split(a, out, mdir, dev, dtype)
    if a.phase == "memory":
        return phase_memory(a, out, mdir, dev, dtype, a.side, a.H, a.T)

    sm = ["--smoke"] if a.smoke else []
    env, sys0 = env_info(), memstat.snapshot()
    print(f"{env}\nsystem at start: available {sys0['sys_available']/1e9:.2f} GB, swap used {sys0['swap_used']/1e9:.2f} GB")
    with open(os.path.join(out, "check_partial.log"), "w") as log:
        if not (a.skip_reference and os.path.exists(os.path.join(out, "reference.pt"))):
            print("\n1) full-model reference", flush=True); print(child(sm + ["--phase", "reference"], log), end="")
        print("\n2) head / middle / tail vs full model (one process, no sockets)", flush=True)
        print(child(sm + ["--phase", "split"], log), end="")
        print("\n3) memory per side (each side alone in a fresh process; MEASURED)", flush=True)
        mem = []
        for H, T in CFG["memory_splits"]:
            for side in ("client", "server"):
                r = json.loads(child(sm + ["--phase", "memory", "--side", side, "--H", str(H), "--T", str(T)], log)
                               .strip().splitlines()[-1])
                s = r["snaps"]
                row = dict(side=side, H=H, T=T, U=16 - H - T if not a.smoke else None, P=r["P"], A=r["A"],
                           footprint_loaded_gb=s["loaded"]["footprint"] / 1e9,
                           footprint_decode_gb=s["decode"]["footprint"] / 1e9,
                           peak_footprint_gb=s["decode"]["peak_footprint"] / 1e9,
                           mps_alloc_gb=s["decode"]["mps_alloc"] / 1e9, mps_driver_gb=s["decode"]["mps_driver"] / 1e9,
                           rss_gb=s["decode"]["rss"] / 1e9, modelled_gb=modelled_gb(side, H, T, r["P"], r["A"], a.smoke),
                           sys_available_start_gb=s["start"]["sys_available"] / 1e9,
                           swap_start_gb=s["start"]["swap_used"] / 1e9, swap_end_gb=s["decode"]["swap_used"] / 1e9)
                row["swap_grew"] = row["swap_end_gb"] > row["swap_start_gb"] + 0.01
                mem.append(row)
                print(f"  {side:6s} H={H} T={T}: footprint loaded {row['footprint_loaded_gb']:.3f} GB, after "
                      f"P={r['P']}+A={r['A']} {row['footprint_decode_gb']:.3f} GB (peak {row['peak_footprint_gb']:.3f}), "
                      f"tensors {row['mps_alloc_gb']:.3f} GB | modelled "
                      f"{row['modelled_gb'] if row['modelled_gb'] is None else round(row['modelled_gb'], 3)} GB | "
                      f"swap {row['swap_start_gb']:.2f}->{row['swap_end_gb']:.2f} GB", flush=True)
    rows = json.load(open(os.path.join(out, "split_rows.json")))
    sys1 = memstat.snapshot()
    summary = dict(env=env, cfg=CFG, smoke=a.smoke, all_passed=all(r["passed"] for r in rows),
                   adapters_change_output=all(any(r["differs_from_public"] for r in rows if r["adapters"] and
                                                      (r["H"], r["T"]) == hs) for hs in CFG["adapter_splits"]),
                   system_start=sys0, system_end=sys1, swap_grew=sys1["swap_used"] > sys0["swap_used"] + 10**7,
                   split_rows=rows, memory_rows=mem)
    json.dump(summary, open(os.path.join(out, "check_partial.json"), "w"), indent=1, default=str)
    for fn, rr in (("check_partial.csv", rows), ("memory.csv", mem)):
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rr for k in r})); w.writeheader(); w.writerows(rr)
    print(f"\nALL SPLIT CHECKS {'PASSED' if summary['all_passed'] else 'FAILED'}; adapters change the output: "
          f"{summary['adapters_change_output']}; system swap {sys0['swap_used']/1e9:.2f} -> {sys1['swap_used']/1e9:.2f} GB")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
