"""
Task 3 analysis: fit the latency model per side and validate it (FITTED values, from MEASURED steps).

Per side (client = CPU fp16, server = MPS fp16), least squares on RELATIVE error (rows divided by the measured time):
  prefill, L layers, P tokens:  t = L * (FLOPs_layer(P) / f_pre + c_pre) + o_pre
  decode,  L layers, context k: t = L * (d0 + d1 * k) + o_dec              (k = positions already in the cache + 1)
  output layer (final norm + lm_head on the last token): mean of the measured times
FLOPs_layer = layer_flops() of step5 (same FLOP count as latency()). Observations:
  client: head_s (L = H), tail_layers_s (L = T > 0), all on client layers_s (L = 16)
  server: server_compute_s (L = U, incl. host<->device copies), all on server layers_s (L = 16)
Fit on P in FIT_P, test on TEST_P (decode: steps of the requests with those P). Per-message overhead on loopback
(serialisation + transport + server handling outside compute) = m0 + m1 * bytes per round trip.

Composition (part compose): predicted request time = sum over steps of fitted client + server + overhead +
network formula RTT + (bytes up + bytes down) * 8 / rate, compared with the measured total and TTFT; the
analytical latency() of step5 (ASSUMED speeds, as in the controller) is shown alongside.

Prefill: two variants are fitted ("flops": one FLOP rate + per-layer constant; "lin_attn": separate rates for
linear-layer and attention FLOPs); the one with the lower TRAINING error is used (chosen before looking at test).

Run:  python fit_task3.py      ->  check_out/task3_fit.json
"""
import collections
import csv
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "step5_per_request"))
import per_request_controller as pr                     # layer_flops, latency, memory_gb, request_workload, PARAMS
from netem import SCENARIOS

OUT = os.path.join(HERE, "check_out")
if os.environ.get("TASK3_QUICK"):
    OUT = os.path.join(OUT, "task3_quick")
FIT_P, TEST_P = [32, 128, 512], [256, 1024]
F_LAYER = {}


def flops_prefill(P):
    if P not in F_LAYER:
        F_LAYER[P] = float(pr.layer_flops(P, 0))
    return F_LAYER[P]


def flops_decode(k):
    return float(pr.layer_flops(1, k - 1))


def load():
    steps = [json.loads(l) for l in open(os.path.join(OUT, "task3_steps.jsonl"))]
    gens = list(csv.DictReader(open(os.path.join(OUT, "task3_gens.csv"))))
    return steps, gens


def observations(steps):
    """(side, kind, L, P, ctx, t, config) for every layer-compute measurement of the public compute part."""
    obs = []
    for s in steps:
        if s["part"] != "compute" or s["variant"] != "public":
            continue
        k = s["pos_start"] + 1                      # decode: context incl. the new token
        if s["config"].startswith("all_on_"):
            side = s["config"][7:]
            obs.append((side, s["kind"], 16, s["P"], k, s["layers_s"], s["config"]))
            continue
        obs.append(("client", s["kind"], s["H"], s["P"], k, s["head_s"], s["config"]))
        if s["T"]:
            obs.append(("client", s["kind"], s["T"], s["P"], k, s["tail_layers_s"], s["config"]))
        obs.append(("server", s["kind"], s["U"], s["P"], k, s["server_compute_s"], s["config"]))
    return obs


PREFILL_MODEL = "flops"      # set per side in main(): "flops" or "lin_attn" (lower TRAINING error wins)


def flops_lin_attn(P):
    """Linear-layer FLOPs and attention FLOPs of one layer for a P-token prefill (sum = layer_flops(P, 0))."""
    lin = 2 * pr.P_LAYER * P
    return lin, flops_prefill(P) - lin


def design(kind, L, P, k, model=None):
    if kind == "prefill":
        if (model or PREFILL_MODEL) == "lin_attn":
            lin, att = flops_lin_attn(P)
            return [L * lin, L * att, 1.0]
        return [L * flops_prefill(P), L, 1.0]
    return [L, L * k, 1.0]


def fit(rows):
    X = np.array([design(*r[1:5]) for r in rows]); y = np.array([r[5] for r in rows])
    beta, *_ = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)
    return beta


def predict(beta, kind, L, P, k):
    return float(np.dot(beta, design(kind, L, P, k)))


def predict_m(model, kind, L, P, k):
    beta, variant = model
    return float(np.dot(beta, design(kind, L, P, k, variant)))


def errors(beta, rows):
    """Relative error of the prediction per (config, P), on the mean over repeats/steps of each group."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r[6], r[2], r[3])].append((r[5], predict(beta, *r[1:5])))
    out = []
    for (cfg, L, P), v in sorted(g.items()):
        meas, pred = st.mean(a for a, _ in v), st.mean(b for _, b in v)
        out.append(dict(config=cfg, L=L, P=P, measured_ms=meas * 1e3, predicted_ms=pred * 1e3,
                        err_pct=100 * (pred - meas) / meas))
    return out


def summary(errs):
    e = [abs(x["err_pct"]) for x in errs]
    return dict(mape_pct=st.mean(e), max_abs_err_pct=max(e), n_groups=len(e)) if e else {}


def main():
    steps, gens = load()
    obs = observations(steps)
    res, models = dict(fit_P=FIT_P, test_P=TEST_P, label="FITTED from MEASURED steps"), {}
    global PREFILL_MODEL
    prefill_model = {}
    for side in ("client", "server"):
        for kind in ("prefill", "decode"):
            rows = [o for o in obs if o[0] == side and o[1] == kind]
            tr = [o for o in rows if o[3] in FIT_P]; te = [o for o in rows if o[3] in TEST_P]
            if not tr:
                continue
            variants = {}
            for variant in (["flops", "lin_attn"] if kind == "prefill" else ["decode"]):
                PREFILL_MODEL = variant if kind == "prefill" else PREFILL_MODEL
                beta = fit(tr); e_tr, e_te = errors(beta, tr), errors(beta, te)
                v = dict(params=beta.tolist(), train=summary(e_tr), test=summary(e_te), test_groups=e_te, train_groups=e_tr)
                if variant == "flops":
                    v.update(f_eff_flops=1 / beta[0], per_layer_fixed_ms=beta[1] * 1e3, per_message_ms=beta[2] * 1e3)
                if variant == "lin_attn":
                    v.update(f_eff_lin_flops=1 / beta[0], f_eff_attn_flops=1 / beta[1], per_message_ms=beta[2] * 1e3)
                variants[variant] = v
            if kind == "prefill":
                best = min(variants, key=lambda k: variants[k]["train"]["mape_pct"])      # chosen on TRAINING error
                prefill_model[side] = best; PREFILL_MODEL = best
                m = dict(variants[best], chosen=best, alternatives={k: dict(train=v["train"], test=v["test"])
                                                                     for k, v in variants.items()})
                res[f"{side}_prefill_variants"] = variants
            else:
                m = variants["decode"]
            beta = np.array(m["params"]); models[(side, kind)] = (beta, PREFILL_MODEL)
            if kind == "decode":
                m.update(d0_ms=beta[0] * 1e3, d1_us_per_token=beta[1] * 1e6, per_message_ms=beta[2] * 1e3,
                         eff_flops_at_ctx={str(k): flops_decode(k) / (beta[0] + beta[1] * k) for k in (64, 512, 1088)})
            res[f"{side}_{kind}"] = m
    # output layer
    outs = collections.defaultdict(list)
    for s in steps:
        if s["part"] == "compute" and s["variant"] == "public":
            side = "server" if s["config"] == "all_on_server" else "client"
            outs[side].append(s["out_s"])
    res["output_layer_ms"] = {k: dict(mean=st.mean(v) * 1e3, median=st.median(v) * 1e3, sd=st.pstdev(v) * 1e3)
                              for k, v in outs.items()}
    # per-message overhead on loopback (split configs): ser + net + (server_total - server_compute)
    ov = [(s["bytes_up"] + s["bytes_down"], s["ser_s"] + s["net_s"] + s["server_total_s"] - s["server_compute_s"])
          for s in steps if s["part"] == "compute" and s["config"].startswith("split_") and s["variant"] == "public"]
    if ov:
        X = np.array([[1.0, b] for b, _ in ov]); y = np.array([t for _, t in ov])
        m0, m1 = np.linalg.lstsq(X, y, rcond=None)[0]
        res["message_overhead"] = dict(m0_ms=m0 * 1e3, m1_ns_per_byte=m1 * 1e9,
                                       decode_median_ms=st.median(t for b, t in ov if b < 20_000) * 1e3)
    else:
        m0 = m1 = 0.0
    out_c = res["output_layer_ms"].get("client", {}).get("mean", 0) / 1e3

    def predict_request(H, T, P, A, scen, steps_of_request):
        """Fitted model + network formula, step by step (same step structure as the runtime)."""
        U = 16 - H - T; sc = SCENARIOS.get(scen)
        tot = ttft = 0.0
        for s in steps_of_request:
            kind, k = s["kind"], s["pos_start"] + 1
            t = predict_m(models[("client", kind)], kind, H, P, k) + predict_m(models[("server", kind)], kind, U, P, k)
            if T:
                t += predict_m(models[("client", kind)], kind, T, P, k)
            b = s["bytes_up"] + s["bytes_down"]
            t += out_c + m0 + m1 * b + ((sc["rtt"] + b * 8 / sc["rate"]) if sc else 0.0)
            tot += t; ttft = ttft or t
        return tot, ttft

    by_req = collections.defaultdict(list)
    for s in steps:
        by_req[(s["part"], s["config"], s["variant"], s["scenario"], s["gap_s"], s["P"], s["rep"])].append(s)
    comp = []
    for g in gens:
        if g["part"] not in ("compose", "compute") or not g["config"].startswith("split_") or g["variant"] != "public":
            continue
        if g["part"] == "compute" and int(g["P"]) not in TEST_P:
            continue                                     # end-to-end check on held-out lengths only
        H, T, P, A = int(g["H"]), int(g["T"]), int(g["P"]), int(g["A"])
        key = (g["part"], g["config"], g["variant"], g["scenario"], float(g["gap_s"]), P, int(g["rep"]))
        pt, pf = predict_request(H, T, P, A, g["scenario"], by_req[key])
        sc = SCENARIOS.get(g["scenario"], dict(rate=1e12, rtt=0.0))
        p = dict(pr.PARAMS, rate_up=sc["rate"], rate_down=sc["rate"], rtt=sc["rtt"])
        lat_assumed = pr.latency(dict(kind="split", H=H, T=T), pr.request_workload(P, A), p)
        comp.append(dict(part=g["part"], config=g["config"], scenario=g["scenario"], P=P, A=A, rep=int(g["rep"]),
                         measured_total_s=float(g["total_s"]), fitted_total_s=pt,
                         err_total_pct=100 * (pt - float(g["total_s"])) / float(g["total_s"]),
                         measured_ttft_s=float(g["ttft_s"]), fitted_ttft_s=pf,
                         err_ttft_pct=100 * (pf - float(g["ttft_s"])) / float(g["ttft_s"]),
                         assumed_latency_s=lat_assumed,
                         err_assumed_pct=100 * (lat_assumed - float(g["total_s"])) / float(g["total_s"]),
                         swap_grew=g["swap_grew"]))
    res["composition"] = comp
    # downclock check
    gap = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in steps:
        if s["part"] == "gaps" and s["kind"] == "decode":
            for k in ("server_compute_s", "head_s", "tail_layers_s", "out_s"):
                gap[s["gap_s"]][k].append(s[k])
    res["gaps"] = {str(gp): {k: dict(median_ms=st.median(v) * 1e3, p10_ms=np.percentile(v, 10) * 1e3,
                                     p90_ms=np.percentile(v, 90) * 1e3, n=len(v)) for k, v in d.items()}
                   for gp, d in sorted(gap.items())}
    # memory table
    mem = json.load(open(os.path.join(OUT, "task3_memory.json")))
    Pm, Am = max(r[3] for r in obs), max(int(g["A"]) for g in gens if g["part"] == "compute")
    mt = []
    for cfg, m in mem.items():
        if cfg.startswith("split_"):
            H, T = map(int, cfg.split("_")[1:]); U = 16 - H - T
            kv = 2 * pr.N_KV_HEADS * pr.HEAD_DIM * 2 * (Pm + Am)
            mt.append(dict(config=cfg, side="client (CPU)", footprint_gb=m["client"]["footprint"] / 1e9,
                           peak_gb=m["client"]["peak_footprint"] / 1e9,
                           modelled_gb=pr.memory_gb(dict(kind="split", H=H, T=T), dict(P_MAX=Pm), dict(bytes_per_value=2, answer_len=Am))))
            mt.append(dict(config=cfg, side="server (MPS)", footprint_gb=m["server"]["footprint"] / 1e9,
                           peak_gb=m["server"]["peak_footprint"] / 1e9, modelled_gb=U * (pr.P_LAYER * 2 + kv) / 1e9))
        else:
            mt.append(dict(config=cfg, side=f"one process ({m['device']})", footprint_gb=m["process"]["footprint"] / 1e9,
                           peak_gb=m["process"]["peak_footprint"] / 1e9,
                           modelled_gb=pr.memory_gb(dict(kind="local"), dict(P_MAX=Pm), dict(bytes_per_value=2, answer_len=Am))))
    for r in mt:
        r["ratio"] = r["peak_gb"] / r["modelled_gb"]
    res["memory"] = dict(P=Pm, A=Am, rows=mt)
    res["swap_grew_requests"] = [dict(part=g["part"], config=g["config"], variant=g["variant"], scenario=g["scenario"],
                                      P=g["P"], rep=g["rep"]) for g in gens if g["swap_grew"] == "True"]
    res["n_requests"] = len(gens)
    json.dump(res, open(os.path.join(OUT, "task3_fit.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("composition", "memory")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
