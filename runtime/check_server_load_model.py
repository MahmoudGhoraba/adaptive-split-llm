"""
Model check before the real server-load trace (MODELLED; no runtime run; 19 Nov 2026).

Replays adaptive_eval.make_trace("server_load", 200, seed 11) in the calibrated model with the SETTINGS OF THE
RUNTIME (run_trace_real.py): M2-4T, A = 64, prompt lengths mapped to the buckets {32, 128, 256, 512}, runtime option
set (no noise options), fixed per class chosen at P = 256 on nominal 4G; the adaptive estimator observes after each
request the RTT and the server-REPORTED share (exact, as the PING reply carries the emulated share) and, for split
options, the effective rate (here r_eff of the true condition; the runtime measures it from the prefill message).
Compares the requests where adaptive and fixed per class differ with step5_per_request/task4_decisions.csv
(adaptive_eval: A = 128, exact P, 3% observation noise, noise seed 0).

Run:  python check_server_load_model.py  ->  prints the comparison; check_out/server_load_model_check.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.join(HERE, "..", "step5_per_request")
sys.path.insert(0, STEP5)
import adaptive_eval as ae                                        # noqa: E402  (loads the multisplit table)
import calibrated_model as cm                                     # noqa: E402
import per_request_controller as prc                              # noqa: E402

PROF, A, BUCKETS = "M2-4T", 64, [32, 128, 256, 512]
OPTS = [o for o in prc.options() if o["kind"] != "noise"]
CL = {c["key"]: c for c in prc.CLASSES}
SPLITLIKE = ("split", "noise", "secret", "secret_u")


def replay(trace, policy, a=A, bucket=True):
    est, out = cm.Estimator(), {}
    fixed = {k: (cm.decide_cal(256, a, c, ae.NOMINAL, PROF, OPTS) or (None,))[0] for k, c in CL.items()}
    for r in trace:
        P = min(BUCKETS, key=lambda b: abs(b - r["P"])) if bucket else r["P"]
        true = {k: r[k] for k in ("rate_up", "rate_down", "rtt", "server_share")}
        if policy == "adaptive":
            o = (cm.decide_cal(P, a, CL[r["cls"]], est.cond(), PROF, OPTS) or (None,))[0]
        else:
            o = fixed[r["cls"]]
        out[r["i"]] = prc.short2(o) if o else "infeasible"
        if policy == "adaptive" and o is not None:               # runtime: no PING for infeasible requests
            est.observe(rtt=true["rtt"], share=true["server_share"])
            if o["kind"] in SPLITLIKE:
                est.observe(rate=cm.r_eff(true))
    return out


def main():
    tr = ae.make_trace("server_load", 200, 11)
    ref = {}
    for r in csv.DictReader(open(os.path.join(STEP5, "task4_decisions.csv"))):
        if r["trace"] == "server_load" and r["profile"] == PROF and r["policy"] in ("adaptive", "fixed_per_class"):
            ref.setdefault(r["policy"], {})[int(r["i"])] = r["choice"]
    d_ref = {i for i in ref["adaptive"] if ref["adaptive"][i] != ref["fixed_per_class"][i]}
    rows = []
    for label, kw in (("runtime settings (A = 64, bucketed P)", {}), ("A = 128, bucketed P", dict(a=128)),
                      ("A = 64, exact P", dict(bucket=False))):
        ad, fx = replay(tr, "adaptive", **kw), replay(tr, "fixed_per_class", **kw)
        d = {i for i in ad if ad[i] != fx[i]}
        print(f"{label}: adaptive vs fixed differ on {len(d)} requests; same set as task4_decisions.csv "
              f"({len(d_ref)}): {d == d_ref}; only here {sorted(d - d_ref)}; only in task4 {sorted(d_ref - d)}")
        for i in sorted(d | d_ref):
            rows.append(dict(setting=label, i=i, cls=tr[i]["cls"], P=tr[i]["P"], share=round(tr[i]["server_share"], 3),
                             adaptive=ad[i], fixed=fx[i], task4_adaptive=ref["adaptive"][i], task4_fixed=ref["fixed_per_class"][i]))
        if not kw:
            served = sum(v != "infeasible" for v in ad.values())
            print(f"  served {served} of {len(ad)}; adaptive choices: "
                  + ", ".join(f"{c} {n}" for c, n in sorted(__import__('collections').Counter(ad.values()).items())))
    os.makedirs(os.path.join(HERE, "check_out"), exist_ok=True)
    with open(os.path.join(HERE, "check_out", "server_load_model_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
