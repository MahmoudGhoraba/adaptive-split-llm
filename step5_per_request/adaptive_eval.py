"""
TASK 4 evaluation of the per-request controller in the CALIBRATED model (calibrated_model.py; MODELLED latency
from FITTED/MEASURED parameters). The calibrated model plays the "true world"; policies only differ in what they
know and when they decide.

Traces (200 requests each; prompts 32-512 tokens and privacy classes from make_stream(), answer A = 128):
  stable       4G all the time (+-20% jitter), server share ~1.0
  mobile       Wi-Fi -> 4G -> poor coverage -> 4G -> Wi-Fi (make_stream phases, +-20% jitter), share ~1.0
  server_load  4G; server share 1.0 -> 0.25 (requests 67-133) -> 1.0
Policies (estimating policies send one PING per request; its cost RTT + overhead is COUNTED in their latency):
  adaptive          per request, calibrated model, ESTIMATED conditions. Default estimator: EWMA of PING RTT,
                    effective rate (prefill message, when the network was used) and the server-REPORTED load in the
                    PING reply (every request)
  adaptive_no_load  as adaptive, but server share only from requests that used the server (reported failure mode:
                    the estimate can get stuck after the server was avoided)
  switch_a / switch_b  switching-aware adaptive: also counts the predicted reload cost from the resident
                    configuration (setting a: this testbed, both sides reload; b: dedicated server, MODELLED)
  oracle            per request, calibrated model, TRUE conditions, not switching-aware, no PING
  fixed_per_class   one option per class, chosen once at nominal 4G (P = 256), no PING
  best_fixed        ONE option for every request, chosen in hindsight: fastest total among options feasible for all
  all_on_client     local
  adaptive_assumed  per request, ASSUMED model (latency()/memory_gb() of step 5 with the assumed device speed),
                    estimated conditions; evaluated in the calibrated world
Accounting: total latency of served requests (compute + network + PING), and the same + predicted reload time under
setting a and under setting b (every policy starts with nothing loaded). Violations: option breaks the class
limits or the calibrated memory budget (request not served). Decision changes: consecutive requests, and within
the same class. Gap to the oracle (same accounting).
20 noise seeds for the estimating policies (paired: same seeds for every policy).

Calibration impact (item 6) and memory-budget sensitivity (2, 3, 4 GB per measured profile) at the end.
Device profiles: M2-4T, M2-2T, M2-1T (FITTED from MEASURED; budgets ASSUMED 4/2/2 GB) and the sensitivity cases
assumed-phone / assumed-edge (MODELLED). Assumed-model counterpart by memory budget:
  M2-4T <-> laptop, M2-2T / M2-1T <-> phone, assumed-phone <-> phone, assumed-edge <-> edge.

Privacy check: --leak-check robust (default; per_request_controller.LEAK_CHECK: max over seeds, or mean + largest
observed seed half range) or mean (earlier check on the seed mean; outputs get the suffix _meancheck).

Run:  python adaptive_eval.py [--leak-check robust|mean]
        ->  task4_summary.csv, task4_decisions.csv, task4_calibration_impact.csv,
            task4_budget_sensitivity.csv, task4_policies.png, task4_mobile_timeline.png
"""
import collections
import copy
import csv
import os

import numpy as np

import argparse

import per_request_controller as pr

HERE = os.path.dirname(os.path.abspath(__file__))
pr.load_multisplit(os.path.join(HERE, "measured_multisplit.txt"))
_ap = argparse.ArgumentParser(); _ap.add_argument("--leak-check", choices=["robust", "mean"], default="robust")
pr.LEAK_CHECK = _ap.parse_known_args()[0].leak_check
SUFFIX = "" if pr.LEAK_CHECK == "robust" else "_meancheck"
import calibrated_model as cm                                    # noqa: E402  (after the multisplit merge)

OPTS = pr.options()
A = pr.PARAMS["answer_len"]
CL = {c["key"]: c for c in pr.CLASSES}
PROFILES = ["M2-4T", "M2-2T", "M2-1T", "assumed-phone", "assumed-edge"]
ASSUMED_DEV = {"M2-4T": "laptop", "M2-2T": "phone", "M2-1T": "phone", "assumed-phone": "phone", "assumed-edge": "edge"}
DEV = {d["key"]: d for d in pr.DEVICES}
NOMINAL = dict(rate_up=pr.PHASES["lte"][1], rate_down=2.5 * pr.PHASES["lte"][1], rtt=pr.PHASES["lte"][2], server_share=1.0)
NOISE = 0.03
SPLITLIKE = ("split", "noise", "secret", "secret_u")
POLICIES = ["adaptive", "adaptive_no_load", "switch_a", "switch_b", "oracle", "fixed_per_class", "best_fixed",
            "all_on_client", "adaptive_assumed"]
STOCHASTIC = ("adaptive", "adaptive_no_load", "switch_a", "switch_b", "adaptive_assumed")
PINGING = STOCHASTIC
N_SEEDS = 20


def make_trace(name, n=200, seed=11):
    net = pr.NETWORKS[1] if name == "mobile" else pr.NETWORKS[0]
    tr = pr.make_stream(n, net, seed)
    rng = np.random.default_rng(seed + 7)
    for i, r in enumerate(tr):
        base = 0.25 if name == "server_load" and n // 3 <= i < 2 * n // 3 else 1.0
        r["server_share"] = float(np.clip(base * (1 + rng.normal(0, 0.03)), 0.02, 1.0))
        r["trace"] = name
    return tr


def name(o):
    return pr.short2(o) if o else "infeasible"


def assumed_decision(P, cls, cond, prof):
    d = DEV[ASSUMED_DEV[prof.split("@")[0]]]
    p = {**pr.PARAMS, "f_client": d["f_client"], "m_client_gb": cm.CAL["client"][prof]["mem_budget_gb"], **cond}
    b = pr.decide_request(P, cls, p, OPTS)
    return b[0] if b else None


def simulate(trace, prof, policy, rng, fixed=None):
    """One row per request: option, latency in the calibrated world (incl. PING for estimating policies),
    predicted reload time under settings a and b, feasibility."""
    est, rows = cm.Estimator(), []
    state = dict(a=cm.EMPTY, b=cm.EMPTY)
    if policy == "fixed_per_class":
        fixed = {k: (cm.decide_cal(256, A, c, NOMINAL, prof, OPTS) or (None,))[0] for k, c in CL.items()}
    local = next(o for o in OPTS if o["kind"] == "local")
    for r in trace:
        cls, true = CL[r["cls"]], {k: r[k] for k in ("rate_up", "rate_down", "rtt", "server_share")}
        if policy in ("adaptive", "adaptive_no_load"):
            b = cm.decide_cal(r["P"], A, cls, est.cond(), prof, OPTS); o = b[0] if b else None
        elif policy in ("switch_a", "switch_b"):
            s = policy[-1]
            b = cm.decide_switch(r["P"], A, cls, est.cond(), prof, OPTS, state[s], s); o = b[0] if b else None
        elif policy == "oracle":
            b = cm.decide_cal(r["P"], A, cls, true, prof, OPTS); o = b[0] if b else None
        elif policy == "adaptive_assumed":
            o = assumed_decision(r["P"], cls, est.cond(), prof)
        elif policy == "fixed_per_class":
            o = fixed[r["cls"]]
        elif policy == "best_fixed":
            o = fixed
        else:
            o = local
        ok = o is not None and cm.feasible(o, cls, r["P"], A, prof)
        lat = cm.latency_cal(o, r["P"], A, true, prof) + (cm.ping_cost(true) if policy in PINGING else 0.0) if ok else None
        sw = {}
        for s in ("a", "b"):
            sw[s], state[s] = cm.transition(state[s], o if ok else None, s)
        rows.append(dict(trace=r["trace"], profile=prof, policy=policy, i=r["i"], cls=r["cls"], P=r["P"],
                         rate_up=r["rate_up"], rtt=r["rtt"], server_share=r["server_share"], choice=name(o),
                         feasible=ok, latency_s=lat, reload_a_s=sw["a"], reload_b_s=sw["b"],
                         est_rtt=est.rtt, est_rate=est.rate, est_share=est.share))
        # what the runtime observes after this request (noisy), used for the NEXT decision
        if policy in PINGING:
            est.observe(rtt=true["rtt"] * (1 + rng.normal(0, NOISE)))
            if policy != "adaptive_no_load":                       # server-reported load in the PING reply
                est.observe(share=true["server_share"] * (1 + rng.normal(0, NOISE)))
            elif o is not None and o["kind"] in SPLITLIKE + ("cloud",):
                est.observe(share=true["server_share"] * (1 + rng.normal(0, NOISE)))
            if o is not None and o["kind"] in SPLITLIKE:
                est.observe(rate=cm.r_eff(true) * (1 + rng.normal(0, NOISE)))
    return rows


def best_fixed_option(trace, prof):
    best = None
    for o in pr.candidates(OPTS):                                  # all on client only as the reference baseline
        if not all(cm.feasible(o, CL[r["cls"]], r["P"], A, prof) for r in trace):
            continue
        tot = sum(cm.latency_cal(o, r["P"], A, r, prof) for r in trace)
        if best is None or tot < best[1]:
            best = (o, tot)
    return best[0] if best else None


def changes(rows):
    ch = sum(rows[i]["choice"] != rows[i - 1]["choice"] for i in range(1, len(rows)))
    last, within = {}, 0
    for r in rows:
        if r["cls"] in last and last[r["cls"]] != r["choice"]:
            within += 1
        last[r["cls"]] = r["choice"]
    return ch, within


def metrics(rows, orc):
    served = [r for r in rows if r["feasible"]]
    ch, chc = changes(rows)
    m = dict(viol=len(rows) - len(served), served=len(served), ch=ch, chc=chc)
    for acc, extra in (("compute", None), ("a", "reload_a_s"), ("b", "reload_b_s")):
        tot = lambda rr: sum(r["latency_s"] + (r[extra] if extra else 0.0) for r in rr if r["feasible"])
        both = [(r, q) for r, q in zip(rows, orc) if r["feasible"] and q["feasible"]]
        m[f"total_{acc}"] = tot(rows) / 60
        m[f"gap_{acc}"] = 100 * (tot([r for r, _ in both]) / tot([q for _, q in both]) - 1) if both else None
    m["reload_a"] = sum(r["reload_a_s"] for r in served) / 60; m["reload_b"] = sum(r["reload_b_s"] for r in served) / 60
    return m


def evaluate(trace, prof, pols=POLICIES, n_seeds=N_SEEDS, keep_rows=False):
    bf = best_fixed_option(trace, prof)
    orc = simulate(trace, prof, "oracle", np.random.default_rng(0))
    out, dec = [], []
    for pol in pols:
        ms = []
        for sd in (range(n_seeds) if pol in STOCHASTIC else [0]):
            rows = orc if pol == "oracle" else simulate(trace, prof, pol, np.random.default_rng(1000 + sd),
                                                         fixed=bf if pol == "best_fixed" else None)
            if sd == 0 and keep_rows:
                dec += rows
            ms.append(metrics(rows, orc))
        s = dict(trace=trace[0]["trace"], profile=prof, policy=pol, n_seeds=len(ms), fixed_choice=name(bf) if pol == "best_fixed" else "")
        for k in ms[0]:
            v = [m[k] for m in ms if m[k] is not None]
            s[k] = float(np.mean(v)) if v else None
            if k.startswith("gap_") and v:
                s[k + "_min"], s[k + "_max"] = float(np.min(v)), float(np.max(v))
        out.append(s)
    return out, dec


def calibration_impact():
    """Item 6: assumed vs calibrated decisions on a grid and on the traces (true conditions)."""
    rows, Ps = [], [32, 128, 256, 512, 1024]
    nets = {k: dict(rate_up=v[1], rate_down=2.5 * v[1], rtt=v[2]) for k, v in pr.PHASES.items()}
    for prof in PROFILES:
        for ck, cls in CL.items():
            n = diff = 0; trans = collections.Counter()
            for P in Ps:
                for nk, net in nets.items():
                    for share in (1.0, 0.25):
                        cond = {**net, "server_share": share}
                        a = name(assumed_decision(P, cls, cond, prof))
                        b = cm.decide_cal(P, A, cls, cond, prof, OPTS); c = name(b[0] if b else None)
                        n += 1; diff += a != c
                        if a != c:
                            trans[(a, c)] += 1
            tn = td = 0
            for tname in ("stable", "mobile", "server_load"):
                for r in make_trace(tname):
                    if r["cls"] != ck:
                        continue
                    cond = {k: r[k] for k in ("rate_up", "rate_down", "rtt", "server_share")}
                    b = cm.decide_cal(r["P"], A, cls, cond, prof, OPTS)
                    tn += 1; td += name(assumed_decision(r["P"], cls, cond, prof)) != name(b[0] if b else None)
            top = "; ".join(f"{a} -> {c} ({k})" for (a, c), k in trans.most_common(2))
            rows.append(dict(profile=prof, assumed_counterpart=ASSUMED_DEV[prof], cls=cls["title"], grid_points=n,
                             grid_changed=diff, trace_requests=tn, trace_changed=td, main_changes=top))
    return rows


def budget_sensitivity():
    """Memory budgets 2 / 3 / 4 GB for each measured profile (the 4/2/2 GB defaults are assumptions)."""
    rows = []
    for base in ("M2-4T", "M2-2T", "M2-1T"):
        for gb in (2.0, 3.0, 4.0):
            p = f"{base}@{gb:g}GB"
            cm.CAL["client"][p] = dict(copy.deepcopy(cm.CAL["client"][base]), mem_budget_gb=gb)
            for tname in ("mobile", "server_load"):
                res, _ = evaluate(make_trace(tname), p, pols=["adaptive", "switch_a", "oracle", "fixed_per_class",
                                                               "all_on_client"], n_seeds=5)
                for s in res:
                    rows.append(dict(base_profile=base, budget_gb=gb, **{k: s[k] for k in
                                ("trace", "policy", "total_compute", "total_a", "total_b", "gap_compute", "gap_a", "viol")}))
    return rows


def main():
    summary, dec = [], []
    for tname in ("stable", "mobile", "server_load"):
        trace = make_trace(tname)
        for prof in PROFILES:
            s, d = evaluate(trace, prof, keep_rows=True)
            summary += s; dec += d
    keys = sorted({k for s in summary for k in s}, key=lambda k: (k not in ("trace", "profile", "policy"), k))
    with open(os.path.join(HERE, "task4_summary" + SUFFIX + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    with open(os.path.join(HERE, "task4_decisions" + SUFFIX + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dec[0])); w.writeheader(); w.writerows(dec)
    fmt = lambda v: f"{v:+6.2f}%" if v is not None else "   -   "
    for s in summary:
        if s["viol"] >= 200:
            print(f"{s['trace']:11s} {s['profile']:13s} {s['policy']:17s} infeasible"); continue
        print(f"{s['trace']:11s} {s['profile']:13s} {s['policy']:17s} compute {s['total_compute']:6.2f} min ({fmt(s['gap_compute'])}) | "
              f"+reload a {s['total_a']:6.2f} ({fmt(s['gap_a'])}) | +reload b {s['total_b']:6.2f} ({fmt(s['gap_b'])}) | "
              f"changes {s['ch']:5.1f} in class {s['chc']:5.1f} | viol {s['viol']:4.1f}" + (f" | {s['fixed_choice']}" if s["fixed_choice"] else ""))
    imp = calibration_impact()
    with open(os.path.join(HERE, "task4_calibration_impact" + SUFFIX + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(imp[0])); w.writeheader(); w.writerows(imp)
    print("\ncalibration impact (assumed vs calibrated decisions):")
    for r in imp:
        print(f"  {r['profile']:13s} ({r['assumed_counterpart']:6s}) {r['cls']:22s} grid {r['grid_changed']:2d}/{r['grid_points']} "
              f"| traces {r['trace_changed']:3d}/{r['trace_requests']:3d} | {r['main_changes']}")
    bs = budget_sensitivity()
    with open(os.path.join(HERE, "task4_budget_sensitivity" + SUFFIX + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bs[0])); w.writeheader(); w.writerows(bs)
    print("\nmemory-budget sensitivity (5 seeds):")
    for r in bs:
        print(f"  {r['base_profile']} {r['budget_gb']:g} GB {r['trace']:11s} {r['policy']:15s} " +
              ("infeasible" if r["viol"] >= 200 else f"compute {r['total_compute']:6.2f} min, +reload a {r['total_a']:6.2f}, "
               f"gap compute {fmt(r['gap_compute'])}, gap a {fmt(r['gap_a'])}, viol {r['viol']:.0f}"))
    plots(summary, dec)


def plots(summary, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    pols = [("adaptive", "Adaptive (default estimator)", "#111"), ("switch_a", "Switching-aware (a)", "#5b8e2f"),
            ("switch_b", "Switching-aware (b)", "#9bc26b"), ("oracle", "Oracle (true)", "#7a3cc9"),
            ("fixed_per_class", "Fixed per class", "#0e6e6e"), ("all_on_client", "All on client", "#2a6fdb"),
            ("adaptive_assumed", "Adaptive, assumed model", "#e0533d")]
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
    for row, acc, lab in ((0, "total_compute", "compute + network + PING"), (1, "total_a", "+ reload, setting a (testbed)")):
        for ax, tname in zip(axs[row], ("stable", "mobile", "server_load")):
            w = 0.11
            for j, (k, l, col) in enumerate(pols):
                v = [next(s for s in summary if s["trace"] == tname and s["profile"] == p and s["policy"] == k) for p in PROFILES]
                xs = np.arange(len(PROFILES)) + (j - 3) * w
                ax.bar(xs, [x[acc] if x["viol"] < 200 else 0 for x in v], w, color=col, label=l)
            ax.set_xticks(range(len(PROFILES))); ax.set_xticklabels(PROFILES, fontsize=8, rotation=15)
            ax.set_title(f"{tname}: {lab}", fontsize=9.5); ax.grid(alpha=.3, axis="y"); ax.set_yscale("log")
        axs[row][0].set_ylabel("Total latency, 200 requests [min] (log)")
    axs[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Task 4: policies in the calibrated model (MODELLED from FITTED/MEASURED parameters), A = 128; "
                 "missing bar = infeasible (all on client does not fit the budget)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "task4_policies" + SUFFIX + ".png"), dpi=200); plt.close(fig)

    rows = [r for r in dec if r["trace"] == "mobile" and r["profile"] == "M2-4T"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5, 1.3]))
    x = np.arange(200)
    ax[0].plot(x, [r["rate_up"] / 1e6 for r in by["oracle"]], color="#444", label="true uplink")
    ax[0].plot(x, [r["est_rate"] / 1e6 for r in by["adaptive"]], color="#d98a1c", ls="--", label="estimated r_eff")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Mbit/s"); ax[0].legend(frameon=False, fontsize=8)
    a2 = ax[0].twinx(); a2.plot(x, [r["rtt"] * 1e3 for r in by["oracle"]], color="#2a6fdb", lw=1)
    a2.plot(x, [r["est_rtt"] * 1e3 for r in by["adaptive"]], color="#2a6fdb", lw=1, ls="--"); a2.set_ylabel("RTT [ms]", color="#2a6fdb")
    names = sorted({r["choice"] for p in ("adaptive", "switch_a") for r in by[p]})
    pal = plt.get_cmap("tab20")
    cols = {nm: pal(i % 20) for i, nm in enumerate(names)}
    ykeys = list(CL)
    for r in by["adaptive"]:
        ax[1].scatter(r["i"], ykeys.index(r["cls"]) + 0.12, color=cols[r["choice"]], s=16, marker="o")
    for r in by["switch_a"]:
        ax[1].scatter(r["i"], ykeys.index(r["cls"]) - 0.12, color=cols[r["choice"]], s=16, marker="x")
    ax[1].set_yticks(range(len(ykeys))); ax[1].set_yticklabels([CL[k]["title"] for k in ykeys])
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=cols[n], label=n) for n in names], frameon=False, fontsize=7,
                 loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax[1].set_title("Choice per request (o = adaptive, x = switching-aware, setting a)", fontsize=10)
    for k, lab, col in pols:
        y = np.cumsum([(r["latency_s"] + r["reload_a_s"]) if r["latency_s"] is not None else 0 for r in by[k]]) / 60
        ax[2].plot(x, y, color=col, label=lab, lw=2 if k in ("adaptive", "switch_a") else 1.3)
    ax[2].set_ylabel("Cumulative latency incl. reload (a) [min]"); ax[2].set_xlabel("Request"); ax[2].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.suptitle("Mobile trace, client M2-4T (calibrated model; MODELLED)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "task4_mobile_timeline" + SUFFIX + ".png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
