"""
Decision-diversity analysis (MODELLED; robust privacy check; existing traces and profiles; no new runs; 15 Nov 2026).

Input: task4_decisions.csv (adaptive_eval.py, robust check; noise seed 0 for the estimating policies).
1. Per (policy, trace, profile): distinct configurations chosen, switches (consecutive requests with a different
   choice; infeasible counts as a state), share of requests per configuration group (split = any split / noise /
   secret option; local = all on client; server = all on server; infeasible = not served).
2. Cause of every switch, for the policies whose decision is a function of the current request only
   (oracle: true conditions; adaptive: the ESTIMATED conditions it used, est_rtt / est_rate / est_share):
   memory limit  - the previous configuration no longer fits the memory budget for the new prompt length;
   otherwise counterfactual: starting from the previous request's inputs, ONE factor at a time is set to the new
   request's value (privacy class; prompt length; network = rate + RTT; server load = server share). The factors
   that alone change the decision are the cause; two or more such factors -> "several"; none alone -> "interaction".
   The replay reproduces every logged decision (checked). switch_a is reported in part 1 only (its decision also
   depends on the resident configuration).
3. Within-class adaptation: the same, restricted to consecutive requests OF THE SAME CLASS (each class's own
   sub-sequence), i.e. how the decision for one class follows network, server load and prompt length.
4. Adapted requests: share of requests whose decision differs from the decision for the same class and prompt
   length at NOMINAL conditions (4G, server share 1) - i.e. requests where the conditions changed the choice
   (a phase-long change counts for every request of the phase, not only at its two switches).

Run:  python decision_diversity.py  ->  task4_diversity.csv, task4_switch_causes.csv
"""
import collections
import csv
import os

import per_request_controller as pr

HERE = os.path.dirname(os.path.abspath(__file__))
pr.load_multisplit(os.path.join(HERE, "measured_multisplit.txt"))
import calibrated_model as cm                                    # noqa: E402

assert pr.LEAK_CHECK == "robust"
OPTS = pr.options()
A = pr.PARAMS["answer_len"]
CL = {c["key"]: c for c in pr.CLASSES}
BY_NAME = {pr.short2(o): o for o in OPTS}
POLICIES = ["adaptive", "oracle", "switch_a"]
ATTRIBUTED = ["adaptive", "oracle"]
NOMINAL = dict(rate_up=pr.PHASES["lte"][1], rate_down=2.5 * pr.PHASES["lte"][1], rtt=pr.PHASES["lte"][2])
CAUSES = ["privacy class", "prompt length", "network", "server load", "memory limit", "several", "interaction"]


def group(choice):
    if choice == "infeasible":
        return "infeasible"
    k = BY_NAME[choice]["kind"]
    return "local" if k == "local" else "server" if k == "cloud" else "split"


def inputs(r, policy):
    if policy == "oracle":
        net = dict(rate_up=float(r["rate_up"]), rate_down=2.5 * float(r["rate_up"]), rtt=float(r["rtt"]))
        share = float(r["server_share"])
    else:
        net = dict(rate_up=float(r["est_rate"]), rate_down=float(r["est_rate"]), rtt=float(r["est_rtt"]))
        share = min(float(r["est_share"]), 1.0)
    return dict(cls=r["cls"], P=int(r["P"]), net=net, share=share)


def decide(x, prof):
    b = cm.decide_cal(x["P"], A, CL[x["cls"]], {**x["net"], "server_share": x["share"]}, prof, OPTS)
    return pr.short2(b[0]) if b else "infeasible"


def cause(prev_r, r, policy, prof, detail=False):
    x0, x1 = inputs(prev_r, policy), inputs(r, policy)
    assert decide(x0, prof) == prev_r["choice"] and decide(x1, prof) == r["choice"]
    if prev_r["choice"] != "infeasible" and x0["cls"] == x1["cls"]:
        o = BY_NAME[prev_r["choice"]]
        if cm.memory_cal(o, x1["P"], A) > cm.CAL["client"][prof]["mem_budget_gb"] + 1e-9:
            return "memory limit"
    flips = []
    for f, key in (("privacy class", "cls"), ("prompt length", "P"), ("network", "net"), ("server load", "share")):
        if x0[key] != x1[key] and decide({**x0, key: x1[key]}, prof) != prev_r["choice"]:
            flips.append(f)
    if detail and len(flips) > 1:
        return "several (" + " + ".join(flips) + ")"
    return flips[0] if len(flips) == 1 else "several" if flips else "interaction"


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "task4_decisions.csv"))))
    seqs = collections.defaultdict(list)
    for r in rows:
        if r["policy"] in POLICIES:
            seqs[(r["policy"], r["trace"], r["profile"])].append(r)
    div, causes = [], []
    for (pol, tr, prof), rr in sorted(seqs.items()):
        rr.sort(key=lambda r: int(r["i"]))
        n = len(rr); ch = [r["choice"] for r in rr]
        g = collections.Counter(group(c) for c in ch)
        sw = [i for i in range(1, n) if ch[i] != ch[i - 1]]
        d = dict(policy=pol, trace=tr, profile=prof, requests=n, distinct=len(set(ch) - {"infeasible"}),
                 switches=len(sw), **{f"share_{k}": round(g[k] / n, 3) for k in ("split", "local", "server", "infeasible")},
                 configurations="; ".join(f"{c} ({k})" for c, k in collections.Counter(ch).most_common()))
        if pol in ATTRIBUTED:
            cc = collections.Counter(cause(rr[i - 1], rr[i], pol, prof) for i in sw)
            for c in CAUSES:
                d[f"cause_{c}"] = cc[c]
            for i in sw:
                causes.append(dict(policy=pol, trace=tr, profile=prof, scope="consecutive", i=rr[i]["i"], prev=ch[i - 1],
                                   new=ch[i], cause=cause(rr[i - 1], rr[i], pol, prof, detail=True)))
            wc = collections.Counter()
            for c in CL:                                           # within-class sub-sequences
                sub = [r for r in rr if r["cls"] == c]
                for a, b in zip(sub, sub[1:]):
                    if a["choice"] != b["choice"]:
                        k = cause(a, b, pol, prof, detail=True); wc[k.split(" (")[0]] += 1
                        causes.append(dict(policy=pol, trace=tr, profile=prof, scope="within_class", i=b["i"],
                                           prev=a["choice"], new=b["choice"], cause=k))
            d["within_class_switches"] = sum(wc.values())
            nom = dict(cls=None, P=None, net=dict(rate_up=NOMINAL["rate_up"], rate_down=NOMINAL["rate_down"],
                                                  rtt=NOMINAL["rtt"]), share=1.0)
            d["adapted_requests"] = sum(decide({**nom, "cls": r["cls"], "P": int(r["P"])}, prof) != r["choice"] for r in rr)
            for c in CAUSES[1:]:
                d[f"within_{c}"] = wc[c]
        div.append(d)
    keys = list(dict.fromkeys(k for d in div for k in d))
    with open(os.path.join(HERE, "task4_diversity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(div)
    with open(os.path.join(HERE, "task4_switch_causes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(causes[0])); w.writeheader(); w.writerows(causes)
    for d in div:
        cs = " ".join(f"{c.split()[0][:6]} {d[f'cause_{c}']}" for c in CAUSES if d.get(f"cause_{c}")) if d["policy"] in ATTRIBUTED else ""
        print(f"{d['policy']:9s} {d['trace']:11s} {d['profile']:13s} distinct {d['distinct']:2d} switches {d['switches']:3d} "
              f"split {d['share_split']:.2f} local {d['share_local']:.2f} server {d['share_server']:.2f} "
              f"infeas {d['share_infeasible']:.2f} | within-class {d.get('within_class_switches', '')} "
              f"adapted {d.get('adapted_requests', '')} | {cs}")
    print("\nswitch causes, all traces and profiles (consecutive requests):")
    for pol in ATTRIBUTED:
        cc = collections.Counter(c["cause"].split(" (")[0] for c in causes if c["policy"] == pol and c["scope"] == "consecutive")
        n = sum(cc.values())
        print(f"  {pol:9s} {n} switches: " + ", ".join(f"{c} {cc[c]} ({100 * cc[c] / n:.1f}%)" for c in CAUSES if cc[c]))
        for prof in sorted({c["profile"] for c in causes}):
            cp = collections.Counter(c["cause"].split(" (")[0] for c in causes if c["policy"] == pol and c["profile"] == prof
                                     and c["scope"] == "consecutive")
            m = sum(cp.values())
            print(f"     {prof:13s} {m:4d}: " + ", ".join(f"{c} {100 * cp[c] / m:.0f}%" for c in CAUSES if cp[c]))

        sev = collections.Counter(c["cause"] for c in causes if c["policy"] == pol and c["scope"] == "consecutive"
                                  and c["cause"].startswith("several"))
        print("     several:", dict(sev))
    print("\nwithin-class switches (same class, consecutive requests of that class):")
    for pol in ATTRIBUTED:
        for prof in sorted({c["profile"] for c in causes}):
            for tr in ("stable", "mobile", "server_load"):
                cc = collections.Counter(c["cause"] for c in causes if c["policy"] == pol and c["profile"] == prof
                                         and c["trace"] == tr and c["scope"] == "within_class")
                if cc:
                    print(f"  {pol:9s} {prof:13s} {tr:11s} {sum(cc.values()):3d}: {dict(cc)}")
        cc = collections.Counter(c["cause"].split(" (")[0] for c in causes if c["policy"] == pol and c["scope"] == "within_class")
        n = sum(cc.values())
        print(f"  {pol} total {n}: " + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in cc.most_common()))


if __name__ == "__main__":
    main()
