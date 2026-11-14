"""
Robust vs mean-based privacy check (15 Nov 2026): compares the two adaptive_eval.py runs
  task4_decisions_meancheck.csv  (python adaptive_eval.py --leak-check mean)
  task4_decisions.csv            (python adaptive_eval.py, robust check = default)
Rows are paired: same trace, profile, policy, request and noise seed 0 (deterministic policies are exact).
Per (trace, profile, policy): requests served under each check, decisions that change, and the latency cost of the
robust check on the requests served under BOTH checks (MODELLED, calibrated model, incl. PING).
Per (profile, class), oracle policy: decision transitions mean -> robust over the three traces.

Run:  python compare_leak_check.py  ->  task4_leak_check_comparison.csv, task4_leak_check_transitions.csv
"""
import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return {(r["trace"], r["profile"], r["policy"], r["i"]): r for r in csv.DictReader(open(os.path.join(HERE, name)))}


def main():
    m, r = load("task4_decisions_meancheck.csv"), load("task4_decisions.csv")
    assert m.keys() == r.keys()
    groups = collections.defaultdict(list)
    for k in m:
        groups[k[:3]].append(k)
    out = []
    for g, ks in sorted(groups.items()):
        fm = [k for k in ks if m[k]["feasible"] == "True"]; fr = [k for k in ks if r[k]["feasible"] == "True"]
        both = [k for k in ks if m[k]["feasible"] == "True" and r[k]["feasible"] == "True"]
        lm = sum(float(m[k]["latency_s"]) for k in both); lr = sum(float(r[k]["latency_s"]) for k in both)
        out.append(dict(trace=g[0], profile=g[1], policy=g[2], requests=len(ks), served_mean=len(fm), served_robust=len(fr),
                        changed=sum(m[k]["choice"] != r[k]["choice"] for k in ks),
                        common_served=len(both), total_mean_min=round(lm / 60, 3), total_robust_min=round(lr / 60, 3),
                        cost_pct=round(100 * (lr / lm - 1), 2) if lm else None))
    with open(os.path.join(HERE, "task4_leak_check_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    tr = collections.Counter(); per = collections.Counter()
    for k in m:
        if k[2] == "oracle":
            per[(k[1], m[k]["cls"])] += 1
            if m[k]["choice"] != r[k]["choice"]:
                tr[(k[1], m[k]["cls"], m[k]["choice"], r[k]["choice"])] += 1
    rows = [dict(profile=p, cls=c, mean_choice=a, robust_choice=b, requests=n, of_class_requests=per[(p, c)])
            for (p, c, a, b), n in sorted(tr.items())]
    with open(os.path.join(HERE, "task4_leak_check_transitions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["profile", "cls", "mean_choice", "robust_choice", "requests", "of_class_requests"])
        w.writeheader(); w.writerows(rows)
    for o in out:
        if o["changed"] or o["served_mean"] != o["served_robust"]:
            print(f"{o['trace']:11s} {o['profile']:13s} {o['policy']:17s} changed {o['changed']:3d}/{o['requests']} "
                  f"served {o['served_mean']:3d}->{o['served_robust']:3d}  common {o['total_mean_min']:7.2f} -> "
                  f"{o['total_robust_min']:7.2f} min ({o['cost_pct']:+.2f}%)")
    print("\noracle transitions (mean -> robust), all three traces:")
    for x in rows:
        print(f"  {x['profile']:13s} {x['cls']:7s} {x['mean_choice']:32s} -> {x['robust_choice']:32s} {x['requests']:3d}/{x['of_class_requests']}")


if __name__ == "__main__":
    main()
