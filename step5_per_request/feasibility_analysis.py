"""
Feasibility analysis of the privacy limits (MODELLED; no new runs). First version 15 Nov 2026; 18 Nov 2026: all on
client is no longer a candidate (pr.candidates), answer leakage = robust proxy + generated-answer gap of task 5a
(pr.ANSWER_CHECK = "generated"), and an attacker-B sensitivity check.

For memory budgets 2 / 3 / 4 / 8 GB and the measured profiles M2-4T, M2-2T, M2-1T (FITTED from MEASURED; the budget
is overridden), every limit combination of the grid
    eps_p (prompt leakage)  in {60, 70, 80, 90, 100} %
    eps_a (answer leakage)  in {20, 30, 40, 50, 100} %
    Q     (agreement)       in {60, 65, 70, 75, 80, 85} %
is tested: feasible = at least one CANDIDATE option passes (robust leakage <= limits, agreement >= Q, calibrated
memory <= budget). The fastest feasible option and its calibrated latency are reported for a 256-token prompt on
nominal 4G (answer A = 128, as in adaptive_eval.py; server share 1.0). Feasibility does not depend on the profile.
Prompt attacker: "A" (attacker without the secret; the controller's check) and, as a SENSITIVITY CHECK only, "B"
(attacker trained on the protected model: 5b showed that 1000 known pairs bring A close to B). Options without a
secret (split, noise, all on server) have B = A.
Also: the minimum robust prompt leakage over the candidate options per (eps_a, Q, budget, attacker) -> Figure 8,
and per privacy class, budget and attacker: servable or not, and for every infeasible class the smallest single-limit
relaxation (prompt limit, answer limit, quality limit, memory budget) that makes it feasible (sensitivity analysis;
the class limits are NOT changed).

Run:  python feasibility_analysis.py  ->  feasibility_grid.csv, feasibility_min_prompt_leak.csv, feasibility_classes.csv,
                                          feasibility_relax.json
"""
import copy
import csv
import json
import os

import per_request_controller as pr

HERE = os.path.dirname(os.path.abspath(__file__))
pr.load_multisplit(os.path.join(HERE, "measured_multisplit.txt"))
import calibrated_model as cm                                    # noqa: E402

assert pr.LEAK_CHECK == "robust" and pr.ANSWER_CHECK == "generated" and not pr.LOCAL_AS_CANDIDATE
CAND = pr.candidates(pr.options())
OPTS_BY = {"A": CAND, "B": [dict(o, leak_r=o["leak_B_r"]) for o in CAND]}   # B: prompt leakage of attacker B
P, A = 256, 128
NOMINAL = dict(rate_up=pr.PHASES["lte"][1], rate_down=2.5 * pr.PHASES["lte"][1], rtt=pr.PHASES["lte"][2], server_share=1.0)
BUDGETS = [2.0, 3.0, 4.0, 8.0]
PROFILES = ["M2-4T", "M2-2T", "M2-1T"]
EPS_P = [0.60, 0.70, 0.80, 0.90, 1.00]
EPS_A = [0.20, 0.30, 0.40, 0.50, 1.00]
QS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def profile(base, gb):
    k = f"{base}@{gb:g}GB"
    cm.CAL["client"][k] = dict(copy.deepcopy(cm.CAL["client"][base]), mem_budget_gb=gb)
    return k


def cls(ep, ea, q):
    return dict(key="grid", title="grid", eps=ep, eps_a=ea, q=q)


def grid():
    rows = []
    for att, opts in OPTS_BY.items():
        for gb in BUDGETS:
            for base in PROFILES:
                prof = profile(base, gb)
                for ep in EPS_P:
                    for ea in EPS_A:
                        for q in QS:
                            c = cls(ep, ea, q)
                            n = sum(cm.feasible(o, c, P, A, prof) for o in opts)
                            b = cm.decide_cal(P, A, c, NOMINAL, prof, opts)
                            rows.append(dict(attacker=att, budget_gb=gb, profile=base, eps_p=ep, eps_a=ea, q=q,
                                             feasible=b is not None, n_feasible_options=n,
                                             fastest=pr.short2(b[0]) if b else "infeasible",
                                             latency_s=round(b[1], 3) if b else None))
    return rows


def min_prompt_leak():
    """Minimum robust prompt leakage over the candidates with answer <= eps_a, agreement >= Q, memory <= budget."""
    rows = []
    for att, opts in OPTS_BY.items():
        for gb in BUDGETS:
            for ea in EPS_A:
                for q in QS:
                    ok = [o for o in opts if o["answer_leak_r"] <= ea + 1e-12 and pr.quality_ok(o, q)
                          and cm.memory_cal(o, P, A) <= gb + 1e-9]
                    best = min(ok, key=lambda o: o["leak_r"]) if ok else None
                    rows.append(dict(attacker=att, budget_gb=gb, eps_a=ea, q=q,
                                     min_prompt_leak=round(best["leak_r"], 4) if best else None,
                                     option=pr.short2(best) if best else "none"))
    return rows


def describe(o, key):
    return dict(value=round(key(o), 4), option=pr.short2(o), leak_r=round(o["leak_r"], 4),
                answer_leak_r=round(o["answer_leak_r"], 4), agreement=round(o["agreement"], 4),
                memory_gb=round(cm.memory_cal(o, P, A), 3))


def relaxations(c, gb, opts):
    """Smallest single-limit relaxation that makes class c feasible at budget gb (None: no relaxation of that limit
    alone helps)."""
    mem = lambda o: cm.memory_cal(o, P, A)
    fit = [o for o in opts if mem(o) <= gb + 1e-9]
    arg = lambda cands, key, best: describe(best(cands, key=key), key) if cands else None
    return dict(
        eps_p=arg([o for o in fit if o["answer_leak_r"] <= c["eps_a"] + 1e-12 and pr.quality_ok(o, c["q"])],
                  lambda o: o["leak_r"], min),
        eps_a=arg([o for o in fit if o["leak_r"] <= c["eps"] + 1e-12 and pr.quality_ok(o, c["q"])],
                  lambda o: o["answer_leak_r"], min),
        q=arg([o for o in fit if pr.leak_ok(o, c["eps"], c["eps_a"])], lambda o: o["agreement"], max),
        memory=arg([o for o in opts if pr.leak_ok(o, c["eps"], c["eps_a"]) and pr.quality_ok(o, c["q"])], mem, min))


def classes():
    rows, relax = [], {}
    for att, opts in OPTS_BY.items():
        for gb in BUDGETS:
            for c in pr.CLASSES:
                r = dict(attacker=att, budget_gb=gb, cls=c["title"])
                for base in PROFILES:
                    b = cm.decide_cal(P, A, c, NOMINAL, profile(base, gb), opts)
                    r[f"fastest_{base}"] = pr.short2(b[0]) if b else "infeasible"
                    r[f"latency_s_{base}"] = round(b[1], 2) if b else None
                r["servable"] = r["fastest_M2-4T"] != "infeasible"
                if not r["servable"]:
                    rx = relaxations(c, gb, opts)
                    relax[f"{att} | {gb:g} GB | {c['title']}"] = rx
                    for k, v in rx.items():
                        r[f"relax_{k}"] = f"{v['value']} ({v['option']})" if v else "none"
                rows.append(r)
    return rows, relax


def write(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    g = grid(); write("feasibility_grid.csv", g)
    write("feasibility_min_prompt_leak.csv", min_prompt_leak())
    cl, relax = classes(); write("feasibility_classes.csv", cl)
    json.dump(dict(answer_gap_max=pr.GEN_GAP_MAX, answer_gap_from=str(pr.GEN_GAP_FROM), relaxations=relax),
              open(os.path.join(HERE, "feasibility_relax.json"), "w"), indent=1)
    n = len(EPS_P) * len(EPS_A) * len(QS)
    for att in OPTS_BY:
        for gb in BUDGETS:
            k = sum(x["feasible"] for x in g if x["attacker"] == att and x["budget_gb"] == gb and x["profile"] == "M2-4T")
            print(f"attacker {att}, {gb:g} GB: {k}/{n} limit combinations feasible")
    print("\nclasses (fastest option, latency at P = 256, 4G, A = 128; M2-4T / M2-2T / M2-1T):")
    for r in cl:
        print(f"  {r['attacker']} {r['budget_gb']:g} GB {r['cls']:22s} " + " | ".join(
            f"{r['fastest_' + b]} {r['latency_s_' + b]}" for b in PROFILES))
    print("\nsmallest single relaxations of the infeasible classes:")
    for k, rx in relax.items():
        print(f"  {k}: " + "; ".join(f"{lim} -> {v['value']} ({v['option']})" if v else f"{lim} -> none"
                                      for lim, v in rx.items()))


if __name__ == "__main__":
    main()
