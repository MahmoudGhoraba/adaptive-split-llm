"""
Feasibility analysis of the privacy limits (MODELLED; robust privacy check; no new runs; 15 Nov 2026).

For memory budgets 2 / 3 / 4 / 8 GB and the measured profiles M2-4T, M2-2T, M2-1T (FITTED from MEASURED; the budget
is overridden), every limit combination of the grid
    eps_p (prompt leakage)  in {60, 70, 80, 90, 100} %
    eps_a (answer leakage)  in {20, 30, 40, 50, 100} %
    Q     (agreement)       in {60, 65, 70, 75, 80, 85} %
is tested: feasible = at least one option passes (robust leakage <= limits, agreement >= Q, calibrated memory <=
budget). The fastest feasible option and its calibrated latency are reported for a 256-token prompt on nominal 4G
(answer A = 128, as in adaptive_eval.py; server share 1.0). Feasibility does not depend on the profile (memory is
profile-independent); the fastest option and its latency do.
Also: the minimum robust prompt leakage over the options that USE THE SERVER (all on client always leaks 0 % when it
fits) per (eps_a, Q, budget) -> Figure 8, and the smallest single-limit relaxation that makes the "private
prompt+answer" class feasible on 2 GB (sensitivity analysis; the class limits are NOT changed).

Run:  python feasibility_analysis.py  ->  feasibility_grid.csv, feasibility_min_prompt_leak.csv, feasibility_relax.json
"""
import copy
import csv
import json
import os

import per_request_controller as pr

HERE = os.path.dirname(os.path.abspath(__file__))
pr.load_multisplit(os.path.join(HERE, "measured_multisplit.txt"))
import calibrated_model as cm                                    # noqa: E402

assert pr.LEAK_CHECK == "robust"
OPTS = pr.options()
P, A = 256, 128
NOMINAL = dict(rate_up=pr.PHASES["lte"][1], rate_down=2.5 * pr.PHASES["lte"][1], rtt=pr.PHASES["lte"][2], server_share=1.0)
BUDGETS = [2.0, 3.0, 4.0, 8.0]
PROFILES = ["M2-4T", "M2-2T", "M2-1T"]
EPS_P = [0.60, 0.70, 0.80, 0.90, 1.00]
EPS_A = [0.20, 0.30, 0.40, 0.50, 1.00]
QS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
SERVER_KINDS = ("cloud", "split", "noise", "secret", "secret_u")


def profile(base, gb):
    k = f"{base}@{gb:g}GB"
    cm.CAL["client"][k] = dict(copy.deepcopy(cm.CAL["client"][base]), mem_budget_gb=gb)
    return k


def cls(ep, ea, q):
    return dict(key="grid", title="grid", eps=ep, eps_a=ea, q=q)


def grid():
    rows = []
    for gb in BUDGETS:
        for base in PROFILES:
            prof = profile(base, gb)
            for ep in EPS_P:
                for ea in EPS_A:
                    for q in QS:
                        c = cls(ep, ea, q)
                        n = sum(cm.feasible(o, c, P, A, prof) for o in OPTS)
                        b = cm.decide_cal(P, A, c, NOMINAL, prof, OPTS)
                        rows.append(dict(budget_gb=gb, profile=base, eps_p=ep, eps_a=ea, q=q, feasible=b is not None,
                                         n_feasible_options=n, fastest=pr.short2(b[0]) if b else "infeasible",
                                         latency_s=round(b[1], 3) if b else None))
    return rows


def min_prompt_leak():
    """Minimum robust prompt leakage over server-using options with answer <= eps_a, agreement >= Q, memory <= budget."""
    rows = []
    for gb in BUDGETS:
        local_fits = cm.memory_cal(next(o for o in OPTS if o["kind"] == "local"), P, A) <= gb + 1e-9
        for ea in EPS_A:
            for q in QS:
                best = None
                for o in OPTS:
                    if o["kind"] not in SERVER_KINDS:
                        continue
                    if (o["answer_leak_r"] <= ea + 1e-12 and pr.quality_ok(o, q) and cm.memory_cal(o, P, A) <= gb + 1e-9
                            and (best is None or o["leak_r"] < best["leak_r"])):
                        best = o
                rows.append(dict(budget_gb=gb, eps_a=ea, q=q, local_fits=local_fits,
                                 min_prompt_leak_server=round(best["leak_r"], 4) if best else None,
                                 option=pr.short2(best) if best else "none"))
    return rows


def relaxations(gb=2.0):
    """Smallest single-limit relaxation that makes 'private prompt+answer' (80 / 50 / 75 %) feasible at gb."""
    c = next(x for x in pr.CLASSES if x["key"] == "both")
    mem = lambda o: cm.memory_cal(o, P, A)
    fit = [o for o in OPTS if mem(o) <= gb + 1e-9]
    def arg(cands, key, best):
        cands = list(cands)
        if not cands:
            return None
        o = best(cands, key=key)
        return dict(value=round(key(o), 4), option=pr.short2(o), leak_r=round(o["leak_r"], 4),
                    answer_leak_r=round(o["answer_leak_r"], 4), agreement=round(o["agreement"], 4), memory_gb=round(mem(o), 3))
    out = dict(
        cls=c,
        budget_gb=gb,
        relax_eps_p=arg((o for o in fit if o["answer_leak_r"] <= c["eps_a"] + 1e-12 and pr.quality_ok(o, c["q"])),
                        lambda o: o["leak_r"], min),
        relax_eps_a=arg((o for o in fit if o["leak_r"] <= c["eps"] + 1e-12 and pr.quality_ok(o, c["q"])),
                        lambda o: o["answer_leak_r"], min),
        relax_q=arg((o for o in fit if o["leak_r"] <= c["eps"] + 1e-12 and o["answer_leak_r"] <= c["eps_a"] + 1e-12),
                    lambda o: o["agreement"], max),
        relax_memory=arg((o for o in OPTS if pr.leak_ok(o, c["eps"], c["eps_a"]) and pr.quality_ok(o, c["q"])),
                         mem, min),
    )
    # same, with the MEAN-based check (shows which part is caused by the robust margin)
    pr.LEAK_CHECK = "mean"
    out["mean_check_feasible_at_budget"] = any(cm.feasible(o, c, P, A, profile("M2-2T", gb)) for o in OPTS)
    pr.LEAK_CHECK = "robust"
    return out


def main():
    g = grid()
    with open(os.path.join(HERE, "feasibility_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(g[0])); w.writeheader(); w.writerows(g)
    m = min_prompt_leak()
    with open(os.path.join(HERE, "feasibility_min_prompt_leak.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m[0])); w.writeheader(); w.writerows(m)
    r = relaxations()
    json.dump(r, open(os.path.join(HERE, "feasibility_relax.json"), "w"), indent=1)
    n = len(EPS_P) * len(EPS_A) * len(QS)
    for gb in BUDGETS:
        k = sum(x["feasible"] for x in g if x["budget_gb"] == gb and x["profile"] == "M2-4T")
        print(f"{gb:g} GB: {k}/{n} limit combinations feasible")
    print("\nclass limits (fastest option, latency at P = 256, 4G, A = 128):")
    for c in pr.CLASSES:
        for gb in BUDGETS:
            for base in PROFILES:
                x = next((x for x in g if x["budget_gb"] == gb and x["profile"] == base and abs(x["eps_p"] - c["eps"]) < 1e-9
                          and abs(x["eps_a"] - c["eps_a"]) < 1e-9 and abs(x["q"] - c["q"]) < 1e-9), None)
                if x is None:                                  # open class: Q = 95 % is outside the grid
                    b = cm.decide_cal(P, A, c, NOMINAL, profile(base, gb), OPTS)
                    x = dict(fastest=pr.short2(b[0]) if b else "infeasible", latency_s=round(b[1], 2) if b else None)
                print(f"  {c['title']:22s} {gb:g} GB {base}: {x['fastest']:32s} {x['latency_s']}")
    print("\nrelaxations, private prompt+answer at 2 GB:", json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
