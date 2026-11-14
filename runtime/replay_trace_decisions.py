"""
Replay the decisions of the MEASURED real-runtime trace (check_out/task4_real_trace.csv) offline, without rerunning it:
for every request the decision is recomputed from the estimator state the run logged after the previous request
(est_rtt_ms, est_rate_mbit, est_share; first request: Estimator() prior), with the switching state following the
replayed choices. With --leak-check mean the replay must reproduce the logged choices (validation of the replay);
with --leak-check robust it shows whether the robust privacy check would change any decision of that run.

Run:  python replay_trace_decisions.py [--leak-check robust|mean]
"""
import argparse
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.join(HERE, "..", "step5_per_request")
sys.path.insert(0, STEP5)
import per_request_controller as prc
prc.load_multisplit(os.path.join(STEP5, "measured_multisplit.txt"))
import calibrated_model as cm                                                   # noqa: E402

PROF, A = "M2-4T", 64                                                           # as in run_trace_real.py
CL = {c["key"]: c for c in prc.CLASSES}
NOMINAL = dict(rate_up=prc.PHASES["lte"][1], rate_down=2.5 * prc.PHASES["lte"][1], rtt=prc.PHASES["lte"][2], server_share=1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--leak-check", choices=["robust", "mean"], default="robust")
    prc.LEAK_CHECK = ap.parse_args().leak_check
    opts = [o for o in prc.options() if o["kind"] != "noise"]
    fixed = {k: prc.short2(cm.decide_cal(256, A, c, NOMINAL, PROF, opts)[0]) for k, c in CL.items()}
    runs = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "check_out", "task4_real_trace.csv"))):
        runs[(r["policy"], r["rep"])].append(r)
    same, diffs = 0, []
    for (pol, rep), rr in sorted(runs.items()):
        rr.sort(key=lambda r: int(r["i"])); state, prev = cm.EMPTY, None
        for r in rr:
            cond = cm.Estimator().cond() if prev is None else dict(
                rate_up=float(prev["est_rate_mbit"]) * 1e6, rate_down=float(prev["est_rate_mbit"]) * 1e6,
                rtt=float(prev["est_rtt_ms"]) / 1e3, server_share=min(float(prev["est_share"]), 1.0))
            P, cls = int(r["P"]), CL[r["cls"]]
            if pol == "adaptive":
                ch = prc.short2(cm.decide_cal(P, A, cls, cond, PROF, opts)[0])
            elif pol == "switch_aware":
                o = cm.decide_switch(P, A, cls, cond, PROF, opts, state, "a")[0]
                _, state = cm.transition(state, o, "a"); ch = prc.short2(o)
            else:
                ch = fixed[r["cls"]]
            if ch == r["choice"]:
                same += 1
            else:
                diffs.append((pol, rep, r["i"], r["cls"], r["P"], r["choice"], ch))
            prev = r
    print(f"leak check {prc.LEAK_CHECK}: {same} of {same + len(diffs)} logged decisions reproduced, {len(diffs)} differ")
    for d in diffs:
        print("   policy %s rep %s request %s class %s P %s: logged %s, replay %s" % d)


if __name__ == "__main__":
    main()
