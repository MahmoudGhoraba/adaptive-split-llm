"""
Thesis figures 1-9 from EXISTING results only (no new runs). Sized for A4 text width (16 cm), 300 dpi PNG.

  fig1_latency_model.png     predicted vs measured end-to-end time under emulated Wi-Fi / 4G / poor coverage:
                             calibrated model (step5_per_request/calibrated_model.latency_cal) and the assumed
                             model (per_request_controller.latency) vs MEASURED (runtime/check_out/task3_gens.csv,
                             part compose: (4,2) and (2,2), P = 256, A = 64, 3 repeats)
  fig2_decode_phases.png     MEASURED per-phase decode step, (4,2), P = 256 (task3_steps.jsonl): loopback (compute
                             part) and the three emulated scenarios (compose part)
  fig3_idle_gap.png          MEASURED slowdown vs idle gap (task3_steps.jsonl, part gaps; (4,2), P = 128)
  fig4_memory.png            MEASURED footprint vs MODELLED memory_gb() per split and side (task3_fit.json, P = 1024 + A = 64)
  fig5_policies_runtime.png  MEASURED real-runtime policy comparison (task4_real_trace.csv, 3 repeats x 30 requests)
  fig6_calibration_impact.png share of decisions that change between the assumed and the calibrated model per device
                             profile (step5_per_request/task4_calibration_impact.csv; MODELLED decisions)
  fig7_multisplit_leakage.png MEASURED prompt and answer leakage vs quality, secret head + tail, all 7 splits
                             (results/multisplit/measured_multisplit.txt; current script only; seed half-ranges)
  fig9_decision_timeline.png chosen configuration per request (adaptive policy, M2 1 thread, 2 GB) with network and
                             server load, mobile and server-load traces (step5_per_request/task4_decisions.csv; MODELLED)
  fig8_feasibility.png       minimum robust prompt leakage of server-using options per (answer limit, quality limit)
                             and memory budget (step5_per_request/feasibility_min_prompt_leak.csv; MODELLED)

Run:  python docs/figures/make_figures.py
"""
import collections
import csv
import re
import json
import os
import statistics as st
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CO = os.path.join(ROOT, "runtime", "check_out")
sys.path.insert(0, os.path.join(ROOT, "step5_per_request")); sys.path.insert(0, os.path.join(ROOT, "runtime"))

# reference categorical palette (dataviz skill), validated light-mode, fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 16 / 2.54                                     # A4 text width [inch]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
                     "font.family": "DejaVu Sans"})
SCEN = [("wifi", "Wi-Fi\n50 Mbit/s, 20 ms"), ("4g", "4G\n10 Mbit/s, 60 ms"), ("poor", "Poor coverage\n1 Mbit/s, 250 ms")]


def save(fig, name):
    fig.savefig(os.path.join(HERE, name)); plt.close(fig); print("wrote", name)


def fig1():
    import per_request_controller as pr
    import calibrated_model as cm
    from netem import SCENARIOS
    rows = [r for r in csv.DictReader(open(os.path.join(CO, "task3_gens.csv"))) if r["part"] == "compose"]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["config"], r["scenario"])].append(float(r["total_s"]))
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.55))
    marker = {"wifi": "o", "4g": "s", "poor": "^"}
    allv = []
    for (cfg, sc), v in sorted(g.items()):
        H, T = map(int, cfg.split("_")[1:]); s = SCENARIOS[sc]
        cond = dict(rate_up=s["rate"], rate_down=s["rate"], rtt=s["rtt"], server_share=1.0)
        meas = st.mean(v)
        cal = cm.latency_cal(dict(kind="split", H=H, T=T), 256, 64, cond, "M2-4T")
        ass = pr.latency(dict(kind="split", H=H, T=T), pr.request_workload(256, 64), dict(pr.PARAMS, rate_up=s["rate"],
                                                                                      rate_down=s["rate"], rtt=s["rtt"]))
        fill = dict(mfc="white") if (H, T) == (2, 2) else {}
        ax.plot(meas, cal, marker[sc], color=BLUE, ms=7, mew=1.5, **fill)
        ax.plot(meas, ass, marker[sc], color=ORANGE, ms=7, mew=1.5, **fill)
        allv += [meas, cal, ass]
    lo, hi = 3, 120
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=0)
    ax.text(70, 58, "y = x", color=INK2, fontsize=8, rotation=38)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [5, 10, 20, 50, 100]
    ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.set_yticks(ticks); ax.set_yticklabels(ticks)
    ax.set_xlabel("Measured end-to-end time [s]"); ax.set_ylabel("Predicted end-to-end time [s]")
    ax.text(3.3, 80, "assumed model\n(+35% to +387%)", color=ORANGE, fontsize=8, va="top")
    ax.text(12, 6.2, "calibrated model\n(−2.5% to +0.6%)", color=BLUE, fontsize=8, va="top")
    h = [Line2D([], [], marker=marker[k], ls="", color=INK2, ms=6, label=lab.split("\n")[0]) for k, lab in SCEN]
    h += [Line2D([], [], marker="o", ls="", color=INK2, ms=6, label="(H, T) = (4, 2)"),
          Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=6, label="(H, T) = (2, 2)")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "fig1_latency_model.png")


def fig2():
    steps = [json.loads(l) for l in open(os.path.join(CO, "task3_steps.jsonl"))]
    sel = {"loopback": [s for s in steps if s["part"] == "compute" and s["config"] == "split_4_2" and s["variant"] == "public"
                        and s["P"] == 256 and s["kind"] == "decode"]}
    for k, _ in SCEN:
        sel[k] = [s for s in steps if s["part"] == "compose" and s["config"] == "split_4_2" and s["scenario"] == k
                  and s["kind"] == "decode"]
    labels = ["Loopback\n(no emulation)"] + [lab for _, lab in SCEN]
    comp = {}
    for k, ss in sel.items():
        comp[k] = dict(client=st.median(s["head_s"] + s["ser_s"] + s["tail_s"] for s in ss) * 1e3,
                       network=st.median(s["net_s"] for s in ss) * 1e3,
                       server=st.median(s["server_total_s"] for s in ss) * 1e3,
                       step=st.median(s["step_s"] for s in ss) * 1e3, n=len(ss))
    fig, ax = plt.subplots(figsize=(W, W * 0.42))
    keys = ["loopback"] + [k for k, _ in SCEN]
    parts = [("client", "Client: head + tail + output layer (CPU)", BLUE), ("network", "Network + transport", ORANGE),
             ("server", "Server: 10 middle layers (MPS)", AQUA)]
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for p, lab, col in parts:
        v = np.array([comp[k][p] for k in keys])
        ax.barh(y, v, left=left, height=0.55, color=col, label=lab, edgecolor="white", linewidth=1.5)
        left += v
    for yi, k in zip(y, keys):
        c = comp[k]
        f = lambda v: f"{v:.1f}" if v < 1 else f"{v:.0f}"
        ax.text(left[list(keys).index(k)] + 4, yi, f"{c['step']:.0f} ms  (client {f(c['client'])}, network {f(c['network'])}, "
                f"server {f(c['server'])})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Decode step, median over all decode steps [ms]"); ax.set_xlim(0, 640)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, ncol=3, fontsize=7.5)
    save(fig, "fig2_decode_phases.png")
    return comp


def fig3():
    steps = [json.loads(l) for l in open(os.path.join(CO, "task3_steps.jsonl"))]
    gaps = sorted({s["gap_s"] for s in steps if s["part"] == "gaps"})
    series = [("server_compute_s", "Server, 10 middle layers (MPS)", BLUE, "o"),
              ("tail_layers_s", "Client, 2 tail layers (CPU)", ORANGE, "s"),
              ("head_s", "Client, 4 head layers (CPU)", AQUA, "^"),
              ("out_s", "Client, output layer (CPU)", YELLOW, "D")]
    fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.5))
    for key, lab, col, mk in series:
        vals = {g: [s[key] for s in steps if s["part"] == "gaps" and s["gap_s"] == g and s["kind"] == "decode"] for g in gaps}
        base = st.median(vals[0.0])
        med = [st.median(vals[g]) / base for g in gaps]
        lo = [np.percentile(vals[g], 10) / base for g in gaps]; hi = [np.percentile(vals[g], 90) / base for g in gaps]
        x = [g * 1e3 for g in gaps]
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, med, "-", color=col, lw=2, marker=mk, ms=6, label=f"{lab} ({base * 1e3:.1f} ms at 0 ms gap)")
        ax.text(255, med[-1], f"x{med[-1]:.2f}", va="center", fontsize=8, color=INK)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Idle gap before each decode message [ms]"); ax.set_ylabel("Median time relative to no gap [-]")
    ax.set_xticks([0, 20, 60, 250]); ax.set_xlim(-8, 285); ax.set_ylim(0.8, 2.7)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig3_idle_gap.png")


def fig4():
    fit = json.load(open(os.path.join(CO, "task3_fit.json")))
    rows = fit["memory"]["rows"]
    splits = ["split_1_1", "split_2_2", "split_4_0", "split_4_1", "split_4_2", "split_6_2", "split_8_2"]
    lab = [s.replace("split_", "(").replace("_", ", ") + ")" for s in splits]
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.45), sharey=True)
    for ax, side, title in ((axs[0], "client (CPU)", "Client (CPU): H + T layers + embedding"),
                            (axs[1], "server (MPS)", "Server (MPS): U = 16 − H − T layers")):
        m = [next(r for r in rows if r["config"] == s and r["side"] == side) for s in splits]
        x = np.arange(len(splits)); w = 0.38
        ax.bar(x - w / 2, [r["peak_gb"] for r in m], w, color=BLUE, label="Measured peak footprint", edgecolor="white", lw=1)
        ax.bar(x + w / 2, [r["modelled_gb"] for r in m], w, color=ORANGE, label="Modelled, memory_gb()", edgecolor="white", lw=1)
        for xi, r in zip(x, m):
            ax.text(xi - w / 2, r["peak_gb"] + 0.04, f"{r['peak_gb']:.2f}", ha="center", fontsize=6.5, color=INK, rotation=90, va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5); ax.set_title(title, fontsize=9)
        ax.set_xlabel("Split (H, T)"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Memory [GB]"); axs[0].set_ylim(0, 3.2)
    for v, t in ((2 * 1.0737, "2 GiB"), (1.0737, "1 GiB")):
        axs[1].axhline(v, color=INK2, lw=0.8, ls=":")
        axs[1].text(6.55, v, " " + t, fontsize=7, color=INK2, ha="left", va="center", clip_on=False)
    axs[0].legend(loc="upper left", frameon=False)
    save(fig, "fig4_memory.png")
    return {(r["config"], r["side"]): r for r in rows}


def fig5():
    # 19 Nov 2026: rerun under the current controller rule (RESULTS_SUMMARY 9m); requests no candidate can serve are
    # logged as infeasible and not run (served == False) -> excluded. Earlier run: task4_real_trace.csv (superseded).
    rows = [r for r in csv.DictReader(open(os.path.join(CO, "task4_real_trace_protected.csv"))) if r["served"] == "True"]
    pols = [("switch_aware", "Switching-aware"), ("fixed_per_class", "Fixed per class"), ("adaptive", "Adaptive")]
    stats = {}
    for p, _ in pols:
        per = []
        for rep in sorted({r["rep"] for r in rows}):
            rr = [r for r in rows if r["policy"] == p and r["rep"] == rep]
            per.append(dict(req=sum(float(r["measured_s"]) for r in rr),
                            rel=sum(float(r["client_load_s"]) + float(r["server_load_s"]) for r in rr),
                            pred=sum(float(r["predicted_true_s"]) + float(r["predicted_reload_s"]) for r in rr)))
        stats[p] = dict(req=np.mean([x["req"] for x in per]), req_sd=np.std([x["req"] for x in per], ddof=1),
                        rel=np.mean([x["rel"] for x in per]), rel_sd=np.std([x["rel"] for x in per], ddof=1),
                        tot=np.mean([x["req"] + x["rel"] for x in per]), tot_sd=np.std([x["req"] + x["rel"] for x in per], ddof=1),
                        pred=np.mean([x["pred"] for x in per]), n=len(per))
    fig, ax = plt.subplots(figsize=(W * 0.8, W * 0.45))
    x = np.arange(len(pols))
    req = [stats[p]["req"] for p, _ in pols]; rel = [stats[p]["rel"] for p, _ in pols]
    ax.bar(x, req, 0.5, color=BLUE, label="Request time (incl. PING)", edgecolor="white", lw=1.5)
    ax.bar(x, rel, 0.5, bottom=req, color=ORANGE, label="Weight reloads (switching)", edgecolor="white", lw=1.5)
    ax.errorbar(x, [stats[p]["tot"] for p, _ in pols], yerr=[stats[p]["tot_sd"] for p, _ in pols], fmt="none",
                ecolor=INK, elinewidth=1.2, capsize=4, label="± 1 s.d. of the total (3 repeats)")
    ax.plot(x, [stats[p]["pred"] for p, _ in pols], "D", color=INK, mfc="white", ms=6, mew=1.3, ls="",
            label="Predicted total (calibrated model + reload model)")
    for xi, (p, _) in zip(x, pols):
        s = stats[p]
        ax.text(xi, max(s["tot"] + s["tot_sd"], s["pred"]) + 2.5, f"{s['tot']:.1f} ± {s['tot_sd']:.1f} s", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(xi, s["req"] / 2, f"{s['req']:.1f}", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(xi, s["req"] + s["rel"] / 2, f"{s['rel']:.1f}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in pols]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Total time, 15 served requests [s]"); ax.set_ylim(0, 95); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig5_policies_runtime.png")
    return stats


PROFILE_LABEL = {"M2-4T": "M2, 4 threads, 4 GB\n(vs assumed laptop)", "M2-2T": "M2, 2 threads, 2 GB\n(vs assumed phone)",
                 "M2-1T": "M2, 1 thread, 2 GB\n(vs assumed phone)", "assumed-phone": "Assumed phone, 2 GB\n(vs assumed phone)",
                 "assumed-edge": "Assumed edge, 8 GB\n(vs assumed edge)"}


def fig6():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "step5_per_request", "task4_calibration_impact.csv"))))
    profs = list(dict.fromkeys(r["profile"] for r in rows))
    classes = list(dict.fromkeys(r["cls"] for r in rows))
    agg = {p: [sum(int(r[k]) for r in rows if r["profile"] == p) for k in ("grid_changed", "grid_points",
                                                                           "trace_changed", "trace_requests")]
           for p in profs}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, W * 0.78))
    x = np.arange(len(profs)); bw = 0.36
    for j, (lab, ic, inn, col) in enumerate([("Grid (5 prompts × 3 networks × 2 loads)", 0, 1, BLUE),
                                             ("Traces (stable, mobile, server load)", 2, 3, ORANGE)]):
        v = [100 * agg[p][ic] / agg[p][inn] for p in profs]
        xx = x + (j - 0.5) * (bw + 0.04)
        a1.bar(xx, v, bw, color=col, label=lab, edgecolor="white", lw=1.0)
        for xi, vi, p in zip(xx, v, profs):
            a1.text(xi, vi + 1.5, f"{agg[p][ic]}/{agg[p][inn]}", ha="center", va="bottom", fontsize=7, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels([PROFILE_LABEL[p] for p in profs], fontsize=6.6)
    a1.set_ylabel("Decisions that change [%]"); a1.set_ylim(0, 100); a1.grid(axis="x", visible=False)
    a1.set_title("(a) All privacy classes", loc="left")
    a1.legend(loc="upper right", frameon=False, fontsize=7)
    cc = [BLUE, ORANGE, AQUA, YELLOW]; bw2 = 0.19
    for j, c in enumerate(classes):
        v = [100 * sum(int(r["trace_changed"]) for r in rows if r["profile"] == p and r["cls"] == c) /
             max(1, sum(int(r["trace_requests"]) for r in rows if r["profile"] == p and r["cls"] == c)) for p in profs]
        a2.bar(x + (j - 1.5) * (bw2 + 0.02), v, bw2, color=cc[j], label=c, edgecolor="white", lw=1.0)
    a2.set_xticks(x); a2.set_xticklabels([PROFILE_LABEL[p] for p in profs], fontsize=6.6)
    a2.set_ylim(0, 100); a2.grid(axis="x", visible=False); a2.set_ylabel("Decisions that change [%]")
    a2.set_title("(b) Traces, per privacy class", loc="left")
    a2.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_calibration_impact.png")
    return agg


def fig7():
    import re
    path = os.path.join(ROOT, "results", "multisplit", "measured_multisplit.txt")
    src = open(path).read(); ns = {}
    exec(src.split("\n# spread")[0], {}, ns); tab = ns["SECRET_U"]
    seeds, spread = {}, {}
    for line in src.splitlines():
        m = re.match(r"\s*(\(.*?\)):.*n_seeds=(\d+)", line)
        if m:
            seeds[eval(m.group(1))] = int(m.group(2))
        m = re.match(r"#\s*(\(.*?\)):\s*(.*?)\s*\(half range", line)
        if m:
            spread[eval(m.group(1))] = {f: float(v) for f, v in re.findall(r"(\w+) \+-([0-9.]+)", m.group(2))}
    splits = sorted({k[:2] for k in tab}, key=lambda s: (s[1], s[0]))
    ramp = {1: "#86b6ef", 2: "#5598e7", 4: "#2a78d6", 6: "#1c5cab", 8: "#0d366b"}   # blue ordinal ramp, H
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.5), sharex=True)
    for ax, field, title in [(axes[0], "leak", "(a) Prompt leakage, attacker A"),
                             (axes[1], "answer_leak", "(b) Answer leakage, server runs the public tail")]:
        for (H, T) in splits:
            ks = sorted((k for k in tab if k[:2] == (H, T)), key=lambda k: k[3])
            xs = [100 * tab[k]["agreement"] for k in ks]; ys = [100 * tab[k][field] for k in ks]
            n = seeds.get(ks[0], 1)
            style = dict(color=ramp[H], ls="-" if T == 2 else "--", lw=1.6, marker="o" if T == 2 else "s", ms=5.5,
                         mfc=ramp[H] if T == 2 else "white", mew=1.4)
            ax.plot(xs, ys, label=f"H={H}, T={T}" + (f" ({n} seeds)" if n > 1 else ""), zorder=3, **style)
            if n > 1:
                ax.errorbar(xs, ys, xerr=[100 * spread[k]["agreement"] for k in ks],
                            yerr=[100 * spread[k][field] for k in ks], fmt="none", ecolor=ramp[H], elinewidth=1.0,
                            capsize=2.5, zorder=2)
        ax.axvline(75, color=INK2, ls=":", lw=1.0)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 105)
    axes[0].axhline(80, color=INK2, ls=":", lw=1.0)
    axes[0].text(94.7, 78, "80 %", fontsize=6.5, color=INK2, ha="right", va="top")
    axes[0].text(94.7, 4, "Dotted: limits\nquality 75 %\nprompt 80 %", fontsize=6.5, color=INK2,
                 ha="right", va="bottom")
    for yv in (40, 50):
        axes[1].axhline(yv, color=INK2, ls=":", lw=1.0)
    axes[1].text(94.7, 51, "50 %", fontsize=6.5, color=INK2, ha="right", va="bottom")
    axes[1].text(94.7, 39, "40 %", fontsize=6.5, color=INK2, ha="right", va="top")
    axes[1].text(94.7, 4, "Dotted: limits\nquality 75 %\nanswer 50 % (prompt+answer)\nanswer 40 % (private answer)",
                 fontsize=6.5, color=INK2, ha="right", va="bottom")
    axes[0].set_ylabel("Leakage [%]"); axes[0].set_xlim(58, 95)
    h, l = axes[0].get_legend_handles_labels()
    hl = dict(zip(l, h)); t1 = [k for k in l if k.split("T=")[1][0] == "1"]; t2 = [k for k in l if k not in t1]
    t1 += [""] * (len(t2) - len(t1)); hl[""] = Line2D([], [], ls="", marker="")
    order = [k for pair in zip(t1, t2) for k in pair]               # legend fills column-major: row 1 T=1, row 2 T=2
    fig.supxlabel("Quality: next-token agreement with the public model [%]", fontsize=9, y=0.105)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend([hl[k] for k in order], order, loc="lower center", ncol=len(t2), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.03))
    save(fig, "fig7_multisplit_leakage.png")
    return tab, seeds, spread


def fig8():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    rows = list(csv.DictReader(open(os.path.join(ROOT, "step5_per_request", "feasibility_min_prompt_leak.csv"))))
    eas = sorted({float(r["eps_a"]) for r in rows}); qs = sorted({float(r["q"]) for r in rows})
    def mat(att, gb):
        M = np.full((len(eas), len(qs)), np.nan)
        for r in rows:
            if r["attacker"] == att and float(r["budget_gb"]) == gb and r["min_prompt_leak"]:
                M[eas.index(float(r["eps_a"])), qs.index(float(r["q"]))] = 100 * float(r["min_prompt_leak"])
        return M
    for att in ("A", "B"):                                    # every candidate fits in 3 GB: 3 / 4 / 8 GB identical
        assert all(np.array_equal(mat(att, 3.0), mat(att, g), equal_nan=True) for g in (4.0, 8.0))
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#f0efec")
    classes = [(0.40, 0.75, 1.00, "A"), (1.00, 0.75, 0.80, "P"), (0.50, 0.75, 0.80, "P+A")]
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.92), sharex=True, sharey=True)
    panels = [(axes[0, 0], "A", 2.0, "(a) Attacker A (controller), 2 GB"),
              (axes[0, 1], "A", 3.0, "(b) Attacker A (controller), 3–8 GB"),
              (axes[1, 0], "B", 2.0, "(c) Attacker B (sensitivity), 2 GB"),
              (axes[1, 1], "B", 3.0, "(d) Attacker B (sensitivity), 3–8 GB")]
    for ax, att, gb, title in panels:
        M = mat(att, gb)
        im = ax.imshow(M, origin="lower", cmap=cmap, vmin=50, vmax=100, aspect="auto")
        for i in range(len(eas)):
            for j in range(len(qs)):
                v = M[i, j]
                ax.text(j, i - 0.12, "none" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color=INK2 if np.isnan(v) else ("white" if v > 75 else INK))
        for ea, q, ep, lab in classes:
            i, j = eas.index(ea), qs.index(q); v = M[i, j]
            ok = (not np.isnan(v)) and v <= 100 * ep + 1e-9
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.0, zorder=3))
            ax.text(j, i + 0.27, f"{lab} {'✓' if ok else '✗'}", ha="center", va="center", fontsize=6, zorder=4,
                    color="white" if v > 75 else INK, fontweight="bold")
        ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{100 * q:.0f}" for q in qs])
        ax.set_yticks(range(len(eas))); ax.set_yticklabels([f"{100 * e:.0f}" for e in eas])
        ax.set_title(title, loc="left", fontsize=8); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Quality limit Q: agreement ≥ [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Answer leakage limit ε_a [%]")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Min. robust prompt leakage over the candidates [%]", fontsize=7.5); cb.outline.set_visible(False)
    save(fig, "fig8_feasibility.png")


def fig9():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "step5_per_request", "task4_decisions.csv")))
            if r["profile"] == "M2-1T" and r["policy"] == "adaptive" and r["trace"] in ("mobile", "server_load")]
    chosen = {r["choice"] for r in rows}
    first = [c for c in ("infeasible", "All on server", "All on client") if c in chosen]
    order = first + sorted(chosen - set(first))                # only configurations that are chosen
    def label(c):
        if c == "infeasible":
            return "not served (infeasible)"
        m = re.match(r"H=(\d+), T=(\d+), (.*)", c)
        return f"({m.group(1)},{m.group(2)}) {m.group(3)}" if m else c.lower()
    labels = {c: label(c) for c in order}
    ccol = {"open": BLUE, "answer": ORANGE, "prompt": AQUA, "both": YELLOW}
    ctit = {c["key"]: c["title"] for c in __import__("per_request_controller").CLASSES}
    fig, axes = plt.subplots(4, 2, figsize=(W, W * 0.95), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1, 1, 1], hspace=0.12, wspace=0.08))
    for col, (tr, title) in enumerate([("mobile", "(a) Mobile trace"), ("server_load", "(b) Server-load trace")]):
        rr = sorted((r for r in rows if r["trace"] == tr), key=lambda r: int(r["i"]))
        i = np.array([int(r["i"]) for r in rr])
        ax = axes[0, col]
        for c in ccol:
            sel = [k for k, r in enumerate(rr) if r["cls"] == c]
            ax.scatter(i[sel], [order.index(rr[k]["choice"]) for k in sel], s=14, color=ccol[c], label=ctit[c],
                       marker="x" if c == "both" else "o", lw=1.2 if c == "both" else 0.6,
                       edgecolor="white" if c != "both" else None, zorder=3)
        ax.set_yticks(range(len(order))); ax.set_title(title, loc="left")
        ax.set_yticklabels([labels[o] for o in order] if col == 0 else [], fontsize=7)
        ax.set_ylim(-0.6, len(order) - 0.4); ax.grid(axis="x", visible=False)
        a = axes[1, col]
        a.plot(i, [2 / (1 / float(r["rate_up"]) + 1 / (2.5 * float(r["rate_up"]))) / 1e6 for r in rr], color=INK2,
               lw=1.0, label="true")                         # effective rate 2/(1/up + 1/down), down = 2.5 x up
        a.plot(i, [float(r["est_rate"]) / 1e6 for r in rr], color=BLUE, lw=1.0, ls="--", label="estimated")
        a.set_yscale("log"); a.set_ylabel("Eff. rate\n[Mbit/s]" if col == 0 else "", fontsize=7.5)
        a = axes[2, col]
        a.plot(i, [1e3 * float(r["rtt"]) for r in rr], color=INK2, lw=1.0)
        a.plot(i, [1e3 * float(r["est_rtt"]) for r in rr], color=BLUE, lw=1.0, ls="--")
        a.set_ylabel("RTT [ms]" if col == 0 else "", fontsize=7.5)
        a = axes[3, col]
        a.plot(i, [float(r["server_share"]) for r in rr], color=INK2, lw=1.0, label="true")
        a.plot(i, [min(float(r["est_share"]), 1.0) for r in rr], color=BLUE, lw=1.0, ls="--", label="estimated")
        a.set_ylim(0, 1.1); a.set_ylabel("Server\nshare" if col == 0 else "", fontsize=7.5)
        a.set_xlabel("Request"); a.set_xlim(-2, 201)
        for a in axes[1:, col]:
            a.tick_params(labelsize=7)
        if col == 1:
            for a in axes[:, 1]:
                a.tick_params(labelleft=False)
            for k in (1, 2, 3):
                axes[k, 1].sharey(axes[k, 0])
    h, l = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[3, 0].get_legend_handles_labels()
    fig.legend(h + h2, l + ["true condition", "estimated by the controller"], loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    save(fig, "fig9_decision_timeline.png")


if __name__ == "__main__":
    fig1(); c = fig2(); fig3(); fig4(); s = fig5(); fig6(); fig7(); fig8(); fig9()
    print({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in c.items()})
    print({k: {kk: round(float(vv), 1) for kk, vv in v.items()} for k, v in s.items()})
