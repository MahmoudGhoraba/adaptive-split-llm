"""
Thesis figures 1-5 from EXISTING results only (no new runs). Sized for A4 text width (16 cm), 300 dpi PNG.

  fig1_latency_model.png     predicted vs measured end-to-end time under emulated Wi-Fi / 4G / poor coverage:
                             calibrated model (step5_per_request/calibrated_model.latency_cal) and the assumed
                             model (per_request_controller.latency) vs MEASURED (runtime/check_out/task3_gens.csv,
                             part compose: (4,2) and (2,2), P = 256, A = 64, 3 repeats)
  fig2_decode_phases.png     MEASURED per-phase decode step, (4,2), P = 256 (task3_steps.jsonl): loopback (compute
                             part) and the three emulated scenarios (compose part)
  fig3_idle_gap.png          MEASURED slowdown vs idle gap (task3_steps.jsonl, part gaps; (4,2), P = 128)
  fig4_memory.png            MEASURED footprint vs MODELLED memory_gb() per split and side (task3_fit.json, P = 1024 + A = 64)
  fig5_policies_runtime.png  MEASURED real-runtime policy comparison (task4_real_trace.csv, 3 repeats x 30 requests)

Run:  python docs/figures/make_figures.py
"""
import collections
import csv
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
    rows = list(csv.DictReader(open(os.path.join(CO, "task4_real_trace.csv"))))
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
        ax.text(xi, s["tot"] + s["tot_sd"] + 5, f"{s['tot']:.1f} ± {s['tot_sd']:.1f} s", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(xi, s["req"] / 2, f"{s['req']:.1f}", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(xi, s["req"] + s["rel"] / 2, f"{s['rel']:.1f}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in pols]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Total time, 30 requests [s]"); ax.set_ylim(0, 290); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig5_policies_runtime.png")
    return stats


if __name__ == "__main__":
    fig1(); c = fig2(); fig3(); fig4(); s = fig5()
    print({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in c.items()})
    print({k: {kk: round(float(vv), 1) for kk, vv in v.items()} for k, v in s.items()})
