"""
Build step5_per_request/calibration.json (task 4) from the task 3 / profile measurements. Every value is labelled.

  client profiles   M2-4T = task 3 fit (check_out/task3_fit.json, all splits, P up to 1024); M2-2T, M2-1T =
                    measure_profiles.py fits (check_out/profiles.json)  -> FITTED from MEASURED
                    memory budgets per profile: ASSUMED (not measurable on one 8 GB machine)
                    sensitivity: old assumed phone (5 GFLOP/s, 2 GB) / edge (100 GFLOP/s, 8 GB) = M2-4T client times
                    scaled by f_eff(M2-4T prefill) / f_assumed  -> MODELLED (sensitivity case only)
  server            task 3 fit, MPS fp16 (FITTED)
  overhead          per-message loopback overhead m0 + m1 * bytes (FITTED)
  gap               measured idle-gap table (task 3, 8c) as slowdown factors vs gap 0 (MEASURED)
  adapters          secret-adapter overhead per client layer: (4,2) adapters - public (MEASURED, task 3)
  memory            client footprint = memory_gb() + 0.40 GB (MEASURED overhead, CPU client, 8e);
                    all on server: client = 0.29 GB (MEASURED empty-process baseline, 7a)
  reload            switching cost (FITTED, few samples): in-situ reloads of the real-runtime trace
                    (check_out/task4_real_trace.csv; the other side resident); isolated per-layer reloads
                    (check_out/reload.json) are NOT linear in the layer count (MAPE 208% / 78%) and are kept only for
                    the report. client: line through (L=2+ends, L=16+ends) in-situ means; server: free only, U = 14,
                    all on server (16 + ends) in-situ means, linear in U between 0 and 14.
                    setting a = this testbed (both sides reload); b = dedicated server, all layers resident
                    (server cost 0; MODELLED)
  ping              cost of the per-request PING with server-reported load: RTT + ping_overhead_s (MEASURED)

Run:  python build_calibration.py
"""
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_out")
DEST = os.path.join(HERE, "..", "step5_per_request", "calibration.json")


def main():
    fit = json.load(open(os.path.join(OUT, "task3_fit.json")))
    prof = json.load(open(os.path.join(OUT, "profiles.json")))["profiles"]
    steps = [json.loads(l) for l in open(os.path.join(OUT, "task3_steps.jsonl"))]

    f4 = 1 / fit["client_prefill"]["params"][0]
    client = {"M2-4T": dict(prefill=fit["client_prefill"]["params"], decode=fit["client_decode"]["params"],
                            out_s=fit["output_layer_ms"]["client"]["mean"] / 1e3, mem_budget_gb=4.0, threads=4,
                            label="FITTED from MEASURED (task 3); memory budget ASSUMED")}
    for n, b in (("2", 2.0), ("1", 2.0)):
        p = prof[n]
        client[f"M2-{n}T"] = dict(prefill=p["prefill"]["params"], decode=p["decode"]["params"],
                                  out_s=p["output_layer_ms"] / 1e3, mem_budget_gb=b, threads=int(n),
                                  label="FITTED from MEASURED (measure_profiles.py); memory budget ASSUMED")
    for name, f_ass, mem in (("assumed-phone", 5e9, 2.0), ("assumed-edge", 100e9, 8.0)):
        k = f4 / f_ass
        base = client["M2-4T"]
        client[name] = dict(prefill=[v * k for v in base["prefill"]], decode=[v * k for v in base["decode"]],
                            out_s=base["out_s"] * k, mem_budget_gb=mem, scale=k,
                            label=f"MODELLED sensitivity case: M2-4T times x {k:.2f} (assumed {f_ass/1e9:g} GFLOP/s)")

    # idle-gap slowdown factors (decode), relative to gap 0
    g = fit["gaps"]; gaps = sorted(g, key=float)
    srv0 = g[gaps[0]]["server_compute_s"]["median_ms"]
    cli = {gp: g[gp]["head_s"]["median_ms"] + g[gp]["tail_layers_s"]["median_ms"] + g[gp]["out_s"]["median_ms"] for gp in gaps}
    gap = dict(gaps_s=[float(x) for x in gaps],
               server_factor=[g[gp]["server_compute_s"]["median_ms"] / srv0 for gp in gaps],
               client_factor=[cli[gp] / cli[gaps[0]] for gp in gaps],
               label="MEASURED (task 3, 8c): (4,2), P=128; beyond the largest gap the last factor is used")

    # adapter overhead per client layer, from the (4,2) runs with and without adapters
    acc = collections.defaultdict(list)
    for s in steps:
        if s["part"] == "compute" and s["config"] == "split_4_2":
            acc[(s["variant"], s["kind"], s["P"])].append(s["head_s"] + s["tail_layers_s"])
    pre = [(st.mean(acc[("adapters", "prefill", P)]) - st.mean(acc[("public", "prefill", P)])) / (6 * P)
           for P in (32, 128, 256, 512, 1024)]
    dec = [(st.median(acc[("adapters", "decode", P)]) - st.median(acc[("public", "decode", P)])) / 6
           for P in (32, 128, 256, 512, 1024)]
    adapters = dict(prefill_s_per_layer_token=st.mean(pre), decode_s_per_layer=st.mean(dec),
                    per_P_prefill=pre, per_P_decode=dec, label="MEASURED (task 3, (4,2) adapters - public)")

    # switching cost from the in-situ reloads of the real-runtime trace
    import csv
    tr = list(csv.DictReader(open(os.path.join(OUT, "task4_real_trace.csv"))))
    cl = lambda pred: [float(r["client_load_s"]) for r in tr if float(r["client_load_s"]) > 0.05 and pred(r["choice"])]
    sv = lambda pred: [float(r["server_load_s"]) for r in tr if float(r["server_load_s"]) > 0.05 and pred(r["choice"])]
    c16, c2 = cl(lambda c: c == "All on client"), cl(lambda c: c.startswith("H=1, T=1"))
    s_all, s14, s_free = sv(lambda c: c == "All on server"), sv(lambda c: c.startswith("H=1, T=1")), sv(lambda c: c == "All on client")
    b_c = (st.mean(c16) - st.mean(c2)) / 14
    reload = dict(client=dict(a=st.mean(c2) - 2 * b_c, b_per_layer=b_c, n=[len(c2), len(c16)]),
                  server=dict(free_s=st.mean(s_free), u14_s=st.mean(s14), all_s=st.mean(s_all), n=[len(s_free), len(s14), len(s_all)]),
                  label="FITTED from MEASURED in-situ reloads (task4_real_trace.csv; few samples); setting b MODELLED")
    cal = dict(
        model_id="meta-llama/Llama-3.2-1B-Instruct", source="runtime/check_out (task 3 + measure_profiles), 14 Nov 2026",
        prefill_model="t = L*(FLOPs_layer(P)/f + c) + o  -> params [1/f, c, o]",
        decode_model="t = L*(d0 + d1*k) + o, k = context incl. new token -> params [d0, d1, o]",
        client=client,
        server=dict(prefill=fit["server_prefill"]["params"], decode=fit["server_decode"]["params"],
                    out_s=fit["output_layer_ms"]["server"]["mean"] / 1e3, label="FITTED from MEASURED (task 3), MPS fp16"),
        overhead=dict(m0_s=fit["message_overhead"]["m0_ms"] / 1e3, m1_s_per_byte=fit["message_overhead"]["m1_ns_per_byte"] / 1e9,
                      label="FITTED (loopback: serialisation + transport + server handling)"),
        wire=dict(bytes_per_token=2048 * 2, header_bytes=52, text_bytes_per_token=4),
        gap=gap, adapters=adapters, reload=reload,
        ping=dict(overhead_s=0.0002, label="MEASURED: PING + JSON reply on loopback 0.06-0.2 ms; cost = RTT + overhead"),
        memory=dict(client_overhead_gb=0.40, cloud_client_gb=0.29,
                    label="MEASURED: CPU client footprint - memory_gb() = 0.40-0.41 GB (8e); empty process 0.29 GB (7a)"),
    )
    json.dump(cal, open(DEST, "w"), indent=1)
    print(f"wrote {os.path.normpath(DEST)}")
    for k, v in client.items():
        print(f"  client {k:14s} f_prefill {1/v['prefill'][0]/1e9:7.1f} GFLOP/s, decode d0 {v['decode'][0]*1e3:6.2f} ms/layer, "
              f"output {v['out_s']*1e3:6.2f} ms, budget {v['mem_budget_gb']} GB  [{v['label'][:40]}]")
    print(f"  reload: client {reload['client']['a']:.2f} s + {reload['client']['b_per_layer']:.3f} s/layer (n={reload['client']['n']}); "
          f"server free {reload['server']['free_s']:.2f} s, U=14 {reload['server']['u14_s']:.2f} s, all {reload['server']['all_s']:.2f} s "
          f"(n={reload['server']['n']})")
    print(f"  gap factors server {[round(x, 3) for x in gap['server_factor']]}, client {[round(x, 3) for x in gap['client_factor']]}")
    print(f"  adapters: prefill {adapters['prefill_s_per_layer_token']*1e6:.2f} us/layer/token (per P {[round(x*1e6, 2) for x in pre]}), "
          f"decode {adapters['decode_s_per_layer']*1e3:.3f} ms/layer (per P {[round(x*1e3, 3) for x in dec]})")


if __name__ == "__main__":
    main()
