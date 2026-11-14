"""
Client on CPU: which dtype? (MEASURED)  Client (4,2) = embedding + 4 head + 2 tail layers + output layer, on CPU.

  1) per dtype (fp16, bf16, fp32), fresh process each: footprint after load and after the runs; per-phase times
     of prefill (P tokens) and decode (A steps): head (4 layers), tail (2 layers), output layer; 1 warm-up, then
     `repeats` timed repeats (the server part is skipped: the head output is fed to the tail)
  2) joint: server.py (4,2) on MPS + SplitClient on CPU with the fastest dtype, one request P + A (fixed length,
     EOS ignored), system available memory and swap sampled every 0.5 s -> did swap grow?

Run:  python check_cpu_dtype.py         ->  check_out/cpu_dtype.json, cpu_dtype.csv
"""
import csv
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "check_out")
CFG = dict(H=4, T=2, P=128, A=32, P_long=512, repeats=5, dtypes=["fp16", "bf16", "fp32"], joint_P=512, joint_A=128)


def worker(dt_name):
    import torch
    import memstat
    import partial_model as pm
    dt = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[dt_name]
    snaps = dict(start=memstat.snapshot())
    mdir = pm.model_dir(); n = pm.load_config(mdir).num_hidden_layers
    m = pm.PartialLlama(mdir, pm.split_segments(CFG["H"], CFG["T"], n)[0], True, "cpu", dt)
    snaps["loaded"] = memstat.snapshot()
    g = torch.Generator().manual_seed(0); ns = time.perf_counter_ns

    @torch.no_grad()
    def one(P, A):
        caches, rows = m.new_caches(), []
        x = torch.randint(0, m.cfg.vocab_size, (1, P), generator=g); pos = 0
        for step in range(1 + A):
            t0 = ns(); h = m.run("head", m.embed_ids(x), pos, caches)
            t1 = ns(); h = m.run("tail", h, pos, caches)
            t2 = ns(); nxt = int(m.logits(h[:, -1:])[0, -1].argmax())
            t3 = ns()
            rows.append(dict(kind="prefill" if step == 0 else "decode", head_s=(t1 - t0) / 1e9, tail_s=(t2 - t1) / 1e9,
                             out_s=(t3 - t2) / 1e9))
            pos += x.shape[1]; x = torch.tensor([[nxt]])
        return rows

    one(CFG["P"], 8); one(CFG["P_long"], 0)                            # warm-up, both prompt lengths
    reps = [one(CFG["P"], CFG["A"]) for _ in range(CFG["repeats"])]
    long = [one(CFG["P_long"], 0)[0] for _ in range(CFG["repeats"])]  # prefill only, P = P_long
    snaps["after"] = memstat.snapshot()
    res = dict(dtype=dt_name, threads=torch.get_num_threads(), snaps=snaps)
    for kind in ("prefill", "decode"):
        for k in ("head_s", "tail_s", "out_s"):
            per_rep = [st.mean(r[k] for r in rows if r["kind"] == kind) for rows in reps]   # mean per repeat
            res[f"{kind}_{k}"] = st.mean(per_rep); res[f"{kind}_{k}_sd"] = st.stdev(per_rep)
    for k in ("head_s", "tail_s", "out_s"):
        res[f"prefill{CFG['P_long']}_{k}"] = st.mean(r[k] for r in long); res[f"prefill{CFG['P_long']}_{k}_sd"] = st.stdev(r[k] for r in long)
    print(json.dumps(res))


def joint(dt_name):
    import torch
    import memstat
    from client import SplitClient
    srv = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "server.py"), "--H", str(CFG["H"]), "--T",
                            str(CFG["T"])], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    port = int(srv.stdout.readline().split()[1])
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            s = memstat.snapshot(); samples.append((time.time(), s["sys_available"], s["swap_used"])); time.sleep(0.5)
    before = memstat.snapshot()
    th = threading.Thread(target=sample); th.start()
    c = SplitClient("127.0.0.1", port, CFG["H"], CFG["T"], device="cpu",
                    dtype=dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[dt_name])
    ids = torch.randint(0, 128000, (1, CFG["joint_P"]), generator=torch.Generator().manual_seed(1))
    c.generate(ids[:, :64], 8)                                        # warm-up
    g = c.generate(ids, CFG["joint_A"])                               # fixed length: no EOS
    cl, sv = memstat.snapshot(), c.server_stats()
    stop.set(); th.join(); c.close(); srv.terminate(); srv.wait(timeout=30)
    dec = g["steps"][1:]
    res = dict(dtype=dt_name, P=CFG["joint_P"], A=CFG["joint_A"], n_steps=len(g["steps"]),
               ttft_s=g["steps"][0]["step_s"], decode_median_step_s=st.median(x["step_s"] for x in dec),
               decode_median={k: st.median(x[k] for x in dec) for k in ("head_s", "rtt_s", "server_compute_s", "tail_s")},
               client_footprint=cl["footprint"], client_peak=cl["peak_footprint"], server_footprint=sv["footprint"],
               server_peak=sv["peak_footprint"], available_before=before["sys_available"],
               available_min=min(s[1] for s in samples), swap_before=before["swap_used"],
               swap_max=max(s[2] for s in samples))
    res["swap_grew"] = res["swap_max"] > res["swap_before"] + 10**7
    print(json.dumps(res))


def child(args):
    r = subprocess.run([sys.executable, __file__] + args, capture_output=True, text=True, timeout=3000)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(f"FAILED {args}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    import memstat
    gb = lambda v: v / 1e9
    s0 = memstat.snapshot()
    print(f"system at start: available {gb(s0['sys_available']):.2f} GB, swap {gb(s0['swap_used']):.2f} GB")
    rows = []
    print(f"\n1) client (4,2) on CPU, P={CFG['P']}, A={CFG['A']}, {CFG['repeats']} repeats after warm-up (mean ± sd over repeats)")
    for dt in CFG["dtypes"]:
        r = child(["--worker", dt]); s = r.pop("snaps")
        row = dict(dtype=dt, threads=r["threads"], footprint_loaded_gb=gb(s["loaded"]["footprint"]),
                   footprint_after_gb=gb(s["after"]["footprint"]), peak_gb=gb(s["after"]["peak_footprint"]),
                   rss_after_gb=gb(s["after"]["rss"]),
                   swap_start_gb=gb(s["start"]["swap_used"]), swap_end_gb=gb(s["after"]["swap_used"]),
                   **{k: v for k, v in r.items() if k not in ("dtype", "threads")})
        row["swap_grew"] = row["swap_end_gb"] > row["swap_start_gb"] + 0.01
        row["prefill_per_layer_ms"] = 1e3 * (row["prefill_head_s"] + row["prefill_tail_s"]) / (CFG["H"] + CFG["T"])
        row["decode_per_layer_ms"] = 1e3 * (row["decode_head_s"] + row["decode_tail_s"]) / (CFG["H"] + CFG["T"])
        row["decode_step_ms"] = 1e3 * (row["decode_head_s"] + row["decode_tail_s"] + row["decode_out_s"])
        L = CFG["P_long"]
        row[f"prefill{L}_ms"] = 1e3 * (row[f"prefill{L}_head_s"] + row[f"prefill{L}_tail_s"] + row[f"prefill{L}_out_s"])
        row[f"prefill{L}_per_layer_ms"] = 1e3 * (row[f"prefill{L}_head_s"] + row[f"prefill{L}_tail_s"]) / (CFG["H"] + CFG["T"])
        # client compute of one request P = P_long, A = 128 (MEASURED parts; decode at context ~P, A from this run)
        row["client_request_s"] = row[f"prefill{L}_ms"] / 1e3 + 128 * row["decode_step_ms"] / 1e3
        rows.append(row)
        print(f"  {dt}: rss {gb(s['after']['rss']):.2f} GB, prefill P={L}: {row[f'prefill{L}_ms']:.0f}"
              f"±{1e3*row[f'prefill{L}_head_s_sd']:.0f} ms ({row[f'prefill{L}_per_layer_ms']:.1f} ms/layer); client compute "
              f"of a P={L}, A=128 request ~{row['client_request_s']:.2f} s")
        print(f"  {dt}: footprint {row['footprint_loaded_gb']:.2f} -> {row['footprint_after_gb']:.2f} GB | prefill "
              f"{row['prefill_per_layer_ms']:.1f} ms/layer, output layer {row['prefill_out_s']*1e3:.1f} ms | decode "
              f"{row['decode_per_layer_ms']:.2f} ms/layer (head {row['decode_head_s']*1e3:.2f}±{row['decode_head_s_sd']*1e3:.2f}, "
              f"tail {row['decode_tail_s']*1e3:.2f}±{row['decode_tail_s_sd']*1e3:.2f}, output {row['decode_out_s']*1e3:.2f}"
              f"±{row['decode_out_s_sd']*1e3:.2f}) = {row['decode_step_ms']:.1f} ms/step | swap grew: {row['swap_grew']}",
              flush=True)
    order = sorted(rows, key=lambda r: r["client_request_s"])
    print(f"\n2) joint run, client dtypes ordered by client compute of a P={CFG['P_long']}, A=128 request: "
          f"{[r['dtype'] for r in order]}")
    joints = []
    for cand in order:                       # fastest first; fall back to the next if swap grows
        j = child(["--joint", cand["dtype"]]); joints.append(j)
        print(f"  client CPU {j['dtype']} + server MPS, P={j['P']}, A={j['A']} ({j['n_steps']} steps): TTFT "
              f"{j['ttft_s']*1e3:.0f} ms, decode {j['decode_median_step_s']*1e3:.1f} ms/step "
              f"({', '.join(f'{k} {v*1e3:.2f}' for k, v in j['decode_median'].items())}) | footprint client "
              f"{gb(j['client_footprint']):.2f} (peak {gb(j['client_peak']):.2f}) server {gb(j['server_footprint']):.2f} "
              f"(peak {gb(j['server_peak']):.2f}) GB | available {gb(j['available_before']):.2f} -> min "
              f"{gb(j['available_min']):.2f} GB, swap {gb(j['swap_before']):.2f} -> max {gb(j['swap_max']):.2f} GB, "
              f"swap grew: {j['swap_grew']}", flush=True)
        if not j["swap_grew"]:
            break
    choice = next((j["dtype"] for j in joints if not j["swap_grew"]), None)
    print(f"\nCHOICE: {choice or 'none of the dtypes ran without swap growth'}")
    json.dump(dict(cfg=CFG, system_start=s0, rows=rows, joint=joints, choice=choice),
              open(os.path.join(OUT, "cpu_dtype.json"), "w"), indent=1)
    with open(os.path.join(OUT, "cpu_dtype.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--joint":
        joint(sys.argv[2])
    else:
        main()
