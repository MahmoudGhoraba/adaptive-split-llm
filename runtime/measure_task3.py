"""
Task 3 measurement campaign (MEASURED). Client = CPU fp16, server = MPS fp16, wire fp16, TCP loopback.
Timing protocol of timing.py: prompt buckets warmed up (one full-length warm-up per bucket), fixed A (EOS ignored),
memory/swap logged around every request (swap_grew flag).

  compute   splits (1,1) (2,2) (4,1) (4,2) (6,2) (8,2) (4,0): P = 32, 128, 256, 512, 1024, A = 64, 5 repeats,
            public weights; (4,2) also with secret adapters (untrained, seeds 2000/2001; overhead only)
            + all on client (16 layers + output layer, CPU fp16, one process) and all on server (MPS fp16, one process)
  gaps      server downclock check: (4,2), P = 128, A = 64, idle gap of 0 / 20 / 60 / 250 ms before every decode
            message, no network emulation, 3 repeats
  compose   end-to-end validation: (4,2) and (2,2) x Wi-Fi / 4G / poor (application-level emulation), P = 256,
            A = 64, 1 warm-up + 3 repeats

Run:  python measure_task3.py [compute gaps compose]      (default: all three, in this order)
Writes check_out/task3_steps.jsonl (every step), task3_gens.csv (every request), task3_memory.json, task3_log.txt
"""
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import partial_model as pm
import timing
from check_partial import env_info

OUT = os.path.join(HERE, "check_out")
CFG = dict(splits=[(1, 1), (2, 2), (4, 1), (4, 2), (6, 2), (8, 2), (4, 0)], buckets=[32, 128, 256, 512, 1024],
           A=64, repeats=5, adapter_split=(4, 2), adapter_seeds=(2000, 2001),
           gaps=[0.0, 0.020, 0.060, 0.250], gap_P=128, gap_repeats=3,
           compose_splits=[(4, 2), (2, 2)], compose_scenarios=["wifi", "4g", "poor"], compose_P=256, compose_repeats=3,
           client_device="cpu", client_dtype="fp16", server_device="mps", server_dtype="fp16")
if os.environ.get("TASK3_QUICK"):         # code test: tiny settings, separate output folder
    OUT = os.path.join(OUT, "task3_quick")
    CFG.update(splits=[(4, 2)], buckets=[32], A=4, repeats=1, gaps=[0.0, 0.02], gap_repeats=1,
               compose_splits=[(4, 2)], compose_scenarios=["wifi"], compose_repeats=1, compose_P=32)
STEPS, GENS, MEM = (os.path.join(OUT, f) for f in ("task3_steps.jsonl", "task3_gens.csv", "task3_memory.json"))
GEN_FIELDS = ["part", "config", "H", "T", "U", "variant", "scenario", "gap_s", "P", "A", "rep", "ttft_s", "total_s",
              "decode_median_step_s", "bytes_up", "bytes_down", "sys_available_before", "sys_available_after",
              "swap_before", "swap_after", "swap_grew", "client_footprint", "client_rss"]


def record(meta, g):
    """Append the steps and one request row."""
    with open(STEPS, "a") as f:
        for s in g["steps"]:
            f.write(json.dumps(dict(meta, **s)) + "\n")
    dec = g["steps"][1:]
    row = dict(meta, ttft_s=g["steps"][0]["step_s"], total_s=sum(s["step_s"] for s in g["steps"]),
               decode_median_step_s=st.median(s["step_s"] for s in dec) if dec else None,
               bytes_up=sum(s.get("bytes_up", 0) for s in g["steps"]), bytes_down=sum(s.get("bytes_down", 0) for s in g["steps"]),
               **g["mem"])
    new = not os.path.exists(GENS)
    with open(GENS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GEN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def save_mem(key, value):
    d = json.load(open(MEM)) if os.path.exists(MEM) else {}
    d[key] = value
    json.dump(d, open(MEM, "w"), indent=1)


def start_server(H, T, log, scenario=None):
    from netem import SCENARIOS
    args = [sys.executable, "-u", os.path.join(HERE, "server.py"), "--H", str(H), "--T", str(T),
            "--device", CFG["server_device"], "--dtype", CFG["server_dtype"]]
    if scenario:
        sc = SCENARIOS[scenario]
        args += ["--rate-mbit", str(sc["rate"] / 1e6), "--delay-ms", str(sc["rtt"] / 2 * 1e3)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=log, text=True)
    line = p.stdout.readline().strip()
    if not line.startswith("READY"):
        p.kill(); raise RuntimeError(f"server did not start: {line!r}")
    return p, int(line.split()[1])


def client(H, T, port, netem=None):
    from client import SplitClient
    return SplitClient("127.0.0.1", port, H, T, device=CFG["client_device"],
                       dtype=dict(fp16=torch.float16, fp32=torch.float32)[CFG["client_dtype"]], netem=netem)


def say(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "task3_log.txt"), "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------------------- children
def child_split(H, T):
    log = open(os.path.join(OUT, "task3_server.log"), "a")
    srv, port = start_server(H, T, log)
    c = client(H, T, port)
    variants = ["public"] + (["adapters"] if (H, T) == CFG["adapter_split"] else [])
    for var in variants:
        if var == "adapters":
            c.add_secret_adapters(*CFG["adapter_seeds"])
        for P in CFG["buckets"]:                                   # warm-up: one full-length request per bucket
            c.generate(timing.synthetic_ids(P, seed=10_000 + P), CFG["A"])
        for P in CFG["buckets"]:
            for rep in range(CFG["repeats"]):
                g = timing.timed_generate(c, timing.synthetic_ids(P, seed=rep), CFG["A"])
                r = record(dict(part="compute", config=f"split_{H}_{T}", H=H, T=T, U=16 - H - T, variant=var,
                                scenario="loopback", gap_s=0.0, P=P, A=CFG["A"], rep=rep), g)
            say(f"  split ({H},{T}) {var:8s} P={P:4d}: TTFT {r['ttft_s']*1e3:7.0f} ms, decode {r['decode_median_step_s']*1e3:6.1f} "
                f"ms/step (last repeat), swap grew: {r['swap_grew']}")
        if var == "adapters":
            c.remove_adapters()
    save_mem(f"split_{H}_{T}", dict(client=memstat.snapshot(), server=c.server_stats()))
    c.close(); srv.terminate(); srv.wait(timeout=30)


@torch.no_grad()
def local_generate(m, ids, A, dev):
    caches, x, pos, steps, ns = m.new_caches(), ids, 0, [], time.perf_counter_ns
    for step in range(A):
        t0 = ns(); h = m.run("all", m.embed_ids(x), pos, caches); pm.sync(dev); t1 = ns()
        nxt = int(m.logits(h[:, -1:])[0, -1].argmax()); t2 = ns()
        steps.append(dict(step=step, kind="prefill" if step == 0 else "decode", n_tokens=x.shape[1], pos_start=pos,
                          layers_s=(t1 - t0) / 1e9, out_s=(t2 - t1) / 1e9, step_s=(t2 - t0) / 1e9))
        pos += x.shape[1]; x = torch.tensor([[nxt]])
    return dict(steps=steps)


def child_local(where):
    dev = CFG["client_device"] if where == "client" else CFG["server_device"]
    m = pm.PartialLlama(pm.model_dir(), dict(all=(0, 16)), True, dev, torch.float16)
    for P in CFG["buckets"]:
        local_generate(m, timing.synthetic_ids(P, seed=10_000 + P), CFG["A"], dev)
    for P in CFG["buckets"]:
        for rep in range(CFG["repeats"]):
            s0 = memstat.snapshot(); g = local_generate(m, timing.synthetic_ids(P, seed=rep), CFG["A"], dev); s1 = memstat.snapshot()
            g["mem"] = dict(sys_available_before=s0["sys_available"], sys_available_after=s1["sys_available"],
                            swap_before=s0["swap_used"], swap_after=s1["swap_used"], client_footprint=s1["footprint"],
                            client_rss=s1["rss"], swap_grew=s1["swap_used"] > s0["swap_used"] + 10**7)
            r = record(dict(part="compute", config=f"all_on_{where}", H=16 if where == "client" else 0, T=0,
                            U=0 if where == "client" else 16, variant="public", scenario="local", gap_s=0.0, P=P,
                            A=CFG["A"], rep=rep), g)
        say(f"  all on {where} ({dev}) P={P:4d}: TTFT {r['ttft_s']*1e3:7.0f} ms, decode {r['decode_median_step_s']*1e3:6.1f} "
            f"ms/step, swap grew: {r['swap_grew']}")
    save_mem(f"all_on_{where}", dict(process=memstat.snapshot(), device=dev))


def child_gaps():
    H, T = CFG["adapter_split"]
    log = open(os.path.join(OUT, "task3_server.log"), "a")
    srv, port = start_server(H, T, log); c = client(H, T, port)
    P = CFG["gap_P"]
    c.generate(timing.synthetic_ids(P, seed=10_000 + P), CFG["A"])
    for gap in CFG["gaps"]:
        for rep in range(CFG["gap_repeats"]):
            g = timing.timed_generate(c, timing.synthetic_ids(P, seed=rep), CFG["A"], gap_s=gap)
            record(dict(part="gaps", config=f"split_{H}_{T}", H=H, T=T, U=16 - H - T, variant="public", scenario="loopback",
                        gap_s=gap, P=P, A=CFG["A"], rep=rep), g)
            dec = g["steps"][1:]
            say(f"  gap {gap*1e3:5.0f} ms rep {rep}: server compute median {st.median(s['server_compute_s'] for s in dec)*1e3:.2f} ms, "
                f"client head {st.median(s['head_s'] for s in dec)*1e3:.2f} ms, tail layers "
                f"{st.median(s['tail_layers_s'] for s in dec)*1e3:.2f} ms, output {st.median(s['out_s'] for s in dec)*1e3:.2f} ms, "
                f"swap grew: {g['mem']['swap_grew']}")
    c.close(); srv.terminate(); srv.wait(timeout=30)


def child_compose(H, T):
    from netem import NetEm
    log = open(os.path.join(OUT, "task3_server.log"), "a")
    P = CFG["compose_P"]
    for scen in CFG["compose_scenarios"]:
        srv, port = start_server(H, T, log, scen); c = client(H, T, port, NetEm.scenario(scen))
        c.generate(timing.synthetic_ids(P, seed=10_000 + P), CFG["A"])             # warm-up
        for rep in range(CFG["compose_repeats"]):
            g = timing.timed_generate(c, timing.synthetic_ids(P, seed=rep), CFG["A"])
            r = record(dict(part="compose", config=f"split_{H}_{T}", H=H, T=T, U=16 - H - T, variant="public",
                            scenario=scen, gap_s=0.0, P=P, A=CFG["A"], rep=rep), g)
            say(f"  compose ({H},{T}) {scen:4s} rep {rep}: total {r['total_s']:.2f} s, TTFT {r['ttft_s']:.3f} s, "
                f"swap grew: {r['swap_grew']}")
        c.close(); srv.terminate(); srv.wait(timeout=30)


# ---------------------------------------------------------------------------------------- driver
def run_child(args):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", __file__, "--child"] + args, stderr=subprocess.PIPE, text=True, timeout=7200)
    if r.returncode:
        err = "\n".join(l for l in r.stderr.splitlines() if "Redirects" not in l)[-2500:]
        say(f"  FAILED child {args} (exit {r.returncode}):\n{err}")
    return time.time() - t0


def main(parts):
    os.makedirs(OUT, exist_ok=True)
    s0 = memstat.snapshot()
    say(f"\n=== task 3 run {time.strftime('%Y-%m-%d %H:%M:%S')} parts {parts}\n{env_info()}\nCFG {json.dumps(CFG)}\n"
        f"system available {s0['sys_available']/1e9:.2f} GB, swap {s0['swap_used']/1e9:.2f} GB")
    if "compute" in parts:
        for H, T in CFG["splits"]:
            say(f"\nsplit ({H},{T}) [{run_child(['split', str(H), str(T)]):.0f} s]")
        for where in ("client", "server"):
            say(f"all on {where} [{run_child(['local', where]):.0f} s]")
    if "gaps" in parts:
        say(f"\ngaps [{run_child(['gaps']):.0f} s]")
    if "compose" in parts:
        for H, T in CFG["compose_splits"]:
            say(f"\ncompose ({H},{T}) [{run_child(['compose', str(H), str(T)]):.0f} s]")
    s1 = memstat.snapshot()
    say(f"\nsystem swap {s0['swap_used']/1e9:.2f} -> {s1['swap_used']/1e9:.2f} GB; done {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        kind, rest = sys.argv[2], sys.argv[3:]
        if kind == "split":
            child_split(int(rest[0]), int(rest[1]))
        elif kind == "local":
            child_local(rest[0])
        elif kind == "gaps":
            child_gaps()
        elif kind == "compose":
            child_compose(int(rest[0]), int(rest[1]))
    else:
        main(sys.argv[1:] or ["compute", "gaps", "compose"])
