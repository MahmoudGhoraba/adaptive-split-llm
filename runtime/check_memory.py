"""
Memory checks before the sockets (MEASURED): fixed overhead vs per-layer cost, and the effect of the loader.

  a) baseline   empty process: python + torch + transformers imported, MPS initialised, no weights
  b) per side   client (4,2) and server (4,2) alone, for each loader mode (partial_model.PartialLlama load=):
                  cpu_keep  bf16 -> CPU -> fp16 -> MPS, no cleanup (the first version)
                  cpu       same + gc + torch.mps.empty_cache() after loading
                  direct    bf16 read straight to MPS, converted there, + gc + empty_cache
  c) together   client (4,2) and server (4,2) as two processes at the same time (best mode of b), holding their
                memory while system available memory and swap are sampled

Each stage: footprint (incl. MPS buffers), tensor bytes, system available memory, swap. P = 512, A = 128.
Run:  python check_memory.py            ->  check_out/memory_check.json, memory_check.csv
"""
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_out")
CFG = dict(H=4, T=2, P=512, A=128, modes=["cpu_keep", "cpu", "direct"], hold_s=20, sample_s=1.0)


def worker(kind, mode, hold):
    import memstat
    snaps = dict(imported=memstat.snapshot())
    import torch
    import partial_model as pm
    snaps["imported"] = memstat.snapshot()
    torch.ones(1, device="mps"); torch.mps.synchronize()
    snaps["mps_init"] = memstat.snapshot()
    if kind != "baseline":
        mdir = pm.model_dir(); n = pm.load_config(mdir).num_hidden_layers
        cseg, sseg = pm.split_segments(CFG["H"], CFG["T"], n)
        m = pm.PartialLlama(mdir, cseg if kind == "client" else sseg, kind == "client", "mps", torch.float16, load=mode)
        pm.sync("mps"); snaps["loaded"] = memstat.snapshot()
        caches, g, P = m.new_caches(), torch.Generator().manual_seed(0), CFG["P"]
        with torch.no_grad():
            for step in range(1 + CFG["A"]):
                k, pos = (P, 0) if step == 0 else (1, P + step - 1)
                if kind == "client":
                    h = m.run("head", m.embed_ids(torch.randint(0, m.cfg.vocab_size, (1, k), generator=g)), pos, caches)
                    m.logits(m.run("tail", h, pos, caches)[:, -1:])
                else:
                    m.run("mid", torch.randn(1, k, m.cfg.hidden_size, generator=g), pos, caches)
                pm.sync("mps")
        snaps["after_P_A"] = memstat.snapshot()
    print("READY", flush=True)
    time.sleep(hold)                              # keep the memory while the parent samples the system
    print(json.dumps(snaps), flush=True)


def run(kind, mode, hold=0):
    r = subprocess.run([sys.executable, __file__, "--worker", kind, mode, str(hold)], capture_output=True,
                       text=True, timeout=1200)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(f"FAILED {kind} {mode}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    import memstat
    gb = lambda v: round(v / 1e9, 3)
    os.makedirs(OUT, exist_ok=True)
    rows, s0 = [], memstat.snapshot()
    print(f"system at start: available {gb(s0['sys_available'])} GB, swap {gb(s0['swap_used'])} GB")

    def add(kind, mode, snaps, together=False):
        for stage, s in snaps.items():
            rows.append(dict(kind=kind, mode=mode, together=together, stage=stage, footprint_gb=gb(s["footprint"]),
                             peak_footprint_gb=gb(s["peak_footprint"]), mps_alloc_gb=gb(s["mps_alloc"]),
                             mps_driver_gb=gb(s["mps_driver"]), rss_gb=gb(s["rss"]),
                             sys_available_gb=gb(s["sys_available"]), swap_gb=gb(s["swap_used"])))
        print(f"  {kind:8s} {mode:8s}: " + " | ".join(f"{k} {gb(v['footprint'])}" for k, v in snaps.items())
              + f"  (tensors {gb(list(snaps.values())[-1]['mps_alloc'])}, swap {gb(list(snaps.values())[0]['swap_used'])}"
                f" -> {gb(list(snaps.values())[-1]['swap_used'])} GB)", flush=True)

    print("\na) baseline (footprint GB per stage)")
    add("baseline", "-", run("baseline", "-"))
    print("\nb) one side alone, (H, T) = (%d, %d), P = %d, A = %d" % (CFG["H"], CFG["T"], CFG["P"], CFG["A"]))
    total = {}                                    # client + server footprint after P + A, per loader mode
    for mode in CFG["modes"]:
        for kind in ("client", "server"):
            sn = run(kind, mode); add(kind, mode, sn)
            total[mode] = total.get(mode, 0) + sn["after_P_A"]["footprint"]
    mode = min(total, key=total.get)
    print(f"\nc) client + server together (two processes, load mode '{mode}', holding {CFG['hold_s']} s)")
    before = memstat.snapshot()
    procs = {k: subprocess.Popen([sys.executable, __file__, "--worker", k, mode, str(CFG["hold_s"])],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for k in ("client", "server")}
    for p in procs.values():                      # wait until both have loaded and run P + A
        assert p.stdout.readline().strip() == "READY"
    samples = []
    for _ in range(int(CFG["hold_s"] * 0.8 / CFG["sample_s"])):
        s = memstat.snapshot(); samples.append((s["sys_available"], s["swap_used"])); time.sleep(CFG["sample_s"])
    for k, p in procs.items():
        out = p.communicate(timeout=600)[0]; add(k, mode, json.loads(out.strip().splitlines()[-1]), together=True)
    after = memstat.snapshot()
    joint = dict(mode=mode, swap_before_gb=gb(before["swap_used"]), swap_max_gb=gb(max(s[1] for s in samples)),
                 swap_after_gb=gb(after["swap_used"]), min_available_gb=gb(min(s[0] for s in samples)),
                 available_before_gb=gb(before["sys_available"]))
    joint["swap_grew"] = joint["swap_max_gb"] > joint["swap_before_gb"] + 0.01
    print(f"  system while both held memory: available {joint['available_before_gb']} -> min {joint['min_available_gb']} GB,"
          f" swap {joint['swap_before_gb']} -> max {joint['swap_max_gb']} GB -> swap grew: {joint['swap_grew']}")
    json.dump(dict(cfg=CFG, system_start=s0, rows=rows, together=joint), open(os.path.join(OUT, "memory_check.json"), "w"),
              indent=1)
    with open(os.path.join(OUT, "memory_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Saved to {OUT}/memory_check.json, memory_check.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.path.insert(0, HERE)
        worker(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        sys.path.insert(0, HERE)
        main()
