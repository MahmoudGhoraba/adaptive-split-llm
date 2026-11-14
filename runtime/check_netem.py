"""
Check the network emulation: measured RTT and throughput vs the configured values (no model, echo server).

  RTT          median of 30 empty PINGs (52-byte header each way); expected = RTT + 2 x 52 x 8 / rate
  throughput   PING with an N-byte payload, echoed back (N sized for ~0.5 s per direction, max 4 MB), 5 repeats;
               throughput = 2 x (N + 52) x 8 / (t - RTT_empty); expected = rate

Modes:
  --mode app       application-level emulation (netem.py) on both sides; scenarios from netem.SCENARIOS
  --mode dummynet  no application emulation; the link is shaped by macOS dummynet, set up beforehand with
                   sudo ./dummynet.sh up <scenario> <port>; the echo server listens on that port
  --mode none      plain loopback (reference)

Run:  python check_netem.py [--mode app] [--scenarios wifi 4g poor]
Writes check_out/netem_check_<mode>.json, .csv
"""
import argparse
import csv
import json
import os
import socket
import statistics as st
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat
import protocol as pr
from netem import NetEm, SCENARIOS

OUT = os.path.join(HERE, "check_out")


def echo_server(srv, netem_factory):
    while True:
        try:
            sock, _ = srv.accept()
        except OSError:
            return
        c = pr.Conn(sock, netem_factory())
        try:
            while True:
                m = c.recv()
                if m.type == pr.BYE:
                    break
                c.send(pr.Msg(pr.PING, m.payload, step=m.step))
        except (ConnectionError, OSError):
            pass
        c.close()


def measure(port, netem_up, rate, n_ping=30, n_thr=5):
    c = pr.Conn(socket.create_connection(("127.0.0.1", port)), netem_up)
    def ping(n):
        t0 = time.perf_counter(); c.send(pr.Msg(pr.PING, bytes(n))); c.recv(); return time.perf_counter() - t0
    for _ in range(3):
        ping(0)                                             # warm-up
    rtts = [ping(0) for _ in range(n_ping)]
    N = int(min(4_000_000, rate / 8 * 0.5)) if rate else 4_000_000
    times = [ping(N) for _ in range(n_thr)]
    c.send(pr.Msg(pr.BYE)); c.close()
    rtt = st.median(rtts)
    thr = [2 * (N + pr.HEADER.size) * 8 / (t - rtt) for t in times]
    return dict(rtt_median_s=rtt, rtt_p10_s=sorted(rtts)[3], rtt_p90_s=sorted(rtts)[-4], payload_bytes=N,
                throughput_mean_bps=st.mean(thr), throughput_sd_bps=st.stdev(thr), transfer_times_s=times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="app", choices=["app", "dummynet", "none"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--port", type=int, default=0, help="dummynet mode: the port the rules shape")
    a = ap.parse_args()
    rows, s0 = [], memstat.snapshot()
    print(f"mode {a.mode}; system available {s0['sys_available']/1e9:.2f} GB, swap {s0['swap_used']/1e9:.2f} GB")
    for name in (a.scenarios if a.mode != "none" else ["none"]):
        sc = SCENARIOS.get(name, dict(rate=0, rtt=0))
        factory = (lambda: NetEm.scenario(name)) if a.mode == "app" else (lambda: None)
        srv = socket.create_server(("127.0.0.1", a.port)); port = srv.getsockname()[1]
        threading.Thread(target=echo_server, args=(srv, factory), daemon=True).start()
        m = measure(port, factory(), sc["rate"])
        srv.close()
        exp_rtt = sc["rtt"] + (2 * pr.HEADER.size * 8 / sc["rate"] if sc["rate"] else 0)
        r = dict(mode=a.mode, scenario=name, rate_mbit=sc["rate"] / 1e6, rtt_ms=sc["rtt"] * 1e3,
                 expected_rtt_ms=exp_rtt * 1e3, measured_rtt_ms=m["rtt_median_s"] * 1e3,
                 rtt_p10_ms=m["rtt_p10_s"] * 1e3, rtt_p90_ms=m["rtt_p90_s"] * 1e3,
                 rtt_err_pct=100 * (m["rtt_median_s"] - exp_rtt) / exp_rtt if exp_rtt else None,
                 measured_mbit=m["throughput_mean_bps"] / 1e6, measured_mbit_sd=m["throughput_sd_bps"] / 1e6,
                 thr_err_pct=100 * (m["throughput_mean_bps"] - sc["rate"]) / sc["rate"] if sc["rate"] else None,
                 payload_bytes=m["payload_bytes"])
        rows.append(r)
        print(f"  {name:5s}: RTT configured {r['rtt_ms']:.0f} ms, expected {r['expected_rtt_ms']:.3f}, measured "
              f"{r['measured_rtt_ms']:.3f} ms (p10-p90 {r['rtt_p10_ms']:.3f}-{r['rtt_p90_ms']:.3f})"
              + (f", error {r['rtt_err_pct']:+.2f}%" if r["rtt_err_pct"] is not None else "")
              + f" | throughput configured {r['rate_mbit']:g} Mbit/s, measured {r['measured_mbit']:.3f} ± "
                f"{r['measured_mbit_sd']:.3f}" + (f", error {r['thr_err_pct']:+.2f}%" if r["thr_err_pct"] is not None else "")
              + f" ({r['payload_bytes']/1e6:.2f} MB payload)", flush=True)
    s1 = memstat.snapshot()
    json.dump(dict(rows=rows, system_start=s0, system_end=s1), open(os.path.join(OUT, f"netem_check_{a.mode}.json"), "w"), indent=1)
    with open(os.path.join(OUT, f"netem_check_{a.mode}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
