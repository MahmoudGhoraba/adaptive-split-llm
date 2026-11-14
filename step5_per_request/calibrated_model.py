"""
CALIBRATED latency/memory model for the per-request controller (task 4). Sits ALONGSIDE latency()/memory_gb() of
per_request_controller.py (kept unchanged for comparison). Parameters: calibration.json (built from the runtime
measurements by runtime/build_calibration.py; labels inside).

One request, prompt P, answer A, greedy: 1 prefill step + (A - 1) decode steps = A round trips and A output-layer
calls (the 1 + A of request_workload() is NOT used here). Per step (split / noise / secret options):
  client   head(H) + tail(T) + output layer + secret-adapter overhead         [x client gap factor, decode only]
  server   middle(U) / server_share                                            [x server gap factor, decode only]
  network  RTT + bytes_up*8/rate_up + bytes_down*8/rate_down + per-message overhead (fitted)
           bytes = n_tokens * 4096 + 52 each way (fp16 hidden states)
  gap      idle gap between messages ~ network time of the message; slowdown factors interpolated from the
           measured gap table (MEASURED at 0/20/60/250 ms; held at the last value beyond 250 ms)
All on client: 16 client layers + output layer, no network. All on server: 16 server layers + server output layer
(/ server_share) + one RTT + text bytes (4 B/token up for the prompt, down for the answer).
Per-layer times: prefill L*(FLOPs_layer(P)/f + c) + o; decode L*(d0 + d1*k) + o (closed-form sums over k).
Memory: client = memory_gb() + 0.40 GB (MEASURED overhead); all on server: 0.29 GB.

Estimator: what the controller knows. EWMA of RTT (one PING per request), effective rate r_eff =
2 / (1/rate_up + 1/rate_down) (from the prefill message; split traffic is the same size both ways, so only r_eff
matters) and server share (predicted / reported server compute). Never sees the true trace.
"""
import json
import os

import numpy as np

import per_request_controller as pr

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "calibration.json")))
N = pr.N_LAYERS
TOK_B = CAL["wire"]["bytes_per_token"]; HDR = CAL["wire"]["header_bytes"]; TXT = CAL["wire"]["text_bytes_per_token"]
GAP = CAL["gap"]


def _prefill(par, L, P):
    if L <= 0:
        return 0.0
    inv_f, c, o = par
    return L * (float(pr.layer_flops(P, 0)) * inv_f + c) + o


def _decode_sum(par, L, P, A):
    """Sum over the A - 1 decode steps (contexts k = P+1 .. P+A-1) of L*(d0 + d1*k) + o."""
    if L <= 0 or A <= 1:
        return 0.0
    d0, d1, o = par
    n = A - 1
    ksum = n * P + n * (n + 1) / 2
    return L * (d0 * n + d1 * ksum) + o * n


def gap_factors(gap_s):
    g = GAP["gaps_s"]
    return (float(np.interp(gap_s, g, GAP["client_factor"])), float(np.interp(gap_s, g, GAP["server_factor"])))


def r_eff(cond):
    return 2.0 / (1.0 / cond["rate_up"] + 1.0 / cond["rate_down"])


def latency_cal(o, P, A, cond, prof, detail=False):
    """Calibrated latency [s] of option o for one request. cond: rate_up, rate_down [bit/s], rtt [s], server_share."""
    cp, srv = CAL["client"][prof], CAL["server"]
    share = max(cond.get("server_share", 1.0), 1e-3)
    k = o["kind"]
    if k == "local":
        t = _prefill(cp["prefill"], N, P) + _decode_sum(cp["decode"], N, P, A) + A * cp["out_s"]
        return (t, dict(client=t, server=0.0, network=0.0)) if detail else t
    if k == "cloud":
        s = (_prefill(srv["prefill"], N, P) + _decode_sum(srv["decode"], N, P, A) + A * srv["out_s"]) / share
        net = cond["rtt"] + P * TXT * 8 / cond["rate_up"] + A * TXT * 8 / cond["rate_down"]
        return (s + net, dict(client=0.0, server=s, network=net)) if detail else s + net
    H, T = o["H"], o["T"]; U = N - H - T
    n_ad = H if k == "secret" else (H + T) if k == "secret_u" else 0
    ad = CAL["adapters"]
    # prefill step
    c_pre = (_prefill(cp["prefill"], H, P) + _prefill(cp["prefill"], T, P) + cp["out_s"]
             + n_ad * ad["prefill_s_per_layer_token"] * P)
    s_pre = _prefill(srv["prefill"], U, P) / share
    b_pre = P * TOK_B + HDR
    n_pre = cond["rtt"] + b_pre * 8 / cond["rate_up"] + b_pre * 8 / cond["rate_down"]
    ov = CAL["overhead"]
    o_pre = ov["m0_s"] + ov["m1_s_per_byte"] * 2 * b_pre
    # decode steps
    nd = A - 1
    b_dec = TOK_B + HDR
    n_dec = cond["rtt"] + b_dec * 8 / cond["rate_up"] + b_dec * 8 / cond["rate_down"]
    o_dec = ov["m0_s"] + ov["m1_s_per_byte"] * 2 * b_dec
    fc, fs = gap_factors(n_dec)
    c_dec = (_decode_sum(cp["decode"], H, P, A) + _decode_sum(cp["decode"], T, P, A) + nd * cp["out_s"]
             + nd * n_ad * ad["decode_s_per_layer"]) * fc
    s_dec = _decode_sum(srv["decode"], U, P, A) / share * fs
    t = c_pre + s_pre + n_pre + o_pre + c_dec + s_dec + nd * (n_dec + o_dec)
    if detail:
        return t, dict(client=c_pre + c_dec, server=s_pre + s_dec, network=n_pre + nd * n_dec, overhead=o_pre + nd * o_dec,
                       ttft=c_pre + s_pre + n_pre + o_pre, gap_factor_client=fc, gap_factor_server=fs)
    return t


def memory_cal(o, P, A):
    if o["kind"] == "cloud":
        return CAL["memory"]["cloud_client_gb"]
    wl = dict(P_MAX=int(P))
    return pr.memory_gb(o, wl, dict(bytes_per_value=2, answer_len=A)) + CAL["memory"]["client_overhead_gb"]


def feasible(o, cls, P, A, prof):
    return (pr.leak_ok(o, cls["eps"], cls["eps_a"]) and pr.quality_ok(o, cls["q"])
            and memory_cal(o, P, A) <= CAL["client"][prof]["mem_budget_gb"] + 1e-9)


def decide_cal(P, A, cls, cond, prof, opts):
    """Fastest feasible option under the CALIBRATED model and the given (true or estimated) conditions."""
    b = None
    for o in opts:
        if not feasible(o, cls, P, A, prof):
            continue
        L = latency_cal(o, P, A, cond, prof)
        if b is None or L < b[1]:
            b = (o, L)
    return b


class Estimator:
    """EWMA estimates of RTT, effective rate and server share from what the runtime observes."""

    def __init__(self, rtt=0.060, rate=2 / (1 / 10e6 + 1 / 25e6), share=1.0, alpha=0.3):   # prior: nominal 4G
        self.rtt, self.rate, self.share, self.alpha = rtt, rate, share, alpha
        self.n = dict(rtt=0, rate=0, share=0)

    def _upd(self, name, x):
        cur = getattr(self, name)
        setattr(self, name, (1 - self.alpha) * cur + self.alpha * x)
        self.n[name] += 1

    def observe(self, rtt=None, rate=None, share=None):
        if rtt is not None:
            self._upd("rtt", rtt)
        if rate is not None:
            self._upd("rate", rate)
        if share is not None:
            self._upd("share", share)

    def cond(self):
        return dict(rate_up=self.rate, rate_down=self.rate, rtt=self.rtt, server_share=min(self.share, 1.0))

    @staticmethod
    def rate_from_message(n_bytes_each_way, net_s, rtt_est):
        """Effective rate from one message pair: net - rtt = bytes*8*(1/up + 1/down) = 2*bytes*8/r_eff."""
        t = net_s - rtt_est
        return 2 * n_bytes_each_way * 8 / t if t > 1e-4 else None


# ------------------------------------------------------------------------------- switching cost (task 4 follow-up)
# What each side holds for an option: client None (all on server) | 16 (all) | H + T layers (+ embedding/output);
# server None (all on client) | "all" (16 layers + embedding/output) | U middle layers. A change of the held set on
# a side reloads that side. Setting "a": this testbed (both sides reload, FITTED in-situ); setting "b": dedicated
# server keeping every layer resident (server never reloads; MODELLED).
def held(o):
    if o is None:
        return None, None
    k = o["kind"]
    if k == "local":
        return ("all",), None
    if k == "cloud":
        return None, ("all",)
    return ("split", o["H"], o["T"]), ("mid", o["H"], o["T"])


def transition(state, o, setting="a"):
    """Predicted reload time [s] and the new resident state when option o is served from resident state
    state = (client_set, server_set). Setting a (this testbed, shared 8 GB): each side holds only what o needs (all on
    server frees the client, all on client frees the server). Setting b (dedicated server, MODELLED): the server holds
    every layer (no server cost) and the client keeps its weights while requests run on the server."""
    if o is None:
        return 0.0, state
    rc, rs = CAL["reload"]["client"], CAL["reload"]["server"]
    pc, ps = state
    nc, ns = held(o)
    if setting == "b" and nc is None:
        nc = pc                                           # client weights stay resident
    t = 0.0
    if nc != pc and nc is not None:
        L = N if nc == ("all",) else nc[1] + nc[2]
        t += rc["a"] + rc["b_per_layer"] * L
    if setting == "a" and ns != ps:
        if ns is None:
            t += rs["free_s"]
        elif ns == ("all",):
            t += rs["all_s"]
        else:
            t += rs["u14_s"] * (N - ns[1] - ns[2]) / 14
    return t, (nc, ns if setting == "a" else ("all",))


EMPTY = (None, None)


def ping_cost(cond):
    """One PING per request (RTT + server-reported load): counted for the estimating policies."""
    return cond["rtt"] + CAL["ping"]["overhead_s"]


def decide_switch(P, A, cls, cond, prof, opts, state, setting):
    """Switching-aware: fastest feasible option counting the predicted reload from the resident state
    (switch only when the predicted saving exceeds the predicted reload cost; myopic, one request ahead)."""
    b = None
    for o in opts:
        if not feasible(o, cls, P, A, prof):
            continue
        L = latency_cal(o, P, A, cond, prof) + transition(state, o, setting)[0]
        if b is None or L < b[1]:
            b = (o, L)
    return b
