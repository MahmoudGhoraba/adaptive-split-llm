"""
Application-level network emulation (label results "application-level emulation", not a real link).

Each side emulates its OWN sending direction (client: uplink, server: downlink), so a round trip gets
  uplink serialisation + one-way delay + server + downlink serialisation + one-way delay.
  - one-way delay   = RTT / 2, added to every message
  - rate            token bucket: tokens (bytes) refill at rate/8 per second up to `burst` bytes; a message of N
                    bytes waits until N tokens are available. Default burst = 0: every message pays its full
                    serialisation time N*8/rate, as on a real link and as in the latency model (bytes / rate);
                    a burst > 0 lets that many bytes through instantly after idle time (a shaper/policer)
The protocol is strictly request/response, so the sender can wait before sendall() without reordering.
Sleeps: macOS oversleeps time.sleep(s) by ~50% of s, up to ~10 ms (measured 14 Nov: 30 ms -> +8.5 ms), so sleeps
are cut into 1 ms slices (+0.5 ms each, absorbed) until < 1.5 ms remain, then the rest is spun.
"""
import time
from dataclasses import dataclass

SCENARIOS = {                      # rate [bit/s], RTT [s] (symmetric rates)
    "wifi":   dict(rate=50e6, rtt=0.020),
    "4g":     dict(rate=10e6, rtt=0.060),
    "poor":   dict(rate=1e6,  rtt=0.250),
}


def precise_sleep(s):
    end = time.perf_counter() + s
    while end - time.perf_counter() > 0.0015:
        time.sleep(0.001)
    while time.perf_counter() < end:
        pass


@dataclass
class NetEm:
    rate: float              # bit/s
    one_way_delay: float     # s
    burst: int = 0           # bytes

    def __post_init__(self):
        self.tokens, self.t_last, self.emulated_s = float(self.burst), time.perf_counter(), 0.0

    def wait(self, n_bytes):
        """Block for serialisation (token bucket) + one-way delay of an n_bytes message; returns the wait [s]."""
        t0 = time.perf_counter()
        self.tokens = min(self.burst, self.tokens + (t0 - self.t_last) * self.rate / 8)
        tx = max(0.0, (n_bytes - self.tokens) * 8 / self.rate)
        self.tokens -= n_bytes                                 # may go negative: debt paid by the wait below
        w = tx + self.one_way_delay
        precise_sleep(w)
        self.t_last = time.perf_counter()
        self.tokens = min(self.burst, self.tokens + (self.t_last - t0) * self.rate / 8)
        self.emulated_s += w
        return w

    @classmethod
    def scenario(cls, name, burst=0):
        s = SCENARIOS[name]
        return cls(s["rate"], s["rtt"] / 2, burst)

    def describe(self):
        return dict(rate_mbit=self.rate / 1e6, one_way_delay_ms=self.one_way_delay * 1e3, burst=self.burst)
