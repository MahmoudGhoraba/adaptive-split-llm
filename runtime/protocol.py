"""
Wire format of the split runtime: one fixed binary header + raw payload, over a TCP stream.

Header (network byte order, 52 bytes):
  magic "USPL" | version u8 | type u8 | dtype u8 | flags u8 (reserved)
  session u32 | step u32 | pos_start u32 | n_tokens u32 | d u32 | payload_len u64
  server_compute_ns u64 | server_total_ns u64          (filled in by the server in replies, 0 otherwise)

Types:
  HELLO    client -> server: JSON {H, T, model_id, dtype}; reply: JSON {U, layers, ...} or ERROR
  PREFILL  hidden states [1, n_tokens, d] at positions pos_start .. pos_start + n_tokens - 1; reply: same shape
  DECODE   as PREFILL with n_tokens = 1
  RESET    clear the server's KV cache (new request / session); reply: RESET
  PING     payload echoed back (RTT and throughput probes)
  STATS    reply: JSON memory snapshot of the server process (memstat)
  BYE      close the connection
  ERROR    UTF-8 message (e.g. position mismatch)
  GENERATE all on server (reconfigurable server): JSON {ids, A}; reply JSON {tokens} + server_compute_ns

server_compute_ns = layers on the device incl. host<->device copies and synchronize;
server_total_ns   = from the request fully received to the reply ready to send (incl. (de)serialisation).
So network + transport time on the client = round trip - server_total_ns, with no shared clock.
"""
import json
import socket
import struct
from dataclasses import dataclass

import torch

MAGIC, VERSION = b"USPL", 1
HEADER = struct.Struct("!4sBBBBIIIIIQQQ")
HELLO, PREFILL, DECODE, RESET, PING, STATS, BYE, ERROR, GENERATE = range(1, 10)
TYPE_NAME = {v: k for k, v in dict(HELLO=HELLO, PREFILL=PREFILL, DECODE=DECODE, RESET=RESET, PING=PING,
                                    STATS=STATS, BYE=BYE, ERROR=ERROR, GENERATE=GENERATE).items()}
DTYPES = {1: torch.float16, 2: torch.bfloat16, 3: torch.float32}
DTYPE_CODE = {v: k for k, v in DTYPES.items()}


class ProtocolError(Exception):
    pass


@dataclass
class Msg:
    type: int
    payload: bytes = b""
    session: int = 0
    step: int = 0
    pos_start: int = 0
    n_tokens: int = 0
    d: int = 0
    dtype: int = 0
    server_compute_ns: int = 0
    server_total_ns: int = 0

    def tensor(self):
        """Payload as a [1, n_tokens, d] tensor (CPU, zero-copy view of the received buffer)."""
        buf = self.payload if isinstance(self.payload, bytearray) else bytearray(self.payload)   # writable buffer
        t = torch.frombuffer(buf, dtype=torch.uint8).view(DTYPES[self.dtype])
        return t.view(1, self.n_tokens, self.d)

    def json(self):
        return json.loads(bytes(self.payload).decode())


def tensor_msg(type_, h, **kw):
    """Hidden states [1, n, d] -> message (one copy device -> host)."""
    h = h.detach().to("cpu").contiguous()
    raw = h.view(torch.uint8).numpy().tobytes() if h.dtype == torch.bfloat16 else h.numpy().tobytes()
    return Msg(type_, raw, n_tokens=h.shape[1], d=h.shape[2], dtype=DTYPE_CODE[h.dtype], **kw)


def json_msg(type_, obj, **kw):
    return Msg(type_, json.dumps(obj).encode(), **kw)


class Conn:
    """A connected TCP socket with message framing and byte counters.
    netem: optional netem.NetEm; every send first waits for its serialisation + one-way delay (this side's direction)."""

    def __init__(self, sock, netem=None):
        self.sock, self.netem = sock, netem
        if sock.family != socket.AF_UNIX:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bytes_sent = self.bytes_recv = 0

    def send(self, m):
        hdr = HEADER.pack(MAGIC, VERSION, m.type, m.dtype, 0, m.session, m.step, m.pos_start, m.n_tokens, m.d,
                          len(m.payload), m.server_compute_ns, m.server_total_ns)
        if self.netem is not None:
            self.netem.wait(HEADER.size + len(m.payload))
        self.sock.sendall(hdr + bytes(m.payload))
        self.bytes_sent += HEADER.size + len(m.payload)
        return HEADER.size + len(m.payload)

    def _recv_exact(self, n):
        buf = bytearray(n); view = memoryview(buf); got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError(f"connection closed after {got} of {n} bytes")
            got += k
        return buf

    def recv(self):
        (magic, ver, type_, dtype, _, session, step, pos, n, d, plen, sc, st) = HEADER.unpack(self._recv_exact(HEADER.size))
        if magic != MAGIC or ver != VERSION:
            raise ProtocolError(f"bad header: magic {magic!r}, version {ver}")
        if type_ not in TYPE_NAME:
            raise ProtocolError(f"unknown message type {type_}")
        payload = self._recv_exact(plen) if plen else bytearray()
        self.bytes_recv += HEADER.size + plen
        if type_ in (PREFILL, DECODE):
            if dtype not in DTYPES:
                raise ProtocolError(f"unknown dtype code {dtype}")
            if plen != n * d * DTYPES[dtype].itemsize:
                raise ProtocolError(f"payload {plen} bytes != {n} x {d} x {DTYPES[dtype].itemsize}")
        return Msg(type_, payload, session, step, pos, n, d, dtype, sc, st)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
