"""
Round-trip test of the wire format (no model). Run:  python test_protocol.py
  1. every message type over socket.socketpair(): all header fields and payloads come back unchanged
  2. hidden-state tensors (fp16, bf16, fp32; 1 x 1 x 2048 and 1 x 512 x 2048) are bitwise identical after the trip
  3. byte counters = header + payload
  4. corrupted header, wrong payload length and a truncated stream are rejected
  5. the same over a real TCP loopback connection (what the runtime uses), with a 4 MB payload
"""
import socket
import struct
import sys
import threading

import torch

import protocol as pr

FAILS = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def pair():
    a, b = socket.socketpair()
    return pr.Conn(a), pr.Conn(b)


def test_types():
    a, b = pair()
    for t in [t for t in pr.TYPE_NAME if t not in (pr.PREFILL, pr.DECODE)]:    # tensor types: test_tensors
        m = pr.Msg(t, f"payload {t}".encode(), session=7, step=t, pos_start=123456, n_tokens=0, d=0,
                   server_compute_ns=2**40 + t, server_total_ns=2**41 + t)
        a.send(m); r = b.recv()
        same = (r.type, bytes(r.payload), r.session, r.step, r.pos_start, r.server_compute_ns, r.server_total_ns) == \
               (m.type, m.payload, m.session, m.step, m.pos_start, m.server_compute_ns, m.server_total_ns)
        check(f"type {pr.TYPE_NAME[t]:7s} round trip", same)
    j = {"H": 4, "T": 2, "model_id": "x"}
    a.send(pr.json_msg(pr.HELLO, j)); check("JSON payload", b.recv().json() == j)
    check("header size 52 bytes", pr.HEADER.size == 52, f"({pr.HEADER.size})")


def test_tensors():
    a, b = pair()
    g = torch.Generator().manual_seed(0)
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for n in (1, 512):
            h = (torch.randn(1, n, 2048, generator=g) * 3).to(dt)
            s0, r0 = a.bytes_sent, b.bytes_recv
            th = threading.Thread(target=a.send, args=(pr.tensor_msg(pr.PREFILL, h, pos_start=5, step=1),))
            th.start(); r = b.recv(); th.join()       # 4 MB > socket buffer: send and receive concurrently
            t = r.tensor()
            ok = t.dtype == dt and t.shape == h.shape and torch.equal(t.view(torch.uint8), h.view(torch.uint8))
            nbytes = pr.HEADER.size + h.numel() * h.element_size()
            check(f"{str(dt):14s} 1x{n}x2048 bitwise", ok)
            check(f"{str(dt):14s} 1x{n}x2048 byte counters", a.bytes_sent - s0 == nbytes == b.bytes_recv - r0,
                  f"({nbytes} bytes)")


def test_errors():
    a, b = pair()
    a.sock.sendall(b"XXXX" + bytes(pr.HEADER.size - 4))
    try:
        b.recv(); check("bad magic rejected", False)
    except pr.ProtocolError:
        check("bad magic rejected", True)
    a, b = pair()
    m = pr.tensor_msg(pr.DECODE, torch.zeros(1, 1, 8, dtype=torch.float16)); m.d = 16     # header says 16, payload 8
    a.send(m)
    try:
        b.recv(); check("payload length mismatch rejected", False)
    except pr.ProtocolError:
        check("payload length mismatch rejected", True)
    a, b = pair()
    a.send(pr.Msg(pr.PREFILL, bytes(8), n_tokens=1, d=4, dtype=0))
    try:
        b.recv(); check("unknown dtype rejected", False)
    except pr.ProtocolError:
        check("unknown dtype rejected", True)
    a, b = pair()
    hdr = pr.HEADER.pack(pr.MAGIC, pr.VERSION, pr.PING, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0)
    a.sock.sendall(hdr + bytes(10)); a.close()
    try:
        b.recv(); check("truncated stream rejected", False)
    except ConnectionError:
        check("truncated stream rejected", True)


def test_tcp():
    srv = socket.create_server(("127.0.0.1", 0)); port = srv.getsockname()[1]
    def echo():
        c = pr.Conn(srv.accept()[0]); m = c.recv()
        c.send(pr.tensor_msg(pr.PREFILL, m.tensor() * 2, pos_start=m.pos_start)); c.close()
    th = threading.Thread(target=echo); th.start()
    c = pr.Conn(socket.create_connection(("127.0.0.1", port)))
    h = torch.randn(1, 1024, 2048).half()
    nodelay = c.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
    c.send(pr.tensor_msg(pr.PREFILL, h, pos_start=9)); r = c.recv(); th.join(); c.close(); srv.close()
    check("TCP loopback 4 MB echo (x2)", torch.equal(r.tensor(), h * 2) and r.pos_start == 9)
    check("TCP_NODELAY set", nodelay != 0)


if __name__ == "__main__":
    for f in (test_types, test_tensors, test_errors, test_tcp):
        print(f.__name__); f()
    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
