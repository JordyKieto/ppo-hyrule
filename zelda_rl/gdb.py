"""Minimal GDB remote-protocol client for Dolphin's GDB stub (memory reads)."""
from __future__ import annotations

import socket


class GDB:
    def __init__(self, host="127.0.0.1", port=2345, timeout=5.0):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b""

    def _recv_packet(self) -> bytes:
        while True:
            i = self.buf.find(b"$")
            j = self.buf.find(b"#", i + 1) if i >= 0 else -1
            if i >= 0 and j >= 0 and len(self.buf) >= j + 3:
                pkt = self.buf[i + 1:j]
                self.buf = self.buf[j + 3:]
                self.s.sendall(b"+")
                return pkt
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("gdb stub closed")
            self.buf += chunk

    def cmd(self, payload: str, expect_reply: bool = True) -> bytes | None:
        data = payload.encode()
        self.s.sendall(b"$" + data + b"#" + f"{sum(data) & 0xFF:02x}".encode())
        return self._recv_packet() if expect_reply else None

    def read(self, addr: int, n: int, chunk: int = 0x800) -> bytes:
        out = bytearray()
        while n > 0:
            k = min(chunk, n)
            r = self.cmd(f"m{addr:x},{k:x}")
            while r.startswith(b"T") or r == b"":   # stale stop replies
                r = self._recv_packet()
            if r.startswith(b"E"):
                raise IOError(f"read {addr:#x},{k:#x}: {r!r}")
            out += bytes.fromhex(r.decode())
            addr += k
            n -= k
        return bytes(out)

    def drain(self, quiet: float = 0.001):
        """Discard stray stop-reply packets the stub may emit (keeps request/reply aligned)."""
        old = self.s.gettimeout()
        self.s.settimeout(quiet)
        try:
            while True:
                chunk = self.s.recv(1 << 16)
                if not chunk:
                    break
        except (socket.timeout, BlockingIOError):
            pass
        finally:
            self.s.settimeout(old)
        self.buf = b""

    def interrupt(self) -> bytes:
        self.s.sendall(b"\x03")
        pkt = self._recv_packet()
        self.drain()
        return pkt

    def cont(self):
        self.cmd("c", expect_reply=False)
