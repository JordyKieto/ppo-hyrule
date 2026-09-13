"""Parse Dolphin (2603a, state header v1) savestates and extract the N64 VC RAM for OoT."""
from __future__ import annotations

import struct

import lz4.block
import numpy as np

N64_RAM_SIZE = 0x400000
N64_RAM_MAGIC = bytes.fromhex("3c1a8000275a260003400008")  # N64 exception vector


def decompress_state(buf: bytes) -> bytes:
    vlen = struct.unpack_from("<I", buf, 0x1C)[0]
    off = 0x20 + vlen
    _hdr_ver, comp, _poff, usize = struct.unpack_from("<HHIQ", buf, off)
    off += 16
    if comp == 0:
        return buf[off:off + usize]
    out = bytearray()
    while off < len(buf) and len(out) < usize:
        n = struct.unpack_from("<i", buf, off)[0]
        off += 4
        out += lz4.block.decompress(buf[off:off + n], uncompressed_size=usize - len(out))
        off += n
    if len(out) != usize:
        raise ValueError(f"short state: {len(out)} / {usize}")
    return bytes(out)


def find_n64_ram(raw: bytes, hint: int | None = None) -> int:
    """Locate N64 RAM: exception vector at +0 and osMemSize (4MB) at +0x318."""
    if hint is not None and raw[hint:hint + 12] == N64_RAM_MAGIC:
        return hint
    i = raw.find(N64_RAM_MAGIC)
    while i >= 0:
        if struct.unpack_from(">I", raw, i + 0x318)[0] == N64_RAM_SIZE:
            return i
        i = raw.find(N64_RAM_MAGIC, i + 1)
    raise ValueError("N64 RAM not found")


class N64Ram:
    def __init__(self, ram: bytes):
        self.ram = ram

    def u8(self, a): return self.ram[a & 0x3FFFFF]
    def s16(self, a): return struct.unpack_from(">h", self.ram, a & 0x3FFFFF)[0]
    def u32(self, a): return struct.unpack_from(">I", self.ram, a & 0x3FFFFF)[0]
    def f32(self, a): return struct.unpack_from(">f", self.ram, a & 0x3FFFFF)[0]
    def vec3(self, a): return struct.unpack_from(">fff", self.ram, a & 0x3FFFFF)

    def as_array(self) -> np.ndarray:
        return np.frombuffer(self.ram, np.uint8)
