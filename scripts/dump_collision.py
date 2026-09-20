"""Read a scene's STATIC COLLISION out of a savestate: is wall geometry available as objects?

    .venv/bin/python scripts/dump_collision.py
    .venv/bin/python scripts/dump_collision.py --state datasets/zelda/combat/deku-baba/states/NACE01.s03 -k 8

Prep for treating static geometry as an OOMDP class (docs/CAUSAL_WORLD_MODEL.md). The paper declares static
entities -- Trap and Food in Mouse, Mineral in StarCraft CMS -- as full classes with no action fields, so the
same class-level attention that handles enemies handles them, and the model can learn the global causality
`Wall.pos -> Link.pos'`. Nothing in the method needs walls to be special; it needs them to be OBJECTS.

This script answers the only question that decides whether that is possible here: can we enumerate the room's
collision polygons, and are there few enough to be objects?

PROVENANCE (AGENTS.md section 3 -- every offset read from the decompilation, none recalled)
  PlayState + 0x007C0   CollisionContext colCtx            include/play_state.h:70
  PlayState + 0x01C24   ActorContext actorCtx              include/play_state.h:71  (matches obs.py's constant,
                                                           so PlayState offsets are NOT shifted; the
                                                           decomp-minus-0x10 rule is Player-specific)
  colCtx    + 0x00      CollisionHeader* colHeader         include/bgcheck.h:322
  header    + 0x0C/0x10 u16 numVertices / Vec3s* vtxList   include/bgcheck.h:246-247
  header    + 0x14/0x18 u16 numPolygons / CollisionPoly*   include/bgcheck.h:248-249
  CollisionPoly (0x10)  type, vIA, vIB, vIC, normal, dist  include/bgcheck.h:51-67
  normal is Vec3s scaled by SHT_MAX (COLPOLY_NORMAL_FRAC)  include/bgcheck.h:33-35

  Floor / wall / ceiling is the ENGINE'S OWN three-way split, not a threshold picked here:
      normal.y >  0.5  -> floor
      normal.y < -0.8  -> ceiling
      otherwise        -> wall
  src/code/z_bgcheck.c:557-562 (StaticLookup_AddPoly).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.obs import PLAY_STATE, read_state_v2  # noqa: E402
from zelda_rl.savestate import decompress_state, find_n64_ram  # noqa: E402

COL_CTX = PLAY_STATE + 0x007C0
NORMAL_FRAC = 1.0 / 32767.0
FLOOR_NY, CEILING_NY = 0.5, -0.8
DEFAULT_STATE = "datasets/zelda/combat/deku-baba/states/NACE01.s03"


def _u32(ram, a):
    return struct.unpack_from(">I", ram, a & 0x3FFFFF)[0]


def _u16(ram, a):
    return struct.unpack_from(">H", ram, a & 0x3FFFFF)[0]


def read_collision(ram) -> dict:
    """-> vertices (V,3) float, polys with vertex ids, normals (P,3) float, centroids (P,3), class (P,)."""
    header = _u32(ram, COL_CTX + 0x00)
    if not (0x80000000 <= header < 0x80400000):
        raise ValueError(f"colHeader pointer {header:#x} is not in N64 RAM")
    n_vtx, vtx_list = _u16(ram, header + 0x0C), _u32(ram, header + 0x10)
    n_poly, poly_list = _u16(ram, header + 0x14), _u32(ram, header + 0x18)

    v = np.frombuffer(ram, ">i2", count=n_vtx * 3, offset=vtx_list & 0x3FFFFF).reshape(n_vtx, 3)
    raw = np.frombuffer(ram, np.uint8, count=n_poly * 0x10, offset=poly_list & 0x3FFFFF).reshape(n_poly, 0x10)
    be16 = lambda c: (raw[:, c].astype(np.int32) << 8) | raw[:, c + 1]        # noqa: E731
    vid = np.stack([be16(2) & 0x1FFF, be16(4) & 0x1FFF, be16(6) & 0x1FFF], 1)
    nrm = np.stack([np.frombuffer(raw[:, c:c + 2].tobytes(), ">i2") for c in (8, 10, 12)], 1) * NORMAL_FRAC

    tri = v[vid].astype(np.float32)                                          # (P, 3 verts, 3 coords)
    cent = tri.mean(1)
    ny = nrm[:, 1]
    kind = np.where(ny > FLOOR_NY, 0, np.where(ny < CEILING_NY, 2, 1))       # 0 floor, 1 wall, 2 ceiling
    return {"header": header, "vertices": v, "vid": vid, "normal": nrm, "centroid": cent,
            "tri": tri, "kind": kind, "n_poly": n_poly, "n_vtx": n_vtx}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("-k", type=int, default=8, help="nearest walls to list")
    args = ap.parse_args()

    raw = decompress_state(Path(args.state).read_bytes())
    base = find_n64_ram(raw)
    ram = raw[base: base + 0x400000]
    st = read_state_v2(ram)
    c = read_collision(ram)

    names = ["floor", "wall", "ceiling"]
    counts = [int((c["kind"] == k).sum()) for k in range(3)]
    print(f"state: {args.state}")
    print(f"  scene {st.scene}  Link at ({st.pos[0]:.1f}, {st.pos[1]:.1f}, {st.pos[2]:.1f})  "
          f"actors {len(st.actors)} (ids {sorted({hex(a.id) for a in st.actors})})")
    print(f"  colHeader {c['header']:#010x}  {c['n_vtx']} vertices, {c['n_poly']} polygons")
    print(f"  " + "  ".join(f"{n} {v}" for n, v in zip(names, counts)))

    lo, hi = c["tri"].reshape(-1, 3).min(0), c["tri"].reshape(-1, 3).max(0)
    print(f"  bounds x[{lo[0]:.0f}, {hi[0]:.0f}]  y[{lo[1]:.0f}, {hi[1]:.0f}]  z[{lo[2]:.0f}, {hi[2]:.0f}]")

    link = np.array(st.pos, np.float32)
    wall = c["kind"] == 1
    wc, wn = c["centroid"][wall], c["normal"][wall]
    d = np.linalg.norm(wc - link, axis=1)
    order = np.argsort(d)[: args.k]
    print(f"\n  nearest {args.k} WALL polygons to Link (candidate instances of a `Wall` class):")
    print(f"    {'dist':>8} {'centroid (x, y, z)':>28} {'normal (x, y, z)':>24}")
    for i in order:
        p, n = wc[i], wn[i]
        print(f"    {d[i]:8.1f} {p[0]:9.1f}{p[1]:9.1f}{p[2]:9.1f}   {n[0]:8.2f}{n[1]:8.2f}{n[2]:8.2f}")

    dw = np.sort(d)
    print(f"\n  wall distances: nearest {dw[0]:.1f}, median {np.median(dw):.1f}, "
          f"{int((dw < 300).sum())} within 300 units, {int((dw < 1000).sum())} within 1000")
    print(f"  Link's own reading: bg_flags {st.sf1 and ''}dist_interact_wall is a DERIVED scalar; the "
          f"nearest wall above is the object it summarises.")


if __name__ == "__main__":
    main()
