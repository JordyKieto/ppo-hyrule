"""OoT NTSC 1.2 (Wii VC NACE) RAM map. Addresses verified against a live savestate."""
from __future__ import annotations

from zelda_rl.savestate import N64Ram

SAVE_CTX = 0x8011AC80      # gSaveContext ("ZELDAZ" at +0x1C)
PLAY_STATE = 0x801C8D60    # gPlayState / GlobalContext

ACTORCAT = ["SWITCH", "BG", "PLAYER", "EXPLOSIVE", "NPC", "ENEMY", "PROP", "ITEMACTION",
            "MISC", "BOSS", "DOOR", "CHEST"]
ACTOR_EN_ITEM00 = 0x0015   # collectibles: rupees, hearts, ...


def read_features(r: N64Ram) -> dict:
    sc, ps = SAVE_CTX, PLAY_STATE
    player = r.u32(ps + 0x1C44)
    f = {
        "magic_ok": r.ram[(sc + 0x1C) & 0x3FFFFF:(sc + 0x22) & 0x3FFFFF] == b"ZELDAZ",
        "entrance": r.u32(sc + 0x00),
        "age": r.u32(sc + 0x04),
        "day_time": r.s16(sc + 0x0C) & 0xFFFF,
        "health_cap": r.s16(sc + 0x2E),
        "health": r.s16(sc + 0x30),
        "rupees": r.s16(sc + 0x34),
        "scene": r.s16(ps + 0xA4),
        "player_ptr": hex(player),
    }
    if 0x80000000 <= player < 0x80400000:
        f["link_pos"] = tuple(round(v, 2) for v in r.vec3(player + 0x24))
        f["link_yaw"] = r.s16(player + 0x32) & 0xFFFF   # shape.rot.y
    # Actor lists: ActorContext at PlayState+0x1C24, lists[12] of {s32 len; Actor* head} at +0x0C
    counts, items = {}, []
    for cat in range(12):
        n = r.u32(ps + 0x1C24 + 0x0C + cat * 8)
        head = r.u32(ps + 0x1C24 + 0x10 + cat * 8)
        counts[ACTORCAT[cat]] = n
        a, guard = head, 0
        while 0x80000000 <= a < 0x80400000 and guard < 64:
            if (r.u32(a) >> 16) == ACTOR_EN_ITEM00:
                items.append((r.s16(a + 0x1C), tuple(round(v, 1) for v in r.vec3(a + 0x24))))
            a = r.u32(a + 0x124)  # Actor.next
            guard += 1
    f["actor_counts"] = counts
    f["collectibles"] = items
    return f
