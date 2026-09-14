"""Replay a recorded boss fight (runs/manual/boss_*.jsonl, 20 Hz) through GohmaTracker at 10 Hz: prints events and
per-component reward totals, to validate the reward against what the human actually did (no Dolphin needed)."""
import json, math, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import CombatConfig, CombatState, Enemy
from zelda_rl.gohma import GohmaTracker, boss_features, gohma_state, kind, larva_state, LARVA_STATE

BOSS_UPD, LARVA_UPDS = 0x801E75F8, (0x801EDED0, 0x801EDC50)   # traces predate the +0x130 read: bases from savestates


def to_state(d):
    en = {}
    for a in d["actors"].values():
        if a["cat"] not in (5, 9) or "raw" not in a:
            continue
        b = bytes.fromhex(a["raw"])
        fn_off = 0x180 if a["cat"] == 9 else 0x2A0
        fn = struct.unpack_from(">I", b, fn_off - 0x14C)[0] if a["id"] == 0x28 else 0
        upd = BOSS_UPD if a["cat"] == 9 else next((u for u in LARVA_UPDS if fn - u in LARVA_STATE), LARVA_UPDS[0])
        en[a["ptr"]] = Enemy(a["ptr"], a["id"], a["hp"], tuple(a["pos"]), fn=fn, yaw_to_player=a["yaw_to_player"],
                             shape_yaw=a["shape_yaw"], cat=a["cat"], params=a["params"], upd=upd,
                             focus=(a["pos"][0], a["pos"][1] + 60, a["pos"][2]), yaw=a["yaw"])
    return CombatState(scene=d["scene"], room=d["room"], hp=d["hp"], hp_cap=64 if d["hp"] > 48 else 48, sf1=d["sf1"],
                       pos=tuple(d["pos"]), yaw=d["yaw"], targeted=d["targeted"], arrow=0, enemies=en,
                       held_item=d["held_item_action"] or 0, seeds=d["ammo"][6], nuts=d["ammo"][1])


def main(path, t_start=0.0, t_end=1e9, max_seconds=300.0):
    rows = [json.loads(l) for l in open(path)]
    t0 = rows[0]["t"]
    rows = [r for r in rows if t_start <= r["t"] - t0 <= t_end][::2]
    cfg = CombatConfig(profile="gohma", max_seconds=max_seconds, target_ids=(0x28,), room_radius=5000, fall_drop=1000,
                       timeout_penalty_per_target=10.0, success_time_bonus=30.0)
    tr = GohmaTracker(cfg)
    states = [to_state(r) for r in rows]
    tr.reset(states[0], int(max_seconds * 10))
    for i in range(1, len(states)):
        r, term, ev = tr.step(states[i - 1], states[i])
        feats = boss_features(states[i], tr)
        if ev:
            g = next((e for e in states[i].enemies.values() if kind(e) == "gohma"), None)
            print(f"  t={rows[i]['t'] - t0:6.1f}s r={r:+6.2f} gohma {gohma_state(g) if g else '-':20s} {ev}")
        if term:
            break
    ep = tr.ep
    print(f"END {ep['end']} | boss dmg {ep['boss_damage']} defeated {ep['boss_defeated']} stuns {ep['stuns']} seed_stuns "
          f"{ep['seed_stuns']}/{ep['seeds_fired']} seeds | larva kills {ep['larva_kills']} | hearts lost {ep['hp_lost'] / 16} | "
          f"blocks {ep['boss_blocks']}/{ep['boss_attacks']} | return {ep['return']:.1f}")
    print("parts", {k: round(v, 2) for k, v in sorted(ep["r_parts"].items(), key=lambda kv: -abs(kv[1]))})
    print("sum(parts) == return:", abs(sum(ep["r_parts"].values()) - ep["return"]) < 1e-6, "| obs block finite:", bool((feats == feats).all()))


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:]))
