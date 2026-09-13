"""Build the Deku Baba combat savestate dataset: manifest.json + README.md, and optionally upload to S3.

Reads every Dolphin savestate (slots 1-10) offline, extracts the start conditions from game RAM (Link, items,
every enemy with its decoded behaviour), merges the hand-written context below and the policy results, and
writes datasets/zelda/combat/deku-baba/{manifest.json,README.md}. With --upload, pushes the states, the
manifest/README and the key checkpoints to s3://zelda-rl/zelda/combat/deku-baba/ (aws profile race-wm =
IAM user `datasets`; boto3 reads the profile itself, no credentials pass through this script).

    .venv/bin/python scripts/build_savestate_manifest.py            # manifest + README only
    .venv/bin/python scripts/build_savestate_manifest.py --upload   # + S3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from zelda_rl.combat import DEKU_BABA, read_combat_state  # noqa: E402
from zelda_rl.obs import SAVE_CTX, read_state_v2  # noqa: E402
from zelda_rl.savestate import decompress_state, find_n64_ram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = Path.home() / "Library/Application Support/Dolphin/StateSaves"
OUT_DIR = ROOT / "datasets/zelda/combat/deku-baba"
BUCKET, REGION, PREFIX, PROFILE = "zelda-rl", "us-west-1", "zelda/combat/deku-baba", "race-wm"
CONSOLE = f"https://{REGION}.console.aws.amazon.com/s3"

# 0x0095 / 0x0037 names are from the decomp actor table (not verified live); they sit on the walls (y 478-678) and
# upper floors (y ~1075), out of reach of the fight, and are not targets.
ACTOR_NAMES = {DEKU_BABA: "Deku Baba", 0x0095: "Skullwalltula (En_Sw)", 0x0037: "Skulltula (En_St)"}
ENV_UPDATE_HZ = 10

# Hand-written context per slot: where Link is, what is happening, why the state is in the set.
CONTEXT = {
    1: dict(title="Entrance, three Babas ahead",
            where="Deku Tree first room, standing just inside the entrance (position -4, 0, 597).",
            what="Full health. All three Deku Babas are underground and 530-830 units away, so Link has to walk up and wake them.",
            why="The baseline full fight. The exit is right behind Link, so it also tests that the policy doesn't leave the room."),
    2: dict(title="Mid-room, Babas already up",
            where="Deku Tree first room, in among the Babas.",
            what="Full health, already Z-targeting. Three Babas at 144-303 units: one idle (up and swaying), one hidden, one retracting.",
            why="Starts mid-engagement: the policy must fight immediately instead of approaching."),
    3: dict(title="A lunge is incoming",
            where="Deku Tree first room, close to the Babas.",
            what="Full health, Z-targeting. Three Babas at 130-355 units: hidden, retracting, and one **preparing a lunge** at Link.",
            why="Tests reacting to an attack already in progress (block, dodge or strike first)."),
    4: dict(title="One down, one hidden",
            where="Deku Tree first room.",
            what="Full health, Z-targeting. One Baba is dying (0 HP, being removed); the one live target is hidden underground 364 units away.",
            why="Late-fight state with a single hidden target that must be found and woken. Only 32% in training (timeouts); input was intermittently ignored on some Linux pods."),
    5: dict(title="Last Baba, hidden",
            where="Deku Tree first room.",
            what="Full health. A single live Baba (2 HP), hidden underground.",
            why="Cleanup: find and kill the last one quickly."),
    6: dict(title="Three Babas, close cluster",
            where="Deku Tree first room, near the middle.",
            what="Full health. Three hidden Babas at 211-276 units.",
            why="Full fight starting at medium range; tests handling several Babas waking at once."),
    7: dict(title="Mid-lunge while two more wait",
            where="Deku Tree first room.",
            what="Full health, Z-targeting. One Baba at 1 HP is **mid-lunge**; two more (2 HP) are hidden.",
            why="Hardest mix: an active attack plus hidden targets to find afterwards. 41% in training."),
    8: dict(title="Hurt Link, two Babas",
            where="Deku Tree first room.",
            what="Link at **2.5 hearts**. Two hidden Babas (2 HP each).",
            why="Lower health: mistakes are costlier. Highest training death rate (21%)."),
    9: dict(title="Low health, stunned Baba",
            where="Deku Tree first room, right next to a Baba.",
            what="Link at **1.5 hearts**. One Baba at 1 HP, **stunned** 91 units away: the kill window is open.",
            why="Tests taking a free kill immediately while one hit from death."),
    10: dict(title="Three Babas, one awake",
             where="Deku Tree first room.",
             what="Full health. Three Babas at 179-271 units: two hidden, one idle.",
             why="Full fight with one Baba already up and able to attack."),
}

# Results for the final policy (global update 150): local watch (sampled actions, one episode per slot) and the
# fleet's per-slot training stats (last 10 updates of gru-v2b rank 4).
LOCAL_G150 = {1: ("success", [8.0, 20.2, 24.3], 0.5, 83.0), 2: ("success", [2.6, 7.1, 14.7], 0.5, 82.9),
              3: ("success", [0.6, 7.3, 10.9], 0.5, 83.5), 4: ("success", [2.7], 0.0, 52.5),
              5: ("success", [2.7], 0.0, 53.6), 6: ("success", [5.2, 11.0, 17.5], 0.0, 87.1),
              7: ("success", [0.6, 11.9, 18.6], 1.5, 76.4), 8: ("success", [5.4, 8.8], 0.0, 70.5),
              9: ("success", [0.5], 0.0, 51.3), 10: ("success", [4.1, 7.1, 12.3], 0.0, 87.0)}
FLEET_BY_SLOT = {1: (0.90, 0.04), 2: (0.77, 0.0), 3: (0.67, 0.04), 4: (0.32, 0.0), 5: (0.93, 0.0),
                 6: (0.77, 0.09), 7: (0.41, 0.0), 8: (0.75, 0.21), 9: (0.90, 0.05), 10: (0.65, 0.10)}

CHECKPOINTS = [  # (local path, global update, run, local update, wandb run id, note)
    ("checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt", 150, "deku-combat-gru-v2b", 120, "vxejacj7",
     "FINAL: 10/10 slots cleared locally; 77% fleet success"),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00100.pt", 130, "deku-combat-gru-v2b", 100, "vxejacj7", ""),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00080.pt", 110, "deku-combat-gru-v2b", 80, "vxejacj7", ""),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00060.pt", 90, "deku-combat-gru-v2b", 60, "vxejacj7", ""),
    ("checkpoints/inbox/deku-combat-gru-v2_u00030.pt", 30, "deku-combat-gru-v2", 30, "nhj5bcde",
     "init for gru-v2b; 1 kill in 67 s on slot 1"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_links(key: str) -> dict:
    return {"s3_uri": f"s3://{BUCKET}/{key}",
            "console_url": f"{CONSOLE}/object/{BUCKET}?region={REGION}&prefix={key}"}


def dolphin_header(buf: bytes) -> dict:
    game_id = buf[:6].decode("ascii", "replace")
    tail = buf[0x20:0x40]
    version = tail[:next((i for i, b in enumerate(tail) if not 32 <= b < 127), len(tail))].decode("ascii")
    return {"game_id": game_id, "dolphin_version_string": version}


def yaw_deg(a: int) -> float:
    return round((a & 0xFFFF) * 360 / 65536, 1)


def slot_facts(path: Path) -> dict:
    buf = path.read_bytes()
    raw = decompress_state(buf)
    base = find_n64_ram(raw)
    ram = raw[base:base + 0x400000]
    g, c = read_state_v2(ram), read_combat_state(ram)
    b_item = ram[(SAVE_CTX + 0x68) & 0x3FFFFF]
    enemies = []
    for e in sorted(c.enemies.values(), key=lambda e: math.dist(e.pos, c.pos)):
        facing_err = ((e.shape_yaw - e.yaw_to_player + 32768) % 65536) - 32768
        enemies.append({
            "actor": ACTOR_NAMES.get(e.id, f"0x{e.id:04X}"), "actor_id": f"0x{e.id:04X}", "hp": e.hp,
            "alive_target": e.id == DEKU_BABA and e.hp > 0, "behaviour": e.state,
            "distance_to_link": round(math.dist(e.pos, c.pos)), "position": [round(v, 1) for v in e.pos],
            "head_facing_link_deg_off": round(abs(facing_err) * 360 / 65536, 1), "action_timer": e.timer,
        })
    targets = [e for e in enemies if e["alive_target"]]
    return {
        "file": path.name, "bytes": len(buf), "sha256": sha256(path), **dolphin_header(buf),
        "game_time_vi": g.vi,
        "scene": c.scene, "room": c.room,
        "link": {"hearts": c.hp / 16, "heart_containers": c.hp_cap / 16, "rupees": g.rupees,
                 "position": [round(v, 1) for v in c.pos], "facing_deg": yaw_deg(c.yaw),
                 "camera_yaw_rad": round(g.cam_yaw, 3),
                 "z_targeting": c.bit(15), "shielding": c.bit(22), "locked_on_enemy": c.targeted in c.enemies,
                 "b_button_item": int(b_item), "equipment_bits": f"0x{g.equipment:04X}"},
        "enemies": enemies,
        "targets_at_start": len(targets),
        "other_enemies": {n: sum(1 for e in enemies if e["actor"] == n) for n in sorted({e["actor"] for e in enemies})
                          if n != "Deku Baba"},
        "target_hp_total": sum(e["hp"] for e in targets),
    }


def ckpt_entry(path: str, g: int, run: str, u: int, wandb_id: str, note: str) -> dict:
    p = ROOT / path
    ck = torch.load(p, map_location="cpu", weights_only=False)
    h = hashlib.sha256(b"".join(v.numpy().tobytes() for v in ck["model"].values())).hexdigest()[:12]
    key = f"{PREFIX}/checkpoints/G{g:03d}_{run}_u{u:05d}.pt"
    return {"global_update": g, "run": run, "local_update": u, "obs_version": ck["config"].get("obs_version"),
            "recurrent": bool(ck["config"].get("recurrent")), "weight_hash": h, "bytes": p.stat().st_size,
            "sha256": sha256(p), "local_path": path, "wandb_run": f"https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/{wandb_id}",
            "note": note, **s3_links(key)}


def readme(m: dict) -> str:
    L = [f"# Deku Baba combat savestates (`{PREFIX}`)", "",
         "Ten Dolphin savestates of the **Deku Tree's first room** in *Ocarina of Time* (Wii VC `NACE01`, NTSC 1.2 ROM), "
         "all on File 1 \"LINK\" (3 heart containers, Kokiri Sword on B, Deku Shield, 0 rupees), "
         "used as episode start states for the combat PPO task (`docs/COMBAT_TASK.md`). "
         "The goal from each is to kill every live Deku Baba without dying or leaving the room. Every state also contains "
         "3 Skullwalltulas on the walls and 3 Skulltulas on the upper floors; they are out of reach and not targets.", "",
         f"- **S3:** `s3://{BUCKET}/{PREFIX}/` ([console]({CONSOLE}/buckets/{BUCKET}?region={REGION}&prefix={PREFIX}/)): "
         "private, SSE-AES256, versioned.",
         f"- **Manifest:** [`manifest.json`](manifest.json) (machine-readable; generated by `scripts/build_savestate_manifest.py` from game RAM).",
         f"- **Emulator:** Dolphin {m['emulator']['version']} (state files are version-locked). Loads on macOS ARM and Linux x86.",
         "", "## Use", "", "```sh",
         f"aws s3 sync s3://{BUCKET}/{PREFIX}/states/ \"$HOME/Library/Application Support/Dolphin/StateSaves/\" --profile {PROFILE}",
         "# play the final policy from any slot",
         f"aws s3 cp s3://{BUCKET}/{PREFIX}/checkpoints/G150_deku-combat-gru-v2b_u00120.pt checkpoints/kept/ --profile {PROFILE}",
         "DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin .venv/bin/python-dbg scripts/play.py \\",
         "  checkpoints/kept/G150_deku-combat-gru-v2b_u00120.pt --slot 7", "```", "",
         "Loading: Dolphin hotkey *Load State Slot N* (the RL env sends it over the hotkey pipe).", "",
         "## Slots", "",
         "| Slot | Title | Link | Live Babas (HP · behaviour · distance) | Final policy (G150) local | Fleet success / death |",
         "|---|---|---|---|---|---|"]
    for s in m["slots"]:
        f, r = s["ram"], s["results"]
        babas = "; ".join(f"{e['hp']} HP · {e['behaviour']} · {e['distance_to_link']}" for e in f["enemies"] if e["alive_target"])
        dying = sum(1 for e in f["enemies"] if e["actor_id"] == "0x0055" and not e["alive_target"])
        L.append(f"| {s['slot']} | {s['title']} | {f['link']['hearts']:.1f} ♥{' · Z-target' if f['link']['z_targeting'] else ''} | "
                 f"{babas}{f' (+{dying} dying)' if dying else ''} | {r['final_policy_local']['end']} in "
                 f"{r['final_policy_local']['kill_times_s'][-1]} s, −{r['final_policy_local']['hearts_lost']:g} ♥ | "
                 f"{r['fleet_training']['success_rate']:.0%} / {r['fleet_training']['death_rate']:.0%} |")
    L += ["", "## Context per slot", ""]
    for s in m["slots"]:
        L += [f"### Slot {s['slot']}: {s['title']}", "",
              f"- **Where:** {s['where']}", f"- **What:** {s['what']}", f"- **Why it's in the set:** {s['why']}",
              f"- **File:** [`states/{s['ram']['file']}`]({s['state']['console_url']}) · `{s['state']['s3_uri']}` · "
              f"sha256 `{s['ram']['sha256'][:16]}…`", ""]
    L += ["## Checkpoints", "", "| Global update | Run · update | Note | S3 | wandb |", "|---|---|---|---|---|"]
    for c in m["checkpoints"]:
        L.append(f"| **{c['global_update']}** | {c['run']} u{c['local_update']} | {c['note']} | "
                 f"[`{c['s3_uri'].split('/')[-1]}`]({c['console_url']}) | [run]({c['wandb_run']}) |")
    L += ["", "Global update G = gru-v2 update (G ≤ 30), gru-v2b update + 30 (G > 30). Training curves: `docs/COMBAT_TASK.md`.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--states-dir", type=Path, default=STATES_DIR)
    args = ap.parse_args()
    slots = []
    for n in range(1, 11):
        path = args.states_dir / f"NACE01.s{n:02d}"
        facts = slot_facts(path)
        end, kills, lost, ret = LOCAL_G150[n]
        slots.append({"slot": n, **CONTEXT[n], "ram": facts,
                      "state": s3_links(f"{PREFIX}/states/{path.name}"),
                      "results": {"final_policy_local": {"global_update": 150, "end": end, "kill_times_s": kills,
                                                         "hearts_lost": lost, "return": ret, "actions": "sampled"},
                                  "fleet_training": {"global_update": "145-154", "success_rate": FLEET_BY_SLOT[n][0],
                                                     "death_rate": FLEET_BY_SLOT[n][1]}}})
        print(f"slot {n}: {facts['link']['hearts']} hearts, {facts['targets_at_start']} targets, "
              f"{[(e['hp'], e['behaviour'], e['distance_to_link']) for e in facts['enemies']]}")
    manifest = {
        "dataset": PREFIX, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bucket": BUCKET, "region": REGION,
        "game": {"title": "The Legend of Zelda: Ocarina of Time", "platform": "Wii Virtual Console", "game_id": "NACE01",
                 "title_id": "000100014e414345", "n64_rom": "NTSC 1.2", "save_file": "File 1 \"LINK\""},
        "emulator": {"name": "Dolphin", "version": "2603a", "savestate_version": 177},
        "location": {"scene": 0, "scene_name": "Inside the Deku Tree", "room": 0, "room_name": "entrance room (first room)"},
        "task": {"doc": "docs/COMBAT_TASK.md", "env_hz": ENV_UPDATE_HZ, "episode_max_s": 180,
                 "success": "all Deku Babas alive at start defeated", "terminal": ["success", "death", "left_room"]},
        "fields": {"hearts": "save ctx +0x30 / 16", "behaviour": "Deku Baba action routine +0x1B0 (docs/RAM_MAP.md §11)",
                   "head_facing_link_deg_off": "|shape yaw +0xB6 - yaw to Link +0x8A|, 0 = looking at Link",
                   "game_time_vi": "VI counter (60 Hz) at save time"},
        "slots": slots,
        "checkpoints": [ckpt_entry(*c) for c in CHECKPOINTS],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT_DIR / "README.md").write_text(readme(manifest))
    print(f"wrote {OUT_DIR}/manifest.json, README.md")
    if not args.upload:
        return
    import boto3
    s3 = boto3.Session(profile_name=PROFILE, region_name=REGION).client("s3")
    uploads = [(args.states_dir / s["ram"]["file"], s["state"]["s3_uri"], s["ram"]["sha256"]) for s in slots]
    uploads += [(ROOT / c["local_path"], c["s3_uri"], c["sha256"]) for c in manifest["checkpoints"]]
    uploads += [(OUT_DIR / "manifest.json", f"s3://{BUCKET}/{PREFIX}/manifest.json", None),
                (OUT_DIR / "README.md", f"s3://{BUCKET}/{PREFIX}/README.md", None)]
    for local, uri, digest in uploads:
        key = uri.split(f"s3://{BUCKET}/", 1)[1]
        extra = {"Metadata": {"sha256": digest}} if digest else {}
        s3.upload_file(str(local), BUCKET, key, ExtraArgs=extra)
        head = s3.head_object(Bucket=BUCKET, Key=key)
        assert head["ContentLength"] == local.stat().st_size, uri
        print(f"uploaded {uri} ({head['ContentLength'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
