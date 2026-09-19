# Adding New Observations and Rewards: The Pattern

How new game state (enemy behaviour, blocks, lock-on, …) goes from "it's somewhere in RAM" to a verified observation feature and reward term. Every step below has been run on this project; the Deku Baba behaviour decode (`docs/COMBAT_TASK.md`) is the worked example throughout.

```
 0. Look up ──▶ 1. Record ──▶ 2. Decode ──▶ 3. Verify ──▶ 4. Implement ──▶ 5. Reward ──▶ 6. Validate ──▶ 7. Document ──▶ 8. Ship
 the decomp     human play      offline        all start      read + feature    pay() terms    replay + smoke    RAM_MAP etc.     fleet run
 (if named)     + RAM trace     candidates     states, live   (versioned)       + metrics      + live run
```

Steps 1–2 are for fields **nobody has named** — actor-specific behaviour, per-enemy timers. For anything in
the engine's own structs (`Actor`, `Player`, collision, camera) **skip to step 0**: the field already has a
name, a type and an offset in the decompilation, and searching RAM for it by hand is wasted effort.

---

## 0. Look it up in the decompilation (do this FIRST)

**Why:** the OoT decompilation names every field of the engine structs. Recalling offsets from memory does
not work and fails *silently* — a wrong offset returns a plausible number, so it reaches the dataset as a
constant and nothing errors. This step was skipped once on this project and cost a full cycle: guessed
offsets for Link's collision state (`+0x84`, `+0x88`, `+0x90`) were rejected as "constants across
savestates", when in fact `+0x84` is `depthInWater` (−32000, i.e. above water) and `+0x88` **is**
`bgCheckFlags` (`0x81` = `GROUND|GROUND_STRICT`). They were correct and read as failures.

**Source of truth** — [github.com/zeldaret/oot](https://github.com/zeldaret/oot):

| what | file |
|---|---|
| `struct Actor` (pos, velocity, speed, collision, dist-to-player) | [`include/actor.h`](https://raw.githubusercontent.com/zeldaret/oot/main/include/actor.h) |
| `struct Player` (action state, attack, Z-target, doors, ledges) | [`include/player.h`](https://raw.githubusercontent.com/zeldaret/oot/main/include/player.h) |
| `BGCHECKFLAG_*`, `ACTOR_FLAG_*` | `include/actor.h` |
| collision / background check | `include/bgcheck.h` |
| per-actor structs | `src/overlays/actors/ovl_<Name>/z_<name>.h` |

```sh
# there is no `gh` on this machine -- use curl
curl -sL 'https://api.github.com/repos/zeldaret/oot/git/trees/main?recursive=1' -o /tmp/oot_tree.json   # find files
curl -sL https://raw.githubusercontent.com/zeldaret/oot/main/include/actor.h -o /tmp/oot_actor.h
sed -n '/typedef struct Actor {/,/} Actor;/p' /tmp/oot_actor.h                                          # read the struct
```

### The version shift: retail `Player` fields are decomp offset − 0x10

The decomp prints offsets for a `DEBUG_FEATURES` build, where `Actor` ends with a debug pad:

```c
    /* 0x138 */ struct ActorOverlay* overlayEntry;
#if DEBUG_FEATURES
    /* 0x13C */ char dbgPad[0x10];
#endif
} Actor; // size = 0x14C
```

We run **retail NTSC 1.2**, which is not a `DEBUG_FEATURES` build, so `Actor` is **0x13C**, not 0x14C:

- **`Actor` fields** (offset < 0x13C) — use the decomp offset **unchanged**.
- **`Player` fields** (everything after the embedded `Actor`) — **decomp offset − 0x10**.

Two offsets already in `zelda_rl/combat.py` confirm the shift independently: `stateFlags1` 0x67C−0x10 =
`0x66C`, and `heldItemAction` 0x151−0x10 = `0x141`.

Watch for the same class of mistake *inside* a struct: an omitted member shifts everything after it. A
recalled `Actor` layout that dropped `Vec3f scale` at `0x50` put the whole collision block 0xC too low.
Read the struct; don't reconstruct it.

### Then verify, because the shift is an assumption too

`scripts/verify_link_fields.py` drives a behaviour that *should* move each field and asserts it responds, so
a wrong offset fails loudly. Prefer **redundant pairs that must agree** over "the value looks sensible":

- `BGCHECKFLAG_WALL` (bit 3 of `bgCheckFlags` `+0x088`) must equal `wallPoly` `+0x074` being non-null
  — measured **1.0000 agreement over 2,510 frames**. Two independent offsets, one conclusion.
- `floorHeight` `+0x080` must track Link's y while grounded — measured mean \|diff\| **0.15**.
- `Player.speedXZ` (`0x838−0x10`) must correlate with `Actor.speed` `+0x068` — measured **0.944**.

**Pitfall — a redundancy check is not a behaviour check.** "Blocked by a wall ⇒ WALL bit set" is *false*:
Link slides **along** walls while still moving (212 such frames, mean speed 3.48 vs 4.97 for
free-and-untouching). That made a correct field score as a failure. Assert the invariant the engine
guarantees, not the one your scenario suggests.

---

## 1. Record: a human performs the behaviour while RAM is traced

**Why:** you can't decode what you haven't seen change. The human supplies the ground truth: "now it lunged, now I blocked, now it's stunned".

**How:**
- Run the signed Dolphin (`~/.cache/zelda-rl/Dolphin.app`, a debuggable copy) with the user's normal profile. `task_for_pid` needs Developer Mode, a `get-task-allow` target and a `cs.debugger` caller (`docs/FINDINGS.md` §3.1).
- Attach a recorder that logs at **20 Hz** (twice the 10 Hz env rate, so short events aren't aliased):
  - known fields (HP, position, flags)
  - **raw hex windows** of the structs you're investigating: the actor-specific region after the Actor header (retail `Actor` is **0x13C**; the Baba window used `+0x14C..+0x3A0`), and the Player region `+0x3F0..+0x900` (colliders)
- Print live events (HP changes, flag toggles, spawns and despawns) so both sides can see what's being captured.
- **Script the session.** Ask for each behaviour explicitly: stand near the enemy without attacking (lunges), shield while it lunges (blocks), take hits, kill one slowly and one quickly, die. The contrast between cases is what makes decoding possible.

**Tools:** `scripts/monitor_play.py` (events + 10 Hz trace), `scripts/monitor_fight.py` (20 Hz + raw struct windows). Traces go to `runs/manual/*.jsonl`.

**Pitfalls:**
- The official Dolphin build can't be read; switch the user to the signed copy.
- Numpy scalars in JSON crash the recorder: use `json.dumps(..., default=int)`.
- A guessed field that is really a float (e.g. `+0x14C` on the Baba) floods the live log. Filter live output to confirmed events and keep the raw data in the trace.

## 2. Decode: find candidate fields offline

Scan the raw windows with simple statistical signatures, then check them against the event timeline.

| Looking for | Signature | Baba example |
|---|---|---|
| **Behaviour / state machine** (action routine) | u32 in N64 RAM range; **few distinct values (≈3–20)**; **identical value set across instances of the same actor**; dwell times > 1 sample | `+0x1B0`: 8 values shared by all 3 Babas |
| Engine mode pointers | u32 with 2 shared values in the main code segment (`0x8008xxxx`) | `+0x19C`: loop vs one-shot animation updater |
| **Positions** | f32 triple, finite, within ±5,000; distance to the actor base varies a lot | `+0x154` swung 4–250 but did **not** track the lunge, so it was rejected |
| **Timers** | s16 ≥ 0, < 300, many 1–6 decrements in a row | `+0x1B6`, `+0x2BA` |
| **Direction** | s16 rotation spanning the full circle | shape rotation `+0xB6` (the head), vs world rotation (fixed base) |
| **Event flags** | a bit that flips at exactly the event frames (± 1) and rarely otherwise | shield flag drops for 1–2 frames on each block |

Then build a **timeline table** per actor: one row per contiguous stretch of each candidate value, with duration, relevant distances, and the overlapping human-labelled events (Link damaged, blocked, enemy hit, HP 0). Name each value by what consistently happens in it:
- the value right before every Link damage → `lunge`
- the value exactly at HP drops → `hit`
- the long value after a hit with the head down → `stunned`
- the value only at HP 0 → `dying`

**Pitfalls:**
- **Relocatable actor code.** An actor's code is loaded wherever there's room, so action-function pointers can change between scene loads. The Gohma larva functions moved by exactly 0x280 after a reload. Decode `fn − update function (actor +0x130)`, which is stable, not the absolute pointer. The Deku Baba table predates this and still uses absolute pointers; they matched in all 10 of its slots.
- **Shared actor ids.** Queen Gohma, her larvae and her debris are all id `0x0028`; they differ only by category and params.
- **Label states by what the action enables, not by name guesses.** Gohma's red-eye states were identified as "the only states in which a seed ever caused a stun" (`docs/GOHMA_TASK.md`).
- A single long segment overlaps many events. Name states by **short, repeated** segments.
- Some behaviours have two routine values (e.g. two `idle`, two `dying`). Merge them into one semantic state.
- A flag that flickers for 1–2 frames is invisible at 10 Hz. Derive the event from a slower signal instead (blocks came from `lunge` → `pull_back` cycles plus "no HP lost").

## 3. Verify: every start state, and live controls

- **Offline over all savestates:** read the field in every slot and require **zero unknown values** (`scripts/probe_v2_ram.py`-style loop). Code pointers are load-address dependent; they matched in all 10 slots here.
- **Live probe for anything control-related:** drive the pipe controller and watch the field (`scripts/probe_combat_controls.py`). This proved GC L = Z-target and GC R = shield, and that lock-on only happens once Navi's arrow is on the enemy.
- **Named engine fields come from the decompilation** (step 0), and the live probe is what confirms the
  offset and the −0x10 version shift. Use redundant-pair assertions (`scripts/verify_link_fields.py`), not
  eyeballing. Mark fields ✅ verified or 🔶 decomp-only in `docs/RAM_MAP.md`.
- **A field that looks constant is not evidence of a wrong offset.** Check whether it *should* vary in the
  states you sampled: `bgCheckFlags` is identical in every start savestate because Link is grounded and dry
  in all of them. Drive the behaviour that changes it before concluding anything.

## 4. Implement: read it, then add a fixed-size, versioned feature block

1. **Read** the field in the task's RAM reader (`read_combat_state` → `Enemy` dataclass fields). Keep reads cheap (µs): a handful of `struct.unpack_from` calls on the zero-copy view.
2. **Semantics in one place:** a decode table (`DEKU_BABA_STATE`) plus a derived property (`Enemy.state`) with an `other` fallback.
3. **Featurize with a fixed layout:**
   - enums → one-hot (include `other`)
   - angles → cos/sin
   - timers and counters → scaled to ~[0, 1] and clipped
   - absent entities → zeros (plus a present flag where relevant)
   - clip everything to ±10, NaN → 0
4. **Declare the layout** in the field list (`COMBAT_FIELDS`): name and size per field. `COMBAT_DIM` and the model's `global_dim` are derived from it, so the network adapts automatically.
5. **Bump the observation version** (`obs_version` in the trainer config). Checkpoints store `obs_version`, `obs_fields` and dims. `policy_io` / `play.py` rebuild the matching env and model from the checkpoint, so old checkpoints keep working, but **a new version can't resume an old run**.

## 5. Reward: every term goes through `pay()`

- Add the term inside `CombatTracker.step` via `pay(name, value, kind)`:
  - **`kind="out"`** for outcomes (kills, damage, death)
  - **`kind="shape"`** for behaviour shaping (lock-on, blocks)
- `pay` accumulates `ep["r_parts"][name]`; the trainer logs each as **`reward/<name>`**. Components sum to `episode/return` by construction.
- **Cap shaping terms per episode** (e.g. `block_cap`) so they can't be farmed, and keep them small relative to outcomes.
- **Count the underlying event too** (`ep["blocks"]`, `ep["lunges"]`, …) and add a `combat/*` metric. Rates like `block_rate` = blocks ÷ lunges are more informative than raw reward.
- **Terminal events** (death, leaving the room) must be penalized at least as hard as each other, so none is an escape hatch. Add a timeout penalty so stalling isn't free.

## 6. Validate: replay first, then smoke, then fleet

1. **Replay recorded traces through the tracker** (no Dolphin). Build `CombatState`s from the trace rows, subsample to 10 Hz, step the tracker, and assert:
   - event counts match what the human did (5 lunge hits = 5 damage events)
   - no fake events around known traps (despawn on death, spawn HP frame)
   - `sum(r_parts) == return`
2. **Local smoke run:** `train_combat.py --no-wandb --num-envs 2 --total-updates 1 --episode-seconds 15`. Checks the observation shape on real RAM, that new metrics appear in `updates.jsonl`, and there are no crashes.
3. **Watch it:** `play.py <ckpt> --slot N` shows events in the HUD.

## 7. Document
- `docs/RAM_MAP.md`: addresses with ✅/🔶 status and evidence.
- The task doc (`docs/COMBAT_TASK.md`): RAM signal table, decoded state table, reward table (with validation numbers), observation layout and version, metric list.
- This file: new tools or pitfalls.

## 8. Ship to the fleet
- `python -m fleet launch ... --run-tag <new-tag>` pushes the Mac's current code to every pod. **Don't edit code while a launch is pushing.**
- New observation version → new run tag / wandb group (not `--resume`).
- Compare against the previous group in wandb (`combat/block_rate`, `combat/death_rate`, `reward/*`).

---

## Checklist

- [ ] **Looked the field up in the decompilation** (`include/actor.h` / `include/player.h`), applied the
      −0x10 shift for `Player` fields, and verified it live with a redundant-pair assertion
- [ ] Recorded a scripted session covering every case (and its contrast cases) at 20 Hz with raw windows
- [ ] Candidate field found by signature and confirmed with an event timeline
- [ ] Zero unknown values across all start savestates; controls probed live
- [ ] Read in the RAM reader; decode table + `other` fallback
- [ ] Fixed-size feature block declared in the field list; observation version bumped
- [ ] Reward via `pay()`, capped if shaping, event counter + `combat/*` metric
- [ ] Trace replay: counts match, no trap events, parts sum to return
- [ ] Local smoke run passes; HUD shows the events
- [ ] RAM_MAP + task doc updated
- [ ] Fleet launched under a new run tag
