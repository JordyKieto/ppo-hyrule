# Field scorecard: what the world model actually predicts

One row per declared field. Generated from the three P3 arms; regenerate with the commands in section 10.

**Read `skill` as:** `1 - model_error / do-nothing_error`, computed ONLY on the steps where the field
actually changed. 0.000 means exactly as good as assuming nothing happens. Negative means worse than that.
The restriction to changed steps matters -- over whole episodes most fields never move, so a frozen model
scores 0.92-1.00 and looks fine (LATENT_WORLD_MODEL.md section 23).

**`n` is the sample count** and it gates everything. Under 50 changed samples in the test split, the number
is not a measurement and the verdict is `no power`, not `FAIL`.

## 1. The three arms

| arm | causal graph | edges | what it tests |
|---|---|---|---|
| **discovered** | learned by CMI, eps 0.05 | 55 | the paper's method |
| **G1** | every edge on, frozen | 8,286 | the paper's own `OOFULL` baseline: same model, no causal graph |
| **empty** | no edges, frozen | 0 | the floor. With no parents a head can only predict do-nothing |

All three: same data (all four v3 collections, 120 random episodes), same seed, same 3-term objective,
event oversampling K=50. Checkpoints `runs/p2-loop-oversample`, `runs/p3-full`, `runs/p3-empty`.
Scored on 81 held-out test-slot episodes, 46,926 transitions.

## 2. `Link` -- the player

*the player character*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `pos` | world position (x,y,z) | approach, shield range | real | 3897 | 8.30 | 5.4060 | +0.144 | +0.132 | -0.010 | **PASS** |
| `vel` | velocity vector |  | real | 2804 | 5.98 | 1.1422 | -0.033 | -0.008 | -0.007 | FAIL |
| `yaw` | world facing angle | facing -> shield_facing | angle | 1892 | 4.03 | 29.1076 | +0.000 | -0.000 | -0.000 | marginal |
| `sf2` | status bits 2 |  | bits | 1746 | 3.72 | 0.0369 | +0.186 | +0.130 | +0.001 | **PASS** |
| `shape_yaw` | rendered body facing (differs from yaw mid-turn) |  | angle | 1701 | 3.62 | 19.8424 | -0.003 | +0.008 | -0.003 | FAIL |
| `speed` | scalar speed |  | real | 1479 | 3.15 | 3.1527 | +0.260 | +0.372 | +0.000 | **PASS** |
| `action_fn_rel` | which action routine Link is running |  | categorical | 1227 | 2.61 | 1.0000 | +0.000 | +0.050 | +0.000 | FAIL |
| `ztarget_timer` | Z-target hold timer |  | count | 1090 | 2.32 | 3.9220 | +0.550 | +0.531 | -0.002 | **PASS** |
| `sf1` | status bits 1 (shielding, dead, z-targeting...) | shield bit, dead bit | bits | 956 | 2.04 | 0.0510 | +0.419 | +0.506 | +0.000 | **PASS** |
| `fall_distance` | how far Link has fallen |  | real | 726 | 1.55 | 4.2603 | +0.198 | +0.233 | -0.000 | **PASS** |
| `melee_state` | is a sword swing live |  | categorical | 468 | 1.00 | 1.0000 | +0.342 | +0.432 | +0.000 | **PASS** |
| `sf3` | status bits 3 |  | bits | 455 | 0.97 | 0.1269 | +0.361 | +0.355 | +0.000 | **PASS** |
| `bg_flags` | collision bits: touching ground / wall / ceiling / water |  | bits | 396 | 0.84 | 0.1765 | +0.017 | +0.522 | +0.000 | **PASS** |
| `floor_height` | height of the floor under Link |  | real | 347 | 0.74 | 1.2705 | +0.022 | +0.112 | +0.003 | **PASS** |
| `dist_interact_wall` | distance to an interactable wall |  | real | 264 | 0.56 | 1.3098 | -0.006 | -0.009 | -0.004 | FAIL |
| `melee_anim` | which swing animation |  | categorical | 203 | 0.43 | 1.0000 | +0.000 | +0.227 | +0.000 | FAIL |
| `floor_pitch` | slope of the floor |  | angle | 199 | 0.42 | 4.4123 | +0.002 | +0.004 | +0.003 | marginal |
| `parallel_yaw` | lock-on facing |  | angle | 119 | 0.25 | 61.9830 | +0.000 | -0.000 | +0.000 | marginal |
| `hp` | health | link_damage, heal, death | categorical | 84 | 0.18 | 1.0000 | +0.000 | +0.000 | +0.000 | FAIL |
| `wall_yaw` | normal of the wall being touched |  | angle | 58 | 0.12 | 63.4185 | -0.000 | -0.000 | -0.000 | FAIL |
| `y_dist_to_ledge` | height drop to a ledge (a sentinel, 399.96 when none) |  | real | 38 | 0.08 | 399.9600 | -0.000 | +0.010 | +0.000 | no power |
| `held_item` | item in hand |  | categorical | 18 | 0.04 | 1.0000 | +0.000 | +0.000 | +0.000 | no power |
| `grass` | grass cut counter |  | count | 12 | 0.03 | 5.0000 | +0.041 | +0.112 | +0.000 | no power |
| `nuts` | deku nuts |  | count | 1 | 0.00 | 5.0000 | +0.000 | +0.000 | +0.000 | no power |
| `hp_cap` | max health |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `seeds` | slingshot ammo |  | count | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `knockback_type` | knockback kind |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `door_type` | door kind |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `door_direction` | door facing |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `door_timer` | door animation timer |  | count | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |

## 3. `Camera` -- Camera

*the game camera; the policy's observation contains its yaw*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `eye` | camera position |  | real | 5057 | 10.78 | 5.9137 | -0.037 | -0.013 | -0.016 | FAIL |
| `pitch` | camera pitch |  | angle | 3668 | 7.82 | 0.9681 | -0.020 | -0.053 | -0.019 | FAIL |
| `yaw` | world facing angle |  | angle | 3159 | 6.73 | 4.8471 | -0.001 | -0.007 | -0.005 | FAIL |

## 4. `Room` -- Room

*which scene/room we are in; leaving the room is a terminal*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `scene` | scene id | left_room | categorical | 12 | 0.03 | 1.0000 | +0.000 | +0.333 | +0.000 | no power |
| `room` | room id within the scene |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |

## 5. `En_Dekubaba` -- Deku Baba

*the plant enemy we fight. 1-3 per room. Actor id 0x0055*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `yaw_to_player` | angle from this actor to Link |  | angle | 7321 | 5.76 | 4.1533 | +0.000 | -0.014 | -0.000 | marginal |
| `timer` | this actor's action timer |  | count | 3463 | 2.72 | 3.8839 | -0.337 | -0.384 | -0.011 | FAIL |
| `pos` | world position (x,y,z) | approach, shield range | real | 3181 | 2.50 | 3.1152 | +0.074 | +0.009 | -0.029 | **PASS** |
| `focus` | focus point used for head-tracking and lock-on |  | real | 3180 | 2.50 | 2.7452 | -0.027 | -0.070 | -0.003 | FAIL |
| `shape_yaw` | rendered body facing (differs from yaw mid-turn) |  | angle | 1440 | 1.13 | 10.3656 | +0.001 | +0.003 | +0.003 | marginal |
| `fn_rel` | which action routine this actor is running |  | categorical | 578 | 0.45 | 1.0000 | +0.000 | +0.066 | +0.000 | FAIL |
| `state_idx` | decoded behaviour state (idle/lunge/hit...) | stun_hit, lunge/block/dodge | categorical | 553 | 0.43 | 1.0000 | +0.067 | +0.089 | +0.000 | **PASS** |
| `is_targeted` | is Link Z-targeting this actor | lockon, lock_hold | boolean | 145 | 0.11 | 1.0000 | +0.000 | +0.483 | +0.000 | FAIL |
| `has_arrow` | does Navi's arrow point at this actor |  | boolean | 106 | 0.08 | 1.0000 | +0.000 | +0.000 | +0.000 | FAIL |
| `vel` | velocity vector |  | real | 27 | 0.02 | 0.8974 | -0.001 | +0.335 | -0.000 | no power |
| `hp` | health | enemy_damage, kill, stun_hit | categorical | 27 | 0.02 | 1.0000 | +0.000 | +0.185 | +0.000 | no power |
| `present` | does this actor exist this step | target_lost | boolean | 25 | 0.02 | 1.0000 | +0.000 | +0.000 | +0.000 | no power |
| `yaw` | world facing angle |  | angle | 5 | 0.00 | 39.2179 | +0.001 | -0.000 | -0.001 | no power |
| `params` | spawn parameters (constant) |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |

## 6. `En_St` -- Skulltula

*big spider hanging from the ceiling. Sways, never dies. Actor id 0x0037*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `focus` | focus point used for head-tracking and lock-on |  | real | 14675 | 10.43 | 0.3212 | +0.677 | +0.733 | +0.000 | **PASS** |
| `pos` | world position (x,y,z) |  | real | 14674 | 10.43 | 0.3204 | +0.663 | +0.715 | -0.000 | **PASS** |
| `vel` | velocity vector |  | real | 14674 | 10.43 | 0.0779 | -0.048 | +0.240 | -0.000 | FAIL |
| `yaw_to_player` | angle from this actor to Link |  | angle | 10218 | 7.26 | 1.5099 | -0.001 | -0.002 | -0.002 | FAIL |
| `present` | does this actor exist this step |  | boolean | 36 | 0.03 | 1.0000 | +0.000 | +0.000 | +0.000 | no power |
| `hp` | health |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `yaw` | world facing angle |  | angle | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `shape_yaw` | rendered body facing (differs from yaw mid-turn) |  | angle | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `fn_rel` | which action routine this actor is running |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `state_idx` | decoded behaviour state (idle/lunge/hit...) |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `timer` | this actor's action timer |  | count | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `params` | spawn parameters (constant) |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `is_targeted` | is Link Z-targeting this actor |  | boolean | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `has_arrow` | does Navi's arrow point at this actor |  | boolean | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |

## 7. `En_Sw` -- Skullwalltula

*small wall spider. Never moves at all. Actor id 0x0095*

| field | what it is | reward term | kind | n changed | chg% | identity err | discovered | G1 | empty | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `yaw_to_player` | angle from this actor to Link |  | angle | 10506 | 7.47 | 1.1684 | -0.016 | -0.017 | -0.018 | FAIL |
| `present` | does this actor exist this step |  | boolean | 36 | 0.03 | 1.0000 | +0.000 | +0.000 | +0.000 | no power |
| `focus` | focus point used for head-tracking and lock-on |  | real | 12 | 0.01 | 2.4020 | -0.000 | +0.000 | +0.000 | no power |
| `pos` | world position (x,y,z) |  | real | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `vel` | velocity vector |  | real | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `hp` | health |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `yaw` | world facing angle |  | angle | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `shape_yaw` | rendered body facing (differs from yaw mid-turn) |  | angle | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `fn_rel` | which action routine this actor is running |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `state_idx` | decoded behaviour state (idle/lunge/hit...) |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `timer` | this actor's action timer |  | count | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `params` | spawn parameters (constant) |  | categorical | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `is_targeted` | is Link Z-targeting this actor |  | boolean | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
| `has_arrow` | does Navi's arrow point at this actor |  | boolean | 0 | 0.00 | 0.0000 | -- | -- | -- | DEAD |
## 8. Totals

| verdict | count | meaning |
|---|---|---|
| **PASS** | 14 | beats do-nothing on the steps where the field changes, with >= 200 samples |
| FAIL | 18 | enough samples, no skill or worse than do-nothing |
| marginal | 5 | positive but under +0.01 |
| no power | 12 | fewer than 50 changed samples: NOT a measurement |
| DEAD | 28 | never changes in this data at all |
| | **77** | declared fields |

**Only 37 of 77 declared fields are even scoreable.** 28 never move (every `Skullwalltula` field, most of
`Skulltula`, all four door fields, `Room.room`, `Link.seeds`, `hp_cap`, `En_Dekubaba.params`), and another 12
move too rarely to judge.

### 8.1 Reward fields specifically

`CombatTracker.step()` reads ten fields. Four pass:

| field | n | discovered | verdict |
|---|---|---|---|
| `Link.sf1` | 956 | +0.419 | **PASS** |
| `Link.pos` | 3897 | +0.144 | **PASS** |
| `En_Dekubaba.pos` | 3181 | +0.074 | **PASS** |
| `En_Dekubaba.state_idx` | 553 | +0.067 | **PASS** |
| `Link.yaw` | 1892 | +0.000 | marginal |
| `En_Dekubaba.is_targeted` | 145 | +0.000 | FAIL |
| `Link.hp` | 84 | +0.000 | FAIL |
| `En_Dekubaba.hp` | **27** | +0.000 | **no power** |
| `En_Dekubaba.present` | **25** | +0.000 | **no power** |
| `Room.scene` | **12** | +0.000 | **no power** |

The four that pass feed the SHAPING terms -- approach, shield-facing, lock-hold. **Every OUTCOME term**
-- damage, kill, death, room-exit -- **rests on a field that either fails or cannot be measured.**

## 9. What the arm comparison says

Counting only fields with >= 200 changed samples:

| field group | discovered | G1 | reading |
|---|---|---|---|
| position (`Link.pos`, `En_Dekubaba.pos`) | **+0.144 / +0.074** | +0.132 / +0.009 | masking WINS, `En_Dekubaba.pos` by 8x |
| state machines and flags (`state_idx`, `sf1`) | +0.067 / +0.419 | **+0.089 / +0.506** | `G1` WINS |
| every field | -- | -- | empty is <= 0 everywhere, so parents matter |

**This is the paper's own documented trade**, not an implementation failure. Yu et al. section F:

> "In AsymBlock where the causal dependencies are **inherently sparse**, OOCDM+ raises the best performance.
> In **Walker where dependencies are inherently denser, OOFULL+ shows the best performance** in the training
> and in-distribution data, whereas **OOCDM+ best generalizes to o.o.d. data**."

Position has sparse dependencies (velocity, collision); state machines have dense ones. We reproduce both
halves.

**And it exposes what we have NOT measured.** The paper's claim for discovery is generalisation to
out-of-distribution data. Every number here is in-distribution -- held-out episodes, same task, same start
slots. **We are measuring the axis `OOFULL` wins on and not the axis OOCDM wins on.**

## 10. Reproduce

    # the three arms
    .venv/bin/python scripts/train_oocdm_loop.py --init runs/p1-paper-obj/model.pt --iters 3 \
        --oversample 50 --eps 0.05 --cmi-batches 5 --cmi-batch-size 2048 --run-dir runs/p2-loop-oversample
    .venv/bin/python scripts/train_oocdm_loop.py ... --graph-mode full  --run-dir runs/p3-full
    .venv/bin/python scripts/train_oocdm_loop.py ... --graph-mode empty --run-dir runs/p3-empty

    # the ten reward fields, with the power column
    .venv/bin/python scripts/eval_reward_fields.py --ckpt runs/p2-loop-oversample/loop_state.pt

wandb: project `zelda-rl-oocdm`, groups `oocdm-p2` and `oocdm-p3`. Each run logs `graph/edges`,
`graph/kind` and `graph/jaccard_prev`, so two runs differing only in their causal graph are distinguishable.
