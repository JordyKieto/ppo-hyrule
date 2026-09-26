# Focus: finish the Queen Gohma fight

**Decision (2026-09-26).** The world model is parked. It is not abandoned as knowledge -- it changed how we
frame game state, and that part is kept and written down below -- but it is not on the critical path to a
defeated Gohma, and it has been the whole path for weeks. One goal now: **a policy that defeats Queen Gohma
from combat-engaged start states.**

`docs/WM_EXPERIMENT_REGIME.md` is the closing record of the world-model work; its checkpoints are in `runs/`
and nothing there needs to be rerun to proceed here.

## Definition of done

Deku Baba was called solved at 78% fleet success / 10-of-10 slots locally (`COMBAT_TASK.md`). Gohma needs a
number set in advance so the run is not judged by vibes:

* **Primary: defeat rate >= 50%** from engaged slots inside the 300 s limit.
* **Leading indicator: `red_eye_seeds / seeds_fired` above the chance rate** (see "the actual blocker"), which
  is what must move before any defeat rate can.
* Reported against a control: the same measurement for the current best checkpoint and for a do-nothing policy.

Ten fleet runs (`gohma-v1` .. `v10`) have produced **0 reliable defeats** -- best was `gohma-v8` at return
-5.6, 0.61 Gohma HP per episode, 3 defeats in 9,124 episodes on `v7`. So the bar above is not a formality.

## Before / after: what the world model actually changed about our input framing

This is the part of the WM work that pays for itself. The left column is the observation that SOLVED Deku Baba;
the right is what the WM established. The last two columns are the honest part.

| Concept | Deku Baba PPO (obs v4, 483 dims, solved) | What the WM established | Measured? | Use in PPO? |
|---|---|---|---|---|
| Entity decomposition | `16 x 24` actor table -> shared per-actor MLP -> **masked mean+max pool** (`EntityActorCritic`) | typed entity tokens with **attention**, so each entity keeps its identity instead of being averaged away | no head-to-head measurement | **yes** -- pooling is exactly wrong for a fight with one eye to aim at and 3 larvae to track |
| Entity classes | ONE actor class, distinguished only by an id embedding | **three** classes: Link, actor, static environment -- each with its own projection, so "I am environment" is learned, not inferred from content | no | yes |
| Static geometry | 9 `bgCheckFlags` contact bits stuffed INSIDE Link's own vector | the environment gets its own token pool, **keys/values only, never queries** (a wall has no dynamics to predict) | ablation: `l_bg_bits` **+39%** on the position head, 3rd strongest input -- the model was leaning on those bits hard | yes |
| Position | raw relative offsets + distance, in world units | nav-grid **cell embedding + within-cell offset**, no raw coordinates anywhere (`zelda_rl/navgrid.py`) | **NOT measured.** An earlier draft of `WM_EXPERIMENT_REGIME.md` claimed "+156% on the position head" for this; that number has no source in this repo and I withdrew it. The cell codec is not even an ablatable class in `ablate_classes.py` | plausible, unproven -- treat as an experiment, not a settled win |
| Rotation frames | world-frame offsets + Link's body yaw | **dual local frames**: camera frame (the stick is camera-relative) and Link's facing frame (the sword hits in front) | ablation: `l_cam` **+21%** on position | **already done** -- obs v6 added camera + body frames by hand for Gohma BEFORE the WM did. This one went PPO -> WM, not the other way |
| Discrete quantities | HP as a fraction, continuous | HP **classified**, and binned on the input side too | damage F1 **0.286 -> 0.906** (`LATENT_WORLD_MODEL.md`) | yes, for HP and any other rare-event field on the input side |
| Behaviour state | 10-way one-hot for the 2 nearest targets | phase variables beat state labels: a Baba's **height** (`a_rel_y`, how far it has emerged) predicted its behaviour state at ~**90x** baseline | ablation, yes | **yes and cheap** -- the Gohma analogue is her height / eye-opening phase, not just the 18-way one-hot |
| History | GRU(256) + hand-rolled "time since last hit / kill / seed" scalars | K=8 frame window **summarised locally per entity into one token** before global attention (Wayformer) | no head-to-head | yes, as an alternative to the GRU |
| Objective vs metric | value loss MSE; reward hand-tuned | **the objective must be the metric.** Same MLP, same data, same epochs: L1 vs MSE moved aggregate one-step skill **-0.379 -> +0.392** and position +0.297 -> +0.393 | yes, measured on this laptop in 70 s (`scripts/smoke_mlp.py`) | yes -- and it is the most transferable single lesson here |
| Baseline discipline | success rate vs nothing | zero-init heads so the model **is** the baseline at step 0 and any gain is provably learned | structural | yes -- report every Gohma number against a stated control |

**What the WM did NOT buy, stated plainly.** No usable dynamics model: positive through H5 and negative at
H100+. No planner, no dream-based training, no transferable weights for the policy. The gain is entirely in
how we represent game state, and one measured lesson about objectives. That is a real win and it is a smaller
win than the time spent on it.

## Start states: begin with the fight already engaged

**The problem.** Three of the ten Gohma slots have her in `ceiling_wait` -- the pre-fight state, hanging at
height -300. Getting out of it means walking in and looking UP at the ceiling to trigger her descent. That is
a navigation-plus-camera skill with nothing in common with the fight it precedes, and the fleet measured the
policy farming it instead of fighting:

> `gohma-v5`: "**entrance slots farm `sling_ready` (~150 steps, 10 seeds) without starting the fight**"

`gohma-v6` suppressed the symptom with reward changes and a curriculum that put half of all episodes in slots
4-6. The slots stayed in the mix.

**The decision: drop them.** Learning to start the fight is a separate task from winning it, and it is being
asked of a policy that cannot yet win the fight.

| Slot | Gohma state | Distance | Verdict |
|---|---|---|---|
| 1 | `ceiling_wait` | 1,286 | **drop** -- entrance, pre-fight |
| 10 | `ceiling_wait` | 1,143 | **drop** -- a copy of slot 1 walked 165 units in |
| 2 | `ceiling_wait` | 376 | **drop** -- close, slingshot drawn, but still the pre-fight state |
| 3 | `stunned` | 188 | keep -- sword window, locked on |
| 4 | `floor_walk` | 190 | keep -- but see the caveat |
| 5 | `ceiling_eye_red` | 438 | keep -- opens on a vulnerable window |
| 6 | `floor_prepare` | 135 | keep |
| 7 | `stunned` | 38 | keep -- inside sword range |
| 8 | `stunned` | 448 | keep |
| 9 | `ceiling_move`, 3 larvae | -- | keep -- the only larvae-alive start |

Implementation is one flag, not a deletion: **`--start-slots 3,4,5,6,7,8,9`**. The savestates stay on S3 and in
the slot directory, so a later "start the fight" curriculum can add them back without remaking anything. Nothing
about slot 10 needs to be rebuilt.

**Caveat to carry:** slots 4 and 5 were measured dying **98% within ~50 s**, and slot 6 had the policy waiting
cocked ~1,400 steps without firing. Engaged does not mean survivable. Report `by_slot/` from update 1 and be
ready to reweight rather than discovering it at update 100.

## The actual blocker, measured 2026-09-26

Two hypotheses have been floating: the vulnerable window is too short for a 10 Hz policy to hit, or the policy
cannot aim. **The first is now dead.** Measured from the three human recordings in `runs/manual/` by decoding
Gohma's action function per frame:

| Recording | `ceiling_eye_red` windows | `floor_eye_red` windows | vulnerable share of the fight |
|---|---|---|---|
| Win (234 s) | 1, median **1.95 s** | 2, median 6.81 s | 4.3% |
| Death (387 s) | 3, median **3.48 s** | 1, 4.21 s | 6.4% |
| Nut fight (707 s) | 9, median **3.50 s** | 2, median 14.26 s | 11.1% |

The red eye is open for a **median 1.1-3.5 s, which is 11-35 policy steps at 10 Hz.** There is an order of
magnitude of margin. Timing and reaction speed are not the problem, and any proposal to raise the control rate
or add a "fire on red" macro is solving a problem we do not have.

**What the same numbers say about the policy.** She is vulnerable 4.3-11.1% of the time. `gohma-v8` fired
**7 seeds per episode** and landed **~0.2 in a red-eye state** -- that is **2.9%**, at or *below* what firing at
random would give. After ~126 updates the policy had not reached chance on the one action that wins the fight.

So the live hypothesis is the remaining one: **it cannot aim, and the aim information was never given a fair
run.** `gohma-v9` (obs v6: raw aim pitch/yaw, camera and body frames) was stopped after **7 updates** against
v8's 126, and `gohma-v10` (obs v7) has no recorded result at all. Two observation upgrades built specifically
to fix aiming, neither run long enough to be judged, and `AGENTS.md` sections 11 and 12 both say an architecture
comparison means training both arms to matched updates.

## STEP 1 RESULT (2026-09-26, local, zero compute): the premise was wrong

`gohma-v10` -- observation v7, the run `GOHMA_TASK.md` left as "(running)" with no result -- **did run**, to
update 56 across 19 ranks, and its per-episode logs were on this laptop the whole time in
`runs/gohma-r*/episodes.jsonl` (15,560 episodes, every field needed). No wandb required. The chance rate does
not have to be estimated from the human recordings either: `vulnerable_steps / length` gives it per episode,
matched to the fight the policy actually experienced.

**Retraction.** I wrote that the policy fires into the red eye at or below chance and that Gohma has 0 reliable
defeats. Both are wrong. They came from `GOHMA_TASK.md`'s `v7`/`v8` rows; the obs-v7 run is a different animal.

| | measured |
|---|---|
| red-eye seeds / seeds fired | **38.55%** against a **11.46%** chance rate = **3.36x chance** |
| trend over updates | 3.14x -> 3.14x -> 3.10x -> 3.71x -> 3.82x -> 3.56x, i.e. **rising** |
| seed stuns | 2,099 |
| defeats | **360** of 15,560 episodes; **3.29% on engaged slots**, rising 2.15% -> 4.27% |

So the aim hypothesis is answered, and answered in favour of obs v7. The aim information helped. Phase 1 as
originally written -- "does obs v7 help?" -- no longer needs running.

### The pre-fight slots are worse than suspected, and the cost is now exact

| | slots 1, 2, 10 | slots 3-9 |
|---|---|---|
| episodes | 4,631 (29.8%) | 10,929 |
| **share of env steps** | **43.7%** | 56.3% |
| Gohma HP dealt, total | **49** across 4,631 episodes | 22,500 |
| defeats | **0** | 360 |
| deaths | 0.2% | 64.5% |
| timeouts | **98.1%** | 32.2% |
| mean return | **-9.1** | +29.6 |

They consume **43.7% of all compute** -- more than their episode share, because they always run to the 300 s
cap -- and return nothing. Slot 10 fired 6,134 seeds across 1,506 episodes and landed **zero** in a red-eye
state. Removing them is a ~1.8x effective speedup before any other change, and it is the largest single win
available.

### The real blocker is SURVIVAL, and it has not moved at all

| engaged slots, by update | u01-10 | u21-30 | u41-50 | u51-60 |
|---|---|---|---|---|
| death rate | 64.9% | 64.2% | 63.8% | 64.6% |
| Gohma HP dealt | 1.92/10 | 2.10/10 | 2.14/10 | 2.13/10 |

Aim improves, damage plateaus at ~2 of her 10 HP, and the death rate is **flat across all 56 updates**.
Died episodes took 7.46 of her attacks and lost 47.3 of 48 HP; survived episodes took 2.04. The policy is not
losing because it cannot shoot her -- it is losing because it dies after two hearts of progress.

**How the 360 wins were actually won:** mean 55 s, `nut_stuns` 2.53, `seed_stuns` **0.05**, `stun_hits` 5.23,
and **324 of 360 started from Gohma at 5 HP** (9 from full HP). So the policy's working kill is *Deku Nut stun
plus sword, from half health* -- not the slingshot skill the reward was built around. Slot 7 (already stunned,
38 units away) supplies 292 of the 360.

### The defensive channel is UNMEASURED, which is its own finding

`shield_steps` is 0.0 in every episode and `blocks` is 0.00 -- but neither is evidence the policy never
shields. `shield_steps` requires a live target **within 150 units** (`shield_radius`, calibrated for a Deku
Baba lunge) and Gohma starts 188-448 away and spends much of the fight on the ceiling. `boss_blocks`
(0.10/episode against 5.57 attacks) requires a whole attack cycle that was *shielded AND took zero damage AND
was inside `g_block_range`* -- it counts defensive **successes**, not attempts. **No logged metric answers
"does the policy ever raise the shield against Gohma".** With survival as the blocker, that is the gap to close
first, and it is a local measurement, not a training run.

## STEP 2 RESULT: the shield action barely registers

Measured locally with `scripts/measure_shield.py` on `gohma-r0_u00050` (obs v7, update 50), 4 Dolphins,
25,395 steps over 12 episodes on slots 3/4/5/7/9:

| | |
|---|---|
| shield action CHOSEN | **6.57%** of steps (1,669) -- against a 10.0% uniform-random baseline |
| `SF1_SHIELD` actually set | **1.15%** of steps (293) |
| **shield presses that registered** | **17.44%** -- 1,378 of 1,669 produced no shield bit |
| shielded while she is ATTACKING | 22 of 949 attack steps = **2.32%** |
| steps within 150 units of her | 5.62% -- which is why `shield_steps` reads 0.0 |
| distance when shielding | median **399**, p10 187, p90 824 |

So it is neither "never explores shield" nor "shields and dies anyway". **The action mostly does not work.**
The policy picks a shield action 1,669 times and Link raises his shield 291 times. 8 of the 9 shield actions
carry a stick direction (`-`, E, NE, N, NW, W, SW, S, SE and `+Shield` = index 4 of the button list), and Link
does not raise his shield while walking, so most of the shield action space is expected to be a no-op -- 1/9
would be 11.1% against the 17.4% measured.

And `shield_steps` is confirmed structurally dead for a boss: Link is within its 150-unit `shield_radius` for
5.62% of steps and shields at a median distance of 399. It cannot report defensive progress here no matter what
the policy does.

## Next steps, in order (revised after step 1)

Everything local. No pods; the fleet is not the bottleneck at 10 Dolphins per machine and remote setup cost
more than the experiments did.

1. **Drop the pre-fight slots.** `--start-slots 3,4,5,6,7,8,9`, plus the rebuilt 1, 2, 10. Recovers 43.7% of
   compute for free. Done as part of phase 0c.
2. **Measure whether the policy uses the shield at all**, from a local rollout of `gohma-r*_u00050` with 10
   Dolphins: the raw action distribution over the 90 actions, and the `SF1_SHIELD` bit per step, logged
   directly rather than through a Baba-scoped counter. This is a ~10-minute local run and it decides whether
   survival is an exploration problem or a reward-measurement problem.
3. **Fix the defensive metrics for a boss** -- `shield_radius` 150 is a Baba number. Whatever step 2 shows,
   `shield_steps` as currently defined cannot report progress on a boss that attacks from 200+ units.
4. **Then** address survival with ONE change, measured against the u56 baseline (3.29% defeat, 64.5% death).
5. **Phase 2 (WM representation port) is deprioritised.** The policy's problem is not representation --
   it aims at 3.4x chance and is still improving. Revisit only if survival work stalls.

## Open questions worth one measurement each

* Does the policy's **aim** correlate with the eye at all? The recordings give the human's distribution of
  `focus.rot.x/.y` while firing a hit versus a miss; the policy's is in its rollouts. One histogram.
* Is `seed_stun` (+15, the key skill) reachable by exploration, or is it so rare that only the *ratio* metric
  can drive learning? 0.04 stuns/episode over 9,124 episodes is ~365 events total -- thin for PPO credit.
* Slots 4 and 5 die 98% within 50 s. Is that a survivable start with a better policy, or an unfair one?

---

# Execution plan (2026-09-26)

## Phase 0, in parallel

| # | Work | Where | Spend | Gate |
|---|---|---|---|---|
| 0a | **DONE.** Back up the savestates to S3. They existed on this Mac ONLY -- `s3://zelda-rl/zelda/combat/gohma/` was empty, so the entire Gohma task rested on one unbacked-up directory. 10 files, ~390 MB, now at `s3://zelda-rl/zelda/combat/gohma/states/`, verified size-identical | local | $0 | **gated 0c; passed** |
| 0b | **Step 1: the zero-compute measurement.** `red_eye_seeds / seeds_fired` per slot for `gohma-v7` and `v8` from wandb history, against the 4-11% chance band | local | $0 | **decides whether phase 1 is the right experiment at all** |
| 0c | **Replace the three `ceiling_wait` slots with engaged states derived from slot 9** (Gohma `ceiling_move`, 3 larvae alive) | local, needs Dolphin + `python-dbg` | $0 | blocks phase 1 |
| 0d | **Provision 5 CPU pods** (16 vCPU) and bootstrap them so off-machine experiments do not wait on setup | RunPod | ~$2.5/hr | blocks phase 1 |

**Slot 10 is a byte-identical copy of slot 1** (md5 `0c627526…` on both). `GOHMA_TASK.md` says slot 10 was
remade with `make_walk_state.py` walking 165 units in; whatever is on disk now is not that. So slots 1, 2 and
10 are two distinct pre-fight states, one of them duplicated -- 30% of the start distribution spent on a state
the policy should never see.

**0c in detail.** Source is slot 9, the only start with larvae alive. Three variants rather than three copies,
because four identical starts would cut start diversity from 10 states to 7.

**Known cost of this choice, accepted deliberately:** slot 9 is a LATE-fight state -- Gohma at 5 of 10 HP with
a larva batch already out. Deriving 1, 2 and 10 from it makes 4 of 10 starts late-fight-with-larvae, so the
policy will see the opening of the fight only from slots 5 and 9's own phase and never from full HP. That is a
narrower curriculum than before, traded for never practising the pre-fight state. If the defeat rate rises but
only from 5 HP, this is the first thing to revisit.

* `make_walk_state.py --source-slot 9 --dest-slot {1,2,10}` with different `--path` values, so Link begins each
  at a different position and range while the fight is already running.
* Every write must pass, and be reported: Link alive, out of cutscene (`SF1_CUTSCENE`), moved >= `--min-move`,
  **stick input check passes**, and -- the check the memory note demands -- **Gohma is present as actor id
  `0x0028` category 9** with the expected larva count, verified with `find_gohma` / `gohma_state`, not assumed.
* The originals are backed up by the script AND by 0a before anything is overwritten.

## Phase 1: finish the abandoned comparison

Only after 0a-0d, and only if 0b has not invalidated the premise.

Two arms, **matched update count**, engaged slots only, nothing else changed:

* **arm A** = obs v7 (`gohma-v10`'s configuration: all larvae, actor velocities, facing, shape, seed impacts)
* **arm B** = obs v4/v5 baseline reproducing `gohma-v8`, which is the only run that has a number worth beating
  (return -5.6 over ~126 updates)

`--start-slots 3,4,5,6,7,8,9` plus the rebuilt 1, 2, 10 -- i.e. all ten slots, none of them pre-fight.
Primary metric `red_eye_seeds / seeds_fired` against chance; secondary defeat rate. Per `AGENTS.md` 11 and 12
the arms are compared at the same update count and nothing is read from an unmatched pair.

**The exact command goes to the user before it runs** (`AGENTS.md` 26).

## Phase 2: port the WM representation, one change at a time

Against arm B as the measured baseline, in this order, each its own run:

1. attention over typed entity tokens instead of the masked mean+max pool
2. the third entity class (static environment as a keys-only pool)
3. nav-grid cell embeddings for position -- last, because it is the one with no measurement behind it

## What this plan deliberately does not do

* No reward changes (30 terms already; four runs of patches aimed at farming that 0c removes at the source).
* No control-rate change and no "fire on red" macro -- the window is 11-35 policy steps, so that is not the bug.
* No new observation version until v7 has been judged at matched updates.
