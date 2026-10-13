# Focus: finish the Queen Gohma fight

**Decision (2026-09-26).** The world model is parked. It is not abandoned as knowledge -- it changed how we
frame game state, and that part is kept and written down below -- but it is not on the critical path to a
defeated Gohma, and it has been the whole path for weeks. One goal now: **a policy that defeats Queen Gohma
from combat-engaged start states.**

`docs/WM_EXPERIMENT_REGIME.md` is the closing record of the world-model work; its checkpoints are in `runs/`
and nothing there needs to be rerun to proceed here.

---

# WHERE WE LEFT OFF -- 2026-10-13, fleet killed, $0.00/hr

**Read this first. It is the state of the task, not a log.**

## The one-line state

**v15 is the best policy we have** (`checkpoints/kept/gohma-kl-v15-r4_latest.pt`, md5 `397e27b2beb6`,
optimizer intact). Boss damage **3.275 of 10** on the mix-invariant metric. Everything after it -- v16, v17,
v18 -- is a regression, and all three were my changes. Nothing is running.

## What actually happened, and the headline lesson

**The project had been improving steadily for eight versions and we could not see it, because the metric was
wrong.** `combat/success_rate` -- boss *defeated* -- is a binary at 1-10% with almost no power to show a
trend. On **boss damage**, the HP actually taken off her out of 10, the same runs read:

| version | `g_approach` | boss damage | slope/upd |
|---|---|---|---|
| v10 | -- | 0.38 -> 1.21 | **+0.003\*** |
| v11 | -- | 1.72 -> 2.10 | **+0.015\*** |
| v12 | -- | 2.06 -> 2.22 | **+0.008\*** |
| v13 | -- | 2.41 -> 2.75 | **+0.007\*** |
| **v14** | **0.0 (absent)** | **2.86 -> 3.22** | **+0.012\*** |
| v15 | 0.5 | 3.17 -> 3.32 | +0.004\* |
| v16 | 1.5 | 3.12 -> 2.85 | **-0.017\*** |
| v17 | 0.5 + deficit sampling | 3.19 -> 2.86 | **-0.022\*** |
| v18 | 0.0 (reverted), from v15 | 3.40 -> 2.94 | -0.021 (n.s. once the start transient is dropped) |

`*` = |t| > 2. **0.38 -> 3.32 HP is an 8.7x improvement and it was reported as six consecutive plateaus.**

**Two things this cost.** `g_approach` was invented to fix a plateau that did not exist, and it measurably
hurt. And three versions of compute went into chasing a flat line that was only flat on the wrong axis.

## The honest open question, which is NOT "which reward next"

Read the damage column as a curve rather than a league table: **0.38, 1.72, 2.06, 2.41, 2.86, 3.17** --
each version starts further along and climbs more slowly. **Damage saturates at 3.2-3.4 of 10 regardless of
reward.** The per-run slope also correlates with the *starting* level, which means the `g_approach`
dose-response is confounded too: runs starting near 2.6 climb, runs starting near 3.3 do not. I reported that
dose-response to you as clean and it is not.

So the live question is **why does damage saturate at ~1/3 of her health**, and the answer is probably not
another reward term. Eight have been tried. Candidate next moves, in the order I would take them:

1. **Watch episodes at the ceiling.** Load v15 and watch ten episodes on slots 3 and 5 where it takes 4+ HP.
   What stops hit 4? Out of nuts? Out of seeds? Dies to the third larva wave? This is a 20-minute
   qualitative question that no amount of reward tuning answers, and it has never been done at this level.
2. **Check the ammunition.** `g_nut_scarcity 20.0` and `g_seed_scarcity 30.0` exist, which means scarcity was
   anticipated. A stun needs a seed or a nut; ten stuns need ten. If the policy runs dry at hit 3-4 the
   ceiling is a RESOURCE limit and no reward fixes it.
3. **Episode length.** Episodes end at 300 s and mean 40-100 s by death. Ten stun-and-hit cycles may simply
   not fit; a human win took 154 s.
4. Only then, reward or architecture.

## The fast-signal fork we stopped at

`--reward damage` is **built, committed and untested** -- `GOHMA_DAMAGE` in `zelda_rl/combat.py`, 39 shaping
terms zeroed, keeping only `boss_damage * dmg * (1 + 2.0 * fraction_of_her_HP_gone)` (4.80 for her first HP,
12.00 for her last), `boss_defeat` 60, timeout, death x `--death-scale`, and `time_penalty` 0.001. Verified
against the dataclass fields so a typo cannot silently do nothing, and asserted to leave no shaping behind.

**The decision not made:** what to initialise it from. "From scratch" and "signal in minutes" conflict --
v10 from scratch *with* shaping took 399 updates to reach 1.21 damage, so random init plus outcome-only
reward is the hardest configuration attempted here and will take hours. The four options as they stood:

| option | starting damage | signal | confound |
|---|---|---|---|
| **from `v13_r12`** | 2.75 | ~10 updates / 30 min | none -- v13 predates every change I made |
| random init | 0 | 1-3 h | none, but slowest |
| split fleet 10/9 | both | half throughput each | none |
| from `v15-r4` | 3.275 | fastest | v15's weights were shaped by `g_approach` |

## What to run, verbatim, when you come back

```sh
python -m fleet create    --experiment gohma-kl --count 20 --vcpu 16 --confirm
python -m fleet bootstrap --experiment gohma-kl                 # S3 path is AUTOMATIC now, ~2 min/pod
python -m fleet ckpt      --experiment gohma-kl --path checkpoints/inbox/v13_r12_latest.pt
python -m fleet preflight --experiment gohma-kl
python -m fleet launch    --experiment gohma-kl --script train_combat.py --run-tag v19 --num-envs 8 \
  --wandb-project zelda-rl-gohma-ppo --train-args "--profile gohma --obs-version 7 --reward damage \
  --episode-seconds 300 --start-slots 3,4,5,6,7,8 --slot-weights 4=1.125,6=1.125,7=2.25 \
  --init-from checkpoints/inbox/v13_r12_latest.pt --death-scale 0 --fleet-aggregator 4 --total-updates 200"
```

No `--kl-anchor`: anchoring to v13 while initialising from v13 does nothing, and with a new reward the anchor
would fight it. Then:

```sh
python -m fleet ppo       --experiment gohma-kl --last 8      # dmgB / dmgH lead; succ is last for a reason
python scripts/train_compare.py --compare gohma-kl-v19 --trend          # slope +/- SE, RISING/flat/FALLING
python scripts/train_compare.py --compare gohma-kl-v19 --exact gohma-kl # exact per-slot pool from the pods
```

**The number to beat is v14's +0.012 HP/update**, judged as a slope with a standard error. It refuses to fit
under 5 updates, which is deliberate.

## The tooling that now exists, and why each piece is there

| command / file | what it is for |
|---|---|
| `fleet ppo --experiment E` | fleet-wide PPO curve. `dmgB`/`dmgH` lead, `succ` last. Parses the log BY HEADER NAME, after a column shift silently relabelled `death` as `hard` |
| `scripts/train_compare.py --trend` | **the answer to "is it improving"**: slope +/- SE on boss damage, RISING/flat/FALLING at \|t\|>2 |
| `scripts/train_compare.py --exact E` | exact per-episode per-slot counts from the live pods. 3,142 episodes in 14 updates vs the eval fleet's 40 |
| `scripts/eval_compare.py` | the HELD-OUT question, with Fisher p and Wilson intervals. Rejects round-based rows by content, dedups on `(checkpoint, rank, ts)`, honours `.withdrawn` / `.run_label` markers |
| `scripts/audit_gohma_slots.py` | reads Gohma's actor state per slot out of the emulator. Exits non-zero on a pre-fight slot. **All 10 slots are engaged; no `ceiling_wait` remains** |
| `fleet ckpt --path ...` | an arbitrary `--init-from` / `--kl-anchor` checkpoint onto every pod by presigned S3 GET, verified by `stat`-ing what the trainer will open. 8.2 MB x 19 pods in 15 s |
| `zelda_rl/combat.band_metrics()` | `boss_damage_balanced`, `boss_damage_hard`, `success_balanced`, `success_hard`, `slots_nonzero`, `by_band/*`. All MIX-INVARIANT |
| `zelda_rl/combat.deficit_weights()` | adaptive start-slot sampling. **Negative result** (v17), kept with its 25 tests |
| `tests/test_band_metrics.py` | asserts mix-invariance on v11's real skill profile |
| `tests/test_deficit_weights.py` | exact-ratio, the anti-forgetting floor, scale-freeness, degenerate inputs |
| `tests/test_gohma_approach.py` | the approach term's mechanics, and asserts it ships at **0.0** with the reason |

`fleet bootstrap` now **takes the S3 path automatically** and refuses with exit 2 if S3 is stale -- that one
cost ~$4 and 25 minutes of a 19-pod fleet because the flag was forgettable.

## Corrections made this session that change earlier numbers

1. **The start-slot distances were 3D.** Six of ten overstated, because Gohma sits ~320-340 units up on the
   ceiling. Slot 2 is 140u not 368u (mid -> **near**), slot 5 is 347u not 472u (far -> **mid**). Rebanded, the
   distance wall is **14x** near-to-far, not 7.5x, with the far band at **1 win in 208 episodes**. The table
   now lives once in `zelda_rl/combat.GOHMA_SLOT_DIST`, measured by `audit_gohma_slots.py`; four copies
   existed and three were wrong.
2. **The held-out checkpoint selector was noise.** It ranked on 16-episode subsets at a 1.68% pooled rate,
   where 6.25% is a 20.9% draw from the null. Selected 2/32 vs others 5/384, **p=0.094**. It needs ~256
   episodes per checkpoint, 16x what it had. And `.best.json` is read by **nothing**.
3. **v11's trend was one slot.** Its climb was slot 7 (38u) alone, 7.32% -> 14.00%, with nine slots at
   exactly 0.00% for 37 updates. v12 was launched because of that number.
4. **The v15-vs-v14 hold-out p was 0.122 against a 559-episode SUBSET** of an 880-episode baseline, in a pool
   that also held 2,831 retracted episodes and counted duplicates. Corrected: **p=0.087**.

Full detail with evidence classes: `PROVENANCE.md` 3.2. Methodology rules: `AGENTS.md` 29-34, where 34 is the
metric failure above.

## Checkpoints kept

| file | md5 | what |
|---|---|---|
| `checkpoints/kept/gohma-kl-v15-r4_latest.pt` | `397e27b2beb6` | **THE BEST POLICY.** damage 3.275, optimizer intact |
| `checkpoints/inbox/v13_r12_latest.pt` | `49cf19963f4f` | v13 u70, the last checkpoint predating any of this session's changes |
| `checkpoints/kept/gohma-kl_r3_u00040.pt` | `8f2640cb3e87` | v14 u40 as published, weights only |
| `checkpoints/kept/gohma-kl-v16-r4_latest.pt` | `0e11fcccb6f5` | v16 u20. Negative result |
| `checkpoints/kept/gohma-kl-v18-r4_latest.pt` | `7bd6b58ed9cb` | v18 u10. Negative result |

---

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

## STEP 3: the action space could not express the defence, and the defence is SPACING

**Two capability holes, found by auditing the button table rather than by training.** The only combination
containing R was `("R",)` alone, so **"shield while Z-targeting" was literally inexpressible** -- the policy
could lock on or shield, never both -- and that is exactly what the `block` reward pays for. The Deku Nut had
the same hole, on the item every measured win depends on (2.53 nut stuns per defeat against 0.05 seed stuns).

| | solo | with Z |
|---|---|---|
| sword (B) | yes | yes |
| A | yes | yes |
| slingshot (C-down) | yes | yes |
| **shield (R)** | yes | **was MISSING -- added as `("L","R")`** |
| **Deku Nut (C-left)** | yes | **was MISSING -- added as `("L","CL")`** |

Both appended last, so actions 0-89 keep their meaning and existing checkpoints transfer unchanged; 90-98 are
`+Z+Shield` and 99-107 `+Z+Nut`, for 108 total. `send_action` already handles them: the C-tap path keys on
`"CL" in held` and L/R come from the held tuple.

**But the shield is probably not the answer.** In 17,182 frames of recorded human play across three fights the
`SF1_SHIELD` bit is set **0.00% of the time**, while Z-target fires 36-63% of frames. The bit is real -- in the
live rollout it co-occurred with a chosen shield action 291 times of 293, 99.3% -- so this is not a decoding
bug. A competent human beats Gohma without ever raising his shield.

What he does instead, measured during her attack states only:

| | median distance to her | attack frames inside 150u | hearts lost |
|---|---|---|---|
| **the WIN** | **311** | 8.7% | 0.5 |
| **the DEATH** | **131** | 61.8% | 3.0, died |

Same player, same boss, and the whole difference is **where he stands while she is attacking**. Survival here
is SPACING, not blocking -- which means a reward for "locked on and shielding while she attacks" pays for
standing inside the range the winning run stayed out of. `g_guard` is implemented and reachable, and it is
**not validated by the only competent play we have**; the spacing term the data supports has not been written.

## The defence, implemented: SPACING, not shielding

`--reward simple` carries `g_spacing` = **0.02 per step spent in the BAND [150, 600] units while Gohma is in an
attack state, capped at 1.5**, and `g_guard` is 0.0. The band is the measurement, not a guess: over her attack
frames the WIN held a median 311 units (p10 162, p90 533) and was inside 150 for 8.7% of them, the DEATH held
131 and was inside for 61.8% -- a 7x separation from the same player on the same boss.

It is a BAND rather than a floor because a floor paid a do-nothing policy exactly as much as it paid the human
win: at 0.05/step capped at 4.0 the cap saturated for the win, the death AND a smoke-run episode that banked
the whole cap over 3,000 attack steps with zero boss damage (slot 6 starts at `floor_eye_red` 357 units away,
so she is "attacking" from t=0 and standing still collected the maximum). `far_attack_steps` is now a separate
unpaid bucket, and the validation that [150, 600] is right is that the human's `far_attack_steps` is 0 in all
three recordings -- nothing about competent play falls outside it.

It pairs with `approach_stunned` / `sword_ready_stunned`, which pay for CLOSING while she is down, so the
learned behaviour is phase-dependent spacing rather than distance for its own sake. The cap is reached only by
spacing correctly through essentially every attack step, and at 4.0 against `boss_defeat`'s 60 it cannot
dominate.

Replayed against the recordings, it separates good play from bad:

| recording | safe attack frames | too close | `spacing` paid | outcome |
|---|---|---|---|---|
| **WIN** | 137 (**90.7%**) | 14 | **+1.50** (cap) | 0.5 hearts |
| **DEATH** | 45 (**40.2%**) | 67 | **+0.90** | 3 hearts, died |
| nut fight | 154 (73%) | 57 | +1.50 | 2.5 hearts |

`close_attack_steps` is the failure counter to watch: it should fall as the defeat rate rises.

## The verified slot set (decoded offline from the savestate files, 2026-09-26)

Not trusted from the agent that wrote them, and not from `GOHMA_TASK.md`, whose table no longer matches the
files on disk for slots 4-8. Decoded with `read_combat_state` over the decompressed savestate RAM, with the
larva/debris split done by category and params (id `0x0028` is shared -- trap 2):

| slot | Gohma state | HP | larvae | Link HP | distance |
|---|---|---|---|---|---|
| 1 | `ceiling_move` | 5 | 3 | 48 | 473 |
| 2 | `ceiling_move` | 5 | 3 | 40 | 368 |
| 3 | `stunned` | 10 | 0 | 48 | 188 |
| 4 | `ceiling_spawn` | 8 | 0 | 40 | 541 |
| 5 | `ceiling_eye_red` | 10 | 0 | 40 | 472 |
| 6 | `floor_eye_red` | 6 | 0 | 48 | 357 |
| 7 | `stunned` | 5 | 0 | 40 | 38 |
| 8 | `stunned` | 10 | 0 | 40 | 448 |
| 9 | `ceiling_move` | 5 | 3 | 48 | 394 |
| 10 | `ceiling_move` | 5 | 3 | 48 | 490 |

**No `ceiling_wait` remains in the set.** 1, 2 and 10 are slot-9-like variants at three distinct ranges rather
than copies (distinct md5s). Curriculum weights, for 50% slot 7 / 25% the slot-9 family / 5% each of 3,4,5,6,8:

```
--start-slots 1,2,3,4,5,6,7,8,9,10 --slot-weights 1=1.25,2=1.25,7=10,9=1.25,10=1.25
```

## What a LOCAL run can and cannot settle

This Mac has 8 logical / 4 performance cores and sustains **~85-111 env steps/s at 4 Dolphins** (it stalls
occasionally; `train_combat` retries 3x and restarts the bad env, so it survives). One update at 4 envs x 3,000
steps = 12,000 env steps ~ 2 minutes.

`gohma-v10` moved its defeat rate 2.15% -> 4.27% over 56 updates at 8 envs per rank across 19 ranks -- roughly
25 million env steps. An hour on this laptop is ~180,000. **A local run cannot reproduce that learning curve**,
and claiming otherwise would repeat the mistake of reading an unmatched comparison (`AGENTS.md` 11, 12).

What it CAN do, and what it is for: validate end to end that the new action space and reward work -- that
`+Z+Shield` and `+Z+Nut` are chosen and register, that `spacing` / `close_attack_steps` / `*_stuns_cashed`
populate, that the rebuilt slots produce engaged episodes with real boss damage. That is the gate before
deciding whether the comparison is worth fleet time at all.

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

---

# The run: `gohma-band` / `v11` (launched 2026-09-26)

20 CPU pods x 8 envs, **$10.32/hr**, ~4,110 env steps/s fleet-wide. 150 updates x 24,000 x 20 = **72M env
steps ~ 4.9 h ~ $50**. wandb group `gohma-band-v11` in `zelda-rl-gohma-ppo`, FedAvg with rank 4 as aggregator.

```
--profile gohma --obs-version 7 --reward simple --episode-seconds 300
--start-slots 1,2,3,4,5,6,7,8,9,10 --slot-weights 1=1.25,2=1.25,7=10,9=1.25,10=1.25
--init-from checkpoints/inbox/gohma-r13_u00050.pt --transfer-action-margin 0.3
--fleet-aggregator 4 --total-updates 150          (+ --num-envs 8)
```

Resumed from `gohma-r13_u00050.pt`, the best of v10's 18 rank checkpoints by late-update defeat rate on engaged
slots (8.43%, against 5.10% for the r0 checkpoint first used by mistake). Widened 90 -> 108 actions
function-preservingly at load.

## What to watch, in priority order

1. **`close_attack_steps` share of attack steps.** 40.5% at launch; the human win is 9.3%. This is the
   mechanism the run is testing and it should move within 10-20 updates.
2. **`nut_stuns_cashed / nut_stuns`.** 15% at launch. The conversion gate withholds the rest, so this rising
   means stuns are being turned into damage rather than farmed.
3. **Defeat rate on engaged slots, against v10's 4.27%.** Needs ~50 updates to read.
4. **`boss_hp_start` of the defeats.** THE TRAP: slot 7 is 50% of episodes and starts with Gohma stunned at
   5 HP, 38 units away -- a rank-19 episode won it in **4.3 s** for +289.4. A rising defeat rate made
   entirely of 5-HP finishes is not progress on the fight. 324 of v10's 360 wins already started at 5 HP.
   Watch the share of defeats with `boss_hp_start` 10.
5. **`far_attack_steps`.** 0 in human play and near-0 at launch. If it climbs, the spacing term is teaching
   retreat rather than positioning, and that is also the trigger to build the boss-room wall map.

## Fleet lessons from this deploy (new, not in RUNPOD.md)

* **`fleet launch` does not check that `--fleet-aggregator N` names a rank that exists.** Creation came up
  9/20 then 18/20 with ranks 1 and 4 missing; launching then would have pointed FedAvg at a nonexistent
  aggregator and every follower would have silently fallen back to its own weights after the timeout. This
  belongs in preflight as a hard refusal.
* **One unreachable pod anywhere in the state file blocks bootstrapping ALL others.** `cmd_bootstrap` calls
  `setup_seed_access` over every pod in the state, not just the selected ranks, so a single dud raised before
  any pod was seeded and the first `--via-seeds` run bootstrapped zero. Terminating and recreating that pod
  fixed it.
* **`fleet audit` is collector-shaped.** It greps tmux session `zrl` and counts episodes under
  `datasets/transitions/v5`, so a TRAINING fleet always shows "episodes on pod disks: 0" and its liveness
  collapses to `pgrep dolphin-emu` -- which reported 4 pods idle that were mid-Dolphin-boot, the exact §22
  trap. For a trainer, check tmux session `train` plus the trainer process.
* **The Linux binary is `dolphin-emu`, not `Dolphin`.** A macOS-shaped `grep "[D]olphin"` returns 0 on a
  perfectly healthy pod.

---

# v11 at 20 updates: the arc is good, and two things it revealed (2026-09-26)

**Verdict: training well, right speed, right curriculum. It is a waiting game now.** 3,023 episodes from 5 of
20 ranks, engaged slots:

| updates | defeat | Gohma HP taken | died | return |
|---|---|---|---|---|
| 1-5 | 3.54% | 1.89 | 69% | +2.6 |
| 6-10 | 3.76% | 1.90 | 68% | +4.4 |
| 11-15 | 5.47% | 2.05 | 66% | +11.3 |
| **16-21** | **6.95%** | **2.16** | **65%** | **+15.7** |

Defeat rate nearly doubled and is already past v10's 4.27% ceiling; return went +2.6 to +15.7. **The death rate
is moving for the first time** -- it sat at 64.5% for all 56 updates of v10 and is now 69% -> 65%.

The reward is shifting toward the outcome as intended: `boss_defeat` entered the top-9 components in the later
half (+3.72/ep, previously outside it) and `boss_defeat_speed_bonus` went +3.69 -> +6.22, so return is
increasingly earned by winning faster rather than by shaping. `death` remains the largest single term at -19.7.

## What is NOT working yet

* **The spacing term has not changed the behaviour it targets.** `close_attack_steps` ran 20.7 / 22.4 / 23.3 /
  19.9 % across the four bands -- noise around 21%, no trend, against the human win's 9.3%. The death rate is
  falling anyway, so something else is driving that. The one change made for survival is not the thing
  producing the survival gain, and that needs saying plainly before anyone credits it.
* **Stun conversion is flat at ~19%** (218 of 1,133 nut stuns cashed, from 20% at the start). The gate is
  correctly withholding the other 80%, but the policy is not learning to cash in -- it stuns and walks away.
* **The rebuilt slots 1/2/10 underperform.** 41% close-during-attack, 79-85% death, 0.4% defeat, return -12 to
  -18. They place Link 473/368/490 units out against slot 9's 394, and he dies at range without closing. They
  are 25% of episodes for almost no return -- better than the pre-fight slots' 0%, but worth rebuilding nearer
  or reweighting down.

## The curriculum is a FINISHER curriculum, and the numbers say so

Gohma has **10 HP**, one per sword hit while stunned. The verified start distribution:

| Gohma HP at start | slots | share of episodes |
|---|---|---|
| **10 (full)** | 3, 5, 8 | **15%** |
| 8 | 4 | 5% |
| 6 | 6 | 5% |
| **5 (half dead)** | 1, 2, 7, 9, 10 | **75%** |

75% of episodes begin with her already at half health, and slot 7 alone -- stunned, 5 HP, 38 units away -- is
50%. **1 full-HP win in 3,023 episodes.** So the doubling defeat rate measures a finisher on a half-dead boss,
which is a real skill and is not the task. Nothing to change while the arc is still climbing; the number to
watch is the share of defeats with `boss_hp_start` 10, and the moment the finisher saturates, the weights
should shift toward slots 3/5/8.

## Note on the slot rebuild, since the naming confuses

The three `ceiling_wait` states were replaced in place, not dropped: `NACE01.s01`, `s02` and `s10` now hold
engaged states derived from slot 9 (`ceiling_move`, 5 HP, 3 larvae), verified by decoding the files. No slot in
the set holds `ceiling_wait` any more. "Slot-9-derived slots 1/2/10" means those rebuilt files.

---

# v13: the ladder worked, and it exposed a DISTANCE wall (2026-09-27)

Three curricula, same reward, same policy lineage, each resumed from the last. 5 of 20 ranks sampled.

| | v11 (finisher-heavy) | v12 (inverted) | **v13 (graded ladder)** |
|---|---|---|---|
| start mix | 75% at 5 HP, 15% full | 20% at 5 HP, 60% full | **30% / 30% / 40% by HP tier** |
| return trend | +0.700/update | +0.098/update | +0.130/update |
| defeat rate, peak | 6.95% | 2.01% | **7.26%** (updates 41-50) |
| **full-HP defeat rate** | **0.03%** (1/3,023) | 0.20% (4/2,008) | **0.64%** (21/3,258) |
| death rate | 70.8 -> 62.2% | 57.8 -> 61.7% (rising) | 59.0 -> 56.1% |

**The ladder is the right shape.** v12's cliff -- jumping straight to 60% full-HP starts -- stalled at +0.098/update
because a full kill needs about four stun-and-hit cycles chained without dying and the policy could do one.
v13's 30/30/40 by HP tier keeps the finisher rung sharp while the 6-8 HP rungs convert, and full-HP kills went
**21x v11's rate**. Nothing else in the project has moved that number.

**But v13 plateaued at update ~50 of 73.** Peak was updates 41-50 (7.26%, +27.1 return) and the last three
blocks drift down (6.65, 6.29, 6.08%). Death is stuck at ~56% for 40 updates; Gohma HP taken stuck at ~2.7 of 10.

## The wall is DISTANCE, not HP

Last 15 updates, per slot:

| slot | Gohma HP | start distance | win rate | died |
|---|---|---|---|---|
| 7 | 5, stunned | **38** | **19.0%** | 62% |
| 6 | 6 | 357 | 3.4% | **29%** |
| 5 | 10 | 472 | 1.8% | 68% |
| 4 | 8 | 541 | 0.9% | **80%** |
| 3 | 10 | 188 | 0.5% | 71% |
| **8** | **10, stunned** | **448** | **0.0%** | **20%** |

Two different failures, and only one of them is what the HP ladder grades:

* Slots 3, 4, 5, 7 **die** (62-80%): they engage and lose.
* **Slot 8 neither dies nor wins** -- 20% death, **0 wins in 225 episodes**. It survives by not fighting. Slot 8
  starts with Gohma ALREADY STUNNED, which is a free damage window, and the policy never crosses 448 units to
  cash it in. Slot 6 at 357 units converts at 3.4% with a 29% death rate; slot 8 at 448 converts at zero.

So the curriculum grades HP and has **no rung between 38 units and 448 units**. The same signature killed the
rebuilt slots 1/2/10 (473/368/490 units, 0.4% defeat, dropped from the mix): whenever Link starts beyond ~400
units he does not close, and a stun he cannot reach is worth nothing.

**The next curriculum should grade by DISTANCE, not by HP** -- intermediate-range starts built from slot 8 by
walking Link partway in, exactly how slots 1/2/10 were made (`scripts/make_walk_state.py`). That attacks the
measured wall rather than the assumed one.

## Kept

`runs/gohma_v13/v13_r12_latest.pt` -- rank 12, the best of the five sampled by late-update defeat rate (7.36%),
verified loadable with optimizer state. Plus `v13_r{0,1,3,7,12}.jsonl`, the per-episode records behind every
number above. The fleet was terminated after this.
