# Plan: fix credit assignment, then capacity, then init — incrementally, with a gate at each step

**Written 2026-10-15, after eight Gohma versions of reward tuning produced no kill.** The thing none of those
versions changed is the one this plan changes first.

---

## 1. The three failures this plan addresses

### 1.1 The credit horizon was never re-derived for a task 10× longer — THE BIG ONE

`gamma`, `gae_lambda` and `chunk` are **identical in all 19 exported RL runs**: `(0.995, 0.95, 60)`. They were
tuned when a Deku Baba died in about 15 seconds. The only thing that changed for the boss is
`episode_seconds` 180 → 300.

At 10 Hz (`vi_per_step 6`) those numbers mean:

| setting | what it bounds | horizon | Deku (15 s kill) | Gohma (45–154 s kill) |
|---|---|---|---|---|
| `gamma 0.995` | the value function | **20 s** | fits | **does not fit** |
| `gae_lambda 0.95` | **the advantage** — what the policy gradient actually sees | **1.8 s** | one swing sequence | **one stun→hit cycle at best** |
| `chunk 60` | BPTT through the GRU | 6 s | fine | cannot span cycles |

GAE weights decay as `(gamma·lambda)^k = 0.94525^k`, so **67.6% of the advantage weight is inside 2 seconds**
and 94% inside 5. And the terminal reward:

- `boss_defeat` **+60**, from the start of a 154 s fight → worth **0.027**. **0.04% of face value.**
- Deku's `w_defeat` **+5.0**, from 15 s out → worth **2.36**. **47% of face value.**

**The policy has never been able to see the kill.** It optimises the next stun-and-hit because that is the
only thing inside its credit horizon — which is exactly the measured outcome: **boss damage saturates at
3.2–3.4 of 10 across eight versions and five reward configurations**, roughly the number of hits reachable
before the discount runs out. Eight reward terms were added inside a 1.8-second window, which also explains
why the reward kept drifting toward dense shaping: immediate terms were the only ones that registered.

### 1.2 The network never grew while the problem more than doubled

| | Deku (SOLVED 78%) | Gohma | change |
|---|---|---|---|
| `obs_dim` | 483 | **1033** | +114% |
| `global_dim` | 99 | **297** | +200% |
| `actor_dim` | 24 | **46** | +92% |
| `n_actions` | 63 | **108** | +71% |
| `hidden` / `gru` | 256 / 256 | **256 / 256** | **unchanged** |
| parameters | 616,064 | 679,725 | +10% |
| params per obs dim | **1,275** | **658** | **halved** |

Measured sizes, so the target is not a guess:

| `hidden`/`gru` | params | per obs dim | vs the ratio Deku solved at |
|---|---|---|---|
| 256 (current) | 679,725 | 658 | 0.52× |
| **384** | **1,404,589** | **1,360** | **1.07×** |
| 512 | 2,391,597 | 2,315 | 1.82× |
| 640 | 3,640,749 | 3,524 | 2.76× |

**`hidden 384` restores the ratio that worked.** Not 512 — that overshoots, and bigger is not the hypothesis.

### 1.3 No PPO hyperparameter was ever retuned

`lr 3e-4`, `clip 0.2`, `epochs 4`, `minibatches 16`, `ent_coef 0.01`, `vf_coef 0.5`, `target_kl 0.03`,
`num_envs 8` — all unchanged from the Deku task. One of them is a measured problem right now: v18 logged
`ppo/value_loss` **14.58** against `ppo/policy_loss` **−0.003**, so with `vf_coef 0.5` the objective is
already ~99.96% value term. Raising `gamma` makes value targets 5–10× larger and the value loss 25–100×
larger, on a **shared trunk**. That is a predicted failure mode, so it gets its own arm rather than a guess.

---

## 2. Failure modes being actively avoided (review before each phase)

Taken from AGENTS 29–34 and from what this project has already paid for:

| mistake | how this plan avoids it |
|---|---|
| **judging on `success_rate`** — a 1–10% binary that called six improving versions "plateaued" (AGENTS 34) | judged on `combat/boss_damage_balanced`, mix-invariant and graded 0–10 |
| **judging a level instead of a slope** | `train_compare.py --trend` reports slope ± SE and refuses to fit under 5 updates |
| **a metric that moves with the curriculum** (AGENTS 29) | `dmgB`/`dmgH` are unweighted means over slots; the mix is held fixed anyway |
| **more than one variable per arm** (AGENTS 13) | one knob per arm, with an explicit control arm sharing the same init |
| **confounding the change with the starting level** — the error that made the `g_approach` dose-response unreadable | **every arm shares one init checkpoint**, so the start is held constant |
| **mistaking a resume transient for a result** — v18's slope was −0.119 at n=8 and −0.019 at n=8-dropped | the control arm carries the identical transient; judged on the window after it |
| **scaling before the cheap check** (AGENTS 9) | Phase 0 is free, Phase 1 is ~$11 and ~1 h, full training only after a gate |
| **burning a fleet on an unverified signal** (AGENTS 22) | GO/NO-GO written down below BEFORE launching, not after seeing results |
| **pushing assets from the Mac** (AGENTS 15/33) | `fleet bootstrap` now takes the S3 path automatically and refuses if S3 is stale |
| **trusting "launched: N"** (AGENTS 19) | `fleet preflight` then read the init/anchor lines off a pod |
| **piping a long job into `tail`** (AGENTS 24) | unbuffered to a file, read with `grep` |
| **holding the fleet while thinking** | each phase has a stated kill condition and a wall-clock budget |

---

## 3. Phase 0 — free, already done

No pods. Verified on the Mac:

- every knob is already a CLI flag: `--gamma --gae-lambda --chunk --hidden --gru --vf-coef`. **No code change.**
- capacity table above, built from the real config through `build_model`.
- advantages **are** normalised per minibatch (`zelda_rl/ppo.py:212`), so the policy gradient is scale-free
  under a longer `gamma`; only the value head sees the scale change.
- the `value_loss` / `policy_loss` imbalance above, which is why Phase 1 has a `vf_coef` arm.

---

## 4. Phase 1 — the credit horizon. 3 arms, 18 pods, ~1 h, ~$11

**Separate fleet experiments per arm**, because FedAvg averages weights across every rank in one experiment —
one fleet with mixed flags would average the arms together and measure nothing.

| arm | experiment | `gamma` | `lambda` | `vf_coef` | value horizon | advantage horizon | what it tests |
|---|---|---|---|---|---|---|---|
| **A** | `gohma-cr-a` | 0.995 | 0.95 | 0.5 | 20 s | 1.8 s | **CONTROL.** Same init, same reward, current horizon — isolates the init/reward transient |
| **B** | `gohma-cr-b` | 0.9997 | 0.997 | **0.05** | 333 s | **56 s** | the hypothesis: a horizon that covers the episode, with the value term rescaled to compensate |
| **C** | `gohma-cr-c` | 0.9997 | 0.997 | 0.5 | 333 s | 56 s | the same horizon **without** compensation — separates "long horizon helps" from "long horizon breaks the critic" |

Held identical across all three: `--reward damage` (outcomes only, 39 shaping terms zeroed),
`--init-from checkpoints/kept/gohma-kl-v15-r4_latest.pt`, `--start-slots 3,4,5,6,7,8`, v13's slot weights,
`--death-scale 0`, `chunk 60`, `hidden/gru 256`, 6 pods × 8 envs. No KL anchor — it would fight a new reward.

**Why `--reward damage`:** the point of a long horizon is that sparse outcome reward becomes usable. Testing
the horizon under eight dense shaping terms would hide the effect. Held fixed, so the arm variable is still
only the horizon.

**Why init from v15:** all arms share it, so it is controlled, and it starts at the 3.2–3.4 ceiling where the
question actually lives. The critic is calibrated to `gamma 0.995` and must re-fit in arms B and C — arm A
carries the same init without the gamma change, which is precisely what makes it the control.

### What is measured

`combat/boss_damage_balanced` slope ± SE over updates 5–15, i.e. after the critic transient.
`python scripts/train_compare.py --compare gohma-cr-X --trend`. Also watched: `ppo/value_loss` (the predicted
failure), `ppo/approx_kl` against `target_kl 0.03`, and `dmgH`.

### GO / NO-GO, written before launching

- **GO to Phase 2** if B or C shows a `dmgB` slope **> +0.012 HP/update** (v14's rate, the best ever measured)
  **and** |t| > 2, while arm A does not.
- **NO-GO, and the horizon is not the answer**: all three arms within noise of each other.
- **Kill immediately** if `ppo/value_loss` exceeds ~1,000 in both B and C (the critic has diverged) — then the
  follow-up is a `vf_coef`/return-normalisation fix, not more training.
- **Hard budget: 20 updates or 75 minutes, whichever first.** Then kill, regardless of result.

---

## 5. Phase 2 — capacity. Only if Phase 1 gates GO. 2 arms, 12 pods, ~1.5 h, ~$10

`hidden/gru 256 → 384` (params per obs dim 658 → 1,360, matching Deku's 1,275), at Phase 1's winning horizon.

**Changing `hidden` changes every weight shape, so a capacity arm cannot load v15's checkpoint — it must
start fresh.** That makes this phase slower and is the reason it is second, not first.

| arm | `hidden`/`gru` | init |
|---|---|---|
| D | 256 | scratch |
| E | **384** | scratch |

Both from scratch so the comparison is matched. Judged on `dmgB` slope over ~30 updates; from-scratch runs
start near 0 damage, so the slope is the only readable signal and the absolute level will stay low.
**Budget: 40 updates or 2 hours.** NO-GO if 384 does not beat 256 on slope with |t| > 2.

---

## 6. Phase 3 — init: does the Deku winner help? Only if Phase 2 gates GO. 2 arms, 12 pods

The question asked directly, run at the winning horizon and capacity:

| arm | init |
|---|---|
| F | `transfer.py`-widened `deku-combat-gru-v2b-20p_u00120.pt` (the 78% policy, 483→1033 obs, 63→108 actions) |
| G | scratch |

`transfer.py` is function-preserving: new global features and new actor fields get zero-initialised columns,
new actions are appended below the mean logit, so arm F computes **exactly what the solved Deku policy
computed** on its first step. **Caveat: if Phase 2 selects `hidden 384`, the Deku checkpoint (`hidden 256`)
cannot be widened by `transfer.py`, which widens inputs and actions but not hidden width.** Then Phase 3 runs
at `hidden 256` and is a separate question, or `transfer.py` needs a hidden-widening path first.

---

## 7. Phase 4 — the full run. Only if 1–3 all point the same way

One 20-pod fleet at the winning horizon, capacity and init, long enough to matter, with the eval fleet
alongside for the held-out question. Not before.

---

## 8. Cost ladder

| phase | pods | wall clock | cost | gate to proceed |
|---|---|---|---|---|
| 0 | 0 | minutes | **$0** | done |
| 1 | 18 | ~1 h | **~$11** | slope > +0.012, \|t\| > 2, in B or C and not A |
| 2 | 12 | ~1.5 h | ~$10 | 384 beats 256 on slope, \|t\| > 2 |
| 3 | 12 | ~1.5 h | ~$10 | Deku init beats scratch, or it does not and we use scratch |
| 4 | 20 | hours | $40+ | only if 1–3 agree |

Total to a justified full run: **~$31 and ~4 hours of ablation**, against the ~$60 already spent on v16–v18,
which produced three regressions because nothing was gated.

---

# RESULTS, 2026-10-15. The diagnosis held; every fix failed. Two winner's-curse errors.

## 9.1 What was established, at **M**

**The diagnosis is correct and stands.** `gamma 0.995 / gae_lambda 0.95` give a **1.8 s advantage horizon** on a
45–154 s fight, and `boss_defeat` +60 is worth **0.027** at episode start. Nothing below changes that.

**A long horizon with an unchanged learning rate is actively destructive, and lowering LR rescues it.** The
12-arm grid, all resumed from v15 (damage 3.275), one pod per arm:

| advantage horizon | lr 3e-4 | lr 1e-4 | lr 3e-5 |
|---|---|---|---|
| 1.8 s (γ.995/λ.95 — v15's own) | 2.40 | 2.42 | 2.32 |
| 9 s (γ.999/λ.99) | 2.39 | 2.30 | **3.16** |
| 30 s (γ.9997/λ.997) | 1.21 | 2.07 | 2.30 |
| whole episode (γ.9999/λ1.0) | **0.76** | 1.74 | 2.08 |

Read the rows: at the whole-episode horizon, LR 3e-4 → 3e-5 moves damage **0.76 → 2.08, 2.7×**. At 30 s it is
1.21 → 2.30. At v15's own short horizon LR does **nothing** (2.40 / 2.42 / 2.32) — which is why this was never
found, and why v16, v17 and v21 all destroyed the policy by lengthening the horizon alone. **M**, a monotone
pattern across 12 cells with effects far larger than single-pod noise.

## 9.2 What FAILED, and the fix is not available

| attempt | result |
|---|---|
| long horizon retrofitted onto v15, `vf_coef 0.05` | critic never converged (value loss oscillating 7–207); damage fell |
| long horizon retrofitted onto v15, `vf_coef 0.5` | `value_loss` 788 → 166 still oscillating; damage **3.275 → 1.84, −44%, t=−3.30** |
| from scratch at whole-episode horizon (v20, `hidden 384`) | 0.39 damage, slope **+0.0030 ± 0.0019** — statistically flat, and the **same rate as v10's +0.0034** |
| the grid's best cell, confirmed at matched n | **tie with the control** (below) |

**A policy trained with a 20 s horizon cannot be switched to a long one.** Its value function is wrong by
~50× and the policy follows the broken advantages downhill faster than the critic recovers. Tried twice, at
two `vf_coef` values, both destructive.

## 9.3 THE TWO WINNER'S-CURSE ERRORS — the reusable part

**Error 1.** The grid ranked 12 arms on the **last-3-update mean of ONE pod** (~13 episodes/update). Its
winner, a6 (9 s / 3e-5), read **3.16**. Scaled to 13 pods, the same settings read **~2.5**. a6's own full-run
mean was 2.80, not 3.16 — **the top of a 12-way ranking on a noisy statistic is biased upward by
construction**, and I selected on exactly that.

**Error 2.** Confirming it properly — two arms, **both initialised from v15**, 7 pods vs 6, matched updates
1–6:

| | mean dmgB |
|---|---|
| Arm A (9 s, lr 3e-5) | 2.623 ± 0.096 |
| Arm B (v15 control) | 2.558 ± 0.067 |
| **difference** | **+0.065 ± 0.117, t=+0.56 — INDISTINGUISHABLE** |

The grid claimed **+32%**. At matched n it is **+2.5%**. The entire apparent gain was selection noise.

**This is AGENTS 30 repeated.** That rule was written hours earlier about the held-out *checkpoint* selector —
ranking on a statistic too noisy to rank with — and the same error was made on *arm* selection. The rule
generalises and is now stated as such in AGENTS 35.

## 9.4 Standing state

**v15 (`checkpoints/kept/gohma-kl-v15-r4_latest.pt`, md5 `397e27b2beb6`, damage 3.275) remains the best
policy. Nothing in this session beat it.** The credit-assignment line is closed for *retrofitting*; the only
untested route is **from scratch at a long horizon with the LR swept**, since from-scratch is the one setting
where the critic learns the long horizon from the beginning and the grid's LR interaction has never been
tested there.

---

# 10. Observation narrowing (2026-10-15): harmless, and it does not help

The actor table is `N_ACTORS=16` rows x 46 dims = **736 of the 1033-dim observation**, and the Gohma room has
at most 4 occupied slots (boss + 3 larvae) — so ~550 dims describe nothing on a typical step, while the
**SOLVED** Deku task was 483 dims total on this same 256-wide network. `--n-actors` keeps only the N nearest
rows (rows are sorted by distance, `zelda_rl/obs.py:203`; Gohma also has her own 69-dim block so she cannot
drop out).

**The enabler, which made this cheap:** the actor encoder is applied **per row and pooled**, so **no weight
shape depends on the row count** — parameters are 679,725 at every `n_actors`. Verified that v15's checkpoint
loads into `n_actors` 4 / 6 / 8 with `missing=0, unexpected=0` and a working forward pass. So the sweep
started from the best policy at damage 3.275 instead of from scratch.

4 arms x 3 pods, all from v15, **only the observation width differs** (v15's own γ/λ/lr retained):

| arm | `n_actors` | obs_dim | params/dim | vs Deku's 1,275 | n | dmgB | SE | slope | t | far% |
|---|---|---|---|---|---|---|---|---|---|---|
| n6 | 6 | 573 | 1,186 | 0.93x | 22 | **2.779** | 0.082 | −0.0153 | −1.20 | 0.19% |
| n4 | 4 | **481** | 1,413 | **1.11x** | 23 | 2.611 | 0.070 | **−0.0249** | **−2.65** | 1.74% |
| n8 | 8 | 665 | 1,022 | 0.80x | 26 | 2.605 | 0.071 | −0.0045 | −0.46 | 0.26% |
| n16 | 16 (control) | 1033 | 658 | 0.52x | 36 | 2.586 | 0.065 | +0.0009 | +0.13 | 0.31% |

**Best vs control: +0.193 ± 0.104, t=+1.85 — not significant.** Slope rule: gap +0.0096 against a combined SE
of 0.0159 → **TIE, region only.**

## 10.1 Two mid-run reports that reversed, recorded because the reversal is the lesson

At 12 updates n4 read slope **+0.0023** and `far` **3.33%**, and both were reported as "the only arm pointing
the right way" and "something to watch". At 23 updates n4's slope is **−0.0249, t=−2.65** — the sign flipped
and it is now the *worst*-trending arm — and its `far` halved to 1.74% while the control's moved too. Three
lucky updates, exactly the failure mode AGENTS 35 was written about hours earlier. **Nothing below 20 updates
per arm should have been quoted at all.**

## 10.2 What holds

- **Cutting the observation 1033 → 481 dims (53%) does no measurable harm.** Every narrowed arm is within
  noise of the control, with v15's weights loaded unchanged. ~550 dims really are dead weight. **M.**
- **Narrowing does not help either.** The capacity-ratio hypothesis — that restoring Deku's 1,275
  params-per-input would unlock learning — is **not supported**: n4 sits *at* that ratio (1,413) and is the
  worst-trending arm. **M.**
- **Every arm, including the control, sits below v15's 3.275 start.** The consistent effect across the whole
  day is the **120 s episode cap plus the removed approach reward**, costing ~0.7 damage regardless of
  anything else varied.

## 10.3 Levers tried on 2026-10-15, all ties

| lever | outcome |
|---|---|
| credit horizon, retrofitted onto v15 | **destroys it** — 3.275 → 1.84, t=−3.30, at two `vf_coef` values |
| credit horizon, from scratch | works but **+0.0030 ± 0.0019/update, the same rate as v10's +0.0034** |
| horizon x LR grid (12 cells) | real mechanism (long horizons need low LR, 2.7x), **no winning cell** |
| learning rate from scratch (4 arms, 30x range) | **four-way tie** |
| observation width (4 arms, 1033 → 481) | **tie**; harmless but not helpful |

**v15 (damage 3.275) is still the best policy. Nothing on 2026-10-15 beat it.**
