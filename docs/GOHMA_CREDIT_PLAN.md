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
