# Rules for working in this repo

Written after a multi-day failure. Every rule below is here because breaking it cost real time, and the
specific incident is named so the rule cannot be argued away as generic advice.

## 1. A number is not a result until its controls are run

Never report, act on, or build on a metric before running all three controls. Run them BEFORE spending
compute on the thing the metric seems to justify, not after.

| control | question it answers |
|---|---|
| **identity / do-nothing** | does the model beat "nothing changes"? A zero-init residual predictor IS the identity, so skill = 0 is the floor, not a bad score. |
| **untrained, same architecture** | did *training* do anything, or is this the shape of the map? |
| **free-signal ablation** | is the signal a clock, an episode constant, or a copy of the input? |

**The incident.** The competent PPO policy's `proj` latent scored skill **+0.742** as a predictive surface
against our encoder's −0.48, and the obvious conclusion — "reuse the competent policy as the encoder" — was
reported before any control ran. The controls then destroyed it completely:

- same architecture, **random untrained weights**: **+0.617**. Competence contributed nothing.
- minus the **control echo**: **+0.240**. `obs` dims 22:26 are `prev_action_*`, a literal copy of `a_t`
  into `obs_{t+1}`.
- minus **velocity**: **+0.053**. With velocity in `z`, next position is linear extrapolation.
- minus **`time_left` and the `goal_*` block**: **−0.012**. A deterministic countdown and episode constants.

The whole +0.742 was artifacts. Cost: days, and a WM/PPO run was nearly launched on it.

## 2. Predictability and informativeness must be reported together

A latent that never moves is trivially predictable and useless; a constant encoder is the global optimum of
any pure prediction objective. So **every** skill number is reported next to:

- `move/rand` = mean |z_{t+1} − z_t| / mean |z_i − z_j|. Low means the latent barely moves per step.
- a **decode probe** (ridge is fine) for the quantities the reward actually reads: position, yaw, HP, counts.

Two columns, always. `scripts/eval_policy_latent.py` is the template.

## 3. Never compute an offset, index, or field width by hand

Derive it from the definition in code and assert it.

**The incident.** The `obs` echo slice was hand-counted as `slice(23, 27)`. `GLOBAL_FIELDS_V2` sums to 22, so
that slice zeroed half the echo and one unrelated `state_flags` bit — which is why the first echo ablation
falsely reported +0.575 surviving. Fixed by building the slices from `GLOBAL_FIELDS_V2` itself.

Same rule for RAM offsets: read them from the [OoT decompilation](https://github.com/zeldaret/oot), never from
recall. See `docs/ADDING_OBSERVATIONS.md`. Retail Player fields are `decomp offset − 0x10`.

## 4. Read the whole definition before claiming a field is absent

**The incident.** "Our encoder never saw relative positions or velocity" was asserted to the user after
reading only the first half of `featurise()`. Both were already there — `d = rpos - lpos`, `dist`, bearing and
per-actor velocity in the row block, and Link's `speed`/`vel_xyz` in the v2 link block. The claim was wrong
and had to be retracted in the next message.

`grep` the symbol, open the function, read it to the `return`. Do not infer absence from a partial read.

## 5. Control echo: check every input vector for a copy of the action

An encoder input that contains the previous action makes the next latent trivially predictable and teaches the
world model nothing. This has now bitten twice: once in our own features (`CONTROL_ECHO_FEATS`, removed — see
`docs/LATENT_WORLD_MODEL.md` §26) and once in the recorded 483-d `obs`, where it is present **by construction**
and was not noticed for days.

Before using any recorded vector as an encoder input — including one produced by another system — list its
fields and find the action-derived ones.

## 6. Actions, action-states and outcomes are different things

- **action**: what the agent commands — stick, buttons. A roll is an action. A block is an action.
- **action-state**: what routine Link is in — `actionFunc`, melee state, airborne.
- **outcome**: what the world does in response — position, HP, damage, kills, collisions.

Conflating them at the encoder or decoder level produces models that "predict" outcomes by echoing actions.
Keep them separate, and be explicit about which one a head predicts.

## 7. One seed is not a measurement

Six seeds of one identical config spanned walk trajectory error 1.38×–3.43× (`docs/LATENT_WORLD_MODEL.md` §25).
Headline numbers reported earlier in this project were seed 0 — the best of six on all three metrics at once.
Report a range, or say explicitly that it is a single seed.

## 8. The barometer is a ground-truth rollout, not a training loss

Anything claimed about the world model is validated by replaying **real action sequences from real start
states** and comparing decoded trajectories against ground truth, at multiple horizons
(`scripts/eval_dream_rollout.py`, `scripts/eval_event_windows.py`). A better training loss, a better tune
objective, or a better skill number is not a result on its own.

## 9. Ablate before you scale

Cheap ablation first, expensive training second. Every hour of training spent on an unvalidated headline
number is an hour that has to be thrown away, plus the time spent believing the conclusion it produced.

## 10. Report failure plainly and immediately

When a measurement kills a hypothesis — including one already reported to the user — say so in the next
message, state which earlier claim is now wrong, and do not soften it. Retracting costs one message.
Not retracting costs days.
