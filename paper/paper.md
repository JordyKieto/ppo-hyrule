# The Legend of PPO: Learning to Fight in Hyrule

**Model-free reinforcement learning from live game memory in *The Legend of Zelda: Ocarina of Time*, and
four things that did not beat it.**

Status: draft, 2026-11-01. Every quantitative statement here is audited in [CLAIMS.md](CLAIMS.md) with an
evidence class and traced to an artefact in [`../PROVENANCE.md`](../PROVENANCE.md). Nothing below should be
read as a result unless it is marked **M** (measured) there. Claims that were published inside this project
and later overturned are listed in section 12 rather than deleted.

---

## Abstract

We train agents to fight in an unmodified commercial 3D game, *Ocarina of Time*, running under the Dolphin
emulator. The agent never sees a pixel. It reads the emulator's memory directly, giving a structured view of
the world: every actor's position, velocity, health and decoded animation routine. This removes perception
from the problem and leaves control, timing and credit assignment.

Two tasks. The first, a three-enemy melee encounter, is solved by recurrent PPO in under five hours on twenty
CPU machines. The second, the dungeon boss Queen Gohma, is not solved, and the useful content of this report
is why. Gohma takes damage only while stunned, a stun must be manufactured, and a kill is ten such
manufactured openings inside a fight that takes a human 154 seconds. Our best policy defeats her on 8.33% of
456 held-out episodes and removes 3.53 of her 10 health on average.

Four alternatives were measured against that policy under matched conditions and **all four lost**. Behaviour
cloning on 100 recorded human victories reaches 74.5% held-out next-action accuracy, nearly four times the
majority-action floor, and takes 0.33 of her 10 health with 1 win in 80 attempts. Fifteen separate mechanisms
for injecting those demonstrations into the policy gradient produced no arm with an improving trend. Offline
return-conditioned behaviour cloning reaches 1.31 health and 3.95% defeats at matched episode caps. A learned
world model trained for 120 epochs ends further from "nothing changes" than it started, ignores the action
entirely, and the policies dream-trained inside it won 0 of 607 real episodes.

The second half of the report is about measurement, because in this project the instrument was a larger source
of error than any method. A defeat is a 1 to 10% binary event. Read through it, six consecutive versions of the
boss policy appeared flat and two experiments were launched to break the apparent plateau. Read through the
graded quantity the reward actually pays for, the same six versions rose **8.7 times**, every step
statistically solid, and one of the experiments launched to fix the plateau measurably caused one. We give the
per-start profile rather than a pooled rate, because the pooled rate moves with a curriculum we choose, and we
record nine claims this project published and then withdrew.

---

## 1. Contributions

1. **A boss fight as an RL benchmark with the mechanism fully instrumented.** Not "sparse reward" in the
   abstract: a measured causal chain, mined from 100 human victories, in which 678 of 678 damage events
   required a stun, a stun converts from a projectile hit on only 40.6% of attempts in the state where the
   agent is most likely to attempt it, and 0% of 71 attempts in two other states (section 4).
2. **A measured profile of start-state dependence** rather than a pooled success rate. Defeat rate across six
   starts of the same task ranges from 28.95% to 0 of 76 episodes, ordered by how much of the fight the start
   hands the policy (section 5).
3. **Imitation measured against the policy that never saw a human.** Frame-level accuracy and closed-loop
   competence are not merely weakly related here, they are **anticorrelated across corpora** at r = -0.875
   (section 6). Two policies that both beat the boss, one human and one learned, agree on **3% of frames**.
4. **A negative result on learned world models, carried to the point where the mechanism is named.** The model
   fails the action-shuffle control outright, so it is not a controllable dynamics model in any sense, which
   explains the dream-trained policies without appealing to the policy (section 7).
5. **One positive representation result that clears its controls**: a masked-prediction objective over the
   structured state beats both an untrained network of identical architecture and a free-signal control on 9
   of 10 frozen probes (section 8).
6. **A metric methodology, derived from its own failures.** Mix-invariance, selector resolution, the
   maximum-of-N bias, pooling as a measurement with its own failure modes, and the difference between a
   discount and a policy-gradient weight (section 9), each with the specific wrong decision it caused.

---

## 2. Setup

### 2.1 Observation: live game memory

The game is *The Legend of Zelda: Ocarina of Time* (GameCube build `NACE01`) under Dolphin 2603a with the
video backend disabled. Observations are read zero-copy from Dolphin's shared memory arena at about 25
microseconds per step. Control runs at 10 Hz, one action every 6 vertical interrupts.

Every memory offset is taken from the [OoT decompilation](https://github.com/zeldaret/oot) rather than from
inspection, with the retail Player structure at the decompilation's offset minus `0x10`. One shared definition
(`zelda_rl/state_layout.py`) is imported by both the data collector and every model, so the two cannot drift.

Reading memory is not a shortcut past the hard part. The state is a variable set of actors over twenty fields
each, and the quantity that matters most for combat, an enemy's current **action routine**, is a function
pointer we decode into a named behaviour state. A Deku Baba telegraphs its lunge; Gohma's eye turns red before
she is vulnerable. Those are the observations a human plays on and they are directly available here, which is
the reason the environment is interesting rather than a reason it is easy.

### 2.2 Actions and reward

Actions are a discrete product of nine world-relative directions and a button set. The melee task uses 63
actions; the boss task needs 108, adding slingshot, Deku Nut, first-person aiming, and the Z-plus-shield and
Z-plus-nut combinations.

Reward is computed by one tracker (`zelda_rl/combat.py`) shared by the real environment and the learned one, so
the two cannot disagree by construction. The boss configuration used by every result here zeroes 39 shaping
terms and pays principally for damage dealt (+4.0 per health point) and the kill (+60).

### 2.3 Policy and optimisation

The policy is an entity-factored recurrent actor-critic: a 16 by 24 actor table through a shared per-actor MLP
with an identity embedding, masked mean and max pooling, a 256-unit GRU, then actor and critic heads. Training
is recurrent PPO on 60-step chunks, each replayed from its stored hidden state. Across machines we average
weights every update rather than averaging gradients.

### 2.4 Compute

Twenty 16-vCPU cloud machines for reinforcement learning, A100 or H100 for world models. The melee task cost
**$51 and 4 hours 52 minutes** for the policy that solved it, about **$1.30 per million environment steps** at
3,300 to 3,800 steps per second fleet-wide.

---

## 3. Task one: the melee encounter is solved

Three Deku Babas in the first room of the Deku Tree, with the room exit behind the agent so that walking out is
an available failure. Success is all three defeated.

At global update 150 the policy clears **10 of 10 start states locally with 0 deaths**, six of them without
taking damage (one watched episode per state, so this is qualitative, not a rate). Across the fleet the same
checkpoints average **78% success and 83% of Babas killed**, over 41.8 million environment steps, which is
about 1,160 hours of game time.

The learned behaviour is recognisable as play rather than as exploitation: it walks in from the entrance, wakes
Babas that are underground, locks on, and times attacks against the lunge animation. The clip accompanying this
report (section 11) is one such episode, 26.7 seconds, three kills at 7.3, 22.4 and 26.8 seconds.

This task matters mainly as calibration. It establishes that the observation, the action space and the
optimiser are jointly sufficient to learn real combat in this engine, which is what makes the boss result a
statement about the boss rather than about the setup.

---

## 4. Task two: what Queen Gohma actually requires

Gohma has 10 health, one point per sword hit, and **she is only damageable while stunned**. A stun must be
manufactured: a slingshot seed into her eye while it is red, or a Deku Nut while she is on the floor. She
spends much of the fight on the ceiling where she cannot be reached, and she spawns larvae.

We mined the structure from 100 recorded human victories (54,626 steps, every one a win). This is the
specification of the task, measured rather than asserted:

| quantity, per human win | value |
|---|---|
| sword hits | 6.78 |
| stuns manufactured | 3.13, so about 2 hits per opening |
| slingshot seeds fired | 2.99 |
| Deku Nuts thrown | 1.71 |
| larvae hit | 3.72 |
| first damage at | step 190, about 19 s, a third of the episode is setup |
| Z-targeted | 63% of steps |
| shield used | **0%**, the human never blocks |
| mean range to her eye | 400 units, but **50 units at the moment damage lands** |

And the gate, from 678 real damage events:

| condition within 2 s before her health falls | share |
|---|---|
| she was stunned | **678 / 678 = 100.0%** |
| a seed impact had landed | 195 / 678 = 28.8% |
| the engine's own "vulnerable" flag was set | **9 / 678 = 1.3%** |

The last row is worth its own sentence: the obvious flag to key a reward on identifies 1.3% of real damage
events. A reward built on it would have been paying for almost nothing.

Conversion is conditional on her state at the moment of impact, across 371 impacts:

| her state at impact | stun follows |
|---|---|
| falling, landing, already stunned | 100% |
| floor, preparing | 95.2% |
| ceiling, eye red | **40.6%** |
| ceiling, idle or moving | **0% across 71 impacts** |

So the projectile hit that looks productive, the red-eye shot on the ceiling, converts about two times in five,
and 71 of 371 impacts were wasted entirely. Hitting the red eye is necessary and not sufficient.

Two behaviours correlate **negatively** with damage dealt across the 100 wins: larvae hit at **-0.598** and
time she spends on the ceiling at **-0.559**. Fighting the larvae is what a human does when the fight is going
badly. Any method that imitates the corpus uniformly learns that behaviour in proportion to how often it
appears.

### 4.1 Why this is hard for a policy gradient, stated precisely

The advantage horizon of our optimiser is `1 / (1 - gamma * lambda)`. At the settings used by every run in this
project, gamma 0.995 and lambda 0.95, that is 18.3 steps, or **1.8 seconds** at 10 Hz, against a fight that
takes 45 to 154 seconds.

The consequence, asserted in a test against the real advantage computation: the terminal +60 for the kill
reaches the first decision of the episode at **0.000% of face value**. Raising the horizon to gamma 0.99967 and
lambda 1.0 brings it to 59.9% at 154 seconds.

A conflation that cost this project time is worth stating, because it is easy to make. The often-quoted
"47% at 15 seconds" is the **value** discount, `gamma^t`. The policy gradient is weighted by `(gamma * lambda)^t`,
which at 15 seconds is **0.023%**. Those are different numbers and only the second one describes what the
gradient sees.

**So the kill bonus has never reached the decisions that earn it, on any task in this project.** The melee task
was solved by dense shaping inside the 1.8 second window. Everything the boss policy knows about assembling a
chain of five manufactured openings, it learned from local shaping, with effectively no gradient from the
outcome.

---

## 5. The result: start-state dependence, not a single number

Our best boss policy is `gohma-kl-v15`, referred to below as v15. The measurement is 456 episodes per arm,
76 on each of six start states, a 300 second episode cap, and the same harness for every arm.

**The episode cap is part of the claim.** An earlier evaluation used a 120 second cap, below the 154 seconds a
human win takes, and so could not observe many of the wins it was counting. It reported 5.00%; the same policy
at 300 seconds reads about 12% in training and 8.33% held out. Every kill rate this project measured at 120
seconds is withdrawn (section 12).

### 5.1 The profile

| start | Gohma at start | distance | wins / 76 | defeat rate | 95% CI | mean damage | share of her starting HP |
|---|---|---|---|---|---|---|---|
| 7 | stunned, 5 of 10 HP | 38 u | 22 | **28.95%** | 19.96 to 39.96 | 3.76 ± 0.13 of 5 | **0.753** |
| 5 | ceiling, eye red, 10 HP | 347 u | 8 | 10.53% | 5.43 to 19.42 | 6.36 ± 0.32 of 10 | 0.636 |
| 3 | stunned, 10 HP | 189 u | 5 | 6.58% | 2.84 to 14.49 | 5.83 ± 0.27 of 10 | 0.583 |
| 4 | ceiling, spawning, 8 HP | 436 u | 1 | 1.32% | 0.23 to 7.08 | 2.93 ± 0.22 of 8 | 0.367 |
| 6 | floor, eye red, 6 HP | 357 u | 2 | 2.63% | 0.72 to 9.10 | 1.01 ± 0.21 of 6 | 0.169 |
| 8 | stunned, 10 HP | 448 u | 0 | **0.00%** | 0 to 4.81 | 1.26 ± 0.27 of 10 | 0.126 |
| **pooled** | | | 38 / 456 | 8.33% | 6.13 to 11.23 | 3.53 ± 0.14 | |

Three readings, in order of importance.

**The pooled rate is not a property of the policy.** It is an average over a curriculum we chose. Change the
mix and it moves without the policy changing. The per-start profile is the thing that transfers, and the
pooled row is included only because comparisons elsewhere in the literature are usually quoted that way.

**The binary cannot resolve the middle of the table and the graded quantity can.** The confidence intervals for
starts 5, 3, 6 and 4 all overlap. On defeat rate alone, only the extremes separate. The damage share orders all
six cleanly at a few percent of standard error: 0.753, 0.636, 0.583, 0.367, 0.169, 0.126. This is the same
phenomenon as section 9.1, appearing inside the headline result.

**It is not one variable.** Starts 5 and 6 differ by ten units of distance, 347 against 357, and by almost
fourfold in damage share. What differs is her action state: on the ceiling with her eye red she presents a
shootable target immediately, on the floor with her eye red she does not. Start 8 is the worst of the six
despite beginning stunned, because she is 448 units away. With six starts we cannot separate distance, her
remaining health and her action state, and we do not fit a model to six points.

### 5.2 What the policy actually learned, and what it did not

Across all 22 recorded wins on start 7, the slingshot converted **zero times**; every stun came from a Deku
Nut. Across all 8 wins on start 5 it converted **exactly once**, presumably the red-eye shot the start hands
it, with every later stun again a Deku Nut.

So the policy largely solved this boss with the wrong tool. The human corpus fires 2.99 seeds and 1.71 nuts per
win; the policy inverts that. In one recorded episode it had three consecutive red-eye windows and dropped all
three before falling back to nuts. This is a real behavioural difference from the demonstrations and it is one
reason imitation does not transfer (section 6).

### 5.3 The trajectory, and the ceiling

Across eight versions and five reward configurations, boss damage rose **0.38 to 3.32 of 10** and return rose
**-12.6 to +56.8**, with every version's slope distinguishable from zero. It then **saturates at 3.2 to 3.4**:
every run beginning near 2.6 climbs and every run beginning near 3.3 does not. The last six health points are
the open problem, and more training time is not the missing ingredient, since one version ran 399 updates and
ended at 1.21% while another ran 129 from scratch and never left 0.1%.

---

## 6. Imitation was not enough

The author beat Queen Gohma 100 times under a recorder that logs, at 10 Hz, the exact observation the policy
sees, the action index the human's input decodes to, the raw controller state, and the camera yaw used for that
decode. The decoder is proved to be the exact inverse of what the environment sends, round-tripping every
emittable action at five camera angles; an earlier version ignored the camera and was silently wrong for every
non-neutral direction.

### 6.1 Cloning the corpus

| | 13 games | 100 games |
|---|---|---|
| held-out next-action accuracy | 53.4% (floor 11.1%) | **74.5%** (floor 20.1%) |
| boss damage, all starts | 0.14 of 10 | 0.33 of 10 |
| defeats | **0 / 80** | **1 / 80** |
| deaths | 83.8% | 81.2% |
| **full-health starts, the actual task** | **0 / 24, damage 0.12** | **0 / 24, damage 0.12** |
| near starts under 200 units | 0.46 | 1.08 |
| mid and far starts | 0.00 / 0.00 | **0.00 / 0.00** |

**Seven times the data bought 2.4 times the damage, the clone's first win ever, and nothing at all on the
task.** On full-health starts the two clones are identical. All of the gain is within 200 units. The single
win came from the 38-unit start against a boss already at half health, which is to say it finished something
it was standing on.

Event-stratified cloning, weighting each step by the decision it belongs to, is the only thing that moved the
number: oversampling decision moments is monotone at matched evaluation, 0.16 then 0.33 then 0.53 of 10, and
**dropping the larva class is worth quadrupling the weight** (0.49 against 0.26 to 0.33), exactly as the
corpus's -0.598 correlation predicts. Across all those arms: **0 defeats in 400 episodes**. Damage scales,
winning does not.

The mechanism is not mysterious. The clone is trained teacher-forced, so every step conditions on the human's
real observation history and it never learns to recover from the states its own mistakes produce. 81% of its
episodes end in death.

### 6.2 Frame accuracy is an anti-selector

Across five stratified cloning arms, the correlation between best validation accuracy and online damage is
**r = -0.11**. Across 14 offline arms with different corpora it sharpens to **r = -0.875** (t = -6.25) on
win-class accuracy, with training loss positively correlated with online damage at +0.624. The best-fitting
arms are the ones with the easiest corpus.

Within a single arm, accuracy does track its own damage, weakly and positively, at r = +0.441 ± 0.097. So the
same number is a usable progress signal inside a run and an actively misleading selector between runs.

And the single most compact statement of the problem: **v15, which does beat her, agrees with the human
demonstrations on 3% of frames.** Two policies that both solve the task share almost no frame-level behaviour,
so matching frames is a weak proxy for the thing being learned.

### 6.3 Fifteen mechanisms, none of which beat the policy trained without the data

Every arm below starts from v15's weights, so the question is whether demonstrations improve an already
competent policy. The bar is v15's own 3.275 on the mix-invariant damage metric.

| mechanism | updates | mean | max | slope ± SE | t |
|---|---|---|---|---|---|
| FitNets projection onto rollout states | 14 | 2.87 | 3.37 | -0.0064 ± 0.0192 | -0.33 |
| latent transition matching | 14 | 2.79 | 3.21 | +0.0100 ± 0.0159 | +0.62 |
| mutual alignment, contrastive | 22 | 2.66 | 3.28 | +0.0017 ± 0.0132 | +0.13 |
| one-way latent to the frozen clone | 17 | 2.47 | 3.19 | -0.0220 ± 0.0188 | -1.17 |
| action-level BC, coefficient decaying | 26 | 2.32 | 2.74 | -0.0086 ± 0.0049 | -1.74 |
| on-policy distillation | 7 | 2.43 | 3.67 | **-0.2779 ± 0.0930** | **-2.99** |
| action-level BC, constant 0.1 | 8 | 2.31 | 2.64 | **-0.0857 ± 0.0172** | **-4.98** |
| clone proposes, Q head selects | 9 | 0.38 | 0.51 | -0.0050 ± 0.0105 | -0.48 |
| clone as initialisation | 6 | 0.28 | 0.35 | rising from nothing | |
| **v15, no human data at all** | | **3.275** | | the bar | |

**Three arms produced a single update above the bar and none produced a trend above it.** The constant
coefficient is the informative row: agreement with the demonstrated action rose from 3% to 23% and the BC loss
fell, so the term was provably doing its job, and the headline it cost was damage, at t = -4.98.

A cleaner decomposition came from a three-arm, three-seed study on the policy's **own** online states.
Aligning only the internal representation to the clone is neutral on every seed (+0.032, -0.066, +0.123).
Adding a penalty on the clone's **actions** at those same states is harmful and not marginally so, at -0.107
(t = -5.50) and -0.110 (t = -4.83). Pressing a 0.53-damage policy's action distribution onto a 3.275-damage
policy pulls the latter toward the former, which is the obvious thing to say afterwards and was not obvious
before it was measured.

### 6.4 What did work offline, and what it says

Return-conditioned behaviour cloning, trained on the same 100 human wins **plus** a few hundred of the policy's
own failures and 200 of its wins, with no policy gradient anywhere, reaches **1.76 ± 0.11 damage and 8 defeats
in 240 episodes (3.3%)**. Both plain clones of the same 100 games scored **0 in 400**.

The informative contrast is the corpus, not the architecture:

| corpus | pooled damage | defeats |
|---|---|---|
| human wins + failures + 200 policy wins | **1.76 ± 0.11** | 3.3% |
| human wins + failures | 0.93 ± 0.17 | about 1% |
| **human wins alone** | **0.30 ± 0.04** | 0.4% |

**Training on victories alone is the worst configuration by a wide margin.** The failures are what label the
fatal action. Adding the policy's own wins to the win class is worth +0.67 ± 0.20 (t = 3.3), and more of them
is not monotonically better: 200 beats all 495 at t = +2.43.

It still loses to the policy gradient. At matched 300 second caps, 456 episodes per arm, same harness:

| | v15 (PPO) | best offline (Dyna-RvS) |
|---|---|---|
| defeats | **38 / 456 = 8.33%** | 18 / 456 = 3.95% |
| starts won from | **5 of 6** | 2 of 6 |
| boss damage | **3.53 ± 0.14** | 1.31 ± 0.08 |

Two-proportion z = +2.76 on the binary; **t = +13.91 on the graded metric**. Off the easiest start, v15
averages 4.26 health to the offline policy's 0.88. Note the ratio of evidence: the graded quantity separates
these arms at 14 standard errors where the binary manages 2.8, on the same episodes.

Two further offline results are worth recording because they close plausible escape routes. Corpus size from
800 to 1,800 episodes at fixed composition is worth **+0.17 ± 0.14**, indistinguishable from zero, while ten
times the episodes at a diluted win fraction is **worse** than two times at fixed composition. And a bug worth
generalising: sharding the 100 demonstrations across seven machines, 14 each and disjoint, produced a 3.4-fold
spread in win-class accuracy across machines and a collapse at -0.0639 ± 0.0109 per epoch. Keeping the scarce
class whole on every machine and pinning its share of the gradient cut that to -0.0165 ± 0.0083, a difference
of +0.0474 ± 0.0137 at t = +3.46. The mechanism was real and the fix was insufficient: the repaired run still
does not rise.

---

## 7. The world model did not get far enough

Four phases over two months, each with its own architecture family, dataset scale and reason for being
abandoned. The baseline throughout is **persistence**, the prediction that nothing changes, and skill is
`1 - model error / persistence error`, so negative means worse than doing nothing.

| phase | scale | verdict |
|---|---|---|
| latent transition models, 6 configurations | 203k transitions | every one **loses to persistence**, best -1.14, worst -7.24 |
| object-centric causal model | same | the adopted paper's actual method was never run; what trained was its baseline |
| entity tokens, fleet scale | **20.5 M** transitions | **validation loss fell 1.8 times while rollout skill collapsed from +0.003 to -0.41** |
| tokens plus history, single H100 | **91.5 M** transitions, resident in VRAM | positive through 5 steps, -2.44 at 1800 |
| boss observation model | 4.7 M transitions, 120 epochs | **getting worse at -0.0874 ± 0.0218 per epoch, t = -4.00** |

The last row is the clearest statement available. After 120 epochs, with the output layer zero-initialised so
that the model **begins** exactly at persistence, 120 epochs of training moved it steadily away from doing
nothing. Its event heads, pushed to catch rare damage events, learned to fire on almost everything: 930 to 950
false events per 1,000 quiet ticks at precision 0.003 to 0.009. And the decisive control:

> **Shuffling the action changes the prediction by a factor of 1.00.** The action is decoration.

That is not a model that is merely inaccurate. It is not a controllable dynamics model at all, so nothing can
be planned inside it. It predicts the diagnosis of the closed loop built on top: a full real-games to model to
dream-policy to real-evaluation loop ran nine iterations on six machines and its dream-trained candidates won
**0 of 607 real episodes**, against the anchor policy's measured rate, with candidate damage flat to declining
at -0.056 ± 0.033 per iteration. The loop's machinery is validated; its model is not.

Three diagnostic findings generalise beyond this game:

- **Motion is not action-conditioned at one step.** Ablating the action costs +0.002 of skill, and the cosine
  between the commanded action and the resulting displacement is 0.006 at lag 1.
- **43% of states are action-insensitive**, measured directly with 150 counterfactual pairs: identical state,
  two different movement actions held for three steps, both real outcomes recorded. The other 57% separate by
  a mean of 21.4 world units. Averaging over both populations is what makes action-conditioning look
  uniformly absent.
- **The objective mattered more than the architecture.** Changing the loss from squared error to absolute
  error, with the same network, data and epochs, moved aggregate skill from **-0.379 to +0.392**. And a
  1.35M-parameter MLP matches both attention models at one step, so one-step skill does not discriminate
  architectures at all.

One operational finding, recorded because it was expensive and avoidable: a reference implementation of the
method we were copying specified a rollout depth of 10. We used 600. At 600 the model degrades at
-0.1675 ± 0.0562 per epoch (t = -2.98); at the reference's depth of 10 it sits at persistence. The number 600
came from a passage in our own earlier write-up describing a **measured regression**, not a recommendation.

---

## 8. One representation result that clears its controls

The four earlier attempts at self-supervised representation learning here all optimised the same objective:
predict the next latent from the current latent and action. All four lost to the identity, for a mechanical
reason that was eventually measured: the latent moves 0.023 to 0.05 of a random pair's distance per step, so
copying is near-optimal.

The masked-prediction objective of I-JEPA and V-JEPA does not pose that task. It masks part of the **input**
and predicts a target network's representation of the masked part from the visible remainder. There is no
persistence shortcut because there is no previous latent. **That objective had never been run here.**

We ran it on the raw 363-field structured state, not the featurised policy observation, because the latter is
dense with derived redundancy: masking a field whose inputs remain visible scores arithmetic rather than
understanding. Masking is structured in the same spirit as V-JEPA's multi-block scheme, translated to time by
entity by field: a whole actor removed for a whole clip, a whole kind of field removed for every actor, a time
block, or scattered cells. Achieved mask ratio 0.791.

Then the encoder is frozen and probed, with three floors. The probe reads a **masked** encoding and predicts
the **hidden** content, because probing an unmasked state for a field that is an input to it scores the
identity map.

| probe | trivial floor | free-signal control | untrained, same architecture | **masked model** |
|---|---|---|---|---|
| which actor is locked on (accuracy) | 0.496 | 0.718 | 0.714 | **0.919** |
| melee animation (accuracy) | 0.327 | 0.593 | 0.603 | **0.798** |
| action routine (accuracy) | 0.378 | 0.603 | 0.627 | **0.791** |
| **enemies alive (mean abs. error)** | 0.772 | 0.539 | 0.565 | **0.181** |
| nearest enemy distance (world units) | 74.3 | 71.7 | 93.1 | **44.1** |
| floor height (mean abs. error) | 4.16 | 3.11 | 4.54 | **2.08** |

**9 of 10 probes beat both an untrained network of identical architecture and a ridge regression on whatever
cells are still visible.** Counting is the strongest: 0.181 average error on how many enemies are alive when
every cell that would tell you is hidden, against 0.539 for the free-signal control. This is the first
representation in this project not beaten by random weights; the previous attempt was worse than untrained on
7 of 10 fields.

What it does not establish, stated in the same breath. **One seed**, where six seeds of one configuration have
previously spanned a factor of 2.5 here. The objective **still loses to its own trivial control**: emitting an
unrelated clip's representation scores 0.0273 against the model's 0.0815, so it is improving toward that
control, from 8.1 times to 3.0 times, and has not passed it. A single regularisation constant across arms of
different scale. And the low-shot axis did not run: a floor on sample count collapsed two fractions to the same
number.

---

## 9. The instrument was the hardest part

This section is the one we would keep if we could keep only one. In this project, measurement error exceeded
method error.

### 9.1 A sparse binary cannot steer a task like this

Every "is it improving" answer came from the defeat rate, a binary at 1 to 10%. On that signal, six consecutive
versions read flat and two experiments were launched to break the apparent plateau. On boss damage, the graded
quantity the reward principally pays for, the same runs read:

| version | boss damage | slope | return |
|---|---|---|---|
| v10 | 0.38 to 1.21 | +0.003 * | -12.58 to 15.46 |
| v11 | 1.72 to 2.10 | +0.015 * | 1.06 to 18.75 |
| v12 | 2.06 to 2.22 | +0.008 * | 4.87 to 6.63 |
| v13 | 2.41 to 2.75 | +0.007 * | 18.93 to 26.36 |
| v14 | 2.86 to 3.22 | +0.012 * | 45.54 to 52.62 |
| v15 | 3.17 to 3.32 | +0.004 * | 54.25 to 56.79 |
| v16 | 3.12 to 2.85 | **-0.017** * | 56.08 to 46.88 |
| v17 | 3.19 to 2.86 | **-0.022** * | 52.35 to 37.55 |

`*` marks a slope distinguishable from zero. **The project improved 8.7 times with every step statistically
solid and it was reported as six consecutive failures.** A defeat requires ten successful openings; damage
counts each one, so one signal can show a trend on a few hundred episodes and the other cannot.

The wrong metric then caused harm rather than merely hiding progress. An approach-shaping term was added in v15
on the strength of a defeat-rate analysis. The last two rows are the versions that raised its coefficient and
added a related change. Both are regressions, confirmed on a mix-invariant per-start form so neither is a
curriculum artefact.

**The rules this produces.** Rank candidate metrics by information per episode before choosing one: a binary at
rate p carries p(1-p) of variance per episode and a graded quantity over 0 to 10 carries far more. Make the
headline the quantity the reward's largest terms pay for. Report a trend with its standard error, because
"flat" and "rising" are claims about a slope. And never let a sparse metric justify adding a reward term.

### 9.2 A metric that moves with the sampling distribution cannot compare policies

The defeat rate is episode-weighted over start states we choose. Version 11 showed the best trend of any
version, +0.270 per update over its final third, and was stopped early to try something else on that basis. Its
per-start history:

| start | distance | first third | last third |
|---|---|---|---|
| **7** | **38 u** | 7.32% | **14.00%** |
| 3, 6, 2, 8, 5, 1, 4 | 188 to 541 u | 0.00% | 0.00% |

It was not learning the boss. It was learning the one start where Link begins on top of her, and the aggregate
climbed because that run weighted that start at 50%. Nine starts sat at exactly zero for 37 updates.

The same instrument then lied in the other direction. Version 15 reads as the flattest run in the table while
being the first version non-zero from **6 of 6** trained starts where v11 was non-zero from 1 of 10, and about
25 times broader off the easy start.

Two consequences. Any headline that averages over a distribution you control must be reported next to a
mix-invariant form of itself, and the mix-invariant form is what a version-to-version claim cites. And, less
obviously: the policy gradient optimises expected return **under** the sampling distribution, so the curriculum
weights are part of the objective, not part of the evaluation.

### 9.3 A selector is not a selector until its statistic is shown to have resolution

A held-out evaluator chose which checkpoint to keep, ranking on 16-episode subsets. The pooled held-out rate
over every evaluation was 1.68%. At that rate a 16-episode subset reads 0.00% **76.2%** of the time and 6.25%
**20.9%** of the time. The 6.25% that crowned a checkpoint was one lucky win in sixteen, an outcome any
checkpoint produces one time in five. Tested directly, the selected checkpoint against every other gives
p = 0.094. Separating a genuinely twice-as-good checkpoint would need about 256 episodes each, sixteen times
what it had.

The same error, made again at the level of an **arm**: a 12-cell grid ranked on roughly 40 episodes per cell
produced a winner reading +32% over control. Confirmed properly as two matched arms from one initialisation,
the gap is **+0.065 ± 0.117, t = +0.56**. The entire +32% was the upward bias of taking the maximum of twelve
noisy estimates. The grid's qualitative finding, that long credit horizons need low learning rates, is monotone
across all twelve cells and survives; the winning cell is fiction.

### 9.4 Pooling is a measurement and has its own failure modes

Writing a script to pool evidence, replacing analysis that had been retyped per question, exposed three errors
in the hand-pooling, all of which had moved p-values toward significance: 2,831 episodes from a withdrawn
evaluator were still in the pool; the same evaluation existed in three copies, so one arm pooled to 1,551
episodes where it ran 880; and the quoted baseline was a 559-episode subset of an 880-episode run. Correcting
all three moved a comparison from p = 0.122 to p = 0.087 on the same data.

A related trap, one level lower: the newest point of a fleet's evaluation curve is usually **incomplete**,
because machines finish at different times and the ones that finish first are not a random subset. One point
read 1.105 on two machines and 1.250 on six. Three consecutive slope claims were made and reversed from this
in forty minutes.

### 9.5 What a claim has to clear here

Four independent measurements of the noise floor, all on identical configurations:

| replicate set | spread |
|---|---|
| five identical policy-gradient controls | 2.00 to 2.97, sd 0.380 |
| two identical offline seeds | 0.43 |
| two groups of three identical runs | differ by 0.25 ± 0.12, **t = 2.1** |
| six seeds of one world-model configuration | trajectory error 1.38 to 3.43 times |

The third row is the useful one: that reading crossed the usual significance bar while being pure seed noise.
A three-seed group mean carries ±0.1 here, so nothing below about 0.3 damage is visible with that design. Two
offline seeds at identical settings produced **opposite** significant slopes, +0.032 (t = 2.2) and -0.030
(t = -2.4), which is the plainest available statement that one seed's slope is not a measurement.

---

## 10. Limitations

- **One seed** for the masked-representation result, and for several world-model comparisons. By this
  project's own measured floors that is below the resolution needed.
- **Start-state coverage.** The boss curriculum trains on six of ten starts, and the four held out are all at
  half health and all beyond 368 units, so "held out" is collinear with "far" and no matched comparison exists.
  We therefore make no generalisation claim.
- **Damage pooled across starts with different starting health.** The pooled 3.53 "of 10" mixes starts at 5, 6,
  8 and 10 health. The share-of-starting-health column is the comparable one.
- **The melee per-state result is one watched episode per state**, so it is qualitative and is not a rate.
- **The wall map used by the world model covers 209 of 1,024 cells**, only where the agent happened to collide
  with something.
- **The long-horizon world-model numbers are thin**: about 31,000 of 52 million possible starts qualify at the
  longest horizon, and all of them are timeout episodes by construction.
- **Reproducing on macOS requires a code-signed interpreter** to attach to emulator memory. Linux does not.
- **The masked-representation substrate exists on one machine.** The 203,822-frame raw-state datasets behind
  section 8 are not mirrored off-site, so that result is currently reproducible only from local disk.

---

## 11. Media

Both clips are the real emulator at 10 Hz control, cropped to the game area, with game audio.

**The melee task, end to end** (`media/deku-slot1/deku-slot1_s01_try01_crop.mp4`, 29.7 s, md5 `d20c62aae595`).
The agent starts just inside the room entrance at full health with all three Babas underground 530 to 830
units away, walks up, wakes them and clears the room. Kills at 7.3, 22.4 and 26.8 s, 1.5 hearts lost. First
attempt.

**Queen Gohma from the full-health start** (`media/gohma-slot5/gohma-slot5_s05_try04_crop.mp4`, 197.7 s,
md5 `a184dbd29233`). She begins at 10 of 10 health on the ceiling with her eye red, 347 units away. The fight,
from the episode's own log:

| time | what happens |
|---|---|
| 0.6 s | slingshot seed into the red eye, she falls. **The only converted seed shot of the episode.** |
| 4.0 to 8.0 s | five sword hits inside that opening, 10 health down to 5 |
| 45.9, 48.6 s | she is back on the ceiling; two larvae killed with the sword |
| 53.9 to 65.4 s | she falls, two Deku Nut stuns, one hit, 5 down to 4 |
| 102.0, 106.2 s | two more larvae killed |
| 111.4 to 119.5 s | falls again, nut stun, two hits, 4 down to 2 |
| 156.5 s | more larvae cleared, eight in total across the fight |
| 181.9 to 191.0 s | final knockdown, nut stun, two hits, **defeated at 191.1 s** |

Ten sword hits, eight stuns (one from the slingshot, four from Deku Nuts), eight larvae killed, two hearts
lost, Link down to half a heart at one point.

**It took four attempts, and the three that failed are in the manifest**: deaths at 2, 5 and 2 of her 10
health. That start wins 10.53% of the time (8 of 76 matched episodes), so four attempts is a slightly lucky
draw on the attempt count and an entirely ordinary win. Every one of those 8 matched wins also kills larvae and
takes all ten health, so the clip is representative **of a win at that start**, and the attempt count is what
says how often a win happens.

Both directories carry an `attempts.json` recording every attempt, won or lost, with each clip's duration
checked against the episode's own length and a frozen-frame fraction confirming the capture is live gameplay
rather than a still. Three separate capture faults were found and fixed to get these, including one that
silently filmed an empty desktop while passing a duration check; they are recorded in `../PROVENANCE.md`
section 17.

## 12. Retractions

Claims this project published internally and then overturned. They are listed because the correction is the
reusable part.

1. **"A competent policy's latent is an excellent prediction target", skill +0.742.** Controls destroyed it:
   random untrained weights of the same architecture scored +0.617, removing the copy of the previous action
   took it to +0.240, removing velocity to +0.053, and removing a countdown and episode constants to -0.012.
   The whole number was artefacts.
2. **"Version 11 is the fastest-improving version", +0.270 per update.** One start state at 50% curriculum
   weight; nine starts at exactly zero for 37 updates (section 9.2).
3. **"Six versions plateaued."** They rose 8.7 times on the graded metric (section 9.1).
4. **"The 9-second, 3e-5 cell beats our settings", +32%.** Winner's curse; matched confirmation gives
   +0.065 ± 0.117 (section 9.3).
5. **"v15 defeats her on 5.00% of episodes."** Measured at a 120-second cap, below the 154 seconds a human win
   takes. Every kill rate measured at that cap is withdrawn, including an offline-versus-online comparison that
   rested on it.
6. **"Offline return-conditioned cloning matches the best policy gradient at killing."** Both sides were
   measured at the short cap. At matched 300-second caps the policy gradient wins by 14 standard errors on the
   graded metric.
7. **"Four machines trained one sharded world model."** They averaged nothing on any round: the aggregator was
   dead from launch and the reduction crashed on device mismatch, so it was four independent quarter-corpus
   models. Visible the whole time in the epoch times, 616 seconds on sync epochs against 15 seconds otherwise.
8. **Three world-model conclusions were artefacts of the instrument**, not the model: scoring a gate on a
   quantity the gate deliberately excludes; measuring a grid by re-fitting it instead of reading the one stored
   in the checkpoint; and a threshold rule derived from a normalisation bug.
9. **"The masked-representation run should be killed."** Recommended twice at five epochs from a slope of
   +0.0026 ± 0.0030 (t = 0.85), read as flat. At twelve epochs the same quantity is +0.0150 ± 0.0015
   (t = +10.29). The arithmetic was right; the number was not decision-grade, and the architectural diagnosis
   behind the recommendation being correct is what made the wrong prediction feel safe.

---

## 13. Reproducibility

Every number above traces to a directory. `../PROVENANCE.md` carries the resume lineage per version, wall clock
and cost, the step-axis discontinuities that make a curve misreadable without them, and why each version
exists including the ones that failed. `CLAIMS.md` is one row per claim with its evidence class.
`../artifacts/MANIFEST.json` identifies every checkpoint, dataset and result file by md5 rather than by
filename, with its remote key where one exists, because a stray checkpoint file cannot otherwise be matched
back to the run that produced it.

Training histories are **exported, not linked**: each run group has a local directory with its configuration,
summary and full history, including the identities of all per-machine runs so a fleet is recoverable and not
just its aggregator.

One warning for anyone pulling artefacts: four world-model checkpoints exist in remote storage under the same
filenames as the ones cited here but at roughly a third of the size. They are an earlier run. Match by md5.

---

## 14. Related work and positioning

The closest relatives are game-playing agents trained from emulator state rather than pixels, model-based
agents that optimise policies inside a learned simulator, and the self-supervised masked-prediction family.
Our contribution is not a new algorithm in any of those lines. It is a measured head-to-head on one hard,
mechanically rich task with the controls run: the same task, the same harness, the same episode cap, and the
same graded metric applied to a policy-gradient agent, a behaviour clone of human experts, fifteen ways of
mixing the two, an offline return-conditioned learner, and a dream-trained agent.

The result of that comparison is unfashionable and we think worth reporting precisely for that reason. On a
task with a long causal chain, a sparse outcome and a strong demonstrator available, the plain model-free
method trained with dense shaping beat every attempt to improve it with demonstrations or with a model, and
the largest single source of wrong conclusions along the way was not the choice of method but the choice of
number used to compare them.

---

## 15. Conclusion

An agent reading a commercial game's memory learns real combat: it clears a three-enemy encounter from every
start state, and on a dungeon boss it manufactures stuns, clears adds and lands ten-hit kills on 8.33% of
held-out episodes from six starts, reaching 28.95% from the most favourable and 0 of 76 from the least.

It does so without the reward for winning ever reaching the decisions that cause it. At the credit horizon used
throughout, the kill bonus arrives at the first decision of the episode at 0.000% of face value, so what the
policy knows it learned from local shaping.

One hundred human victories did not help. Cloning them reaches 74.5% next-action accuracy and 0.33 of 10
health. Fifteen mechanisms for folding them into the policy gradient produced no improving arm, and the one
that demonstrably worked mechanically cost damage at t = -4.98. Offline return-conditioned learning on the same
games plus the agent's own failures does substantially better than cloning and still loses to the policy
gradient by 14 standard errors. A world model trained to 120 epochs ends further from "nothing changes" than it
began and ignores the action entirely, and policies dreamed inside it won 0 of 607 real episodes.

The last six health points of the boss remain unreached, and the honest summary of this report is that we know
a great deal about why, mostly because we eventually started measuring the right quantity.
