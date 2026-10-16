# The event classes: which moments of human play the clone is told to care about

Uniform behaviour cloning spends gradient in proportion to how often something happens. In this corpus that
means **walking**: the human's first damage lands at step 190 of a 546-step episode and only 2.99 seeds are
fired per win, so the overwhelming majority of frames are approach and waiting. The clone trained that way
reaches 74.5% frame accuracy and takes **0.33 of her 10 HP** (`PROVENANCE.md` §3.3).

So each step gets a loss weight by what DECISION it belongs to. Every class below was measured from the corpus
(`scripts/mine_corpus.py`, `scripts/mine_slingshot.py`), not guessed.

## The classes

Weight applied to a step = **`1.0 + scale × factor`**, where `scale` is `--bc-event-weight` /
`--event-weight` and a step takes the **maximum** over the classes covering it (overlaps do not compound).
A class covers the `--event-span` steps *before* the event as well as the event step itself, because that
span is where the decision was actually made.

| class | factor | what marks it, exactly | events | steps | % of corpus | weight at scale 3 |
|---|---|---|---|---|---|---|
| `seed_hit` | **1.5** | a seed was fired AND an impact followed within 12 steps (1.2 s) | 260 | 3,062 | 5.6% | 5.5 |
| `boss_hp` | **1.5** | `gohma_hp_frac` fell — the payoff moment | 678 | 6,935 | 12.7% | 5.5 |
| `stun` | 1.2 | `gohma_stunned` went 0→1 — the window that permits sword damage | 313 | 4,068 | 7.4% | 4.6 |
| `sword` | 1.0 | action contains **B** while she is stunned and within 120 u | 3,975 | 9,904 | 18.1% | 4.0 |
| `seed_fire` | 0.8 | `since_last_seed` reset — any shot, landed or not | 299 | 3,440 | 6.3% | 3.4 |
| `sling_out` | 0.8 | `held_slingshot` went 0→1 — the equip decision before every shot | 179 | 2,327 | 4.3% | 3.4 |
| `larva` | **0.3** | a larva's HP fell | 372 | 4,269 | 7.8% | 1.9 |
| *(everything else)* | — | positioning, approach, waiting | — | 32,793 | **60.0%** | 1.0 |

At `scale 3, span 12`: **40.0% of steps are up-weighted, 60.0% stay at 1.0.** Positioning is never discarded —
that would break the GRU's continuity across an episode — it is just no longer the majority of the gradient.

## Why those factors, and the evidence behind each

- **`boss_hp` and `seed_hit` highest.** A stun gates **678 of 678** recorded hits, and a stun comes from a
  seed reaching the red eye. These two classes are the mechanism the fight turns on.
- **`stun` just below them.** It is the gate, but the human gets ~2 hits per stun (6.78 hits from 3.13 stuns),
  so the hits themselves carry more signal per event.
- **`sword` at 1.0 despite being the largest class** (3,975 events, 18.1% of the corpus). It is deliberately
  not the top factor: swinging is cheap and frequent, and a swing outside a stun is wasted. The definition
  therefore requires `gohma_stunned` AND range < 120 u, so only *productive* swings are counted.
- **`seed_fire` and `sling_out` at 0.8.** Firing beats waiting — the human lands **87%** of shots (260 of
  299) — but the shot that matters is the red-eye one, which `seed_hit` already covers at 1.5. These two
  classes are the setup, worth less than the outcome.
- **`larva` at 0.3, deliberately the lowest.** Larva hits correlate **−0.598** with damage dealt across the
  100 episodes: engaging the larvae is what a human does when the fight is going badly. Included at all only
  because they are sometimes forced (she reopens her eye after they die), and an arm of the sweep removes them
  entirely to test exactly that.

## What the sweep varies

`--event-weight` (scale), `--event-span` (steps before the event), `--event-classes` (restrict which classes
count — empty means all). Seven arms, each training on one pod and **evaluating its own checkpoint on that same
pod**, so a checkpoint is scored seconds after it is written:

| arm | scale | classes | the question |
|---|---|---|---|
| `uniform` | 1.0 | — | the control: does stratification do anything at all? |
| `s1.5` | 1.5 | all | is a gentle nudge enough? |
| `s3` | 3.0 | all | the default |
| `s6` | 6.0 | all | does more pressure keep helping? |
| `s12` | 12.0 | all | where does it break? |
| `nolarva` | 3.0 | all except `larva` | does dropping the −0.598 class help? |
| `chainonly` | 3.0 | `seed_hit, seed_fire, sling_out, stun, boss_hp` | is the seed→stun→HP chain sufficient on its own, with no sword or larva? |

## How these are read

Frame accuracy is **not** the metric for a stratified arm: by construction it stops optimising the uniform
frame distribution, so accuracy can fall while the clone gets better at what matters. The 100-game uniform
clone scores 74.5% and the scale-3 one 72.7%, and that difference says nothing. **Every arm is judged on
online boss damage over the same 80-episode suite** (8 episodes on each of the 10 starts), against the
uniform arm and against the 0.33 the unstratified 100-game clone achieved.
