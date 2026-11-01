> **SUPERSEDED 2026-11-01 by `paper/paper.md`** ("The Legend of PPO: Learning to Fight in
> Hyrule"). Kept for the record; its section numbering and several of its figures were
> overtaken, including the v11 trend it cites, which is withdrawn.

---

# Outline

Working title: **Two negative results and one positive one, from learning Ocarina of Time from RAM.**

The honest shape of what this repository contains. The RL track works and the WM track does not, and the
interesting paper is the pair, not either alone.

## 1. Setup
Dolphin 2603a, Null backend, RAM read zero-copy from shared memory, 10 Hz control, offsets from the OoT
decompilation. One shared state layout so collector and models cannot drift (`zelda_rl/state_layout.py`).
Why RAM and not pixels: the reward is defined on RAM quantities, and nothing has to be rendered.

## 2. RL: combat is learnable (R1, R3, R5, R6, R8)
Deku Baba solved in 4 h 52 m / ≈$51 on 20 CPU pods, FedAvg across ranks. Then Queen Gohma: 13 numbered
attempts, still unsolved, and the useful content is *why*. Three findings that generalise beyond this game:

* **Capability before credit.** Two actions the task requires (`Z+Shield`, `Z+Nut`) were absent from the action
  space; the shield registered on 17% of presses. No reward shaping could have fixed that, and four versions of
  reward shaping were spent before it was found (R7).
* **Shaping budget competes with the goal.** On engaged slots shaping was 88% of reward magnitude and
  `red_eye_seed` paid more per episode than defeating the boss. Removing 19 terms and paying stun credit only on
  conversion doubled the defeat rate (R6).
* **Curriculum shape, not difficulty.** A finisher-heavy mix climbs fast and learns the wrong skill; inverting it
  stalls; a graded ladder gives 21× the full-difficulty success rate (R8).

## 3. WM: the world model does not get far enough (W1–W4, W12)
Six latent architectures lose to the identity. The token model is positive to about five steps and worse than
doing nothing by a hundred. The rollout viewer makes the failure legible: Link ends 429 world units away.

## 4. Why: the metric, the objective, and the action (W2, W5, W6, W7, W8)
The four things that cost the most time, each with a measurement:
* **loss is not the metric** — val loss fell 1.8× while rollout skill collapsed to −0.41;
* **the objective must be the metric** — L1 vs MSE moved aggregate skill −0.379 → +0.392 with nothing else changed;
* **a 1.35 M-param MLP matches both token models at one step** — so one-step skill does not discriminate architectures;
* **motion is not action-conditioned, and 43% of states cannot be** — the counterfactual set separates the two populations.

## 5. What a negative result needs to be usable
The instrument matters more than the arm. Three of this session's conclusions were artefacts of the *metric*:
scoring a gate on position skill when the gate excludes position; measuring a grid on a refit rather than the
grid in the checkpoint; and a threshold rule derived from a normalisation bug. All three are recorded with their
retractions (`docs/WM_MOTION.md`).

## 6. Threats to validity
Straight from `paper/CLAIMS.md` and `PROVENANCE.md` §5: single seeds, five simultaneous changes never ablated,
an incomplete wall map (209/1024 cells), a biased H=1800 subset, and one watched-episode-per-slot result.

## Not in scope
The dream-PPO line (`dream-deku-baba`, `results/andojkgi/`) and the causal/OOCDM attempt
(`docs/CAUSAL_DEBRIF.md`) are recorded but not defended.
