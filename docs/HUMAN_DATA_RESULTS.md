# Human-data experiments, 2026-10-15/16 — results register

Every number here is a FLEET-ARM MEAN over a stated window, or an 80-episode held-out suite. Single-rank
numbers are not results: five pure-PPO replicates spanned **2.15–3.77 damage at update 1**, so any claim from
one rank-update is inside the noise. That mistake was made twice this session and both are retracted below.

**The bar: v15 = 3.275 boss damage (of 10), trained with no human data at all.** Nothing has beaten it.

## 1. The corpus

100 human wins, 54,626 steps, 9 start states (22 pre-fight), 92% action coverage. Versioned with per-episode
md5 / slot / start state / timestamp / recorder commit: **v1 = 13 episodes `5663643e97f8`, v2 = 100 episodes
`e57da6027a30`** (`scripts/dataset_manifest.py`), v1 a prefix of v2 so v1's published numbers stay
reproducible. Mirrored to `s3://zelda-rl/zelda/human/gohma/`.

Plus a PPO corpus collected from v15: **9,340 episodes, 495 wins (5.3%), ~5M steps**, per-step observations in
the same format, at `s3://zelda-rl/zelda/ppo/gohma_steps/`. It contains what the human corpus does not:
~8,800 losing episodes with deaths.

## 2. What the corpus says about the fight (M)

| finding | number |
|---|---|
| a stun gates EVERY hit | **678 of 678** hits had `gohma_stunned` within 2 s |
| `gohma_vulnerable` is the wrong signal | set for only **9 of 678** (1.3%) real damage events |
| slingshot accuracy | **299 fired, 260 landed (87%)** |
| the productive shot | `ceiling_eye_red` is **52% of all shots at 92.9% land**; `ceiling_wait` lands **7.1%** |
| hit vs miss discriminator | aim-to-eye cosine **0.607 vs 0.270**; first-person aim **0.115 vs 0.359** (associated with MISSING) |
| what an impact buys | 0.70 of 10 HP in the next 3.5 s, but **0% stun across 71 impacts** in `ceiling_idle`/`ceiling_move` |
| larva fighting hurts | r = **−0.598** with damage dealt |
| her ceiling time hurts | r = **−0.559** |
| the recorded reward barely tracks damage | `return ≈ 253 + 8.19 × damage` → **~80% damage-independent** |

## 3. Events are predictable; perception is not the bottleneck (M)

A **linear** probe on the raw 1033-dim observation, held-out episodes, 1 s lookahead:

| event | AUC | base rate |
|---|---|---|
| Gohma about to lose HP | **0.983** | 11–13% |
| Link about to be hurt | **0.965** | 1.4% / 0.3% (~13 positive steps — thin) |

So the information a reward needs is linearly accessible. The failure is credit assignment, not seeing.

## 4. Behaviour cloning: more data barely helps, event weighting helps more

| clone | val accuracy | online damage | defeats |
|---|---|---|---|
| 13 games | 53.4% | 0.14 | 0/80 |
| 100 games | 74.5% | 0.33 | 1/80 |
| **100 games, full-HP starts only** | — | **0.12** | **0/24** |

7.7× the data bought 2.4× damage overall and **nothing on full-HP starts**, which are the actual task. All of
the gain sits in the <200u band (0.46 → 1.08); mid and far range are 0.00 for both.

### Event-stratified BC (80-episode suites, matched)

| arm | scale | classes | damage |
|---|---|---|---|
| uniform (control) | 1.0 | — | **0.16** |
| s1.5 | 1.5 | all | 0.16 |
| s3 | 3 | all | 0.26 |
| s6 | 6 | all | 0.33 |
| **s12** | 12 | all | **0.53** |
| **nolarva** | 3 | no larva | **0.49** |
| chainonly | 3 | chain only | 0.34 |

Monotone in scale from the control (0.16 → 0.53, **3.3×**). Dropping the larva class is worth **quadrupling
the scale**. The chain alone is too narrow — the sword class carries real signal.

Expanded (new weighting: `kill` class added, `sword` 1.0→2.0, selection by sword accuracy): nl-s12 **0.29**,
nl-s24 **0.40**, nl-s48 **0.36**, nl-only **0.29**, all-s24 **0.28**. **Scale saturates between 12 and 24 and
then declines**, killing the log-log extrapolation that implied scale ~4,022 for a kill. Isolating arms:
removing the `kill` class changes nothing (**0.31 vs 0.29**); dropping `sword` to 1.0 gives **0.12**.

Offline metrics vs online damage across five arms: **`acc_sword` r = +0.83**, overall accuracy **r = −0.43**,
`acc_larva` **r = −0.62**. Accuracy on the uniform frame distribution is actively misleading; accuracy on the
sword moment is the only cheap proxy. n = 5 arms — a hint, not a result.

## 5. RvS and the PPO hybrid

RvS v1 (pod-local 219-episode mix): val accuracy on winning play **52.5%**.

First hybrid run (v15 init + PPO + return-conditioned CE + Dyna refresh), 5 ranks, 6 updates:

| | update 1 | update 6 | slope |
|---|---|---|---|
| damage | 3.16 | 1.98 | declining |
| success | 8.2% | 3.8% | **−0.92 ± 0.39 pts/update, t = −2.36** |

It **degrades v15 on both metrics**. Two caveats on whether it was a fair test: a fixed `win-weight 8` put
**83% of the conditioning gradient on wins** (the pod-local corpus was 48.6% wins, not the 5% the multiplier
assumed), so it was closer to anchored BC than RvS; and six updates is far too few — the decay arm needed ~26
before its slope meant anything.

## 6. Retractions

1. **"The decay beats the constant coefficient" (t = +1.92).** Came from comparing the decay arm's updates
   18–25 against the constant arm's 1–8, because `fleet ppo` prints only the last 8 rows without `--last N`.
   At matched updates 1–8: **t = +0.65**, not distinguishable.
2. **"Wins are climbing in the hybrid" (0% → 15%).** That was rank 1 of five. The fleet mean **declines**
   (8.2% → 3.8%, t = −2.36). Quoted one rank two messages after warning against exactly that.

## 7. Infrastructure built because something was lost or slow

- **Checkpoints back themselves up** on every `torch.save` (`zelda_rl/ckpt_s3.py`) — a 34-pod fleet was
  terminated for lack of account funds and took **fifteen arms' weights** with it, because `fleet sync` was
  manual and nobody had run it.
- **Recordings back themselves up per episode** (`record_row`) — the manifest was written once at session end,
  so 29 of 55 human wins had no md5 and no backup.
- **Parallel corpus pull** (`scripts/pull_corpus.py`, 16 workers) — serial boto3 downloads cost ~3 min per pod
  per 600 files; 80 files now take **4 s**.
- **Win share is solved, not guessed** (`rvs_data.solve_win_weight`) — `w = s(1−p)/(p(1−s))` from the corpus,
  because the same 8× that suits a 5%-win corpus puts wins at 83% of the gradient on a 48%-win one.
- **Fresh-data share is continuous** (`--rvs-fresh-share`) — "Dyna on/off" answers a cruder question than
  *how much* of the gradient should come from the policy's own fresh episodes; the fresh windows' weights are
  rescaled to hit the target exactly (verified at 0, 0.25, 0.5, 0.75, 0.9).
- **The evaluator reports damage per minute** and writes per-episode rows; its `--ckpt` branch used to
  `return` before the output sinks were opened, so `--dump-steps` and `--per-episode-out` were silently inert
  for every single-checkpoint run.
