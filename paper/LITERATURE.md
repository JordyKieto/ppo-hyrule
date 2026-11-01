# Literature and positioning

Verified citations for `paper.md`, with what each establishes and how our result relates. Follows the format
of `../../neural-simulators/paper/LITERATURE.md`.

**Verification depth, stated honestly.** Every entry below was checked by web search against at least two
independent sources for authors, venue, year and central claim: typically the arXiv abstract page plus the
publisher's proceedings entry. **We did not read every paper in full for this file**, so the summaries are of
the abstract and the widely reported central result, not of the experiments. Entries where that distinction
could matter are marked. Anything we could not verify to that standard is not here, because an unverified
reference is worse than a missing one.

**Status: partial.** Written on the main thread after a delegated literature pass produced nothing in 35
minutes. The clusters below cover the five citations the paper cannot ship without. Clusters still missing
are named at the end.

---

## 1. RL in commercial games from engine state rather than pixels

**Firoiu, Whitney and Tenenbaum (2017), "Beating the World's Best at Super Smash Bros. Melee with Deep
Reinforcement Learning", arXiv:1702.06230.** Trains agents for a 2001 Nintendo GameCube fighting game under
emulation. Rather than pixels, it reads features from the game's memory each frame: **each player's position,
velocity, and action state**. Reports agents competitive with and in some cases beating human professionals,
and notes a weakness to specific exploitative strategies such as edge-crouching.

> **Relation, and this is the nearest precedent to our work by some distance.** Same emulator family, same
> commercial console game, same decision to read structured engine state including the *action state* rather
> than render pixels, same comparison against a human expert. Our observation is a direct descendant of
> theirs. What differs is the task structure rather than the method: Melee is a reactive two-player fighting
> game, where our boss encounter is a single-agent chain of manufactured openings over 45 to 154 seconds, and
> our interest is in what fails to improve a policy on it. **We must not claim novelty for observing decoded
> emulator memory.** The defensible line is that we apply it to a 3D action-adventure boss fight and report
> the decoded enemy action routine as the observation that makes the task legible.

Also in this cluster, not individually verified for this file and therefore cited only as background: ALE's
RAM observation mode, Gym Retro, and community work rewarding progress read from Pokemon Red's memory.

## 2. Imitation that hurts, and why

**Ross, Gordon and Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online
Learning" (DAgger), AISTATS 2011, arXiv:1011.0686.** Establishes the compounding-error problem: behaviour
cloning assumes training states are i.i.d., but the learner's own mistakes carry it into states the expert
never demonstrated, and errors compound. Naive cloning has regret growing **quadratically** in the horizon;
DAgger reduces imitation to no-regret online learning and achieves linear growth.

> **Relation.** This is the textbook explanation of our section 6.1, and we should present our numbers as an
> instance of it rather than as a discovery. Our contribution is the magnitude at a long horizon with a
> strong demonstrator: 74.5% held-out next-action accuracy converts to 0.33 of 10 health and 1 win in 80, and
> 81% of the clone's episodes end in death.

**Weihs, Jain et al., "Bridging the Imitation Gap by Adaptive Insubordination" (ADVISOR), NeurIPS 2021,
arXiv:2007.12173.** Names the **imitation gap**: when a teacher acts on information or capability the student
does not have, imitating it is actively harmful, and pure imitation can be beaten by pure RL. ADVISOR weights
the imitation and RL losses per state, using an auxiliary actor that checks whether the student can reproduce
the expert's action from its own perspective.

> **Relation, and it reframes our strongest negative result.** Our fifteen demonstration-injection arms are an
> instance of the imitation gap, and the paper should say so. Two differences are worth stating. First, their
> canonical case is a *privileged* teacher; ours sees exactly the same observation the policy does, so the gap
> here is capability and strategy rather than information. Second, the usual framing has the demonstrator
> helping a weak learner; in our case the demonstrator is far better at the task (100 of 100 against 8.33%)
> and the demonstrations still degrade the policy, with the term that provably worked mechanically, agreement
> 3% to 23%, costing damage at t = -4.98. ADVISOR's per-state weighting is also the obvious thing we did not
> try.

## 3. Reward shaping and its pathologies

**Ng, Harada and Russell, "Policy Invariance Under Reward Transformations", ICML 1999, pp. 278-287.** Shows
that a shaping term of the form `F(s, s') = gamma * Phi(s') - Phi(s)`, a difference of a potential over
states, is both necessary and sufficient to leave the optimal policy unchanged under arbitrary dynamics. Any
other additive shaping can change what is optimal.

> **Relation.** Our shaping terms are **not** potential-based, and this is the theory that says what we then
> observed empirically: a shaping cluster grew to 88% of total reward magnitude, the policy optimised it
> correctly, driving red-eye aim to 3.36 times chance, and still died 64.5% of the time with the boss at 8 of
> 10 health. That is Ng et al.'s failure mode with a measured price tag. The paper should cite it where it
> describes removing 19 terms, and should state plainly that the reward we ran is not policy-invariant by
> construction.

## 4. Masked prediction over trajectories and structured state

**Wu, Majumdar, Stone, Lin, Mordatch, Abbeel and Rajeswaran, "Masked Trajectory Models for Prediction,
Representation, and Control" (MTM), ICML 2023, PMLR 202:37607-37623, arXiv:2305.02968.** One network trained
to reconstruct a trajectory from random subsets of itself. Because the masking is randomised, one set of
weights serves as a forward model, an inverse model or an offline RL agent depending on the mask at inference.
Representations speed up conventional RL; competitive with dedicated offline RL methods.

> **Relation, and it narrows what section 8 may claim.** MTM is the nearest prior work and it already masks
> over trajectories of states and actions. **We cannot claim the objective.** What is ours is (a) the masking
> *structure*, over an entity-by-field grid of raw engine fields, with whole-actor tubes, whole-field-group
> tubes and time blocks rather than random subsets, and (b) the **free-signal floor**, a ridge fit on the
> cells still visible after masking. For a state vector where speed is the norm of velocity and position is
> the previous position plus velocity, a masked cell is often exact arithmetic on a visible one, and we know
> of no prior work reporting that floor. Claim the protocol, not the method.

**I-JEPA (arXiv:2301.08243) and V-JEPA (arXiv:2404.08471)** are the objective we implemented, and the paper
already discloses the two places our implementation departs from them.

## 5. World models and the persistence baseline

**"Your world model may be worse than copying the last frame", NeurIPS 2026** (listing:
`neurips.cc/virtual/2026/166122`). Reports that repeating the previous frame explained 99.6% of the same
targets against 97.9% for a learned model, and that **of 20 recent world-model papers, 12 report a
prediction-quality number and 9 of those report nothing to read it against.** Proposes a normalised lift
between the copy-last-frame floor and a ceiling. *Verified by conference listing and search summary; we have
not read the paper.*

**Context from video prediction:** the copy-last-frame baseline is long established as a floor and is known to
beat learned models on mostly static data. Villegas, Pathak, Kannan, Erhan, Le and Lee, "High Fidelity Video Prediction with Large
Stochastic Recurrent Neural Networks" (arXiv:1911.01655), report it outperforming all their models on Human
3.6M and conclude per-frame metrics are unreliable there. Villegas et al. (arXiv:1706.08033) report it beating
their model on UCF101 and omit it on KITTI where most pixels move. The `vp-suite` library ships a
`CopyLastFrame` model for exactly this purpose.

> **Relation, and it changes our framing from discovery to confirmation.** Our persistence baseline is the
> same instrument, and the NeurIPS 2026 result says the field is only now standardising it. **We should cite
> this and present our world-model section as confirming a known and under-reported failure mode**, not as
> finding it. What remains ours is the sharper version: our model is **zero-initialised so that it begins
> exactly at persistence**, and 120 epochs of training then move it steadily away at -0.0874 ± 0.0218 per
> epoch; and we carry the failure through to a closed loop where the dream-trained policies win 0 of 607 real
> episodes. Starting at the floor and walking away from it is a stronger statement than failing to reach it.

## 6. Evaluation methodology in deep RL

**Agarwal, Schwarzer, Castro, Courville and Bellemare, "Deep Reinforcement Learning at the Edge of the
Statistical Precipice", NeurIPS 2021 (Outstanding Paper), arXiv:2108.13264.** Argues that point estimates of
mean or median across tasks, from few runs, ignore run-to-run uncertainty, and shows this changes conclusions
on Atari 100k. Recommends stratified bootstrap interval estimates, performance profiles, and robust aggregates
such as the interquartile mean. Ships `rliable`.

**Henderson, Islam, Bachman, Pineau, Precup and Meger, "Deep Reinforcement Learning That Matters", AAAI 2018,
32(1):3207-3214, arXiv:1709.06560.** Reproducibility study: environment non-determinism plus intrinsic method
variance makes reported results hard to interpret; shows how metrics shift against common baselines; proposes
reporting guidelines.

> **Relation, and this is where we must be most careful not to claim what is already known.** Section 9 of our
> paper is largely a **confirmation** of these two, at unusual resolution and with costs attached:
>
> | our finding | status against this literature |
> |---|---|
> | few seeds give unreliable conclusions; our four measured noise floors, including two groups of three identical runs differing at t = 2.1 | **known.** Henderson 2018, Agarwal 2021. Cite, do not claim |
> | point estimates without intervals mislead | **known.** Agarwal 2021 |
> | maximum of N noisy arms is biased upward; our 12-cell grid's +32% collapsing to +0.065 ± 0.117 | **known** in general, and the concrete worked example with a confirmation run is a useful instance |
> | a selector needs enough n to resolve what it selects on; our 16-episode subsets at a 1.68% base rate | **known in principle**, rarely shown with the arithmetic of the null and the decision it corrupted |
> | **a metric averaged over a start distribution the experimenter controls is not a property of the policy**, and our mix-invariant form | **we have not found this stated this way.** Closest in spirit to Agarwal's aggregation critique, but that is across *tasks* in a fixed benchmark, where ours is across *initial conditions whose weights we choose and also optimise under*. Candidate contribution |
> | **a sparse binary outcome hid an 8.7x improvement that a graded metric showed**, and the binary then motivated two harmful reward changes | **we have not found this measured end to end.** Candidate contribution |
> | **an episode cap below the task's own duration censored half the successes** | mundane and, we think, unreported. Worth one sentence |

---

## Novelty assessment, with recommended wording

| claim | verdict | safe wording |
|---|---|---|
| RL from decoded emulator memory | **NOT novel.** Firoiu et al. 2017 | "following Firoiu et al.'s approach of reading structured state from emulator memory" |
| Applying it to a 3D Zelda boss fight with decoded enemy action routines | **Novel as a setting**, not as a method | "to our knowledge the first agent for this encounter; the observation approach is not new" |
| Behaviour cloning fails in closed loop despite high frame accuracy | **NOT novel.** Ross et al. 2011 | "an instance of the compounding-error problem, at a magnitude worth recording" |
| Demonstrations from a strong expert degrade a competent policy | **PARTIALLY novel.** Weihs et al. 2021 names the effect; the reversed-competence case is less reported | "an instance of the imitation gap, with the unusual feature that the demonstrator is far stronger than the learner" |
| Shaping dominating and misdirecting the objective | **NOT novel.** Ng et al. 1999 | "the non-potential-based shaping failure mode, with a measured price" |
| World models losing to persistence | **NOT novel**, and newly systematised by the NeurIPS 2026 copy-last-frame paper | "confirming a known and under-reported failure mode" |
| A model that *starts* at persistence and trains away from it, plus the action-shuffle control and the closed loop to 0 of 607 | **Novel as a measurement** | claim this, narrowly |
| Masked prediction over structured state | **NOT novel.** MTM 2023 | "following MTM's masked-trajectory objective" |
| Entity-by-field structured masking plus a free-signal floor | **Novel as a protocol** | claim this |
| Seeds, intervals, winner's curse, selector resolution | **NOT novel.** Agarwal 2021, Henderson 2018 | "confirming, with costs attached" |
| **Mix-invariance under an experimenter-controlled start distribution** | **Candidate novel** | claim cautiously, invite correction |
| **A sparse binary hiding an 8.7x improvement and then motivating harmful changes** | **Candidate novel as an end-to-end case** | claim cautiously |
| A matched head-to-head of five method families on one task, one harness, one cap, one graded metric | **Novel as breadth**, not as any single arm | claim this |

---

## Still missing from this file

Named so the gap is visible rather than implied. None of these are cited in `paper.md` yet:

- PPO itself (Schulman et al. 2017) and GAE (Schulman et al. 2015), which the paper leans on arithmetically.
- Large-scale PPO in complex games: OpenAI Five, AlphaStar, Capture the Flag, GT Sophy.
- Sparse-reward exploration: Montezuma's Revenge, Go-Explore, RND.
- Demonstration-augmented RL specifics: DQfD, DAPG, AWAC, RLPD, and the Q-filter we reimplemented.
- Offline RL and return conditioning: Decision Transformer, RvS (Emmons et al.), Trajectory Transformer.
- The Dreamer line, MuZero, TD-MPC, IRIS, DIAMOND.
- MAE and BERT as the masking ancestors.
- Colas et al. on how many random seeds.
- The author's own companion work: `neural-simulators` (which uses the same copy baseline) and the Dyna paper
  whose rollout depth of 10 we overrode with 600.

## Sources

- [Firoiu, Whitney, Tenenbaum 2017, arXiv:1702.06230](https://arxiv.org/abs/1702.06230)
- [Ross, Gordon, Bagnell, DAgger, arXiv:1011.0686](https://arxiv.org/abs/1011.0686) and [PMLR](https://proceedings.mlr.press/v9/ross10a.html)
- [Weihs et al., ADVISOR, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9fc664916bce863561527f06a96f5ff3-Abstract.html), [arXiv:2007.12173](https://arxiv.org/abs/2007.12173)
- [Ng, Harada, Russell, ICML 1999](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)
- [Wu et al., MTM, ICML 2023](https://icml.cc/virtual/2023/poster/23703), [arXiv:2305.02968](https://arxiv.org/abs/2305.02968)
- [Agarwal et al., Statistical Precipice, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf), [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- [Henderson et al., Deep RL That Matters, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694), [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- [Your world model may be worse than copying the last frame, NeurIPS 2026](https://neurips.cc/virtual/2026/166122)
- [Villegas et al., arXiv:1911.01655](https://arxiv.org/pdf/1911.01655) and [Villegas et al., arXiv:1706.08033](https://arxiv.org/pdf/1706.08033)
