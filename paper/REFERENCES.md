# References

Verified bibliography for `paper.md`. Every entry below was checked against arXiv's own API, the publisher's
record (Crossref, PMLR, CVF open access, NeurIPS/ICLR proceedings) or the conference's own listing. Entries
copied from `LITERATURE.md` had their arXiv ids re-resolved rather than trusted.

House style follows `../../neural-simulators/paper/paper.tex`'s `thebibliography` block: author initials
first, tied to the surname with `~`, venue spelled out with its acronym, then the arXiv id or DOI. One
deliberate difference from the sibling: titles here are reproduced in the **exact capitalisation the publisher
uses**, because the title string is the field a fabricated citation gets wrong first, and the sibling's
sentence-casing destroys it. `et~al.` is used only past six authors.

`VERIFIED` means the title wording, the author list, the venue and the year were each read off a primary
record. Nothing in the table is from memory.

## How to use this file

`paper.tex`'s `thebibliography` block is **generated** from the table below by `scripts/md_to_tex.py`, which
reads the `authors`, `title` and `venue` columns separately so it can emit the house style's `\newblock`
structure; only rows marked `VERIFIED` are emitted. The build **fails** if `paper.md` cites a key with no row
here, cites a row that is not `VERIFIED`, or if a `VERIFIED` row exists that nothing in the paper cites —
enforced by `tests/test_md_to_tex.py`, so an uncited reference is a build error rather than an untidiness.
Two parsing notes for the generator, because both cells already carry their own terminal punctuation and must
not be given a second one: an `authors` cell ending in `et~al.` and the one `title` cell ending in `?`
(`emmons2022rvs`). A pipe character in any cell breaks the parse; there are none.

---

## Verified entries

| key | authors | title | venue | used for | status |
|---|---|---|---|---|---|
| `agarwal2021precipice` | `R.~Agarwal, M.~Schwarzer, P.~S. Castro, A.~Courville and M.~G. Bellemare` | `Deep Reinforcement Learning at the Edge of the Statistical Precipice` | `In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2021. arXiv:2108.13264` | §9, §14 — few-seed point estimates mislead; interval estimates and robust aggregates | VERIFIED |
| `andrychowicz2021onpolicy` | `M.~Andrychowicz et~al.` | `What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study` | `In \emph{International Conference on Learning Representations (ICLR)}, 2021. arXiv:2006.05990` | §2.3 — on-policy/PPO implementation choices dominating outcomes; the "we ran no head-to-head ablation" caveat | VERIFIED |
| `assran2023ijepa` | `M.~Assran et~al.` | `Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)` | `In \emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2023, pp.~15619--15629. arXiv:2301.08243` | §8 — the masked-prediction objective we implemented, and the first of the two deviations disclosed | VERIFIED |
| `bardes2024vjepa` | `A.~Bardes et~al.` | `Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)` | `\emph{Transactions on Machine Learning Research (TMLR)}, 2024. arXiv:2404.08471` | §8 — the multi-block masking scheme our entity-by-field masking is translated from | VERIFIED |
| `bellemare2013ale` | `M.~G. Bellemare, Y.~Naddaf, J.~Veness and M.~Bowling` | `The Arcade Learning Environment: An Evaluation Platform for General Agents` | `\emph{Journal of Artificial Intelligence Research}, 47:253--279, 2013. arXiv:1207.4708` | §2.1, §14 — emulator-based RL benchmarking, including ALE's RAM observation mode | VERIFIED |
| `burda2019rnd` | `Y.~Burda, H.~Edwards, A.~Storkey and O.~Klimov` | `Exploration by Random Network Distillation (RND)` | `In \emph{International Conference on Learning Representations (ICLR)}, 2019. arXiv:1810.12894` | §4.1, §9.1 — sparse-reward exploration, against which our shaping-inside-1.8-seconds approach is the alternative | VERIFIED |
| `chen2021dt` | `L.~Chen et~al.` | `Decision Transformer: Reinforcement Learning via Sequence Modeling` | `In \emph{Advances in Neural Information Processing Systems (NeurIPS)} 34, 2021. arXiv:2106.01345` | §6.4 — return-conditioned sequence modelling, the family our offline arm belongs to | VERIFIED |
| `cho2014gru` | `K.~Cho et~al.` | `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation` | `In \emph{Conference on Empirical Methods in Natural Language Processing (EMNLP)}, 2014. arXiv:1406.1078` | §2.3 — the gated recurrent unit in the policy's recurrent core | VERIFIED |
| `chung2014gru` | `J.~Chung, C.~Gulcehre, K.~Cho and Y.~Bengio` | `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling` | `NIPS Deep Learning and Representation Learning Workshop, 2014. arXiv:1412.3555` | §2.3 — the standard GRU-as-a-unit citation beside `cho2014gru` | VERIFIED |
| `ecoffet2021goexplore` | `A.~Ecoffet, J.~Huizinga, J.~Lehman, K.~O. Stanley and J.~Clune` | `First Return, Then Explore (Go-Explore)` | `\emph{Nature}, 590:580--586, 2021. DOI 10.1038/s41586-020-03157-9` | §4.1, §9.1 — hard-exploration under a sparse terminal reward | VERIFIED |
| `emmons2022rvs` | `S.~Emmons, B.~Eysenbach, I.~Kostrikov and S.~Levine` | `RvS: What is Essential for Offline RL via Supervised Learning?` | `In \emph{International Conference on Learning Representations (ICLR)}, 2022. arXiv:2112.10751` | §6.4 — the method of our best offline arm, and the source of the "RvS" name used throughout | VERIFIED |
| `fan2022minedojo` | `L.~Fan et~al.` | `MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge` | `In \emph{Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks}, 2022. arXiv:2206.08853` | §14 — positioning only: large-scale open-world game agents | VERIFIED |
| `firoiu2017melee` | `V.~Firoiu, W.~F. Whitney and J.~B. Tenenbaum` | `Beating the World's Best at Super Smash Bros. with Deep Reinforcement Learning` | `\emph{arXiv preprint arXiv:1702.06230}, 2017` | §2.1, §14 — the nearest precedent: structured state, including action state, read from emulator memory | VERIFIED |
| `guss2019minerl` | `W.~H. Guss et~al.` | `MineRL: A Large-Scale Dataset of Minecraft Demonstrations` | `In \emph{International Joint Conference on Artificial Intelligence (IJCAI)}, 2019. arXiv:1907.13440` | §6, §14 — human demonstration corpora at scale, beside our 100 recorded wins | VERIFIED |
| `ha2018worldmodels` | `D.~Ha and J.~Schmidhuber` | `World Models` | `\emph{arXiv preprint arXiv:1803.10122}, 2018` | §7 — learned models used to train policies in imagination | VERIFIED |
| `hafner2020dreamer` | `D.~Hafner, T.~Lillicrap, J.~Ba and M.~Norouzi` | `Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)` | `In \emph{International Conference on Learning Representations (ICLR)}, 2020. arXiv:1912.01603` | §7 — the dream-to-policy loop our nine-iteration closed loop instantiates | VERIFIED |
| `hafner2025dreamerv3` | `D.~Hafner, J.~Pasukonis, J.~Ba and T.~Lillicrap` | `Mastering Diverse Control Tasks through World Models (DreamerV3)` | `\emph{Nature}, 640:647--653, 2025. arXiv:2301.04104` | §7 — the current standard for model-based control, against which our model's skill is reported | VERIFIED |
| `henderson2018matters` | `P.~Henderson, R.~Islam, P.~Bachman, J.~Pineau, D.~Precup and D.~Meger` | `Deep Reinforcement Learning that Matters` | `In \emph{AAAI Conference on Artificial Intelligence (AAAI)}, 32(1):3207--3214, 2018. arXiv:1709.06560` | §9, §14 — environment and seed variance making reported RL results hard to interpret | VERIFIED |
| `kieto2026dyna` | `J.~Kieto` | `Learning in Dreams, Winning in Reality: A Continuous Dyna Loop for a Ten-Hero MOBA` | `\emph{arXiv preprint arXiv:2610.08033}, 2026` | §7, §9.5 — the companion Dyna work whose published rollout depth of 10 we overrode with 600 | VERIFIED |
| `kingma2015adam` | `D.~P. Kingma and J.~Ba` | `Adam: A Method for Stochastic Optimization` | `In \emph{International Conference on Learning Representations (ICLR)}, 2015. arXiv:1412.6980` | §2.3 — the optimiser for every policy and every world model here | VERIFIED |
| `mnih2015dqn` | `V.~Mnih et~al.` | `Human-Level Control through Deep Reinforcement Learning` | `\emph{Nature}, 518:529--533, 2015. DOI 10.1038/nature14236` | §14 — deep RL on emulated console games; **cite this version**, not the 2013 workshop paper, unless the point is specifically the original NIPS workshop result | VERIFIED |
| `ng1999shaping` | `A.~Y. Ng, D.~Harada and S.~Russell` | `Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping` | `In \emph{International Conference on Machine Learning (ICML)}, 1999, pp.~278--287` | §2.2, §9.1, §14 — potential-based shaping is necessary and sufficient for policy invariance; ours is not potential-based | VERIFIED |
| `nichol2018gymretro` | `A.~Nichol, V.~Pfau, C.~Hesse, O.~Klimov and J.~Schulman` | `Gotta Learn Fast: A New Benchmark for Generalization in RL` | `\emph{arXiv preprint arXiv:1804.03720}, 2018` | §2.1, §14 — the Gym Retro technical report; emulated console games as an RL benchmark | VERIFIED |
| `openai2019dota` | `OpenAI: C.~Berner et~al.` | `Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)` | `\emph{arXiv preprint arXiv:1912.06680}, 2019` | §14 — positioning only: large-scale PPO on a complex commercial game | VERIFIED |
| `pleines2022recurrentppo` | `M.~Pleines, M.~Pallasch, F.~Zimmer and M.~Preuss` | `Generalization, Mayhems and Limits in Recurrent Proximal Policy Optimization` | `\emph{arXiv preprint arXiv:2205.11104}, 2022` | §2.3 — recurrent PPO in practice: truncated BPTT, hidden-state handling, the choices our recurrent core makes silently | VERIFIED |
| `ross2010reductions` | `S.~Ross and J.~A. Bagnell` | `Efficient Reductions for Imitation Learning` | `In \emph{International Conference on Artificial Intelligence and Statistics (AISTATS)}, PMLR 9:661--668, 2010` | §6.1 — the earlier reduction establishing the error-compounding bound for pure behaviour cloning | VERIFIED |
| `ross2011dagger` | `S.~Ross, G.~J. Gordon and J.~A. Bagnell` | `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)` | `In \emph{International Conference on Artificial Intelligence and Statistics (AISTATS)}, PMLR 15:627--635, 2011. arXiv:1011.0686` | §6.1, §14 — regret quadratic in the horizon for cloning, linear for DAgger; our 81%-death clone is an instance | VERIFIED |
| `schrittwieser2020muzero` | `J.~Schrittwieser et~al.` | `Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)` | `\emph{Nature}, 588:604--609, 2020. arXiv:1911.08265` | §7 — planning in a learned model whose value is exactly the action-difference our shuffle control found absent | VERIFIED |
| `schulman2016gae` | `J.~Schulman, P.~Moritz, S.~Levine, M.~Jordan and P.~Abbeel` | `High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)` | `In \emph{International Conference on Learning Representations (ICLR)}, 2016. arXiv:1506.02438` | §4.1 — the `1/(1 - gamma*lambda)` advantage horizon the whole section's arithmetic rests on | VERIFIED |
| `schulman2017ppo` | `J.~Schulman, F.~Wolski, P.~Dhariwal, A.~Radford and O.~Klimov` | `Proximal Policy Optimization Algorithms (PPO)` | `\emph{arXiv preprint arXiv:1707.06347}, 2017` | §2.3 and throughout — the optimiser for every policy in this report | VERIFIED |
| `surazhsky2026copyframe` | `M.~Surazhsky` | `Your World Model May Be Worse Than Copying the Last Frame` | `NeurIPS 2026 Workshop on Robot Learning with World Models, 2026. \url{https://neurips.cc/virtual/2026/166122}` | §7, §14 — the copy-last-frame floor and the survey finding that 9 of 12 world-model papers report nothing to read their number against | VERIFIED |
| `sutton1991dyna` | `R.~S. Sutton` | `Dyna, an Integrated Architecture for Learning, Planning, and Reacting` | `\emph{ACM SIGART Bulletin}, 2(4):160--163, 1991. DOI 10.1145/122344.122377` | §7 — the original interleaving of real experience with model-generated experience, which our closed loop implements | VERIFIED |
| `villegas2017mcnet` | `R.~Villegas, J.~Yang, S.~Hong, X.~Lin and H.~Lee` | `Decomposing Motion and Content for Natural Video Sequence Prediction` | `In \emph{International Conference on Learning Representations (ICLR)}, 2017. arXiv:1706.08033` | §7 — copy-last-frame beating a learned model on UCF101 | VERIFIED |
| `villegas2019highfidelity` | `R.~Villegas, A.~Pathak, H.~Kannan, D.~Erhan, Q.~V. Le and H.~Lee` | `High Fidelity Video Prediction with Large Stochastic Recurrent Neural Networks` | `In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2019. arXiv:1911.01655` | §7 — copy-last-frame outperforming every model on Human 3.6M, and per-frame metrics declared unreliable there | VERIFIED |
| `vinyals2019alphastar` | `O.~Vinyals et~al.` | `Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning (AlphaStar)` | `\emph{Nature}, 575:350--354, 2019. DOI 10.1038/s41586-019-1724-z` | §14 — positioning only: large-scale agents on a commercial game, trained from human replays plus RL | VERIFIED |
| `weihs2021advisor` | `L.~Weihs et~al.` | `Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR)` | `In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2021. arXiv:2007.12173` | §6.3, §14 — the imitation gap; the per-state weighting we did not try | VERIFIED |
| `wu2023mtm` | `P.~Wu et~al.` | `Masked Trajectory Models for Prediction, Representation, and Control (MTM)` | `In \emph{International Conference on Machine Learning (ICML)}, 2023. arXiv:2305.02968` | §8, §14 — masked prediction over decision data; the reason we claim the protocol and not the objective | VERIFIED |

38 entries, all verified.

---

## Where two versions exist, and which to cite

- **DQN.** `mnih2015dqn` (Nature 518) is the version to cite. The NIPS 2013 workshop version had a row here
  and was **removed on 2026-11-01**: no sentence in this paper is about the original workshop result
  specifically, and citing both on one clause is padding.
- **Andrychowicz et al.** The arXiv preprint is titled *"What Matters In On-Policy Reinforcement Learning? A
  Large-Scale Empirical Study"*; the **ICLR 2021 paper was retitled** *"What Matters for On-Policy Deep
  Actor-Critic Methods? A Large-Scale Study"*. The table carries the published title. If the arXiv id is
  printed beside it, the title mismatch is expected and is not an error.
- **V-JEPA.** Published in **TMLR 2024**, and also listed as an **ICLR 2025** poster. Cite the TMLR version:
  it is the archival record and matches the arXiv text we implemented against.
- **Dreamer.** `hafner2020dreamer` (ICLR 2020) is the right cite for "train a policy inside a learned model";
  `hafner2025dreamerv3` (Nature 640) is the right cite for current model-based performance. The paper's §7
  needs the first; a positioning sentence needs the second. These are the two standard ones — PlaNet,
  DreamerV2 and Dreamer 4 are not needed here.
- **Ross.** `ross2010reductions` (AISTATS 2010) is the compounding-error bound; `ross2011dagger` (AISTATS
  2011) is DAgger. §6.1 as written argues the bound, so it needs both, and the 2010 paper is the one that
  carries the quadratic-regret result for *cloning*.
- **Sutton's Dyna.** The SIGART Bulletin version above is the one with a DOI. Sutton's own publication list
  also attributes it to AAAI 1991 at the same pagination; the SIGART record is the citable one.
- **Go-Explore.** Cite the **Nature 2021** paper (`First Return, Then Explore`). The 2019 arXiv preprint
  (arXiv:1901.10995) has a different title and its own page explicitly redirects readers to the later version,
  so the preprint is not the right record.

---

## Dropped, and why

- **Gym Retro as a standalone technical report.** There is no separate "Gym Retro" paper. The citable record
  is `nichol2018gymretro` (*Gotta Learn Fast*, arXiv:1804.03720), which introduces the benchmark built on
  Gym Retro. Anything citing "Gym Retro (2018)" as its own report would be fabricated; the entry above is
  the honest substitute, and the paper's prose should name *Gotta Learn Fast*, not "the Gym Retro report".
- **Babaeizadeh et al., arXiv:1911.01655.** Dropped **as attributed**, because the attribution is wrong and
  is wrong in the paper — see the note below. The work itself is kept as `villegas2019highfidelity`.
- **"The 37 Implementation Details of PPO" / CleanRL.** Not included. The ICLR 2022 Blog Track post has no
  archival identifier I could resolve, and the CleanRL paper (arXiv:2111.08819) is a software paper whose
  journal record I could not confirm today. `andrychowicz2021onpolicy` and `pleines2022recurrentppo` cover
  the same ground with records that resolve.
- **A DOI for Ng, Harada and Russell (1999).** ICML 1999 predates PMLR and the proceedings have no DOI I
  could resolve. The field is omitted rather than guessed.
- **MuZero DOI vs arXiv.** Both exist (DOI 10.1038/s41586-020-03051-4); the entry prints the arXiv id to
  match the sibling paper's convention for the Nature-published DeepMind results.
- Nothing was dropped for an unresolvable arXiv id. **Every arXiv id in this file resolved through
  `export.arxiv.org/api/query` today**, including `2610.08033`, which does resolve and is the author's own
  Dyna paper.

---

## Unverified

Empty. No entry was carried into this file that could not be confirmed against a primary record. If an
entry is added later and cannot be confirmed, it belongs in this section and not in the table above.

---

## Three places the paper's prose is inaccurate about a work it cites

These matter more than the count, and all three are in text already written.

1. **Firoiu et al.'s title does not contain the word "Melee".** The exact title is *"Beating the World's Best
   at Super Smash Bros. with Deep Reinforcement Learning"*. `LITERATURE.md` §1 prints it as *"…at Super Smash
   Bros. Melee with Deep Reinforcement Learning"*, which is a fabricated title string. The paper's §2.1 and
   §14 describe the game as *Super Smash Bros. Melee*, which is correct as a statement about the game and
   must not be folded into the title.
2. **arXiv:1911.01655 is not Babaeizadeh et al.** It is **Villegas, Pathak, Kannan, Erhan, Le and Lee**,
   *High Fidelity Video Prediction with Large Stochastic Recurrent Neural Networks*, NeurIPS 2019. Both
   `paper.md` §7 ("Babaeizadeh et al., arXiv:1911.01655; Villegas et al., arXiv:1706.08033") and
   `LITERATURE.md` §5 attribute it to Babaeizadeh. The id and the finding are right; the author is wrong, and
   as printed it reads as two different groups when both papers are Villegas-led.
3. **The copy-last-frame paper is a NeurIPS 2026 *workshop* poster, not a NeurIPS 2026 paper.** It is a
   poster in the *Robot Learning with World Models: Capabilities, Frontiers, and Challenges* workshop, single
   author, OpenReview id `iF1NJvEfwS`. `paper.md` §7 and §14 both call it "a NeurIPS 2026 paper", and §14
   leans on its survey numbers. The venue should say workshop, and the existing "we have not read the paper"
   caveat should stay attached.

One smaller correction: Ng, Harada and Russell's title has a subtitle — *"Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping"*. `LITERATURE.md` §3 prints only the first half.
