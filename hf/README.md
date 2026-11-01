---
library_name: pytorch
tags:
  - reinforcement-learning
  - ppo
  - imitation-learning
  - world-model
  - game-playing
  - zelda
---

# The Legend of PPO: Learning to Fight in Hyrule

Policies and models from a study of reinforcement learning in *The Legend of Zelda: Ocarina of Time*, played
under an unmodified Dolphin emulator. The agent never sees a pixel: it reads the emulator's memory and
receives every actor's position, velocity, health and decoded animation routine. That removes the
pixel-processing problem, not the perception problem, since nothing in the 1,033-number observation is
labelled.

Two tasks. A three-enemy melee encounter is **solved**. The dungeon boss **Queen Gohma is not**, and the
report is mostly about why: four alternatives were measured against the policy gradient under matched
conditions and all four lost.

| | melee task | Queen Gohma |
|---|---|---|
| best policy | `deku-baba-g150.pt` | `gohma-kl-v15.pt` |
| result | 10 of 10 start states cleared, 0 deaths (one watched episode each) | **8.33%** of 456 matched episodes, 3.53 health of 10 |
| per-start spread | not applicable | **28.95% down to 0 of 76**, by what the start hands the policy |
| cost | $51, 4 h 52 m, 41.8 M steps | part of $1,381 and 509.6 M steps across the project |

- **Code, paper, claim audit and provenance:** https://github.com/JordyKieto/ppo-hyrule
- **Queen Gohma, full fight, unedited:** https://youtu.be/FBD1wEsdY3E
- **Deku Babas, unedited:** https://youtu.be/fMMvd1HmMjI

## Files

| File | Contents |
|---|---|
| `gohma-kl-v15.pt` | The best boss policy, `gohma-kl-v15` rank 4 update 40. Entity-factored recurrent actor-critic, observation v7 (1,033), 108 actions. md5 `397e27b2beb688b9b14d6a260ec07a22` |
| `deku-baba-g150.pt` | The solved melee policy, global update 150. Observation v4 (483), 63 actions. md5 `51cd69f624b8…` |
| `bc-human-100-strat.pt` | Event-stratified behaviour clone of 100 human victories. The best clone we trained, and it takes 0.53 of her 10 health. md5 `6408314168de…` |
| `rvs-m5.pt` | Return-conditioned behaviour cloning, no policy gradient anywhere. 1.76 damage, 8 defeats in 240 episodes. md5 `b4aafda19665…` |
| `sjepa-p1.pt` | Masked-prediction encoder over the raw 363-field state, epoch 12. Beats an untrained network of identical architecture and a free-signal control on 9 of 10 frozen probes, on one seed. md5 `fcad90898b4c…` |
| `wm-obs-epoch120.pt` | The observation-space world model at epoch 120. Included because it is the negative result: it ends further from "nothing changes" than it started. md5 `ba85403916e1…` |

Every file is identified by md5 in `artifacts/MANIFEST.json` in the GitHub repository, because a stray
checkpoint cannot otherwise be matched to the run that produced it. Four world-model checkpoints exist in our
own S3 under the same filenames as different, earlier runs at a third of the size, which is why we match by
hash and not by name.

## The task, measured rather than asserted

Gohma has 10 health, one point per sword hit, and is **only damageable while stunned**. A stun has to be
manufactured. Mined from 100 recorded human victories:

| | |
|---|---|
| damage events requiring a stun | **678 of 678** |
| the engine's own "vulnerable" flag, among those | **9 of 678**, so the obvious flag to reward is almost never set |
| projectile hit converts to a stun, on the ceiling with her eye red | **40.6%** |
| the same, while she is ceiling-idle or moving | **0% across 71 attempts** |
| larvae hit, correlation with damage dealt | **-0.598** |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("jordykieto/ppo-hyrule", "gohma-kl-v15.pt")
ck = torch.load(path, map_location="cpu", weights_only=False)
print(ck["cfg"]["obs_version"], ck["update"])
```

To watch it play, clone the GitHub repository, which carries the environment, the savestates manifest and the
reward tracker, and run `scripts/play.py`. Attaching to emulator memory needs a code-signed interpreter on
macOS; Linux does not.

## What these models do not show

- **No generalisation claim.** The four start states excluded from training are all at half health and all
  beyond 368 units, so "held out" is collinear with "far" and no matched comparison exists.
- **`sjepa-p1.pt` is one seed**, and its own training objective still loses to a trivial control, even though
  the frozen probes clear theirs. Those are two different conclusions.
- **`wm-obs-epoch120.pt` is published as a negative result.** Shuffling the action changes its predictions by
  a factor of 1.00 at the horizon evaluated, so nothing can be planned in it.
- **The imitation models are included because they failed.** A clone reaching 74.5% held-out next-action
  accuracy takes 0.33 of her 10 health and wins 1 of 80, which is the compounding-error problem at a 154
  second horizon rather than a bug.
- The paper's claim audit separates measured results from inferred ones, and lists eleven claims this project
  made internally and then withdrew.

## Citation

```bibtex
@techreport{kieto2026legendofppo,
  title  = {The Legend of PPO: Learning to Fight in Hyrule},
  author = {Kieto, Jordy},
  year   = {2026},
  type   = {Technical Report},
  url    = {https://github.com/JordyKieto/ppo-hyrule}
}
```

## Links

- **Code, paper, claim audit and provenance:** https://github.com/JordyKieto/ppo-hyrule
