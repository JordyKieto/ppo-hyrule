"""Write a Gohma-task (obs v5 / 90 actions) checkpoint warm-started from an obs-v4 combat checkpoint, so the
untrained transfer can be watched with scripts/play.py (or used as --init-from).

    .venv/bin/python scripts/make_transfer_checkpoint.py checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt \
        checkpoints/kept/gohma-init-from-deku-G150.pt
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_combat import GOHMA_DEFAULTS  # noqa: E402
from zelda_rl.combat import COMBAT_DIM, COMBAT_FIELDS, CombatConfig  # noqa: E402
from zelda_rl.env import N_COMBAT_ACTIONS_V5  # noqa: E402
from zelda_rl.gohma import BOSS_DIM, BOSS_FIELDS  # noqa: E402
from zelda_rl.obs import ACTOR_DIM_V2, GLOBAL_DIM_V2, GLOBAL_FIELDS_V2, N_ACTORS  # noqa: E402
from zelda_rl.policy_io import load_policy  # noqa: E402
from zelda_rl.transfer import transfer_state_dict  # noqa: E402

src, dst = sys.argv[1], Path(sys.argv[2])
model, cfg, ck = load_policy(src)
global_dim = GLOBAL_DIM_V2 + COMBAT_DIM + BOSS_DIM
new_cfg = {**cfg, "profile": "gohma", "obs_version": 5, "global_dim": global_dim,
           "obs_dim": global_dim + N_ACTORS * ACTOR_DIM_V2, "n_actions": N_COMBAT_ACTIONS_V5,
           "obs_fields": GLOBAL_FIELDS_V2 + COMBAT_FIELDS + BOSS_FIELDS, "episode_seconds": 300.0,
           "start_slots": "1,2,3,4,5,6,7,8,9,10", "max_seconds": 300.0,
           "combat_cfg": asdict(CombatConfig(profile="gohma", max_seconds=300.0, **GOHMA_DEFAULTS)),
           "transferred_from": {"checkpoint": src, "update": ck.get("update")}}
sd = transfer_state_dict(model, cfg, new_cfg)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": sd, "optimizer": None, "update": ck.get("update", 0), "global_step": ck.get("global_step", 0),
            "episode_count": 0, "config": new_cfg}, dst)
print(f"wrote {dst}: obs {new_cfg['obs_dim']} (global {global_dim}), actions {N_COMBAT_ACTIONS_V5}, from {src} update {ck.get('update')}")
