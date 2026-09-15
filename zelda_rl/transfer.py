"""Warm-start a policy for a larger observation / action space from an existing checkpoint (e.g. the Deku Baba
combat policy, obs v4 / 63 actions -> the boss-capable obs v5 / 81 actions).

Function-preserving surgery on RecurrentEntityActorCritic / EntityActorCritic:
- Observation: new global features are inserted at the END of the global block (before the pooled actor-table
  features). The first trunk layer gets zero-initialised columns for them, so the transferred network computes
  exactly what it did before until training gives the new inputs weight.
- Actor table: new per-actor fields are inserted right before the trailing `present` flag; the actor encoder's first
  layer gets zero columns for them (so pooled features, hence the whole network, are unchanged).
- Actions: new actions are appended after the old ones. Their policy-head rows are zero and their bias is set a
  fixed margin below the old mean logit, so the old behaviour is kept and the new actions start rarely but not never.
Everything else (actor encoder, id embedding, GRU, critic) is copied unchanged.
"""
from __future__ import annotations

import torch

from zelda_rl.ppo import build_model


def _widen_input(w: torch.Tensor, old_global: int, n_new: int) -> torch.Tensor:
    """Linear weight (out, in): insert n_new zero columns right after the first old_global inputs."""
    zeros = torch.zeros(w.shape[0], n_new, dtype=w.dtype)
    return torch.cat([w[:, :old_global], zeros, w[:, old_global:]], 1)


def transfer_state_dict(old_model, old_cfg: dict, new_cfg: dict, new_action_logit_margin: float = 2.0) -> dict:
    new_model = build_model(new_cfg)
    old_sd, new_sd = old_model.state_dict(), new_model.state_dict()
    n_new_obs = new_cfg["global_dim"] - old_cfg["global_dim"]
    n_new_act = new_cfg["n_actions"] - old_cfg["n_actions"]
    assert n_new_obs >= 0 and n_new_act >= 0, "transfer only widens"
    recurrent = bool(new_cfg.get("recurrent"))
    assert old_cfg["n_actors"] == new_cfg["n_actors"], "actor count must match"
    n_new_actor = new_cfg["actor_dim"] - old_cfg["actor_dim"]
    assert n_new_actor >= 0, "transfer only widens"
    enc = "enc." if recurrent else ""
    actor_first = f"{enc}actor_enc.0.weight"
    first_layers = ["proj.0.weight"] if recurrent else ["actor.0.weight", "critic.0.weight"]
    head = "pi.2" if recurrent else "actor.4"
    out = {}
    for k, v in new_sd.items():
        src = old_sd[k]
        if k in first_layers:
            out[k] = _widen_input(src, old_cfg["global_dim"], n_new_obs)
        elif k == actor_first and n_new_actor:
            id_emb = old_sd[f"{enc}id_embedding.weight"].shape[1]
            out[k] = _widen_input(src, id_emb + old_cfg["actor_dim"] - 2, n_new_actor)   # before `present`
        elif k == f"{head}.weight":
            out[k] = torch.cat([src, torch.zeros(n_new_act, src.shape[1], dtype=src.dtype)], 0)
        elif k == f"{head}.bias":
            fill = src.mean() - new_action_logit_margin
            out[k] = torch.cat([src, torch.full((n_new_act,), float(fill), dtype=src.dtype)], 0)
        else:
            out[k] = src
        assert out[k].shape == v.shape, (k, tuple(out[k].shape), tuple(v.shape))
    return out


def transfer_model(old_model, old_cfg: dict, new_cfg: dict, **kw):
    model = build_model(new_cfg)
    model.load_state_dict(transfer_state_dict(old_model, old_cfg, new_cfg, **kw))
    return model
