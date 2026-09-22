"""Targets, weights and the loss for the reward-only decode heads, in ONE place.

Stage 1 (`scripts/train_encoder_reward.py`) trains an encoder to reconstruct these; stage 2
(`scripts/train_jepa_stage2.py`) pushes the PREDICTED latent to decode to them. If those two computed the loss
separately they would drift, and this project has already paid for that kind of duplication twice -- five
scripts each rebuilt the predictor by hand (hence `zelda_rl/dynamics.py`), and a checkpoint once saved a
reference `Norm` instead of the one used for training, silently mis-normalising every consumer.

The fields are exactly what `CombatTracker.step` reads; see `docs/JEPA_PLAN.md` for the field-to-reward-term
table and `bc.REWARD_GLOBAL` / `bc.REWARD_ROW` for the list.

Conventions kept from what was already measured here, not reinvented:
  * HP is CLASSIFIED, never regressed (damage F1 0.286 -> 0.906, LATENT_WORLD_MODEL.md section 3).
  * Class weights are sqrt-inverse-frequency; a raw inverse gave the rare class weight 0.0005 and it was
    simply never predicted.
  * HP-transition frames are oversampled 20x, because damage detection lives on that minority of frames.
  * `link_facing_target` is masked to frames where the Baba is OUT of the ground: 64.7% of Baba frames are
    `hidden`, where the angle is near-uniform (R=0.12), and fitting it there swamped the 35% that carries
    signal (R=0.84-0.99).
  * Real-valued fields are standardised before the MSE so one head in raw world units cannot swamp the rest.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from zelda_rl.bc import (LINK_HP_VALUES, N_BABA_STATES, REWARD_GLOBAL, REWARD_ROW, ROW_HP_VALUES,
                         decode_targets, hp_class, transition_frames)

REAL_FIELDS = ("link_pos", "near_dist", "row_pos", "row_dist")


class RewardTargets:
    """Everything the reward-field decode loss needs, built once.

    `scale` reuses a checkpoint's stored standardisation instead of refitting -- stage 2 must score against
    the exact scale stage 1's decoder was trained under.
    """

    def __init__(self, state, obs, epid, train_mask=None, scale: dict | None = None):
        Y = decode_targets(state, obs, epid=epid)
        Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
        Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
        self.scale = {}
        for k in REAL_FIELDS:
            if scale is not None and k in scale:
                mu, sd = np.asarray(scale[k][0]), np.asarray(scale[k][1])
            else:
                v = Y[k].reshape(-1, Y[k].shape[-1])
                ok = np.isfinite(v).all(1)
                mu, sd = v[ok].mean(0), v[ok].std(0) + 1e-6
            Y[k] = ((Y[k] - mu) / sd).astype(np.float32)
            self.scale[k] = (np.asarray(mu).tolist(), np.asarray(sd).tolist())

        from zelda_rl.combat import BABA_STATES
        hidden = BABA_STATES.index("hidden") if "hidden" in BABA_STATES else 0
        keep = set(REWARD_GLOBAL) | set(REWARD_ROW) | {"link_hp_c", "row_hp_c", "row_state"}
        self.Y = {k: torch.from_numpy(np.asarray(v)) for k, v in Y.items() if k in keep}
        self.VIS = torch.from_numpy((Y["row_state"] != hidden).astype(np.float32))

        pres = Y["row_present"] > 0.5
        self.CW = {"link_hp": _cw(Y["link_hp_c"], len(LINK_HP_VALUES)),
                   "row_hp": _cw(Y["row_hp_c"], len(ROW_HP_VALUES), pres)}
        self.wl = torch.from_numpy(1 + 20.0 * transition_frames(Y["link_hp_c"], epid).astype(np.float32))
        self.wr = torch.from_numpy(1 + 20.0 * transition_frames(Y["row_hp_c"], epid).astype(np.float32))

    def loss(self, p: dict, i: torch.Tensor, w_pos: float = 6.0, skip: tuple = ()) -> torch.Tensor:
        """Reconstruction loss for decoder output `p` against the reward fields at frame indices `i`.

        `skip` drops named terms, so a caller that represents a field differently (e.g. position as a nav-grid
        cell + offset rather than a regressed coordinate) can own that field without reimplementing the rest.
        """
        Y, m, mb = self.Y, self.Y["row_present"][i], self.Y["row_is_baba"][i]
        L = (F.mse_loss(p["link_yaw"], Y["link_yaw"][i])
             + F.mse_loss(p["near_dist"], Y["near_dist"][i])
             + (0.0 if "link_pos" in skip else w_pos * F.mse_loss(p["link_pos"], Y["link_pos"][i]))
             + F.binary_cross_entropy_with_logits(p["link_bits"], Y["link_bits"][i])
             + F.cross_entropy(p["targeted"], Y["targeted"][i])
             + F.cross_entropy(p["scene"], Y["scene"][i])
             + F.cross_entropy(p["room"], Y["room"][i])
             + F.cross_entropy(p["baba_count"], Y["baba_count"][i])
             + F.binary_cross_entropy_with_logits(p["row_present"], m)
             + F.binary_cross_entropy_with_logits(p["row_is_baba"], mb)
             + (F.cross_entropy(p["link_hp"], Y["link_hp_c"][i], weight=self.CW["link_hp"],
                                reduction="none") * self.wl[i]).mean())
        rh = (F.cross_entropy(p["row_hp"].reshape(-1, len(ROW_HP_VALUES)), Y["row_hp_c"][i].reshape(-1),
                              weight=self.CW["row_hp"], reduction="none").reshape(m.shape) * m * self.wr[i])
        L = L + rh.sum() / m.sum().clamp(min=1)
        if "row_pos" not in skip:
            L = L + w_pos * (((p["row_pos"] - Y["row_pos"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        L = L + (((p["row_dist"] - Y["row_dist"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        ce = F.cross_entropy(p["row_state"].reshape(-1, N_BABA_STATES), Y["row_state"][i].reshape(-1),
                             reduction="none").reshape(m.shape)
        L = L + (ce * mb).sum() / mb.sum().clamp(min=1)
        vis = mb * self.VIS[i]
        L = L + (((p["link_facing_target"] - Y["link_facing_target"][i]) ** 2).mean(-1)
                 * vis).sum() / vis.sum().clamp(min=1)
        return L

    def report(self, dp: dict, dr: dict | None, i: torch.Tensor) -> list[tuple]:
        """Per-field accuracy / world-unit MAE for a decoded prediction, optionally beside a decoded real
        latent. Returns rows of (name, predicted, real_or_nan, metric_kind)."""
        Y = self.Y
        m, mb = Y["row_present"][i] > 0.5, Y["row_is_baba"][i] > 0.5
        out = []
        for name, key, msk in (("link_hp", "link_hp_c", None), ("targeted", "targeted", None),
                               ("scene", "scene", None), ("baba_count", "baba_count", None),
                               ("row_hp", "row_hp_c", m), ("row_state", "row_state", mb)):
            lab = Y[key][i]
            pc = dp[name].argmax(-1)
            rc = dr[name].argmax(-1) if dr is not None else None
            if msk is not None:
                lab, pc = lab[msk], pc[msk]
                rc = rc[msk] if rc is not None else None
            out.append((name, float((pc == lab).float().mean()),
                        float((rc == lab).float().mean()) if rc is not None else float("nan"), "acc"))
        for name, msk in (("link_pos", None), ("near_dist", None), ("row_pos", m), ("row_dist", m)):
            sd = np.asarray(self.scale[name][1])
            w = float(np.mean(sd)) * (100.0 if name in ("near_dist", "row_dist") else 1.0)
            t_, pv = Y[name][i], dp[name]
            rv = dr[name] if dr is not None else None
            if msk is not None:
                t_, pv = t_[msk], pv[msk]
                rv = rv[msk] if rv is not None else None
            out.append((name, float((pv - t_).abs().mean()) * w,
                        float((rv - t_).abs().mean()) * w if rv is not None else float("nan"), "mae"))
        return out


def _cw(lab, k, m=None):
    v = lab.reshape(-1)[(m.reshape(-1) if m is not None else slice(None))]
    c = np.bincount(np.asarray(v).astype(np.int64), minlength=k).astype(np.float64)
    w = np.sqrt(c.sum() / np.maximum(c, 1.0))      # sqrt: a raw inverse collapsed the rare class to 0.0005
    return torch.from_numpy((w / w.mean()).astype(np.float32))
