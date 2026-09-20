"""How far can the world model roll forward? Autoregressive, action-conditioned, up to a full episode.

    .venv/bin/python scripts/eval_rollout.py checkpoints/dynamics/joint_v6.pt

Feeds the model the dataset's real actions and lets it predict its own next latent each step, then decodes
and scores against the truth. The control throughout is `identity` -- decoding the STARTING latent at every
horizon, i.e. assuming nothing ever changes. A model that cannot beat that has not learned dynamics.

The real environment caps episodes at 1800 steps (180 s at 10 Hz), so that is the horizon a dream has to
survive if it is to contain a whole fight.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dynamics import load_predictor  # noqa: E402
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, UnifiedDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class, wants_v2)
from scripts.train_bc_decode import DEFAULT_DATA, load_many  # noqa: E402

HORIZONS = (1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 200, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--min-len", type=int, default=400, help="episodes at least this long")
    ap.add_argument("--out", default="results/rollout.json")
    a = ap.parse_args()
    torch.set_num_threads(6)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = (UnifiedDecoder(latent=LAT) if ck.get("unified_decoder") else StateDecoder(latent=LAT))
    dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)      # residual or flat, per what the checkpoint stored
    norm = Norm.load(ck["norm"])

    meta, state, action, slot, epid, obs, has = load_many([a.data, *a.extra_data])
    Y = decode_targets(state, obs)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    Fn = norm.apply(featurise(state, link_v2=wants_v2(ck['link_dim'])))
    A = F.one_hot(torch.from_numpy(action), NA).float()

    test = set(ck["split"]["val_slots"])
    eps = [e for e in np.unique(epid)
           if np.isin(slot[epid == e][0], list(test)) and (epid == e).sum() >= a.min_len]
    idx = [np.where(epid == e)[0] for e in eps]
    L = min(min(len(i) for i in idx), max(HORIZONS) + 1)   # every episode must reach every horizon
    print(f"{len(eps)} held-out episodes of >= {a.min_len} steps; common horizon {L - 1}")

    starts = np.array([i[0] for i in idx])
    with torch.no_grad():
        z = enc.encode(torch.from_numpy(Fn["link"][starts]), torch.from_numpy(Fn["rows"][starts]),
                       torch.from_numpy(Fn["ids"][starts]), torch.from_numpy(Fn["mask"][starts]))
        z0 = z.clone()
    rows_out = []
    print(f"\n{'h':>6} {'behaviour':>20} {'baba_count':>20} {'link_hp':>18} {'row_hp':>18} {'link_pos MAE':>22}")
    print(f"{'':>6} {'model / identity':>20} {'model / identity':>20} {'model / ident':>18} "
          f"{'model / ident':>18} {'model / identity':>22}")
    print("-" * 110)
    with torch.no_grad():
        for h in range(1, L):
            act = torch.stack([A[i[h - 1]] for i in idx])
            z = pred(torch.cat([z, act], -1))
            if h not in HORIZONS:
                continue
            tgt = np.array([i[h] for i in idx])
            p, p0 = dec(z), dec(z0)
            mb = Y["row_is_baba"][tgt] > .5
            m = Y["row_present"][tgt] > .5
            sc = ck["decode_scale"]["link_pos"]
            f = lambda pp: (                                            # noqa: E731
                float((pp["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][tgt][mb]).mean()),
                float((pp["baba_count"].argmax(-1).numpy() == Y["baba_count"][tgt]).mean()),
                float((pp["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][tgt]).mean()),
                float((pp["row_hp"].argmax(-1).numpy()[m] == Y["row_hp_c"][tgt][m]).mean()),
                float(np.abs(pp["link_pos"].numpy() * sc[1] + sc[0] - Y["link_pos"][tgt]).mean()))
            mo, id_ = f(p), f(p0)
            rows_out.append({"horizon": h, "model": mo, "identity": id_})
            print(f"{h:>6} {mo[0]:>9.3f} / {id_[0]:<8.3f} {mo[1]:>9.3f} / {id_[1]:<8.3f} "
                  f"{mo[2]:>8.3f} / {id_[2]:<7.3f} {mo[3]:>8.3f} / {id_[3]:<7.3f} "
                  f"{mo[4]:>10.1f} / {id_[4]:<9.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "episodes": len(eps), "rows": rows_out}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
