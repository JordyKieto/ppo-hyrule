"""Train latent dynamics over the frozen behaviour-shaped encoder, and measure it where it matters.

    .venv/bin/python scripts/train_dynamics.py --encoder checkpoints/bc/bc-decode-final.pt

Predicts z_{t+1} = M(z_t, a_t) against the frozen encoder's own embedding of the true next state. The
encoder and decoder are never updated. Evaluation decodes the *predicted* latent through the same frozen
decoder used in paper/paper.md Table 5, so every field and event number is directly comparable to the
autoencoding upper bound reported there.

Reported at horizons 1, 5, 10, 50 by iterating the model on its own output, since compounding error is the
expected failure mode and the horizon at which behaviour state decays to chance is what bounds a dream.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)
from zelda_rl.dynamics import LatentDynamics, norm_target  # noqa: E402
from scripts.train_bc_decode import DEFAULT_DATA, load  # noqa: E402

HORIZONS = (1, 2, 5, 10)      # prediction steps; each spans --stride frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="checkpoints/bc/bc-decode-final.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="early stop after this many evals with no gain")
    ap.add_argument("--unroll", type=int, default=2,
                    help="rollout loss horizon; V-JEPA 2-AC adds a 2-step term to cut error accumulation")
    ap.add_argument("--stride", type=int, default=5,
                    help="frames per predicted step. At 10 Hz stride 1 leaves 74%% of targets unchanged, "
                         "which is the trivial-prediction failure mode JEPA work engineers around")
    ap.add_argument("--residual", action="store_true",
                    help="predict z_t + delta (no JEPA does this; it makes identity the initialisation)")
    ap.add_argument("--out", default="checkpoints/dynamics/latent-dynamics.pt")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)

    # ---- frozen encoder / decoder
    ck = torch.load(a.encoder, weights_only=False, map_location="cpu")
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=ck["n_actions"],
                latent=ck["latent"], gru=0)
    enc.load_state_dict(ck["model"]); enc.eval()
    dec = StateDecoder(latent=ck["latent"]); dec.load_state_dict(ck["decoder"]); dec.eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    norm = Norm.load(ck["norm"])
    split = ck["split"]

    meta, state, action, slot, epid = load(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)

    print(f"encoding {len(state):,} frames with the frozen encoder...", flush=True)
    Fn = norm.apply(featurise(state))
    Z = []
    with torch.no_grad():
        for s in range(0, len(state), 8192):
            sl = slice(s, s + 8192)
            Z.append(enc.encode(torch.from_numpy(Fn["link"][sl]), torch.from_numpy(Fn["rows"][sl]),
                                torch.from_numpy(Fn["ids"][sl]), torch.from_numpy(Fn["mask"][sl])))
    Z = torch.cat(Z)
    act = torch.from_numpy(action)
    print(f"latents {tuple(Z.shape)}  | z mean {Z.mean():.3f} std {Z.std():.3f}")

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_mask = ~np.isin(slot, list(test | tune))
    # a training index is valid only if the next `unroll` frames are in the same episode
    def valid_starts(mask, k):
        ok = mask.copy()
        for j in range(1, k + 1):
            shifted = np.r_[epid[j:], np.full(j, -1)] == epid
            ok &= shifted
        return np.where(ok)[0]
    tr_idx = valid_starts(tr_mask, a.unroll * a.stride)
    tu_idx = valid_starts(np.isin(slot, list(tune)), max(HORIZONS) * a.stride)
    te_idx = valid_starts(np.isin(slot, list(test)), max(HORIZONS) * a.stride)
    print(f"train starts {len(tr_idx):,} (slots {sorted(set(slot[tr_mask].tolist()))}) | "
          f"tune {len(tu_idx):,} (slots {sorted(tune)}) | test starts {len(te_idx):,} (slots {sorted(test)})")

    model = LatentDynamics(latent=ck["latent"], n_actions=ck["n_actions"], hidden=a.hidden, layers=a.layers,
                           dropout=a.dropout, residual=a.residual)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    rng = np.random.default_rng(a.seed)

    # baseline: predict no change at all. Any model must beat this to be worth having.
    with torch.no_grad():
        idle = float(F.l1_loss(norm_target(Z[te_idx]), norm_target(Z[te_idx + a.stride])))
    print(f"stride {a.stride} ({a.stride * 100} ms) | identity baseline (z_hat = z_t) 1-step L1: {idle:.5f}\n")

    t0 = time.time()
    best_mse, best_state, bad = float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = n = 0.0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = Z[i]
            loss = 0.0
            for k in range(a.unroll):
                z = model(z, act[i + k * a.stride])
                loss = loss + F.l1_loss(norm_target(z), norm_target(Z[i + (k + 1) * a.stride]))
            loss = loss / a.unroll
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        model.eval()                                    # model selection on TUNE slots, never on test
        with torch.no_grad():
            zp = model(Z[tu_idx], act[tu_idx])
            tune_mse = float(F.l1_loss(norm_target(zp), norm_target(Z[tu_idx + a.stride])))
        if tune_mse < best_mse - 1e-5:
            best_mse, best_state, bad = tune_mse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 2 == 0 or ep == 1:
            print(f"  epoch {ep:>3}/{a.epochs}  train {tot / n:.5f}  tune 1-step L1 {tune_mse:.5f}"
                  f"{'  *' if bad == 0 else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"  early stop at epoch {ep} (best tune L1 {best_mse:.5f})")
            break
    model.load_state_dict(best_state)

    # ---------------- evaluation: decode the PREDICTED latent through the frozen decoder
    model.eval()
    sub = te_idx[rng.permutation(len(te_idx))[:12000]]
    rows_out = []
    print(f"\n{'steps':>8} {'latent L1':>11} {'cos':>7} {'link_hp':>9} {'row_hp':>8} {'row_state':>10} "
          f"{'row_pos MAE':>12} {'present':>8}")
    print("-" * 82)
    with torch.no_grad():
        maxh = max(HORIZONS)
        acts = torch.stack([act[sub + k * a.stride] for k in range(maxh)], 1)
        traj = model.rollout(Z[sub], acts)                      # (B, maxh, latent)
        for h in HORIZONS:
            zp = traj[:, h - 1]
            tgt_i = sub + h * a.stride
            mse = float(F.l1_loss(norm_target(zp), norm_target(Z[tgt_i])))
            cos = float(F.cosine_similarity(zp, Z[tgt_i], -1).mean())
            p = dec(zp)
            lhp = float((p["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][tgt_i]).mean())
            m = Y["row_present"][tgt_i] > 0.5
            rhp = float((p["row_hp"].argmax(-1).numpy()[m] == Y["row_hp_c"][tgt_i][m]).mean())
            mb = Y["row_is_baba"][tgt_i] > 0.5
            rst = float((p["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][tgt_i][mb]).mean())
            sc = ck["decode_scale"]["row_pos"]
            rpos = float(np.abs((p["row_pos"].numpy() * sc[1] + sc[0])[m] - Y["row_pos"][tgt_i][m]).mean())
            pres = float(((p["row_present"].numpy() > 0) == (Y["row_present"][tgt_i] > 0.5)).mean())
            # identity control at the same horizon: z_hat = z_t, decoded the same way
            i_mse = float(F.l1_loss(norm_target(Z[sub]), norm_target(Z[tgt_i])))
            pi = dec(Z[sub])
            i_rst = float((pi["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][tgt_i][mb]).mean())
            i_lhp = float((pi["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][tgt_i]).mean())
            rows_out.append({"horizon": h, "frames": h * a.stride, "skill_vs_identity": 1 - mse / max(i_mse, 1e-9),
                             "latent_l1": mse, "cosine": cos, "link_hp": lhp, "row_hp": rhp,
                             "row_state": rst, "row_pos_mae": rpos, "row_present": pres,
                             "identity_latent_mse": i_mse, "identity_row_state": i_rst,
                             "identity_link_hp": i_lhp})
            print(f"{h:>8} {mse:>11.5f} {cos:>7.4f} {lhp:>9.4f} {rhp:>8.4f} {rst:>10.4f} "
                  f"{rpos:>12.2f} {pres:>8.4f}")
            print(f"{'  (identity)':>8} {i_mse:>11.5f} {'-':>7} {i_lhp:>9.4f} {'-':>8} {i_rst:>10.4f}"
                  f"    skill vs identity: {1 - mse / max(i_mse, 1e-9):+.3f}")

    # upper bound: the same decoder on the TRUE latent (this is paper Table 4, on the same frames)
    with torch.no_grad():
        p = dec(Z[sub + a.stride])
        m = Y["row_present"][sub + a.stride] > 0.5
        mb = Y["row_is_baba"][sub + a.stride] > 0.5
        sc = ck["decode_scale"]["row_pos"]
        ub = {"link_hp": float((p["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][sub + a.stride]).mean()),
              "row_hp": float((p["row_hp"].argmax(-1).numpy()[m] == Y["row_hp_c"][sub + a.stride][m]).mean()),
              "row_state": float((p["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][sub + a.stride][mb]).mean()),
              "row_pos_mae": float(np.abs((p["row_pos"].numpy() * sc[1] + sc[0])[m]
                                          - Y["row_pos"][sub + a.stride][m]).mean()),
              "row_present": float(((p["row_present"].numpy() > 0) == (Y["row_present"][sub + a.stride] > 0.5)).mean())}
    print(f"{'ENCODER':>8} {'-':>11} {'-':>7} {ub['link_hp']:>9.4f} {ub['row_hp']:>8.4f} "
          f"{ub['row_state']:>10.4f} {ub['row_pos_mae']:>12.2f} {ub['row_present']:>8.4f}   <- upper bound "
          f"(decode the TRUE latent)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(a), "encoder": a.encoder, "split": split,
                "latent": ck["latent"], "n_actions": ck["n_actions"],
                "metrics": {"horizons": rows_out, "upper_bound": ub, "identity_mse": idle}}, a.out)
    res = Path("results/dynamics.json")
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps({"checkpoint": a.out, "config": vars(a), "split": split,
                               "horizons": rows_out, "upper_bound": ub, "identity_mse": idle}, indent=2))
    print(f"\n-> {a.out}\n-> {res}")


if __name__ == "__main__":
    main()
