"""Simplest possible latent dynamics: frozen encoder, 3-layer MLP, plain MSE on the raw latent.

    .venv/bin/python scripts/train_dynamics_simple.py --stride 1

    input   [ z_t (128) || one-hot action (63) ]
    body    Linear -> ReLU -> Linear -> ReLU -> Linear      (no activation on the output)
    loss    MSE( predicted z, actual z_{t+k} )

No residual, no target normalisation, no rollout term, no EMA -- deliberately none of the JEPA machinery,
so this isolates whether the plain regression works at all. The control is the do-nothing baseline
(predict z_t unchanged), which is strong here because the game state is highly autocorrelated at 10 Hz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)
from scripts.train_bc_decode import DEFAULT_DATA, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="checkpoints/bc/bc-decode-final.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--stride", type=int, default=1, help="frames ahead to predict")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--residual", action="store_true",
                    help="add z_t to the output. A plain ReLU MLP on zero-mean input has to rebuild the "
                         "signal through half-rectified layers, so it struggles to express even the "
                         "identity map -- which is the function to beat here.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out", default="checkpoints/dynamics/simple.pt")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.encoder, weights_only=False, map_location="cpu")
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=ck["n_actions"],
                latent=ck["latent"], gru=0)
    enc.load_state_dict(ck["model"]); enc.eval()
    dec = StateDecoder(latent=ck["latent"]); dec.load_state_dict(ck["decoder"]); dec.eval()
    norm, split, LAT, NA = Norm.load(ck["norm"]), ck["split"], ck["latent"], ck["n_actions"]

    meta, state, action, slot, epid = load(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    Fn = norm.apply(featurise(state))
    Z = []
    with torch.no_grad():
        for s in range(0, len(state), 8192):
            sl = slice(s, s + 8192)
            Z.append(enc.encode(torch.from_numpy(Fn["link"][sl]), torch.from_numpy(Fn["rows"][sl]),
                                torch.from_numpy(Fn["ids"][sl]), torch.from_numpy(Fn["mask"][sl])))
    Z = torch.cat(Z)
    A = F.one_hot(torch.from_numpy(action), NA).float()

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    def starts(mask, k):
        ok = mask.copy()
        for j in range(1, k + 1):
            ok &= (np.r_[epid[j:], np.full(j, -1)] == epid)
        return np.where(ok)[0]
    K = a.stride
    tr = starts(~np.isin(slot, list(test | tune)), K)
    tu = starts(np.isin(slot, list(tune)), K)
    te = starts(np.isin(slot, list(test)), 10 * K)

    body = nn.Sequential(nn.Linear(LAT + NA, a.hidden), nn.ReLU(),
                         nn.Linear(a.hidden, a.hidden), nn.ReLU(),
                         nn.Linear(a.hidden, LAT))

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.body = body
        def forward(self, xin):
            y = self.body(xin)
            return xin[..., :LAT] + y if a.residual else y
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    x = lambda i: torch.cat([Z[i], A[i]], -1)

    ident_te = float(F.mse_loss(Z[te], Z[te + K]))
    print(f"stride {K} ({K * 100} ms) | train {len(tr):,} tune {len(tu):,} test {len(te):,}")
    print(f"identity baseline (predict z unchanged), test MSE: {ident_te:.5f}\n")

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tot = n = 0.0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            loss = F.mse_loss(model(x(i)), Z[i + K])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        model.eval()
        with torch.no_grad():
            tl = float(F.mse_loss(model(x(torch.from_numpy(tu))), Z[tu + K]))
        if tl < best - 1e-6:
            best, best_state, bad = tl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 4 == 0 or ep == 1:
            ident_tu = float(F.mse_loss(Z[tu], Z[tu + K]))
            print(f"  ep {ep:>3}/{a.epochs}  train {tot / n:.5f}  tune {tl:.5f}  "
                  f"(identity {ident_tu:.5f}, skill {1 - tl / ident_tu:+.3f}){'  *' if bad == 0 else ''}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"  early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)

    # ---- test: rollout by feeding the model its own output
    model.eval()
    sub = te[rng.permutation(len(te))[:12000]]
    print(f"\n{'steps':>6} {'model MSE':>10} {'identity':>10} {'skill':>7} {'row_state':>10} "
          f"{'link_hp':>8} {'row_pos':>8}")
    print("-" * 68)
    out = []
    with torch.no_grad():
        z = Z[sub]
        for h in range(1, 11):
            z = model(torch.cat([z, A[sub + (h - 1) * K]], -1))
            if h not in (1, 2, 5, 10):
                continue
            tgt = sub + h * K
            mse = float(F.mse_loss(z, Z[tgt]))
            idn = float(F.mse_loss(Z[sub], Z[tgt]))
            p = dec(z)
            mb, m = Y["row_is_baba"][tgt] > .5, Y["row_present"][tgt] > .5
            rst = float((p["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][tgt][mb]).mean())
            lhp = float((p["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][tgt]).mean())
            sc = ck["decode_scale"]["row_pos"]
            rpos = float(np.abs((p["row_pos"].numpy() * sc[1] + sc[0])[m] - Y["row_pos"][tgt][m]).mean())
            out.append({"steps": h, "frames": h * K, "mse": mse, "identity_mse": idn,
                        "skill": 1 - mse / idn, "row_state": rst, "link_hp": lhp, "row_pos_mae": rpos})
            print(f"{h:>6} {mse:>10.5f} {idn:>10.5f} {1 - mse / idn:>+7.3f} {rst:>10.4f} "
                  f"{lhp:>8.4f} {rpos:>8.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(a), "metrics": out}, a.out)
    Path(f"results/dynamics_simple_s{K}.json").write_text(json.dumps({"config": vars(a), "horizons": out}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
