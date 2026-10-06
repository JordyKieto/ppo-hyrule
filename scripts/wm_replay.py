"""Roll the token world model AUTOREGRESSIVELY over a recorded episode and write a local viewer page.

    # one command, opens in your browser
    .venv/bin/python scripts/wm_replay.py --html viz/out/dream.html --open

    # pick the episode / window / checkpoint
    .venv/bin/python scripts/wm_replay.py --episode datasets/transitions/v5/episodes/ep00001_G030_s04_success.npz \
        --start 20 --horizon 300 --ckpt runs/wm_v2/ck_0080.pt --html viz/out/v2.html

    # terminal, no page: every field of one entity at one step
    .venv/bin/python scripts/wm_replay.py --frame 30 --entity link
    .venv/bin/python scripts/wm_replay.py --frame 30 --entity 2        # actor row 2

WHAT THE ROLLOUT IS. Step 0 is the real state at `start`. From there the model sees ONLY ITS OWN OUTPUT: each
step it is given its predicted state plus the REAL action the player took at that frame, and predicts the next
state. It never invents an action and it never re-reads the true state. That is the whole point of the view --
error compounds, and you watch where it goes. `--sample` draws from the discrete heads instead of argmaxing
(a Baba's inner state is unobservable, so its next behaviour is genuinely stochastic); default is argmax so
the page is reproducible.

The advance step here is a port of `advance()` in scripts/train_wm_tokens.py. Keeping them in sync matters: if
you change the head layout or the token fields there, change them here. The alternative was refactoring the
trainer, which is not something a viewer should force.

See docs/WM_VIEWER.md for the controls and what each colour means.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ONE THREAD. Dolphin needs ~2 threads per instance (emulation + the Null GPU thread) and this Mac has 4
# performance cores; torch defaults to taking all 4 for a forward pass, which starves the emulation thread past
# the 10 s Null-backend stall timeout and raises "emulation stalled (VI stuck)". 27 scripts in this repo
# already set this -- the three I added did not, and I spent three runs blaming the hardware. AGENTS.md 28.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass            # already initialised in this process; the intra-op setting is the one that matters

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_wm_tokens import (AHPD_K, AHPD_N, CENTRE, DC, DW, HPD_K, HPD_N, HPD_STEP,  # noqa: E402
                                     LOGIT_K, N_STATES, TokenWM)
from zelda_rl import wm_data  # noqa: E402
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.state_layout import ENEMY_FIELDS, ROW_DIM, link_fields  # noqa: E402
from zelda_rl.tokens import BG_BITS, REL_CLIP, SF1_KEEP  # noqa: E402
from zelda_rl.wm_context import local_frames, patch_offsets, wall_map_from_packed, wall_patch  # noqa: E402

NB = len(SF1_KEEP)
L_RELY, L_HP, L_HPF, L_YAW, L_SPEED, L_VEL, L_CAM, L_HEAD, L_FLOOR, L_NPRES, L_NBABA = \
    0, 1, 2, slice(3, 5), 5, slice(6, 9), slice(9, 11), slice(11, 13), 13, 14, 15
A_RELY, A_BEAR, A_HP, A_YAW, A_SHAPE, A_Y2P, A_TIMER, A_VEL = \
    0, slice(1, 3), 3, slice(4, 6), slice(6, 8), slice(8, 10), 10, slice(11, 14)
HIST = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
        "actor_rcell", "actor_off", "cell_xy", "lhp_bin", "ahp_bin", "frames")
LONG = ("ids", "cats", "states", "link_cell", "actor_rcell")

BABA_STATES = ["hidden", "emerge_retract", "idle", "prepare_lunge", "lunge", "pull_back", "hit", "stunned",
               "dying", "other"]
# Actor names come from zelda_rl.oomdp.ACTOR_CLASSES, which was built from the decompilation's own
# actor_table.h -- so this viewer and the object-centric model agree on what a thing is called, and an id we
# have not identified shows as its hex id rather than a guess.
_OOMDP = {}
try:
    from zelda_rl.oomdp import ACTOR_CLASSES
    _PRETTY = {"En_Dekubaba": "Deku Baba", "En_St": "Skulltula", "En_Sw": "Skullwalltula"}
    for _c in ACTOR_CLASSES:
        for _i in _c.ids:
            _OOMDP[_i] = _PRETTY.get(_c.name, _c.name)
except Exception:
    pass
# kind drives the colour, and the colour is IDENTICAL in both panels
KINDS = {0x0055: ("Deku Baba", "baba"), 0x0037: ("Skulltula", "baba"), 0x0095: ("Skullwalltula", "baba"),
         0x0125: ("Grass", "prop"), 0x0056: ("Deku Nut", "prop"), 0x005F: ("Heart", "pickup"),
         0x005D: ("Blue warp", "prop"), 0x0015: ("Rupee", "pickup"), 0x0011: ("Recovery heart", "pickup")}
# Link action-state -> a visible outline. action_fn_rel values are the engine's; the ones we can name.
# SF1_KEEP indices whose bit numbers we can name from zelda_rl.combat. The rest stay as bit<n> -- naming a
# flag we have not verified would be worse than leaving it numbered (AGENTS.md 3).
FLAG_AT = {1: "dead", 7: "shield", 8: "damaged", 10: "cutscene"}
def flags_from(bits):
    """bits: the 11 kept sf1 flags as 0/1. Returns {name: 0|1} with unnamed ones as bit<n>."""
    out = {}
    for i, v in enumerate(bits):
        out[FLAG_AT.get(i, f"sf1_{SF1_KEEP[i]}")] = int(v > .5)
    return out


def kind_of(aid: int, cat: int) -> tuple[str, str]:
    """Name and colour-kind for an actor. Gohma and her larvae share id 0x0028 -- category tells them apart."""
    if aid == 0x0028:
        return ("Gohma", "gohma") if cat == 9 else ("Larva", "larva")
    if aid in KINDS:
        return KINDS[aid]
    if aid in _OOMDP:
        return (_OOMDP[aid], "other")
    return (f"actor 0x{aid:04X}", "other")


class Replay:
    """One episode, one model, rolled forward on its own predictions. Queryable without any page."""

    def __init__(self, ckpt, episode, start=20, horizon=200, sample=False, device="cpu"):
        self.dev = torch.device(device)
        ck = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
        self.ck, c = ck, ck["config"]
        self.cfg, self.scales = c, ck["scales"]
        self.grid = NavGrid.load(ck["grid"])
        self.K, self.R, self.hb = c["hist"], c["rows"], c["hp_bins"]
        self.ENV_OFF = patch_offsets(c["env_radius"], self.dev)
        self.net = TokenWM(ck["n_actions"], c["dim"], c["heads"], n_rows=self.R,
                           attn=c.get("attn", "softmax"), layers=c["layers"], hist=self.K,
                           env_p=int(self.ENV_OFF.shape[0]), hp_bins=self.hb).to(self.dev)
        self.net.load_state_dict(ck["model"]); self.net.eval()
        self.ckpt, self.episode, self.sample = str(ckpt), str(episode), sample

        d = np.load(episode)
        st, act = d["state"].astype(np.float32), d["action"].astype(np.int64)
        st = st[:len(act) + 1]
        self.state, self.act = st, act
        epid = np.zeros(len(st), np.int64)
        P = wm_data.tokens_and_targets(st, np.r_[act, act[-1]], epid, epid, grid=self.grid,
                                       scales=self.scales, d_link_y=float(self.scales.get("l_floor_dy", 20.7)),
                                       rows_keep=self.R, n_actions=ck["n_actions"])
        self.P = P
        self.tt = {k: torch.from_numpy(np.ascontiguousarray(P[k])).to(self.dev) for k in HIST if k in P}
        self.WMAP = torch.from_numpy(wall_map_from_packed(P["link_cell"], P["link"][:, 16 + NB:], 0,
                                                         self.grid.n)).to(self.dev)
        self.CELLWH = torch.tensor(np.asarray(self.grid.cell, np.float32), device=self.dev)
        self.lf = link_fields(st.shape[1]); self.li = {n: i for i, n in enumerate(self.lf)}
        self.ei = {n: i for i, n in enumerate(ENEMY_FIELDS)}; self.L = len(self.lf)
        n = len(act)
        self.start = int(np.clip(start, 1, max(n - 2, 1)))
        self.H = int(max(1, min(horizon, n - 1 - self.start)))
        self._roll()

    # ---- token window from REAL frames (the burn-in) ------------------------
    def _window(self, i):
        w = torch.clamp(torch.arange(i - self.K + 1, i + 1, device=self.dev), min=0)[None]
        o = {k: (self.tt[k][w].long() if k in LONG else self.tt[k][w].float()) for k in self.tt}
        return self._derive(o)

    def _derive(self, o):
        o["lhp_bin"] = (o["link"][:, :, L_HP].clamp(0, 1) * (self.hb - 1)).round().long()
        o["ahp_bin"] = (o["actors"][:, :, :, A_HP].clamp(0, 1) * (self.hb - 1)).round().long()
        B, K = 1, o["link"].shape[1]
        rel = o["actor_rcell"].reshape(B * K, -1, 2).float() - REL_CLIP
        o["frames"] = local_frames(rel, o["actor_off"].reshape(B * K, -1, 2), o["link_off"].reshape(B * K, 2),
                                   o["link"].reshape(B * K, -1)[:, 3:5], o["link"].reshape(B * K, -1)[:, 9:11],
                                   self.CELLWH, scale=float(max(self.grid.cell)) * 8.0).reshape(B, K, -1, 4)
        occ, _ = wall_patch(o["link_cell"][:, -1], self.WMAP, self.ENV_OFF)
        o["env_occ"] = occ
        ent = torch.cat([torch.zeros(B, 1, 2, device=self.dev), o["cell_xy"][:, -1]], 1)
        o["env_d"] = (ent[:, :, None, :] - self.ENV_OFF.float()[None, None]).norm(dim=-1)
        o["rel_xz"] = o["actors"][:, -1, :, :2]
        return o

    # ---- the autoregressive step, ported from advance() in the trainer ------
    def _pick(self, lg):
        if not self.sample:
            return lg.argmax(-1)
        sh = lg.shape[:-1]
        return torch.multinomial(lg.reshape(-1, lg.shape[-1]).softmax(-1), 1).reshape(sh)

    def _flip(self, lg):
        pr = lg.sigmoid()
        return (torch.rand_like(pr) < pr).float() if self.sample else (pr > 0.5).float()

    def _heads(self, ol, oa, cur):
        j = 0
        dcl, j = ol[:, j:j + DC], j + DC
        dof, j = ol[:, j:j + 2], j + 2
        dyv, j = ol[:, j:j + 1], j + 1
        dyw, j = ol[:, j:j + 2], j + 2
        hpd, j = ol[:, j:j + HPD_N], j + HPD_N
        bits, j = ol[:, j:j + NB], j + NB
        scn, j = ol[:, j:j + 2], j + 2
        dvel, j = ol[:, j:j + 3], j + 3
        dcam, j = ol[:, j:j + 2], j + 2
        dfl, j = ol[:, j:j + 1], j + 1
        bg = ol[:, j:j + len(BG_BITS)]
        one = lambda v, n: F.one_hot(torch.tensor([v], device=self.dev), n).float()   # noqa: E731
        dcl = dcl + LOGIT_K * one(CENTRE, DC)
        hpd = hpd + LOGIT_K * one(HPD_K, HPD_N)
        bits = bits + LOGIT_K * (2 * cur["bits"] - 1)
        bg = bg + LOGIT_K * (2 * cur["bg"] - 1)
        m = 0
        rdcl, m = oa[..., m:m + DC], m + DC
        rdof, m = oa[..., m:m + 2], m + 2
        rhpd, m = oa[..., m:m + AHPD_N], m + AHPD_N
        rpr, m = oa[..., m:m + 1], m + 1
        rst, m = oa[..., m:m + N_STATES], m + N_STATES
        ardy, m = oa[..., m:m + 1], m + 1
        adyw, m = oa[..., m:m + 2], m + 2
        adsh, m = oa[..., m:m + 2], m + 2
        adti, m = oa[..., m:m + 1], m + 1
        advl = oa[..., m:m + 3]
        rdcl = rdcl + LOGIT_K * F.one_hot(torch.full(oa.shape[:2], CENTRE, device=self.dev), DC).float()
        rhpd = rhpd + LOGIT_K * F.one_hot(torch.full(oa.shape[:2], AHPD_K, device=self.dev), AHPD_N).float()
        rpr = rpr + LOGIT_K * (2 * cur["pres"][..., None] - 1)
        rst = rst + LOGIT_K * F.one_hot(cur["st"], N_STATES).float()
        return ((dcl, dof, dyv, dyw, hpd, bits, scn, dvel, dcam, dfl, bg),
                (rdcl, rdof, rhpd, rpr, rst, ardy, adyw, adsh, adti, advl))

    @torch.no_grad()
    def _advance(self, tok, ol, oa, cur):
        (dcl, dof, dyv, dyw, hpd, bits, scn, dvel, dcam, dfl, bg), \
            (rdcl, rdof, rhpd, rpr, rst, ardy, adyw, adsh, adti, advl) = self._heads(ol, oa, cur)
        now = {k: (tok[k][:, -1] if (self.K > 1 and k in HIST) else tok[k]) for k in tok}
        g, sc = self.grid, self.scales
        k = self._pick(dcl)
        dcell = torch.stack([torch.div(k, DW, rounding_mode="floor"), k % DW], -1) - DW // 2
        n_cell = (now["link_cell"] + dcell).clamp(0, g.n - 1)
        n_off = (now["link_off"].float() + dof).clamp(-0.5, 0.5)
        Lo = now["link"].float()
        rely = (Lo[:, L_RELY:L_RELY + 1] + dyv).clamp(-1, 1)
        hp = (Lo[:, L_HP:L_HP + 1] + (self._pick(hpd).float() - HPD_K)[:, None] * HPD_STEP
              / max(sc.get("l_hp", 48.0), 1e-6)).clamp(0, 1)
        yaw = (Lo[:, L_YAW] + dyw).clamp(-1, 1)
        vel = (Lo[:, L_VEL] + dvel).clamp(-1, 1)
        cam = (Lo[:, L_CAM] + dcam).clamp(-1, 1)
        floor = (Lo[:, L_FLOOR:L_FLOOR + 1] + dfl).clamp(-1, 1)
        nbits, nbg = self._flip(bits), self._flip(bg)
        speed = vel[:, :2].norm(dim=-1, keepdim=True).clamp(-1, 1)
        head = (yaw - cam).clamp(-1, 1)
        rk = self._pick(rdcl)
        rd = torch.stack([torch.div(rk, DW, rounding_mode="floor"), rk % DW], -1) - DW // 2
        abs_cell = (now["link_cell"][:, None, :] + now["actor_rcell"] - REL_CLIP) + rd
        rel = (abs_cell - n_cell[:, None, :]).clamp(-REL_CLIP, REL_CLIP)
        n_rcell = (rel + REL_CLIP).clamp(0, 2 * REL_CLIP)
        n_aoff = (now["actor_off"].float() + rdof).clamp(-0.5, 0.5)
        n_mask = self._flip(rpr.squeeze(-1))
        n_states = self._pick(rst)
        Ao = now["actors"].float()
        a_hp = (Ao[:, :, A_HP:A_HP + 1] + (self._pick(rhpd).float() - AHPD_K)[..., None]
                / max(sc.get("a_hp", 2.0), 1e-6)).clamp(0, 1)
        a_rely = (Ao[:, :, A_RELY:A_RELY + 1] + ardy).clamp(-1, 1)
        a_yaw = (Ao[:, :, A_YAW] + adyw).clamp(-1, 1)
        a_shape = (Ao[:, :, A_SHAPE] + adsh).clamp(-1, 1)
        a_timer = (Ao[:, :, A_TIMER:A_TIMER + 1] + adti).clamp(-1, 1)
        a_vel = (Ao[:, :, A_VEL] + advl).clamp(-1, 1)
        dx = rel[..., 0].float() + n_aoff[..., 0] - n_off[:, None, 0]
        dz = rel[..., 1].float() + n_aoff[..., 1] - n_off[:, None, 1]
        ang = torch.atan2(dx * float(g.cell[0]), dz * float(g.cell[1]))
        yaw_ang = torch.atan2(yaw[:, 0], yaw[:, 1])[:, None]
        bear = torch.stack([ang.sin(), ang.cos()], -1)
        y2p = torch.stack([(ang - yaw_ang).sin(), (ang - yaw_ang).cos()], -1)
        n_actors = torch.cat([a_rely, bear, a_hp, a_yaw, a_shape, y2p, a_timer, a_vel], -1) * n_mask[:, :, None]
        npres = (n_mask.sum(1, keepdim=True) / max(sc.get("l_n_present", 9.0), 1e-6)).clamp(-1, 1)
        nbaba = (((now["ids"] == 0x0055).float() * n_mask).sum(1, keepdim=True)
                 / max(sc.get("l_n_baba", 3.0), 1e-6)).clamp(-1, 1)
        n_link = torch.cat([rely, hp, hp, yaw, speed, vel, cam, head, floor, npres, nbaba, nbits, nbg], -1)
        newf = {"link": n_link, "actors": n_actors, "ids": now["ids"], "cats": now["cats"],
                "states": n_states, "mask": n_mask, "link_cell": n_cell, "link_off": n_off,
                "actor_rcell": n_rcell, "actor_off": n_aoff, "cell_xy": rel.float()}
        nt = {}
        for kk, v in tok.items():
            if kk in HIST and kk in newf:
                nt[kk] = torch.cat([v[:, 1:], newf[kk][:, None].to(v.dtype)], 1) if v.shape[1] > 1 \
                    else newf[kk][:, None].to(v.dtype)
            elif kk in HIST:
                nt[kk] = v
            else:
                nt[kk] = v
        nt = self._derive({k: nt[k] for k in HIST if k in nt})
        nxt = {"bits": nbits, "bg": nbg, "pres": n_mask, "st": n_states}
        return nt, nxt

    # ---- decode a token state into display fields --------------------------
    def _show(self, tok, real_frame=None):
        """Display dict from either the model's token state or (when real_frame is given) the true state."""
        if real_frame is not None:
            return self._show_real(real_frame)
        now = {k: (tok[k][:, -1] if self.K > 1 else tok[k]) for k in tok if k in HIST}
        g, sc = self.grid, self.scales
        cell = now["link_cell"][0].cpu().numpy().astype(float)
        off = now["link_off"][0].float().cpu().numpy().astype(float)
        Lo = now["link"][0].float().cpu().numpy()
        yaw = math.atan2(float(Lo[3]), float(Lo[4])) % (2 * math.pi)
        link = dict(self._geo(cell, off), hp=round(float(Lo[L_HP]) * sc.get("l_hp", 48.0), 1), hp_cap=48,
                    yaw=yaw * 65536 / (2 * math.pi), speed=round(float(Lo[L_SPEED]) * sc.get("l_speed", 10.0), 2),
                    vel_x=round(float(Lo[6]) * sc.get("l_vel_x", 8.7), 2),
                    vel_z=round(float(Lo[7]) * sc.get("l_vel_z", 8.7), 2),
                    floor_dy=round(float(Lo[L_FLOOR]) * sc.get("l_floor_dy", 20.7), 2),
                    on_ground=int(Lo[16 + NB] > .5), name="Link", kind="link",
                    **flags_from(Lo[16:16 + NB]))
        actors = []
        ids = now["ids"][0].cpu().numpy(); cats = now["cats"][0].cpu().numpy()
        rc = now["actor_rcell"][0].cpu().numpy().astype(float) - REL_CLIP
        ao = now["actor_off"][0].float().cpu().numpy().astype(float)
        Ao = now["actors"][0].float().cpu().numpy()
        msk = now["mask"][0].cpu().numpy(); sts = now["states"][0].cpu().numpy()
        for r in range(self.R):
            nm, kd = kind_of(int(ids[r]), int(cats[r]))
            actors.append(dict(self._geo(cell + rc[r], ao[r]), i=r, name=nm, kind=kd,
                               present=int(msk[r] > .5), hp=round(float(Ao[r, A_HP]) * sc.get("a_hp", 2.0), 2),
                               state=int(sts[r]), state_name=BABA_STATES[int(sts[r]) % len(BABA_STATES)],
                               timer=round(float(Ao[r, A_TIMER]) * sc.get("a_timer", 38.0), 1),
                               rel_y=round(float(Ao[r, A_RELY]) * sc.get("a_rel_y", 1088.0), 1),
                               actor_id=int(ids[r]), cat=int(cats[r])))
        return {"link": link, "actors": actors}

    def _geo(self, cell, off):
        g = self.grid
        c = np.clip(np.asarray(cell, float), 0, g.n - 1)
        o = np.clip(np.asarray(off, float), -0.5, 0.5)
        lo, hi = np.asarray(g.lo), np.asarray(g.hi)
        w = lo + (c + 0.5 + o) * ((hi - lo) / g.n)
        return {"cell": int(c[0] * g.n + c[1]), "cx": float(c[0] + .5 + o[0]), "cz": float(c[1] + .5 + o[1]),
                "wx": round(float(w[0]), 1), "wz": round(float(w[1]), 1)}

    def _show_real(self, fi):
        s, g = self.state[fi], self.grid
        lc, lo_ = g.to_cell(s[[self.li["pos_x"], self.li["pos_z"]]][None])
        link = dict(self._geo(lc[0], lo_[0]), hp=round(float(s[self.li["hp"]]), 1), hp_cap=48,
                    yaw=float(s[self.li["yaw"]] % 65536),
                    speed=round(float(s[self.li["speed"]]), 2) if "speed" in self.li else 0.0,
                    vel_x=round(float(s[self.li.get("vel_x", 0)]), 2),
                    vel_z=round(float(s[self.li.get("vel_z", 0)]), 2),
                    floor_dy=round(float(s[self.li["floor_height"]]), 2) if "floor_height" in self.li else 0.0,
                    on_ground=int(int(s[self.li["bg_flags"]]) & 1) if "bg_flags" in self.li else 1,
                    action_fn=int(s[self.li["action_fn_rel"]]) if "action_fn_rel" in self.li else 0,
                    melee=int(s[self.li["melee_state"]]) if "melee_state" in self.li else 0,
                    name="Link", kind="link",
                    **flags_from([(int(s[self.li["sf1"]]) >> b) & 1 for b in SF1_KEEP]))
        actors = []
        for r in range(self.R):
            row = s[self.L + r * ROW_DIM:self.L + (r + 1) * ROW_DIM]
            aid, cat = int(row[self.ei["id"]]), int(row[self.ei["cat"]])
            nm, kd = kind_of(aid, cat)
            ac, aof = g.to_cell(row[[self.ei["pos_x"], self.ei["pos_z"]]][None])
            actors.append(dict(self._geo(ac[0], aof[0]), i=r, name=nm, kind=kd,
                               present=int(row[self.ei["present"]] > 0), hp=round(float(row[self.ei["hp"]]), 2),
                               state=int(row[self.ei["state_idx"]]),
                               state_name=BABA_STATES[int(row[self.ei["state_idx"]]) % len(BABA_STATES)],
                               timer=round(float(row[self.ei["timer"]]), 1),
                               rel_y=round(float(row[self.ei["pos_y"]]), 1), actor_id=aid, cat=cat))
        return {"link": link, "actors": actors}

    # ---- roll ---------------------------------------------------------------
    @torch.no_grad()
    def _roll(self):
        from zelda_rl.env import action_name
        tok = self._window(self.start)
        cur = {"bits": tok["link"][:, -1, 16:16 + NB], "bg": tok["link"][:, -1, 16 + NB:],
               "pres": tok["mask"][:, -1], "st": tok["states"][:, -1]}
        self.frames = []
        pe, pp = [], []
        t0 = self._show_real(self.start)
        for h in range(self.H + 1):
            i = self.start + h
            truth = self._show_real(i)
            pred = truth if h == 0 else self._show(tok)
            a = int(self.act[min(i, len(self.act) - 1)])
            self.frames.append({"k": h, "frame": i, "action": a,
                                "action_name": action_name(a, "combat"), "truth": truth, "pred": pred})
            t, m = truth["link"], pred["link"]
            pe.append(math.dist((t["wx"], t["wz"]), (m["wx"], m["wz"])))
            pp.append(math.dist((t["wx"], t["wz"]), (t0["link"]["wx"], t0["link"]["wz"])))
            if h < self.H:
                oh = F.one_hot(torch.tensor([a], device=self.dev), self.ck["n_actions"]).float()
                ol, oa = self.net(tok, oh)
                tok, cur = self._advance(tok, ol, oa, cur)
        assert self.frames[0]["pred"] == self.frames[0]["truth"], "step 0 must be the real state"
        self.pos_err, self.persist_err = pe, pp

    def to_json(self):
        fl = (["cell", "cx", "cz", "wx", "wz", "hp", "yaw", "speed", "vel_x", "vel_z", "floor_dy",
               "on_ground", "action_fn", "melee"]
              + [FLAG_AT.get(i, f"sf1_{b}") for i, b in enumerate(SF1_KEEP)])
        fa = ["name", "present", "cell", "cx", "cz", "wx", "wz", "hp", "state_name", "state", "timer",
              "rel_y", "actor_id", "cat"]
        return {"meta": {"episode": self.episode, "n_frames": len(self.frames), "start": self.start,
                         "horizon": self.H, "burn": self.K, "checkpoint": self.ckpt, "grid_n": self.grid.n,
                         "mode": "autoregressive (open loop, real actions)",
                         "sampled": bool(self.sample), "states": BABA_STATES},
                "walls": [int(v) for v in self.WMAP.cpu().numpy().reshape(-1)],
                "fields": {"link": fl, "actor": fa}, "frames": self.frames,
                "metrics": {"pos_err_world": [round(v, 2) for v in self.pos_err],
                            "persist_err_world": [round(v, 2) for v in self.persist_err]}}

    def table(self, kind, k, i=0):
        f = self.frames[min(k, len(self.frames) - 1)]
        t = f["truth"]["link"] if kind == "link" else f["truth"]["actors"][i]
        m = f["pred"]["link"] if kind == "link" else f["pred"]["actors"][i]
        nm = "Link" if kind == "link" else f"{t.get('name', 'actor')} (row {i})"
        out = [f"{nm}   step {f['k']}  episode frame {f['frame']}  action {f['action_name']}",
               f"{'field':<14}{'real':>13}{'model':>13}{'diff':>13}"]
        for x in [q for q in t if q in m]:
            a, b = t[x], m[x]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                out.append(f"{x:<14}{a:>13.2f}{b:>13.2f}{b - a:>+13.2f}")
            else:
                out.append(f"{x:<14}{str(a):>13}{str(b):>13}{'' if a == b else '   <-- differs':>13}")
        return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/wm_v3/ck_0085.pt")
    p.add_argument("--episode", default=None)
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--sample", action="store_true", help="sample discrete heads instead of argmax")
    p.add_argument("--json", default=None, help="write the raw replay JSON here")
    p.add_argument("--html", default=None, help="write a self-contained local viewer page here")
    p.add_argument("--open", action="store_true", help="open the page in your browser when done")
    p.add_argument("--frame", type=int, default=None, help="print one step's field table and exit")
    p.add_argument("--entity", default="link", help="'link' or an actor row number")
    a = p.parse_args()

    ep = a.episode
    if ep is None:
        for f in sorted(glob.glob("datasets/transitions/v5/episodes/*.npz"))[:4000]:
            if len(np.load(f)["action"]) >= 160:
                ep = f
                break
    if ep is None:
        raise SystemExit("no episode found; pass --episode")
    print(f"episode  {ep}\ncheckpoint {a.ckpt}")
    rp = Replay(a.ckpt, ep, a.start, a.horizon, sample=a.sample)
    pe, pp = np.array(rp.pos_err), np.array(rp.persist_err)
    n = len(rp.frames)
    print(f"{n - 1} autoregressive steps from frame {rp.start}, {rp.R} actor rows, "
          f"grid {rp.grid.n}x{rp.grid.n}, {'sampled' if a.sample else 'argmax'}")
    print(f"\n{'step':>6}{'model':>10}{'persist':>10}{'skill':>9}   (world units from the true position)")
    for s in [x for x in (1, 2, 5, 10, 20, 50, 100, n - 1) if 0 < x < n]:
        sk = 1 - pe[s] / pp[s] if pp[s] > 1e-9 else float("nan")
        print(f"{s:>6}{pe[s]:>10.1f}{pp[s]:>10.1f}{sk:>+9.3f}")
    ok = pp > 1e-9
    print(f"overall  model {pe.mean():.2f}   persistence {pp.mean():.2f}   "
          f"skill {1 - pe[ok].sum() / pp[ok].sum():+.4f}")
    if a.frame is not None:
        print()
        print(rp.table("link" if a.entity == "link" else "actor", a.frame,
                       0 if a.entity == "link" else int(a.entity)))
    blob = rp.to_json()
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(blob))
        print(f"\nwrote {a.json}  ({Path(a.json).stat().st_size / 1e6:.2f} MB)")
    if a.html:
        tpl = ROOT / "viz" / "wm_viewer.html"
        if not tpl.exists():
            raise SystemExit(f"missing template {tpl}")
        out = Path(a.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        page = tpl.read_text().replace("/*__REPLAY__*/null", json.dumps(blob))
        out.write_text(page)
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)  -- open it in any browser, no server needed")
        if a.open:
            subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
