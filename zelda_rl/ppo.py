"""Minimal CleanRL-style PPO pieces: actor-critic MLP, GAE, clipped update."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                   _layer(hidden, n_actions, 0.01))
        self.critic = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                    _layer(hidden, 1, 1.0))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None, deterministic=False):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


def gae(rewards, values, dones, next_value, gamma, lam):
    """rewards/values/dones: (T, N); dones[t]=1 if the episode terminated at step t."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def ppo_update(model, opt, batch, *, epochs, minibatches, clip, ent_coef, vf_coef, max_grad_norm, target_kl=None):
    obs, act, logp_old, adv, ret, val_old = (torch.as_tensor(batch[k]) for k in ("obs", "act", "logp", "adv", "ret", "val"))
    n = obs.shape[0]
    mb = max(n // minibatches, 1)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    skipped = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, mb):
            i = perm[s:s + mb]
            _, logp, ent, v = model.act(obs[i], act[i])
            log_ratio = (logp - logp_old[i]).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            a = adv[i]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - clip, 1 + clip)).mean()
            vl = 0.5 * ((v - ret[i]) ** 2).mean()
            loss = pg - ent_coef * ent.mean() + vf_coef * vl
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(vl.item())
                stats["entropy"].append(ent.mean().item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > clip).float().mean().item())
        if target_kl is not None and np.mean(stats["approx_kl"][-minibatches:]) > target_kl:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["skipped_minibatches"] = skipped
    y, yp = batch["ret"], batch["val"]
    out["explained_variance"] = float(1 - np.var(y - yp) / (np.var(y) + 1e-8))
    return out


class EntityActorCritic(nn.Module):
    """v2 policy: global features + a table of nearest actors (shared per-actor encoder with actor-id embedding,
    masked mean+max pooling). Same act()/value() interface as ActorCritic; obs stays one flat float vector."""

    def __init__(self, global_dim: int, n_actors: int, actor_dim: int, n_actions: int, id_vocab: int = 512,
                 id_emb: int = 16, actor_hidden: int = 64, hidden: int = 256):
        super().__init__()
        self.global_dim, self.n_actors, self.actor_dim = global_dim, n_actors, actor_dim
        self.id_vocab = id_vocab
        self.id_embedding = nn.Embedding(id_vocab, id_emb)
        self.actor_enc = nn.Sequential(_layer(id_emb + actor_dim - 1, actor_hidden), nn.ReLU(),
                                       _layer(actor_hidden, actor_hidden), nn.ReLU())
        trunk_in = global_dim + 2 * actor_hidden
        self.actor = nn.Sequential(_layer(trunk_in, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                   _layer(hidden, n_actions, 0.01))
        self.critic = nn.Sequential(_layer(trunk_in, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                    _layer(hidden, 1, 1.0))

    def _features(self, x):
        g = x[..., :self.global_dim]
        t = x[..., self.global_dim:].reshape(*x.shape[:-1], self.n_actors, self.actor_dim)
        ids = t[..., 0].long().clamp(0, self.id_vocab - 1)
        present = t[..., -1:]
        h = self.actor_enc(torch.cat([self.id_embedding(ids), t[..., 1:]], -1)) * present
        n = present.sum(-2).clamp(min=1.0)
        mean = h.sum(-2) / n
        mx = (h - (1 - present) * 1e4).max(-2).values * (present.sum(-2) > 0)
        return torch.cat([g, mean, mx], -1)

    def value(self, x):
        return self.critic(self._features(x)).squeeze(-1)

    def act(self, x, action=None, deterministic=False):
        f = self._features(x)
        dist = Categorical(logits=self.actor(f))
        if action is None:
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(f).squeeze(-1)


class RecurrentEntityActorCritic(nn.Module):
    """Entity encoder (as EntityActorCritic) -> GRU memory -> actor / critic heads.
    The hidden state is carried across steps and zeroed where `first` = 1 (episode start)."""

    recurrent = True

    def __init__(self, global_dim: int, n_actors: int, actor_dim: int, n_actions: int, hidden: int = 256, gru: int = 256):
        super().__init__()
        self.enc = EntityActorCritic(global_dim, n_actors, actor_dim, n_actions, hidden=hidden)
        trunk_in = global_dim + 2 * self.enc.actor_enc[-2].out_features
        self.proj = nn.Sequential(_layer(trunk_in, gru), nn.Tanh())
        self.gru = nn.GRUCell(gru, gru)
        self.hidden_size = gru
        self.pi = nn.Sequential(_layer(gru, hidden), nn.Tanh(), _layer(hidden, n_actions, 0.01))
        self.v = nn.Sequential(_layer(gru, hidden), nn.Tanh(), _layer(hidden, 1, 1.0))
        del self.enc.actor, self.enc.critic       # only the entity feature encoder is reused

    def init_hidden(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden_size)

    def encode(self, x):
        """The pre-GRU latent: a pure function of the observation, with no recurrence in it.

        Alignable separately from the GRU state, which is this encoding plus everything the episode has
        accumulated. Measured on this task: v15 and the BC clone differ by 0.497 here and 0.938 at the GRU,
        so most of their divergence is accumulated memory rather than perception.
        """
        return self.proj(self.enc._features(x))

    def step(self, x, h, first):
        h = h * (1.0 - first.unsqueeze(-1))
        h = self.gru(self.encode(x), h)
        return self.pi(h), self.v(h).squeeze(-1), h

    def act(self, x, h, first, action=None, deterministic=False):
        logits, v, h = self.step(x, h, first)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), v, h

    def value(self, x, h, first):
        return self.step(x, h, first)[1]


def latent_gap(student, teacher, obs, first, w, site: str = "gru"):
    """Mean squared gap between student and frozen-teacher latents, weighted by `w` (B, L).

    `site` picks WHAT is aligned: "proj" the pre-GRU encoding of the observation, "gru" the recurrent state.
    The caller picks WHERE by choosing `obs`: demonstration windows, or the policy's own rollout states.
    """
    if site == "proj":
        flat = obs.reshape(-1, obs.shape[-1])
        sp = student.encode(flat)
        with torch.no_grad():
            tp = teacher.encode(flat)
        d = ((sp - tp) ** 2).mean(-1).reshape(obs.shape[0], obs.shape[1])
    else:
        sh, th = student.init_hidden(obs.shape[0]), teacher.init_hidden(obs.shape[0])
        rows = []
        for k in range(obs.shape[1]):
            _, _, sh = student.step(obs[:, k], sh, first[:, k])
            with torch.no_grad():
                _, _, th = teacher.step(obs[:, k], th, first[:, k])
            rows.append(((sh - th) ** 2).mean(-1))
        d = torch.stack(rows, 1)
    return (d * w).sum() / w.sum().clamp(min=1.0)


class ActionValueHead(nn.Module):
    """Q(s, .) over the discrete action set, read off the policy's GRU state.

    IBRL (Hu et al. 2023, arXiv 2311.02198) selects between the RL actor's action and the BC policy's action
    with `argmax_a Q(s, a)`. This project's critic is a V head, so there is nothing to compare two candidate
    actions with -- hence this head. It is a SEPARATE module, not part of the policy, so `--init-from` still
    loads v15's checkpoint unchanged.

    DEVIATIONS from the paper, stated here because they are structural (AGENTS 13):
    1. IBRL is built on off-policy RL (RLPD/SAC, replay buffer, n-step targets). This is on-policy PPO, so Q
       is fitted SARSA-style to the same GAE returns the V head uses -- it is the value of the action actually
       taken under the current behaviour, not an off-policy optimum.
    2. Executing the BC policy's action makes the behaviour policy differ from pi, which PPO only tolerates
       through its ratio clipping. The paper uses off-policy RL precisely to avoid that.
    3. No ensemble, no bootstrap proposal in the TD target -- actor proposal only.
    """

    def __init__(self, gru: int, hidden: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(_layer(gru, hidden), nn.Tanh(), _layer(hidden, n_actions, 1.0))

    def forward(self, h):
        return self.net(h)


def latent_delta_gap(student, teacher, obs, first, w):
    """Align how the two policies UPDATE their belief, not what their belief is: MSE between `h_t - h_{t-1}`.

    The absolute-state form forces the student's memory to equal the teacher's, which the measurement says is
    stronger than necessary (encoder gap 0.497 vs GRU gap 0.938 -- the divergence is accumulated memory). Two
    policies can track the fight the same way while holding different absolute states; matching transitions
    asks for that and nothing more.
    """
    sh, th = student.init_hidden(obs.shape[0]), teacher.init_hidden(obs.shape[0])
    rows = []
    for k in range(obs.shape[1]):
        s_prev, t_prev = sh, th
        _, _, sh = student.step(obs[:, k], sh, first[:, k])
        with torch.no_grad():
            _, _, th = teacher.step(obs[:, k], th, first[:, k])
        if k > 0:
            rows.append((((sh - s_prev) - (th - t_prev)) ** 2).mean(-1))
    if not rows:
        return torch.zeros((), device=obs.device)
    d = torch.stack(rows, 1)
    return (d * w[:, 1:]).sum() / w[:, 1:].sum().clamp(min=1.0)


def projected_gap(student, teacher, obs, first, w, proj: nn.Module, site: str = "proj"):
    """FitNets-style: compare a LEARNED LINEAR MAP of the student's latent to the teacher's, not the raw one.

    Forcing two encoders to produce identical vectors demands they share a basis. A regressor between them
    (Romero et al. 2015, FitNets) asks only that the student's latent CONTAIN the teacher's information up to
    a linear transform -- the weaker and more appropriate constraint when both nets solved the task
    differently (3% action agreement here).
    """
    if site == "proj":
        flat = obs.reshape(-1, obs.shape[-1])
        sp = proj(student.encode(flat))
        with torch.no_grad():
            tp = teacher.encode(flat)
        d = ((sp - tp) ** 2).mean(-1).reshape(obs.shape[0], obs.shape[1])
    else:
        sh, th = student.init_hidden(obs.shape[0]), teacher.init_hidden(obs.shape[0])
        rows = []
        for k in range(obs.shape[1]):
            _, _, sh = student.step(obs[:, k], sh, first[:, k])
            with torch.no_grad():
                _, _, th = teacher.step(obs[:, k], th, first[:, k])
            rows.append(((proj(sh) - th) ** 2).mean(-1))
        d = torch.stack(rows, 1)
    return (d * w).sum() / w.sum().clamp(min=1.0)


def align_loss(a_lat, b_lat, w, kind: str = "mse", tau: float = 0.1):
    """Pull two encoders' activations for the SAME state together, with gradients into BOTH sides.

    `mse` forces equality -- the strongest constraint, and measured to be stronger than needed here.
    `infonce` is the contrastive form of the teacher-student alignment work (arXiv 2605.28372): the same
    state's two encodings are the positive pair and every other state in the batch is a negative, so each
    encoder must IDENTIFY the state in the other's space rather than reproduce its numbers. Scale- and
    basis-free, which matters when both encoders are moving.
    """
    if kind == "mse":
        d = ((a_lat - b_lat) ** 2).mean(-1)
        return (d * w).sum() / w.sum().clamp(min=1.0)
    a = F.normalize(a_lat.reshape(-1, a_lat.shape[-1]), dim=-1)
    b = F.normalize(b_lat.reshape(-1, b_lat.shape[-1]), dim=-1)
    logits = a @ b.t() / tau
    tgt = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def ppo_update_recurrent(model, opt, batch, *, chunk, epochs, chunks_per_mb, clip, ent_coef, vf_coef, max_grad_norm,
                         target_kl=None, kl_coef=0.0, bc=None, bc_coef=0.0, bc_adaptive_frac=0.0,
                         bc_teacher=None, bc_latent_coef=0.0, bc_latent_site="gru",
                         bc_latent_states="demo", peer=None, peer_opt=None, peer_bc_coef=1.0,
                         align_coef=0.0, align_kind="mse", latent_mode="state", latent_proj=None,
                         q_head=None, q_coef=0.5, rvs=None, rvs_coef=0.0):
    """Recurrent PPO: minibatches of length-`chunk` sequences, each re-run from its stored initial hidden state.
    batch arrays are (T, N, ...): obs, act, logp, adv, ret, val, first, h (hidden state *before* each step).

    `bc_adaptive_frac` > 0 ignores `bc_coef` and sets the weight each minibatch so the BC term is that
    fraction of |policy loss|. `bc_teacher` + `bc_latent_coef` > 0 adds an MSE between the student's GRU state
    and a frozen BC policy's on the same frames. `bc["keep"]` filters which demonstration frames are paid for.

    `bc_coef` > 0 adds a BEHAVIOUR-CLONING PENALTY on human demonstrations: `bc` is the dict from
    `zelda_rl.bc_demos.load_bc_chunks` and each minibatch pays cross-entropy on an equal number of human windows.
    It regularises against the DATA, where `kl_coef` regularises against a POLICY -- use both or either.

    `kl_coef` > 0 adds a KL PENALTY toward an anchor policy, using `batch["anchor_logp"]` (T, N, A): the
    anchor's full log-probabilities, recorded during the rollout so the anchor's GRU carries its OWN hidden
    state rather than the trainee's.

    This is a different instrument from `target_kl`, which early-stops the epoch loop on the KL between the
    policy before and after THIS update. That bounds each step and says nothing about cumulative drift: in
    gohma-band-v13 the measured per-update KL was 0.005-0.007 against a 0.03 threshold, so it never fired
    once, while the defeat rate drifted from 7.26% down to 6.08% over ~23 updates that were each individually
    fine. A penalty toward a known-good policy bounds the distance travelled, not the step size. Adopted from
    puffertank E39, where it was the only arm that stopped an equivalent decline (the arms are in
    ../puffertank/docs/moba/experiments/E39-continuous-dyna.md); there coef 1.0 held the policy at the anchor
    without improving it and 0.3 let it improve, so 0.3 is the default worth trying first.
    """
    T, N = batch["act"].shape
    starts = [(s, i) for i in range(N) for s in range(0, T - chunk + 1, chunk)]
    tt = {k: torch.as_tensor(v) for k, v in batch.items()}
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    skipped = 0
    for _ in range(epochs):
        order = np.random.permutation(len(starts))
        for m0 in range(0, len(order), chunks_per_mb):
            sel = [starts[j] for j in order[m0:m0 + chunks_per_mb]]
            idx_t = torch.as_tensor([[s + k for k in range(chunk)] for s, _ in sel])          # (B, L)
            idx_n = torch.as_tensor([[i] * chunk for _, i in sel])
            g = lambda k: tt[k][idx_t, idx_n]
            obs, act, logp_old, adv, ret, first = g("obs"), g("act"), g("logp"), g("adv"), g("ret"), g("first")
            h = tt["h"][idx_t[:, 0], idx_n[:, 0]]
            anc = g("anchor_logp") if (kl_coef > 0 and "anchor_logp" in tt) else None
            logps, ents, vals, kls = [], [], [], []
            qs = []
            for k in range(chunk):
                logits, v, h = model.step(obs[:, k], h, first[:, k])
                dist = Categorical(logits=logits)
                logps.append(dist.log_prob(act[:, k])); ents.append(dist.entropy()); vals.append(v)
                if q_head is not None:
                    qs.append(q_head(h).gather(-1, act[:, k:k + 1].long()).squeeze(-1))
                if anc is not None:
                    lp = torch.log_softmax(logits, -1)
                    # KL(current || anchor), summed over actions: mode-seeking, so it penalises the policy
                    # putting mass where the anchor puts none.
                    kls.append((lp.exp() * (lp - anc[:, k])).sum(-1))
            logp, ent, v = torch.stack(logps, 1), torch.stack(ents, 1), torch.stack(vals, 1)
            kl_anchor = torch.stack(kls, 1).mean() if kls else None
            log_ratio = (logp - logp_old).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            a = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - clip, 1 + clip)).mean()
            vl = 0.5 * ((v - ret) ** 2).mean()
            loss = pg - ent_coef * ent.mean() + vf_coef * vl
            if q_head is not None and qs:
                # Q(s, a_taken) toward the same return target the V head is fitted to. Without this the
                # selection in the rollout loop is made by an untrained head and IBRL degenerates into
                # "sometimes act like the clone at random".
                q_taken = torch.stack(qs, 1)
                q_loss = 0.5 * ((q_taken - ret) ** 2).mean()
                loss = loss + q_coef * q_loss
                with torch.no_grad():
                    stats.setdefault("q_loss", []).append(float(q_loss))
            # Every term that reads `b_obs` must be in this gate, or it is unreachable and silently
            # does nothing. That has now happened twice (bc_adaptive_frac, then align_coef), both
            # times caught only because tests/test_bc_regulariser.py asserts each term TRAINS.
            if (bc_coef > 0 or bc_adaptive_frac > 0 or bc_latent_coef > 0 or align_coef > 0) \
                    and bc is not None:
                # Same number of human windows as rollout chunks in this minibatch, so the term's weight does
                # not drift with the minibatch size. Fresh hidden per window (see zelda_rl/bc_demos.py).
                j = torch.randint(0, bc["obs"].shape[0], (min(chunks_per_mb, bc["obs"].shape[0]),))
                b_obs, b_act, b_first = bc["obs"][j], bc["act"][j], bc["first"][j]
                b_keep = bc["keep"][j] if "keep" in bc else torch.ones_like(b_act, dtype=torch.float32)
                bh = model.init_hidden(b_obs.shape[0])
                th = bc_teacher.init_hidden(b_obs.shape[0]) if bc_teacher is not None else None
                b_logits, s_hid, t_hid = [], [], []
                for k in range(b_obs.shape[1]):
                    lg, _, bh = model.step(b_obs[:, k], bh, b_first[:, k])
                    b_logits.append(lg)
                    if bc_teacher is not None:
                        s_hid.append(bh)
                        with torch.no_grad():
                            _, _, th = bc_teacher.step(b_obs[:, k], th, b_first[:, k])
                        t_hid.append(th)
                b_lg = torch.stack(b_logits, 1)                                   # (B, L, A)

                # The FILTER: cross-entropy is paid only on kept frames. An unweighted mean would quietly
                # include the dropped ones at weight 0 in the numerator and still divide by all of them,
                # scaling the term down instead of focusing it.
                w = b_keep.reshape(-1)
                ce = F.cross_entropy(b_lg.reshape(-1, b_lg.shape[-1]), b_act.reshape(-1), reduction="none")
                denom = w.sum().clamp(min=1.0)
                bc_loss = (ce * w).sum() / denom

                # ADAPTIVE weight: hold the BC term at a fixed fraction of the policy loss. At a constant 0.1
                # the term contributed ~0.44 against a policy loss of 0.01-0.1 and dominated the objective,
                # which is the measured cause of v15's decline (PROVENANCE 3.3, claim R47). This makes the
                # ratio the thing that is chosen, instead of a weight whose effect depends on both losses.
                coef_now = bc_coef
                if bc_adaptive_frac > 0:
                    with torch.no_grad():
                        coef_now = float((bc_adaptive_frac * pg.detach().abs() /
                                          bc_loss.detach().clamp(min=1e-6)).clamp(1e-5, 1.0))
                if bc_coef > 0 or bc_adaptive_frac > 0:
                    loss = loss + coef_now * bc_loss
                with torch.no_grad():
                    stats.setdefault("bc_loss", []).append(float(bc_loss))
                    hit = ((b_lg.argmax(-1) == b_act).float() * b_keep).sum() / b_keep.sum().clamp(min=1.0)
                    stats.setdefault("bc_acc", []).append(float(hit))
                    stats.setdefault("bc_coef_applied", []).append(float(coef_now))
                    stats.setdefault("bc_kept_frac", []).append(float(b_keep.mean()))

                # LATENT ALIGNMENT: match the GRU state the frozen BC policy carries on the same frames,
                # instead of forcing the same action. The two policies agree on 3% of actions (R46) while
                # both beating her, so aligning the representation is a weaker constraint than aligning
                # behaviour -- it shares what the clone encodes without overwriting what v15 decided to do.
                if bc_teacher is not None and bc_latent_coef > 0 and bc_latent_states == "demo":
                    if latent_mode == "delta":
                        lat = latent_delta_gap(model, bc_teacher, b_obs, b_first, b_keep)
                    elif latent_mode == "projected":
                        lat = projected_gap(model, bc_teacher, b_obs, b_first, b_keep, latent_proj,
                                            bc_latent_site)
                    else:
                        lat = latent_gap(model, bc_teacher, b_obs, b_first, b_keep, bc_latent_site)
                    loss = loss + bc_latent_coef * lat
                    with torch.no_grad():
                        stats.setdefault("bc_latent", []).append(float(lat))

                # MUTUAL alignment: `peer` is a SECOND TRAINABLE network (normally the clone). It keeps its
                # own BC objective while both encoders are pulled into a common representation space, so
                # nothing is frozen and the shared space has to explain the human's behaviour AND the return.
                # One-way distillation imposes the teacher's basis; this negotiates one.
                if peer is not None and align_coef > 0:
                    flat = b_obs.reshape(-1, b_obs.shape[-1])
                    al = align_loss(model.encode(flat), peer.encode(flat), b_keep.reshape(-1), align_kind)
                    ph, plog = peer.init_hidden(b_obs.shape[0]), []
                    for k in range(b_obs.shape[1]):
                        lg_p, _, ph = peer.step(b_obs[:, k], ph, b_first[:, k])
                        plog.append(lg_p)
                    p_lg = torch.stack(plog, 1)
                    p_ce = F.cross_entropy(p_lg.reshape(-1, p_lg.shape[-1]), b_act.reshape(-1),
                                           reduction="none")
                    p_ce = (p_ce * b_keep.reshape(-1)).sum() / b_keep.sum().clamp(min=1.0)
                    loss = loss + align_coef * al + peer_bc_coef * p_ce
                    with torch.no_grad():
                        stats.setdefault("align", []).append(float(al))
                        stats.setdefault("peer_acc", []).append(
                            float(((p_lg.argmax(-1) == b_act).float() * b_keep).sum()
                                  / b_keep.sum().clamp(min=1.0)))
            if kl_anchor is not None:
                loss = loss + kl_coef * kl_anchor
                stats.setdefault("kl_anchor", []).append(float(kl_anchor.detach()))
            if bc_teacher is not None and bc_latent_coef > 0 and bc_latent_states == "rollout":
                # Align where the policy actually goes, not where the demonstrator went: 13 human episodes
                # versus the distribution the policy is scored on.
                w1 = torch.ones_like(first)
                if latent_mode == "delta":
                    lat = latent_delta_gap(model, bc_teacher, obs, first, w1)
                elif latent_mode == "projected":
                    lat = projected_gap(model, bc_teacher, obs, first, w1, latent_proj, bc_latent_site)
                else:
                    lat = latent_gap(model, bc_teacher, obs, first, w1, bc_latent_site)
                loss = loss + bc_latent_coef * lat
                with torch.no_grad():
                    stats.setdefault("bc_latent", []).append(float(lat))
            if rvs is not None and rvs_coef > 0:
                # RETURN-CONDITIONED term. The windows carry each step's ACTUAL outcome in the observation,
                # so a fatal action is labelled with the low prompt that preceded it and the policy -- which
                # ACTS with the target prompt -- is pushed toward the action that produced the outcome we ask
                # for, not the average of winning and losing behaviour at that state.
                j = torch.randint(0, rvs["obs"].shape[0], (min(chunks_per_mb, rvs["obs"].shape[0]),))
                # LONG CONTEXT: sample a window, then replay its EPISODE from the start so the GRU arrives
                # carrying the episode's history instead of cold-starting every 60 steps. The label is
                # already episode-length (zelda_rl/rvs.py), and a cold context was the only short horizon
                # left in the loop. Earlier windows are run under no_grad -- they are context, not targets --
                # so the cost is forward passes, not memory.
                if "ep" in rvs and "pos" in rvs:
                    r_obs, r_act, r_first, r_w = [], [], [], []
                    prefix = []
                    for jj in j.tolist():
                        e_, p_ = int(rvs["ep"][jj]), int(rvs["pos"][jj])
                        same = (rvs["ep"] == e_).nonzero().flatten()
                        earlier = [int(x) for x in same.tolist() if int(rvs["pos"][x]) < p_]
                        prefix.append(earlier[-4:])          # up to 4 windows (4 s) of carried context
                        r_obs.append(rvs["obs"][jj]); r_act.append(rvs["act"][jj])
                        r_first.append(rvs["first"][jj]); r_w.append(rvs["keep"][jj])
                    r_obs = torch.stack(r_obs); r_act = torch.stack(r_act)
                    r_first = torch.stack(r_first); r_w = torch.stack(r_w)
                    rh = model.init_hidden(r_obs.shape[0])
                    depth = max((len(x) for x in prefix), default=0)
                    with torch.no_grad():
                        for d in range(depth):
                            idxs, rows = [], []
                            for b, pre in enumerate(prefix):
                                if len(pre) >= depth - d:
                                    idxs.append(b); rows.append(pre[len(pre) - (depth - d)])
                            if not idxs:
                                continue
                            sel = torch.tensor(idxs)
                            ctx = rvs["obs"][torch.tensor(rows)]
                            cf = rvs["first"][torch.tensor(rows)]
                            hh = rh[sel]
                            for k in range(ctx.shape[1]):
                                _, _, hh = model.step(ctx[:, k], hh, cf[:, k])
                            rh = rh.clone()
                            rh[sel] = hh
                else:
                    r_obs, r_act, r_first, r_w = (rvs["obs"][j], rvs["act"][j], rvs["first"][j],
                                                  rvs["keep"][j])
                    rh = model.init_hidden(r_obs.shape[0])
                rlog = []
                for k in range(r_obs.shape[1]):
                    lg_r, _, rh = model.step(r_obs[:, k], rh, r_first[:, k])
                    rlog.append(lg_r)
                r_lg = torch.stack(rlog, 1)
                rw = r_w.reshape(-1)
                r_ce = F.cross_entropy(r_lg.reshape(-1, r_lg.shape[-1]), r_act.reshape(-1), reduction="none")
                r_loss = (r_ce * rw).sum() / rw.sum().clamp(min=1.0)
                loss = loss + rvs_coef * r_loss
                with torch.no_grad():
                    stats.setdefault("rvs_loss", []).append(float(r_loss))
                    stats.setdefault("rvs_acc", []).append(
                        float((((r_lg.argmax(-1) == r_act).float()) * r_w).sum() / rw.sum().clamp(min=1.0)))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            if peer_opt is not None:
                peer_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            if peer_opt is not None:
                # One backward populates both graphs, so both optimisers step on the same alignment
                # gradient. That is what makes it mutual rather than a frozen teacher.
                nn.utils.clip_grad_norm_(peer.parameters(), max_grad_norm)
                peer_opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(pg.item()); stats["value_loss"].append(vl.item())
                stats["entropy"].append(ent.mean().item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > clip).float().mean().item())
        if target_kl is not None and stats["approx_kl"] and np.mean(stats["approx_kl"][-chunks_per_mb:]) > target_kl:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["skipped_minibatches"] = skipped
    y, yp = batch["ret"].reshape(-1), batch["val"].reshape(-1)
    out["explained_variance"] = float(1 - np.var(y - yp) / (np.var(y) + 1e-8))
    return out


def build_model(config: dict) -> nn.Module:
    """Rebuild the policy described by a checkpoint/run config (obs_version 1 -> MLP, 2 -> entity model)."""
    if config.get("recurrent"):
        return RecurrentEntityActorCritic(config["global_dim"], config["n_actors"], config["actor_dim"],
                                          config["n_actions"], hidden=config.get("hidden", 256), gru=config.get("gru", 256))
    if config.get("obs_version", 1) >= 2:
        return EntityActorCritic(config["global_dim"], config["n_actors"], config["actor_dim"], config["n_actions"],
                                 hidden=config.get("hidden", 256))
    return ActorCritic(config["obs_dim"], config["n_actions"], config.get("hidden", 256))
