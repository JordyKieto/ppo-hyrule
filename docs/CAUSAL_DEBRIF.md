1. Your understanding, corrected in three places
Your summary is right in spirit. Three things are off, and each one matters for how we'd build it.

(a) The graph is not over entities. It's over (class, field) pairs. This is the whole compression trick and it's easy to miss.

There are exactly two kinds of edge (core/causal_graph.py):

local_matrices[C] — shape (n_state_fields(C), n_fields(C)). "Does field U of an object affect field V' of that same object?" e.g. Link.speed → Link.pos'.
global_matrix — one matrix, shape (n_state_fields_total, n_fields_total), indexed by (class, field) across all classes. "Does field U of any other object of class Cl affect V' of any object of class Ck?" e.g. Baba.pos → Link.hp'.
So the graph is ~6k booleans for our state and does not grow with the number of actors at all. A flat CDL over our 363-d state would test ~130k edges and re-test them whenever the actor count changes. (Those counts are illustrative — per AGENTS §3, they get derived from the schema in code and asserted, not hand-counted.)

(b) "Acyclic" is free, not a constraint you enforce. Theorem 3.2: if state variables transit independently given (S, A), the true graph is bipartite — every edge goes from a t variable to a t+1 variable, and there are no edges within S'. So the DAG is just an adjacency matrix from (state ∪ action) at t to state at t+1. Nothing to project, nothing to regularise.

And for us that assumption is unusually safe: OoT is a deterministic emulator. Given (s_t, a_t), s_{t+1} is a delta function. Factorisation across fields is exact, not an approximation. The paper's models pay a price for stochasticity that we largely don't owe.

(c) p and the supervision are two different loops.

p̂ is the factored transition distribution, p(S'|S,A) = Π_j p(S'_j | Pa(S'_j)). In code each state field gets a shared-per-class head f_{C.V} emitting distribution parameters — Normal(mean, scale), Categorical(logits) (core/ptype.py). Supervision is per-field negative log-likelihood, averaged per instance (their AILL, eq. 8) so the loss is invariant to how many actors are alive.
The graph is supervised separately, and this is the elegant part: no second model. During training the graph is randomly masked (each edge kept w.p. λ), so the model learns to answer "what would you predict without this parent?". Then discovery is just a conditional mutual information estimate — E[log f(V' | G_full) − log f(V' | G_minus_this_edge)] — thresholded at ε.
The ablation is the training objective. That is literally AGENTS.md rule §1 compiled into a loss function.

2. "Is it a layer?" — no, it's four modules and a mask multiply
From alg/model/object_oriented_model.py, with shapes:

piece	what it is
schema	plain Python: c = EnvObjClass('Baba'); c.declear_field('hp', vtype, role='state'). Not a tensor.
ClassAttributeEncoder	HeterogenousLinear — one Linear per field (fields have different widths) → (B, n_obj, n_field, d)
the causal graph	masked_retain(x, attrmask) — zero out non-parent fields' embeddings, then flatten n_field × d and Linear. That is the entire "causal" mechanism: a boolean broadcast over the field axis before a flatten.
class sharing	MultiLinear with a leading axis over fields; the object axis is just batched. Instances share weights because they're a batch dim.
causation symmetry	key-value attention with self masked out (m[:, i, i] = False). Query = object i's local encoding, keys/values = every other object's global encoding. Attention is permutation-equivariant, so Assumption 4.1 is architectural, not learned. Also how varying object counts work.
output	per-field DistributionDecoder → PType → log_prob
One thing you asked that the code answers differently than you'd hope: EnvObjClass(name, base=...) copies field declarations, it does not share weights. Cross-class invariants are expressible in the graph (the global matrix spans classes) but every class gets its own encoder/decoder parameters. So "different classes share core invariants" is not something this architecture gives you for free.

3. The reframe that actually matters: this deletes the latent
This is the part I want to push hardest on, because I think you're framing it as "another world-model architecture" and it isn't.

Go through our failure log. Every single failure is a failure of z_{t+1} = M(z_t, a_t) with one monolithic bottleneck:

§	failure	signature
22	dream worse than identity past h=25	position error 103 vs 50 for "Link never moves"
24	position and events compete for the 128 dims	w_pos 200 → trajectory 1.06×, damage 0%. Detach → damage back, walking 2.36×. Falsified as avoidable.
25	seed variance 1.38×–3.43×	swamps every reported effect
28	our encoder worse than random as a predictive surface	and the +0.742 "good" surface was a control echo + a clock
29	3 architectures, 3 negatives	latent moves 0.023–0.044/step; action explains 0.04–0.07 of a transition
30	discrete latents	predictor learns to copy z_t; skill exactly +0.000
b88868f	latent recursion	3× worse than do-nothing at H=10; |z| drifts 13.3 → 41.0
OOCDM has no bottleneck and no latent recursion. It predicts fields in their own units, one head per field, and the state is re-encoded each step. Mapping the failures onto that:

§24 dissolves structurally. Link.pos' and Baba.hp' are separate heads with separate parents and separate loss terms in different units. Weighting position cannot erase damage, because there is nothing for them to compete over. The w_pos tradeoff was an artifact of the bottleneck, not a property of the data — and §24's own numbers say both objectives are individually achievable.
§30's collapse can't happen the same way. There is no z to copy. (Be honest: per-field "predict no change" is still a strong baseline, and rare fields still need §23-style oversampling. But it moves from unavoidable capacity contention to a per-field loss weight, which is a tunable.)
b88868f's drift is bounded. Errors accumulate in world units, not in an unbounded |z|. And you already have the preview: your STATE-space arm beat the LATENT arm 2.3× at H=100 and pinned |z| at 12.8. OOCDM is the principled version of the experiment you ran yesterday — with a head for every field instead of only position, which is exactly the "camera, velocity and collision have no heads yet" caveat in that script's docstring.
The control echo becomes impossible by construction. In an OOMDP the action is a field with role='action' — it is an input to other fields and never a prediction target. There is no prev_action to leak. And if you did leak one, causal discovery would print the edge Link.prev_action → pos' with a huge CMI. The §28 disaster would have been a line of output instead of days.
§29's own data is the best argument for this. In those same runs, the action-conditioned heads on real latents work: moved AP 0.435 (6.8× base), enemy-hit AP 0.628, nearest-Baba distance MAE 26.8 vs a 64.0 mean-predictor. Those all read (z_t, a_t) → what happens, never z_t → z_{t+1}. The component of ours that works already has OOCDM's shape.
The retrospective, in one line: §16–§30 changed the network six times while holding the interface fixed — 128-d monolithic z, next-latent regression, a 363-d state flattened into one vector, one predictor. When six changes fail with the identical signature, the invariant across them is the hypothesis. AGENTS.md has a rule for "ablate before you scale"; it doesn't yet have one for "the thing you never varied is the thing that's wrong."

4. The schema sketch — and don't use the 483-d obs
Use the v3 state (363-d), not the 483-d PPO obs. Direct application of AGENTS §5 and §28:

the obs contains prev_action by construction (dims 22:26) — the exact leak that cost days;
it carries time_left (a clock) and the goal_* block (episode constants);
it's goal-conditioned and reward-shaped, i.e. task-specific — a world model shouldn't be;
it's poorer on physics: the actor table has no velocity, no fn_rel/state_idx, no timer. The v3 state has all of those.
And the v3 state is already an object table — state_layout.py is LINK_FIELDS_V3 + 16 × ENEMY_FIELDS. This is a schema declaration job, not a re-engineering job. That's the single biggest reason I think this fits.

Rough classing:

class	n	state fields	action fields
Link	1	pos, yaw, shape_yaw, vel, speed, hp, sf1/2/3, action_fn_rel, melee_state, bg_flags, wall_yaw, floor_height, dist_interact_wall, ztarget_timer, …	stick_dir, buttons
Camera	1	cam_yaw, cam_pitch, cam_eye	—
Baba (per actor id)	0–n	present, hp, pos, vel, yaw, yaw_to_player, fn_rel, state_idx, timer, focus	—
Item / Projectile	0–n	pos, vel, params, present	—
Room	1	scene, room	—
Three changes that are required, not optional:

targeted_row and arrow_row must die. They are indices into a row table, which is the one thing Assumption 4.1 forbids — swap two Babas and the prediction changes. Re-express as a Boolean field on each actor: Actor.is_targeted, Actor.has_arrow. This also removes a whole bug class: today rows are kept permutation-stable by sorting, so a Baba dying reshuffles rows and the MLP sees a discontinuity. Attention makes sorting unnecessary.
VTypes, not floats. hp → Categorical (§3 of our doc already measured HP-as-classification taking damage F1 0.286 → 0.458 → 0.746 — OOCDM gives it for free). action_fn_rel, state_idx, melee_state → Categorical. bg_flags → Booleans. Angles → sin/cos Normal pair or sector Categorical. Note §29: dist_interact_wall MAE 2.8 against a 2.9 mean-predictor means that continuous head is not learned at all — per-field vtypes are the lane to fix it.
Residual/delta parameterisation with zero-init output layers, so an untrained model is exactly do-nothing. That preserves AGENTS §1's first control row for free: skill 0 is the floor.
One bonus nobody's mentioned: Normal emits a learned scale. A field whose true driver we never read from RAM will show a large scale rather than silently regressing to a mean. That's a diagnostic we have never had.

And the loop closes cleanly: the dream state is a game state, so featurize/combat_features can run on it directly and the existing PPO policy consumes it with no latent anywhere. (Needs a check that all 483 dims are reconstructible from v3 — the episode-memory counters have to be integrated, not read.)

5. Where this could still fail here
Stated plainly, since I'd rather name these now than retract later:

It's a Markov model over declared fields. If a Baba's lunge is triggered by an animation counter we don't read, the class transition function is forced to guess. timer/fn_rel/state_idx are the right instinct; we may need more, read from decomp per AGENTS §3.
10 Hz decisions = 2 game-logic ticks per transition. The causal parents at 2-tick separation are wider than at 1 tick — influence propagates. It blurs the graph; it doesn't break it.
Nothing in the paper promises h=1800. Their evaluation is log-likelihood and short-horizon planning return. Our barometer (AGENTS §8) is real-action replay. Set the target at beating do-nothing per-field at h=10–100 in world units — exactly what eval_rollout_space.py already measures — and do not repeat §22's mistake of measuring the scaffolding.
Their envs are tiny (position as an int in [-99,99], health as an int). 43 messy Link fields is well outside what they tested. The CMI loop stays cheap because it scales in fields, not variables — but the per-class heads on rare events (damage = 0.13% of steps) are marginal at 120k transitions.
Seed variance doesn't go away (§25). But the graph gives a second measurement axis: edge-set stability across seeds is a much sharper "did training learn anything" test than a skill number.
6. Controls first, per AGENTS §1 — including two the paper doesn't have
Before any of this earns compute:

per-field do-nothing table on the existing dataset — which fields are even nontrivial?
untrained OOCDM, same architecture (§28's lesson; with zero-init residuals this should be exactly do-nothing, which is an assertion, not an experiment)
full graph G₁ vs discovered Ĝ vs empty graph — if empty scores the same, parents don't matter
shuffled-class control — assign actors to classes at random. If that does as well, class sharing contributed nothing. This is the OO-specific control and the paper doesn't run it.
permutation unit test — shuffle row order, predictions must be bit-identical. Attention guarantees it, so this is a test, not an experiment.
graph stability across ≥3 seeds (Jaccard on edge sets). If unstable, ε is making the decisions, not the data.
7. Staging, so we don't change four things at once
The risk with adopting OOCDM wholesale is that it changes the latent, the recursion, the input structure and adds causal discovery simultaneously — and a win wouldn't tell us which one mattered.

Stage 0 (hours, no training): write the OOMDP schema as a declaration, assert it round-trips the 363-d vector exactly. Produce the per-field do-nothing table.
Stage 1: per-field, class-shared, attention-based predictor with the full graph G₁ and no causal discovery. This isolates "object-oriented + per-field heads + state-space recursion" from "causality". Compare against do-nothing, untrained, and our best latent model on the same rollout metric.
Stage 2: add masking + CMI discovery and test whether Ĝ beats G₁.
I'd bet Stage 1 is where most of the win is, and Stage 2 buys generalisation (varying actor counts, unseen rooms) plus an explainability artifact — which is also the paper's own emphasis. Doing them together would hide that.

Want me to start on Stage 0 — the schema declaration plus the per-field do-nothing table? That's cheap, it's pure measurement, and it tells us whether the fields we'd be predicting are nontrivial before anything gets trained. I can also put this write-up in a doc if you'd rather have it alongside LATENT_WORLD_MODEL.md as §31.