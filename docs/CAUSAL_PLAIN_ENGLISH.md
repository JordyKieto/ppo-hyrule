# The causal world model, in plain English

A companion to [`CAUSAL_WORLD_MODEL.md`](CAUSAL_WORLD_MODEL.md), which is the scientific record. Same facts,
no jargon. If the two ever disagree, the other one is right.

---

## What we are trying to do

Teach a model to predict what happens next in Ocarina of Time, so an agent can practise inside its
imagination instead of inside the emulator. Earlier attempts squashed the whole game state into one 128-number
summary and tried to predict the next summary. That failed for about thirty documented reasons, and the
short version is that everything competed for the same 128 numbers and nothing won.

## The new idea

Stop treating the game as one big blob. Treat it as **objects** -- Link, the camera, the room, each Deku Baba,
each Skulltula -- where every object has **fields** (position, health, velocity) and objects of the same kind
behave the same way. Then learn which field actually affects which other field. That last part is a **causal
graph**, and it is the whole point of the paper we are following.

## Why the old approach was failing (the big finding)

We measured it directly. If you ask a simple straight-line formula to predict where Link moves next:

- give it **only his velocity** -> it does well
- give it **all 38 of Link's fields** -> it does *worse than useless*

Same data, same formula. The extra 33 fields don't add noise on top of the signal; they bury it. The model had
been handed **8,286 possible cause-effect links** when only a few dozen are real, so ~99.6% of what it was
reading was irrelevant.

That explains the whole history: a dream that barely moved, heads that learned nothing, and a 4.7-million
parameter model losing to a formula with no parameters at all.

## The plan, in four steps

| step | question | verdict |
|---|---|---|
| **P0** | If we hand-pick the right inputs, does it get better? | **PASS** |
| **P1** | Can we train it the way the paper does? | **PASS** |
| **P2** | Can it find the right inputs *by itself*? | **PARTIAL** |
| **P3** | Does the whole thing actually hold up? | not started |

### P0 -- does trimming the inputs help?

We told the model that Link's next position depends only on his position, velocity, speed, stick direction
and collision state, and hid everything else. Prediction quality jumped **54%**, landing exactly where an
independent estimate said the ceiling was.

So yes. Trimming is not a tweak, it is the mechanism.

### P1 -- train it the paper's way

The paper trains the model three ways at once: with all inputs, with a random 90% of them, and with only the
inputs it currently believes matter. The random one is the important one -- it teaches the model to cope when
information is missing, which is what lets us later ask "how much worse are you without velocity?".

We confirmed it works by checking the model does worst with no inputs, better with 90%, and best with all of
them. That ordering can only happen if the hiding is genuinely hiding things.

### P2 -- can it find the inputs itself?

**The good news: yes, and it found real physics.**

Asked what determines Link's next position, it answered **velocity** and **whether he's touching
ground or a wall** -- nothing else came close. Nobody told it that. It also worked out that swinging a sword
slows you down, that falling distance comes from velocity, and that a Deku Baba being targeted changes Link's
status flags -- which is it noticing that two different objects interact.

It also confirmed the diagnosis: only about **0.4% of the 8,286 possible links are real**.

**The bad news: the answers aren't repeatable.** Run the search twice on two different chunks of gameplay and
the two answers overlap by only a third. Most of what it "found" is coincidence from looking at too little
data -- and you can see it, because alongside the sensible answers it also claimed *"the height of the floor
under Link causes a Deku Baba to move"*, which is nonsense.

## Is that a critical failure?

**No, but it is a hard stop.** It looks like a sample-size problem, not a broken method: we asked it to detect
small effects from about 250 moments of gameplay, and the noise swamped the signal.

It is a stop rather than a nuisance because the next step is to feed the discovered links back into training.
Do that with a list that is two-thirds noise and everything afterwards is built on sand -- and it would
*look* like progress, which is worse than failing loudly.

## It was short of data, and two settings were wrong

Re-running the search on **10x more gameplay** fixed most of it. Two mistakes on our side:

1. We were searching using about 250 moments of gameplay. **The paper uses its entire memory -- 10,000 to
   50,000.** We were running the method at a fiftieth of its intended scale.
2. We treated the sensitivity dial as a fixed constant. **The paper tunes it per game** -- their own six
   environments use six different values spanning a 10x range.

Fixed both: agreement went from **19% to 66%**, and the edges we know are real came back -- including
"velocity moves Link", which had gone missing entirely at the small scale.

## Then we closed the loop

The real method alternates: search for the links, train the model using only those links, search again with
the improved model, and so on until the answer stops changing. Three rounds:

| round | links found | changed since last | agreement |
|---|---|---|---|
| 1 | 48 | all new | -- |
| 2 | 39 | 20 dropped, 11 added | 48% |
| 3 | 48 | **2 dropped**, 11 added | **74%** |

**It is settling.** Churn collapsed from 20 links changing to 2, and agreement rose to 74%. Meanwhile the
model itself improved every round -- its prediction of where Link moves next roughly doubled, and six more
quantities started beating the "assume nothing happens" baseline.

That is the phase working as intended: better links make a better model, which finds better links.

**And the biggest result: the model's own answer beat ours.** Earlier we hand-picked what we thought should
determine Link's position, from a separate analysis. The links the model discovered by itself did better --
84% of the "simple formula" benchmark against our hand-picked 75%. We are no longer better than it at this.

## The honest caveats

- **Damage is still not modelled at all.** Whether a Deku Baba loses health scores exactly zero, as it has
  since the very first attempt. Nothing in this phase touched it, and it is what the reward is made of.
- **Three rounds is not proof of convergence.** The paper runs up to a hundred. The trend is right; the
  claim "it converged" would be premature.
- **One earlier result was measured on easier data.** A bug meant several runs used one of our four gameplay
  collections rather than all of them, and that one has Link standing still more often. Their numbers are
  real but not directly comparable; every figure is now quoted against its own baseline.

## The honest scoreboard

**What is working:** the object-based structure, the trimming mechanism, the paper's training method, and a
discovery step that genuinely recovers physics nobody told it.

**What is not:** the discovery step is not yet reproducible, so we cannot trust the list it produces.

**What we have not tested at all:** whether any of this makes the imagined gameplay actually match reality
over many steps. The one time we measured that, the model barely moved at all -- it travelled 4% of the real
distance. That test gets rerun once the graph is trustworthy, and it is the one that decides whether any of
this was worth it.
