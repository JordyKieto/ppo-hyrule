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

## What is happening right now

Re-running the search with 2x and 4x more gameplay to see whether the two runs start agreeing.

- **If agreement climbs to around 80%** -> it was just short of data. We use the bigger sample and carry on
  to the last step.
- **If it stays low** -> these effects cannot be reliably detected on the gameplay we have. That is a real
  result, and the answer would be to record more gameplay rather than to buy more compute.

## The honest scoreboard

**What is working:** the object-based structure, the trimming mechanism, the paper's training method, and a
discovery step that genuinely recovers physics nobody told it.

**What is not:** the discovery step is not yet reproducible, so we cannot trust the list it produces.

**What we have not tested at all:** whether any of this makes the imagined gameplay actually match reality
over many steps. The one time we measured that, the model barely moved at all -- it travelled 4% of the real
distance. That test gets rerun once the graph is trustworthy, and it is the one that decides whether any of
this was worth it.
