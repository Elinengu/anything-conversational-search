# S6 — Rerank Signal Investigations: What Shipped and What Didn't

**Files changed: `src/rerank.py`, `src/text.py`, `src/state.py`** (two new
signals + housekeeping); supporting edits in `src/facets.py`, `src/policy.py`,
`tools/sweep.py`, `tests/test_components.py`.

**Result:** public set 0.9128 → **0.9159** (200/200 kept), dev split 0.9207 →
0.9233, holdout 0.9010 → 0.9048, adversarial hard set 0.7914 → **0.7944** with
no bucket regressing and the worst bucket (`homogeneous_cluster`) up 4.7 MRR
points.

This is the record of the rerank-signal investigations. Two signals shipped
(negative facet evidence; association-preserving pair spans, plus a
word-boundary fix to span matching), one was implemented, measured, and removed
(profile-weighted agreement), one was rejected before a line of code was
written (budget/price closeness), three fragment-grouping variants were
prototyped and rejected, and the no-span early return was challenged and
upheld. §6-§8 add a later round that shipped no code at all: turn-1 exclusion
from facet extraction and multi-valued facet extraction were both measured and
rejected, and the explicit constraint ledger was specified operation by
operation and not built. That round also corrected the stated diagnosis in §1 —
the `focused_text()` guard is right, but not for the reason first recorded.
§9 is the one signal that was fully built and then removed: a neural
cross-encoder, which loses on every split and every setting, and whose optimum
weight is zero (its code now lives on the `semantic-rerank` branch). It also establishes the oracle ceiling — +0.043 public, +0.084
hard — that bounds every future reranking idea in this document. §10 dissects
where that ceiling actually lives (a pure tie-break regime in which the
retrieval score picks the impostor 33/33 and popularity picks the target
31/33) and carries the weight re-fit that follows from it.
§11 closes the other half of that tie-break and fails: document length is a real,
popularity-independent discriminator in the near-miss anatomy (33/37 on dev) and
does not survive contact with the adversarial set in any of three forms. §5 was
re-opened in the same round — change 12 looked like it should revive the no-span
rescore — and upheld a second time, on a dev split that does not move by a single
digit.
The negative results are documented with the same care as the positive ones —
knowing *why* a plausible signal does not work is what stops it being rebuilt
later. Three of the last four rounds shipped no code at all; that is the method
working, not the method failing.

**This doc is the decision log: what was tried and what the measurements said.**
Its companion `signal_descriptions.md` is the as-built spec — every shipped
signal, exactly how it is computed, and the house method for choosing a weight.
Read that one to know what the reranker *does*; read this one to know *why*.
Keep them in step: a signal added or a weight changed needs an edit in both.

All numbers were measured on the current tree in one session, so before/after
pairs are directly comparable. (The hard-set baseline here is 0.7914, not the
0.8007 quoted for the tail signal in `signal_descriptions.md` and
`category_tail_match.md` — the anchor-retrieval merge moved it. Hard-set numbers
are only comparable within a single measurement session.)

---

## 1. Shipped: negative facet evidence

### The gap

Every existing rerank term is positive evidence: span coverage, facet
agreement, category and tail alignment. None of them can tell apart a candidate
that *satisfies* "color: grey" from one that merely *doesn't contradict* it —
a black-only shirt matches "cotton shirt" spans exactly as well as a grey one
and loses nothing for being black. The customer's constraint rules products
out; the scorer only ever ruled products in.

### The signal

`_facet_conflicts(customer_facets, product_facets, product_text)` counts each
stated facet value the candidate contradicts, and the score subtracts
`facet_conflict_weight × conflicts`. A conflict requires all three of:

1. **The customer stated a value** for the attribute (via
   `extract_query_facets` — material, color, style, use_case, size).
2. **The candidate resolves that attribute too.** Silence is never punished:
   a product that mentions no colour at all is not in conflict with "grey".
   Missing data is not disagreement.
3. **The stated value appears nowhere in the candidate's text** as a
   word-bounded substring, aliases included (`grey`/`gray` is the only synonym
   pair in the facet vocabulary). This guards against `extract()`'s
   first-match-wins behaviour: a "black/grey reversible" belt extracts
   `color: black` but still contains "grey", and must not be penalised.

### The bug the first version had — and what it taught

The first implementation judged conflicts against `full_text()` (everything
the customer ever said). On the adversarial override bucket this *punished the
target for obeying the intent override*: after "actually, ignore black — I
need grey", the stale "black" was still in the transcript, still extracted
first, and grey-only products (the target among them) were penalised for
contradicting it. Measured cost: `generic_override` MRR 0.673 → 0.626.

The fix judges conflicts against `focused_text()` — the currently
authoritative turns only, identical to `full_text()` until an override fires.
That restored `generic_override` to exactly 0.673 and is now covered by a unit
test (`test_override_discards_stale_facet_for_conflict_scoring`).

### Correction: the fix is right, the diagnosis above was wrong

The paragraph above was written from the shape of the problem, not from the
sessions. Investigated properly (see §6), the mechanism is different:

* **The "stale black → grey" example never happens.** `behavior_for()` draws
  `old_value` and `new_value` from the *same target's* intent card, and across
  all 46 override sessions in the two eval sets, not one replaces an exclusive
  facet value with a different one. 25/30 public overrides are cross-slot
  (`"Buckle closure"` → `"leather"`); 4/30 are `feature → feature`; the single
  `material → material` case is `"Leather Loafers Women…"` → `"leather"`, the
  same value. The override in this evaluator is an *emphasis shift*, not a
  retraction.
* **Single-value extraction over full history picks a value the post-override
  turns contradict in 0 of 30 sessions.** There is no staleness to guard.
* **What `focused_text()` actually does here is drop turn 1.** The one
  regressing session is `hard_generic_override_08`, whose conflict comes from
  the opening line: `coarse_category()` emits the target's two most specific
  category levels, and those are drawn from the same vocabulary as the
  `style`/`use_case` facets, so `"I'm looking for Pants Casual"` extracts
  `style=casual` and punishes a target whose own style resolves to
  `regular fit`.

Proof that this is the whole effect: variants A and B in §6 — exclude turn 1
keeping `focused_text()`, and exclude turn 1 using full history — score
**bit-identically on all four splits**.

The general lesson stands, but restated: **negative evidence is more sensitive
to what counts as a constraint than positive evidence is.** A spurious positive
term merely boosts some wrong candidates; a spurious negative term actively
demotes the right one. Turn-1 category tokens are the spurious constraint here,
not stale ones.

### Measurements (weight 0.0 → 0.4)

| split | score off | score on | MRR off | MRR on |
|---|---:|---:|---:|---:|
| dev (120) | 0.9207 | **0.9224** | 0.862 | 0.870 |
| holdout (80) | 0.9010 | **0.9021** | 0.811 | 0.815 |
| public full (200) | 0.9128 | **0.9143** | 0.842 | 0.848 |
| hard (96) | 0.7914 | **0.7917** | 0.695 | 0.696 |

Hard set per bucket (MRR): every bucket flat or up, none down —
budget_only_signal 0.732 → 0.736, homogeneous_cluster 0.429 → 0.431, the other
four unchanged. Public hit rate stays 200/200, which was the disqualifying
check: a false conflict that demotes a true target would show up there first.

Weight choice: dev scores 0.9224 at 0.4 and 0.9226 at 0.8 — a plateau, and a
penalty term gets the smallest weight on it, so 0.4 shipped. At 0.4, one
conflict (−0.4) can reorder saturated ties but cannot overturn a genuine span
lead (each matched span is worth at least 1.12).

### An honest note on the original hypothesis

This signal was proposed to attack the `homogeneous_cluster` bucket (MRR
0.429), where positive signals saturate. Measurement showed that hypothesis
was wrong: bucketmates in a homogeneous cluster share the stated facets by
construction, so there is nothing to contradict *within* the cluster and the
bucket barely moves (+0.002). The gain comes from the general distribution
instead — cross-cluster impostors that match the spans but not the stated
colour or material. The signal earned its place on the measured gain, not on
the story it was designed around.

---

## 2. Shipped: association-preserving pair spans (+ word-bounded matching)

### The gap — spotted by reading one transcript

The customer's turn-2 message in `public_0020`:

> For that, what matters is: color: grey; Solid colors: 100% Cotton;
> Heather Grey: 90% Cotton, 10% Polyester; All Other Heathers: 50% Cotton,
> 50% Polyester.

`constraint_spans()` splits on `[.;:,\n]`, producing `heather grey`,
`90 cotton`, `10 polyester`, … — eight fragments. But the line is a *mapping*:
colour-variant → composition. Splitting on colons and commas severs "heather
grey" from *its* "90 cotton 10 polyester", so a candidate that pairs the
composition differently (an 80/20 heather grey) matches all eight fragments
exactly as well as the target. The fragments ask "does this product mention
90% cotton at all?"; the evidence in the message is "does it say that *about
heather grey*?"

### What was tried first — and rejected

The obvious diagnosis is that eight fragments from one line over-weight that
line, so three de-weighting variants were prototyped: per-utterance groups
scored as sum over sqrt(k) (public 0.9139), mean (0.9134), and best-single-span
(0.9025, −1.2 points). **All flat or worse.** The fragment gradient — target
matches 8/8, impostor matches 5/8 — is load-bearing and must not be
compressed. The fix is not to weaken the fragments but to *add* the
association they lost.

### The signal

`pair_spans()` (`src/text.py`) splits only on sentence separators (`.;\n` and
`" - "`), keeping colon/comma-joined content together, stripping the leading
simulator framing, minimum 3 words. `query_pair_spans()` (`src/state.py`)
applies the same turn-1/declined exclusions as `query_spans()` and drops
anything already emitted as a fragment. The reranker adds
`pair_weight (0.8) × matched pairs`, flat 1.0 per pair — the pair's evidential
value is the intact association, not its length. Catalog copy repeats these
blocks verbatim, so the joined form is still an exact substring of the target:
df("heather grey 90 cotton 10 polyester") = 511 products vs 612 for
"heather grey" alone.

Alongside it, one latent bug fixed: coverage tested `span in text` unanchored,
so `"90 cotton"` also matched `"190 cotton"` (1–4 catalog products per numeric
span). Both fragment and pair matching are now word-bounded — the product text
is token-joined, so padding with single spaces anchors every span at token
edges.

### Measurements

| step | public | public MRR | hard | hard MRR |
|---|---:|---:|---:|---:|
| baseline (conflict shipped) | 0.9143 | 0.848 | 0.7917 | 0.696 |
| + word-bounded matching | 0.9149 | 0.850 | 0.7917 | 0.696 |
| + pair spans (0.8) | **0.9159** | **0.851** | **0.7944** | **0.705** |

Hard set per bucket: the entire gain sits in `homogeneous_cluster`
(MRR 0.431 → **0.478**) and `generic_override` (0.673 → 0.680); every other
bucket is exactly unchanged, and no hit is lost anywhere. This is the bucket
the conflict signal could not move — bucketmates in a homogeneous cluster
share every *fragment* by construction, but they do not all pair the
compositions the same way, so the intact association is the discriminator that
survives saturation. `public_0020` itself, the session that exposed the gap,
goes from rank 4 to **rank 1**.

Weight: 0.4–1.5 score identically on dev (0.9233) and holdout (+0.0008);
0.8 sits mid-plateau per house convention.

---

## 3. Implemented, measured, removed: profile-weighted facet agreement

`user_profile.preference_tags` (e.g. `["material", "fit"]`) is unused in the
scored path. The idea: give extra credit when a facet agreement lands on an
attribute the profile says this customer cares about — a personalisation
prior, additive-only, mapped through the shared `TAG_HINTS` vocabulary.

| config | dev | holdout |
|---|---:|---:|
| baseline (conflict shipped) | 0.9224 | 0.9021 |
| profile_weight 0.1 | 0.9237 | 0.9013 |
| profile_weight 0.2 | 0.9217 | 0.9005 |
| profile_weight 0.4 | 0.9217 | — |

The dev gain at 0.1 does not reproduce on holdout (−0.0008, and monotonically
worse as the weight grows). The shape is a knife-edge, not a plateau — the
signature of fitting noise. Removed per the repo convention (dead options are
deleted, not kept at weight 0; the pool-local-rarity precedent).

Why it fails is worth keeping: **the profile is redundant with the
transcript.** The simulated customer already discloses their profiled
preferences verbatim during the session — a customer whose profile says
"material" *says the material out loud* by turn 2 or 3, and the facet-agreement
term already scores it. Re-weighting that same agreement adds variance without
adding information. A profile signal would only pay off where the profile
carries information the conversation lacks, which this evaluator's design
never produces.

One durable side effect kept: `TAG_HINTS` moved from `InfoGainPolicy` to
`src/facets.py` (module level) with identity entries added, so a literal tag
like `"material"` now maps to the `material` attribute. `InfoGainPolicy`
aliases it unchanged; profile tags that name an attribute directly previously
did nothing in its answerability prior.

### 3b. Same verdict, second try: rating-disposition modulated popularity

A different profile field, a fresh hypothesis: `average_prior_rating` /
`rating_style` are read by nothing and are genuinely orthogonal to the
transcript (a customer never says "I rate harshly"). Hypothesis — a generous
rater buys mainstream, well-reviewed products; a critical one shops away from
the bestseller shelf. Added term:
`popularity_profile_weight x rating_disposition x _popularity(product)`, with
`rating_disposition = clip(average_prior_rating - 4.0, -1, +1)` (the public
groups avg 5 / 4 / <=3 map to +1 / 0 / -1).

| weight | dev | holdout |
|---|---:|---:|
| 0.00 | 0.9233 | 0.9048 |
| 0.02 | 0.9235 | 0.9048 |
| 0.05 | 0.9254 | 0.9033 |
| 0.10 | 0.9285 | 0.9014 |
| 0.20 | 0.9293 | 0.9036 |

Dev climbs monotonically (+0.006); holdout is best at 0 and down ~0.003 across
the range — the identical knife-edge. Full public "+0.0018" at 0.10 is the dev
gain diluted by the holdout loss; the hard set "+0.005" is not independent (its
profiles are synthesised with a rating correlation by `tools/hard_cases.py`).
Code reverted. Two independent profile attempts have now failed the same way:
**the `user_profile` carries no non-redundant rerank signal on this evaluator.**
The disposition-to-popularity correlation is real in the 120 dev sessions and
noise in the 80 holdout ones.

---

## 4. Rejected before implementation: budget/price closeness

The idea — long recorded in `docs/team/hard_cases.md` as "budget is triply
dead" — was to parse the customer's "budget around $X" and score candidates by
price closeness, since the disclosed figure is the target's own price.

Measurement against the evaluator's own `intent_card()` killed it before any
code was written:

| population | cards containing a budget constraint |
|---|---:|
| public set (200 sessions) | **0** |
| hard set (96 sessions) | 4 — all rendered `budget around $—` (no number) |
| full catalog (50,000 intent cards) | 224 (**0.45%**), of which 110 parseable (**0.22%**) |

The root cause is structural: `intent_card()` appends the budget line *after*
all feature/detail candidates and keeps only the first four, so a budget
surfaces only for near-empty "degenerate card" products. The four hard-set
sessions that do carry one are exactly the four `never_retrieved` misses — the
target is not in the pool, so no rerank term of any kind can reach them. And
their dollar figure renders as "—" anyway.

Expected effect on a private 800 drawn the same way: ~2 sessions, likely
zero. The signal was not built. If a future evaluator version emits budgets
routinely, the design is on file: parse the last-mentioned dollar amount,
score 1.0 within ±10% with linear decay to ±50% (never exact-match — robust to
rounding), weight ~0.6.

---

## 5. Challenged and upheld: the no-span early return

`rerank()` returns the retrieval order untouched when `state.query_spans()` is
empty, and `query_spans()` excludes turn 1. The objection is fair on its face:
*"no spans" is not "no information"* — the opening always names a category, and
in buying sessions it also states a hard requirement. Three situations were
separated and measured.

### How much is even at stake

The early return fires on a turn that **actually shows a list** in only 20 of
595 public turns (3.4%) and 18 of 372 hard turns (4.8%). Most no-span turns are
turns 1-2, where the list is withheld anyway. That ceiling caps any possible
gain before a single variant is written.

### Case: the buying opening's hard requirement — a non-issue

`intent_card()` inserts the matched material at position 0, so the
`hard_constraints[0]` that the opening quotes is almost always a **single
word**: "cotton", "leather", "silk", "Material:alloy". `constraint_spans()`
requires `min_words=2`, so there is no span to extract even if turn 1 were
included, and nothing is being discarded: that requirement already reaches
ranking twice, as a BM25 query term and as a `material` facet via
`extract_query_facets`. What turn 1 *would* add is mostly framing noise
("i m looking for jewelry necklaces", "but i m still exploring"). The one real
exception is `intent_override`, whose opening carries genuine product copy
("stainless steel band", "buckle closure") — which is precisely the value the
customer later discards.

### Case: a later turn with no preference — already correct

Declined turns are held out of spans *and* of `full_text()`, while earlier real
constraints remain. Prior constraints therefore continue to drive ranking after
a decline; the early return fires only when nothing was **ever** disclosed.

### Case: vague opening with only a category — measured

Two variants, and their combination:

* **A** — when there are no spans, rescore with category signals only
  (retrieval + popularity + category + tail, no span/pair/facet/conflict terms).
* **B** — include turn-1 spans minus the leading framing fragment.

| variant | dev | holdout | public | hard |
|---|---:|---:|---:|---:|
| baseline (shipped) | 0.9233 | 0.9048 | 0.9159 | 0.7944 |
| A only | 0.9233 | 0.9070 | 0.9168 | 0.7938 |
| B only | 0.9237 | 0.9048 | 0.9162 | 0.7940 |
| A + B | 0.9237 | 0.9070 | 0.9170 | 0.7960 |

A alone is dev-flat; B alone is holdout-flat; only the combination moves both,
and the interaction has no mechanism behind it. The session counts show what
the combination actually is: across 296 sessions it moves **4 better / 1 worse
on public** and **5 better / 2 worse on hard**. The holdout +0.0022 is
arithmetically one session going from rank 3 to rank 1 — an order of magnitude
below the ~0.02 noise floor `tools/sweep.py` documents for that split. Per
bucket, A+B trades `degenerate_card` (MRR 0.527 -> 0.555) against
`generic_override` (0.680 -> **0.664**).

The refinement the bucket split suggests — keep turn-1 spans but drop them once
an override fires — was tested and made that bucket **worse** (0.623), not
better. That independently re-confirms the reasoning already recorded in
`apply_override` (src/state.py): in this evaluator the discarded preference is
still derived from the target product, so erasing it destroys usable signal.

**Not shipped.** With hit@10 already at 1.0, a change that moves 5 of 296
sessions on an unexplained interaction, while regressing a bucket, is the
overfitting signature this document exists to catch. The early return and the
turn-1 exclusion stand.

### Re-opened after change 12, and upheld a second time

Change 12 raised `popularity_weight` 0.02 -> 0.4, which looked like it should
revive variant A: the whole point of A is to rescore with the non-span signals,
and popularity is the strongest of them. `public_0198` is the motivating case -
its four disclosed constraints (`leather`, `color: black`, `PU`, `Imported`) are
all single words, so `constraint_spans` (min 2) and `pair_spans` (min 3) both
return nothing for the entire session, the early return fires on every turn, and
the pool is served in raw RRF order. The target sits at pool rank 51 and only
surfaces at **turn 9** via the elimination scan. Replayed with A, it ranks 1 at
turn 4 - and the margin is 0.003, entirely popularity (4.7*/4718 ratings against
the impostor's 2.8*/6). At weight 0.02 the target loses that comparison; at 0.4
it wins it.

Measured at the shipped 0.4, on the splits:

| | dev (120) | holdout (80) | hard (96) |
|---|---:|---:|---:|
| baseline | 0.941757 | 0.913619 | **0.801978** |
| variant A | **0.941757** | 0.918765 | 0.799968 |

**Dev is bit-for-bit unchanged.** The regime change did not revive it: A was
dev-flat at 0.02 and is still exactly dev-flat at 0.4. Everything that moves is
on the holdout (4 sessions better, 0 worse - `public_0145`, `public_0149`,
`public_0162`, `public_0198`) against 5 hard-set sessions worse
(`hard_degenerate_card_{01,07,10,15}`, `hard_generic_override_13`). A change
whose selector split does not move while the gate split gains is not a candidate;
it is the thing the split exists to detect.

The natural explanation - that popularity causes the hard-set loss, because
`hard_cases.py` draws thin, unreviewed targets - was tested and is **wrong**:

| variant A on the hard set | score |
|---|---:|
| baseline (early return) | **0.801978** |
| popularity x1.0 | 0.799968 |
| popularity x0.5 | 0.797753 |
| popularity x0.0 | 0.799299 |

Damping the prior makes it worse. The cost is in `_tail_match` /
`_category_match` / `_facet_agreement` / `_facet_conflicts` firing as the *only*
evidence on `degenerate_card`, where the customer has disclosed almost nothing.

`public_0198` is worth naming because it is the honest hard case for this
document's own method: a real nine-turn efficiency cost, with a real fix, that
the protocol nonetheless refuses. The untried variant, if anyone re-opens this a
third time, is a **separability gate** - rescore only when the non-span signals
actually discriminate within this pool, rather than whenever spans are absent -
which is selective reranking / query-performance prediction in its cheapest
form. It needs a dev-side win to justify the work. There isn't one yet.

---

## 6. Measured and rejected: turn-1 exclusion from facet extraction

### The hypothesis

`extract_query_facets` runs over the whole conversation, including turn 1. But
turn 1 is the simulator's own framing — `initial_message()` renders
`coarse_category()`, i.e. the target's two most specific category levels — and
those levels are drawn from the same vocabulary as the `style` and `use_case`
facets (`casual`, `athletic`, `running`, `winter`, `work`, `outdoor`, …). So the
category is being read as a style/use_case *constraint*.

`query_spans()` already excludes turn 1 for exactly this reason
(`src/state.py:126-133`: "the opening line is the simulator's own framing … not
quoted product copy"). The facet path never got the same guard. Turn 1 is also
already consumed correctly as category evidence by `_tail_match`
(`tail_weight=0.8`) and `_category_match` (`category_weight=0.4`), so the facet
path double-counts it.

The static evidence looked strong:

| | public (200) | hard (96) |
|---|---|---|
| turn-1 framing injects a facet no later turn states | 46 slot-instances (26 `use_case`, 18 `style`, 1 `material`, 1 `size`) | 19 |
| target wrongly penalised by conflict, turn 1 **included** | 8 | 5 |
| target wrongly penalised by conflict, turn 1 **excluded** | **0** | **0** |

### Measurements

| variant | dev (120) | holdout (80) | public (200) | hard (96) |
|---|---|---|---|---|
| baseline (`focused_text`) | **0.9233** | **0.9048** | **0.9159** | **0.7944** |
| A: −turn1, keep focused | 0.9226 | 0.9035 | 0.9150 | 0.7944 |
| B: −turn1, full history | 0.9226 | 0.9035 | 0.9150 | 0.7944 |
| C: full history, +turn1 | 0.9233 | 0.9048 | 0.9159 | 0.7920 |

Hit rate stays 1.000 on every public split throughout.

### Why the hypothesis failed

The 8-and-5 figures above count only the *harm* — targets wrongly penalised —
and never the benefit. The category framing is a genuine constraint: "Athletic
Shoes Running" really does mean the customer wants athletic/running items, and
demoting a formal-shoe impostor on that basis is correct. The impostor
demotions outweigh the target penalties, so removing turn 1 loses score
(−0.0009 public, −0.0013 holdout) despite removing every false penalty.

C reproduces the original regression the `focused_text()` guard was introduced
for (−0.0024 hard). **Baseline is the optimum of all four**: keep turn 1 on the
85% of sessions with no override, where it helps; drop it on override sessions,
where it hurts. No code change — only the wrong explanation in §1 needed fixing.

A ≡ B bit-identically, which is what proves `focused_text()` is doing nothing
in this path except filtering turn 1.

---

## 7. Measured and rejected: multi-valued facet extraction

### The hypothesis

`extract_query_facets` and `extract` both use `pattern.search()` — **first match
wins, one value per attribute**. A customer disclosing `"Heather Grey: 90%
Cotton, 10% Polyester"` yields `material=cotton`; the polyester is invisible to
both `_facet_agreement` and `_facet_conflicts`. Measured on the public set,
**89 of 200 sessions state more than one distinct material** and lose all but
the first; 98 of 200 have at least one multi-valued slot.

This also predicts the open-ended-slot problem directly: `"Water Resistant"`
should not displace `"machine washable"` merely because both land in `feature`.

Variants: hold each slot's stated values as a list; count agreement when the
product's value is anywhere in that list; count a conflict only when **none** of
the stated values appear in the candidate's text (plus a `fractional` variant
scoring `absent / stated` to preserve discrimination).

### Measurements

| variant | dev (120) | holdout (80) | public (200) | hard (96) |
|---|---|---|---|---|
| baseline | **0.9233** | 0.9048 | **0.9159** | 0.7944 |
| multi agree only | 0.9225 | 0.9045 | 0.9153 | 0.7921 |
| multi conflict only | **0.9233** | **0.9048** | **0.9159** | **0.7949** |
| multi both | 0.9225 | 0.9045 | 0.9153 | 0.7919 |
| multi both, fractional | 0.9225 | 0.9050 | 0.9155 | 0.7932 |

### Why the hypothesis failed

**Multi-value agreement is the harmful half, consistently** (−0.0006 public,
−0.0023 hard). It loosens the match test from "the product's value equals the
customer's" to "the product's value is somewhere in the customer's list", so it
fires for strictly more candidates. More coverage, less discrimination — a
diluted signal, not a richer one. The same reason the fragment de-weighting
variants in §2 failed: the gradient is load-bearing.

**Multi-value conflict is exactly neutral** on dev, holdout and public, and
+0.0005 on hard — well inside the documented ~0.02 noise floor. It was not
shipped despite being harmless: the house rule is that dead options are deleted
rather than parked, and a code path no measurement justifies does not earn its
place. The robustness argument for the private set (a multi-material disclosure
judged against one material) is real but unevidenced, and was weighed against
the added surface and rejected.

Note that `_facet_conflicts` already carries a narrower fix for the same
first-match-wins problem on the *product* side — guard 3, the substring check,
which is why a "black/grey reversible" product is not punished. That guard is
cheap and measured; generalising it to the query side is not.

---

## 8. Not pursued: the explicit constraint ledger

A typed `Constraint` ledger (slot, value, turn, polarity, status) with
CARRY / UPDATE / ADD / DELETE / DONTCARE / NEGATE operations was specified and
each operation measured against this evaluator before any of it was built:

| Op | Status | Verdict |
|---|---|---|
| `CARRY` / `ADD` | `full_text()` already accumulates every non-declined turn | already built |
| `DONTCARE` | `NO_PREFERENCE_CUES` → `Utterance.declined` + `dead_attributes`, read by both policies | already built |
| `UPDATE` | global down-weight (`apply_override`) | **no case to fire on** — see the §1 correction: 46/46 override sessions are emphasis shifts, not retractions |
| `DELETE` / `NEGATE` | absent | `customer_reply()` only ever *adds* constraints; the simulator never retracts or negates |
| multi-valued slots | absent | measured flat or worse — §7 |

Four of the six operations are already implemented or provably unable to fire,
and the two that could be tested end-to-end measured flat or worse. Building the
ledger would have been new surface area carrying no signal.

This also closes two logged open items as not worth pursuing: "fix or delete the
override weight" (`docs/team/hard_cases.md`) — there is nothing for a weight to
express — and the `PRE_OVERRIDE_WEIGHT` tuning / slot-erasure ideas in
`IMPLEMENTATION.md` §S3.

---

---

## 9. Built, measured, and removed: neural cross-encoder reranking (S6b)

> **Status: removed from the working branch.** The stage was never used — off by
> default, and a no-op even when enabled without weights — so the code was
> deleted rather than carried as an unused dependency-bearing module. It is
> preserved in full on the **`semantic-rerank`** branch (`src/semantic.py`,
> `tools/fetch_model.py`, `requirements.txt`,
> `docs/team/semantic_rerank_setup.md`), reproducible with the commands in
> §12. Everything below is the measurement, which is the reason the section
> stays.

### The ceiling, measured first

Every signal above is lexical. Before writing a semantic one, an **oracle**
reranker — target forced to rank 1 whenever it is anywhere in the pool — fixed
how much any reranking work could possibly be worth:

| | dev (120) | holdout (80) | public (200) | hard (96) | generated |
|---|---|---|---|---|---|
| baseline | 0.9268 | 0.9096 | 0.9199 | 0.7981 | 0.9197 |
| oracle | 0.9638 | 0.9620 | 0.9631 | 0.8823 | 0.9590 |
| **gap** | **+0.037** | **+0.052** | **+0.043** | **+0.084** | **+0.039** |

Worth having, and also the whole prize. The addressable population is small and
hard: at the first slate, 100 of 142 public sessions already have the target at
rank 1, and 25 of the remaining 42 need a rank-2-to-4 promotion among
near-identical cluster-mates.

### The gate, also measured first

The proposal gated the model on ambiguity:
`tied_span_leaders >= 8 or same_facet_cluster >= 10 or distinctive_span_count == 0`.
Implemented and measured at the first slate turn:

| | fires | mean RR firing | mean RR quiet |
|---|---|---|---|
| public | 104/142 (73%) | 0.774 | 0.987 |
| hard | 50/69 (72%) | 0.576 | 0.947 |

It discriminates well and gates badly — at 73% it is an always-on stage with
extra steps. Shipped thresholds are tighter (`tied_leaders >= 15`,
`facet_cluster >= 12`, two of three conditions) and fire on 28% of rerank calls.

### The signal

`src/semantic.py`: `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M parameters,
Apache-2.0), fused with the symbolic ranking by RRF rather than score addition —
logits are uncalibrated and unbounded while one matched span is worth ~1.12, so
adding them puts an arbitrary scale in charge. Runtime is `onnxruntime` +
`tokenizers` over a 23.2 MB int8 graph; upstream publishes ONNX exports, so no
torch, no transformers, no export step.

### Measurements

| variant | dev (120) | hard (96) | sec |
|---|---|---|---|
| **off** | **0.9268** / 0.885 | **0.7981** / 0.725 | 26 |
| on, semantic weight 0.7 | 0.9211 / 0.872 | 0.7944 / 0.713 | 347 |
| on, semantic weight 0.3 | 0.9249 / 0.882 | 0.7959 / 0.717 | 341 |
| on, depth 20 | 0.9236 / 0.879 | 0.7940 / 0.711 | 146 |

Latency: mean turn 30.7 ms → 389.8 ms, p95 73.7 → 1347.8 ms, max 1.48 s.

### Why it failed

Read the weight column downward: 0.7 → 0.3 → 0.0 recovers the baseline
monotonically. **The optimum weight is zero**, which is what a signal carrying no
usable information looks like — there is no threshold to tune toward, and no
plateau to sit mid-way along.

The mechanism is one number. On the 162 fired turns where the target was in the
rescored head, fusion moved it **up 46 times and down 74**, mean rank 7.63 →
8.77. The model is anti-correlated with the target here, not miscalibrated.

The likely cause is domain mismatch, which was flagged as the principal risk
before building and turned out to be the one that mattered. MS MARCO pairs a
natural-language question with a prose passage. Here the query is simulator
boilerplate (*"For that, what matters is: full grain leather; buckle closure"*)
and the document is a token-joined blob of title, features, description and
details. The task it was handed — separating cluster-mates that share every
stated facet — is also the hardest discrimination in the pool, which is exactly
why it was chosen and exactly why a mismatched model has nothing to add.

### Why it was removed rather than parked

It was first kept in the tree, disabled, as a reproducible artifact. That was
the wrong call by this repo's own rule — dead options are deleted, not parked —
and the stage was doubly dead: off by default, and a silent no-op even when
enabled, because the weights it needs are gitignored and absent on any clean
clone. It was never on the scored path for a single evaluation. Carrying it
meant carrying a `requirements.txt`, a download tool, an ONNX runtime import
path and a macOS teardown quirk for code that never ran.

So the code moved to the **`semantic-rerank`** branch and this section kept the
numbers. Nothing measured is lost: the branch is one `git checkout` away and
§13 carries the commands. The oracle ceiling this work established (+0.043
public, +0.084 hard) outlived the code and directly produced §10.

### One sub-signal that was deleted

"Protect strong lexical evidence" — a candidate matching a span no other
candidate matches is never demoted by the neural score. Built first as a `+1.0`
bonus on the fused score, which was wrong: RRF scores here top out near 0.028,
so the constant did not protect, it promoted, hoisting a unique-span holder from
symbolic rank 40 to rank 1. Rewritten as a rank clamp. Then measured: it fires on
**0 of 8750** candidates examined, because inside a pool retrieved by those very
spans no span is unique. Removed rather than kept as an inert flag.

---

---

## 10. The tie-break regime: near-miss anatomy and the weight re-fit

### The measurement that frames everything else

After change 11 fixed the ceiling (+0.043 public / +0.084 hard, §9), the next
question was *where the remaining headroom lives*. Answer: dissect every
near-miss session — target at rank 2-10 at a slate turn — comparing the target
against the impostor holding rank 1, feature by feature.

Public set, 33 near-miss sessions (hard set, 15, directionally similar):

| feature | target mean | impostor mean | tgt>imp | tgt<imp |
|---|---|---|---|---|
| span coverage | 3.027 | 3.027 | 0 | 0 |
| pair coverage | 0.424 | 0.424 | 0 | 0 |
| category / tail / conflict | tied | tied | 0 | 0-1 |
| **retrieval score (norm.)** | 0.759 | **0.922** | **0** | **33** |
| **popularity** | **0.752** | 0.363 | **31** | 2 |
| text length (tokens) | 195 | 126 | 25 | 8 |
| match density | 0.050 | 0.088 | 6 | 27 |

Three facts:

1. **Every lexical signal is exactly tied, 33/33.** The regime holding all
   remaining public headroom is a pure tie-break regime; no new lexical
   evidence separates these candidates.
2. **The tie is broken by the retrieval score, and it points at the impostor
   33/33.** Mechanism: BM25 length normalization. The impostor is a thin
   listing (126 vs 195 tokens) where the same matched words are a larger
   share of the document (density 0.088 vs 0.050), so BM25 rates identical
   evidence higher.
3. **Popularity points at the target 31/33** — the target is a real purchase,
   so it tends to be a reviewed, documented product — but at weight 0.02
   against retrieval's 1.0 it is drowned 50:1. A signal right 94% of the time
   in this regime loses to one wrong 100% of the time.

The same table kills three plausible new signals *before implementation*:
title boost (`title_hits` favours the impostor 11:6 — thin listings are mostly
title), match density (impostor 27:33), and span contiguity (exactly tied).
Recorded here so nobody builds them.

### Step 1 — bracketing the two implicated weights (read-only, shipped as sweep rows)

| variant | dev (120) | holdout (80) | public (200) | hard (96) / hit |
|---|---|---|---|---|
| base r1.0 p0.02 | 0.9268 | 0.9096 | 0.9199 | 0.7981 / .885 |
| p0.10 | 0.9331 | 0.9122 | 0.9248 | 0.7985 / .885 |
| p0.30 | 0.9422 | 0.9095 | 0.9291 | **0.8047 / .896** |
| p0.50 | **0.9441** | 0.9131 | **0.9317** | 0.8016 / .896 |
| r0.50 p0.02 | 0.9282 | 0.9072 | 0.9198 | 0.7987 / .885 |
| r0.70 p0.30 | 0.9430 | **0.9147** | 0.9317 | 0.7990 / .896 |

Raising `popularity_weight` alone is a plateau from 0.30 to 0.50: public
+0.012, holdout never regresses beyond noise, hard up, and **hard hit
0.885 → 0.896 — a converted miss**, not a reshuffle. Public hit stays 200/200
throughout. One weight captures ~27% of the oracle ceiling. Muting
`retrieval_weight` alone does little: the fix is amplifying the signal that is
right in this regime, not merely quieting the wrong one.

`tools/sweep.py` rows: `pop002 / pop010 / pop030 / pop050`.

### Step 2 — direct-metric fit of the whole mixture (`tools/fit_weights.py`)

The reranker is a linear feature-based model in the Metzler & Croft sense
(Information Retrieval 10:257-274, 2007), and their estimator — coordinate
ascent directly on the IR metric — fits it without gradients, which matters
because the technical score is non-smooth. The repo proposed learned weights
in four places (IMPLEMENTATION.md §S6 ideas, ideas.md idea 4, hard_cases.md
item 6); this is that idea, now with a mechanism-level reason the hand weights
are wrong.

Protocol, per the house rules above: fit on **dev only**; `span_weight` stays
1.0 as the definitional unit; holdout is run once, on the final vector, as a
gate; what ships is a rounded, plateau-checked point, never the dev argmax.
Each candidate vector costs a full dev evaluation (~11-26 s) because the
session transcript is weight-dependent — the session ends at the first hit and
the confidence gate reads scores — so cached feature vectors cannot shortcut
the objective.

### Step 2 outcome — the fit, the gates, and what actually shipped

The fit converged in 168 evaluations (29 min): dev 0.9268 → 0.9520, moving
`popularity 0.02→0.8`, `retrieval 1.0→0.1`, `facet 0.3→0.5`, `tail 0.8→1.2`,
`conflict 0.4→0` (category and pair never moved — the hand values were already
on their plateaus). The one-shot gates on the argmax:

| vector | dev | holdout (sealed until now) | public | hard / hit |
|---|---|---|---|---|
| baseline | 0.9268 | 0.9096 | 0.9199 | 0.7981 / .885 |
| raw argmax | 0.9520 | **0.9290** | **0.9428** | **0.7824** / .885 |
| argmax, conflict kept 0.4 | 0.9500 | 0.9276 | 0.9410 | 0.7825 / .885 |

Holdout **confirmed the direction** (+0.019 on data the fit never saw) — this
was not dev overfit. But the argmax **regresses the adversarial set by 0.016**
(MRR 0.725 → 0.675): `hard_cases.py` deliberately draws targets from thin,
unreviewed catalog regions where popularity is neutral (the anatomy's hard
column: 6:9), so `popularity 0.8 / retrieval 0.1` overshoots for that
distribution. Two side findings: holdout is indifferent to zeroing the conflict
penalty (Δ0.0014), so by the smaller-change rule the measured signal stays at
0.4; and tempered vectors (`pop .5-.8` with `retrieval .5-.7`) all still left
hard below baseline.

**Shipped: the one-weight change, `popularity_weight 0.02 → 0.4`** — the only
vector under the pre-declared rule (holdout keeps gains, hard ≥ baseline,
smallest departure wins):

| | dev | holdout | public | hard / hit |
|---|---|---|---|---|
| before | 0.9268 | 0.9096 | 0.9199 | 0.7981 / .885 |
| **pop 0.4** | **0.9418** | **0.9136** | **0.9305** | **0.8020 / .896** |

Every split up, a hard-set miss converted, public hit 200/200 kept, and no
public scenario regresses (boundary MRR 0.704 → 0.86). Official evaluator:
public **0.930502**, hard **0.801978**. Plateau: 0.1 / 0.3 / 0.4 / 0.5 are all
≥ baseline on all four splits, so 0.4 sits mid-plateau with both neighbours
measured. The argmax is kept reproducible as the `weights_argmax` sweep row —
a documented trade (public +0.012 more, hard −0.020) that the private set's
uniform sampling might justify, but the no-bucket-regresses rule does not.

## 11. Built, measured, and rejected: the document-length tie-break

§10 named BM25 length normalization as the mechanism that hands rank 1 to the
impostor — the thin listing where the same matched words are a larger share of
the document — but fixed only the other half of the tie-break, by raising
`popularity_weight`. This section is the attempt to fix the length half, and the
reason it does not ship.

### The signal is real, and it is not popularity in disguise

Near-miss anatomy on the **dev split only** (target at rank 2-10 at a slate
turn, against the impostor holding rank 1), re-run on the post-change-12 tree:

| | dev (n=37) |
|---|---|
| popularity picks the target | 31/37 |
| **text length picks the target** | **33/37** |
| mean tokens, target vs impostor | **221.4 vs 103.8** |
| of the 6 near-misses popularity gets wrong, length rescues | **5** |

The two are complementary rather than redundant: their deltas correlate 0.418,
and together they cover 36 of 37. This is exactly the shape of
Singhal, Buckley & Mitra's pivoted length normalisation (SIGIR 1996) — here
P(relevance) *rises* with length while BM25's normalisation makes P(retrieval)
*fall*, which is the gap pivoting exists to close.

The same anatomy killed one candidate signal before implementation:
**category-path precision**. `public_0002` makes it look irresistible — the
target is `Men > Accessories > Belts` while the impostors are
`Sport Specific Clothing > Golf > Women > Accessories > Belts`, so penalising
category levels the customer never named would separate them, and neither
`_category_match` nor `_tail_match` can (both score every one of them 2.0). It
ties on **34 of 37** dev near-misses. One vivid session is not a signal.

### Three forms, all measured, all rejected

**Additive log-length** (`+ w · log10(len)/3`, unconditional): directionally
positive and numerically negligible. Normalised log-length separates target from
impostor by ~0.09 against a retrieval gap of ~0.5, so even at w=0.4 it moves
0.036. Not enough to matter.

**Percentile within the pool** (unconditional), which fixes the scale by
spreading the same ordering across [0, 1]:

| | dev | holdout | hard |
|---|---:|---:|---:|
| baseline | 0.941757 | 0.913619 | **0.801978** |
| w=0.2 | 0.942375 | 0.915806 | 0.795164 |
| w=0.5 | 0.936694 | 0.916896 | 0.794613 |
| w=1.0 | 0.931583 | 0.903760 | 0.787850 |

Best dev result +0.0006, inside the noise floor, against −0.0068 on the hard
set, and no plateau — dev falls off a cliff between 0.2 and 0.5.

The mismatch is in the reading, and naming it is the useful part. The anatomy
measures length **conditionally**: among near-misses, where every content signal
is already tied, the longer document is the target. Applying it *unconditionally*
asserts something stronger and false — "longer is better" — and the hard set,
whose targets are deliberately thin, is precisely where that is wrong.

**Percentile applied only inside a content tie**, which is the form the anatomy
actually supports. Content evidence (`span + pair + facet + category + tail −
conflict`) is summed separately from the priors; candidates whose content
subtotal is within `length_tie_tolerance` of the leader's get
`length_weight × pool_length_percentile`, everyone else gets nothing. This works
on the motivating session — `public_0002` (a dev session) moves the target from
rank 9 to 6 at w=0.1 and to 3 at w=0.3 — and on dev it is worth about +0.0015
across the whole bracket. On the hard set at exact-tie tolerance:

| length_weight (tolerance 0.0) | dev | hard | hard hit |
|---|---:|---:|---:|
| 0.0 (baseline) | 0.941757 | **0.801978** | 0.896 |
| 0.05 | 0.943042 | 0.800511 | 0.896 |
| 0.08 | 0.943146 | 0.800381 | 0.896 |
| **0.10** | 0.943229 | **0.805064** | 0.896 |
| 0.12 | 0.943354 | 0.799075 | 0.885 |
| 0.15 | 0.943979 | 0.795659 | 0.885 |
| 0.20 | 0.942604 | 0.791179 | 0.885 |

**This is why it does not ship.** One point clears the hard-set gate and both its
neighbours fail, on either side. Dev is flat-to-rising across the whole range
while hard swings non-monotonically by 0.014 — a spread of roughly one session on
a 96-session set. `w=0.10` is an argmax on noise, not a plateau, and the rule
this document has applied since change 12 is that a rounded, plateau-checked
value ships and an argmax does not. A looser tolerance (0.5, roughly half a
matched span) is worse still: every weight from 0.05 up regresses hard, and
hit@10 drops 0.896 → 0.885 from w=0.1.

### What would have to change

The term reaching the reranker as `retrieval_score` is an **RRF score**
(`src/retrieval.py`), i.e. a function of BM25 *rank*, not BM25 itself. Rank
fusion has already discarded the score magnitudes that pivoted normalisation
operates on, so every form above is a correction bolted onto fused ranks rather
than a corrected scorer. The faithful version — and the one the literature
actually describes — is to recompute a length-corrected BM25 over the
300-candidate pool inside the reranker, where `k1` and `b` are yours to choose;
SQLite FTS5's `bm25()` fixes them and exposes no knob. That is ~30 lines and
still stdlib-only. It is the one untried route, and it is the right one to try
before anybody re-derives the additive prior a fourth time.

The code for the three rejected forms is not kept: dead options are deleted, not
parked. Rebuilding it is a `length_weight` / `length_tie_tolerance` pair on
`RerankConfig`, a pool-local length percentile, and splitting the existing
`total` in `rerank()` into a content subtotal plus priors so the tie can be
tested without the priors masking it.

## 12. Housekeeping shipped alongside

* Deleted the dead commented-out block in `_facet_agreement` (the pre-lowercase
  `extract()` call kept as a `'''…'''` string).
* `extract_query_facets(state.full_text())` was recomputed for **every
  candidate** (300× per turn); it is now computed once per `rerank()` call and
  passed in. Verified bit-identical before any signal work: public run exactly
  0.912801 before and after.
* The scoring formula in `docs/team/ideas.md` was three signals out of date
  (still showed span + bm25 + popularity over depth 200); now matches
  `RerankConfig`.

## 13. Reproduction

```bash
python3 -m unittest discover -s tests            # 57 tests
python3 -m evaluator.local_evaluator             # 0.9159, hit 1.0
python3 -m evaluator.local_evaluator \
    --dataset data/hard_set.jsonl                # 0.7944
python3 tools/sweep.py --split dev \
    --configs conflict00,conflict04,pair00,pair08   # the ablation rows
git checkout semantic-rerank                        # §9 lives on its own branch
pip install -r requirements.txt && python3 tools/fetch_model.py
python3 tools/sweep.py --split dev \
    --configs semantic_off,semantic_on              # §9, needs the model
python3 tools/sweep.py --split dev \
    --configs pop002,pop010,pop030,pop050           # §10 step-1 bracket
python3 tools/fit_weights.py                        # §10 step-2 fit (dev only)
python3 tools/observe.py --only public_0020      # the motivating session, now rank 1
python3 tools/observe.py --only public_0002,public_0198   # §5 and §11 diagnostics
```

### The split discipline these numbers were measured under

`tools/sweep.py:split_samples` partitions the **public set** into dev (120) and
holdout (80). "Public 200" therefore *contains* the holdout, so a variant chosen
by reading a public score has already spent the gate. §5's re-opening is the
cautionary example: on public 200 the no-span rescore reads +0.0021 and looks
shippable, and every point of that comes from the holdout half while dev does not
move at all. Select on dev; run holdout once, on the final candidate; report
public last. §11's near-miss anatomy was re-derived on dev alone for the same
reason — the first pass was computed over all 200 sessions, which would have made
the feature-selection step itself a read of the test set.

The hard set has now been read across changes 11 and 12 and §§5, 11, so it is
partially spent as an independent test. `tools/generate_adversarial_set.py` can
draw a fresh one; doing so before the next weight re-fit is worth the few
minutes.

§6-§8 were measured with a throwaway harness rather than `tools/sweep.py` rows,
because the variants change the *shape* of `_facet_agreement` /
`_facet_conflicts` rather than a weight, and the house rule is not to park dead
options in `RerankConfig`. The harness monkeypatches `src.rerank.rerank` and
`starter.agent.rerank` in-process, touches no repo file, and reproduced the
baseline exactly on all four splits (dev 0.9233, holdout 0.9048, public 0.9159,
hard 0.7944) before any variant ran — which is the check that makes the
before/after pairs trustworthy. To redo it, wrap `rerank()` with the one line
changed and evaluate against `split_samples()` from `tools/sweep.py`.

New unit tests (`tests/test_components.py`): conflict demotes a contradicting
candidate; a multi-value product containing the stated value is not punished;
silence about a facet is not a conflict; an override discards the stale value
for conflict scoring; pair spans keep key:value associations and strip leading
filler; an intact association outranks recombined fragments; span matching is
word-bounded; each new weight at 0.0 reproduces the previous ranking exactly.
