# Demo video — plan and script

For the Devpost "Demo Video" requirement (backend/NLP track: a walkthrough
showing API usage, inference examples, or result analysis is accepted — no
front-end needed). Target length **2:45–3:15**. Everything below uses commands
and sessions that already work on `state-encoder-eval-hdu1cw` — no new code,
no new measurements needed to record this.

## Before you record — a checklist

- [ ] `git checkout state-encoder-eval-hdu1cw && git pull` — confirm you're on
      the current HEAD so the numbers on screen match what's committed.
- [ ] `python3 -m unittest discover -s tests -t .` once, off-screen, so you
      know the suite is green before recording (mention the count on camera,
      don't run the full 30s suite live — dead air).
- [ ] Regenerate a fresh `viewer.html` right before recording so the terminal
      timestamps look current: `python3 tools/observe.py --tag demo`
- [ ] **Do not show `.env`, any API key, or your terminal's shell history if
      it contains one.** Close other tabs/panes that might have a key in
      scrollback.
- [ ] **No Amazon logos, no Amazon UI screenshots.** `DATA_ATTRIBUTION.md`:
      the dataset is text/structured metadata only (product titles, features,
      categories) — real product *titles* on screen are fine and expected,
      but don't overlay any Amazon branding, favicon, or trademarked UI chrome
      to imply this is an official Amazon surface.
- [ ] Pick one clean 1080p recording pass for the terminal (font size ≥ 16pt —
      assume judges watch on a phone) and record `viewer.html` separately in
      a browser at 100% zoom, then edit together. Don't try to do both live.
- [ ] Have `runs/demo-<timestamp>/viewer.html` open in a browser tab, already
      scrolled/filtered to session `public_0100` before you start recording
      that segment (see Segment 3) — dead time hunting for it on camera reads
      badly.

## The narrative arc

Five beats, in order of what actually earns judging weight (Technical
Execution 35% / Innovation 20% / Impact 20% / Feasibility 15% / Presentation
10% — see `docs/team/ideas_to_integrate_llm.md`'s framing):

1. **The problem is real** (hook) — one-shot search fails a multi-turn need.
2. **The core system, proven** — 0.1067 → 0.8592 with zero AI, zero network.
3. **Watch it actually think** — one real session, live, with the *why*.
4. **It's been stress-tested, not just scored** — paraphrase/vague customers.
5. **The innovation layer, honestly reported** — opt-in LLM, measured,
   including where it doesn't help. This is the "we don't oversell" beat —
   judges notice a team that reports a real trade-off instead of a clean win.

---

## Script

### Segment 1 — Hook (0:00–0:15)

**On screen:** title card or terminal, nothing running yet.

> "A shopper says 'I need a loafer.' A one-shot search engine has to guess
> everything else — color, material, size — from four words. Our agent
> instead does what a real salesperson does: it asks. This is a multi-turn
> conversational shopping agent that finds one hidden product out of 50,000,
> across up to ten turns of dialogue."

### Segment 2 — The core system, proven (0:15–0:50)

**On screen:** terminal, run:
```bash
python3 -m evaluator.local_evaluator
```
Let it run (~30s — cut the dead air in editing, keep the printed score on
screen for a beat).

> "The supplied baseline — plain keyword search with no memory — scores
> `0.1067`. Our pipeline scores [read the current number off the screen —
> `0.9235` as of this branch], using nothing but the Python standard library:
> no model API, no network call, no external dependency, for this whole core
> path. Every stage — retrieval, reranking, dialogue state, clarification
> policy — is deterministic and reproducible from this one command."

**Cut to:** `README.md`'s score table or `IMPLEMENTATION.md` §7 "Results" for
half a second, showing the dev/holdout split — signals "we didn't just tune
on the number that counts," without dwelling on it.

### Segment 3 — Watch it think (0:50–1:50) — the centerpiece

**On screen:** `runs/demo-<timestamp>/viewer.html`, session `public_0100`
already loaded (browsing scenario: Dockers Proposal loafer).

> "Here's one real session from that run. The customer opens vague: 'I'm
> looking for Shoes, Loafers & Slip-Ons, but I'm still exploring.' Watch what
> the agent does — it doesn't guess. It asks a targeted question grounded in
> what's actually in the candidate pool right now, not a generic 'tell me
> more.'"

**Scroll to turn 2 in the viewer:** show the customer's answer ("manmade
sole, platform about half an inch") landing, and the retrieval pool
narrowing — call out the pool size dropping.

> "That answer gets parsed into structured 'slots' — not just remembered as
> text, but tracked with what turn it came from and whether it's still
> active. If the customer later says 'actually, ignore that,' the old slot
> doesn't get erased — it's demoted to a superseded archive, so the system
> has an audit trail of *why* it changed its mind."

**Scroll to the hit turn:** show the target landing at position 1 (or
whatever rank in that run), and the "why" panel — matched spans, popularity
tie-break, category match.

> "By turn three, the target is recommended at rank one. The scoring only
> rewards a hit if the product actually appears in the shown list — and the
> session ends the instant it does, so *where* it ranks matters as much as
> *whether* it's found at all."

### Segment 4 — Stress-tested, not just scored (1:50–2:20)

**On screen:** terminal.
```bash
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs router_on
```

> "The official evaluator is cooperative — it always discloses exactly what's
> asked, in a fixed template. Real customers paraphrase, ramble, and go quiet.
> So we built a second, adversarial evaluator: a stress harness that
> paraphrases every constraint, substitutes synonyms, and — this is the hard
> one — only discloses information when asked a *pointed* question, never on
> a generic 'anything else?' On that harness the score drops to
> [read the number — `0.77065`], which tells us exactly where the system is
> weakest, instead of hiding behind one easy number."

### Segment 5 — The innovation layer, honestly reported (2:20–2:55)

**On screen:** terminal.
```bash
python3 tools/sweep.py --split holdout --configs router_on,llm_rerank_gated
```

> "On top of that deterministic core, we built one opt-in layer: a real LLM
> call — DeepSeek — that only fires when the candidate pool is genuinely
> ambiguous, measured live by a pool-confidence signal from the dialogue
> state. It's fused into the existing ranking, never replaces it, and it
> fails closed on any error — no key, no network, a timeout — so a
> network-restricted scoring run still gets the full deterministic score,
> unmodified."

**On screen (cut to the two numbers on the same terminal output):**

> "Gated on, it moves the holdout score from `0.9149` to `0.9218`, and the
> stress-harness score up by about half a point. It's a real, measured gain —
> and it's off by default, because the gain is smaller than we'd need to
> justify flipping a network dependency on for everyone. We measured it
> honestly instead of shipping the version that looks best in a demo."

### Segment 6 — Close (2:55–3:10)

**On screen:** terminal or title card, `README.md`'s architecture diagram if
time allows.

> "Deterministic core that beats the baseline eight-to-one on the standard
> library alone, an adversarial harness that finds its real weak points, and
> one measured, opt-in AI layer on top — reported with its actual trade-offs,
> not just its wins. Thanks for watching."

---

## Alternative / shorter cut (if the track's limit is under 2 minutes)

Drop Segment 4 entirely (stress harness) and compress Segment 5 to one
sentence + the two numbers on screen. Keep Segments 1, 2, 3 and 6 — the live
session walkthrough (Segment 3) is the one beat that should never be cut; it's
the only part that actually *shows* the system working end-to-end rather than
reporting a score.

## Recording notes

- **Narration:** record voiceover separately from screen capture and sync in
  editing — much easier to get a clean take of the words without also
  needing perfect terminal timing live.
- **Pacing:** the numbered segments above sum to ~3:10; trim pauses in
  editing rather than talking faster live.
- **Captions:** burn in the two or three numbers that matter (`0.1067 →
  0.9235`, `0.9149 → 0.9218`) as on-screen text when you say them — judges
  skim.
- **Upload:** YouTube, set to **Public** (per the submission requirement —
  Unlisted does not satisfy "public visibility"), title it something judges
  can find later, and paste the link into the Devpost description.
