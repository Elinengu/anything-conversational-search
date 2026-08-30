# Stress conversation observer

This directory visualizes conversations produced by `tools/stress_harness.py`
without modifying either `tools/observe.py` or `tools/stress_harness.py`.

```bash
# Ten browsing sessions where the customer paraphrases and needs pointed questions
python3 -m tools.stress_observe \
  --customer paraphrase:heavy+browse-gated \
  --scenario browsing \
  --limit 10

# Inspect one session
python3 -m tools.stress_observe \
  --customer browse-gated \
  --only public_0008

# Compare using a named sweep configuration
python3 -m tools.stress_observe \
  --customer browse-gated \
  --config router_off \
  --tag gated-router-off
```

Each run writes beneath `runs/stress-observe/`:

- `viewer.html` — self-contained interactive transcript viewer
- `sessions/<sample_id>.md` — one annotated transcript per session
- `trace.jsonl` — machine-readable turn records
- `summary.json` — aggregate scores and failure diagnoses
- `index.md` — worst sessions first

The runner records the actual stress-customer disclosure before paraphrasing, so
the viewer can label a paraphrased sentence with the constraint it represents.
It also traces the per-turn buying/browsing route and forwards the branch's
`track=` reranking argument correctly.
