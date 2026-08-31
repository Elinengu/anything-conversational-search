# tools/fit_common.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, MAX_TURNS
from src.policy import FixedPolicy
from src.rerank import RerankConfig
from starter.agent import Agent, AgentConfig
from tools.sweep import split_samples

FITTED = ("popularity_weight", "retrieval_weight", "facet_weight", "category_weight",
          "tail_weight", "pair_weight", "facet_conflict_weight")

def make_config(weights: dict, variant: str) -> AgentConfig:
    rc = RerankConfig(**{k: float(v) for k, v in weights.items()})
    if variant == "plain":
        return AgentConfig(use_router=False, policy=FixedPolicy(), rerank=rc)
    if variant == "default":
        return AgentConfig(rerank=rc)
    raise ValueError(variant)

def load_all(catalog="data/catalog.jsonl") -> dict:
    ids, cats, prods = catalog_index(catalog)
    pub = load_jsonl("data/public_set.jsonl")
    return {"catalog": catalog, "ids": ids, "cats": cats, "prods": prods,
            "dev": split_samples(pub, "dev"), "holdout": split_samples(pub, "holdout"),
            "public": pub, "hard": load_jsonl("data/hard_set.jsonl")}

def scalar_from_sessions(sessions: list) -> dict:
    if not sessions:
        return {"score": 0.0, "hit": 0.0, "mrr": 0.0, "eff": 0.0, "mttc": 11.0, "n": 0}
    n = len(sessions)
    hit = sum(int(s["hit"]) for s in sessions) / n
    mrr = sum(s["reciprocal_rank"] for s in sessions) / n
    mttc = sum((s["first_hit_turn"] if s["first_hit_turn"] is not None else MAX_TURNS + 1)
               for s in sessions) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"score": 0.5 * hit + 0.3 * mrr + 0.2 * eff, "hit": hit, "mrr": mrr,
            "eff": eff, "mttc": mttc, "n": n}

class Scorer:
    def __init__(self, data: dict, variant: str, samples_key: str = "dev"):
        self.data, self.variant, self.samples = data, variant, data[samples_key]
        self.cache: dict = {}
        self.evals = 0
    def sessions(self, weights: dict) -> list:
        key = tuple(round(float(weights[k]), 6) for k in FITTED)
        if key not in self.cache:
            r = evaluate(Agent(self.data["catalog"], make_config(weights, self.variant)),
                         self.samples, self.data["ids"], self.data["cats"], self.data["prods"])
            self.cache[key] = r["sessions"]
            self.evals += 1
        return self.cache[key]
    def score(self, weights: dict) -> float:
        return scalar_from_sessions(self.sessions(weights))["score"]

def gate(weights: dict, data: dict, variant: str) -> dict:
    cfg = make_config(weights, variant)
    out = {}
    for split in ("dev", "holdout", "public", "hard"):
        r = evaluate(Agent(data["catalog"], cfg), data[split],
                     data["ids"], data["cats"], data["prods"])
        out[split] = {"score": r["recommended_technical_score"], "hit": r["hit_rate_at_10"],
                      "mrr": r["mrr"], "scenario": r["scenario_metrics"]}
    return out

ROUND_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
def snap(weights: dict) -> dict:
    return {k: min(ROUND_GRID, key=lambda g: abs(g - float(weights[k]))) for k in FITTED}

def plateau(weights: dict, scorer: Scorer) -> dict:
    grid = {"_base": scorer.score(weights)}
    for k in FITTED:
        grid[k] = {f: scorer.score({**weights, k: round(float(weights[k]) * f, 4)}) for f in (0.5, 1.5)}
    return grid

def stress(rows: list, catalog="data/catalog.jsonl") -> str:
    out = []
    for spec in ("official", "paraphrase:heavy+browse-gated"):
        cmd = [sys.executable, "tools/stress_harness.py", "--customer", spec,
               "--configs", ",".join(rows), "--catalog", catalog]
        out.append("$ " + " ".join(cmd))
        out.append(subprocess.run(cmd, capture_output=True, text=True).stdout)
    return "\n".join(out)
