#!/usr/bin/env python3
"""Interactive Demonstration Tool for the Hybrid LLM Layer.

Demonstrates:
1. Deterministic Offline Mode (0 tokens, instant fallback, score-optimal).
2. Live/Mock LLM Mode (Natural recommendation justifications and pool clarifications).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter.agent import Agent, AgentConfig
from src.llm import LLMConfig, LLMClient


def demo_offline_mode():
    print("=" * 70)
    print("1. DETERMINISTIC OFFLINE MODE (0 Tokens, Benchmark Floor)")
    print("=" * 70)

    config = AgentConfig(
        llm=LLMConfig(enabled=False),
        first_recommend_turn=1,
        confidence_margin=0.0,
    )
    agent = Agent("data/catalog.jsonl", config=config)
    agent.reset("offline_demo", {})

    response = agent.respond("offline_demo", "I want a waterproof black leather jacket.", turn=1, top_k=3)
    print(f"Message (Offline):      {response['message']}")
    print(f"Ask Attribute:          {response['ask_attribute']}")
    print(f"Top 3 Recommendations:  {[r['parent_asin'] for r in response['recommendations']]}")
    print(f"Token Usage Reported:   {response['usage']}")


def demo_llm_mode():
    print("\n" + "=" * 70)
    from src.llm import _load_dotenv_if_present
    _load_dotenv_if_present()
    provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "mock"
    print(f"2. HYBRID LLM MODE (Provider: {provider})")
    print("=" * 70)

    config = AgentConfig(
        llm=LLMConfig(enabled=True, provider=provider),
        first_recommend_turn=1,
        confidence_margin=0.0,
    )
    agent = Agent("data/catalog.jsonl", config=config)
    agent.reset("llm_demo", {})

    response = agent.respond("llm_demo", "I want a waterproof black leather jacket.", turn=1, top_k=3)
    print(f"Message (LLM Reasoned): {response['message']}")
    print(f"Ask Attribute:          {response['ask_attribute']}")
    print(f"Top 3 Recommendations:  {[r['parent_asin'] for r in response['recommendations']]}")
    print(f"Token Usage Reported:   {response['usage']}")


def main():
    print("======================================================================")
    print("HYBRID LLM LAYER DEMONSTRATION & VERIFICATION")
    print("======================================================================\n")

    demo_offline_mode()
    demo_llm_mode()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
