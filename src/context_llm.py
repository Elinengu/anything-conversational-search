"""Gemini-backed state rewriting for context distillation.

When an intent override is detected, this module uses an LLM to parse the
conversation and determine which constraints to erase, keep, or add.

The distiller falls back to regex-based extraction when Gemini is unavailable
or rate-limited, preserving deterministic behavior under quota constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm import get_llm_client


@dataclass
class StateRewriteInstruction:
    """Parsed state rewrite directive from Gemini."""
    erase: list[str]
    keep: list[str]
    add: list[str]

    @classmethod
    def from_llm(cls, payload: dict | None) -> StateRewriteInstruction | None:
        """Validate and construct from LLM response payload."""
        if not isinstance(payload, dict):
            return None
        # All three keys must be present
        if not all(k in payload for k in ("erase", "keep", "add")):
            return None
        erase = payload.get("erase", [])
        keep = payload.get("keep", [])
        add = payload.get("add", [])
        if not isinstance(erase, list) or not isinstance(keep, list) or not isinstance(add, list):
            return None
        return cls(
            erase=[str(x).strip() for x in erase if x],
            keep=[str(x).strip() for x in keep if x],
            add=[str(x).strip() for x in add if x],
        )


def rewrite_state_for_override(
    conversation: list[str],
    override_turn: int,
) -> StateRewriteInstruction | None:
    """Use Gemini to determine what constraints to erase/keep/add after an override.

    Returns None if the LLM is unavailable or rate-limited; the caller should fall
    back to deterministic state handling in that case.
    """
    client = get_llm_client()
    if not client.is_configured:
        return None

    # Only call Gemini if we have enough conversation to reason about
    if not conversation or len(conversation) < override_turn:
        return None

    # Build a concise conversation narrative for the LLM
    conversation_text = "\n".join(
        [f"{i + 1}. {msg}" for i, msg in enumerate(conversation[:override_turn])]
    )

    prompt = (
        "You are analyzing a shopping conversation where the customer has just reversed "
        "or revised a prior preference. Based on the conversation history, return a JSON "
        "object with three keys:\n"
        '- "erase": list of constraint keywords to discard from before the override\n'
        '- "keep": list of constraint keywords that remain valid and should persist\n'
        '- "add": list of new constraint keywords the customer just introduced\n'
        "Return ONLY valid JSON with these keys. Do not include explanation.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Return:"
    )

    payload = client.generate_json(prompt)
    return StateRewriteInstruction.from_llm(payload)
