"""Test package init.

The default test suite asserts the agent is fully deterministic and offline
(see IMPLEMENTATION.md: "unset or unreachable, the agent is bit-identical to
the offline path"). ``src/llm.py`` loads ``DEEPSEEK_API_KEY`` from a repo-root
``.env`` at import time, and ``src/phrasing.py``'s optional polish pass
(``_llm_polish``) fires whenever that key is present - independent of
``AgentConfig.llm.enabled`` - so a real key in ``.env`` (used for the live
DeepSeek demos/investigation runs) would otherwise make clarification wording
non-deterministic and break the offline-guarantee tests. Stripping it here,
before any test module imports ``src.llm``, keeps the default suite
deterministic regardless of what a developer's local ``.env`` contains.
Tests that want to exercise the real DeepSeek path do so explicitly (setting
the key back on a client instance), not via ambient environment state.
"""

import os

# Set (not pop): src/llm.py's dotenv loader runs at import time and, by default,
# only fills in keys *absent* from os.environ - a bare pop() here would just be
# re-populated from .env the moment src.llm (or anything importing it) loads.
# An empty string is present-but-falsy, so DeepSeekClient.is_configured stays
# False without dotenv silently overwriting it back to the real key.
os.environ["DEEPSEEK_API_KEY"] = ""
