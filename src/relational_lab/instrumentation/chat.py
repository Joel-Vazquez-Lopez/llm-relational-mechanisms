"""Shared chat-template settings for all OTR runners.

``CHAT_TEMPLATE_KWARGS`` is passed to every ``tokenizer.apply_chat_template``
call made by HFScorer (OTR-01) and by the patching code (OTR-02/03/04). It is
empty by default, so earlier runs are reproduced exactly. Runners set
``{"enable_thinking": False}`` with ``--no-thinking`` for models whose template
otherwise opens a reasoning block before the answer (Qwen3-type, OLMo-3 Think);
templates that do not use the variable ignore it.
"""

from __future__ import annotations

CHAT_TEMPLATE_KWARGS: dict = {}


def configure(no_thinking: bool = False) -> dict:
    """Set the shared kwargs from runner flags; returns them for the result JSON."""
    CHAT_TEMPLATE_KWARGS.clear()
    if no_thinking:
        CHAT_TEMPLATE_KWARGS["enable_thinking"] = False
    return dict(CHAT_TEMPLATE_KWARGS)
