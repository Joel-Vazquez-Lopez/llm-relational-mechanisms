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
# Text appended after the template's generation prompt, e.g. " to=user<|message|>"
# so that a model with a reasoning channel (Muse Glimmer) starts its final answer
# directly. Empty by default.
ASSISTANT_PREFIX: list[str] = [""]


def assistant_prefix() -> str:
    return ASSISTANT_PREFIX[0]


def configure(no_thinking: bool = False, template_kwargs=(), assistant_prefix: str = "") -> dict:
    """Set the shared settings from runner flags; returns them for the result JSON.
    ``template_kwargs``: strings "key=value" passed to apply_chat_template."""
    CHAT_TEMPLATE_KWARGS.clear()
    if no_thinking:
        CHAT_TEMPLATE_KWARGS["enable_thinking"] = False
    for kv in template_kwargs or ():
        if "=" not in kv:
            raise ValueError(f"--template-kwarg expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        CHAT_TEMPLATE_KWARGS[k] = {"true": True, "false": False}.get(v.lower(), v)
    ASSISTANT_PREFIX[0] = assistant_prefix or ""
    out = dict(CHAT_TEMPLATE_KWARGS)
    if ASSISTANT_PREFIX[0]:
        out["_assistant_prefix"] = ASSISTANT_PREFIX[0]
    return out
