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
# Optional system message for every chat-template prompt (prompt-robustness runs). Empty = none,
# which reproduces all earlier runs.
SYSTEM_PROMPT: list[str] = [""]


def assistant_prefix() -> str:
    return ASSISTANT_PREFIX[0]


def messages(prompt: str) -> list[dict]:
    """The chat turns for one prompt: an optional system turn, then the prompt as one user turn."""
    turns = [{"role": "user", "content": prompt}]
    if SYSTEM_PROMPT[0]:
        turns = [{"role": "system", "content": SYSTEM_PROMPT[0]}] + turns
    return turns


def as_id_list(out) -> list[int]:
    """apply_chat_template(tokenize=True) returns a list in older transformers and a
    BatchEncoding (dict-like) in transformers >= 5, sometimes in batched form [[...]]."""
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def apply_template(tokenizer, prompt: str, tokenize: bool):
    """The one place where OTR prompts are wrapped in a chat template, used by both
    scoring (HFScorer) and patching, so the two always see the same sequence.
    Adds the generation prompt and ``CHAT_TEMPLATE_KWARGS``; the assistant prefix is
    appended by the callers (it is text after the template). If the template rejects a
    system turn, the system text goes in front of the user turn instead. With no system
    prompt this is exactly the single-user-turn call that all earlier runs used."""
    turns = messages(prompt)
    try:
        return tokenizer.apply_chat_template(turns, add_generation_prompt=True, tokenize=tokenize,
                                             **CHAT_TEMPLATE_KWARGS)
    except Exception:
        if len(turns) == 1:
            raise
        merged = [{"role": "user", "content": SYSTEM_PROMPT[0] + "\n\n" + prompt}]
        return tokenizer.apply_chat_template(merged, add_generation_prompt=True, tokenize=tokenize,
                                             **CHAT_TEMPLATE_KWARGS)


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
