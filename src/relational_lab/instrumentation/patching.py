"""Residual-stream activation patching for Hugging Face decoder-only models.

torch is imported lazily. The residual stream "after layer l" is the output of
decoder block l (hidden states entering block l + 1). Patching replaces that
vector at one token position per row with the vector a *source* prompt
produced at its aligned position, then lets the forward pass continue.
"""

from __future__ import annotations

from collections.abc import Sequence

_LAYER_PATHS = (
    "model.layers",  # Llama, Qwen, Mistral, Gemma
    "model.language_model.layers",  # Gemma 3/4 multimodal wrappers
    "model.model.layers",
    "transformer.h",  # GPT-2
    "gpt_neox.layers",  # Pythia
)


def decoder_layers(model):
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"cannot find decoder layers on {type(model).__name__}")


def token_index(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> int:
    """Index of the LAST token overlapping [char_start, char_end)."""
    hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < char_end and b > char_start]
    if not hits:
        raise ValueError(f"no token covers characters {char_start}-{char_end}")
    return hits[-1]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def chat_text(tokenizer, prompt: str) -> tuple[str, int]:
    """The prompt wrapped in the tokenizer's chat template as one user turn, with the
    generation prompt appended, plus the character offset of ``prompt`` inside it.
    The same text that ``HFScorer`` scores in chat-template mode."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    at = text.find(prompt)
    if at < 0 or text.find(prompt, at + 1) >= 0:
        raise ValueError("prompt must occur exactly once, verbatim, in the chat-templated text")
    return text, at


def _encode_one(tokenizer, prompt: str, chat_template: bool):
    """input_ids and offset mapping in PROMPT coordinates (template tokens get offsets
    outside [0, len(prompt)), so item spans can be looked up unchanged)."""
    if not chat_template:
        e = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        return list(e["input_ids"]), [tuple(o) for o in e["offset_mapping"]]
    text, at = chat_text(tokenizer, prompt)
    # The template already contains BOS and turn markers: add no special tokens.
    e = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(e["input_ids"])
    ref = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    if not isinstance(ref, list):  # transformers >= 5: BatchEncoding
        ref = ref["input_ids"]
    if ref and isinstance(ref[0], list):
        ref = ref[0]
    if ids != list(ref):
        raise ValueError("re-tokenizing the templated text does not reproduce apply_chat_template's ids")
    offsets = [(a - at, b - at) if b > a else (a, b) for a, b in e["offset_mapping"]]
    return ids, offsets


def encode_batch(tokenizer, prompts: Sequence[str], pad_id: int, chat_template: bool = False):
    """Right-padded batch plus per-row offset mappings (requires a fast tokenizer).
    Offsets are always relative to the raw prompt string, also in chat-template mode."""
    import torch

    enc = [_encode_one(tokenizer, p, chat_template) for p in prompts]
    width = max(len(ids) for ids, _ in enc)
    ids_t = torch.full((len(enc), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for i, (ids, _) in enumerate(enc):
        ids_t[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    lengths = [len(ids) for ids, _ in enc]
    offsets = [off for _, off in enc]
    return ids_t, mask, lengths, offsets


def answer_token(tokenizer, name: str, leading_space: bool = True) -> int:
    """First token of the answer continuation: " Name" after "Answer:", "Name" at the
    start of a chat assistant turn. Logit differences use this first token only."""
    ids = tokenizer.encode((" " if leading_space else "") + name, add_special_tokens=False)
    if not ids:
        raise ValueError(f"name {name!r} encodes to nothing")
    return ids[0]


class ResidualPatcher:
    def __init__(self, model):
        self.model = model
        self.layers = decoder_layers(model)
        self.n_layers = len(self.layers)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def capture(self, ids, mask) -> list:
        """Run once; return [n_layers] tensors of shape [B, T, D] (on CPU) plus logits."""
        import torch

        store: list = [None] * self.n_layers
        handles = []
        for i, layer in enumerate(self.layers):
            def hook(_m, _inp, out, i=i):
                store[i] = _hidden(out).detach()
            handles.append(layer.register_forward_hook(hook))
        try:
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
        finally:
            for h in handles:
                h.remove()
        return store, logits

    def run_patched(self, ids, mask, layer: int, positions: Sequence[int], vectors):
        """Forward pass with block ``layer``'s output replaced at ``positions[b]`` by ``vectors[b]``."""
        import torch

        rows = torch.arange(len(positions), device=self.device)
        pos = torch.tensor(list(positions), device=self.device)

        def hook(_m, _inp, out):
            h = _hidden(out).clone()
            h[rows, pos] = vectors.to(h.dtype).to(h.device)
            return _replace_hidden(out, h)

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
        finally:
            handle.remove()
