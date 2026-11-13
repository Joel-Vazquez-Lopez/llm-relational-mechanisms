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


def encode_batch(tokenizer, prompts: Sequence[str], pad_id: int):
    """Right-padded batch plus per-row offset mappings (requires a fast tokenizer)."""
    import torch

    enc = [tokenizer(p, add_special_tokens=True, return_offsets_mapping=True) for p in prompts]
    width = max(len(e["input_ids"]) for e in enc)
    ids = torch.full((len(enc), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for i, e in enumerate(enc):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        mask[i, :n] = 1
    lengths = [len(e["input_ids"]) for e in enc]
    offsets = [list(e["offset_mapping"]) for e in enc]
    return ids, mask, lengths, offsets


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
