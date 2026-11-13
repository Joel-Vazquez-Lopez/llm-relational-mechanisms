"""Candidate scoring with Hugging Face causal language models.

torch and transformers are imported lazily, so importing this module (and the
rest of ``relational_lab``) stays dependency-free.

For each item the continuation scored is " " + name (no space after a chat
template's assistant tag), encoded without special
tokens and appended to the encoded prompt, so the boundary between prompt and
answer never merges into a single token. Multi-token names are summed over
their tokens; ``answer_is_top1`` checks only the first token.
"""

from __future__ import annotations

from collections.abc import Sequence

from relational_lab.experiments.order_role_behavior import Score
from relational_lab.experiments.order_role_stimuli import Item


def name_token_ids(tokenizer, name: str, leading_space: bool = True) -> list[int]:
    """Raw prompts end in "Answer:", so the answer follows a space. After a chat
    template's assistant tag the answer starts a fresh line, with no space."""
    return tokenizer.encode((" " if leading_space else "") + name, add_special_tokens=False)


def single_token_checker(tokenizer, leading_space: bool = True):
    """For ``order_role_stimuli.filter_names``."""
    return lambda name: len(name_token_ids(tokenizer, name, leading_space)) == 1


def _prompt_ids(tokenizer, prompt: str, use_chat_template: bool) -> list[int]:
    if use_chat_template:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        # transformers >= 5 returns a BatchEncoding (dict-like); older versions a list.
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):  # batched form [[...]]
            out = out[0]
        return list(out)
    return tokenizer.encode(prompt, add_special_tokens=True)


class HFScorer:
    """Callable scorer: ``scorer(items) -> list[Score]``."""

    def __init__(self, model, tokenizer, batch_size: int = 16, use_chat_template: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.pad_id = pad

    def _continuation_logprobs(self, sequences: list[tuple[list[int], list[int]]]):
        """Return, per (prompt_ids, cont_ids): (sum logprob of cont, top1 id at first cont position)."""
        import torch

        device = next(self.model.parameters()).device
        full = [p + c for p, c in sequences]
        width = max(len(x) for x in full)
        input_ids = torch.full((len(full), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(full), width), dtype=torch.long)
        for i, seq in enumerate(full):  # right padding: positions stay aligned with the unpadded sequence
            input_ids[i, : len(seq)] = torch.tensor(seq)
            attention[i, : len(seq)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()
        out = []
        for i, (p, c) in enumerate(sequences):
            total = 0.0
            for k, tok in enumerate(c):
                total += float(logprobs[i, len(p) - 1 + k, tok])
            top1 = int(logprobs[i, len(p) - 1].argmax())
            out.append((total, top1))
        return out

    def __call__(self, items: Sequence[Item]) -> list[Score]:
        scores: list[Score] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            seqs = []
            for it in batch:
                p = _prompt_ids(self.tokenizer, it.prompt, self.use_chat_template)
                space = not self.use_chat_template
                seqs.append((p, name_token_ids(self.tokenizer, it.answer, space)))
                seqs.append((p, name_token_ids(self.tokenizer, it.distractor, space)))
            results = self._continuation_logprobs(seqs)
            for j, it in enumerate(batch):
                (lp_a, top1), (lp_d, _) = results[2 * j], results[2 * j + 1]
                first_answer_tok = name_token_ids(self.tokenizer, it.answer, not self.use_chat_template)[0]
                scores.append(
                    Score(
                        item_id=it.item_id,
                        logprob_answer=lp_a,
                        logprob_distractor=lp_d,
                        answer_is_top1=(top1 == first_answer_tok),
                    )
                )
        return scores
