"""Print how one OTR prompt looks after a model's chat template (tokenizer only, no GPU).

Use on the login node before running a new chat model, to check that the
generation prompt ends where the answer (a name) should start, e.g. that
--no-thinking really closes the reasoning block.

    python scripts/show_chat_prompt.py Qwen/Qwen3.8-27B --no-thinking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.instrumentation import chat


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--assistant-prefix", default="")
    args = p.parse_args(argv)
    chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                   assistant_prefix=args.assistant_prefix)

    from relational_lab.instrumentation.hf_scoring import check_names_for_patching, load_tokenizer
    from relational_lab.instrumentation.patching import chat_text

    tok = load_tokenizer(args.model)
    item = ors.build_item("Leon", "Mira", "praise", "praised", "object_cleft", "patient", "active")
    text, _ = chat_text(tok, item.prompt)
    print("=== templated prompt (repr of the last 300 characters) ===")
    print(repr(text[-300:]))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print("=== last 12 tokens ===")
    print([tok.decode([i]) for i in ids[-12:]])
    problems = check_names_for_patching(tok, ors.NAMES, chat_template=True)
    print("=== name list check (chat mode) ===")
    print("OK" if not problems else "; ".join(problems))


if __name__ == "__main__":
    main()
