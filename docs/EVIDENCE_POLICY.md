# Evidence and reproducibility policy

- Freeze exact model/checkpoint/tokenizer IDs and revisions, quantization, prompts, software/hardware, seeds and decoding.
- Verify source file versions and SHA-256 digests; retain immutable raw results privately and record hashes.
- Label each claim: illustration, exploratory, controlled development, confirmatory or independently replicated.
- Unit of analysis must be independent source/document/event family where appropriate. Correlated prompts and counterfactual rotations are not independent replications.
- Separate generated answers, constrained candidate scores, first-token correctness, full-string validity, task gold and causal effects.
- Prespecify strongest plausible competitors: syntax-aware copying, position, lexical/name priors, query format, output competition and generic depth. Use competent controls rather than weak straw men.
- Historical stopped competence/format/identifiability gates remain failed; changing criteria requires a new prospective protocol.
- Preserve all negative outcomes, excluded cases and deviations.
- Never commit secrets, third-party copyrighted text without rights, model weights, protected evaluation items or user-private information.
