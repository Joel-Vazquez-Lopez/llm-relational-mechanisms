# EXP-101 RC2 — fixed position-index coverage amendment

**Decision:** deterministic nine-step training-only absolute-position ID cycle; one remedy, selected pretraining and without viewing model outcomes.

R3 uses learned absolute positions and training sequences of at most 34 tokens; the withheld L+I composition has 37–39 tokens. Increasing table capacity to 64 gives vectors at positions 34–38 but no gradient updates. The fixed remedy is an offset `offset(t)=t mod 9` for optimizer update `t` (zero-based). The original token sequence, token order, causal mask, readout index, label, and text are unchanged. Only the learned embedding ID associated with each *existing* token is shifted. Both paired arms use the same offset at each update and retain paired initialization and identical shared batch RNG.

All test/development inference uses `position_offset=0`. A checkpoint stores the schedule identifier and update step; the next offset after resume is determined entirely by that step. No extra trainable parameters, steps, training examples or study seeds are introduced. **This changes the training regime from ML-080 and prior EXP-101 candidates**, hence effects cannot be attributed solely to exposure under the former unaugmented training protocol. It does not train the L+I surface composition.

Static validation must compute nonpadding *question-region* training-token opportunities at every L+I evaluation index (34–38), not merely verify table size. The engineering fixture must backpropagate from a familiar development-only 34-token prompt shifted by eight and record strictly nonzero position-embedding gradients for indexes 34–38. This is a coverage/engineering criterion, not evidence of successful relational learning or sufficient gradient mass during the eventual study.

**No remedy comparison or study training was performed.** The release remains unapproved until external grammar sign-off and on-device Mac MPS engineering verification.
