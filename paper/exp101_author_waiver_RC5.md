# EXP-101 RC5 — Author grammar acceptance and independent-review waiver

**Decision recorded before EXP-101 study training or protected evaluation, 13 November 2026.**

The study author reported that the rendered sentences had been personally checked and that the examples passed their grammatical/semantic review. In a subsequent request, the author explicitly elected to **waive the independent human reviewer requirement** and continue to release preparation. This is a **protocol deviation** from RC2–RC4, which had prespecified an independent 16-template accept/revise review. The author did not provide a separately dated, item-by-item signed external review sheet. This record must not be presented as independent linguistic validation or as 16 independently confirmed acceptances.

**Review disposition:** `AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED`.

**Scope and limitations:** No additional modifications to the syntactic templates are being made here. The template renderer and independent software oracle still enforce participant assignments, explicit gate owner and near person, and question-to-answer consistency, but those checks do not demonstrate native-speaker naturalness or unambiguous discourse pragmatics. Preservation of **agent/patient role assignments** under active/passive transformation is the intended semantic invariant, **not** complete preservation of sentence-level information structure, discourse ordering or focus. Any later linguistic issues must be reported as limitations, not used for post-hoc training or protected-set tuning.

**Scientific safeguards unchanged:** two arms, eight paired seeds, exactly 12,800 updates per model, the offset schedule, predefined development competence gate, same-cohort bare/L/I references, one L+I primary, cross-event correctness check, fixed analysis and separate later authorization for protected evaluation.

**Hardware evidence:** two fresh-process MPS fixture outputs were shared in the conversation, each stating `ENGINEERING_FIXTURE_PASS_NOT_APPROVAL`, a same-origin checkpoint roundtrip difference of zero, optimizer resume within tolerance and positive gradients at indices 34–38. The complete raw files/logs are still on the author's Mac. The local release-sealing script requires their actual bytes and stores their SHA-256 hashes before permitting any release approval. We must not claim the raw reports have been uploaded or independently inspected here.

**Release policy:** the source-bound candidate remains `training_authorized=false`. After the raw Mac fixture reports and logs pass local validation, the author may explicitly seal a training-only frozen release. A separate one-time protected unlock remains mandatory after the prespecified competence gate. No study training was conducted while recording this waiver.
