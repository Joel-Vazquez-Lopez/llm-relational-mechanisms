# POSITION-CB core-transfer v0 — pre-run protocol

**Status:** prepared, not yet scientifically executed. Exploratory read-only audit motivated by the existing train-pool-to-development gap; **not a confirmatory prospective neural experiment**.

## Question

The previous study was trained on one event core per each of 300 TRAIN name quartets and nearly exhausted 13,500 surface-variant indices. Its unused variant control still reused trained cores. Does the existing model fail on unfamiliar **role assignments, event orders, or participant pairings** when the original four-name set is held fixed?

## Fixed data and controls

Select 144 distinct four-name sets from the **TRAIN partition**, selection seed 1282026, 16 each for nine modifier pairs. For each set, reconstruct the original j=2 training event core, and render twelve controlled assignments:

- Original unchanged agent/patient/event configuration (1).
- Flip agent/patient identities in the first event, second event, or both (3).
- Swap first/second event membership, with four within-event role orientations (4).
- Repartition the quartet into new event pairs; choose one orientation from each of the four possible first-event cross-pair groups, cycling all four orientations across sampled quartets (4).

Render in all four voices AA/PP/AP/PA, using the same two verbs, modifier pair and event-layout flags. Adjust near/gate adjunct references to the other event. **Source lengths at fixed voice/modifier are equal** across all twelve variants. Modifiers can change lexical name counts; the bare/bare subset (16 quartets) must match *exact full-token multisets* for every variant at fixed voice.

Preflight constructs exactly 6,912 surface presentations: 576 unchanged original train surfaces and 6,336 novel surfaces. Novel surfaces must be disjoint from all original 54,000 TRAIN pool voice/variant surfaces. Independently parse every new sentence; enforce exactly four unique old names, matching gold labels, source length cap, and no within-event LI or reserved/protected four-name sets. This does **not** add an independent grammar, new names, or external data.

## Frozen checkpoint inference

Evaluate only the **familiar voice combinations** for each saved model: same_voice uses AA+PP; mixed_voice uses AP+PA. Use all 8 original seeds, 2 MPS numerical repetitions, 2 arms, 32 frozen final checkpoints. There is no retraining, weight update, model selection or re-scoring of existing development. The familiar development comparison is copied from original archived result JSON.

Report all variant-level strict joint four-question accuracy (C_query), four-name question accuracy, full-vocabulary accuracy, and mutually exclusive pair error categories for each of two events. Average variants within the named tier, repeats within seed, then eight independent seeds. Also report the bare/bare lexical-bag control. Source texts and item-level predictions are not stored.

Summarize **three fixed exploratory contrasts**: original vs role_flip, original vs event_swap, original vs repartition. Show descriptive two-sided seed-level Student-t confidence intervals; no p-value hierarchy or claims of preregistered confirmation. The original nominal primary/secondary studies remain unchanged.

## Interpretation and stop

- Role-flip drop: inability to reassign agent/patient within known event groups.
- Event-swap drop: reliance on event-position / sequence order.
- Repartition drop: reliance on learned participant-pair membership.
- Strong performance across all three but weak archived development: stronger evidence that change in four-name combinations or dev-specific source details matters.

Every one of these interpretations remains conditional because reassigning adjunct names can alter local lexical evidence. The bare/bare control removes that particular confound with smaller sample size.

**Before any pretrained checkpoint is scored**, run prepare and random-weight smoke, then commit the exact runner, launcher, protocol and generated source manifest. The runner verifies committed bytes of these four files. No protected/reserved/L+I splits are opened. No seed/checkpoint filtering is permitted.

The output should determine **one new frozen training-diversity intervention**, not trigger additional post-hoc parameter/head searches.
