# EXP-101 RC2 — grammatical template review sheet (external sign-off pending)

**Scope:** every condition × voice × gate-layout (4 × 2 × 2 = 16 template configurations). The generator/parser checks semantic roles, named locative slots, and ordinal event-query identity automatically. It cannot provide independent human judgments of grammaticality, pragmatics, coreference or scope. All human acceptance fields are **PENDING**, not silently approved.

The only surface correction in RC2 is postposed `by X's gate` → `beside X's gate`. Thus the original awkward `Bero was helped near Dovan by Aven by Cira's gate` becomes `Bero was helped near Dovan by Aven beside Cira's gate`. The fronted `By Cira's gate, ...` is retained, and both clauses remain balanced in adjunct status. No measurement or training outcome motivated this amendment.

**Review rubric:** (1) grammaticality and naturalness; (2) can four distinct event participants be read unambiguously? (3) does `by Aven` unequivocally denote passive agent, and the gate/near names only nonarguments? (4) is the *agent/patient role relation* invariant under active↔passive, without claiming exact equivalence of discourse focus, information structure or pragmatic implications? (5) is `near Dovan` possibly construed as patient/event location rather than an extra participant? (6) do the two event predicates remain distinguishable under order reversal and same-sentence query change?

| Condition | Voice | Gate layout | Example (single clause) | Machine role/slot check | Independent grammar decision |
|---|---|---|---|---|---|
| bare | active | 0 | Aven helped Bero | PASS: structural labels | **PENDING** |
| bare | active | 1 | Aven helped Bero | PASS: structural labels | **PENDING** |
| bare | passive | 0 | Bero was helped by Aven | PASS: structural labels | **PENDING** |
| bare | passive | 1 | Bero was helped by Aven | PASS: structural labels | **PENDING** |
| L | active | 0 | By Cira's gate, Aven helped Bero | PASS: structural labels | **PENDING** |
| L | active | 1 | Aven helped Bero beside Cira's gate | PASS: structural labels | **PENDING** |
| L | passive | 0 | By Cira's gate, Bero was helped by Aven | PASS: structural labels | **PENDING** |
| L | passive | 1 | Bero was helped by Aven beside Cira's gate | PASS: structural labels | **PENDING** |
| I | active | 0 | Aven, near Dovan, helped Bero | PASS: structural labels | **PENDING** |
| I | active | 1 | Aven, near Dovan, helped Bero | PASS: structural labels | **PENDING** |
| I | passive | 0 | Bero was helped near Dovan by Aven | PASS: structural labels | **PENDING** |
| I | passive | 1 | Bero was helped near Dovan by Aven | PASS: structural labels | **PENDING** |
| LI | active | 0 | By Cira's gate, Aven, near Dovan, helped Bero | PASS: structural labels | **PENDING** |
| LI | active | 1 | Aven, near Dovan, helped Bero beside Cira's gate | PASS: structural labels | **PENDING** |
| LI | passive | 0 | By Cira's gate, Bero was helped near Dovan by Aven | PASS: structural labels | **PENDING** |
| LI | passive | 1 | Bero was helped near Dovan by Aven beside Cira's gate | PASS: structural labels | **PENDING** |

**Interpretation restriction:** role-preservation is the modeled semantic invariant. Voice and event-order changes can change focus, information structure, discourse chronology and attachment preferences; therefore do not claim strict entire-sentence equivalence.

**Human reviewer requirements before release:** record reviewer identity/date, ACCEPT/REVISE on each of the 16 entries, optional example corrections and whether the semantic-task interpretation remains stable. A machine-generated assessment or a language model cannot be counted as independent human approval. The current release is locked until this record is completed.
