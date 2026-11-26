# FREEZE addendum — Spanish and Galician on test set 0 (26 Nov 2026)

Status: FROZEN by the commit that adds this file, together with scorer v1.1.0
(`scripts/otr_freeze_score.py`) and `scripts/slurm/otr_xl_test0_submit.sh`. Written after the English test0
results (FREEZE.md) were opened and before any Spanish or Galician command ran on a test split. Everything
here was chosen from the dev es/gl results of 17–21 Nov. After the commit, sections 2–5 do not change;
anything else is labelled exploratory. FREEZE.md stays as it was; its section 7 ("XL is not part of test0")
described the main freeze and is superseded only by this addendum.

## 0. Prior access to held-out data
- No result file with an es/gl test split exists (checked 26 Nov: 0 files matching `test[0-3]_es_` or
  `test[0-3]_gl_`, 0 es/gl files with status `TEST_SET_OPENED`).
- Disclosure: the English test0 results are known. The addendum's predictions come from the dev es/gl runs
  only; where they differ from English (handover layer, role − function), the dev es/gl values decide.
- Disclosure: dev OTR-03 runs for es/gl drew donor pairs from all name pairs (as for English, FREEZE.md §0).
  Test runs draw donors from test0 only and abort on any leak.

## 1. Scope
- Same five models, revisions, runtime and family partition as FREEZE.md §1 (es/gl share the core pair
  assignment: test0 = 59 families, 236 events per construction).
- Stimuli: XL generator v1.0.0, the same templates as dev. Instructions: es "Responde con un solo nombre.",
  gl "Responde cun só nome."; chat template; Qwen3.8 and Granite with thinking off.
- Failing question: the passive agent question ("¿Por quién fue V-ado alguien?" / "Por quen foi V-ado
  alguén?"). Working question: the active agent question ("¿Quién V a alguien?" / "Quen V a alguén?").
- Constructions: C1 and C4/C5 use the constructions where the working question is reliable on dev
  (es: active, passive, object cleft; gl: active, passive), as in the dev runs. C2/C3 use the four
  transplant constructions, as in the dev runs.
- Spanish is the primary test, Galician the secondary one. Both are scored and reported in full.
- Native-speaker judgements of the stimuli were not settled for the main freeze. Before submitting, Joel
  (native speaker of both) reads a random sample of 24 dev prompts per language and records the outcome in
  section 6. A template problem found there stops the submission; it is fixed on dev and the addendum is
  re-frozen with a new commit.

## 2. Definitions
As FREEZE.md §2, with these differences:
- C1 measure: proportion of agent-question rows the extractor credits (`correct` in the generation output),
  working minus failing, on the same run (`--n 1640`, balanced over all cells); family bootstrap on the
  per-family means.
- Handover layer (C4): the first even layer with residual repair ≥ .5 in the dev es/gl OTR-09 runs (files
  in section 7). This is not the English layer: on dev, Gemma (46 vs 40) and Qwen3.8 (50 vs 54) shift in
  es/gl. The English layer is reported next to each result, descriptively.
- C5b control: the frozen heads patched with the head outputs of another item's working run (different
  family). This replaces `other_failing <= .10` as the content control: English test0 showed in phi-4 that
  replacing the heads' output with any other item's output repairs about .27 on its own (removing its own
  wrong-way push), so `other_failing` does not test item-specific content (FREEZE.md stays as it was; see
  the results log, 26 Nov). `other_failing`, `random_nudge` and `mean_failing` are reported, not scored.

## 3. Claims and pass rules (per model and language, test0)
| ID | Claim | Measure | Passes if |
|---|---|---|---|
| XL-C1 | The failing question fails selectively | agent questions: working − failing accuracy | CI lower bound > 0 |
| XL-C2 | Roles live on the names | OTR-03 ordinary questions, correct bases, role coefficient at the frozen English role layer (order and function in the same fit) | role CI lower bound > 0 (role − function reported, not scored: on dev, function also carries signal in es/gl) |
| XL-C3 | Wrong answers read the code weaker or reversed | OTR-03 failing question (passive agent question), wrong agent bases, role coefficient at the same layer; C2 − C3 on the same resampled families | CI lower bound > 0; at least 30 wrong bases, else not evaluable |
| XL-C4 | The handover happens at the frozen es/gl layer | OTR-09 residual repair, even layers up to handover + 2 | first layer with repair ≥ .5 within ±2 of the frozen layer, and all layers ≤ frozen − 4 below .5 |
| XL-C5 | The English heads repair es/gl | frozen English top-20 heads vs 20 random heads from the same layers (5 draws), paired by item | top − random CI lower bound > 0, and top ≥ .25 |
| XL-C5b | The repair carries item-specific content | the same heads with the item's own working-run output vs another item's working-run output, paired by item | CI lower bound > 0 |

Frozen values:

| Model | Role layer (English) | es handover | gl handover | English handover (descriptive) |
|---|---|---|---|---|
| Gemma 4 31B-it | 28 | 46 | 46 | 40 |
| Qwen3.8-27B | 29 | 50 | 50 | 54 |
| phi-4 | 15 | 22 | 22 | 22 |
| OLMo-3.1-32B-Instruct | 18 | 24 | 26 | 26 |
| Granite-4.2-30b | 37 | 52 | 52 | 52 |

Heads: the English dev top-20 per model (the same files as FREEZE.md §4).

## 4. Commands
`bash scripts/slurm/otr_xl_test0_submit.sh` prints them; `--submit` runs them (40 jobs, one H100 each) and
logs every job ID in `docs/order_to_role/test0_attempts.md`. Per model and language:
- XL-C1: `otrgen_pelle.sbatch <model> --chat-template --split test0 --stimuli <lang> --n 1640 --instruction <instr>`
- XL-C2: `otr03_pelle.sbatch <model> --chat-template --split test0 --stimuli <lang> --n-events 64 --layers <role>`
- XL-C3: the same plus `--question-voices passive --roles agent --include-incorrect`
- XL-C4/C5/C5b: `otr09_pelle.sbatch <model> --chat-template --split test0 --stimuli <lang> --instruction <instr>
  --rank-from <English dev file> --skip-scan --n-random 5 --controls --controls-extra --n-per 64 --top-k 20
  --layers <even layers up to handover + 2>`

Scoring: `python scripts/otr_freeze_score.py --results results --split test0 --stimuli es` (then `gl`).

## 5. What counts as what
- Every claim is reported for every model and language, pass or fail, with its CI.
- Technical failures: as FREEZE.md §6 (identical rerun, at most twice more, else not evaluable).
- Expectations from dev, written down so a miss is visible (not pass rules): XL-C1 passes in all ten
  model–language cells on dev (es .29–.76, gl .38–.51). XL-C3 failed on dev for Gemma in Spanish (C2 − C3
  = .08, CI includes 0) and passed wherever evaluable elsewhere. XL-C5 top-20 repair on dev: es .36–.73,
  gl .34–.76, random ≤ .11. XL-C5b had no dev run for es/gl; English dev (phi-4 and Gemma) supports it.
  Dev role layers 29, 15 and 37 were not on the dev es/gl grid; the neighbouring layers gave role > 0 in
  every model and language.

## 6. Pre-submission check (filled in before `--submit`)
- Native read of 24 random dev prompts per language (Joel): es ___ ; gl ___ ; problems found: ___
- Queue empty when submitted: ___

## 7. Dev files this addendum is frozen from (sha256)
```
edc89286ef87d897c980b845a73a2d6ed67114913ca9dde605cc97a9af8e4b24  otr09_Qwen__Qwen3.8-27B_dev_es_20261121T194155.json
03339dff0e77a8b98805df57da9459217bf500a569aaedbe50dde572d17ff832  otr09_allenai__Olmo-3.1-32B-Instruct_dev_es_20261121T171302.json
8d9cbcf221d8e79595143eea5a44b0fb97e5a003f1b25bde0a0ad4758b028cb1  otr09_google__gemma-4-31B-it_dev_es_20261121T154613.json
69f3beb67bdd7e852dabaab4104413dfcd74116b0d1ff547b212ad3b347213dd  otr09_ibm-granite__granite-4.2-30b_dev_es_20261121T142211.json
7d68fb96473254a2b987accfb163303988e3059a9c783067e4f9010e539c48e6  otr09_microsoft__phi-4_dev_es_20261121T131940.json
f6ff9c3846c67830559ff544c345b80df718093c819a90331e1b3c33a4e4194b  otr09_Qwen__Qwen3.8-27B_dev_gl_20261121T193403.json
6a71e5091019f6f5effbb572ff60471420c43e3ea18620d52ef2de1d7cc2d167  otr09_allenai__Olmo-3.1-32B-Instruct_dev_gl_20261121T173017.json
dc99cf56b6b9cdc4809e0fd15044078496b78bf928ededcf8e89ae6b378bc97c  otr09_google__gemma-4-31B-it_dev_gl_20261121T161508.json
8d8c2aa14347b8ed957d4d0798e88938bd07b8150de92b76764b7bf5b8c12a0b  otr09_ibm-granite__granite-4.2-30b_dev_gl_20261121T142632.json
a51d03d9210fdbf03da0c91cef5b54d690a3e096bad7dc8ff61f3b064140d9e0  otr09_microsoft__phi-4_dev_gl_20261121T132125.json
```
