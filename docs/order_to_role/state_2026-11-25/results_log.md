## NEW 20 Nov (overnight batch): the failure generalises
Accuracy on the failing question (bare name). Defaults → failing.
- **BX transitive (new names and verbs, 18 constructions):** Gemma .99 → .86, Qwen3.8 .90 → .61, OLMo-3.1 .88 → .41, phi-4 .90 → .13, Granite .90 → .85. Active-voice sentences are worst (Gemma active .72, coord_active .78, perfect_active .71). Anomaly to inspect: long_object_cleft is low on DEFAULT questions for OLMo/phi/Granite/Qwen (.34–.56) but high under the failing question. Check the raw outputs before using it.
- **BX ditransitive ("Who was someone given {theme} by?"):** Gemma 1.00 → .56 (pd_active .17), phi-4 .21, Qwen3.8 .82, OLMo .80, Granite .96.
- **BX psych:** Gemma .95 → .45. fear-class active (es_active) .07: Gemma answers the grammatical object (the one who is feared). This supports (b') beyond agentivity: the local "who was V-ed" chunk fetches the object, whatever its thematic role. Qwen3.8 .31, phi-4 .35, OLMo .63, Granite .74.
- **Spanish/Galician, new models (failing agent question on actives):** OLMo es .00 / gl .22; phi-4 es .09 / gl .18. A full sentence recovers Spanish (OLMo .93, phi-4 1.00) but Galician less.
- **Swedish (UNCHECKED, do not report):** cleft agent questions are fine (Gemma 1.00). Stranded "Vem blev någon {pp} av?" fails: Gemma .36 (active .48, by_cleft .18), phi-4 .29, OLMo .10. s-passive is worse (phi-4 .09, OLMo .14). Pied-piped also fails (Gemma active .19). A full sentence recovers it: Gemma 1.00, phi-4 .87, OLMo .69. So there is a "knows but answers wrong" gap in a third language.
- **Missing:** Swedish for Qwen3.8 and Granite; es/gl for Granite (not submitted or no output).
- Instructions used: sv "Svara med ett enda namn." / "Svara med en fullständig mening." (n 512); es "Responde con un solo nombre." / "Responde con una frase completa."; gl "Responde cun só nome." / "Responde cunha frase completa." (n 1640).
- **Next build:** OTR-03 + OTR-09 on BX items, to test whether the name role code and the repair heads transfer to new verbs, names and constructions, and whether the code is "agent" or "proto-agent / subject-of-active" in the psych block. Then es/gl mechanism. Then UD.


## 20 Nov evening: mechanism on BX (patch 0051) and OTR-07 in all five models
- **Role code transfers (OTR-03, BX transitive, normal questions):** Gemma +1.13 @0.48, Qwen3.8 +0.71, phi-4 +0.55, OLMo +0.49, Granite +0.87; validity null 0; predictors exactly decorrelated. Function now small positive (+0.08 to +0.36).
- **Psych:** role (stimulus vs experiencer) +0.65 to +1.01 in fear-class, where subject = experiencer → the name code is thematic, not subject. Within-class order–function corr .33.
- **Failing question on BX:** psych wrong items reversed in 5/5 (Gemma −0.53, Qwen −0.13, phi −0.40, OLMo −0.33, Granite −0.09); transitive reversed phi −0.75, OLMo −0.28, Qwen −0.14; Gemma +0.30 and Granite +0.27 weakened (Gemma was reversed on the original set). Ignore `order` in single-role fits.
- **Repair heads transfer (OTR-09 --rank-from):** onset same depth as original (Gemma L40, phi L20–22, OLMo L24, Granite L52, Qwen L50–60); old top-20 heads repair BX as well as fresh ones (transitive: Gemma .78, phi .74, Granite .50, OLMo .35, Qwen .34; random ≈0); 13–19 of top-20 overlap; same in ditransitive and psych.
- **OTR-07, all five:** wrong-bare items still give the agent at the full-sentence "<patient> was V-ed by" slot (Gemma −7.5/+19.0, Qwen −1.2/+8.8, phi −6.8/+11.1, OLMo −4.7/+14.1, Granite −2.1/+8.2). Caveat: prefix names the patient → control with "Someone was V-ed by".
- 11 of 48 BX names dropped for non-Gemma tokenizers (recorded in results).
- Pelle push needs a live VS Code connection (credential helper); reconnect + new terminal if "vscode-git sock" error.

## 21 Nov: OTR-07 by-slot control (patch 0052, --by-prefix "Someone was {verb} by")
Wrong-bare items, share whose full-sentence slot still picks the agent (final margin): Gemma 1.00 (+13.7), Granite .85 (+2.2), Qwen .83 (+2.2), OLMo .59 (+1.5; actives −0.19, passives/by-clefts +3.5), phi-4 .29 (−1.0). Knows-but-answers-wrong holds unaided in Gemma, Granite, Qwen; partly OLMo; not phi-4 (the patient name in the prefix did the work). Report both prefixes.

## 21 Nov evening: mechanism in Spanish/Galician (patch 0053)
- OTR-03 normal: role code es/gl Gemma +0.68/+0.92, Granite +0.63/+0.43, Qwen +0.74/+0.70, OLMo +0.46/+0.24, phi +0.37/+0.35; function ALSO positive in es/gl (+0.1–0.6; unlike English). Null 0.
- OTR-03 failing (passive agent q, wrong items): es Gemma +0.55, Granite −0.11, Qwen +0.11, OLMo −0.25, phi −0.29; gl Gemma −0.82, Granite +0.12, Qwen −0.08, OLMo −0.11, phi −0.13 → reversed in 7/10.
- OTR-09: English top-20 heads repair es/gl as well as fresh (Gemma .73/.76, Granite .59/.62, phi .69/.62, OLMo .44/.39, Qwen .36/.34; random ≈0–.1); overlap 14–17/20; onset as in English except Gemma 0.78 (vs 0.68).
- Pending: CIs + final figures (not started yet).

## 21 Nov night: natural sentences (UD, patch 0054, OTR-09 --stimuli ud-en)
188 UD EWT/GUM sentences (348 active / 26 passive items), names substituted. Failing q. on actives / working q.: Gemma .50/1.00, Granite .71/.99, phi .41/1.00, Qwen .22/1.00, OLMo .15/.99. Repair onset Gemma L40, Granite L52, phi L22, OLMo L24, Qwen L60 (0.95, later than templates). Template-ranked top-20 heads repair real sentences as well as fresh: Gemma .81, phi .67, Granite .52, Qwen .42, OLMo .39 (random ≈0); overlap 15–18/20.

## 23 Nov: WIKI-ROLES (patches 0055–0056) — our dataset
4,500 Wikipedia sentences (3,000 active / 1,500 passive, 87 verbs, ≤150/verb/voice) from 1.12M articles in 1.8 h (GPU node CPU cores; CPU queue was days). First run died on BlockingIOError (fixed: node-local scratch + retry). OTR-09 800 items/model, real vs neutral names: failing q actives Gemma .53/.50, Granite .78/.77, phi .53/.42, Qwen .28/.25, OLMo .33/.24; passives phi .22/.13, OLMo .20/.30 fail too; working q ≥.96. Template heads repair real Wikipedia as well as fresh (Gemma .76/.82, phi .66/.71, Granite .42/.47, OLMo .37/.45, Qwen .34/.36; random ≤.1); overlap 13–19. Real ≈ neutral → not memory. Check sheet (200) sent to Joel 23 Nov.
Pelle push gotcha: if a tracked results file changed, commit BEFORE `git pull --rebase`.

## 23 Nov: OTR-13 question words (patch 0057)
Instruction sits on "by" (not the verb, contrary to prediction). Failing "by" into working q destroys margin: Granite 1.55, OLMo 1.12, Gemma .78, Qwen .59, phi .35 (peaks 0.13–0.48 depth); working "by" into failing repairs OLMo .75, phi .57, Granite .49, Gemma .32, Qwen .22. "someone": earliest layers only. Verb ablation worsens the error. Whole-question swap repairs .85–1.0 early-mid and drops to 0 where the answer-position patch rises (Gemma ~.65, phi ~.5, OLMo ~.38, Granite/Qwen ~.75) → question→answer handover. Figure otr13_question_to_answer.png (made in chat).

## 23 Nov evening: OTR-13 checks (patch 0058)
- BX: "by" carries the instruction in 5/5 (reverse damage Granite 1.06, OLMo .95, Gemma .58, Qwen .49, phi .26; repair OLMo .84, phi .60, Granite .45, Gemma .37, Qwen .18); handover depths same. phi: verb also .50 on BX.
- Second working q "Who was the {ing} of someone done by?": end-of-question still carries the difference ("by" damage Granite .67, Qwen .49, OLMo .32, phi .30; Gemma moves to "?" .53); "someone" matters more mid-layers. → depends on "someone V-ed" directly before "by", not on "done". Few wrong items in Gemma/Granite (8–16). Ignore joint for src2 (fewer counterpart words).
- Next: WIKI-ROLES v0.2 (patch 0060) extraction; API runs (0059) after.

## 24 Nov evening: WIKI-ROLES v0.2 OTR-09 (8 of 10; Qwen held in queue)
Bare-answer logit check (agent preferred), failing question, active/passive; working question ~.99 everywhere.
| model | names | fail act | fail pas | handover (first ≥ .5) | frozen dev 20 heads | random 20 | overlap with fresh top-20 |
|---|---|---|---|---|---|---|---|
| Gemma | real / neutral | .55 / .54 | .76 / .80 | 40 / 40 (dev 40) | .77 / .83 | .08 / .09 | 15 / 17 |
| Granite | real / neutral | .73 / .76 | .98 / .95 | 52 / 52 (dev 52) | .42 / .46 | −.04 / −.04 | 14 / 16 |
| phi-4 | real / neutral | .51 / .39 | .18 / .15 | 22 / 22 (dev 22) | .67 / .71 | .08 / .10 | 18 / 19 |
| OLMo | real / neutral | .23 / .20 | .24 / .34 | 22 / 24 (dev 26) | .37 / .45 | .00 / .00 | 14 / 17 |
- On ~800 natural Wikipedia sentences per run, the dev handover layer reappears exactly in Gemma, Granite, phi-4; OLMo 2–4 layers earlier.
- The 20 heads frozen on templates repair natural text far above random; real and neutral names agree (not memory).
- Role layers for the freeze (dev OTR-03 argmax): Gemma 28, Qwen3.8 29, phi-4 15, OLMo 18, Granite 37.
