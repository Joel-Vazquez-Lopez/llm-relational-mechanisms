# Test set 0: attempt log

Every submission of a test0 job, as FREEZE.md section 6 requires. `scripts/slurm/otr_test0_submit.sh --submit`
appends one row per job; add a row by hand for each rerun after a technical failure (crash, timeout,
out of memory, partial output), with the same label and the outcome.

| When | Commit | Job | Slurm ID | Outcome |
|---|---|---|---|---|
| 2026-11-26 10:10 | f6ea0e1 | C1 fail google/gemma-4-31B-it | 8028342 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 work google/gemma-4-31B-it | 8028343 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C2 google/gemma-4-31B-it | 8028344 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C3 google/gemma-4-31B-it | 8028345 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C4/C5 google/gemma-4-31B-it | 8028346 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C6/C8 google/gemma-4-31B-it | 8028347 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C7 google/gemma-4-31B-it | 8028348 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 fail Qwen/Qwen3.8-27B | 8028349 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 work Qwen/Qwen3.8-27B | 8028350 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C2 Qwen/Qwen3.8-27B | 8028351 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C3 Qwen/Qwen3.8-27B | 8028352 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C4/C5 Qwen/Qwen3.8-27B | 8028353 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C6/C8 Qwen/Qwen3.8-27B | 8028354 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C7 Qwen/Qwen3.8-27B | 8028355 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 fail microsoft/phi-4 | 8028356 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 work microsoft/phi-4 | 8028357 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C2 microsoft/phi-4 | 8028358 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C3 microsoft/phi-4 | 8028359 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C4/C5 microsoft/phi-4 | 8028360 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C6/C8 microsoft/phi-4 | 8028361 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 fail allenai/Olmo-3.1-32B-Instruct | 8028362 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 work allenai/Olmo-3.1-32B-Instruct | 8028363 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C2 allenai/Olmo-3.1-32B-Instruct | 8028364 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C3 allenai/Olmo-3.1-32B-Instruct | 8028365 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C4/C5 allenai/Olmo-3.1-32B-Instruct | 8028366 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C6/C8 allenai/Olmo-3.1-32B-Instruct | 8028367 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 fail ibm-granite/granite-4.2-30b | 8028368 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C1 work ibm-granite/granite-4.2-30b | 8028369 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C2 ibm-granite/granite-4.2-30b | 8028370 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C3 ibm-granite/granite-4.2-30b | 8028371 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C4/C5 ibm-granite/granite-4.2-30b | 8028372 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C6/C8 ibm-granite/granite-4.2-30b | 8028373 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | C7 ibm-granite/granite-4.2-30b | 8028374 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C2 google/gemma-4-31B-it | 8028375 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C3 google/gemma-4-31B-it | 8028376 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C5 google/gemma-4-31B-it | 8028377 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C2 Qwen/Qwen3.8-27B | 8028378 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C3 Qwen/Qwen3.8-27B | 8028379 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C5 Qwen/Qwen3.8-27B | 8028380 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C2 microsoft/phi-4 | 8028381 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C3 microsoft/phi-4 | 8028382 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C5 microsoft/phi-4 | 8028383 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C2 allenai/Olmo-3.1-32B-Instruct | 8028384 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C3 allenai/Olmo-3.1-32B-Instruct | 8028385 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C5 allenai/Olmo-3.1-32B-Instruct | 8028386 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C2 ibm-granite/granite-4.2-30b | 8028387 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C3 ibm-granite/granite-4.2-30b | 8028388 | submitted |
| 2026-11-26 10:10 | f6ea0e1 | BX C5 ibm-granite/granite-4.2-30b | 8028389 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 es google/gemma-4-31B-it | 8046976 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 es google/gemma-4-31B-it | 8046977 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 es google/gemma-4-31B-it | 8046978 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 es google/gemma-4-31B-it | 8046979 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 es Qwen/Qwen3.8-27B | 8046980 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 es Qwen/Qwen3.8-27B | 8046981 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 es Qwen/Qwen3.8-27B | 8046982 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 es Qwen/Qwen3.8-27B | 8046983 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 es microsoft/phi-4 | 8046984 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 es microsoft/phi-4 | 8046985 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 es microsoft/phi-4 | 8046986 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 es microsoft/phi-4 | 8046987 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 es allenai/Olmo-3.1-32B-Instruct | 8046988 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 es allenai/Olmo-3.1-32B-Instruct | 8046989 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 es allenai/Olmo-3.1-32B-Instruct | 8046990 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 es allenai/Olmo-3.1-32B-Instruct | 8046991 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 es ibm-granite/granite-4.2-30b | 8046992 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 es ibm-granite/granite-4.2-30b | 8046993 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 es ibm-granite/granite-4.2-30b | 8046994 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 es ibm-granite/granite-4.2-30b | 8046995 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 gl google/gemma-4-31B-it | 8046996 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 gl google/gemma-4-31B-it | 8046997 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 gl google/gemma-4-31B-it | 8046998 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 gl google/gemma-4-31B-it | 8046999 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 gl Qwen/Qwen3.8-27B | 8047000 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 gl Qwen/Qwen3.8-27B | 8047001 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 gl Qwen/Qwen3.8-27B | 8047002 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 gl Qwen/Qwen3.8-27B | 8047003 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 gl microsoft/phi-4 | 8047004 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 gl microsoft/phi-4 | 8047005 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 gl microsoft/phi-4 | 8047006 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 gl microsoft/phi-4 | 8047007 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 gl allenai/Olmo-3.1-32B-Instruct | 8047008 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 gl allenai/Olmo-3.1-32B-Instruct | 8047009 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 gl allenai/Olmo-3.1-32B-Instruct | 8047010 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 gl allenai/Olmo-3.1-32B-Instruct | 8047011 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C1 gl ibm-granite/granite-4.2-30b | 8047012 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C2 gl ibm-granite/granite-4.2-30b | 8047013 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C3 gl ibm-granite/granite-4.2-30b | 8047014 | submitted |
| 2026-11-26 19:11 | c40a1f7 | XL-C4/C5 gl ibm-granite/granite-4.2-30b | 8047015 | submitted |
