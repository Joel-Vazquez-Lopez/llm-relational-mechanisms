# Third-party data in this folder

## `ud_en_natural.json`: natural English sentences from Universal Dependencies

This file holds 188 sentences, with tokens, argument spans and verb forms, used by `ud_stimuli.py` for the natural-sentence transfer test (OTR-09 `--stimuli ud-en`, development only). Each record names its source treebank, file and `sent_id`, which serves as the attribution for that sentence. The selection procedure is described in the file header. It was built by `scripts/ud_extract_natural.py` from the treebanks' master branches, downloaded 21 November 2026.

| Source | Sentences | Licence |
|:--|--:|:--|
| [UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) | 89 | CC BY-SA 4.0 |
| [UD_English-GUM](https://github.com/UniversalDependencies/UD_English-GUM) | 99 | CC BY-NC-SA 4.0 (repository licence). Some GUM genres carry their own source licences; the used sentences include `whow` (wikiHow, CC BY-NC-SA) and `fiction` (CC BY-NC-SA) |

**Terms.** Sentences from this file may be redistributed only with their attribution (treebank and `sent_id`) and under the same licence. Reuse of the GUM sentences is limited to non-commercial purposes. This notice covers the third-party sentences only. The rest of the repository does not yet declare a licence.

The OTR-09 UD result files in `results/` (`otr09_*_dev_ud-en_*.json`) contain no sentence text, only aggregate repair values.

## Not in the repository: WIKI-ROLES

The WIKI-ROLES item sets (English Wikipedia sentences, dump 20231101.en, CC BY-SA 4.0) are kept out of Git until their licensing in the repository is decided. They are preserved in the Order-to-Role results archive (`papers/order-to-role/archive/`). They can be rebuilt with `scripts/wiki_download.sh` and `scripts/wiki_extract_roles.py`.
