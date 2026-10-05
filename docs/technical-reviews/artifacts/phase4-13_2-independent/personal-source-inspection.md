# 13.2 independent original-source inspection

Reviewer: `/root/phase4_factual_coordinator/factual_13_2`, fresh context, 2026-10-05.
This file records my own inspection, rather than an earlier review's conclusions.

I read the original UTF-8 bytes of `course/chapters/13.md#13.2`, lines 35–69,
and the necessary prerequisite 13.1, lines 7–34. I did not read the chapter
introduction because 13.2 is not the first section. `frozen-input-chapter13.md`
is the full Markdown input saved at extraction time, not a assertion of the
current whole-chapter version. The section bytes are saved in `section.md`.
The referenced-image inventory is empty. The token lists and indices are
fully present in the text, so no visual rendering is required for these claims.

I read `factual-reviewer-instructions.md`, the technical checker schema,
`section_facts.py`, `clear-tutorial/SKILL.md` and its `review-protocol.md`.
I followed the factual-review stage rather than the blind first-reading stage.
I read no previous technical/reader reports, history or repository author
result interpretations. JSON fields named `scope`, `purpose`, `warnings`,
`evidence_status` and related interpretation values were not displayed or used.
Original JSON snapshots retain their full bytes and hashes, including unread
fields. Ordinary upstream dataset-card notes were read as part of that original
authority, not as repository review conclusions.

## Authorities personally inspected

- DPO paper: `https://arxiv.org/pdf/2305.18290v3`, first-page title, authors,
  arXiv version/date; §3, printed page 3, preferred/dispreferred completions
  under the same prompt x and equation (1); §4, printed page 4, equation (7)
  uses separate probabilities of yw and yl conditioned on that x. The PDF's
  own version is `2305.18290v3`, 29 July 2024; I verified its bytes rather than
  treating a locator-index assertion as evidence. Saved PDF and pdftotext output
  contain the original source. Text locators are lines 145–168 and 195–240.
- Author implementation, eric-mitchell/direct-preference-optimization,
  commit `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`:
  `trainers.py` lines 90–115 ignore -100 labels and shift raw sequence labels
  once; lines 118–142 concatenate and pad each candidate's own input/labels;
  lines 210–220 feed those matching sequences to the model. In
  `preference_datasets.py` lines 214–277, the same prompt is prepended separately
  to chosen/rejected, EOS is appended, and prompt labels are masked. This
  upstream API starts from unshifted labels; our render_chat has already shifted,
  so its downstream sequence_log_probability must not apply that shift again.
  Both files were fetched from immutable raw.githubusercontent.com URLs with
  default TLS verification. The HTTP log records the commit and content hashes.
- HuggingFaceH4/ultrafeedback_binarized card, immutable revision
  `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`: YAML features/splits, Dataset
  Description, Data Splits and the preference-record schema. The card explains
  that GPT-4 scores full completions; binarization picks the highest overall_score
  and one of the remaining responses. The `train_prefs` and `test_prefs` splits
  are official; the repository source is the first 100 train_prefs rows, not
  its official test set. A fresh HTTPS fetch of the immutable README exactly
  matches the locally saved original, SHA c774e58776969093f0dad0a3ddf031e943f36aeabd34471d29b21fac36469d25.
  The opening arithmetic example's labels are manually assigned; I do not
  interpret that sentence as saying UltraFeedback itself was human-scored.

## Repository contracts and original measurement

AST located functions before I read their computation. For current `data.py`
I read lines 3–28, 46–86: ByteTokenizer, shifted, render_chat, pad_batch.
The data.py hash is also the version recorded by the original DPO experiment.
For `alignment.py` I read only sequence_log_probability, lines 29–35.
For `common.py` I read write_json, lines 33–38, and split_records, lines 50–70;
its full hash matches the original experiment. For current behavior.py I read
imports 10–19, _preference_parts/_pair_examples 550–575 and data/fit method
668–698, excluding the return's scope interpretation. For text.py I read
_digest/_save_splits/_asset_rows, lines 43–88 and _utf8_prefix 374–375.

I retrieved original behavior.py and text.py using `git show` at experiment
revision `8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d`; their complete hashes
match the raw result code_sha256. I personally read original behavior.py
_pair_examples 552–562 and _natural_dpo_pilot 655–685, and original text.py
_json_bytes/_digest/_save_splits 39–62, _asset_rows 75–88 and _utf8_prefix
374–375. I excluded the original return's extra scope text. The current
behavior.py full hash differs; claims about the original pilot use the preserved
original program, not an assumption that the current whole file is identical.

Raw pointer access is listed in `ultrafeedback-verification-result.json`.
Additional raw pointers `/results/ultrafeedback_pilot/sft_training/steps`,
`.../sft_training/records` and `.../training/steps` were inspected: all are 80.
The primary checks execute only the original data-preparation statements,
not any SFT or DPO steps. Training-path pointers identify the original existing
record and do not represent a new run, quality assessment or weight inspection.

## My checks and their limits

The original fence was actually run by section_facts.py in the CPU .venv,
exit 0, no guard events. The result gives exactly the printed input/target lists.
Independent byte construction and small variants check rejected=4, unequal
answer lengths, multibyte bytes, mask invariance, and one-shift alignment.
A declared conditional probability table gives p(3,EOS)=0.3×0.8 using 3's
prefix but 0.3×0.2 after crossing in 2's input. This is a mechanism calculation,
not fitted-model performance. Its log-probability difference is log(4), checked
to 1e-12 with float64. The eight prompt positions are ignored in summation but
remain input context; the exercise does not claim they are removed from attention.

The 100 source rows exactly equal the original viewer response rows, with no
upstream truncated_cells. Original family splitting gives 80/10/10, no overlaps;
each serialized split SHA exactly matches the original measurement's raw SHA.
Prompts/chosen/rejected are actually truncated in 68/88/90 rows. Every excerpt
has at most 120 UTF-8 bytes. Multibyte-boundary checks show complete UTF-8
code points, not a promise to preserve extended grapheme clusters or reasoning.
Input length is bounded by 1+1+120+1+1+120+1−1=244 tokens, below the pilot's
256-token maximum. Each candidate carries its own full token history.

The card's full-answer score cannot certify the label after information is
discarded. I did not relabel the excerpts, prove actual preference reversals,
evaluate chat quality, load an existing model, download training data, train,
use GPUs, or modify course text/figures. The section already states those limits.
