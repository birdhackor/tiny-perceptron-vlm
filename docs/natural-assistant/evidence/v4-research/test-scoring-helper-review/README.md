# Final-test scoring helper review

`scripts/score_natural_v4_test.py` reads the actual persisted `review/result.json`,
selected `generations-*.json`, and `transcripts.json`. The outer workflow
`result.json` is a wrapper, not an inference result. This helper performs no
inference, training, downloads, publication, or model selection. Actual v4
validation/test predictions were unavailable while this helper was built; its
tests use constructed artifacts and synthetic grade decisions.

The input selection must already be committed. Its exact bytes, manifest,
validation protocol, ASR selection, and runtime source files must match the
actual test's `execution.revision`. The selection's validation evidence hashes,
foundation model pin and adapter run/checkpoint/weight hash must agree with the
one executed variant, `execution.selection_sha256`, and `pretest_selection`.
The actual report must declare `split=test`, `execution.stage=evaluate`, and
`selected_only=true`; extra variants/files and partial or repeated rows fail.
The helper never emits `selection.json` or makes a new selection decision.

The blind private binding and final scores also include `scoring_sources`: exact
SHA-256 and byte counts for the actual loaded final-test scorer, imported
validation scorer, and imported core source, read through their `__file__`
locations under stable literal source keys. These are current grading-source
hashes; the new scorer need not have existed at the older inference commit.
Export and score recheck these bytes against artifact-load hashes, including
when using a stale in-memory artifact object. Reloading changed scoring code
still fails the original blind packet's private binding. This closes the fresh
reviewer's reproduced preserve-case OCR gap: casefolding the copied scorer after
blind export previously changed nine correct OCR responses to ten; the same
change now rejects scoring and writes no output.

The foundation pin here is the model ID and immutable Hugging Face revision,
plus actual prepare/download hash evidence recorded elsewhere. The inference
runtime uses its pinned local cache and does not physically rehash the full
foundation weight file on every forward. This helper does not claim such a
per-forward weight-file measurement.

There are 178 LM cases: 170 visual/text rows plus four paired voice questions
evaluated once through their reference text and once through actual ASR. The
eight group denominators are 42 summary, 84 narrow facts, 18 presence, 10 single
OCR, three ordered OCR, 13 chat, four typed voice, and four actual voice.
The 147 semantic cases require independent manual review under the frozen
per-question rubrics. There are 22 ASR hypotheses, including 18 transcription
recordings and the four voice questions.

The helper preserves the frozen validation rubric and raw-EOS mask. Every LM
case remains in its group denominator; a missing, unknown or truncated EOS
completion fails even if a manual grade passes. An EOS at token 384 is complete.
Complete photo answers require `source_image_inspected: true`. Graders must name
themselves, attest `independent_of_training_and_selection: true`, and attest
`private_mapping_consulted: false`. These attestations are audit evidence, not a
programmatic proof of reviewer independence. An incomplete answer still needs a
grade/reason, but does not require another photo inspection.

Presence uses whole-answer NFKC/outer-strip equality. Single OCR additionally
removes Unicode whitespace. Ordered OCR retains internal spaces and newlines.
There is no case, punctuation or Traditional/Simplified conversion beyond NFKC.
Chat uses semantic grades, regardless of any runtime exact-match score.

The summary includes group counts, exact rational group accuracy, the same
five-capability descriptive arithmetic as the frozen protocol with **test**
denominators, and correct-response micro accuracy over 178. It does not reuse the
validation integer weights or `/75600` denominator, compare candidate scores,
apply validation selection gates, or claim population performance. Related
questions are not independent trials.

ASR raw and normalized micro CER are independently recomputed from reference
and actual hypothesis strings with codepoint Levenshtein distance, ignoring
saved runtime CER. The declared normalization is NFKC plus removal of Unicode
whitespace, preserving case/punctuation after NFKC and without script conversion.
The report keeps errors, reference characters, exact rational CER, task groups,
per-record counts and ASR completion evidence; CER can exceed one. ASR completion
is reported separately and is not an extra, newly invented LM grading gate.
ASR CER does not contribute to LM capability scores.

Raw generations verify the current user, source image, family and reference
answer. Actual voice responses must use exactly the corresponding actual ASR
hypothesis and retain the original reference question. The runtime does not log
rendered system/history tokens per generation: those contexts are reconstructed
from the committed manifest and bound runtime sources. The report explicitly
records that evidence limit; it does not invent prompt-log fields.

After the actual selected test has completed, run this from the repository root
with the real committed selection path and persisted review directory:

```bash
.venv/bin/python scripts/score_natural_v4_test.py export \
  --selection docs/natural-assistant/v4/selection.json \
  --test-dir outputs/natural-v4/selected-test/review \
  --blind-dir outputs/natural-v4/test-review
```

These selection/test paths are illustrative, not evidence that an actual v4
test or selection currently exists. Defaults use the frozen v4 manifest,
protocol, and `outputs/natural-v4/data`; `--manifest`, `--protocol`, and
`--data-root` can specify local paths. The selection, manifest, protocol and ASR
selection paths must remain inside the repository and match the execution
commit. The source files bound by this helper are
`tiny_perceptron/natural_assistant.py`, `scripts/natural_assistant.py`, and
`scripts/modal_natural.py`.

Give a fresh independent grader only `blind-packet.json`,
`grades-template.json`, and the corresponding actual source images. The packet
contains alias A and conceals the selected model identity; there is only one
candidate because model selection already happened. Keep `private-map.json`
and original candidate filenames from the grader. Complete the template as
`grades.json`, then run:

```bash
.venv/bin/python scripts/score_natural_v4_test.py score \
  --selection docs/natural-assistant/v4/selection.json \
  --test-dir outputs/natural-v4/selected-test/review \
  --blind-dir outputs/natural-v4/test-review \
  --grades outputs/natural-v4/test-review/grades.json \
  --output outputs/natural-v4/test-scores
```

Only `scores.json` is written. Existing nonempty review/score output directories
are refused. Save the score, completed grades, blind/private binding, selection,
and source artifacts for audit. Test results cannot alter the selected weights,
gold, normalization, generation settings, or prompt context.

The executed artifact tests and source hashes are recorded in `review.json`;
the final raw output, JUnit results and source snapshots are under `run-002/`.
The prior `run-001/` and `review-before-grading-source-fix.json` retain the
initial checks before fresh review exposed the grading-source binding gap.
These tests verify artifact rejection and scoring behavior, not the trained
model's performance.

The [fresh independent review](../test-scoring-independent-review/read-report.json)
closed the reproduced grading-source gap: the original post-export casefold
change now exits with a binding error and creates no scores directory. The
reviewer independently passed all 129 final-test helper checks; the author's
155-check combined suite also covers the unchanged validation scorer and
selected-only runtime. No further concrete defect was found within that scope.

A subsequent fixture-only portability patch sets `core.autocrlf=false` inside
each temporary test Git repository. Its [supplemental receipt](fixture-portability/receipt.json)
records five passing targeted checks, including constructed CRLF JSON under an
inherited `core.autocrlf=true`: the unfixed baseline loses byte equality while
the fixed full fixture preserves it. This was run on the current Linux host,
without an actual Windows execution claim. The production scorer hash is
unchanged; the earlier 155-pass receipt remains evidence for its original test
snapshot, and a new full-suite run is not claimed for this fixture-only patch.
