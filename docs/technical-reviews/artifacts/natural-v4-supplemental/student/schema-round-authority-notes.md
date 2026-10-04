# Same-owner factual evidence completion after first FINAL

The first genuine FINAL report is preserved byte-for-byte as
`report.first-genuine-FINAL.json` (SHA-256
`966e2712c5dc949accfb9d62dbed993368b523847efcadb7cd49eecefd8a8039`).
The actual coordinator receipt with schema errors is preserved as
`coordinator-observed-first-FINAL-errors.json`. I personally read the unchanged
official `_artifacts`, `_sources` and `_claims` implementations. Software,
numeric and empirical claims need explicit verification; expected/observed are
nonempty strings. My first FINAL omitted software verification dictionaries
and used structured objects in three expected/observed fields. These were
report defects, not newly discovered contradictions in the guide.

I personally read additional necessary paragraphs of the original pinned
Whisper card, Evaluated Use and Performance and Limitations, lines 487–510.
The official card recommends evaluation in the particular context/domain,
warns that ASR may produce unspoken text and that accuracy differs across
languages/conditions. It does not establish this project's chat performance.
This directly supports retaining separate stage/task checks and declining to
infer unrestricted quality from one successful operation.

I retrieved and personally read the full original HF Evaluate CER metric
source at pinned space revision `76a87418a42c9c8e6e6832daed7979456774aba8`:
`https://huggingface.co/spaces/evaluate-metric/cer/raw/76a87418a42c9c8e6e6832daed7979456774aba8/cer.py`.
Full bytes stay in ignored `outputs/natural-v4/factual-research/student/`;
actual retrieval hashes are in `authority-retrieval-schema-round.json`.
The `_DESCRIPTION`, `_KWARGS_DESCRIPTION` and `_compute` define comparison
to a supplied reference, count character substitutions/deletions/insertions,
and use reference character count as denominator. Thus a character-copy task
has a different target from a semantic summary or a downstream chat answer.
My original transparent task reasoning covers that distinction. I do not
claim that this course uses HF CER's default normalization, that its published
scores were reproduced, or that a zero ASR CER proves a correct chat answer.

Existing actual CPU/browser stdout and original record audits supply explicit
expected/observed statements. Installation, new Git clone, true inference and
GPU work remain unexecuted. Additional bounded offline probes specifically
exercise an interrupted two-small-file stage, successful retry to the same
output, missing-file rejection with remaining files preserved, and actual
tiny WAV/FLAC/MP3/OGG decoding/upload validation in current SoundFile0.14.0.
Those probes are scoped to the actual current environment and do not replace
the pinned runtime's original code/records. Original numeric objects remain
preserved as expected_values/observed_values alongside their string fields.
