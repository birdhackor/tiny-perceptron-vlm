# Phase 5 implementation contract

Phase 4 closed on 2026-10-06; see docs/course-revision-20261005/verification/phase4-closure.json.
This is engineering coordination, not textbook prose or measured capability.
Authoritative scope: docs/course-revision-20261005/sources/selftrained-capstone.md.
Keep old toy milestones and mature-model extensions intact. New code belongs in
tiny_perceptron/selftrained/ and scripts/selftrained/. Do not edit course, notebooks,
old review reports, or prior experiment results in this phase. Root integrates and
controls commits, paid dispatches, dataset upload, and weight release.

## Shared records

Asset directory: outputs/selftrained/data/. Producers own text-tools, vision, ocr,
and voice JSONL plus separate provenance manifests under outputs/selftrained/manifests/.
Each record has id, group_id, split (train/validation/test), task, and ordered messages
with role and content. Roles: system/user/assistant/tool. Last assistant is the target.
Optional image/audio paths are relative to asset directory. Optional supervision
holds training/evaluation labels and NEVER enters inference prompts/neural inputs.
All descendants/paraphrases/augmentations of a source group share one split.
Freeze actual IDs and evaluation rules before final training/test; use validation
for choices. Do not adjust test criteria after results. Preserve real dialogue history.

Every modality answers through the same language core. All neural components start
random; freezing one's own separately trained perception weights is allowed.
Disclosed neural perception symbols may feed the same core, as the scope permits.
A runtime lookup must not return full assistant answers. Never feed gold labels,
test transcripts, filenames' class labels, or answer tables to the model.

## Ownership

Visual/OCR data: scripts/selftrained/prepare_vision_ocr.py, vision/ocr data, images,
fonts/notices/manifests. Fashion-MNIST source IDs split before augmentation.
Finite known-glyph strings/layout families separated; legally attributed fonts.

Voice data: scripts/selftrained/prepare_voice.py, voice data/audio/notices/manifest.
Report actual source/class/duration/duplicate counts. Missing speaker/session IDs
stay explicit; no invented speaker-disjoint or manual-listening claim. No mature ASR.

Text/tools: scripts/selftrained/prepare_text_tools.py, tiny_perceptron/selftrained/tools.py,
text-tools data/manifest. Original limited Chinese dialogue, history/format variants,
changed-information controls, and meaningful unsupported/clarification cases.
Calculator add/subtract/multiply with bounded integers, model-generated JSON call,
true executor, tool role, then second generation by the same core. No eval/shell.
Changed-result control verifies tool-message use; non-tool cases are separately tested.

Architecture: tiny_perceptron/selftrained/model.py and tokenizer/encoding module.
Reuse this repo's learned-layer code, never mature weights. Main MoE plus transparent
Dense comparison. Agree concrete forward/generation/tensor API with trainer owner.
Report actual total/active/perception parameter counts and comparison assumptions.

Training/evaluation: scripts/selftrained/train.py, evaluate.py and dataset/collation module.
Auditable pretrain/SFT/perception/joint stages, resume/checkpoints, seed, validation
selection, frozen final test, raw outputs and per-task/control metrics. CPU small
correctness smoke first; root controls paid jobs. Supervision stays out of prompt.

GPU infrastructure: bounded new Modal runner/workflow and HF release plumbing.
Preserve the REAL shared /course/budget.json; total authorized ceiling US$40.
Earlier reservations are not invoices. Read the live ledger and rates before jobs;
reserve failures/retries and save resumable artifacts. Modal HF secret: codex_cloud.
Modal token ID is a repository variable; token secret remains secret. Never print tokens.
No paid dispatch/publication until root integrates concrete code/data/bounds.
Dataset packages use Git LFS; weights use HF birdhackor/tiny-perceptron-course-models.

Producers/consumers message each other for concrete interfaces. Report actual
downloads/tests and limitations to root. Capabilities/results are honestly reported
in Phase 6; do not rewrite established teaching around experimental numbers.
