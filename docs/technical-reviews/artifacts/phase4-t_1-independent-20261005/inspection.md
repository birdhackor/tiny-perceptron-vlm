# T.1 independent factual inspection

Reviewer: /root/phase4_factual_coordinator/factual_t_1. Current T.1 and its chapter introduction were personally read, not inherited as summaries. The review is factual, not a blind reader session. No old reader/technical/history/review-dispatch bodies or old T.1 report were read. See unchanged scope-stop.json and subsequent scope-resume.json for the real pause and classification history.

## Inspected original ranges

Current training.md lines 1–170, especially the complete introduction and T.1. Current chapter 05 lines 1–140 plus complete 5.10 were read as contextual tutorial claims, not authority. Current chapter 19 lines 1–84 were read to verify that the limited product, OCR and speech material in T.1 refers to a separate planned scope; this review did not evaluate chapter 19's other results. W.1/W.7 and DATA.md were only checked for file/heading existence and raw SHA; DATA.md contents were not read.

Required method files were read in full. Original code was first located using rg --files, then AST function/assignment/import spans were inspected: fetch_training_assets.py main 13–42; assets.py sha256 10–15 and unpack_asset 18–72; prepare_data.py conversation 12–19, generate_records 22–81 and main 84–128; data.py shifted 46–51 and render_chat 54–68; modal_data.py modal_example 13–34. General comments/API/docstrings were read. build_training_assets.py was inspected only as AST names/imports/line spans. The snapshot of data.py is retained whole, but its other implementation bodies were not read.

Training manifest and source JSON were not printed recursively or in full. Top-level key/type inspection preceded value reads. The named metadata pointers for each package are recorded in bounded stdout's manifest-list-provenance item. For TinyStories, /source_url, /source_revision, /source_split, /local_split, /selection_method, /normalization, /license, /training_permission, /redistribution_permission, /attribution, /modifications, /license_evidence, /checks and /files were inspected. The permission and selection claims remain package method assertions; they do not substitute for official original sources. TinyStories JSONL raw first three lines were preserved and personally read at /text, /source_revision and /license; their full raw member fingerprint and excerpt fingerprint are recorded. Other archive members were hashed as bytes, not semantically read.

## Personally checked original external sources

OpenAI's original 2019 GPT-2 report, page 2 §2 Approach Eq. (1), specifies sequential language modeling and p(output|input,task), and page 3 contrasts output-only supervised objective with a whole-sequence objective. This supports task-dependent answers and next-token labels from text; no reported GPT-2 score is used here.

Official scikit-learn release 1.7.2 cross_validation.rst lines 9–20 and 60–70 define held-out evaluation and train/validation/test purposes. Lines 626–657 specify dependent groups, unseen-group evaluation and ensuring the same group does not appear on both sides. common_pitfalls.rst lines 77–115 explains leakage and optimistic evaluation. These support keeping derived questions/crops/noisy recordings with the original family when evaluating new families, without guaranteeing a numeric effect for every model/run.

TinyStories maintainers' README at f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 was personally read in full (short original document). It declares text generation, English short stories, CDLA-Sharing-1.0, and train/validation filenames. The Linux Foundation's original CDLA-Sharing 1.0 definitions and §§2–3 were read, including computational use, conditional Use/Publish, attribution and license retention. This confirms the linked TinyStories license label and its conditions, not a blanket legal certification of every package.

All HTTPS URLs, retrieval date, byte counts and SHA-256 are in external/retrieval.json. Snapshots are original documents. scikit-learn 1.7.2 is an inspected documentation release, not asserted to be installed; execution versions are independently recorded in execution-provenance.json.

## Execution and support limits

The first original shell command --list ran successfully with python resolved to the repository .venv. The second command can pull only the selected archive via Git LFS when it is a pointer, then validates archive size/SHA and member size/SHA before committing data/training files. It does not import or run a model/optimizer. The long download recipe was not run. A two-record mini archive exercised the actual unpack helper; wrong archive SHA and changed destination contents were rejected. The existing local TinyStories archive/member SHA were independently checked, and only three original JSONL records were preserved for content inspection. No new data or model weights were downloaded.

The generator produced the exact A/B/C rows, including describe→shape and a shape?→color? variation. Its actual small CLI generated 60 synthetic questions across 12 families, splitting complete families; A and B were on train and C was held out in validation at seed 42. Those counts describe the verification fixture, not a new model score. The original next-token helper mapped [11,12,13,14] to X=[11,12,13], Y=[12,13,14]. No parameters were created or updated.

The modality helper converted a readable 7×9 RGB image to its required 3×16×16 tensor. A readable 8 kHz waveform was rejected; a synthetic 16 kHz mono waveform passed. This checks the stated need for dimensions/sample-rate/task-answer contracts and does not certify the suitability of a natural dataset or the assistant's ability.

There are no referenced figures in T.1. Literal input/answer/family strings are sufficient to inspect the grouping example; no render/view was required or performed. No claim in this section reports measured learning performance. All substantive concepts, table conventions, commands, data-provenance references and scope advice are covered by report claims. No substantive unresolved issue was found.
