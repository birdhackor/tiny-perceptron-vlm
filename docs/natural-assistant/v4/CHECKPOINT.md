# Natural-input v4 execution handoff

This file records engineering state, not a student lesson or a completed release.

## Frozen and verified

- Data manifest: `docs/natural-assistant/v4/manifest.json`, SHA-256 `0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60`. Three data archives total 146,543,309 compressed bytes; 1,513 declared files total 160,683,823 unpacked bytes.
- 2,371 text/image records: 2,077 train, 124 validation, 170 test. There are also 38 heldout audio recordings: 16 validation and 22 test. The CPU encoding audit passed all rows, longest full sequence 630 tokens, 39,436 supervised training tokens. Current source-status binding is preserved as an append, rather than changing earlier audit bytes.
- Legal-source data: Google DOCCI CC BY 4.0; NVIDIA OCR CC BY 4.0; individually approved Commons CC BY/CC0/public-domain photos; OpenAssistant Apache 2.0; FLEURS CC BY 4.0; AISHELL-1 Apache 2.0. Excluded NC/SA preview collections are absent from the snapshots.
- Accepted DOCCI photographs: 439 train, 28 validation and 42 test. All added activity and heldout photographs received actual visual author and independent AI peer review. One rejected activity candidate remains excluded from training.
- OCR: 710 single-line plus 117 multiline NVIDIA crops, 62 approved Commons crops, whole-image region questions and balanced text-presence training. A genuinely fresh public-source reconstruction in this cloud workspace verified all 978 OCR artifacts from 89 photos and 153 selected NVIDIA pages; NVIDIA range transfer was 69,982,238 bytes, below 128 MiB, with no HTTP429. This is distinct from the GitHub runner replay that was interrupted by Wikimedia HTTP429.
- ASR was selected on the same 16 validation recordings before test: small normalized CER 117/510, turbo 50/510; raw CER also improved. All LM candidates use the same turbo transcripts. No ASR training occurred.
- Validation protocol is committed before LM outputs. It compares base and archives at updates 1,039 and 2,077, with blinded semantic grades, exact row normalization, raw EOS completion guards, integer macro scoring and explicit nonregression gates. The scorer has export/score commands and generates an immutable pre-test selection.
- Actual prepare/train/validation execution source is `9a61ecf524c9518f33f1501c28aa72997d4a82d0`; the feature branch also contains subsequent independent final-test scorer fixes. Its three archive paths contain exact 133-byte Git LFS pointer blobs matching the frozen manifest. Native Actions upload and independent LFS download verified all three objects; the exact receipt is preserved in [data-publication evidence](../evidence/v4-research/data-publication/lfs-transfer-receipt.json).
- Canonical chapter 20 has 13 coherent sections, four additional SVGs, three v4 operation guides and migrated links. CPU examples and static links passed. Final test/release information is still pending. Fresh readability review has started; factual, continuity and final reading-time review must follow in that order.
- Optional research-proof base and supplement are packaged and verified twice; small indices/notices are committed. Raw proof is outside ordinary Git. Publication has not been verified.

## Current execution state

Data publication completed in [Actions run 37212727035](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37212727035). The native uploader registered the three frozen objects, and independent LFS downloads matched their sizes and SHA-256 hashes: 146,543,309 bytes in total. The temporary `natural-v4-lfs-byte-transfer` ref is absent remotely and locally. The ordinary feature tree contains none of the temporary transfer paths. Exact-copy, committed-blob and read-only ref checks are recorded in [publication-closure.json](../evidence/v4-research/data-publication/publication-closure.json).

The two CPU source-bootstrap attempts, `37211774915` and `37212355816`, failed and retained US$0.62 each. Their final cumulative reservation was US$26.05 before the subsequent model/data prepare attempt. These are conservative ledger reservations, not an invoice or measured billing total. The shared authorization remains US$40; no ledger reset occurred.

The [CPU model/data prepare run 37212856741](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37212856741) completed and its actual 23 pinned model files were audited. A separate fresh, unauthenticated student data download verified every archive and all 1,513 declared unpacked files. Both receipts remain separate from model inference and capability grading.

The [training run 37213067566](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37213067566) completed 2,077 optimizer updates, 4,154 row visits and 78,872 supervised target tokens. Every one of the 2,077 training rows was visited twice. Both archived adapters and their private HF receipt were audited; the two hashes remain in [training evidence](../evidence/v4-runtime/train-37213067566/training-audit-summary.json). Loss on different training batches is not heldout accuracy.

The [original validation run 37215395335](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37215395335) completed 396 LM generations and one shared pass over 16 actual ASR recordings. Four actual fresh reviewers completed its 303 blinded semantic judgments, viewing all 28 source photographs. Both adapters failed the frozen completion/nonregression gates. Its raw results, original selection and reviewer records remain unchanged in [original blind-review evidence](../evidence/v4-runtime/validation-37215395335/blind-review/).

A separate, validation-informed follow-up was frozen before its own execution: [experiment plan](experiment-plan-lower-lr.json) and [protocol](validation-protocol-lower-lr.json). It changes only the fresh training learning rate from 0.0001 to 0.00003. It is not described as preregistered before the original validation.

The [lower-learning-rate training run 37217452291](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37217452291) completed the same 2,077 updates, 4,154 visits and 78,872 supervised target tokens, with identical fresh LoRA initialization and row order. It did not resume an earlier checkpoint. Both actual archived weights and the private HF backup at commit `296576ce7de20efa8b35bdb2817e570ee141f35e` were independently checked against their byte counts and SHA-256 hashes; see [training audit](../evidence/v4-runtime/train-37217452291/training-audit-summary.json).

The [lower-learning-rate validation run 37219466611](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37219466611) completed 396 LM generations and 16 shared ASR transcriptions. The base answers and all ASR transcriptions are byte-equivalent to the original validation records. Four new actual fresh reviewers completed all 303 blinded semantic cases and viewed all 28 full source photos. Both archives again failed completion and voice-chat nonregression gates; their primary scores also fell below base. These owner judgments, exact scores and pre-test closure are preserved in [follow-up blind-review evidence](../evidence/v4-runtime/validation-37219466611/blind-review/).

The final active [selection](selection.json) is **base**, SHA-256 `8570755345e08c55ab77c165b405f41f6e5012b4c37a93909472b7f17334c42f`. It was committed and pushed at Git `1e71b8abffb34eccd297747d39827135a627107a` before the sole selected-only final test. This choice retains Qwen3-VL-2B-Instruct with the already selected Whisper-large-v3-turbo. It is not a claim that base answers are all correct or complete.

The [selected-only final test 37221188153](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37221188153) was dispatched once at 2026-10-04 17:36:59 UTC against that exact Git source, with all adapter/checkpoint arguments empty. It is currently running. Its expected scope is 178 LM generations and 22 ASR recordings, followed by 147 fresh independent manual grades. These are requested counts, not recorded completed results. No final scores or public v4 release exist yet.

The last fully audited durable reservation before final test was US$33.83/40.00. Final-test preflight permits at most another US$1.79; the actual live ledger must be retrieved after completion. These are conservative reservations, not an invoice. No reset occurred.

The six actual independent whole-book scans cover all 279 pre-existing numbered sections; their preserved records are [public review evidence](../evidence/v4-research/wholebook-independent-scan/). They do not replace fresh ordered review of changed text or the new chapter. Seven corrected conceptual sections have completed fresh readability checks; remaining cleanup/new-chapter readability work is ongoing. Do not claim the factual/continuity/reading-time gates have passed yet.

## Continue

1. Finish the sole selected-only test, verify actual source/output/HF/budget receipts, and export its 147 semantic cases with the unchanged committed helper and follow-up protocol.
2. Assign four new independent blind graders, preserve every actual judgment/source-image inspection and owner independence attestation, then apply the fixed test scorer. Do not change the committed pre-test selection or tune on test results.
3. Independently review a concrete base-only release approval and model card, publish the selected configuration, construct the actual pinned public manifest, and verify anonymous student download and actual browser use.
4. Fill the canonical book with only current measured results. Keep useful explanations, remove incidental failure histories from student prose.
5. Run fresh readability, then factual correctness, then continuity review; fix and recheck each finding. Update reading times only afterward, rebuild the whole course, commit/push and verify Pages.
