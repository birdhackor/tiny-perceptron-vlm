# Natural-input v4 execution handoff

This file records engineering state, not a student lesson or a completed release.

## Frozen and verified

- Data manifest: `docs/natural-assistant/v4/manifest.json`, SHA-256 `0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60`. Three data archives total 146,543,309 compressed bytes; 1,513 declared files total 160,683,823 unpacked bytes.
- 2,371 text/image records: 2,077 train, 124 validation, 170 test. There are also 38 heldout audio recordings: 16 validation and 22 test. The CPU encoding audit passed all rows, longest full sequence 630 tokens, 39,436 supervised training tokens. Current source-status binding is preserved as an append, rather than changing earlier audit bytes.
- Legal-source data: Google DOCCI CC BY 4.0; NVIDIA OCR CC BY 4.0; individually approved Commons CC BY/CC0/public-domain photos; OpenAssistant Apache 2.0; FLEURS CC BY 4.0; AISHELL-1 Apache 2.0. Excluded NC/SA preview collections are absent from the snapshots.
- Accepted DOCCI photographs: 439 train, 28 validation and 42 test. All added activity and heldout photographs received actual visual author and independent AI peer review. One rejected activity candidate remains excluded from training.
- OCR: 710 single-line plus 117 multiline NVIDIA crops, 62 approved Commons crops, whole-image region questions and balanced text-presence training. A genuinely fresh public-source reconstruction in this cloud workspace verified all 978 OCR artifacts from 89 photos and 153 selected NVIDIA pages; NVIDIA range transfer was 69,982,238 bytes, below 128 MiB, with no HTTP429. This is distinct from the GitHub runner replay that was interrupted by Wikimedia HTTP429.
- ASR was selected on the same 16 validation recordings before test: small normalized CER 117/510, turbo 50/510; raw CER also improved. Both LM candidates must use the same turbo transcripts. No ASR training or new test inference occurred.
- Validation protocol is committed before LM outputs. It compares base and archives at updates 1,039 and 2,077, with blinded semantic grades, exact row normalization, raw EOS completion guards, integer macro scoring and explicit nonregression gates. The scorer has export/score commands and generates an immutable pre-test selection.
- Actual prepare/train/validation execution source is `9a61ecf524c9518f33f1501c28aa72997d4a82d0`; the feature branch also contains subsequent independent final-test scorer fixes. Its three archive paths contain exact 133-byte Git LFS pointer blobs matching the frozen manifest. Native Actions upload and independent LFS download verified all three objects; the exact receipt is preserved in [data-publication evidence](../evidence/v4-research/data-publication/lfs-transfer-receipt.json).
- Canonical chapter 20 has 13 coherent sections, four additional SVGs, three v4 operation guides and migrated links. Ten CPU examples and static links passed. Real model results and release pins remain author comments; formal fresh review/reading times have not started.
- Optional research-proof base and supplement are packaged and verified twice; small indices/notices are committed. Raw proof is outside ordinary Git. Publication has not been verified.

## Current execution state

Data publication completed in [Actions run 37212727035](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37212727035). The native uploader registered the three frozen objects, and independent LFS downloads matched their sizes and SHA-256 hashes: 146,543,309 bytes in total. The temporary `natural-v4-lfs-byte-transfer` ref is absent remotely and locally. The ordinary feature tree contains none of the temporary transfer paths. Exact-copy, committed-blob and read-only ref checks are recorded in [publication-closure.json](../evidence/v4-research/data-publication/publication-closure.json).

The two CPU source-bootstrap attempts, `37211774915` and `37212355816`, failed and retained US$0.62 each. Their final cumulative reservation was US$26.05 before the subsequent model/data prepare attempt. These are conservative ledger reservations, not an invoice or measured billing total. The shared authorization remains US$40; no ledger reset occurred.

The [CPU model/data prepare run 37212856741](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37212856741) completed and its actual 23 pinned model files were audited. A separate fresh, unauthenticated student data download verified every archive and all 1,513 declared unpacked files. Both receipts remain separate from model inference and capability grading.

The [training run 37213067566](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37213067566) completed 2,077 optimizer updates, 4,154 row visits and 78,872 supervised target tokens. Every one of the 2,077 training rows was visited twice. Both archived adapters and their private HF receipt were audited; the two hashes remain in [training evidence](../evidence/v4-runtime/train-37213067566/training-audit-summary.json). Loss on different training batches is not heldout accuracy.

The [validation run 37215395335](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37215395335) completed 396 LM generations and one shared pass over 16 actual ASR recordings. The original frozen source files, source/weight hashes, raw outputs and exact counts were audited. Decoder completion was 126/132 for base and 129/132 for each adapter. Actual truncated answers remain failures under the original protocol. Four fresh independent reviewers are now grading the 303 blinded semantic cases against actual source images and the frozen rubrics. No chosen version, final test, or public release is recorded here yet.

The latest durable cumulative reservation is US$30.25/40.00 after validation. It is a conservative reservation ledger, not the invoice. No reset occurred.

## Continue

1. Complete the four actual independent blind grading packets without disclosing candidate mapping; preserve every judgment, source-image inspection and owner identity.
2. Apply the unchanged protocol, preserve its original selection evidence, and commit the selected exact weights before opening the new test. Any further training experiment would require a separate pre-run plan and budget reservation; neither frozen gold nor the existing protocol may be rewritten to accommodate outputs.
3. Run the selected-only new test, obtain fresh semantic grades, publish the selected configuration and actually verify anonymous student download and browser use.
4. Fill the canonical book with only current measured results. Keep useful explanations, remove incidental failure histories from student prose.
5. Run fresh readability, then factual correctness, then continuity review; fix and recheck each finding. Update reading times only afterward, rebuild the whole course, commit/push and verify Pages.
