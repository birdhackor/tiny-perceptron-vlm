# Natural-input teaching release: verified state

This is an engineering record. Student instructions are in `STUDENT.md`, `DATA.md`, `TRAINING.md` and chapter 20. Original unsuccessful execution attempts remain in evidence packets, outside the teaching narrative.

## Data and actual training

The frozen manifest is `manifest.json`, SHA-256 `0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60`. It contains 2,371 text/image examples: 2,077 training, 124 validation and 170 test. The 38 heldout recordings are 16 validation and 22 test; they do not train Whisper. Three Git LFS archives contain 1,513 declared files, 146,543,309 compressed bytes and 160,683,823 unpacked bytes. Native publication and independent anonymous retrieval verified their exact bytes at public data revision `9a61ecf524c9518f33f1501c28aa72997d4a82d0`.

Training sources are Google DOCCI (CC BY 4.0), NVIDIA OCR (CC BY 4.0), individually approved Commons photos (CC BY 4.0, CC BY 3.0 or CC0), and OpenAssistant (Apache 2.0). FLEURS (CC BY 4.0) and AISHELL-1 (Apache 2.0) are heldout speech evaluation sources. Per-file provenance, source-family splits and derivative attributions are preserved. AI-authored Traditional Chinese labels were visually checked by separate AI reviewers; they are not official human gold labels.

Two real Modal L4 LoRA training runs each completed 2,077 optimizer updates and 4,154 visits to the same 2,077 training examples. The 1,605,632 trainable parameters are 112 language-attention q/v LoRA tensors. The second run changed only learning rate, started from fresh LoRA and was informed by the first validation; it did not resume the first weights. Original commands, outputs, runtime boundaries and intermediate private Hugging Face checkpoints remain preserved. The selected ASR is Whisper-large-v3-turbo, chosen from 16 validation recordings before the final test. No ASR fine-tuning occurred.

## Selected and actually tried release

The immutable pre-test selection is BASE, committed at `1e71b8abffb34eccd297747d39827135a627107a`. Neither LoRA candidate passed the delivery guards. The release uses Qwen3-VL-2B-Instruct at `89644892e4d85e24eaac8bacfd4f463576704203` and Whisper-large-v3-turbo at `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`: 2,936,410,112 unique parameters across two separate models and no adapter. This is a pretrained dense extension; chapter 19 separately teaches the integrated small MoE. No improvement from the rejected adapters is claimed for the selected product.

The sole selected-version final test ran 178 LM and 22 ASR cases. Its 147 semantic cases were graded by four genuine independent AI owners. Scores are summary 25/42, image facts 58/84, text presence 18/18, single-line OCR 8/10, ordered OCR 1/3, text chat 2/13, correct-transcript chat 2/4 and actual-ASR chat 2/4. Normalized ASR CER is 96/661. These are limited, related teaching cases, not independent population estimates. The unchanged pre-test selection was not tuned against final-test results.

Public release metadata is in `birdhackor/tiny-perceptron-course-models` at `d3954d6900b3cf81e593d99d9b8b1a91e6f9741d`. Upstream base weights are pinned, not copied into the Git repository. Student runtime Git revision is `59a1eda4ed7b6e8609892ec2b9013c821ac93e69`. Actual anonymous metadata retrieval and a fresh CPU plus Chromium browser trial exercised text, training-photo input, ASR, editable transcript, shared history, mobile layout and reset. Existing verified 5,889,111,977-byte model caches were reused: this was not a fresh full-weight download. Peak service RSS was 13,496,104 KiB (12.8709 GiB), excluding browser and driver; this is an observation, not a minimum RAM requirement. The trial was AI headless operation, not a human usability study.

## Book and verification

Current book source is frozen at `9e963cb0863c9f0139e02f00a4c9209741cec7f9`. The coherent rewrite includes thirteen chapter-20 sections, three operation guides and the surrounding prerequisites, training explanations and entry/reference pages. Genuine factual findings and continuity suggestions were corrected and reread by the original owners before closure.

Completed current readability covers 292 numbered reports: 58 fresh current owners and 234 unchanged baseline reports, plus eight whole-file owner groups covering thirteen supplemental files. Current factual review covers 69 fresh numbered owners and 223 unchanged baseline reports, plus eight independent whole-file owner groups covering thirteen files. Native factual and identity/introduction checks passed all 292. Seven fresh continuity owners cover all 292 numbered sections and 38 full files. Current closure packets preserve genuine FINALs, original failures and corrective rereads without changing original reviewer judgments.

All 266 notebooks passed actual serial independent CPU kernels in 728.82 seconds. Later changed lessons individually passed fresh kernels, and strict export checks all current notebook cells against actual executed outputs. This is not a second full 266-kernel run. The application passed 934 tests with one skipped; the skip reason is not asserted. Native book synchronization, reader/factual/identity checks, Ruff lint/format and the frozen dependency lock passed. Final reading-time and browser verification receipts accompany this release. Deployment status is recorded by the public `Publish course` Actions workflow for the publishing commit; local build success alone is not evidence of a live deployment.

The authorized cumulative ceiling is US$40. The conservative reservation ledger is US$36.20, not the actual provider invoice. No further GPU tuning or final-test-based candidate selection is part of this release.
