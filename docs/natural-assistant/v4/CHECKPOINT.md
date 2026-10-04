# Natural-input v4 preparation checkpoint

This is an engineering handoff, not a published lesson or a claim of finished training.

## Completed

- Legal-source research: Google DOCCI CC BY 4.0, NVIDIA multilingual OCR CC BY 4.0, individually licensed Commons photos (CC BY / CC0 / public domain), OpenAssistant Apache 2.0, FLEURS CC BY 4.0 and AISHELL-1 Apache 2.0. Excluded NC/SA or unclear-rights image collections are not training inputs.
- Real photograph training: 439 photos, 878 Traditional-Chinese scene/fact QA. Every author viewed the actual photos and read full descriptions. All 39 added human-activity photos received a separate actual-visual peer review; 11 answers were corrected before inference.
- OCR: 710 single-line and 117 multi-line NVIDIA crops, 62 approved Commons crops, whole-image region questions and balanced Chinese-presence examples. Ambiguous or clipped crops were removed. Fresh local-source reconstruction verified all 978 declared OCR artifacts byte-for-byte; the receipt explicitly records zero fresh HTTP requests.
- Six additional natural multi-line heldout tasks passed a second whole-image/detail visual review. Their score measures ordered transcription, not isolated reading-order ability.
- Chat: 144 records (122 train / 9 validation / 13 test), independently read in full; source tree grouping, source replay, peer corrections and question-specific heldout rubrics preserved.
- Speech: 30 fresh FLEURS transcription recordings and eight AISHELL human read questions (four validation / four test). These are evaluation inputs; no ASR fine-tuning occurred.
- Real CPU ASR validation: same 16 recordings, small normalized CER 117/510 (22.94%), turbo 50/510 (9.80%). Raw CER also improved. Turbo was selected before test; final base/adapter comparisons must use the same ASR. See `asr-selection.json` and its immutable evidence hashes.
- CPU processor audit: 2,341 text/image rows plus 38 audio files passed; longest full sequence 630 tokens, 39,436 current-assistant supervised train tokens. This is stable provisional data, not the final freeze.
- Runtime supports bounded fresh LoRA training, two immutable update checkpoints, validation of base plus two adapters, selected-checkpoint release and separate transcription/chat evaluation.
- Download helper accepts complete fixed v3 or v4 archive sets, verifies the correct remote manifest path and all file hashes. Actual provisional archives were extracted and verified locally. Anonymous network download is not yet verified.
- Three provisional LFS archives are reproducible locally, total 145,577,586 compressed bytes. They are under `outputs/natural-v4/provisional-assets`, not uploaded or committed final snapshots.
- Chapter 20 was rewritten as 13 coherent draft sections, with four new SVGs and rewritten STUDENT/DATA/TRAINING drafts. Ten CPU teaching examples were actually run. Drafts are under `drafts/`; they do not replace the formal book yet.

## Transport blocker

Public GCS/HF downloads, GitHub API/git and Modal API return HTTP 503 with Envoy `cloudflare_https_tunnel` / `Invalid argument`. Chromium reproduced the same result. Direct connection reported network unreachable. No network policy, TLS checks or proxy configuration were changed. This observation does not establish the infrastructure root cause.

Latest anonymous probes are in `../evidence/v4-research/environment-network/` (relative to the parent natural-assistant directory). No v4 Modal/GPU job has started, no v4 weights were produced, no v4 LFS upload succeeded and no formal v4 book was published.

## Continue in order

1. Restore the environment's network. Retrieve the ten already-selected new activity heldout photos: four validation and six test, in the existing heldout similarity clusters. Actually view each, author the labels, then obtain independent visual review before any inference. The unused 40th activity training candidate stays excluded; 39 good training photos are sufficient.
2. Finish the source/gold freeze, delta processor audit and exact validation-selection rule. `experiment-plan.json` holds the intended fresh learning rate 1e-4, rank 8, q/v targets, accumulation 2 and seed 42. Current train count is 2,077: archive updates 1,039 and 2,077 correspond to 2,078 (approximately one pass) and 4,154 (two passes) visits. Confirm final counts before dispatch.
3. Build the final three archives and `manifest.json`. Commit them as native LFS pointers on the `natural-assistant-v4` branch. `.github/workflows/natural-data-v4.yml` reconstructs exact pinned sources and uploads matching LFS objects without a GPU. Verify a fresh anonymous student data download afterward.
4. Use the existing `natural-assistant.yml` workflow serially: prepare, train with both checkpoints, then one validation of base plus both adapters. Keep the cumulative durable budget: US$40 cap, last verified reservation US$24.81, US$15.19 remaining. Reservations are not invoices; never reset the ledger.
5. Grade predeclared semantic rubrics independently, select on validation and commit exact selection/weight hashes before the new paired final test. Preserve all actual completion/timeout counts. Test outputs must not be used to choose training settings or rewrite gold.
6. Release the exact selected checkpoint, verify student model download/UI, then integrate actual results into the coherent draft. Do not copy old v3 performance into v4 or attribute ASR replacement gains to LM fine-tuning.
7. Apply semantic crosslinks from `drafts/crosslink-migration.json`, generate notebooks and figures, then complete three ordered fresh review rounds: readability, factual/source accuracy, continuity. Recheck changed runtime-dependent sections 11.15 / 12.14 / 19.12 / R.2 / R.4 as well as all new Chapter 20 sections.
8. Add actual fresh reading-time estimates for changed pages, validate all source/figure fingerprints, run appropriate builds/CI, commit/push main, verify Pages and the real student route. Remove learning-irrelevant failure history from student prose; keep genuine technical evidence separately.

Do not blanket-add old untracked files. Some historical research previews were excluded from the permitted dataset. Preserve the current raw review evidence; large image data and image evidence need the LFS/release strategy before final publication.
