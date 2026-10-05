# Independent 10.8 inspection receipt

Reviewer: `/root/phase4_factual_coordinator/factual_10_8`; access date: 2026-10-05.
This reviewer read the current 10.8 section, not a previous review of this section.

## Current inputs personally inspected

- `course/chapters/10.md#10.8`, raw lines 250–283, complete section and original fence.
- Necessary prerequisite prose and code: 10.6, 10.7, 4.6, and 16.8. These were read from the current chapter bytes. Their empirical results were not reverified and do not supply evidence for 10.8.
- Complete `tiny_perceptron/model.py`, `multimodal.py`, `data.py`, `attention.py`, and `modern.py`, plus `scripts/build_course.py` lines 1–115, `pyproject.toml`, factual-reviewer instructions, both clear-tutorial skill documents, and the complete section extraction, review-round and technical-review checker implementations.
- `inputs/manifest.json` records the full original file hashes and precisely extracted section hashes. The source bytes were not normalized.
- 10.8 has no referenced figure or existing result JSON. No original result JSON pointer was read, and no empirical model score is supplied as evidence here.

## Original authoritative source inspection

The versioned PyTorch documentation URLs returned HTTP 403. `sources/fetch-receipts.json` preserves those failures. The successful fallback consists of original files under the official `pytorch/pytorch` repository at commit `5c4886908584029761b579af026dcfb627c84070`, identical to this wheel's `torch.version.git_version`. `installed-source-comparison.json` compares the actual bytes with the installed wheel. The HTTPS URLs, original bytes and their full hashes are retained; no locator index was used as proof.

- `module-official.py` lines 1976–2038: assignment of a child module stores that module object in `_modules`; lines 2894–2945: `train(False)` recurses, `eval()` delegates to it, only particular module classes alter behavior, and `requires_grad_` serves freezing/fine-tuning. This supports reference identity and evaluation-mode semantics, not a general promise that every random operation is disabled or that gradients are disabled.
- `torch-docstrings.py` lines 4238–4282: `torch.equal` tests equal sizes and elements; NaNs never compare equal, and dtype itself is not distinguished. Lines 12472–12512: `unsqueeze` inserts a singleton dimension and shares data. Lines 14490–14521 were also inspected for the generator seed documentation; the actual `torch.manual_seed` API is separately checked below.
- `random-official.py` lines 28–72: `torch.manual_seed` seeds random number generation on devices and delegates to the seed implementation. This supports the example's initialization control, not cross-version or cross-device reproducibility.
- `sgd-official.py` lines 152–234, 325–387: original SGD formula and parameter update `param.add_(grad, alpha=-lr)`. This supports the possibility of changed text computations when language parameters are updated. No optimizer is invoked in this review.

## Actual execution and limits

The helper extracted and ran the original fence in an offline CPU process with exit 0 and no guard events. Its original code, bootstrap, command, stdout, stderr and environment are in `original-helper/`.

The exact fence was additionally executed without the course bootstrap in the repository's `.venv`; the original code itself contains all required imports and inputs. The two successful commands and their exit codes, stdout/stderr hashes, time limits and offline variables are in `execution-receipt.json`. Full logs are retained. `variants.json` gives Python/PyTorch versions, device, shapes, dtype and counts.

The four local forward pairs use seeds 0 and 1 with `[1,20,30]` and `[1,40,50]`. All have shape `[1,3,264]`, 792 finite float32 scores, exact equality, maximum absolute difference 0, the same language and parameter objects, and unchanged language state. Image marker 5, audio marker 6, and both markers with no input each raise the expected `ValueError`. One scalar-output derivative pair checks all 17 language parameter tensors with exact equal gradients; no parameter update occurs. The default language path has no Dropout or BatchNorm layers. `eval()` is a defensive mode choice here rather than a necessary fix for an observed stochastic layer.

Exact equality is appropriate to these identical CPU computations. This is not the SDPA-versus-manual backend comparison in 16.8, where numerical tolerance has a different role. No wrapping claim is extended to an already batched tensor, labels, arbitrary user modules or other keyword inputs. The tested wrapper input is one unbatched text sequence with no modal marker, image or waveform.

No optimizer step, training, existing model evaluation, generation quality benchmark, model/data download, GPU/paid computation, or neural `.pt` artifact occurred. A fixed text benchmark for skill retention remains a future evaluation recommendation, not a result of this section. Visual rendering is not applicable to 10.8 because there is no image or spatial diagram to inspect; the two entry paths and all axes are explicit in prose and code.

## Search-scope incident record

The first file-name-only `rg --files` lookup used repository-wide filename globs and listed other review artifact paths. No body, verdict, author summary, or another review's execution evidence was read. The event was reported to the coordinator; all later content inspection used exact current-source paths. It does not constitute a clean-room filesystem isolation claim. No stop-triggering previous verdict or author correction summary was encountered.
