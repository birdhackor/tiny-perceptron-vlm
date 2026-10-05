# T.2 independent factual inspection, 2026-10-05

Reviewer task: `/root/phase4_factual_coordinator/factual_t_2`; this reviewer was newly assigned T.2, has not authored it, has not performed its reader review, and did not read earlier technical/reader/history/dispatch conclusions or author correction worknotes. No child agents were created. File-name-only discovery initially returned many review-artifact paths; their contents were not opened.

The instructions personally read were `docs/review-tools/factual-reviewer-instructions.md`, all of `scripts/check_technical_reviews.py`, all of `docs/review-tools/section_facts.py`, `.agents/skills/clear-tutorial/SKILL.md`, and its `references/review-protocol.md`. The cloud onboarding skill and its onboarding reference were read to check the existing environment. No installation, service or configuration change was necessary.

## Actual input and code reading

- `course/training.md`: the complete raw T.2 section, initial introduction and T.1 necessary context, and the opening of T.3 exposed by the exact line-range read (lines 1–70). The T.2 raw SHA is `ca8dd78a47909aeac6a194e0c2ae6e7a81b2f72aeb2b35df6f6d08ff19de23b2`. The whole-file frozen SHA `22f3115b0d09a83d82342e82c101df8b8516aa5de7796b137a5243c213396324` refers to the true preserved whole Markdown bytes at first read, not a future or current whole-file assertion. Introduction summary: choose a finite task, prepare its material, check the numerical path, then update and assess held-out questions; commands run at the project root and short exercises may use CPU.
- `course/chapters/05.md`: lines 1–100, including all of 5.1 and the accidentally broader contiguous read into 5.2/5.3. This is the current textbook, whose claims were treated as input, not proof. The T.2 link to 5.1 points to an actual parameter-specific embedding-gradient check and separately explains an optimizer update.
- `course/chapters/10.md`: all of 10.7 through the next heading; this current-textbook cross-reference describes shifted answer positions. No claim in those other sections was reviewed or endorsed by this T.2 report.
- `scripts/train.py`: AST located `parser` (37–71), `prepare_examples` (74–113), `modal_loss` (116–146), and `main` (188–393); the corresponding exact ranges and final entry point (396–397) were personally read. Constructor/import contracts were read through the AST import list. No result-commentary constant was encountered. Dry-run line 305 selects one iteration; lines 353–356 compute backward and aggregate norm and guard step; lines 360–392 guard checkpoint/weight/report saves. Default text and sft use generated examples; vision uses a generated shape image and asks `shape?`.
- `tiny_perceptron/data.py`: AST located and read `ByteTokenizer` (14–28), `shifted`/`render_chat`/`pad_batch` (46–86), and toy material (112–122). Text labels are the next positions; assistant content and its end marker are scored in sft; -100 and padding are excluded from scored positions.
- `tiny_perceptron/model.py`: `TinyLM` (53–89), `loss_sum`/`masked_loss` (92–105); score shape is batch × position × vocabulary and the loss denominator is the number of labels unequal to -100, with empty targets rejected.
- `tiny_perceptron/multimodal.py`: `VisionEncoder` (30–41), `expand_modalities`/`MultiModalLM` (84–135), and `scene` (177–190). The image features remain in the causal context and do not directly receive text labels; answer labels are aligned after expansion. Unused audio components may lack gradients in vision.
- `tiny_perceptron/training.py`: `choose_device` and `seed_everything` (13–27). The explicit cpu argument resolves to `torch.device('cpu')`.
- `pyproject.toml`: read only for the existing interpreter/dependency workflow. This report does not review its package/version guidance.

The permanent frozen-input files were copied byte for byte and hashed personally. They are snapshots rather than altered source. No ignored training data/cache or weights were required.

## Personally inspected original authority

The installed torch version is 2.14.1+cpu, git commit `5c4886908584029761b579af026dcfb627c84070`. HTTPS retrieval of the official PyTorch GitHub source at that exact commit succeeded. A probe for the 2.11 hosted API page returned HTTP 403, so this report uses the exact-commit original source instead. A second successful v2.11 raw source probe was only a retrieval candidate; it is not cited as evidence.

`fetch_official_sources.py` retained the exact relevant function bytes and hashes the full HTTP responses for provenance. The first AST selector for `Optimizer.step` selected an overload stub; the selector was corrected to the implementation and retrieval rerun before any substantive source judgment. Only the final implementation snippet supports the claim.

- Original `torch/_tensor.py`, `Tensor.backward`, lines 566–625: computes gradients by the chain rule, accumulates them in leaves, and differentiates only the leaves used by the current calculation.
- Original `torch/optim/optimizer.py`, actual `Optimizer.step` implementation and its docstring (see official manifest): explicitly performs an optimization step to update parameters.
- Original `torch/nn/utils/clip_grad.py`, `_get_total_norm` lines 48–116 and `clip_grad_norm_` lines 184–232: total norm is that of the available gradients as a single vector; parameters with `grad is None` are excluded; gradients are changed in place and the original total norm is returned. The helper computes the default L2 total as sqrt(sum of squares), with no sample/parameter-count denominator.

## Actual CPU inspection and interpretation

`verify_dry_runs.py` launches each of the three original command argument lists with the `.venv` interpreter. Each completes with exit 0, one progress JSON record and one final report. The final root fields are `mode`, `task`, and `device`; the diagnostic `loss`, `grad_norm`, `effective_tokens` occur in `history[0]` and the progress record. The textbook table does not claim these are root-level fields.

The bounded observation pass executes the same unmodified script again, observes labels and aggregate gradients, compares every captured parameter tensor bytewise, and fails if any optimizer step or weight-saving function is called. All parameter tensors stayed equal, all three step/save counters are zero, and the diagnostic records match the direct-run records exactly. Text has 111 scored positions and sft has 8. Vision has four separately averaged examples with 7 scored answer positions each (28 in total), while its interface reports null because the branch does not publish that count. This is not an additional model score or retraining result.

The pre-clipping aggregate gradient norms match an independent float64 sqrt(sum of squares) calculation within relative/absolute 2e-6. Vision has 26 parameter tensors with gradients and 8 without, while its total norm is positive. A two-scalar variation confirms the same logic: `(p-1)^2` at p=2 gives loss 1 and gradient 2, leaves p unchanged after backward/gradient clipping, and leaves an unused parameter without gradient.

There is no embedded numerical worked example or model-performance result to recompute. The numeric check here verifies the sign/finiteness criterion and aggregate-norm interpretation. There are no Python fences or referenced figures in T.2; no figure render/view is applicable. The batch flow is sufficiently specified by the command choices and report table for this factual check; no missing material is needed to establish the stated interface behavior.

These observations establish that these default diagnostic paths can compute a finite loss and nonzero aggregate gradient without updating weights. They do not establish learned task ability, every-parameter gradient coverage, generalization, full-train behavior, other CLI options, GPU readiness or capability of a finished product. No unresolved substantive claim was found in T.2.
