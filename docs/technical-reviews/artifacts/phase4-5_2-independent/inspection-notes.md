# Independent inspection of 5.2, 2026-10-05

Reviewer: `/root/phase4_factual_coordinator/factual_5_2`. I did not read prior technical or reader review conclusions. This is an independent factual review of the current 5.2, not a reading-order review or a review of adjacent lessons.

## Current repository material actually read

- `course/chapters/05.md`, raw section 5.2, source lines 50–82. One Python fence; no SVG, raster image, or other cited figure. The axes, labels, and numeric inputs are explicit in the prose and code. There is no cited figure to render or view. No desktop/mobile page-render claim is made.
- `tiny_perceptron/model.py`, especially `loss_sum` lines 92–100 and `masked_loss` lines 103–105. The helper counts `(labels != IGNORE).sum()`, rejects zero valid targets, flattens only sample/position axes to an `[8,5]` score matrix, passes `ignore_index=IGNORE, reduction="sum"`, and returns total and count separately. This helper performs no model forward and no optimizer update.
- `tiny_perceptron/data.py`, lines 1–30; `IGNORE=-100` at line 10. The defined toy candidate list in 5.2 is not the ByteTokenizer's complete vocabulary.
- The review instructions, extractor, checker schema, review protocol, and bootstrap in `scripts/build_course.py` were inspected before execution. The extraction helper is limited to ignored outputs or `/tmp`; its exact results were copied into the permanent docs directory without relying on temporary tensor/weight outputs.
- Necessary prerequisite: 1.8 defines `log` as the natural logarithm and the score as `-log(p)`. This is a prerequisite read, not a verdict about 1.8. The five-character list in 1.1 was confirmed in current source lines 23–39 as `['。','狗','看','貓','，']`. The chapter introduction was not read or endorsed. An adjacent source range was displayed while locating the list; no claim about that adjacent lesson is made.
- The linked `course/training.md` initial workflow and recipe contracts were read without executing any recipe. `scripts/train.py` lines 304–365 and `scripts/course_experiments/common.py` lines 105–144 were inspected to distinguish text-token averaging, forward/backward, and the guarded optimizer step. These reads do not endorse other task-specific loss rules. In 5.2, “本課選後者” is supported within this section's explicitly described two-short-text/label example and `loss_sum`/`masked_loss` text contract; it is not an assertion that every other training task in the repository uses the same objective.

## Official sources personally read

All successful official snapshots use the exact installed PyTorch 2.14.1+cpu git commit `5c4886908584029761b579af026dcfb627c84070`, retrieved from the PyTorch organization's original GitHub repository over HTTPS. Fetch receipts preserve URLs and raw SHA-256. I read the following original ranges, not another review's summaries:

- `torch/utils/data/dataloader.py`, lines 153–176: batching/collation, `batch_size` as samples per batch, and collation merging samples into a mini-batch.
- `torch/nn/modules/loss.py`, `CrossEntropyLoss` lines 1200–1242 and 1272–1310: unnormalized class logits, the negative natural logarithm of softmax target probability, ignored-target indicator, sum and weighted mean reductions, input/target axes, and no input-gradient contribution for `ignore_index`.
- `torch/nn/functional.py`, `softmax` lines 2176–2217 and `cross_entropy` lines 3478–3563: normalization along `dim`; default `ignore_index=-100`; class-target shapes; `reduction="sum"`; call into the core cross-entropy implementation. This verifies the helper's class-last flattening contract, rather than incorrectly treating `[B,T,C]` as the raw API's `[B,C,T]` input.
- `docs/source/notes/autograd.md`, lines 12–34 and 194–229: recording the executed operations, reverse chain rule, graph construction, and `requires_grad=True` leaf-gradient accumulation. An initial `.rst` URL returned 404; the corresponding `.md` file at the same immutable commit was then retrieved successfully and personally read. The failed URL is not claimed as supporting evidence.
- `torch/_tensor.py`, `Tensor.backward` lines 566–627: differentiates the current scalar with respect to graph leaves, uses the chain rule, and accumulates gradients into leaves. It is not an optimizer step.
- `torch/_tensor_docs.py`, `item` lines 2788–2804 and `requires_grad_` lines 4125–4142: a one-element tensor converts to a standard Python number; the conversion is not differentiable; gradient recording is controlled by the flag. The original fence only prints `.item()` results and calls backward on `mean_loss`, preserving its computation graph.
- `torch/optim/sgd.py`, `SGD.step` lines 105–152, algorithm lines 156–184, and in-place parameter update lines 372–379: one step consumes the available gradient; updates alter the parameter point used by later forwards.

## Scope of actual execution

The raw fence was executed by the extractor's bounded offline CPU worker (exit 0, 30-second limit, no guard events) and again by my inspection probe. The exercise was executed as that same float32 fence with only its second-row second label changed to 2, then restored to -100. I separately used float64 synthetic tensors to verify precise loss values, sample/token weights, ignored-position gradients, and gradient accumulation at a fixed parameter point.

The optimizer comparison uses only a shared vector of five logits and a few SGD steps. One full-batch backward plus one step and two globally normalized micro-batch backwards plus one step agree. Intermediate two-sample/five-answer steps use the same global denominator but recompute at changed logits, so they differ. This checks arithmetic and forward/backward/step boundaries, not model quality or an optimal batch size. No training data/model was downloaded, no tutorial model was retrained, and no GPU, upload, or paid compute was used. There are no empirical model-quality claims in 5.2 to audit.

No unresolved substantive factual issue was found in this scoped section. The all-zero logits make the two averaging rules have the same scalar loss, but the section says their weights differ, which the gradient check confirms. The extra nonuniform-logit check verifies that differing weights can also change the scalar loss; its numbers are evidence only and are not proposed as new tutorial scores.
