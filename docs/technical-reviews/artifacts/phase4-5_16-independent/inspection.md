# 5.16 fresh independent technical inspection

Reviewer: `/root/phase4_factual_coordinator/factual_5_16`. Access date: 2026-10-05.
No previous reader/technical report contents were read. No child agent was used.

## Scope personally read

- `docs/review-tools/factual-reviewer-instructions.md`, entire methodology.
- `scripts/check_technical_reviews.py`, complete schema/checking contract.
- `docs/review-tools/section_facts.py`, entire extractor/guard/worker contract.
- `.agents/skills/clear-tutorial/references/review-protocol.md`, entire protocol.
- Current 5.16 original raw UTF-8 section; nearby 5.13–5.15 and start of 5.17 for context; referenced 7.16 for the original empirical report location. This is not a chapter-opening section.
- `tiny_perceptron/data.py` lines 8–28, 54–86: `IGNORE=-100`, byte tokenization, assistant content plus EOS target, single shift, question/system/role/padding ignore behavior. `pad_batch` also rejects a record left without valid labels after cropping.
- `tiny_perceptron/model.py` lines 92–105: sum unweighted class-index CE over all nonignored targets and divide by exactly their count.
- `scripts/course_experiments/common.py` lines 45–97, 105–203: SFT records stay intact, overlong SFT input is rejected rather than silently cropped; `fit_lm` default batch size 16, separate `random.Random(seed)` with replacement, one mean loss per sampled batch, cumulative valid-label count.
- `scripts/course_experiments/text.py` lines 35–66, 556–576, 597–663: original saving/hash convention, base max length 128, arithmetic/replay branch records, 500-step scale, identical base. No long recipe was executed.
- Original `docs/course-experiments/results/sft_ablation.json`: run identity, data manifests, b-only/replay training counts, source record counts, reported code hashes and scope. All four relevant current implementation hashes exactly match the recorded experiment hashes.
- Existing original `outputs/text-behavior-interface-check/sft_ablation/{attributes,arithmetic}/train.jsonl`; copied only these 15.6 KB of records, never regenerated. Their raw file hashes exactly match the original report data manifests, and combined structured record hashes match both training runs.

## Official primary source inspection

Downloaded pinned official raw text over HTTPS with normal certificate checks. Personally read:

- PyTorch v2.14.1 `torch/nn/modules/loss.py` lines 1200–1252 (class-index CE formula, weighted nonignored mean denominator) and 1275–1300 (ignore_index/reduction contracts). In this repository no class weights are passed, so the denominator is the unweighted number of valid target positions.
- PyTorch v2.14.1 `torch/nn/functional.py` lines 3478–3572 (cross_entropy default ignore_index=-100, integer-target masking, class weighting, reduction dispatch). This version matches installed 2.14.1+cpu API family. Source weighting is a custom per-source factor illustrated separately; it is not the built-in class-weight argument.
- CPython v3.13.5 `Lib/random.py` lines 458–492: choices draws with replacement; absent weights each population entry has equal probability; implementation is floor(random()*n) for each draw. This is the same primitive used in original `fit_lm` and matches installed Python 3.13.5.

These sources support averaging/masking and sampling contracts. They do not establish a universal optimum data mixture or a guarantee of model quality.

## Actual execution and its limits

Original Python fence was executed by the supplied CPU helper under its offline/artifact-only guard, exit 0, no guard events. The independent `verify_cpu.py` then executed the identical fence and its two literal-only exercises; checked exact fractions and float64 CE on 2,200 synthetic padded cells with 290 valid labels; changed ignored logits; and inspected a 12-cell crop. All assertions passed.

The position-share counterexample has long-position share 0.689655 but long summed NLL share 0.117087: actual loss contributions depend on the error at each valid position. The text explicitly makes that distinction. Explicit custom source factors 1 and 1/20 change weighted masses to 90 and 10 while leaving all 290 valid positions intact; no source-weighted training is claimed.

For stipulated fixed lengths 1 and 20, 180:1 sampled record counts yield cumulative token totals 180:20, hence 90:10. This is an algebraic global/cumulative condition. Finite random batches need not have that ratio: with batch 16, iid long-answer draw probability 1/181 gives P(no long answer)=0.915172. The expected per-batch fraction of long valid positions is 0.049069, while the ratio of expected cumulative long tokens to all tokens is 0.1. Independent per-batch mean normalization therefore is not interchangeable with a single global token mean. The section claims the latter only under its explicit “all 290 positions summed, divided by 290” assumption, mentions batch feasibility, and recommends cumulative accounting. It does not assert fixed per-batch composition or equality of gradient influence. Preserve these qualifications in any later edit.

The original 500×16 sample sequence was replayed solely to count existing labels: no model object, forward pass, gradients, optimizer, checkpoint/weight read or training. Both branches sample 8,000 records; b-only uses 49 distinct available input records, replay 94 (45 A+49 B). Counts reproduce 18,453 and 36,384 exactly. Replay draws A 3,813 and B 4,187; effective labels include the assistant EOS. The ratio is 1.9717119, correctly described as nearly double. This checks the original GPU evidence, not a new GPU result, retention score, or causal equal-token experiment.

One launcher attempt incorrectly resolved the venv executable symlink to `/usr/bin/python3.13` and lacked torch; that attempt is retained under `launcher-path-failure-*`. The corrected command invoked `.venv/bin/python` without resolving its symlink, completed in 3.05 seconds, and is the actual verification evidence. The error concerned the review launcher, not the lesson.

## Figures and environment

The section has no image, SVG, HTML image reference or diagram. No figure was rendered or viewed; figure consistency is legitimately not applicable. Counts and tables of scalar ratios are fully represented by text and code, so no missing diagram is needed for the checked technical claim. No desktop/mobile page rendering was performed or claimed.

The required cloud-environment-onboarding:setup skill and its onboarding reference were read. Existing `.venv` Python 3.13.5 / PyTorch 2.14.1+cpu successfully executed the selected CPU review workflow; no installation, repository configuration change or environment draft save was necessary. No GPU, model/data download, upload or paid compute was used. Official-source text retrieval is the sole network operation for this review.

Verdict: pass. No substantive unresolved technical issue identified. The finite-batch/global-mean limits above are evidence boundaries, not claims that the small example proves model quality or realizes the long training recipe.
