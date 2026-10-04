# Independent 9.2 factual derivation

Reviewer: `/root/v4_review_coordinator/factual_v4_9_2`, fresh context.

## Grounding is relative to the specified information

Let a world have hidden ball count H and observable information O. With no visible
count and no count rule, world A has H=3 and world B has H=4 but both have the same
O: an opaque box and an unanswered count question. A deterministic answer using
only O is identical in A and B. Answering 3 is correct in A but false in B.
Consequently O does not entail H=3; a lucky correct answer does not establish
grounding. This is a finite counterexample, not a statement about a model's private
beliefs, confidence calibration, or consciousness. If O explicitly includes the
trusted count 3, the admissible worlds in the toy task have H=3, so answering 3 is
supported. A sufficiently clear and complete view that permits counting can also
supply O; this lesson does not execute any vision model or guarantee recognition.

Add observed color red to both A and B. They remain indistinguishable unless a
count rule is supplied. If O additionally contains the trusted rule
`red => H=3`, both red-box worlds must satisfy H=3. Color only becomes decisive
under that stated task rule, rather than by changing an unrelated dictionary key.
This does not deny probabilistic color/count correlations in other tasks.

## Label leakage

In the lesson's rule T(v), T(3)='3' and T(None) is the missing-information sentence.
The rule reads v, rather than H, so changing H from 3 to 99 leaves T(None) unchanged.
If a dataset instead labels an input with no count information as answer '3', its
supervised negative-log-likelihood includes -log p('3'|O_missing). Decreasing that
loss raises the probability of '3' for missing-information inputs. This shows the
direction of the supervision signal; it does not prove any particular fitted
model will converge or learn a universal rule. Hidden H may remain available for
the dataset author's audit without entering the answering input or target.

## Exact code predictions before execution

Row 1: visible=3; (3 is not None)=True; choose str(3)='3'. `print` joins its four
arguments with one space and adds one newline: `可見 3 → 3`.

Row 2: visible=None; (None is not None)=False; choose the missing-information
literal. Prediction: `可見 None → 看不到球數，請提供數量或圖片。`.

The complete expected stdout is two lines, each ending with LF. The conditional
expression evaluates the condition first and only the selected branch. There is
no `row['hidden_count']` subscription. The actual executable probe installs a
dictionary subclass that raises if that key is read: all four executed rows only
read `visible_count`. With visible=0, `0 is not None` is True, so the answer is '0'.
With visible=None and an added red color field, the answer remains the literal.

The single code block defines examples, assigns strings, and prints. It neither
creates a learned model nor runs training. This is evidence of a human labeling
rule, and cannot be evidence of self-awareness or successful generalization.

## Why evaluation requires both groups

Use a toy evaluation with two answerable counts [0,3] and two missing-information
examples. Always saying '3' scores 1/2 on the answerable group and 0/2 for appropriate
clarification. Always asking for count scores 0/2 on the answerable group and 2/2
on the missing-information group. A clarification-frequency metric alone cannot
distinguish that second policy from a properly conditional one. A hidden-truth
accuracy metric can credit unsupported lucky guesses. These are hand calculations
for an invented transparent counterexample, not new measured model results.

Testing changed numbers and wording asks a different question from checking
training labels or string equality. The frozen model's same box-1 missing-count
input receives a clarification under the original wording but '6' after changing
only the question phrase. One observed failure suffices to refute an unrestricted
claim that this model always recognizes missing information under different
wordings. It does not prove that the model cannot generalize anywhere, or isolate
a specific internal memorization mechanism.

## Execution and historical limits

`probe.py` ran on Python 3.13.5 / torch 2.14.1+cpu with one CPU thread, seed 42,
FP32, greedy generation, 128 requested new tokens and the actual model context
limit 128. It used the preexisting 575558-byte public checkpoint after checking
its SHA against the official pinned manifest; no checkpoint was downloaded.
The model has width 64, two layers, 141568 parameters, manual attention, no RoPE,
and byte tokenization. There were zero optimizer steps or training updates.

The original recorded run used Python 3.13.3 / torch 2.14.1+cu126 on NVIDIA L4,
seed 42, 900 updates, batch size 16, AdamW LR 0.003, constant learning rate,
auxiliary coefficient 0.01, and gradient clipping 1.0. The mixed training set has
102 behavior examples plus 49 arithmetic examples. This task reconstructed the
eight disjoint rule families and exact train/validation/test JSONL SHAs, as well
as the selection denominators 394888 (behavior only) and 275389 (mixed) effective
supervised tokens. This replays random selection and token counting, not training.

The whole current behavior.py differs from the original run at the style-metric
function. The original source was actually retrieved at the recorded revision;
its raw SHA matches the run record. AST comparison and direct inspection confirm
`_conversation`, `_safety_records`, `_safety_evaluations`, and `run_safety` are
unchanged. The common, text, tokenizer, model and checkpoint-loader full file
SHAs also match the original run.

Frozen CPU inference regenerated every token of all 17 original-test samples and
all six paraphrase samples identically to the original record. Denominators are
17 examples / 466 supervised target tokens, and six examples / 117 target tokens.
Original missing-count subgroup: three appropriate clarifications out of three;
paraphrased missing-count subgroup: zero exact target matches out of three.
Full-test matching is 16/17; paraphrase matching is 0/6; all six paraphrases end
with EOS. Parameter-bytes SHA before and after inference is identical. CPU NLL
roundoff need not equal GPU NLL; exact output token equality is the claim checked.

The record's 22.208399321 seconds covers training, evaluation, and local checkpoint
saves, excluding image build, startup, and uploads. This task did not benchmark
that time or GPU memory, and 9.2 makes no speed or memory claim. The results
support one fixed checkpoint, one seed, eight rule families, and two mild kinds
of wording changes. They do not establish arbitrary natural-language, multimodal,
or multi-turn information-sufficiency judgments.

## Figures and introduction

The entire raw 9.2 section and the fully read necessary 9.1, 8.6, and 9.8 sections
have no SVG references. The complete applicable SVG map is `{}`. No renderer or
image viewer was run, because no applicable figure exists. Assigned section 9.2
is not the first numbered section, so a chapter-introduction SHA is inapplicable.
The exact source and prerequisite bytes are preserved in the raw snapshots and
`reading-manifest.json`; no whitespace normalization was performed.
