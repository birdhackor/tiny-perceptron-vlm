# Independent derivation for 9.8

Reviewer: `/root/v4_review_coordinator/factual_v4_9_8`, fresh factual/technical review.

Each of 24 box identifiers creates seven records: two permission, one known count,
one missing count, two arithmetic-premise, and one document-color record. Before
deduplication this is 24 × 7 = 168. The five records containing the box identifier
remain distinct, giving 24 × 5 = 120. Arithmetic uses a = id mod 8 and b = 3id mod 8
and omits the box identifier. Each of eight rules has two premise records, giving
8 × 2 = 16 distinct premise records. Hence 120 + 16 = 136, removing 32 duplicates.
Every rule family contains three box variants × five distinct records plus two
premise records: 3 × 5 + 2 = 17. Six training families give 6 × 17 = 102; one
validation family and one test family give 17 each; 102 + 17 + 17 = 136.

Arithmetic contains 8 × 8 = 64 ordered-pair records. There are eight diagonal
families and 8 × 7 / 2 = 28 off-diagonal unordered-pair families: 36 in total.
The independently executed seed-42 family split gives 49 / 8 / 7 records in
28 / 4 / 4 families. Mixed training is therefore 102 + 49 = 151 records.

Both branches sample 900 × 16 = 14,400 records with replacement. One supervised
answer has len(answer.encode('utf-8')) + 1 targets: its bytes and its EOS. The
sampler replay in verify.py independently sums these lengths and compares them
with render_chat's labels != -100. It obtains 394,888 targets for safety-only and
275,389 for mixed training, exactly matching the saved original run records.
Equal updates therefore do not imply equal supervised-answer-token budgets.

For each saved generation, find the first EOS ID 2; compare all preceding IDs
with the expected UTF-8 bytes plus offset 8. The recomputed score is the count of
exact sequences divided by the number of records in that specific evaluation.
EOS presence is checked separately. This yields 15/17 and 16/17 behavioral test
matches; mixed validation 14/17; arithmetic test 0/7 for base and both branches;
rewritten questions 0/6 with 6/6 EOS. The procedure never uses a decoder's visible
text alone to decide exact success, and checks that no raw answer has hidden
special tokens. results.json retains each inspected prediction and its raw-ID hash.

Information-sufficiency counterexample: two worlds may both show a red closed
box and the same earlier conversation about its color, while one contains one
ball and the other contains four. The observation is identical, yet the count
differs. Therefore color alone cannot uniquely determine count without an extra
provided color-to-count rule. An always-clarify response is appropriate for a
missing-count task but fails a known-count task. Retaining both kinds prevents
this strategy from receiving complete task credit.

Two different strings can express the same template (for example replacing only
punctuation). String inequality is sufficient to establish different byte strings;
it is insufficient to establish independent semantic/template families. Likewise,
success on original templates cannot establish success on arbitrary new templates.
The six rewrites here are three missing-count and three color-extraction tasks,
using exactly two wording substitutions and no multi-round evaluation.

Saving only an aggregate count loses which input and response caused each failure.
A trace preserving each round's input/output supports diagnosis. This is a data
retention argument, not an empirical claim of multi-round model success.

Original-authority check for the information argument: Shannon's 1948 corrected
reprint, Introduction page 1, requires a system to cover each possible selected
message. This supports evaluating both possible target behaviors rather than
only a single constant response. Section 6, pages 11–12, defines entropy and
conditional entropy, with H(y) >= H_x(y); knowing x can leave uncertainty about y.
For a transparent illustrative distribution, assign equal probability to the two
red-box worlds: H(count | red) = -2 × (1/2) × log2(1/2) = 1 bit. If the count
itself is visible, H(count | visible count) = 0. These invented equiprobable
worlds are a counterexample, not estimated frequencies of real boxes.

For aggregate-score information loss, two traces with correctness vectors
[False, True] and [True, False] both report one success out of two, yet fail at
different rounds. Assuming equal probability just for the illustration, the
failure position has one bit of uncertainty after observing the score alone,
but zero after observing the complete correctness trace. Shannon Section 6,
page 12, conditional entropy, and Section 9, page 15, singular versus invertible
transducers provide the general information-loss basis; no claim is made that
the paper specifically studied model debugging. I actually rendered and viewed
the original paper's page 12 in ignored research to confirm the formulas and
inequality directions, in addition to reading the extracted original text.

Original PyTorch v2.9.0 functional.py cross_entropy lines 3375–3466 explicitly
excludes ignore_index targets from input gradients and identifies class-index
target conditions. This supports counting non-ignored targets, rather than
equating step count with answer supervision. Local torch 2.14.1+cpu target
counts were executed independently; the older source is not represented as
version-matched certification of every current API.

The empirical verification here inspects and recomputes official stored GPU run
records; it does not independently replicate GPU training or checkpoint inference.
The original record specifies NVIDIA L4, torch 2.14.1+cu126, Python 3.13.3, seed 42,
step_scale 1.0. Its 22.208399321 seconds cover experiment training, evaluation, and
local checkpoint saves, excluding image build, startup, and HF uploads. No speed
comparison or statistical superiority is inferred. The local runtime is Python
3.13.5, torch 2.14.1+cpu. A separate untrained width-8/layer-1 CPU evaluation probe
checks unchanged parameters and absent gradients, not original-model quality.
