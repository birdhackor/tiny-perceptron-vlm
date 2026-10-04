# Original-source inspection, 9.3

Read by `/root/v4_review_coordinator/factual_v4_9_3` on 2026-10-04.
Retrieval hashes and resolved URLs are in `retrieval-receipts.json`. Complete
third-party PDFs and extracted text remain only in ignored research output.

## Sharma et al., Towards Understanding Sycophancy in Language Models

Read original https://arxiv.org/pdf/2310.13548v4, arXiv v4 dated 10 May 2025,
ICLR 2024 conference paper. Inspected PDF pages 1–8, especially abstract and §2
(p. 1–2), §3.2–3.4 (p. 3–5), §4.1 (p. 5–6), §4.2 (p. 6–7), and §4.3–4.3.1
(p. 7–8). The paper defines sycophancy as undesirable approval-seeking, studies
answer changes and repeated user mistakes, and finds evidence that preference
judgments can favor agreement at the cost of truth. The 9.3 claim says “may”:
this agrees with the original's conditional evidence. In §4.2 optimization has
mixed effects; it is incorrect to infer every preference model or optimization
procedure necessarily increases every form of sycophancy.

§4.3 explicitly distinguishes a bare correction, helpful truth with explanation,
and convincing agreement with a misconception. This supports the task's rubric
distinction, without guaranteeing that the toy examples train a capable fact
checker. The human experiment restricts internet/fact-checking tools and is
proof-of-concept. I use no paper percentages as claimed local measurements.

## Rafailov et al., Direct Preference Optimization

Read original https://arxiv.org/pdf/2305.18290v3, arXiv v3 dated 29 July 2024,
NeurIPS 2023. Inspected original PDF page 1 for version and context; page 3, §3
Reward Modelling Phase, for labels `y_w` and `y_l`; page 4, §4/Eq. 7 for the
preference dataset's role. §3 describes human labelers selecting a preferred and
dispreferred completion from a pair for the same prompt. This supports the
meaning of chosen/rejected in the displayed human-authored example. It does
not make a preference label a guarantee of factual correctness, and 9.3 does not
run DPO or claim its optimization objective is executed.

## Original project run code

The saved GPU record names revision
`a864a60bbf72583afc9bbaf45e052bd4fe076c62`. Current `behavior.py` has a different
full SHA, so I actually retrieved its original source at the pinned revision:
https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/a864a60bbf72583afc9bbaf45e052bd4fe076c62/scripts/course_experiments/behavior.py
Its full SHA exactly matches the saved run's recorded hash. I personally read
original `_conversation` (lines 21–26), `_safety_records` (363–418),
`_safety_evaluations` (421–451), and `run_safety` (490–534). Their actual statements
confirm the prompt/target generation, shared-base copying, 900-update versions,
split use and frozen paraphrase evaluation. A fresh executed AST comparison
confirms these four functions are equivalent to the current corresponding
functions. The current common.py, text.py, data.py and model.py full hashes match
the saved run's recorded hashes. This resolves the relevant version boundary;
it does not assert that every current behavior.py function equals the old file.

## Official objective/subjective distinction and target semantics

Read OpenAI Model Spec, explicitly pinned historical version 2025-12-18:
https://model-spec.openai.com/2025-12-18.html#avoid_sycophancy
I read the full “Don't be sycophantic” subsection, including its examples, and
the relevant “Express uncertainty” opinion passage. The former distinguishes
objective answers, whose factual content should not change solely to agree with
the user, from subjective questions, where interpretation and thoughtful rationale
are appropriate. The latter treats an opinion as subjective. This is a published
behavior specification, not empirical proof of any model complying with it.
Applying that distinction to “I like blue” requires the transparent logic in my
derivation; the document does not independently verify the speaker's preference.

Read original PyTorch 2.14 CrossEntropyLoss documentation:
https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
Inspected the criterion description, class-index target definition and unreduced
class-index loss equation. The loss scores the supplied target, not the factual
correctness of its meaning. Current project model.py lines 92–105 uses this
cross-entropy target semantics through F.cross_entropy and its answer mask.
My separate float64 CPU probe (one explicit two-class example; zero updates)
checks a stale answer4 target for prompt3+3 versus correct answer6. The stale
target makes the loss gradient favor class4 even though arithmetic requires6.
This supports the label-consistency warning; it does not assert a measurable
whole-model degradation from one example. Actual runtime remains2.14.1+cpu.
