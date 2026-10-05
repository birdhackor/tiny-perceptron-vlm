# C.2 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_c_2`.
Date: 2026-10-05. This is a new technical review. No earlier technical or
reader report body, author result commentary, training/data worklog, or correction
summary was read. The original source locator index was inspected only for
source paths, identifiers and hashes; the papers were then fetched and read anew.

## Actual input and read scope

The current C.2 bytes are retained in `original-fence/section.md`. SHA-256:
`959cfac0b9ca1a19aeede2abb6c9186de3df41a3ec2cb11249273ceb35676c2b`.
The full Markdown read was C.1–C.7 and the chapter opening, for context; only
C.2 is judged. `inputs/0C-frozen.md` is the actual frozen whole-file input,
SHA-256 `d9959533b8acd624fcc8d20120bb7830905f60737f92f78f200d37d582d9f577`.
This whole-file hash describes that frozen input, not an assertion that every
other section is independently verified or unchanged later. C.2 is not the
chapter's first section. All section hashing used original UTF-8 bytes without
stripping or newline normalization.

C.2 has one Python fence and no image reference, SVG, mathematical display,
shell recipe, or long-running recipe. No figure was available or necessary to
judge its arithmetic: the explicit tuples and equations contain all quantities.
Figure render/view is therefore not applicable; no render was claimed.

The following instructions were personally read: factual-reviewer-instructions,
the checker schema, section_facts, clear-tutorial SKILL.md and review-protocol.
Cloud-environment-onboarding:setup and its onboarding reference were also read;
existing Python CPU tools sufficed and no configuration was changed.

## Original code inspection

AST was used first to locate the necessary methods. Read locations:

- `scripts/course_experiments/applications.py`: `_sample` lines 42–103;
  `_reasoning_records` 679–695; `_reasoning_sft` 698–704;
  `_verify_reasoning` 707–747; `_reasoning_samples` 750–811;
  `run_reasoning` 1071–1103 (creation, shared initialization, split and evaluation
  path only; its result-summary return literals were not read).
- `scripts/course_experiments/common.py`: imports and `split_records` 50–70;
  `records_sha256` 73–74.
- `tiny_perceptron/data.py`: SPECIALS and ByteTokenizer lines 12–28.

`git show 910aebc6419c9fc6217279a27fde9851c5cfad30:scripts/course_experiments/applications.py`
was saved without printing the file. Its complete SHA matches both the existing
report code provenance and current code: `8fbb7e1c1a5ee33c50884a3cb6658352aacd80d2131ac6561c272e8c75b2ade0`.
Necessary common and tokenizer copies also matched the report's code hashes
in the executed audit. No model construction, checkpoint loading, training,
model download, training-data download or new model scoring was performed.

## Raw result access

The top-level keys/types were inspected before any values. Necessary pointers
were then inspected; result commentary and unrelated result sections were not read.

- `/schema_version`, `/experiment_id`, `/revision`, `/device`, `/seed`,
  `/torch_version`, `/python_version`, `/gpu`, `/step_scale`, `/evidence_status`,
  `/unfinished_schedules`: original run provenance.
- `/code_sha256/scripts~1course_experiments~1applications.py`,
  `/code_sha256/scripts~1course_experiments~1common.py`,
  `/code_sha256/tiny_perceptron~1data.py`: original code version checks.
- `/results/split/{train,validation,test}`: original records, families and hashes.
- `/results/comparison/steps/budgets/0..3/candidate_count` and
  `/results/comparison/steps/budgets/0..3/samples`: raw rows and candidates only.
  The audit checks each row's family/a/b/c/question/truth and each candidate's
  generated text, generated_ids, eos, invalid_special_tokens, generated_tokens,
  final_answer, parsed, steps, final_correct, equations_valid, linked,
  task_operands_valid, fully_verified. Existing aggregate success-rate values
  and author explanations are not used as proof.
- Exemplar: `/results/comparison/steps/budgets/1/samples/7/candidates/0`.

The complete original result is retained unmodified in `inputs/reasoning-original.json`.
Original/copy SHA equality was checked before use. It is the formal evidence
copy; no ignored cache is the only evidence. Source and saved method SHA checks
are in the executable audit output.

## Independent calculation and criteria

Original fence: `(2,2,5),(5,-1,4)` yields equations `[False,True]`, linkage
`[True]`, final correctness `True`. Changing only the first claimed value to 4
yields `[True,True]`, `[False]`, `True`. Thus neither final correctness nor a
collection of correct local equations establishes a valid linked solution.

The bounded variants also include a different problem `(2+3)-1=4` and the
arithmetically valid redundant path `2+2=4;4+1=5;5-1=4`. Both pass the local
flags, which proves those flags omit task membership and acceptance criteria.
Redundant correct algebra is not inherently a false mathematical proof. C.2
says that local checks alone cannot accept unmotivated extra steps; it does not
claim that every redundant step is necessarily mathematically wrong. The
original paper's positive/neutral distinction must not be replaced by such a
blanket rule.

Original `_verify_reasoning` accepts an exact two-addition syntax, rejects
invalid special tokens, checks both arithmetic equations, subtotal/total/final
linking, exact original operands, and equality of final answer to truth. A
fallback can separately parse a final answer even when the full form fails.
AST-extracted original code was exercised with valid, false equation, unlinked,
wrong operand, final-only parse and invalid-token cases. This is a narrow
task-specific contract, not a verifier for arbitrary natural-language proofs.

The original candidate is `5+0=5;5+3=10;answer=10`, for truth 8. It is parseable,
linked and uses the task's operands, but its second equation and final answer
are wrong. ByteTokenizer decoding of its recorded 23 IDs (including EOS) exactly
reproduces that text. The illustrative old candidate does not itself show a
correct final answer with a wrong process; the manual first example does.

The original test split was independently reconstructed from 216 arithmetic
rows: 24 test rows in six unseen permutation families; its hash matches the raw
report. Existing step candidates across k=1,2,4,8 contain 24/48/96/192 candidate
instances, totaling 360, all on those same 24 tasks. All stored per-candidate
verification and token records were independently recalculated and matched.
These are checks of saved evidence, not new model performance measurements.

## Personally checked external sources

1. HTTPS `https://arxiv.org/pdf/2305.20050v1`: Lightman et al., *Let's Verify
   Step by Step*, arXiv v1, 31 May 2023, OpenAI authors. Personally inspected
   abstract, introduction pp.1–2, Methods and Scope §2/§2.1 p.3, generator/data
   collection §2.3/§2.4 p.4, and Appendix D p.19. The primary paper distinguishes
   final-result from step supervision and explicitly discusses incorrect
   reasoning reaching correct final answers. It describes fixed generators,
   reward-model supervision and best-of-N final-answer evaluation on MATH;
   those model performance figures are not asserted for C.2's tuples.
   §2.4's label description and Appendix D's operational instructions both
   concern appropriateness/reasonableness, not arithmetic alone. Appendix D
   labels appropriate/correct/easily verified steps neutral and adds progress
   for positive labels; it does not prove that any extra valid step is false.

2. HTTPS `https://arxiv.org/pdf/2305.04388v2`: Turpin et al., *Language Models
   Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought
   Prompting*, arXiv v2, 9 December 2023, NeurIPS 2023 paper. Personally inspected
   abstract/introduction pp.1–2, counterfactual faithfulness method §2 pp.3–4,
   tested models/data/perturbation/prompting conditions §3.1 pp.4–5, and
   qualitative analysis §3.3 p.6. The primary research compares biased and
   unbiased inputs for text-davinci-003 and claude-v1.0 on 13 BBH tasks and a
   separate BBQ setup. Some explanations have no obvious errors while failing
   to mention causal bias. It supports the limited statement that plausibility
   or arithmetic correctness does not establish faithfulness; it does not
   measure the internal reasoning of this tiny model, and no new mature-model
   scores were generated.

Both immutable version PDFs were fetched directly from arXiv with normal HTTPS
verification on 2026-10-05. Their visible first-page identifiers were checked.
Downloaded/saved PDF SHA pairs match. Full primary files plus concise inspected
text-page snapshots are retained. The index was used only to locate a candidate
version and was not evidence of any factual judgment.

No unresolved substantive factual issue was found in the current C.2. This
review does not perform an engineering, continuity or reading-time review.
