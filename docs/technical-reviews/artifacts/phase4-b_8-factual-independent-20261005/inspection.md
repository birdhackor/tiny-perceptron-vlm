# B.8 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_b_8`; fresh task, only B.8.
Date: 2026-10-05 (UTC).

Read the factual reviewer instructions, complete checker schema, section_facts helper,
clear-tutorial SKILL.md, and its review protocol. The cloud onboarding skill was read
for the existing CPU environment; no package installation or configuration changes
were needed. No previous technical/reader report content or author correction/result
summary was read. Initial filename discovery included artifact filenames but no
previous review content; subsequent reads were exact original files. The two allowed
locator indexes were inspected by top-level types and URL/arXiv identity fields only.

## Frozen material and reading scope

- B.8: current `course/chapters/0B.md`, source lines 313–335, read in full;
  raw UTF-8 section bytes preserved in `original-fence/section.md`.
- Necessary current context: chapter introduction and B.1 (lines 1–43), B.3
  (82–127), B.4 (128–175), and B.7 (247–312). These are claims, not proof.
- `frozen-input/course/chapters/0B.md` is the whole-chapter frozen input from this
  inspection. Its whole-file hash means only those saved input bytes, not the later
  current chapter version; B.8's separate raw section SHA is the report source hash.
- AST function locations were found before reading methods in applications.py.
  Read `_sample` 42–103, `_parse_json_action` 496–524, `_tool_episode` 527–605,
  and `run_tools` 608–660 (raw generation and criteria only). Read retrieval.py's
  `call_tool` 23–34 and `tool_loop` 37–49. The two whole original code snapshots
  match `/code_sha256` in the recorded result file exactly.
- Original tools.json was first inspected for necessary top-level keys/types and
  `/results` keys/types. Directly read provenance `/revision`, `/device`, `/seed`,
  `/torch_version`, `/python_version`, named `/code_sha256` entries, and protocol
  roles, serialization, allowlist, final contract and action cap. Read raw sample
  `/results/samples/2` for the referenced failure, including its original trace.
  The bounded script inspected the named raw sample/criterion fields for all 20
  normal episodes and independently replayed their recorded generated text.
  This recomputes recorded criteria, with no model weights or new generations.
  The full original result file was preserved unchanged; no author note or limits
  value was used as an answer. No training/data/worklog summaries were read.

## Primary external source inspections

Sources were fetched from the official HTTPS URLs in `sources/fetch-provenance.json`.
The calibration PDF was located through the allowed original paper index, copied
byte-for-byte, and its own first-page title, authors and `1706.04599v2`, 3 Aug 2017
identifier inspected. Index descriptions were not treated as evidence.

- OpenAI's original 2019 GPT-2 report, *Language Models are Unsupervised Multitask
  Learners*: read original title/authors and §2, page 2, equation (1). The
  conditional-probability product defines sequence likelihood; a single next-token
  probability is not that product and neither quantity by itself certifies semantic
  task correctness. No GPT-2 benchmark scores are used here.
- Guo et al., *On Calibration of Modern Neural Networks*, arXiv:1706.04599v2:
  first-page title/authors/version, §2 page 2 definition of calibration, equation
  (1), reliability-diagram definition, and §4 page 4 first paragraph. The paper's
  classification-confidence definition distinguishes confidence from actual
  correctness and explicitly assumes matched train/validation/test distributions.
  It does not demonstrate calibration of this course's models or prescribe the
  hypothetical probabilities in B.8.
- scikit-learn official source-tag 1.7.2 `classification_threshold.rst`: read
  entire original document, especially introduction lines 9–16 (probability
  estimation versus decisions), 47–60 (cost tradeoff), 64–80 (utility metric),
  and 120–137 (fresh validation data; do not reuse training data to tune a
  threshold). The classification example supports the general validation and
  utility principle, not a claim that B.8 runs TunedThresholdClassifierCV.
- scikit-learn official source-tag 1.7.2 `cross_validation.rst`: read original
  lines 10–22, 60–83, 312–332. Changing decisions based on final test results
  leaks information; final test data must stay separate. The same-generative-
  process assumption limits transfer to changed prompts or tool conditions.
- OpenAI official function calling guide, resolved URL
  `https://developers.openai.com/api/docs/guides/function-calling`, fetched
  2026-10-05, HTML SHA-256 recorded. It is an undated live-document snapshot,
  not a claimed immutable API release. Read original “Tools”, “Tool calls”,
  “Tool call outputs”, “The tool calling flow”, and “Incorporating results into
  response” (derived text lines 1690–1777 and 4290–4305). Application execution,
  output return and final model response are distinct steps. These steps do not
  guarantee that the final answer is correct.

## Arithmetic and execution

The original Python fence ran unchanged through the section_facts CPU/offline worker
with exit 0, empty stderr and no guard events. Saved stdout is 1.0, 0.12, TOOL.
The worker environment is Python 3.13.5 and PyTorch 2.14.1+cpu (CUDA build None,
CUDA unavailable); the elementary Python arithmetic is independent of PyTorch APIs.

`bounded_verify.py` ran with exit 0 and empty stderr. It changes only the original
fence's direct_accuracy to 0.999, reproduces 0.01, 0.12, DIRECT, and independently
uses Decimal to check both cases and the tie. With binary success/failure and zero
success loss, E[L] = 0*p + wrong_cost*(1-p) + fixed_extra_cost. Costs are common
arbitrary points, not converted seconds or dollars. For fixed p_tool=0.99, L=10,
c=0.02, tool is preferred iff p_direct < 0.988. On a tie the original strict
comparison chooses DIRECT. The separate probability counterexample 0.999*0.1=
0.0999 illustrates sequence probability only, not calibrated semantic confidence.

All 20 recorded normal episodes were replayed through the exact original method
functions with recorded model text supplied as input. Arithmetic and done/answer/
tool criteria matched the stored fields for every episode. Contextual totals are
19/20 correct and 18 actual calls. B.8's 0.99 remains explicitly hypothetical.
Sample 2 re-executes multiply(9,9)=81, returns TOOL_RESULT:81, then rejects the
stored text `{"done":true,"answer":8}}` with Extra data at char 24. Therefore it
is a failed task despite a successful calculator call. No training was repeated.

## Figure inspection

B.8 itself has no figure references. Because its failure example refers back to
B.3, the original contextual SVG `rewrite-B-tool-flow.svg` was copied, rendered
with Inkscape 1.4 and actually viewed as a PNG. It shows request → validation →
execution (123*45=5535) → user TOOL_RESULT:5535 → newly generated done+answer,
with content separately checked. The arrow order and distinction between the
outer tool record and model template roles agree with the inspected contract.
Inkscape emitted Pango/GTK wrapper warnings but exited 0 and produced a fully
readable rendered image, viewed after the logged render too.

## Judgment

No unresolved substantive discrepancy was found. B.8 is a fixed task-type policy
proposal and a hypothetical loss calculation. It claims no fitted confidence
estimator, trained selector, measured latency, dollars, or new end-to-end score.
Empirical path comparison on separate validation questions followed by untouched
final questions is methodologically appropriate, with the stated condition changes
requiring reassessment. No guarantee about arbitrary prompts, tools or per-question
calibration is supported or claimed.
