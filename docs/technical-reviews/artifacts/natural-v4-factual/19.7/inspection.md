# Independent factual inspection of 19.7

Reviewer: `/root/v4_review_coordinator/factual_v4_19_7`, fresh assigned context.
Read the entire current raw UTF-8 body, including all trailing blank lines, from
`## 19.7` through immediately before the next heading. Its SHA-256 is
`9bb0e3e6a5f4748790bf47f6bb7ed9c5487c58325b7d1819a07f8d78db93621c`.
No introduction is applicable. The previous report was copied as bytes without
reading it; its preserved SHA is `ddb37343572de6a14f60c6c6623633a649d1019026fbf93435531c6f4fe2ba09`.

Read the complete explicit prerequisites B.1–B.4 and 19.1; also read 19.4 for
the same-question pretrain comparison, 13.2 for paired preferences, and 19.12
for the final tool denominators. Their current raw section hashes are recorded
in the report. No earlier review judgments or author checks were used.

Original authority inspection, retrieved on 2026-10-04:

- OpenAI function-calling guide, heading `#the-tool-calling-flow` and preceding
  “Tool calls” / “Tool call outputs”: a generated request, application execution,
  second model request with actual output, and final model response are distinct.
  This supports the mechanism, not this project's model-quality numbers.
- RFC 3629 (November 2003), sections 1 and 3, table row U+0000–U+007F:
  ASCII characters encode in one octet; the request's 19 ASCII characters yield
  19 UTF-8 bytes. `ByteTokenizer.encode` maps each byte to one ordinary ID.
- OWASP Input Validation Cheat Sheet, “Input Validation Strategies”,
  “Allowlist vs Denylist”, and “Parse Safely, Then Validate”:
  parsing and exact allowed-set/type/range validation are separate checks.
  The lesson's calculator-only executor implements a particularly narrow rule.
- Rafailov et al., arXiv 2305.18290v3, sections 3 and 4, Eq. (7), and “DPO
  outline”: the preferred and dispreferred completions share a prompt; the
  policy objective uses their likelihood ratios against a reference. The brief
  DPO definition is correct; it does not promise tool or modality improvement.

Read actual current source locations: `tiny_perceptron/data.py` lines 14–29
(bytes and EOS); `tiny_perceptron/capstone.py` lines 139–282 (dataset families,
fixed templates and reserved 1+2), 299–319 (serialized prompts), 379–497
(actual generated IDs and stopping), 499–606 (scoring, parse, runtime and
same-model two-call orchestration); training script lines 70–278 (stage inputs,
frozen data, schedules and validation). Full-file SHA values are registered.
Pinned public source and per-question record bytes match the local files.

Executed the literal lesson code, unknown-tool exercise, integer/type/range
boundaries and an explicitly scripted generation injection into the real
`run_assistant`. The injected model's final `DIRECT:4` is preserved despite
the runtime's `3`, and both generation calls receive the same model object.
This is an orchestration test, not trained-model generation.

Independently decoded token IDs and checked EOS, original prompts, parsed
actions, ordered arguments, runtime sums, follow-up prompts and final answers
for all 426 original rows across pretrain/SFT/joint/DPO validation and joint
test. Recomputed all stored scores rather than trusting summary booleans.
SFT/joint/DPO each give 10/10 tool requests, executions and final answers,
10/10 unavailable ASK without execution and 5/5 independent tool-return
answers. The joint test gives 12/12 requests and executions, 10/12 final
answers, 12/12 unavailable ASK without execution and 5/6 independent returns.
The exact failure IDs and held-out 1+2/2+1 successes are in `audit-results.json`.

The audit rebuilds only the existing small synthetic dataset, verifies all
726 rows and split digests, disjoint families and predeclared numbers:1:2
reservation. Original records identify seed 42, completed schedules 300,
1400, 600 and 100, Python 3.13.3, torch 2.14.1+cu126 and NVIDIA L4.
Parent export hashes and deployment joint checkpoint hash agree. Recorded
training seconds exclude validation; whole-experiment seconds include
training, evaluation and local saves and exclude startup and HF uploads.
These are original run records, not my own GPU replication. My bounded audit
uses Python 3.13.5, torch 2.14.1+cpu, two CPU threads, zero optimizer updates.

Personally viewed all three newly rendered PNGs. The direct vertical diagram
shows user → model request → validated true calculation → same-model final
generation. 1+2→3 is correct, and “人工流程示意” prevents confusing its example
with a measured success. The bottom checklist separates action, arguments,
execution and final correctness. The B.3 prerequisite figure runs left-to-right
through request/validation/pure function/actual return/stopping; it agrees with
B.3's stated flow. The pipeline prerequisite shows A→B→C and a dashed C→D
preference branch, plus final tool return to the same core, consistent with the
stage provenance relevant here. All labels and arrow positions are visible.
Chromium's file navigation was blocked; the retained initial stderr documents
that failure. Rendering the exact local SVG bytes with `page.set_content`
succeeded without changing the SVG or browser settings.

The content is supported within the fixed, narrow template and arithmetic
world. Separate tool-return rows are not counted as full loops; ordered-string
accuracy is not a mathematical ability score; held-out families are not novel
natural-language templates; perfect request selection does not imply reliable
reading of every result or general uncertainty calibration. No unresolved
factual issue was found in the current body.
