# Independent factual review record: 14.10

Reviewer task: `/root/phase4_factual_coordinator/factual_14_10`.
Access date: 2026-10-05. This is a fresh technical review of this section only.

Personally read: the current original 14.10, original prerequisite sections 14.9
and 14.3, and the referenced SVG source and actual rendered image. The frozen
chapter and introduction preserve original raw bytes; the introduction is
**not reviewed by this non-first-section reviewer**. The full chapter hash in
`frozen-input.json` describes that saved input, not a claim about later chapter
versions.

Read workflow/schema first: `docs/review-tools/factual-reviewer-instructions.md`,
`scripts/check_technical_reviews.py`, `docs/review-tools/section_facts.py`,
`.agents/skills/clear-tutorial/SKILL.md`, and its `references/review-protocol.md`.
No old technical or reader reports, author review, result interpretation or
revision summary was consulted. The only additional lesson text read is the
named prerequisite original text. Its ordinary scope restrictions are method
descriptions, not a prior review verdict. No contamination event occurred.

Locator indexes were inspected for top-level key/type, then immutable original
locator fields (`path`, `sha256`, `first_page_arxiv_identifiers` for `/records/*`;
`url`, `version_label`, `original_path`, `sha256` for `/locators/*`). No exact
YaRN v3 locator was found. No author result JSON was needed or read. Downloaded
the exact cited original PDF directly from arXiv. Personally inspected its first
page identifier: `arXiv:2309.00071v3 [cs.CL] 6 Feb 2026`.

Primary reading: §2.1 Eq.(2); §3.1 uniform PI contraction; §3.2 relative-local
motivation, Eq.(10) rotations ratio, Eq.(11) ramp, Definition 1 Eqs.(12–13),
case-specific α/β; §3.3 Eq.(14), Definition 2 and model-specific Eq.(15);
§4.1 training method, §4.2 long sequence language-model evaluation method,
§4.3 passkey-retrieval evaluation method, §4.4 short-task benchmark method.
Viewed rendered original pages 5 and 6 to verify the fractions that text
extraction flattens. The paper's own outcome prose and tables appeared while
reading those original sections; these are original published authority, not
repository author commentary. Only definitions and methodology support the
claims here; no repository model win, usable context length, or training result
is inferred.

The γ in the lesson follows original Definition 1: 0 means θ/s, 1 means θ,
with intermediate convex interpolation. Original r=L/λ=Lθ/(2π) measures
rotations over the original context; α and β are model-dependent boundaries.
No universal speed thresholds are claimed. Eq.(14)'s score multiplier is 1/t;
when both Q/K are scaled by m=√(1/t), the dot product gets m². The lesson's
c is the score multiplier, and its example c=2 is explicitly just a toy value.
Original Eq.(15) is a recommendation for LLaMA/Llama2, not a universal constant.

CPU verification: no original fence exists (Python=0, other=0). Executed the
saved bounded `check_math.py`; no training, backward, parameter updates, dataset
or model download. Softmax is over two candidate positions; score differences
are dimensionless. Float64 PyTorch results were independently checked using
stable scalar exp and the two-exponential denominator. Rounded two-decimal
values use strict error <0.005; exact ties equal [0.5,0.5]. Additional c=0.5,1,2
cases preserve argmax and reduce entropy as c rises. This establishes a weight
mechanism, not semantic correctness. Endpoint/midpoint frequency checks use
chosen L=4096, s=8, α=1, β=32 to exercise original Eqs.(10–13), not to estimate
model scores. No empirical denominator or model performance claim is present.

Figure: rendered `frozen-figure.svg` with Inkscape 1.4 and personally viewed the
640×763 PNG. The fast/intermediate/slow rows, transition arrows, and score→
positive scale→softmax arrows agree with the original definitions and prose.
Necessary labels and arrowheads are visible and not clipped. This is a standalone
figure visual check, not a claim of desktop/mobile full-site acceptance.
The attempted lookup of a literal `site/` directory failed because that directory
does not exist; it was not used as rendering evidence. The actual SVG rendering
succeeded, with return code 0 and empty stderr in `command-records.json`.
Both original-PDF page renders also succeeded; original stdout/stderr files are
retained. No failed rendering was relabeled as success.

Support boundaries: mathematical mechanism and the authors' declared method
and task scopes only. The lesson says model adoption needs appropriate choices
and verification; the original paper includes fine-tuned and non-fine-tuned
settings, so this text is not evidence that every YaRN use requires fine-tuning.
Changed positional calculations or more concentrated attention cannot by
themselves certify longer-context answering or short-task preservation.

Outcome: no unresolved substantive issue found in the reviewed original.
No course text, SVG, training configuration, commit, or push was changed.
