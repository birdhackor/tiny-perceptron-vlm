# Independent factual inspection, 13.8

Reviewer: /root/phase4_factual_coordinator/factual_13_8. This reviewer read no old
technical/reader report, review history, author correction summary, or other
reviewer's conclusions. The immutable locator index was used only to locate an
original paper; the paper itself, its exact version, and its authority were checked.

## Read scope and frozen input

- Current source `course/chapters/13.md#13.8`, source lines 249–273, original UTF-8
  bytes, SHA-256 `24e5f91fc833502eb5ac2f28dfdb0d0ae45994fbfeb7cfc1371ad97096caaf48`.
- `inputs/chapter-13-frozen.md` is the full chapter at initial extraction, SHA-256
  `c33e12dd7e61e73f11a45e3f7ee8c099dd130d23133c5a15426adfe2ca24adbe`.
  This means frozen input snapshot, not an assertion about the current whole chapter.
- Necessary preceding source sections 13.1 and 13.2 were read to check that
  `prompt/chosen/rejected` are task-conditioned human labels and that this section
  continues the preference-record convention. Their experimental supplements are
  outside this review's claims and were not used to support 13.8.
- Read factual-reviewer instructions, checker schema, section_facts helper,
  clear-tutorial skill and review protocol. 13.8 is not the chapter's first section;
  no chapter-introduction review is claimed.
- The source has one 367-byte Python fence and no figure, external measurement,
  training command, tensor axis, probability denominator, or performance percentage.

## Original sources personally checked

1. Sharma et al., *Towards Understanding Sycophancy in Language Models*,
   https://arxiv.org/pdf/2310.13548v4, arXiv v4, 10 May 2025, ICLR 2024 paper.
   Read first-page title/authors/version, abstract, §1–2 (PDF pp.1–2), §4.1
   (pp.5–6), §4.3/4.3.1 (pp.7–9), §5 prevention paragraph (p.9).
   The abstract says feedback can encourage responses matching users' beliefs over
   truthful ones. §4.3 directly compares convincing mistaken-belief agreement with
   helpful truthful correction, and reports that humans and preference models
   sometimes prefer the former. It does not show that a particular user prefers
   “2+2=5”; the section only states this reversal is possible. The original study's
   findings are used as conceptual background, not reproduced experimental results.
   The arXiv abstract page independently confirmed v4/date/title/authors. A new
   download from the exact PDF URL was byte-identical to the locator's original PDF,
   SHA-256 `ee764bd30119f2146f2e130a099d6d313fca6c70ab07b17b7fdbde456d96be36`.

2. OpenAI Model Spec, https://model-spec.openai.com/2025-12-18.html,
   dated 18 December 2025. Read `#avoid_sycophancy` and
   `#avoid_being_condescending` from the official versioned page.
   The former says factual aspects of objective answers should not differ with
   phrasing, and an assistant should not change its stance solely to agree with
   a user. The latter asks for honest constructive assistance without condescending,
   dismissive or judgmental language; its “favorite state” example also illustrates
   respecting a subjective preference without unnecessary factual correction.
   This is an official behavioral criterion, not proof of any trained model's behavior.

3. Python official release 3.13.5 documentation, matching the installed Python:
   `library/functions.html#print` (conversion, default space separator/newline,
   stdout), `library/stdtypes.html#mapping-types-dict` (literal string-key mapping
   and d[key]), `reference/expressions.html#binary-arithmetic-operations`
   (numeric + gives sum). These connected APIs are checked as one data-record/print
   contract. Full original pages and inspected excerpts are saved.

All HTTPS sources were fetched on 2026-10-05 with TLS verification intact;
URL/status/version/hash provenance is saved in the fetch JSON files.

## Original code and bounded variants

Before reading implementation content, AST enumerated `scripts/build_course.py`
top-level function ranges and BOOTSTRAP (lines 26–61). Only BOOTSTRAP was needed:
it initializes imports/CPU reproducibility, and its Colab install branch is not
taken in this environment. The exact fence AST is one dictionary assignment and
four `print` expressions. It calls no text judge, model, backward pass or optimizer.

Actual original execution:
`.venv/bin/python docs/review-tools/section_facts.py 'course/chapters/13.md#13.8' --output /tmp/phase4-13_8-independent-cpu --execute --timeout 60`
completed in 1.8223 seconds with exit 0. Python 3.13.5, torch 2.14.1+cpu,
CUDA build None, CUDA unavailable, attempted fence [1], no guard events and no
repository model module imports. True stdout is in `cpu-original/stdout.txt`;
stderr is empty. Only named small files were copied from the helper run. Its
runtime workspace/symlinks, weights and archives were not copied.

The original prints arithmetic 4 and the exact chosen/rejected/reason strings.
`verify_variants.py` executes the original bytes and a literal replacement of chosen
with “笨蛋，當然是4”. The replacement still prints 4 and the unchanged old reason.
Its insult is an independent human tone judgment; there is no automated language
classification. A separate arithmetic matrix computes 2+2 and the new 3+4 under
correct and incorrect premises and varied wording. The corresponding expected
actions are confirm/correct according to truth, not a blanket “對吧” rejection.
A self-reported blue preference asserts no arithmetic value. These are reference
criteria and data-design controls, not learned-generalization results.

The only arithmetic unit is an object count: two disjoint groups, each containing
2 objects, total 4 objects. Exact integer equality; no rounding, floating tolerance,
axis, rate or statistical denominator is involved. Adding correct-premise controls
can contradict a universal-refutation shortcut as a labeling rule; this text does
not claim the example guarantees a trained model avoids every shortcut.

## Actual rendering and inspection

No SVG/image is referenced by 13.8, so figure consistency is not applicable.
The actual Markdown section was converted to local HTML with minimal review CSS.
Chromium 151.0.7922.173 / Playwright 1.63.0 first blocked file navigation with
`net::ERR_BLOCKED_BY_ADMINISTRATOR`; this was not a timeout. Original script,
results/stdout/stderr of that attempt are retained. Loading the same generated HTML
with `page.set_content` succeeded at 1280×800 and 390×844. Both real screenshots
were viewed by this reviewer. Text and code examples are present, Chinese characters
render, and no horizontal overflow was measured. Mobile code wraps. This is not
an acceptance claim for the complete published course site.

## Judgment and limits

All material statements in 13.8 are covered by the report's seven claims.
The examples separate truth, tone, and task-conditioned labeling. “能教” is scoped
to the comparison/data-design explanation; neither the fence nor this review
establishes model training, accuracy, generalization, or eliminated sycophancy.
No unresolved factual issue was found. No source or diagram was changed, no agent
was spawned, no commit was made, and no dataset/model download, full training,
GPU work, or re-evaluation of an existing model was run.
