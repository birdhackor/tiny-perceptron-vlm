# 5.16 original technical owner continuity callback

Reviewer: `/root/phase4_factual_coordinator/factual_5_16`; 2026-10-05.
This is my real callback, not a new fresh initial review or an author/reader verdict.
Before parsing my prior canonical report, I copied it opaquely into this callback's
`history/prior-pass-5.16.json`; `archive-record.json` records its identity. All prior
artifacts, prior issues (none), proof scopes and the original PASS remain preserved.
No other technical, reader, continuity or author correction report was read.

## Current material personally read

I read the complete current `docs/review-tools/factual-reviewer-instructions.md`,
then all of current 5.16, current 5.2's explanation of token-mean loss and ignored
labels, the final paragraphs of 5.15 at chapter source lines 572–586, and the full
linked 7.16 disclosure. The saved 5.15 excerpt is a complete section for version
context; I do not claim to have reread its opening code in this callback.
These are manuscript claims, not proof of themselves. Current section bytes and
necessary context bytes are preserved under `current/`, with original newline
semantics and SHA-256 in `current-inputs.json`.

The 5.16 diff contains exactly one replacement paragraph: PAD is now explained as
the placeholder used to align different input lengths, and question/padding cells
are said not to enter the supervision denominator. All other paragraphs, the
Python fence, exact numerical values, exercises, record/token ratio caveat and the
original replay disclosure are unchanged. I personally read the new sentence and
the entire surrounding section; I did not infer acceptance from unchanged hashes.

## Personally rechecked primary contracts

I reread my original pinned official PyTorch v2.14.1 `loss.py` lines 1200–1254
and `functional.py` lines 3478–3525, plus CPython v3.13.5 `random.py` lines
458–492. Their raw snapshot hashes still match their original HTTPS retrieval
receipts. No new source search or retrieval was needed.

I reread current `tiny_perceptron/data.py` lines 54–86 and
`tiny_perceptron/model.py` lines 92–105, and current
`scripts/course_experiments/common.py` lines 122–145. An AST query identified the
actual functions and line ranges. The original implementation fingerprints and
result/input fingerprints remain identical to the independent review's preserved
original evidence. The replay input pools/500-step branch are in original
`text.py` lines 614–632; callback byte comparison confirms that original contract
is unchanged. No result-interpretation constants or author correction notes were
used as proof.

The original result was inspected only at named measurement/provenance pointers:
`/revision`, `/seed`, `/results/runs/b-only/training/{steps,effective_tokens}`,
`/results/runs/replay/training/{steps,effective_tokens}`, and the original data
manifest hashes under `/results/data/{attributes,arithmetic}/train/sha256`.
The complete original JSON fingerprint is retained; no notes, review or correction
fields were printed or used. These checks establish that the unchanged original
18453/36384 counting proof still applies, not a newly run retention experiment.

## Present claim-by-claim judgment

1. The record/token arithmetic and both exercises remain exact: 90 short tokens,
   200 long, total 290; ratios 9/29 and 20/29. Long length 1 gives 90/10; long
   count 5 at length 20 gives 90/100, total 190, long share 10/19. The unchanged
   original-fence and literal-only exercise evidence is preserved; the current
   fence was really executed again in this callback.
2. The equal valid-position mean is still explicitly conditional on summing all
   290 losses and dividing by 290. A position share is not the numerical loss
   contribution or a guarantee of gradient influence; the paragraph explicitly
   retains this distinction, and the official formula/source supports it.
3. The newly explained PAD terminology matches actual behavior. `pad_batch`
   fills input padding with PAD ID 0, target padding with IGNORE -100 and marks
   attention validity false. `render_chat` excludes user/system/role targets and
   supervises assistant content plus EOS; `loss_sum` counts only nonignored
   targets. The new tiny CPU check below tests this contract directly.
4. The 180:1 relation still concerns cumulative fixed-length token totals or
   their expected cumulative ratio, not exact random batch composition. The
   section retains its batch feasibility qualification and cumulative accounting
   recommendation. My prior explicit finite-batch/global-mean boundary proof is
   preserved and is still the support scope; no per-step coefficient claim was
   added by the PAD sentence.
5. Truncation, record resampling and custom source loss factors remain distinct
   accounting operations. Neither the current prose nor my report treats
   PyTorch's class-weight argument as an automatic source-weight API. The original
   SFT loader continues to reject overlong records rather than silently crop them.
6. The original empirical disclosure still refers to 500 updates ×16 sampled
   records =8000 draws per branch, not equal unique record pools or token budgets.
   The saved, originally recounted labels 18453/36384 and ratio 1.9717119 are
   unchanged, original-record/code identities still match, and the result scope
   has not widened.

## New bounded execution and remaining limits

`verify_current.py` executes the actual current math fence and two tiny hard-coded
chat fixtures through the real `render_chat`, `pad_batch`, `loss_sum` and
`masked_loss`. It creates no model, optimizer, gradient graph or dataset file.
The two examples have a [2,11] padded shape: 22 cells, 5 padding cells, 11 ignored
nonpadding question/role positions, and [2,4] valid assistant-plus-EOS targets.
The loss denominator is 6; changing all ignored scores leaves mean loss unchanged.
It also asserts the current/context bytes, unchanged code/data/result/source
fingerprints, and the original specified measurement pointers. The real .venv CPU
process has a 30-second timeout; its complete stdout/stderr, environment and exit
status are retained here. Prior exercise/count-only replay checks were not rerun
because neither their code nor the numerical/source contracts changed.

The section still has no image/SVG or other figure reference. No visual render,
browser layout check or missing-figure claim is made. No GPU work, long recipe,
model/data download, upload, paid compute, network request, weight read or model
quality measurement occurred. No manuscript or other owner's report was edited.

Verdict after this personal reinspection: PASS. No unresolved substantive issue or
external dependency. Prior and current evidence boundaries remain in force.
