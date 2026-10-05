# 14.8 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_14_8`, fresh task for 14.8 only. I read the current 14.8 section in full, prerequisite 14.7 in full (original lines 242–268), and the RoPE explanatory portion of 14.1 (original lines 5–25). I did not review the chapter introduction. `inputs/chapter-14-frozen-input.md` is the actual whole-chapter raw input frozen for this run, SHA-256 `292b95b0fdd08459dde76bc6c237da58129852b35f7aabd8bd5f6ee7f88b37a3`; its hash does not assert that other sections remain unchanged later. The formal section hash applies to `inputs/section.md` only.

I used the factual-reviewer instructions, checker schema, section-facts extraction helper, and clear-tutorial review protocol. The cloud-environment-onboarding setup skill was read as required; the existing Python 3.13.5 `.venv` supports this bounded CPU review without setup changes.

No old technical/reader report, author review, course source-note interpretation, or extra author result explanation was read. The broad Git status output exposed path names of unrelated changes and opaque history files, but no contents or judgments. Primary locator indexes were examined as locator metadata only and yielded no matching Position Interpolation entry. I downloaded and personally read the exact v2 paper directly. No repository experiment result JSON is relevant to this section, and none was opened.

## Primary inspection and support

- Chen et al., *Extending Context Window of Large Language Models via Position Interpolation*, arXiv:2306.15595v2, 28 June 2023, https://arxiv.org/pdf/2306.15595v2, downloaded 2026-10-05. Title/version personally checked from the original first page. §2.1 equations 1–2 (PDF p.3; extracted lines 134–158) define rotations through `m theta_j`, with Q/K comparison depending on relative displacement for fixed Q/K. §2.3 Eq.4 (PDF p.4; extracted lines 236–252) replaces `f(x,m)` by `f(x,mL/L′)` and maps `[0,L′)` to `[0,L)`. This supports dividing indices by the length ratio before RoPE, rather than dropping cards or mapping both discrete endpoints identically. Fractional coordinates arise directly from Eq.4; it does not imply prior training at every fractional grid value.
- The same paper §2.3 Fine-tuning (p.5; lines 293–299) describes next-token prediction on the extended context with interpolated position encodings. §3.1 (p.6; 317–341) tests particular LLaMA variants and changes the position rescaling while retaining the architecture. §3.2 (p.6–7; 342–351,369–385) evaluates long-sequence language modeling and notes original-window degradation and improvement with fine-tuning. §3.3 (p.7; 414–434) defines the passkey retrieval evaluation. §3.4 (p.9; 508–517) separately evaluates standard tasks within the original context. These support the section's bounded instruction to adapt and test long and short tasks; they do not promise unchanged quality for arbitrary models or zero-training extension. I did not independently rerun any paper experiment.
- Python 3.13 official documentation, https://docs.python.org/3.13/library/stdtypes.html, inspected `#numeric-types-int-float-complex`, `#common-sequence-operations`, `#lists`, `#ranges`; and https://docs.python.org/3.13/library/functions.html, `#len`, `#print`. The actual inspected excerpts are saved with original extracted line locators. Together these support the ordinary APIs as a group: ordered `list(range(n))`, `range` default start 0 / step 1 / exclusive stop, zero-origin indexing, list construction, multiplication/division, item counts, and string printing. Installed Python is 3.13.5; online branch 3.13 documentation is maintained independently of that patch version.

## CPU results

The frozen original Python fence ran unmodified in a separate isolated Python process and printed both counts as 16 and mappings `0→0.0`, `1→0.5`, `2→1.0`, `14→7.0`, `15→7.5`. The prescribed exercise variant ran with `L_prime=24`, printed both counts as 24 and `5→1.6666666666666667`, `23→7.666666666666667`, matching four-decimal values 1.6667 and 7.6667. Exact `Fraction` arithmetic additionally checked identity length 8, length 16, length 24, distinct coordinate count, step size, interval bounds, and final coordinate `8−8/L_prime`. All values agree with the float calculation; floating tolerance is 2e-15, half-grid values are exact, and four-decimal rounding tolerance is 0.00005.

The toy angle of 60 degrees per coordinate unit gives 120 degrees at old position 2, 60 degrees after scaling, and reduces the adjacent angle difference from 60 to 30 degrees. The distance from position 0 to 15 changes from 15 to 7.5 coordinate units. Degrees are only the stated teaching example; the independent cos/sin check explicitly converts to radians. The original fence has only built-in coordinate operations and prints; it loads no model, computes no gradients, performs no parameter update, and measures no task quality.

## Personally viewed figures

I rendered and viewed native PNGs with Inkscape 1.4 of all four figures in the section or actually read prerequisites: `rewrite-14-8-position-interpolation.svg`, `rewrite-14-7-position-range.svg`, `rewrite-14-shared-rotation.svg`, and `rewrite-14-rotation-components.svg`. The prerequisite images correctly provide the unchanged sixteen-card wording, training positions 0–7, added positions 8–15, and RoPE angle context; they are prerequisite inspection, not a formal re-review of those sections.

The 14.8 image visibly contains the same sixteen characters in order, original indices 0–15, and one downward arrow per card to 0,0.5,...,7.5. Row 2 continues from index 8. Its footer agrees with the formula and interval bound. A separate 342-pixel-wide Inkscape rendering was also viewed; all indices, halves, arrows and footer remain readable. XML checks matched all sixteen token labels, indices, output coordinates, and arrow directions to the independent rational calculation. Visual inspection, not XML alone, supports this result.

Chromium 151 page screenshots genuinely timed out: the initial default-profile call at 45 seconds, then an independent-profile call at 30 seconds. The actual commands and second attempt's partial stderr are preserved; the first attempt's partial browser output was not captured by its initial runner. Inkscape succeeded and supplies the factual diagram inspection. No successful browser screenshot or complete desktop/mobile course page validation is claimed. There is no unresolved substantive claim in 14.8.

## Actual command sequence

All shell commands used `bash` with `login:false`, working directory `/workspace/tiny-perceptron-vlm`. Principal reproducible commands:

```bash
.venv/bin/python docs/review-tools/section_facts.py course/chapters/14.md#14.8 --output outputs/reviewer-tools/phase4-14_8-independent
curl --fail --location --max-time 40 --silent --show-error https://arxiv.org/pdf/2306.15595v2 --output docs/technical-reviews/artifacts/phase4-14_8-independent/sources/position-interpolation-2306.15595v2.pdf
curl --fail --location --max-time 40 --silent --show-error https://docs.python.org/3.13/library/stdtypes.html --output docs/technical-reviews/artifacts/phase4-14_8-independent/sources/python-3.13-stdtypes.html
curl --fail --location --max-time 40 --silent --show-error https://docs.python.org/3.13/library/functions.html --output docs/technical-reviews/artifacts/phase4-14_8-independent/sources/python-3.13-functions.html
.venv/bin/python docs/technical-reviews/artifacts/phase4-14_8-independent/code/prepare_evidence.py
.venv/bin/python docs/technical-reviews/artifacts/phase4-14_8-independent/code/verify_coordinates.py
```

`execution/*.json` saves exact child-process argv, outcomes, and environment where applicable; corresponding stdout and stderr remain alongside them. The preparation script was revised to preserve the second timeout and to use an independent Chromium profile after the first failure. `execution/chromium-initial-timeout.json` retains the first command and the honest initial capture limitation. No runtime workspace was copied, no symlink was followed into evidence, and no weights, dataset, training setup, GPU run or full model evaluation was created.
