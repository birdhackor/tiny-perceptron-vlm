# 15.2 independent factual inspection

Reviewer task: `/root/phase4_factual_coordinator/factual_15_2`; fresh context assigned only this section. Date: 2026-10-05. No subagents, chapter/figure edits, commits, pushes, training, trained-model evaluation, dataset/model download, or engineering preparation.

## Frozen input and actual reading

`execution-original/section.md` is the original UTF-8 byte extraction of `course/chapters/15.md#15.2`, beginning source line 43. SHA-256: `c40e5c8326c8513eaa1ed74e70f8cd5cce0534519f6f847438e9a8eb76db6a5e`.

`inputs/course/chapters/15.md` is the full chapter frozen during this inspection, not a full-chapter review or a claim about future chapter versions. Its actual snapshot SHA-256, matching the extractor's initial full-file fingerprint, is `d261089b88d4d274b09276f06322e527a7862492768c06a8236614f2efea75be`.

I read all of 15.2 and its optional details. For context I also read current chapter lines 1–160: the chapter introduction, 15.1, 15.3, 15.4, and the opening of 15.5; I read the exact original-byte 14.6 snapshot for shared-parameter context. I do not audit those sections or the introduction. No figure is referenced in 15.2. The vectors and small matrices are specified in prose/code, and the independent-weight mechanism does not require readers to infer hidden spatial material.

I personally read the factual-reviewer instructions, checker schema, section_facts extraction/execution helper, clear-tutorial skill, and its review protocol. Initial broad `rg --files` returned old-artifact filenames only; no old technical/reader report contents, author review, correction summary, or extra result interpretation was opened. Subsequent implementation searches used AST inventories and precise selected methods. Ordinary API/docstring/shape, initialization-comparison restrictions, and computational method comments were read as method, not as result interpretations. No contamination event occurred.

## Primary original inspections and support limits

The locator indexes supplied only immutable original PDF paths and version identifiers. I copied the exact original PDF bytes, verified the first-page arXiv versions, and personally read the following original paragraphs. Those index entries are not evidence.

* Shazeer et al., arXiv:1701.06538v1 (2017-01-23), https://arxiv.org/pdf/1701.06538v1: §1.2/§2, pp. 2–3, independent expert parameters, matching input/output shape, sparse gating, and Eq. (1)'s avoided expert computations; Appendix C.1, p. 14, one ReLU hidden layer and one output layer as a concrete two-linear-layer expert. The original §1.2 reports observed syntax/semantic specialization after training, not innate subject labels. This supports architectural terminology and a conventional FFN example, not universal expert architecture or guaranteed specialization.
* Jiang et al., arXiv:2401.04088v1 (2024-01-08), https://arxiv.org/pdf/2401.04088v1: §2.1, pp. 2–3, separate SwiGLU weight sets and computation only for selected experts; §5, p. 7, an actual investigation of domain-based routing with no obvious topic patterns. This supports the need to observe trained behavior rather than assign an English/math label based on a module count. I do not generalize that model's finding to all MoEs.
* Official PyTorch repository, tag v2.9.0, fetched 2026-10-05 over verified HTTPS from https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/: `torch/nn/modules/container.py` ModuleList lines 332–400 and 462–493; `torch/nn/modules/module.py` parameters/named_parameters/_named_members lines 2636–2711 and named_modules 2823–2870; `torch/nn/modules/linear.py` Linear lines 53–140; `torch/autograd/grad_mode.py` no_grad lines 21–85; `torch/_tensor_docs.py` copy_ 1204–1222, mul_ 3413–3420, tolist 5513–5533; `torch/_torch_docs.py` eye 4158–4186, numel 8397–8417, tensor 9223–9276. This is an official stable-version contract, not a claim that v2.9.0 equals the installed v2.14.1+cpu build; the installed behavior is independently executed. Official generated 2.9 webpages for no_grad/copy_/numel returned HTTP 403, preserved in download-manifest.json; they are not claimed as inspected successful pages.
* Python official 3.13 builtins documentation, https://docs.python.org/3.13/library/functions.html#zip (resolved to https://docs.python.org/3.13/builtins/functions.html), `zip` description and examples, accessed 2026-10-05. Supports positional pairing of the equal-length three-element iterables used here, not any claim that unequal lengths are checked by default.

## Raw result and implementation scope

The complete raw `moe.json` is preserved without deleting its annotations; only named configuration, budget, dataset-count/hash, and provenance pointers were read as values. The actual selected pointers and values are in `checks/raw-pointer-selection.json`. Additional inventories read only key/type lists at `/results`, `/results/variants`, each variant record, `/results/variants/top2_aux0.01/heldout`, its validation/test dictionaries, `/results/variants/top2_aux0.01/validation_routing` and its layer dictionaries. I did not inspect the raw values of `comparison`, `auxiliary_note`, `limitations`, or other extra author interpretations.

The experiment revision is `48a4f3e912b483d70aee57c42c2aac226534a9a6`; its recorded model.py, modern.py, common.py, attention.py and architecture.py were extracted from Git and their full SHA-256 checked against raw provenance. Current model.py and modern.py match recorded full SHA. Current architecture.py differs as a full file; exact AST-selected `_text_dataset`, `_parameter_budget`, `_heldout`, and `_routing` methods are byte-equivalent to the recorded revision. Likewise current and recorded `new_lm`, `text_examples`, and `evaluate_lm` methods are byte-equivalent. I read current methods and those original-equivalent method bodies, plus original `run_moe` lines 428–477 (the configuration, structural comparison and raw result assembly, excluding the final author's return annotations). These are the support for the short report's measurement scope. `_routing` groups by layer/expert over validation tokens; `evaluate_lm` scores/generates text for the entire model. Neither implements per-language or per-subject expert specialization evaluation. This statement is limited to inspected measurement schema/methods.

## CPU results and limits

The exact original fence ran under section_facts with Python 3.13.5, torch 2.14.1+cpu, CPU, CUDA unavailable, no guard events, exit 0. The permanent copy retains raw fence, stdout/stderr, environment, exact worker command and result. It creates/fills weights and computes outputs; it performs no backward or optimizer update. Original output: `[[[1.0,2.0]],[[2.0,4.0]],[[2.0,1.0]]]`, `False`, 12 and 4.

`checks/check_cpu.py` separately runs the exact fence and verifies unique parameter/storage identities and unchanged shapes. The exercise multiplies only expert 0's weight by 3 under no_grad, obtaining `[[[3,6]],[[2,4]],[[2,1]]]`. A same-value copy retains independent storage. ModuleList alone raises NotImplementedError. An asymmetric weight matrix explicitly verifies Linear's `x @ W.T` convention, giving `[5,11]` for `[1,2]` and weight `[[1,2],[3,4]]`.

For four recorded MoE conditions, a constructor-only CPU model yields 2 layers × 4 independent experts = 8, each with `64×256+256+256×64+64=33,088` parameters. The expert total is `8×33,088=264,704`; routers total `2×64×4=512`; other shared model parts total 75,392; total `264,704+512+75,392=340,608`. All assertions pass. No weights were loaded, saved, or reevaluated, and no new model performance is inferred. These are structural counts, not empirical quality measurements.

## Visual execution and real limitations

The first CLI Chromium file:// attempt produced no screenshot for approximately 35 seconds and was stopped by its exact owned shell/browser PIDs; the initial view_image call failed because no file existed. A desktop and mobile retry each timed out at 30 seconds with no screenshot. I preserve actual stderr and timeout records; these failures do not establish a file:// policy rejection.

One fallback used Playwright's system Chromium 151.0.7922.173 `set_content` with the same frozen standalone HTML, DOMContentLoaded, and external requests aborted. It rendered screenshots at actual viewports 1280×800 and 390×844, then I personally viewed both with view_image. Full-page heights were 1409 and 2371 pixels. The prose/numbers are visible, and the code uses horizontal scrolling on mobile. No SVG/figure or arrow/label claim exists in this section.

This is a standalone Python-Markdown preview, not a published-site/theme/mobile-navigation acceptance. The optional details' internal Markdown link remains literal in this preview because the helper does not parse Markdown inside its raw HTML block; I do not treat that as a defect in the chapter or claim the actual website link was verified. The figure-consistency check is not applicable because 15.2 has no referenced figure; screenshots document the honest visual scope only.

## Judgment

Within 15.2's substantive claims, no technical contradiction or unresolved substantive uncertainty was found. The toy computation is labeled as a manually specified independence demonstration, and the optional larger counts are limited to capacity rather than expert specialization. No chapter/figure corrections are requested. External-page 403 and browser file:// timeouts have successful original-source/set_content alternatives with the stated limits above.
