# 5.5 fresh technical review, 2026-10-05

The selected original section is `course/chapters/05.md#5.5`, source lines 150–181. I also personally read actual source lines 60–149 of chapter 05 (incidental end of 5.2 and full 5.3/5.4), plus the complete linked 1.13. This is technical review, not a new blind-reader session. I did not read any previous technical/reader review report or its judgments. Methods and schema were read in full.

## Primary reading

- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, arXiv:1711.05101v3, dated 4 January 2019, ICLR 2019. Newly downloaded directly from arXiv verified HTTPS. Personally read title/version, Abstract, Introduction and §2, printed pages 1–4; specifically Equation (1), Proposition 1, Algorithm 2, Proposition 2 and the explanatory paragraph after it. Actually rendered and viewed PDF page 3: purple line 6 is the L2-only `+λθ` gradient term, green line 12 is the AdamW-only decay term. The text extractor alone loses that color distinction.
- Exact PyTorch upstream commit `5c4886908584029761b579af026dcfb627c84070` matches the CPU wheel's reported git commit. Downloaded `adamw.py`, `adam.py`, `optimizer.py`, `parameter.py`, `_torch_docs.py`, `_tensor.py` from the project's official raw GitHub repository. Four installed implementation files (`adamw`, `adam`, `optimizer`, `parameter`) were independently snapshotted and byte-hash matched to upstream. Actual installed version is `2.14.1+cpu`; this report does not pretend older v2.5.1 snapshots were the execution environment. The v2.5.1 files were acquisition candidates; decisions use matching commit.
- Read AdamW constructor lines 19–48 and official algorithm docs lines 59–108; actual Adam group filter lines 150–189, step dispatch lines 214–272, update lines 413–475 and 528–546. Read `zero_grad` docs/body 1048–1093, group contract 267–269 and 1127–1187, Parameter 30–57, zeros_like docs 12647–12677, tensor docs 9582–9610, allclose docs 834–865, detach 798–813.
- Read Google's original BERT `optimization.py` at immutable commit `eedf5716ce1268e56f0a50264a88cafad334ac61`: header authority, optimizer factory 59–65, `apply_gradients` 108–157 and `_do_use_weight_decay` 159–167. Supports the existence of an explicit policy excluding bias and LayerNorm; it does not imply PyTorch excludes these automatically.
- Read scikit-learn's official cross-validation documentation source, tag 1.7.2, lines 10–98, particularly 60–70: held-out validation selects hyperparameters while final test evaluation is separate. Only that methodological recommendation is used; sklearn code was not installed or executed.

## Own CPU evidence and scope

Original extraction helper completed with exit 0 and no guard events; stdout `tensor([1.9600])`. Raw source and fence bytes, helper/version, bootstrap and original stdout/stderr/environment/execution are permanent under `original/`.

The separate 45-second bounded CPU script executed the original fence unchanged and the exact two suggested modifications, including adjusted assertions. Results: fresh zero-gradient decay -> 1.9600000381469727 with m=v=0 and step 1; fresh no decay -> exactly 2; fresh None -> exactly 2 and no optimizer state. Decimal calculation independently gives factor 0.98, remaining 1.96, shrink 0.04. Float32 absolute error is approximately 3.815e-8, below 1e-6 review tolerance; original allclose defaults are rtol=1e-5 and atol=1e-8.

Autograd verifies a task minimum `(w−2)^2` has task gradient 0 at w=2, while explicit penalty `(0.2/2)w²` gives total gradient 0.4. Adam's first step leaves w=1.9000000025 and records m=0.04, v=0.00016. Decoupled AdamW leaves w≈1.96 with m=v=0. Plain SGD with the same penalty produces 1.96, matching the original paper's limited standard-SGD equivalence. This is a counterexample and mechanism check, not training or generalization evidence.

A task-loss calculation shows decay can raise `(w−2)^2` from 0 to approximately 0.0016. An explicit PyTorch parameter-group demonstration changes the weight to 1.96 while leaving bias and normalization scale at 2. These scope checks support possibility and API policy, not a universal instruction that biases/norms must never decay.

One important boundary was checked: after a previous nonzero-gradient step, a subsequent zero gradient retains momentum, so the update is not only decay; `grad=None` instead preserves both parameter and state. The selected fence constructs a fresh optimizer and describes this particular experiment, so its isolated-decay explanation is supported. Similarly, the paper's Algorithm 2 uses a separate schedule multiplier and a different decay-coefficient convention. The textbook's factor `1−ηλ` is precisely PyTorch's documented API convention, not a literal symbol mapping of that algorithm.

## Figures, result inputs and limits

No image or SVG is referenced in 5.5; no textbook figure render is applicable. The scalar calculation and zero/None state distinction are directly inspectable in the formula and short code; no image geometry, arrows or unseen materials are required. The original paper's colored algorithm really was rendered/viewed. This review does not claim desktop/mobile page rendering.

No empirical measurement JSON, sample/token counts, checkpoint or model benchmark is asserted or referenced by 5.5. I did not invent or rerun training results. The new bounded-check JSON was personally read and verified against actual stdout/state values, and every original empirical-input category is recorded as not applicable. No datasets, models, weights, GPU, full training recipe, paid operation or upload were used.

All seven substantive claims have per-claim original-source locators and scope in the report. No material contradiction or unresolved substantive claim found; verdict is pass. No正文/figure edit or commit was made.
