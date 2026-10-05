# Fresh 5.10 independent inspection, 2026-10-05

Reviewer task: `/root/phase4_factual_coordinator/factual_5_10`. No child agents, no old technical/reader report judgments, no chapter edits, no training or model loading.

## Actual read scope

Read all four mandated method/schema files personally. Read exact original 5.10 (chapter lines 357–395 in captured version); training T.4, lines 125–205; 5.15 as referenced context. Also saw the end of 5.8 and 5.9 when locating chapter line numbers (lines 305–356); these sections are outside this review's verdict. Capturing the full chapter is input version preservation, not a claim to reviewing the whole chapter or its introduction.

Read `text.py` `_deduplicate_text` lines 91–100, `_save_splits` 47–62, `_evaluations` 65–72, `_asset_rows` 75–88, and `run_real_text` 317–350. Read `common.py` `split_records` 50–71, `text_examples` 77–100, `_nll` 106–120, `fit_lm` 122–203 and `evaluate_lm` 243–304. The train-only input is the explicit `parts["train"]` argument in run_real_text; fit_lm constructs examples and samples solely from those records. Original commit `5581462ef01959636425eb7795ae3153142dbfeb` byte hashes are independently checked against the result's `code_sha256` by git show and by fresh captured current files.

## Original authoritative passages personally read

- Cawley and Talbot, JMLR 11(70), 2010, pp. 2079–2107. Downloaded directly from the journal HTTPS PDF and matched bibliographic publisher page. Read original title/abstract/introduction pp. 2079–2080; §4 and §4.1 pp. 2084–2086 (finite sample variance and optimizing sample peculiarities); §5 and §5.1 pp. 2094–2095 (model selection forms part of fitting and must be inside evaluation). In own `cawley-talbot-2010.txt`, inspected lines 1–119, 305–482 and 1080–1186. This supports selection bias as a methodological risk, not a universal rate of bias for every candidate list. The bounded enumerated +/-0.1 case shows the mechanism under its explicit independence/equal-quality assumptions; it is not empirical model performance.
- Official scikit-learn source tag 1.7.2, `doc/modules/cross_validation.rst`: lines 60–78 explain tuning/test contamination and training/validation/final test uses; lines 628–657 explain dependent groups and evaluating unseen groups. Read the original project source text, not a search snippet. The project example `plot_nested_cross_validation_iris.py` lines 6–26 explains optimistic non-nested selection and separate inner validation/outer test. No sklearn execution is claimed or needed for these conceptual citations.
- CPython tag v3.13.5 official documentation, `Doc/tutorial/controlflow.rst`: `tut-if` and `tut-for`, lines 11–89, conditional execution and sequential list iteration. `Doc/tutorial/datastructures.rst`: `tut-dictionaries`, lines 490–544, literal dictionaries and key lookup. Actual CPU execution covers the specific float comparisons, slicing, reassignment and printed output in the fence.
- TinyStories authors' pinned Hugging Face README at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, full 21 lines: identifies upstream `TinyStories-train.txt`, separate validation text, and evaluation prompts. Fresh HTTPS snapshot equals the existing original card bytes. This card does not define or certify a course test set. The original acquisition receipt and raw prefix show all 512 retained stories complete, and every input row has upstream `source_split=train`.
- chinese-poetry project's README at `b8594f81a89752241442f2ce267d6f66f96704ee`, read introduction and dataset listing lines 1–110. This identifies a JSON anthology. Existing original `tang300-source.json` consists of 366 complete poem objects, has no train/validation/test split fields, and renders to the 365 retained unique complete texts; fresh CPU audit checks the exact rendering. No anthology download was performed.

## Execution and permanent evidence

All shell execution used `bash`, `login:false`; `.venv` Python 3.13.5, torch 2.14.1+cpu; torch.version.cuda=None and cuda unavailable. `prepare.py` invokes section_facts helper with a 30-second worker bound, captures original UTF-8 bytes and hashes, and copies real original execution artifacts from temporary outputs into this docs directory. Exact fence succeeds. `probe.py` runs under `timeout 40`, CPU/offline environment, no model construction/update/inference. It executes the original code plus 0.6, 0.9 and tie variants; reconstructs original grouped splits and checks original raw result counts, hashes, training-record digest, window counts, byte/EOS denominators, and NLL and BPB arithmetic. The records/split checks are exact; floating aggregate arithmetic uses relative tolerance 1e-12. Stored samples are only eight generations per text side; NLL uses every effective target.

Commands actually run:

```bash
.venv/bin/python docs/technical-reviews/artifacts/phase4-5_10-independent/prepare.py > docs/technical-reviews/artifacts/phase4-5_10-independent/prepare.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_10-independent/prepare.stderr.txt
.venv/bin/python docs/technical-reviews/artifacts/phase4-5_10-independent/fetch_sources.py > docs/technical-reviews/artifacts/phase4-5_10-independent/source-fetch.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_10-independent/source-fetch.stderr.txt
pdftotext -layout docs/technical-reviews/artifacts/phase4-5_10-independent/sources/cawley-talbot-2010.pdf docs/technical-reviews/artifacts/phase4-5_10-independent/sources/cawley-talbot-2010.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 40 .venv/bin/python docs/technical-reviews/artifacts/phase4-5_10-independent/probe.py > docs/technical-reviews/artifacts/phase4-5_10-independent/probe.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_10-independent/probe.stderr.txt
.venv/bin/python docs/technical-reviews/artifacts/phase4-5_10-independent/version_probe.py > docs/technical-reviews/artifacts/phase4-5_10-independent/version-probe.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_10-independent/version-probe.stderr.txt
```

## Limits and figure applicability

No referenced figure or original shell recipe in 5.10, so there is no section figure to render/view. There is no spatial mechanism beyond the explicit list comparison that requires a missing diagram. No browser or desktop/mobile page visual verification is claimed.

0.7/0.8 and exercise 0.6/0.9 are specified illustrative scores, not measured success rates; there is no sample-count denominator behind them. The program neither owns nor secures a test dataset; its final line labels the demonstration's scope, and AST/execution confirms no test score exists.

Original real_text result is a historical CUDA pilot. Its test is local holdout drawn from the upstream train prefix, not an official TinyStories score. `before` and `after` both contain test evaluations. The fixed fit_lm path has no test-dependent hyperparameter/seed selection. Evidence establishes train-only weight updates and full-document disjoint exact-identity groups; it does not establish that the test was viewed only once across all repository development, or that near-duplicate related stories/poems are eliminated (the raw result explicitly warns no near-duplicate clustering). Neither stronger assertion appears in this section. No new performance score is manufactured.
