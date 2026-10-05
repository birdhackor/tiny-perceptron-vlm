# Phase 4 independent factual review: 14.3

Reviewer task: `/root/phase4_factual_coordinator/factual_14_3`. Date: 2026-10-05.

Read only the current 14.3 raw section and necessary original authority/source methods. No chapter introduction or other section was needed or read; 14.3 is not the first section. No old technical/reader reviews, author review, or precomputed correction summaries were read. The whole chapter snapshot is saved as a frozen input fingerprint only, not as an assertion that other sections were reviewed.

`frozen/extraction.json` preserves extractor hashes and original bytes (section starts at chapter line95; fence at line104). The original fence was directly executed with the existing CPU `.venv`; the extractor bootstrap was captured but not executed. No chapter or figure was modified. No referenced figure exists (`figure_sha256={}`); no image rendering was applicable.

Actual execution:

- `.venv/bin/python docs/technical-reviews/artifacts/phase4-14_3-independent/execution/run_checks.py`: original fence and numeric/axis variants exited0; the initial scope check exited1 because current architecture.py full-file SHA differs from the historical result provenance.
- The original result `/revision` is `4ef6555710e9915179a4a69738cdfabe80c299cf`. `git show 4ef6555710e9915179a4a69738cdfabe80c299cf:scripts/course_experiments/architecture.py` retrieved the original source, whose SHA exactly matches `/code_sha256/scripts/course_experiments/architecture.py`. The AST-located `run_modern` node was then independently read.
- `.venv/bin/python docs/technical-reviews/artifacts/phase4-14_3-independent/execution/run_scope_corrected.py`: corrected check uses that exact historical source and exited0. Original fence and numeric checks were not repeated.

The initial raw run files were renamed with `initial-` prefixes after execution to preserve the failure. `initial-runs.json` still contains the original creation-time stdout/stderr paths. Map its `execution/<name>` references to the permanent `execution/initial-<name>` files; their bytes and SHA remain unchanged. `initial-check_scope.py` preserves the exact source that failed. No failure was discarded or represented as a passing first run.

Raw-result read scope: top-level necessary keys/types, `/results` keys/types, `/results/variants` names and each variant's necessary method schema, then exact `/experiment_id`, `/revision`, `/results/variants/<name>/changes`, `/results/variants/<name>/model/config`, and `/code_sha256/{tiny_perceptron/model.py,tiny_perceptron/attention.py,scripts/course_experiments/architecture.py}`. No training/heldout numbers or result-interpretation strings were read. The complete unchanged raw JSON and full SHA are saved under frozen inputs.

Repository code was located by AST before method contents were read. `sources/repository-code-inspected-nodes.json` records exact current nodes/ranges. `sources/run-modern-result-revision-excerpt.txt` records the exact historical method. Its comparison and limitation strings state ordinary control/method scope and no empirical winner or precomputed correction; reading them did not contaminate this review.

Original authority snapshots are concise text excerpts personally read from original papers or official upstream source. `sources/original-provenance.json` records URL, version, original full hash, authority, date and the actual read locations. Large unrelated source copies were removed only after the raw full hashes and relevant original excerpts were preserved. No prior review's interpretation was copied.

Supported scope: nonzero vectors above the normalize epsilon in this two-candidate example; forward dot-product/normalization/softmax math; positive temperature and reciprocal scale; one actual RMS-based Q/K implementation example; absence of a QK normalization group in the recorded modern comparison and absence of a dedicated TinyLM option. This is not a whole-model QKNorm training, runtime, quality, stability or ranking result. No model or data was downloaded, no model was instantiated or evaluated, and no gradients or parameter updates were run.
