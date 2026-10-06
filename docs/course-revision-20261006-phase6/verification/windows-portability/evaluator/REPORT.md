# Evaluator Windows portability verification

Completed on 2026-10-06 against baseline revision `77d0589887847777d85aebdbc5b74351f12237b0`.

The production patch changes two lines in `scripts/selftrained/evaluate.py`:

- `code_fingerprints()` uses `relative_to(root).as_posix()` so protocol map keys have canonical `/` separators on Windows as well as Linux. The source boundary and SHA-256 computation are unchanged; native Linux key spelling is unchanged.
- The initial test receipt is opened with `r+b` before `os.fsync()`. Windows requires a write-capable descriptor for this operation. Opening with this mode does not truncate or rewrite the receipt, and the sync still precedes advancement of the latest-attempt marker and generation.

The original Windows CI job, `112325891684`, failed with receipt `fsync` errors (`OSError: [Errno 9] Bad file descriptor`) and backslash fingerprint keys. The supplied job log is `outputs/phase6-main-publication/112325891684-actual-failed-job.log`.

The focused software regressions in `tests/test_selftrained_training_eval.py` check canonical keys for native and simulated Windows-relative paths, enforce a writable receipt descriptor on every operating system, and compare exact receipt bytes before and after the sync. The existing frozen-condition and second-test protections remain exercised.

| Check | Result | Evidence |
| --- | --- | --- |
| Original evaluator loaded from Git HEAD in memory; focused regressions | 2 failed as expected; pytest exit status 1 | [baseline-regression.log](baseline-regression.log) |
| Entire existing evaluator/training software-test module | 31 passed; 0 failures/errors/skips | [evaluator-tests.log](evaluator-tests.log), [evaluator-tests.xml](evaluator-tests.xml) |
| Final focused regressions, including explicit native-key separator assertion | 2 passed; 0 failures/errors/skips | [focused-regressions.log](focused-regressions.log), [focused-regressions.xml](focused-regressions.xml) |
| Ruff lint | Passed | [lint.log](lint.log) |
| Ruff formatting | 2 files already formatted | [format.log](format.log) |
| `git diff --check` for the two owned files | Passed | [evaluator.patch](evaluator.patch) |

Commands ran from `/workspace/selftrained-v2` with `/workspace/tiny-perceptron-vlm/.venv/bin/python` (Python 3.13.5):

```bash
/workspace/tiny-perceptron-vlm/.venv/bin/python -m pytest -q tests/test_selftrained_training_eval.py --junitxml=docs/course-revision-20261006-phase6/verification/windows-portability/evaluator/evaluator-tests.xml
/workspace/tiny-perceptron-vlm/.venv/bin/python -m pytest -q tests/test_selftrained_training_eval.py::test_code_fingerprints_use_posix_keys_for_windows_relative_paths tests/test_selftrained_training_eval.py::test_frozen_protocol_rejects_changed_conditions_and_second_test --junitxml=docs/course-revision-20261006-phase6/verification/windows-portability/evaluator/focused-regressions.xml
/workspace/tiny-perceptron-vlm/.venv/bin/python -m ruff check scripts/selftrained/evaluate.py tests/test_selftrained_training_eval.py
/workspace/tiny-perceptron-vlm/.venv/bin/python -m ruff format --check scripts/selftrained/evaluate.py tests/test_selftrained_training_eval.py
git diff --check -- scripts/selftrained/evaluate.py tests/test_selftrained_training_eval.py
```

For the baseline proof, `git show HEAD:scripts/selftrained/evaluate.py` was compiled and executed into the imported evaluator module's in-memory namespace before invoking `pytest.main()` for the two focused cases. No source file was reverted or overwritten. The outer verification wrapper accepted only pytest exit status 1, and the log identifies both expected failures.

The existing tests use disposable tiny CPU fixtures under pytest temporary directories. No actual course model, candidate record, heldout set, saved output, or frozen protocol was used or altered; no Modal run occurred. The hashing and frozen-protocol validation code was not relaxed. The live evaluator's own digest naturally changes with its source patch, so existing frozen course protocols continue to bind their historical source rather than silently accepting this version.

Verification ran on Linux. The guards reproduce the relevant Windows descriptor and path requirements, but a native Windows CI rerun remains necessary before claiming that job passes. This work created no commit or push and changed only the two owned source/test files plus this evidence directory. Concurrent publisher-fixture changes belong to another task.

The complete owned diff is [evaluator.patch](evaluator.patch); machine-readable run metadata and source digests are [verification.json](verification.json).
