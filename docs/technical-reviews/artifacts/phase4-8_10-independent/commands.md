# Actual commands and scope

All repository shell commands were launched with `login:false`, cwd `/workspace/tiny-perceptron-vlm`.

Original extraction: `.venv/bin/python docs/review-tools/section_facts.py course/chapters/08.md#8.10 --output /tmp/phase4-8_10-extract`.

Original execution: `.venv/bin/python docs/review-tools/section_facts.py course/chapters/08.md#8.10 --output /tmp/phase4-8_10-original-cpu --execute --timeout 30`. Original helper worker argv, cwd, environment, exit code and elapsed time are in original-cpu/execution.json and environment.json. Artifacts copied byte-for-byte to the permanent review directory; no symlinked workspace or weights copied.

Original authority retrieval: `.venv/bin/python docs/technical-reviews/artifacts/phase4-8_10-independent/sources/fetch_originals.py`; HTTP receipts saved in sources/fetch-receipts.json. Conversion: `pdftotext -layout docs/technical-reviews/artifacts/phase4-8_10-independent/sources/mt-bench-v1.pdf docs/technical-reviews/artifacts/phase4-8_10-independent/sources/mt-bench-v1.txt`.

Short CPU variations were first run with explicit offline environment values, then executed through `.venv/bin/python docs/technical-reviews/artifacts/phase4-8_10-independent/variants/run_with_receipt.py` to preserve true subprocess exit status. Actual child argv, result, timeout, source/stdout/stderr SHA and offline values are in variants/execution.json; software/device facts are in variants/environment.json.

Visual render: `timeout 30 .venv/bin/python docs/technical-reviews/artifacts/phase4-8_10-independent/visual/render_section.py`, stdout/stderr and render-receipt.json saved. Initial file:// navigation failed; the same command after replacing navigation with page.set_content succeeded and produced both PNGs; initial script/stderr saved in visual/attempt1-file-url. Personally opened both PNGs with view_image. No HTTP server or occupied port was used.

Final schema/evidence checker: `.venv/bin/python scripts/check_technical_reviews.py --lesson 8.10`, invoked via check-report.py with true subprocess exit code/stdout/stderr saved. This only validates schema/current source/evidence hashes and identity uniqueness, not truth of the claims. Global round checking is the coordinator's task after all sections complete.
