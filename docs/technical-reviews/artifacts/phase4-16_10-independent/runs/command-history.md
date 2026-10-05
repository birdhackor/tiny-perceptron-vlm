Actual CPU execution was captured by subprocess.run(command_argv, cwd=/workspace/tiny-perceptron-vlm, env=current environment plus the listed offline overrides, capture_output=True, timeout=30). The underlying exit status is preserved in execution-receipts.json.

Commands:
/workspace/tiny-perceptron-vlm/.venv/bin/python /workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_10-independent/code/cpu_probe.py
/workspace/tiny-perceptron-vlm/.venv/bin/python /workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_10-independent/code/raw_verify.py

Source acquisition: .venv/bin/python heredoc called urllib.request.urlopen(url, timeout=25).read() for checkpoint.py, autograd.md, activation.py and linear.py, and timeout=20 for checkpoint.md, dropout.py and memory.py; exact HTTPS URLs, hashes and errors are in sources/download-receipts*.json. A guessed docs/source/checkpoint.rst path returned 404; docs/source/checkpoint.md was retrieved successfully.

Original implementation acquisition:
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:scripts/course_experiments/architecture.py
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:scripts/course_experiments/common.py
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:tiny_perceptron/model.py
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:tiny_perceptron/data.py
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:tiny_perceptron/attention.py
git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:tiny_perceptron/modern.py

Extraction was a byte-preserving call to section_facts.original_section, then section_facts.fences; source whole file and extracted fence bytes were saved before execution.
