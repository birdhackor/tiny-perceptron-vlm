"""Run exact short Python fences; never run training or shell fences."""
from pathlib import Path
import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch

records = []
for chapter in ("0A", "0B", "0C"):
    source = (OUT / f"{chapter}.source.md").read_text()
    for match in re.finditer(r"^```python\n(.*?)^```", source, re.M | re.S):
        prefix = source[:match.start()]
        heading = re.findall(r"^## ([A-C]\.\d+) (.*)$", prefix, re.M)[-1]
        code = match.group(1)
        filename = f"{heading[0]}.example.py"
        (OUT / filename).write_text(code)
        stream = io.StringIO()
        error = None
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                exec(compile(code, filename, "exec"), {"__name__": "__main__"})
            except Exception:
                error = traceback.format_exc()
                print(error)
        record = {
            "section_id": heading[0],
            "chapter_source": f"course/chapters/{chapter}.md",
            "chapter_sha256": hashlib.sha256((OUT / f"{chapter}.source.md").read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "saved_code": filename,
            "stdout_and_stderr": stream.getvalue(),
            "outcome": "executed_without_exception" if error is None else "failed",
        }
        records.append(record)
        print(f"{heading[0]}: {record['outcome']}\n{stream.getvalue()}")

# Exact elementary probability calculation, not a model result.
probability_comparison = []
for k in (1, 2, 4, 8):
    probability_comparison.append({
        "k": k,
        "same_p_0_3_for_each_task": 1 - 0.7**k,
        "equal_task_mixture_p_0_6_and_p_0": ((1 - 0.4**k) + 0) / 2,
    })
print("C.3 illustrative task mixture, independent draws within each fixed task:")
print(json.dumps(probability_comparison, indent=2))

report = {
    "executed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "command": ".venv/bin/python outputs/natural-v4/wholebook-review/foundations/run_local_examples.py",
    "records": records,
    "c3_probability_comparison": probability_comparison,
    "limitations": [
        "No model-training commands or notebook cells executed in this check.",
        "Successful snippet execution is not evidence for reported trained-model metrics.",
        "The mixture numbers are elementary probability calculations, not measured pass@k.",
    ],
}
(OUT / "local-example-execution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
if any(r["outcome"] == "failed" for r in records):
    sys.exit(1)
