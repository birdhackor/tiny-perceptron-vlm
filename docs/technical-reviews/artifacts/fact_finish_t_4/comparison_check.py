"""Execute the exact T.4 comparison block on current CPU evaluation JSON, including a failed guard."""

import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
body = (OUT / "section.md").read_text()
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
code = next(block for block in blocks if "compare_validation" not in block and 'kind = "attributes"' in block)
(OUT / "comparison-snippet.py").write_text(code)
execution = json.loads((OUT / "execution.json").read_text())
results = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "cases": [],
}
with tempfile.TemporaryDirectory(prefix="fact_finish_t_4-comparison-") as temporary:
    directory = Path(temporary)
    (directory / "outputs").mkdir()
    for kind, target in [("attributes", "cpu_validation_sft"), ("text", "cpu_validation_text")]:
        report = {**execution[target], "data": "same-validation.jsonl", "declared_split": "validation"}
        for when in ("before", "after"):
            (directory / f"outputs/{kind}-{when}.json").write_text(json.dumps(report))
            (directory / f"outputs/{kind}-prompt-{when}.txt").write_text(
                "real CPU baseline used to test comparison guards"
            )
        script = directory / "compare.py"
        script.write_text(code.replace('kind = "attributes"', f'kind = "{kind}"'))
        proc = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(script)], cwd=directory, capture_output=True, text=True
        )
        assert proc.returncode == 0
        actual = json.loads((directory / f"outputs/{kind}-comparison.json").read_text())
        assert actual["before"]["effective_tokens"] == actual["after"]["effective_tokens"]
        results["cases"].append({"kind": kind, "returncode": proc.returncode, "result": actual, "stderr": proc.stderr})
    after = directory / "outputs/text-after.json"
    report = json.loads(after.read_text())
    report["data"] = "different.jsonl"
    after.write_text(json.dumps(report))
    (directory / "outputs/text-comparison.json").unlink()
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), str(script)], cwd=directory, capture_output=True, text=True)
    assert proc.returncode != 0 and not (directory / "outputs/text-comparison.json").exists()
    results["cases"].append(
        {
            "kind": "different-data-guard",
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "no_comparison_written": True,
        }
    )
results["scope"] = (
    "Identical actual untrained CPU reports test the comparison block and both task schemas; no learning improvement is inferred."
)
(OUT / "comparison-check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "pass", "cases": len(results["cases"])}))
