"""Execute the B.5 original fence and its two explicitly prescribed edits."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = Path(__file__).resolve().parent
code = (ART / "tool-choice-b5-original-code.py").read_text()
cases = {
    "original": code,
    "flag-only": code.replace('"calculator": True, "action": "TOOL"', '"calculator": False, "action": "TOOL"', 1),
    "repaired": code.replace('"calculator": True, "action": "TOOL"', '"calculator": False, "action": "ASK"', 1),
}
expected = {
    "original": ["TOOL", "DIRECT", "DIRECT", "ASK", "ASK"],
    "flag-only": ["TOOL", "DIRECT", "DIRECT", "ASK", "ASK"],
    "repaired": ["ASK", "DIRECT", "DIRECT", "ASK", "ASK"],
}
results = {}
for name, text in cases.items():
    filename = ART / f"tool-choice-b5-{name}-code.py"
    filename.write_text(text)
    tree = ast.parse(text)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    assert [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)] == ["print"]
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(text, str(filename), "exec"), namespace)
    rows = namespace["examples"]
    actions = [r["action"] for r in rows]
    assert actions == expected[name]
    assert len(rows) == 5
    assert all(type(r["calculator"]) is bool for r in rows)
    assert rows[0]["question"] == rows[-1]["question"]
    assert "加" in rows[1]["question"] and rows[1]["action"] == "DIRECT"
    (ART / f"tool-choice-b5-{name}-stdout.txt").write_text(output.getvalue())
    results[name] = {
        "code_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "row_count": len(rows),
        "actions": actions,
        "calculator_values": [r["calculator"] for r in rows],
        "calculator_types": [type(r["calculator"]).__name__ for r in rows],
        "stdout": output.getvalue(),
        "first_row": rows[0],
        "source_calls": ["print"],
        "source_imports": [],
    }
assert results["flag-only"]["first_row"]["calculator"] is False
assert results["flag-only"]["first_row"]["action"] == "TOOL"
assert results["repaired"]["first_row"]["action"] == "ASK"
assert 1 + 2 == 3
section = (ART / "tool-choice-b5-current-section.md").read_text()
table_actions = [line.split("|")[-2].strip() for line in section.splitlines() if line.startswith("| ") and line.split("|")[-2].strip() in {"TOOL", "DIRECT", "ASK"}]
assert table_actions == expected["original"]
svg = ET.parse(ROOT / "course/figures/tool_choice.svg").getroot()
ns = {"s": "http://www.w3.org/2000/svg"}
texts = [element.text for element in svg.findall("s:text", ns)]
arrows = [element.attrib for element in svg.findall("s:path", ns)]
assert [arrow["d"] for arrow in arrows] == ["M383 141H438", "M383 233H438", "M383 325H438"]
assert all(arrow["marker-end"] == "url(#arrow)" for arrow in arrows)
for label in ["TOOL：需要工具", "DIRECT：直接回答", "ASK：先求助", "工具可用 → TOOL　　工具停用 → ASK", "人工策略圖；不是模型成績，也不是工具已執行。"]:
    assert label in texts
marker = svg.find("s:defs/s:marker", ns)
assert marker.attrib["orient"] == "auto"
record = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-b5-exercise-audit.py",
    "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "device": "CPU", "model": "none", "platform": platform.platform()},
    "result": "All original/flag-only/repaired assertions passed; each executed fence printed five rows, no model or calculator invoked.",
    "expected_actions": expected,
    "arithmetic_sanity": {"input": [1, 2], "operation": "integer addition", "expected": 3, "observed": 1 + 2},
    "cases": results,
    "table_actions": table_actions,
    "figure": {"texts": texts, "arrows": arrows, "marker": marker.attrib, "source_sha256": hashlib.sha256((ROOT / "course/figures/tool_choice.svg").read_bytes()).hexdigest()},
    "scope": "Exact local Python fence and the two edits prescribed in B.5; no trained classifier or model accuracy was tested.",
}
(ART / "tool-choice-b5-exercise-execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(record["result"])
print(json.dumps({name: item["actions"] for name, item in results.items()}, ensure_ascii=False))
