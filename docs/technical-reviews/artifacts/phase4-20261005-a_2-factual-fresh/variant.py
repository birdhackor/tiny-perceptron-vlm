"""Execute the original A.2 fence and only override its input source in short variants."""
import ast
import contextlib
import io
import json
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "original-fence/fence-1.py").read_bytes()
tree = ast.parse(raw)
namespace = {}
baseline_stdout = io.StringIO()
with contextlib.redirect_stdout(baseline_stdout):
    exec(compile(raw, "original-A.2-fence-1.py", "exec"), namespace)
baseline = namespace["source"].copy()
assert baseline == {"id": "公告1", "date": "2027年3月1日", "address": "青街8號"}
tail = ast.Module(body=tree.body[1:], type_ignores=[])
cases = []
for field, value in (("address", "紅街9號"), ("id", "公告2")):
    changed = {**baseline, field: value}
    output = io.StringIO()
    ns = {"source": changed}
    with contextlib.redirect_stdout(output):
        exec(compile(tail, "A.2-input-override-only", "exec"), ns)
    assert ns["context"] == "小小書店在" + changed["date"] + "搬到" + changed["address"] + "。"
    assert ns["prompt"].startswith("[來源" + changed["id"] + "] ")
    assert "由資料確定的答案 " + changed["address"] in output.getvalue()
    assert changed["date"] == baseline["date"] and namespace["source"] == baseline
    cases.append({"changed_field": field, "source": changed, "stdout": output.getvalue()})
print(json.dumps({"environment": {"python": platform.python_version(), "device": "cpu"}, "baseline_stdout": baseline_stdout.getvalue(), "variants": cases, "scope": "dict field access, f-string prompt construction and print only; no model or generation"}, ensure_ascii=False, indent=2))
