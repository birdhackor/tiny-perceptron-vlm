from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import platform
import re
import torch

root = Path(__file__).resolve().parents[3]
text = (root / "course/chapters/20.md").read_text(encoding="utf-8")
section = re.search(r"^## 20\.2 .*?(?=^## )", text, re.M | re.S).group()
snippet = re.search(r"```python\n(.*?)```", section, re.S).group(1)
records = []
for label, code, expected in [
    ("chapter", snippet, [(8510128128, 7.93), (4255064064, 3.96)]),
    ("exercise", snippet.replace("2_127_532_032", "328_128"), [(1312512, 0.0), (656256, 0.0)]),
]:
    captured = StringIO()
    with redirect_stdout(captured):
        namespace = {}
        exec(compile(code, f"20.2-{label}", "exec"), namespace)
    observed = [(namespace["parameters"] * n, round(namespace["parameters"] * n / 1024**3, 2)) for n in (4, 2)]
    assert observed == expected
    records.append({"case": label, "code": code, "stdout": captured.getvalue(), "observed": observed})
from tiny_perceptron.capstone import CapstoneModel, default_config
capstone = CapstoneModel(default_config())
assert capstone.description()["parameters"] == 328128
total = 2127532032 + 241734912 + 1605632
assert total == 2370872576
result = {
    "reviewer_task": "/root/natural_factual_20_2",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "source_sha256": hashlib.sha256(section.encode()).hexdigest(),
    "records": records,
    "capstone_parameters": capstone.description()["parameters"],
    "combined_parameters": total,
    "combined_billions": total / 1000000000,
    "weights_only_scope": "arithmetic only; excludes tensors during execution, headers, buffers, ASR and LoRA in the chapter table",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
