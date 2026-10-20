"""核對候選重排、機率、平手規則與位置標籤的確定性算例。"""

import hashlib
import difflib
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
import io
import itertools
import json
import math
import platform
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).with_name("verification-current-01.json")
torch.set_num_threads(1)
raw = (ROOT / "docs/course-review-20261019/freeze-05/sources/9.12.md").read_bytes()
current = (ROOT / "course/chapters/09.md").read_bytes()
section = re.search(rb"(?ms)^## 9\.12 .*?(?=^## |\Z)", current).group()
assert raw == section
namespace = {"__name__": "__main__"}
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(code, "frozen_section", "exec"), namespace)
    exec('print(backward[0])', namespace)

scores = namespace["scores"]
names = list(scores)
assert names == ["查卡片進度", "更改地址", "以上皆不適合"]
assert namespace["backward"][0] == "以上皆不適合"
base = torch.tensor([scores[n] for n in names], dtype=torch.float64)
probability = base.softmax(-1)
manual_denominator = sum(math.exp(scores[n]) for n in names)
manual = [math.exp(scores[n]) / manual_denominator for n in names]
assert max(abs(a - b) for a, b in zip(probability.tolist(), manual)) < 1e-15
assert [round(p, 3) for p in probability.tolist()] == [0.844, 0.042, 0.114]
loss = F.cross_entropy(base, torch.tensor(0))
assert abs(loss.item() + math.log(manual[0])) < 1e-15

permutations = []
for order in itertools.permutations(names):
    logits = torch.tensor([scores[n] for n in order], dtype=torch.float64)
    p = logits.softmax(-1)
    new_target = order.index(names[0])
    mapped = {n: p[i].item() for i, n in enumerate(order)}
    assert max(abs(mapped[n] - probability[names.index(n)].item()) for n in names) < 1e-15
    corrected_loss = F.cross_entropy(logits, torch.tensor(new_target)).item()
    assert abs(corrected_loss - loss.item()) < 1e-15
    assert order[int(p.argmax())] == names[0]
    permutations.append({"order": order, "probabilities": p.tolist(), "target": new_target,
                         "corrected_loss": corrected_loss,
                         "unchanged_label_loss": F.cross_entropy(logits, torch.tensor(0)).item()})

tie_scores = {"甲": 1.0, "乙": 1.0, "丙": -1.0}
fixed_ids = {"甲": "b", "乙": "a", "丙": "c"}
ties = []
for order in [list(tie_scores), list(reversed(tie_scores))]:
    logits = torch.tensor([tie_scores[n] for n in order], dtype=torch.float64)
    naive = order[int(logits.argmax())]
    maximum = max(tie_scores.values())
    stable = min((n for n in order if tie_scores[n] == maximum), key=fixed_ids.get)
    ties.append({"order": order, "argmax_choice": naive, "fixed_id_choice": stable})
assert ties[0]["argmax_choice"] != ties[1]["argmax_choice"]
assert ties[0]["fixed_id_choice"] == ties[1]["fixed_id_choice"] == "乙"

added = torch.cat([base, torch.tensor([-0.5], dtype=torch.float64)]).softmax(-1)
removed = base[[0, 2]].softmax(-1)
duplicate = torch.cat([base, base[:1]]).softmax(-1)
missing = torch.tensor([2.0, -1.0], dtype=torch.float64).softmax(-1)
assert added[0] < probability[0] < removed[0]
assert duplicate[0] == duplicate[-1] and duplicate[0] < probability[0]
assert missing.numel() == 2 and abs(missing.sum().item() - 1) < 1e-15

previous = re.search(rb"(?ms)^## 9\.11 .*?(?=^## |\Z)", current).group()
previous_code = re.search(r"```python\n(.*?)```", previous.decode(), re.S).group(1)
previous_namespace = {"__name__": "__main__"}
previous_output = io.StringIO()
with redirect_stdout(previous_output):
    exec(compile(previous_code, "previous_section", "exec"), previous_namespace)
assert previous_namespace["judgment"]["candidates"][0] == names[0]


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


source_url = "https://docs.python.org/3.13/library/functions.html#round"
source_command = ["curl", "--fail", "--silent", "--show-error", "--location", source_url]
source_html = subprocess.check_output(source_command)
page = PageText()
page.feed(source_html.decode())
source_text = " ".join(" ".join(page.parts).split())
source_quotes = [
    "Return number rounded to ndigits precision after the decimal point.",
    "rounding is done toward the even choice",
]
assert all(quote in source_text for quote in source_quotes)
rounding_cases = [{"input": x, "input_hex": x.hex(), "ndigits": 3, "observed": round(x, 3)}
                  for x in [0.0625, 0.1875]]
assert [case["observed"] for case in rounding_cases] == [0.062, 0.188]
assert len(namespace["p"].tolist()) == 3
assert all(isinstance(x, float) for x in namespace["p"].tolist())
old_section = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.12.md").read_bytes()
old_code = re.search(r"```python\n(.*?)```", old_section.decode(), re.S).group(1)
assert old_code == code
assert old_section.replace("把機率四捨五入到三位方便閱讀".encode(),
                           "把機率取到小數三位方便閱讀".encode()) == raw
original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-12/technical-9-12-original-01.json"
original_report = json.loads(original_path.read_bytes())
assert hashlib.sha256(original_path.read_bytes()).hexdigest() == "9ea40e982ad750602281855202d04ac8c6304c5310b03a93e3e887355dd82f37"
assert not original_report["issues"]
original_artifacts = []
for artifact in original_report["artifacts"]:
    observed_sha = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
    assert observed_sha == artifact["sha256"]
    original_artifacts.append({"path": artifact["path"], "sha256": observed_sha, "unchanged": True})
frozen_previous = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.11.md").read_bytes()
frozen_previous_code = re.search(r"```python\n(.*?)```", frozen_previous.decode(), re.S).group(1)
assert frozen_previous_code == previous_code

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-12/verify-claims-current-01.py",
    "official_round_source": {
        "url": source_url,
        "command_argv": source_command,
        "version": "Python 3.13 官方版本化文件；本機 Python 3.13.16",
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "retrieved_html_sha256": hashlib.sha256(source_html).hexdigest(),
        "locator": "Built-in Functions: round(number, ndigits=None)",
        "short_excerpts": source_quotes,
        "inspection_note": "已讀 ndigits 的小數精度、最接近倍數、中點取偶數及二進位浮點限制；原文取到小數三位與契約一致。這不代表截斷，也不承諾補齊尾端零。",
    },
    "rounding_cases": rounding_cases,
    "original_artifacts": original_artifacts,
    "original_issues": original_report["issues"],
    "source_diff": list(difflib.unified_diff(old_section.decode().splitlines(), raw.decode().splitlines())),
    "python_fence_unchanged": True,
    "previous_python_fence_unchanged": True,
    "changed_wording_verified": True,
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu",
                    "platform": platform.platform(), "threads": "1", "seed": "不使用亂數"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "current_section_equals_freeze": True,
    "previous_section_sha256": hashlib.sha256(previous).hexdigest(),
    "previous_section_output": previous_output.getvalue(),
    "exercise_output": captured.getvalue(),
    "dtype": "torch.float64；原文另以預設 torch.float32 實跑",
    "manual_denominator": manual_denominator,
    "manual_probabilities": manual,
    "float64_probabilities": probability.tolist(),
    "permutations": permutations,
    "tie_cases": ties,
    "candidate_set_changes": {
        "base_probability": probability[0].item(),
        "after_addition": added[0].item(),
        "after_removal": removed[0].item(),
        "after_duplicate": duplicate.tolist(),
        "no_none_class": missing.tolist(),
    },
    "result": "全部斷言通過；未訓練或評估任何模型的語意品質。",
}
with DESTINATION.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
