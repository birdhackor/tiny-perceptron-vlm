"""核對原始證據是否仍有效，並驗證目前比較 API 的敘述。"""

import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_14-2/technical-14-2-original-01.json"
assert sha(original_path) == "07369338c154496236a50ea856c9cd7073f5c8629538f3c0fc1cccf5dfc04dd0"
original = json.loads(original_path.read_text())
old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/14.2.md"
current_path = ROOT / "docs/course-review-20261019/freeze-05/sources/14.2.md"
old = old_path.read_bytes()
current = current_path.read_bytes()
assert hashlib.sha256(current).hexdigest() == "db1fb689dae4b4e4c550c3583938c54383efe77251715cd73eeae595b065037b"
old_text, current_text = old.decode(), current.decode()
old_phrase = "逐位元比較的 `torch.equal`"
current_phrase = "要求每個元素完全相等的 `torch.equal`"
assert old_text.count(old_phrase) == 1
assert old_text.replace(old_phrase, current_phrase) == current_text
old_fences = re.findall(r"```python\n(.*?)```", old_text, re.S)
current_fences = re.findall(r"```python\n(.*?)```", current_text, re.S)
assert old_fences == current_fences

version_checks = []
for item in original["artifacts"] + [s for s in original["sources"] if s["kind"] == "repository_code"]:
    actual = sha(ROOT / item["path"])
    assert actual == item["sha256"], "沿用證據內容已改變，必須重新核對"
    version_checks.append({"path": item["path"], "expected_sha256": item["sha256"], "current_sha256": actual, "unchanged": True})

old_code = ROOT / "docs/technical-reviews/artifacts/review-20261019-14-2/verification.py"
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    runpy.run_path(str(old_code), run_name="__main__")
rerun = json.loads(captured.getvalue())

left = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
right = left.clone()
right[0] = torch.nextafter(right[0], torch.tensor(float("inf"), dtype=torch.float32))
cases = {
    "same_values": torch.equal(left, left.clone()),
    "different_shape": torch.equal(left, left.reshape(1, 3)),
    "one_ulp_difference_equal": torch.equal(left, right),
    "one_ulp_difference_allclose": torch.allclose(left, right),
    "one_ulp_left": left.tolist(),
    "one_ulp_right": right.tolist(),
    "signed_zero_equal": torch.equal(torch.tensor([0.0]), torch.tensor([-0.0])),
    "different_dtype_same_values_equal": torch.equal(torch.tensor([1], dtype=torch.int32), torch.tensor([1.0], dtype=torch.float32)),
}
assert cases["same_values"] and not cases["different_shape"]
assert not cases["one_ulp_difference_equal"] and cases["one_ulp_difference_allclose"]
assert cases["signed_zero_equal"] and cases["different_dtype_same_values_equal"]

values = torch.arange(1, 97, dtype=torch.float32).reshape(-1, 3)
mean_square = values.square().mean(-1, keepdim=True) + 1e-5
multiplied = values * mean_square.rsqrt()
divided = values / mean_square.sqrt()
rounding = {"input": values.tolist(), "different_elements": int((multiplied != divided).sum()), "maximum_absolute_difference": float((multiplied - divided).abs().max()), "allclose": torch.allclose(multiplied, divided)}
assert rounding["different_elements"] > 0 and rounding["allclose"]

url = "https://raw.githubusercontent.com/pytorch/pytorch/v2.12.0/torch/_torch_docs.py"
command = ["curl", "-fLsS", "--max-time", "30", url]
download = subprocess.run(command, capture_output=True, check=True).stdout
assert hashlib.sha256(download).hexdigest() == "a580b2e1cf873f8d76371120b8a81c8ff40f848ac0e991d0b007fd062801ce52"
text = download.decode()
start = text.index("    torch.equal,")
end = text.find("\nadd_docstr(", start)
block = text[start:end]
assert "same size and elements" in block
assert "differentiate between the data types" in block
assert "same size and elements" in torch.equal.__doc__

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", "")},
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path)},
    "current_source": {"path": str(current_path.relative_to(ROOT)), "sha256": sha(current_path)},
    "difference": "".join(difflib.unified_diff(old_text.splitlines(True), current_text.splitlines(True), fromfile=str(old_path.relative_to(ROOT)), tofile=str(current_path.relative_to(ROOT)))),
    "only_expected_phrase_changed": True,
    "python_fences_identical": True,
    "version_checks": version_checks,
    "original_verifier_rerun": {"code_path": str(old_code.relative_to(ROOT)), "code_sha256": sha(old_code), "result": rerun},
    "comparison_cases": cases,
    "rounding_difference_demonstration": rounding,
    "official_source": {"url": url, "version": "PyTorch v2.12.0 tag; also checked installed PyTorch 2.14.1 docstring", "accessed_on": "2026-10-20", "command_argv": command, "downloaded_sha256": hashlib.sha256(download).hexdigest(), "locator": "torch.equal add_docstr", "excerpt": "True if two tensors have the same size and elements.", "inspection_note": "原始文件定義大小與元素值相同，且明記不區分 dtype。只保存必要短摘錄；本次實跑一個 ULP 差異、形狀差異與相同值例子。"},
    "result": "唯一改句符合比較 API 契約；既有數值與軟體驗證重跑成功，所有沿用檔案的 SHA-256 相同。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
