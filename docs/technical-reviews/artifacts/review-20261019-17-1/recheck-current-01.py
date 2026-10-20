"""核對凍結來源、沿用證據與 Tensor.to 的副本契約。"""

import difflib
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = Path(__file__).resolve().parent
ORIGINAL_REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_17-1/technical-17-1-original-01.json"
OLD_SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/17.1.md"
NEW_SOURCE = ROOT / "docs/course-review-20261019/freeze-05/sources/17.1.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads(ORIGINAL_REPORT.read_bytes())
assert sha(ORIGINAL_REPORT) == "dff9a55893330ebc926cf4fd725288781b4bf84ad15e0e52023cd5daa9a7f231"
assert sha(NEW_SOURCE) == "4e4ec2270b750dafed249f11ec9e586fb60e7d8bc8be86b610540f4589ad2116"
retained = []
for category in ("sources", "artifacts"):
    for item in report[category]:
        if "path" not in item:
            continue
        actual = sha(ROOT / item["path"])
        assert actual == item["sha256"], item["path"]
        retained.append({"category": category, "id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})

chapter = (ROOT / "course/chapters/17.md").read_bytes()
start = chapter.index(b"## 17.1 ")
end = chapter.index(b"\n## ", start) + 1
assert chapter[start:end] == NEW_SOURCE.read_bytes()
old_lines = OLD_SOURCE.read_text().splitlines(keepends=True)
new_lines = NEW_SOURCE.read_text().splitlines(keepends=True)
matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
changes = []
for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
    if tag != "equal":
        changes.append({"kind": tag, "old_text": "".join(old_lines[old_start:old_end]), "new_text": "".join(new_lines[new_start:new_end])})
assert len(changes) == 1

commit = str(torch.version.git_version)
url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_tensor_docs.py"
command = ["curl", "-fLsS", "--max-time", "30", url]
fetched = subprocess.run(command, check=True, capture_output=True)
text = fetched.stdout.decode("utf-8")
needle = "If the ``self`` Tensor already"
index = text.index(needle)
end = text.index(".. note::", index)
excerpt = text[index:end].strip()
assert "then ``self`` is returned" in excerpt
assert "returned tensor is a copy" in excerpt
assert "copy=False" in torch.Tensor.to.__doc__

torch.set_default_device("cpu")
weight = torch.zeros(64, 32)
rows = []
for dtype in (torch.float32, torch.float16, torch.int8):
    converted = weight.to(dtype)
    same_type = dtype == weight.dtype
    row = {
        "dtype": str(converted.dtype),
        "device": str(converted.device),
        "is_original": converted is weight,
        "shares_storage": converted.data_ptr() == weight.data_ptr(),
        "numel": converted.numel(),
        "element_bytes": converted.element_size(),
        "tensor_bytes": converted.numel() * converted.element_size(),
    }
    assert row["is_original"] == same_type
    assert row["shares_storage"] == same_type
    rows.append(row)
assert [r["tensor_bytes"] for r in rows] == [8192, 4096, 2048]
forced = weight.to(torch.float32, copy=True)
assert forced is not weight
assert forced.data_ptr() != weight.data_ptr()

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-1/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": commit, "device": "cpu"},
    "original_report": {"path": ORIGINAL_REPORT.relative_to(ROOT).as_posix(), "sha256": sha(ORIGINAL_REPORT)},
    "source_freeze": {"path": "docs/course-review-20261019/freeze-05.json", "sha256": sha(ROOT / "docs/course-review-20261019/freeze-05.json")},
    "source_sha256": sha(NEW_SOURCE),
    "current_chapter_section_equals_freeze": True,
    "unchanged_inputs": retained,
    "changes": changes,
    "official_original": {
        "url": url,
        "version": f"PyTorch {torch.__version__}, commit {commit}",
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command": command,
        "exit_code": fetched.returncode,
        "full_response_sha256": hashlib.sha256(fetched.stdout).hexdigest(),
        "full_response_bytes": len(fetched.stdout),
        "locator": "Tensor.to note: same dtype/device returns self; otherwise a copy",
        "excerpt": excerpt,
    },
    "dtype_conversion": rows,
    "copy_true_control": {"is_original": forced is weight, "shares_storage": forced.data_ptr() == weight.data_ptr()},
    "result": "修訂句符合預設 copy=False 的官方契約；不同 dtype 建立副本，相同 dtype/device 沿用原 tensor；容量與 numel/element_size 的敘述相符。",
    "scope": "只測 CPU 與本節 to(dtype) 呼叫；copy=True 的控制例說明額外參數可改變契約，未把預設呼叫的說明外推到全部 to 變體。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
