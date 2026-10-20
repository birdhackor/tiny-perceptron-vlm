"""核對凍結原文差異、既存證據與共用表示算例。"""

import ast
import difflib
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


artifact_dir = Path(__file__).resolve().parent
report_path = Path("docs/course-review-20261019/reviews/rv1019_technical_3-3/technical-3-3-original-01.json")
assert sha(report_path) == "88f10548f51ad7a226cd0ec722eae90ab05cbd42d77691f4ad16cbdb7702276b"
original = json.loads(report_path.read_text())
old_source = Path("docs/course-review-20261019/freeze-04/sources/3.3.md")
new_source = Path("docs/course-review-20261019/freeze-07/sources/3.3.md")
freeze_path = Path("docs/course-review-20261019/freeze-07.json")
old_raw, new_raw = old_source.read_bytes(), new_source.read_bytes()
assert sha(old_source) == original["source_sha256"]
assert sha(new_source) == "d8b342973cd58fec1ab85bcf8d38ed9b132128b74e4f7ac4190014f05fd59e30"
assert sha(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"

old_clause = "各位置向量長度相同時，跟自己的分數也最高（或並列最高）。"
new_clause = "本例 A、B 跟自己都得 1、彼此得 0，所以最高分都落在自己。"
assert old_raw.count(old_clause.encode()) == 1
assert old_raw.replace(old_clause.encode(), new_clause.encode()) == new_raw

lesson = original["lesson_id"]
chapter_path = Path(original["source"].split("#")[0])
chapter = chapter_path.read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
selected = [index for index, header in enumerate(headers) if header[0].startswith(("## " + lesson + " ").encode())]
assert len(selected) == 1
index = selected[0]
end = headers[index + 1].start() if index + 1 < len(headers) else len(chapter)
assert chapter[headers[index].start():end] == new_raw

old_blocks = re.findall(rb"```python\n(.*?)```", old_raw, re.S)
new_blocks = re.findall(rb"```python\n(.*?)```", new_raw, re.S)
assert old_blocks == new_blocks
assert len(new_blocks) == 2

literal_inputs = {}
for block in new_blocks:
    for node in ast.parse(block.decode()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "torch" and function.attr == "tensor":
                literal_inputs[node.targets[0].id] = ast.literal_eval(node.value.args[0])

has_rows = literal_inputs["has"]
manual = [[sum(a * b for a, b in zip(left, right, strict=True)) for right in has_rows] for left in has_rows]
has = torch.tensor(has_rows, dtype=torch.float32, device="cpu")
scores = has @ has.T
assert manual == scores.tolist() == [[1.0, 0.0], [0.0, 1.0]]
assert scores.diag().tolist() == [1.0, 1.0]
assert [scores[0, 1].item(), scores[1, 0].item()] == [0.0, 0.0]
assert scores.argmax(dim=-1).tolist() == [0, 1]

prior_file_checks = []
for artifact in original["artifacts"]:
    actual = sha(artifact["path"])
    assert actual == artifact["sha256"]
    prior_file_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
prior_check = artifact_dir / "self-check.txt"
assert sha(prior_check) == "e76f03efecb1cf1f25a67666e139e3998b439593547b4bd4cc15a0b545ab86b5"
prior_file_checks.append({"path": str(prior_check.relative_to(Path.cwd())), "sha256": sha(prior_check), "unchanged": True})

current_run = json.loads((artifact_dir / "section-run-current-01.json").read_text())
extraction = json.loads(current_run["files"]["extraction.json"]["utf8"])
environment = json.loads(current_run["files"]["environment.json"]["utf8"])
execution = json.loads(current_run["files"]["execution.json"]["utf8"])
assert extraction["source_sha256"] == sha(new_source)
assert execution["exit_code"] == 0
assert len(extraction["python_fences"]) == len(new_blocks)
assert not extraction["svg_references"]
for item, block in zip(extraction["python_fences"], new_blocks, strict=True):
    assert item["sha256"] == hashlib.sha256(block).hexdigest()
    assert current_run["files"][item["file"]]["utf8"].encode() == block

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-3/verify-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": str(scores.device), "dtype": str(scores.dtype), "platform": platform.platform()},
    "source_sha256": sha(new_source),
    "source_freeze_sha256": sha(freeze_path),
    "current_section_matches_freeze": True,
    "only_body_change": {"before": old_clause, "after": new_clause},
    "diff": "".join(difflib.unified_diff(old_raw.decode().splitlines(keepends=True), new_raw.decode().splitlines(keepends=True), fromfile=str(old_source), tofile=str(new_source))),
    "original_issue_ids": [issue["claim_id"] for issue in original["issues"]],
    "unchanged_fence_sha256": [hashlib.sha256(block).hexdigest() for block in new_blocks],
    "current_section_execution": {"exit_code": execution["exit_code"], "source_sha256": extraction["source_sha256"], "environment": environment},
    "literal_has": has_rows,
    "manual_scores": manual,
    "torch_scores": scores.tolist(),
    "self_scores": scores.diag().tolist(),
    "cross_scores": [scores[0, 1].item(), scores[1, 0].item()],
    "argmax": scores.argmax(dim=-1).tolist(),
    "calculation": "A·A=1×1+0×0=1；B·B=0×0+1×1=1；A·B=1×0+0×1=0；B·A=0×1+1×0=0；每列 1 大於 0，所以各自唯一最高分在自己。",
    "prior_file_checks": prior_file_checks,
    "result": "全部斷言通過；現行小節與凍結原文相同，程式未變，舊證據未變，新增的自身分數敘述與手算及 CPU 實跑一致。",
}, ensure_ascii=False, indent=2))
