"""核對 API 修訂、凍結原文差異與沿用證據的完整性。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import fingerprint
import torch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    return digest((ROOT / path).read_bytes())


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


original_path = "docs/course-review-20261019/reviews/rv1019_technical_5-11/technical-5-11-original-01.json"
original_hash = "a250fa80a72a585a59e6cc2cd2180ce6ed74786aaacbfccac0c1617d5593d6de"
assert file_digest(original_path) == original_hash
original = json.loads((ROOT / original_path).read_bytes())
old_path = "docs/course-review-20261019/freeze-01/sources/5.11.md"
new_path = "docs/course-review-20261019/freeze-05/sources/5.11.md"
old = (ROOT / old_path).read_bytes()
new = (ROOT / new_path).read_bytes()
assert digest(old) == original["source_sha256"]
assert digest(new) == "b404bc1842d17819e7edfc44cb207258f6efb40e00c1c62a0a7c0a7024e95c25"
old_clause = "因為 `hashlib.sha256` 只接受 bytes"
new_clause = "因為 `hashlib.sha256` 不能直接接受字串"
assert old.count(old_clause.encode()) == 1
assert old.replace(old_clause.encode(), new_clause.encode()) == new
fences = lambda raw: re.findall(rb"```python\n(.*?)```", raw, flags=re.S)
assert fences(old) == fences(new)
chapter = (ROOT / "course/chapters/05.md").read_bytes()
section = re.search(rb"(?ms)^## 5\.11 .*?(?=^## |\Z)", chapter).group()
assert section == new
freeze_path = "docs/course-review-20261019/freeze-06.json"
assert file_digest(freeze_path) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"

artifacts = []
for item in original["artifacts"]:
    actual = file_digest(item["path"])
    assert actual == item["sha256"]
    artifacts.append({"path": item["path"], "expected_sha256": item["sha256"], "actual_sha256": actual, "matches": True})
repo_sources = []
for item in original["sources"]:
    if item["kind"] != "repository_code":
        continue
    actual = file_digest(item["path"])
    unchanged = actual == item["sha256"]
    if item["path"] == "course/training.md":
        assert actual == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    else:
        assert unchanged
    repo_sources.append({"path": item["path"], "old_sha256": item["sha256"], "current_sha256": actual, "unchanged": unchanged})
old_verification = json.loads((DEST / "verification.json").read_bytes())
for path, recorded in old_verification["repository_sha256"].items():
    if path != "course/training.md":
        assert file_digest(path) == recorded

api = {}
text = "貓 看狗"
for value in [text, text.encode(), bytearray(text.encode()), memoryview(text.encode())]:
    try:
        api[type(value).__name__] = {"accepted": True, "hexdigest": hashlib.sha256(value).hexdigest()}
    except TypeError as error:
        api[type(value).__name__] = {"accepted": False, "exception": type(error).__name__, "message": str(error)}
assert not api["str"]["accepted"]
assert api["str"]["exception"] == "TypeError"
assert all(api[t]["accepted"] for t in ["bytes", "bytearray", "memoryview"])
assert len({api[t]["hexdigest"] for t in ["bytes", "bytearray", "memoryview"]}) == 1
assert fingerprint("貓  看狗") == api["bytes"]["hexdigest"]

source_url = "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/library/hashlib.rst"
curl_command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", source_url]
raw_source = subprocess.check_output(curl_command)
old_receipts = json.loads((DEST / "source-receipts.json").read_bytes())
old_source = next(s for s in old_receipts["sources"] if s["id"] == "python-hashlib")
assert digest(raw_source) == old_source["full_source_sha256"]
assert b"bytes-like objects <bytes-like object>" in raw_source
source_receipt = {"url": source_url, "command_argv": curl_command, "version": "CPython v3.13.16", "accessed_on": "2026-10-20", "full_source_sha256": digest(raw_source), "matches_original_receipt": True, "locator": "Hash algorithms; Constructors", "excerpt": "bytes-like objects <bytes-like object>", "inspection_note": "官方契約接收 bytes-like，不是 str；與本次 TypeError 及成功編碼的對照一致。全文僅保存在記憶體。"}

training = (ROOT / "course/training.md").read_text()
training_section = re.search(r"(?ms)^## T\.4 .*?(?=^## |\Z)", training).group()
real_text_command = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
assert real_text_command in training_section
previous_training = subprocess.check_output(["git", "show", "5ef56bc756f3:course/training.md"], cwd=ROOT, text=True)
assert digest(previous_training.encode()) == next(s["sha256"] for s in original["sources"] if s.get("path") == "course/training.md")
previous_training_section = re.search(r"(?ms)^## T\.4 .*?(?=^## |\Z)", previous_training).group()
assert training_section == previous_training_section
cli = command([sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", "real_text"])
assert cli["exit_code"] == 0
assert set(cli["stdout"].splitlines()) == {"assets/training/tinystories-v1.tar.gz", "assets/training/chinese-poetry-v1.tar.gz"}
source_diff = command(["git", "diff", "--no-index", old_path, new_path])
assert source_diff["exit_code"] == 1
training_diff = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
assert training_diff["exit_code"] == 0
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-11/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "repo_revision": revision,
    "source_change": {"before": old_clause, "after": new_clause, "only_change": True, "python_fences_identical": True, "current_chapter_equals_freeze": True, "source_sha256": digest(new), "diff": source_diff},
    "original_report": {"path": original_path, "sha256": original_hash},
    "retained_artifacts": artifacts,
    "repository_sources": repo_sources,
    "api_cases": api,
    "official_source": source_receipt,
    "training_dependency": {"path": "course/training.md", "current_sha256": file_digest("course/training.md"), "supporting_section_identical": True, "command_found": real_text_command, "diff": training_diff, "conclusion": "差異位於裝置使用說明、多模態三題與量化單題的限制；本頁引用的真實文字實驗命令及完整補充區沒有變動，既有命令主張仍成立。"},
    "cli": cli,
    "retained_claim_ids": [c["id"] for c in original["claims"] if c["id"] != "c08"],
    "result": "字串直接傳入 sha256 會拋出 TypeError，UTF-8 編碼後的 bytes 成功；新版敘述成立。其他主張的依據與沿用 artifact hash 未變，真實文字實驗的連結命令仍存在。"
}
with (DEST / "recheck-current-01.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
