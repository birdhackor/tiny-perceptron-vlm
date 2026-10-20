"""保存複查收據；沿用相同證據，重新執行程式並核對日期適用性。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-A-6"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_A-6/technical-A-6-original-01.json"
RUN = (ROOT / sys.argv[1]).resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = json.loads(ORIGINAL.read_bytes())
assert digest(ORIGINAL) == "3c7fd52e6bc4fc3adec376e57e4aa6b556c76c0514af286a6930b5c1addfd4c2"
old_source = ROOT / "docs/course-review-20261019/freeze-01/sources/A.6.md"
new_source = ROOT / "docs/course-review-20261019/freeze-05/sources/A.6.md"
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert digest(old_source) == original["source_sha256"]
assert digest(new_source) == "a77707471cff7ea3716fd701da173b13bbf3a25318200c64aa42964aedec81c6"
assert digest(freeze_path) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
freeze = json.loads(freeze_path.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == original["lesson_id"])
assert page["source_sha256"] == digest(new_source)
assert page["figures_sha256"] == {}
chapter = (ROOT / "course/chapters/0A.md").read_bytes()
start = chapter.index(b"## A.6 ")
next_heading = chapter.find(b"\n## ", start)
live = chapter[start:next_heading + 1] if next_heading >= 0 else chapter[start:]
assert live == new_source.read_bytes(), "現行小節與凍結原文須逐位元組相同"
fences = lambda raw: re.findall(rb"```python\n(.*?)```", raw, flags=re.S)
assert fences(old_source.read_bytes()) == fences(new_source.read_bytes())

artifacts = []
for artifact in original["artifacts"]:
    actual = digest(ROOT / artifact["path"])
    assert actual == artifact["sha256"], "沿用證據的內容不得改變"
    artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual})
repo_sources = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(ROOT / source["path"])
        assert actual == source["sha256"], "來源若改動需重新核對其實作"
        repo_sources.append({"id": source["id"], "path": source["path"], "sha256": actual})

execution = json.loads((RUN / "execution.json").read_bytes())
environment = json.loads((RUN / "environment.json").read_bytes())
stdout = (RUN / "stdout.txt").read_bytes()
stderr = (RUN / "stderr.txt").read_bytes()
assert execution["source_sha256"] == digest(new_source)
assert execution["exit_code"] == 0
assert stdout == (BASE / "section-stdout.txt").read_bytes()
assert stderr == b""
assert (RUN / "fence-1.py").read_bytes() == (BASE / "section-fence-1.py").read_bytes()

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
verify_command = [str(ROOT / ".venv/bin/python"), str(BASE / "verify.py")]
verified = subprocess.run(verify_command, cwd=ROOT, env=env, capture_output=True, check=True)
assert verified.stderr == b""
assert verified.stdout == (BASE / "verification.json").read_bytes(), "重跑結果須與沿用證據逐位元組相同"
verification = json.loads(verified.stdout)
rag_path = ROOT / "docs/course-experiments/results/rag.json"
assert digest(rag_path) == verification["inputs"]["docs/course-experiments/results/rag.json"]

source_command = [str(ROOT / ".venv/bin/python"), str(BASE / "inspect_sources.py")]
source_run = subprocess.run(source_command, cwd=ROOT, env=env, capture_output=True, check=True)
assert source_run.stderr == b""
primary_sources = json.loads(source_run.stdout)

as_of = date(2026, 10, 20)
notices = [
    {"store": "bookshop", "issued": "2026-10-01", "valid_from": "2026-01-01", "valid_until": "2026-10-31", "address": "青街8號"},
    {"store": "bookshop", "issued": "2026-10-15", "valid_from": "2026-11-01", "valid_until": "2027-12-31", "address": "紅街9號"},
    {"store": "other_bookshop", "issued": "2026-10-20", "valid_from": "2026-10-20", "valid_until": "2027-12-31", "address": "紅街9號"},
]
applicable = [
    notice for notice in notices
    if notice["store"] == "bookshop"
    and date.fromisoformat(notice["valid_from"]) <= as_of <= date.fromisoformat(notice["valid_until"])
]
assert [notice["address"] for notice in applicable] == ["青街8號"]
assert notices[1]["issued"] > notices[0]["issued"]
assert date.fromisoformat(notices[1]["valid_from"]) > as_of

diff_command = ["git", "diff", "--no-index", str(old_source.relative_to(ROOT)), str(new_source.relative_to(ROOT))]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1 and not diff.stderr

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + sys.argv[1],
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "torch": environment["torch"]},
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(ORIGINAL)},
    "source_checks": {"old_source_sha256": digest(old_source), "source_sha256": digest(new_source), "freeze_sha256": digest(freeze_path), "current_section_equals_freeze": True, "python_fences_unchanged": True, "figures": page["figures_sha256"]},
    "diff": {"command_argv": diff_command, "exit_code": diff.returncode, "stdout": diff.stdout},
    "reused_artifacts": artifacts,
    "repository_sources": repo_sources,
    "section_run": {"origin_directory": str(RUN.relative_to(ROOT)), "execution": execution, "environment": environment, "stdout": stdout.decode(), "stderr": stderr.decode(), "code_sha256": digest(RUN / "fence-1.py"), "code_equals_original_artifact": True},
    "verification_replay": {"command_argv": verify_command, "exit_code": verified.returncode, "stdout_equals_saved_artifact": True, "saved_artifact": "docs/technical-reviews/artifacts/review-20261019-A-6/verification.json", "stdout_sha256": hashlib.sha256(verified.stdout).hexdigest(), "result": verification["result"], "unknown_recomputed": verification["unknown_recomputed"], "evaluation": verification["evaluation"], "code_provenance_limit": verification["code_provenance_limit"]},
    "primary_sources": {"command_argv": source_command, "exit_code": source_run.returncode, "result": primary_sources},
    "temporal_derivation": {"kind": "constructed_example", "as_of": as_of.isoformat(), "notices": notices, "applicable_addresses": [notice["address"] for notice in applicable], "details": "較新發布的同店公告可在未來才生效；別家店即使已生效仍不適用。依查詢對象與生效期間核對後，此例當日只有青街地址適用。這是透明的構造算例，不是實際店家調查或一般頻率估計。"},
    "result": "全部斷言成立；新版原文與現行小節相同，程式成功，沿用證據與重跑輸出雜湊相符，日期適用性構造算例符合修訂後說明。",
}, ensure_ascii=False, indent=2))
