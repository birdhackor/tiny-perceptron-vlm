"""核對凍結原文的差異、沿用證據的完整性與五字示例的適用範圍。"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-5-2"
ORIGINAL_REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-2/technical-5-2-original-01.json"
OLD_SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/5.2.md"
NEW_SOURCE = ROOT / "docs/course-review-20261019/freeze-05/sources/5.2.md"
NEW_FREEZE = ROOT / "docs/course-review-20261019/freeze-05.json"
RUN = Path(sys.argv[1]).resolve()
assert RUN.is_relative_to(ROOT / "outputs/reviewer-tools/runs")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(ORIGINAL_REPORT) == "3ca285813941fcafaca299f183aa2e5a6757682b6cb44c8351bf89d42a7aa73f"
assert sha(OLD_SOURCE) == "3edb710d4431a5ad8223cab698f58e8248dfe4fb6f9b0e3641235a323fca57b4"
assert sha(NEW_SOURCE) == "e4c7eef3b10fa3acc18efbd7d7334cb849ba2085b42775d9d6eb9029347bb306"
assert sha(NEW_FREEZE) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
original = json.loads(ORIGINAL_REPORT.read_bytes())
old = OLD_SOURCE.read_bytes()
new = NEW_SOURCE.read_bytes()
before = "本章例子一個字就是一個 token"
after = "本節的五字示例裡一個字就是一個 token"
assert old.count(before.encode()) == 1
assert old.replace(before.encode(), after.encode()) == new

difference_command = ["git", "diff", "--no-index", str(OLD_SOURCE.relative_to(ROOT)), str(NEW_SOURCE.relative_to(ROOT))]
difference = subprocess.run(difference_command, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
assert difference.returncode == 1

integrity = []
for category in ("sources", "artifacts"):
    for item in original[category]:
        if "path" not in item:
            continue
        current_hash = sha(ROOT / item["path"])
        assert current_hash == item["sha256"]
        integrity.append({"category": category, "id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": current_hash, "unchanged": True})
assert sha(BASE / "self-check.txt") == "c91a557c0b0fdf7e91286a10d255746e931eb018a222ce644b6537961474fbb5"

section = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.2" --execute',
    "extraction": json.loads((RUN / "extraction.json").read_bytes()),
    "execution": json.loads((RUN / "execution.json").read_bytes()),
    "environment": json.loads((RUN / "environment.json").read_bytes()),
    "stdout": (RUN / "stdout.txt").read_text(),
    "stderr": (RUN / "stderr.txt").read_text(),
    "copied_file_sha256": {name: sha(RUN / name) for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt")},
}
assert (RUN / "section.md").read_bytes() == new
assert section["extraction"]["source_sha256"] == sha(NEW_SOURCE)
assert section["execution"]["exit_code"] == 0
assert section["environment"]["guard_events"] == []
assert section["extraction"]["figure_sha256"] == {}
old_run = json.loads((BASE / "section-run.json").read_bytes())
assert [fence["sha256"] for fence in section["extraction"]["python_fences"]] == [fence["sha256"] for fence in old_run["extraction"]["python_fences"]]
assert section["stdout"] == old_run["stdout"]
assert section["environment"]["torch"] == old_run["environment"]["torch"]
assert section["environment"]["torch_git_version"] == old_run["environment"]["torch_git_version"]

supplement_command = [str(ROOT / ".venv/bin/python"), str(BASE / "verify_facts.py")]
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
supplement_result = subprocess.run(supplement_command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60, check=True)
supplement = json.loads(supplement_result.stdout)
old_supplement = json.loads((BASE / "verification.json").read_bytes())
for key in ("original", "changed", "restored_equals_original", "all_ignored", "reshape_order_preserved", "excerpt_matches_library_ast", "unequal_parts", "optimizer_comparison", "experiment_evaluation", "tokenization"):
    assert supplement[key] == old_supplement[key]
assert supplement["training_dry_run"]["exit_code"] == 0
assert not supplement["training_dry_run"]["weights_updated"]

candidate_context = "五個候選沿用句號、狗、看、貓、逗號"
assert candidate_context.encode() in new
characters = ["。", "狗", "看", "貓", "，"]
char_to_id = {character: index for index, character in enumerate(characters)}
assert len(char_to_id) == 5
assert all(len(character) == 1 for character in characters)
example = "狗看貓，"
example_ids = [char_to_id[character] for character in example]
assert example_ids == [1, 2, 3, 4]
assert len(example_ids) == len(example)
assert len(supplement["tokenization"]["single_character"]["byte_ids"]) == 3

paragraph = next(part for part in new.decode().split("\n\n") if part.startswith("為什麼要分成兩步？"))
claim_checks = []
for claim in original["claims"]:
    claim_checks.append({"id": claim["id"], "kind": claim["kind"], "text_changed": claim["id"] == "chapter_character_token_scope", "status": "verified", "basis": "新版把對應限於五字示例；逐字對應與目標ID已核對。" if claim["id"] == "chapter_character_token_scope" else "全文差異確認原句未改；既有來源與證據SHA-256未變；原始程式與補充算例本回合重跑一致。"})

result = {
    "command_argv": [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), str(RUN)],
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_git_version": str(torch.version.git_version)},
    "source_sha256": sha(NEW_SOURCE),
    "source_freeze": {"path": str(NEW_FREEZE.relative_to(ROOT)), "sha256": sha(NEW_FREEZE)},
    "original_report": {"path": str(ORIGINAL_REPORT.relative_to(ROOT)), "sha256": sha(ORIGINAL_REPORT)},
    "diff": {"command_argv": difference_command, "exit_code": difference.returncode, "stdout": difference.stdout, "stderr": difference.stderr, "only_changed_phrase": {"before": before, "after": after}, "changed_paragraph": paragraph},
    "reused_evidence_integrity": integrity,
    "section_execution": section,
    "supplement_execution": {"command_argv": supplement_command, "exit_code": supplement_result.returncode, "stderr": supplement_result.stderr, "parsed_stdout": supplement},
    "five_character_mapping": {"candidate_context": candidate_context, "characters": characters, "ids": list(char_to_id.values()), "characters_per_candidate": [len(character) for character in characters], "example": example, "example_ids": example_ids, "scope": "只核對本節手工五字字表；未概括ByteTokenizer。"},
    "claim_checks": claim_checks,
    "independence": "只讀自己的原報告與證據、凍結稿、必要實作與本次section_facts新產生的執行目錄；未讀其他角色或舊輪報告。",
    "result": "差異只有一處措辭；所有原始證據與repo來源未變；新版三段程式與補充核對成功，全部主張已核實。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
