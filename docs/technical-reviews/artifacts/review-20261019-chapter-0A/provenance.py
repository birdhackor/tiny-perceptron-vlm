"""核對實驗程式版本與正文引用的生成紀錄。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TARGET = "scripts/course_experiments/applications.py"
record_path = ROOT / "docs/course-experiments/results/rag.json"
record = json.loads(record_path.read_bytes())
expected_hash = record["code_sha256"][TARGET]


def show(revision):
    command = ["git", "show", f"{revision}:{TARGET}"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    return command, result


command, pinned = show(record["revision"])
history_command = ["git", "rev-list", "--all", "--", TARGET]
history = subprocess.run(history_command, cwd=ROOT, capture_output=True, check=True)
candidates = []
matched_functions = []
for revision in history.stdout.decode().splitlines():
    candidate_command, candidate = show(revision)
    assert candidate.returncode == 0
    sha = hashlib.sha256(candidate.stdout).hexdigest()
    candidates.append({"command": candidate_command, "sha256": sha, "matches_recorded_code": sha == expected_hash})
    if sha == expected_hash:
        source = candidate.stdout.decode()
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in {"_sample", "run_rag"}:
                matched_functions.append({"revision": revision, "function": node.name,
                    "source": ast.get_source_segment(source, node)})

freeze = json.loads((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes())
rows = record["results"]["samples"]
body_examples = []
for page in freeze["pages"]:
    if page["source"] != "course/chapters/0A.md" or page["kind"] != "lesson":
        continue
    raw = (ROOT / page["snapshot"]).read_text()
    body = re.sub(r"<details>.*?</details>", "", raw, flags=re.S)
    for paragraph in body.split("\n\n"):
        if "模型卻回答" in paragraph:
            body_examples.append({"page_id": page["page_id"], "snapshot": page["snapshot"],
                "snapshot_sha256": hashlib.sha256(raw.encode()).hexdigest(), "paragraph": paragraph})
                
sample = next(row for row in rows if row["family"] == "K003" and row["mode"] == "correct_with_distractor")
assert sample["generation"]["samples"][0]["generated"] == "C9[D9]"
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-0A/provenance.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU"},
    "recorded_code_sha256": expected_hash,
    "pinned_revision_lookup": {"command": command, "returncode": pinned.returncode,
        "stderr": pinned.stderr.decode(), "stdout_sha256": hashlib.sha256(pinned.stdout).hexdigest()},
    "history_lookup": {"command": history_command, "returncode": history.returncode,
        "stdout": history.stdout.decode()},
    "reachable_versions": candidates,
    "matched_functions": matched_functions,
    "body_generated_examples": body_examples,
    "corresponding_saved_generation": {
        "family": sample["family"], "mode": sample["mode"], "documents": sample["documents"],
        "expected": sample["expected"], "generated": sample["generation"]["samples"][0]["generated"],
        "generated_ids": sample["generation"]["samples"][0]["generated_ids"],
        "citation_valid": sample["citation_valid"], "supported_by_cited_source": sample["supported_by_cited_source"]},
    "limits": ["只讀取本地 Git 與實驗紀錄；未重新執行訓練或模型生成。"]
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
