"""核對引用文件的變動、既有證據完整性與當前算例執行結果。"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, identifier):
    found = re.search(rb"(?ms)^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)", raw)
    assert found is not None, "缺少指定小節"
    return found[0]


def main():
    previous_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-12/technical-5-12-current-01.json"
    previous_raw = previous_path.read_bytes()
    assert sha(previous_raw) == "e5a61d9a0307318f8d47dbfa84fb0beb1bc21e7737104ceba733ab2772f787ca"
    previous = json.loads(previous_raw)
    assert previous["issues"] == []
    original_path = ROOT / previous["original_report"]["path"]
    assert sha(original_path.read_bytes()) == previous["original_report"]["sha256"]
    artifact_checks = []
    for item in previous["artifacts"]:
        digest = sha((ROOT / item["path"]).read_bytes())
        assert digest == item["sha256"], "既有證據雜湊不符"
        artifact_checks.append({"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": digest, "unchanged": True})
    sources = []
    for item in previous["sources"]:
        if item["kind"] != "repository_code":
            continue
        digest = sha((ROOT / item["path"]).read_bytes())
        if item["path"] == "course/training.md":
            assert digest == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
        else:
            assert digest == item["sha256"], "引用來源出現非預期變動"
        sources.append({"source_id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": digest, "unchanged": digest == item["sha256"]})
    diff = execute(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
    old = execute(["git", "show", "ecf6dec71280:course/training.md"])["stdout"].encode()
    current = (ROOT / "course/training.md").read_bytes()
    committed = execute(["git", "show", "c6b9339d2595:course/training.md"])["stdout"].encode()
    assert sha(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert committed == current
    old_t4, current_t4 = section(old, "T.4"), section(current, "T.4")
    assert old_t4 == current_t4, "引用的文字配方有變動"
    recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
    assert recipe.encode() in current_t4
    before = json.loads((DIRECTORY / "recheck-output-current-01.json").read_bytes())
    context = section((ROOT / "course/chapters/02.md").read_bytes(), "2.5").split("多看字也有成本。".encode())[0]
    assert sha(context) == before["recheck"]["dependency_comparisons"]["context_definition"]["sha256"]
    freeze_path = ROOT / "docs/course-review-20261019/freeze-07.json"
    freeze_raw = freeze_path.read_bytes()
    assert sha(freeze_raw) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
    page = next(item for item in json.loads(freeze_raw)["pages"] if item["page_id"] == "5.12")
    frozen = (ROOT / page["snapshot"]).read_bytes()
    assert frozen == section((ROOT / "course/chapters/05.md").read_bytes(), "5.12")
    assert sha(frozen) == "832c6295bab044426dd82bff227820875caa016ee41f22dbba6fe748265339e4"
    assert page["figures_sha256"] == {}
    supplemental = execute([str(ROOT / ".venv/bin/python"), str(DIRECTORY / "verification.py")])
    payload = json.loads(supplemental["stdout"])
    baseline = json.loads((DIRECTORY / "verification-output.json").read_bytes())
    for key in ("fingerprints", "grams", "repeated_fragments", "empty_cases", "synonym_example", "different_answer_example", "family_grouping", "deduplication", "recorded_experiment", "recipe"):
        assert payload[key] == baseline[key], "重跑結果與原始證據不同"
    run = ROOT / "outputs/reviewer-tools/runs/20261020T152107508910Z"
    copied = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (run / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
    assert copied["execution.json"]["content"]["exit_code"] == 0
    assert copied["extraction.json"]["content"]["source_sha256"] == sha(frozen)
    assert copied["stdout.txt"]["content"] == baseline["original_stdout"]
    assert copied["stderr.txt"]["content"] == ""
    payload["command"] = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-12/recheck-current-02.py"
    payload["result"] = "全部斷言通過；引用的文字配方未改，原始小節程式與補充驗證重跑成功。"
    payload["recheck"] = {
        "previous_report": {"path": previous_path.relative_to(ROOT).as_posix(), "sha256": sha(previous_raw)},
        "original_report": previous["original_report"],
        "previous_issues": previous["issues"],
        "resolved_issue_ids": [],
        "head": execute(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_raw)},
        "source_rechecks": sources,
        "original_artifact_rechecks": artifact_checks,
        "git_diffs": [{"path": "course/training.md", **diff}],
        "dependency_comparisons": {
            "training_recipe_section": {"unchanged": True, "previous_sha256": sha(old_t4), "current_sha256": sha(current_t4), "command": recipe},
            "context_definition": {"unchanged": True, "sha256": sha(context), "content": context.decode()},
        },
        "supplemental_execution": {"command_argv": supplemental["command_argv"], "exit_code": supplemental["exit_code"], "stderr": supplemental["stderr"], "stdout_sha256": sha(supplemental["stdout"].encode()), "stdout_fields": "本檔根層 fingerprints、grams、empty_cases、family_grouping、deduplication、recorded_experiment、recipe 等欄位複製自本次實際 stdout。"},
        "source_current_matches_freeze": True,
        "claim_count": len(previous["claims"]),
        "added_or_changed_page_claims": [],
        "scope_note": "引用文件的改句在多模態練習段落，未改動本次審閱頁面或它引用的文字配方。",
    }
    payload["section_execution"] = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/05.md#5.12' --execute", "files": copied}
    target = DIRECTORY / "recheck-output-current-02.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target.read_bytes()), "result": payload["result"], "source_rechecks": sources, "environment": payload["environment"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
