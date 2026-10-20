"""保存引用文件差異、原始程式重跑與舊證據完整性的複查收據。"""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"command_argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def read_section(raw, name):
    matched = re.search(rb"(?ms)^## " + re.escape(name.encode()) + rb" .*?(?=^## |\Z)", raw)
    assert matched is not None, "找不到指定小節"
    return matched[0]


def main():
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-12/technical-5-12-original-01.json"
    original_raw = original_path.read_bytes()
    assert sha(original_raw) == "c4bfae18b8e37afef2adb868b9e4dabea62378a03e8f4f5874fca0d4ae2fef22"
    original = json.loads(original_raw)
    assert original["issues"] == []
    artifact_checks = []
    for item in original["artifacts"]:
        actual = sha((ROOT / item["path"]).read_bytes())
        assert actual == item["sha256"], "舊證據的雜湊已改變"
        artifact_checks.append({"path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": actual, "unchanged": True})
    expected_changed = {
        "course/training.md": "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94",
        "course/chapters/02.md": "55f87e23709ef968b1f6f611fa2ea3f77d88b44579ed7687df27e3a5c4ddbd29",
    }
    source_checks = []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        raw = (ROOT / source["path"]).read_bytes()
        actual = sha(raw)
        if source["path"] in expected_changed:
            assert actual == expected_changed[source["path"]]
        else:
            assert actual == source["sha256"], "非預期的引用程式變動"
        source_checks.append({"source_id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
    git_diffs = []
    old_files, new_files = {}, {}
    for path in expected_changed:
        git_diffs.append({"path": path, **command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", path])})
        old = command(["git", "show", "5ef56bc756f3:" + path])["stdout"].encode()
        new = command(["git", "show", "6a7aa1de5227:" + path])["stdout"].encode()
        assert sha(new) == expected_changed[path]
        assert new == (ROOT / path).read_bytes()
        previous = next(item for item in source_checks if item["path"] == path)
        assert sha(old) == previous["previous_sha256"]
        old_files[path], new_files[path] = old, new
    old_context = read_section(old_files["course/chapters/02.md"], "2.5")
    new_context = read_section(new_files["course/chapters/02.md"], "2.5")
    marker = "多看字也有成本。".encode()
    old_context_evidence = old_context.split(marker)[0]
    new_context_evidence = new_context.split(marker)[0]
    assert old_context_evidence == new_context_evidence
    assert "prefix[-length:]" in new_context_evidence.decode()
    old_recipe_section = read_section(old_files["course/training.md"], "T.4")
    new_recipe_section = read_section(new_files["course/training.md"], "T.4")
    assert old_recipe_section == new_recipe_section
    recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
    assert recipe.encode() in new_recipe_section
    freeze_path = ROOT / "docs/course-review-20261019/freeze-06.json"
    freeze_raw = freeze_path.read_bytes()
    assert sha(freeze_raw) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
    page = next(item for item in json.loads(freeze_raw)["pages"] if item["page_id"] == "5.12")
    frozen = (ROOT / page["snapshot"]).read_bytes()
    current = read_section((ROOT / "course/chapters/05.md").read_bytes(), "5.12")
    assert frozen == current
    assert sha(frozen) == "832c6295bab044426dd82bff227820875caa016ee41f22dbba6fe748265339e4"
    assert page["figures_sha256"] == {}
    supplemental = command([str(ROOT / ".venv/bin/python"), str(DIRECTORY / "verification.py")])
    payload = json.loads(supplemental["stdout"])
    previous_output = json.loads((DIRECTORY / "verification-output.json").read_bytes())
    for key in ("fingerprints", "grams", "repeated_fragments", "empty_cases", "synonym_example", "different_answer_example", "family_grouping", "deduplication", "recorded_experiment", "recipe"):
        assert payload[key] == previous_output[key], "補充算例與原紀錄不一致"
    run = ROOT / "outputs/reviewer-tools/runs/20261020T102235463547Z"
    section_files = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (run / name).read_bytes()
        section_files[name] = {"sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
    assert section_files["execution.json"]["content"]["exit_code"] == 0
    assert section_files["extraction.json"]["content"]["source_sha256"] == sha(frozen)
    assert section_files["stdout.txt"]["content"] == previous_output["original_stdout"]
    payload["command"] = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-12/recheck-current-01.py"
    payload["result"] = "所有斷言通過；引用的視窗定義與重做配方未變，原始程式及補充驗證結果相符。"
    payload["recheck"] = {
        "original_report": {"path": original_path.relative_to(ROOT).as_posix(), "sha256": sha(original_raw)},
        "original_issues": original["issues"],
        "resolved_issue_ids": [],
        "head": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_raw)},
        "source_rechecks": source_checks,
        "original_artifact_rechecks": artifact_checks,
        "git_diffs": git_diffs,
        "dependency_comparisons": {
            "context_definition": {"unchanged": True, "content": new_context_evidence.decode(), "sha256": sha(new_context_evidence)},
            "training_recipe_section": {"unchanged": True, "sha256": sha(new_recipe_section), "command": recipe},
        },
        "supplemental_execution": supplemental,
        "source_current_matches_freeze": True,
        "claim_count": len(original["claims"]),
        "added_or_changed_page_claims": [],
        "scope_note": "複查的是完整教材小節及引用來源對它的支持；引用頁其他段落的修改不擴張成本次逐頁審閱。",
    }
    payload["section_execution"] = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/05.md#5.12' --execute",
        "files": section_files,
    }
    target = DIRECTORY / "recheck-output-current-01.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target.read_bytes()), "result": payload["result"], "page_unchanged": True, "original_issues": [], "source_rechecks": source_checks, "environment": payload["environment"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
