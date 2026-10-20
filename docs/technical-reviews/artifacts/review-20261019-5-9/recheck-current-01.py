"""核對不可改寫的既有證據、來源差異與本次 CPU 執行結果。"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-9/technical-5-9-original-01.json"
ORIGINAL_SHA = "041b62310e950ee27d47dea4a29cf904a5e71ab2c4f3680b0f919c7649a01cae"
EXPECTED_TRAINING_SHA = "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
EXPECTED_FREEZE_SHA = "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
DESTINATION = BASE / "recheck-evidence-current-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(argv):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def section(raw, heading):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, match in enumerate(headers) if match[0].startswith(("## " + heading + " ").encode()))
    end = headers[index+1].start() if index+1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def main():
    assert not DESTINATION.exists()
    original_raw = ORIGINAL.read_bytes()
    assert digest(original_raw) == ORIGINAL_SHA
    original = json.loads(original_raw)
    artifact_checks = []
    for artifact in original["artifacts"]:
        observed = digest((ROOT / artifact["path"]).read_bytes())
        item = {"id": artifact["id"], "path": artifact["path"], "expected_sha256": artifact["sha256"], "observed_sha256": observed, "matches": observed == artifact["sha256"]}
        assert item["matches"]
        artifact_checks.append(item)
    source_checks = []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        observed = digest((ROOT / source["path"]).read_bytes())
        source_checks.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": observed, "unchanged": observed == source["sha256"]})
    assert [item["path"] for item in source_checks if not item["unchanged"]] == ["course/training.md"]
    assert digest((ROOT / "course/training.md").read_bytes()) == EXPECTED_TRAINING_SHA
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.9.md").read_bytes()
    current = section((ROOT / "course/chapters/05.md").read_bytes(), "5.9")
    assert current == frozen
    assert digest(current) == original["source_sha256"]
    freeze_path = ROOT / "docs/course-review-20261019/freeze-06.json"
    assert digest(freeze_path.read_bytes()) == EXPECTED_FREEZE_SHA
    page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == "5.9")
    assert page["source_sha256"] == digest(current) and page["figures_sha256"] == {}
    diff = execute(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    assert diff["exit_code"] == 0
    versions = []
    training_raw = (ROOT / "course/training.md").read_bytes()
    training_section = section(training_raw, "T.4")
    for revision in ("5ef56bc756f3", "6a7aa1de5227"):
        argv = ["git", "show", revision + ":course/training.md"]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
        raw = result.stdout
        relevant = section(raw, "T.4")
        versions.append({"revision": revision, "command_argv": argv, "exit_code": result.returncode, "file_sha256": digest(raw), "section_sha256": digest(relevant), "section_equals_current": relevant == training_section})
    assert all(v["section_equals_current"] for v in versions)
    assert versions[0]["file_sha256"] == next(s["sha256"] for s in original["sources"] if s["id"] == "training_recipe")
    recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda"
    assert recipe in training_section.decode()
    runner = (ROOT / "scripts/course_experiments/run.py").read_text()
    parser_excerpt = next(line.strip() for line in runner.splitlines() if 'parser.add_argument("--device"' in line)
    assert 'default="cpu"' in parser_excerpt
    invocation = execute([".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/05.md#5.9", "--execute"])
    assert invocation["exit_code"] == 0
    tool_result = json.loads(invocation["stdout"])
    run_dir = Path(tool_result["artifact_directory"])
    assert run_dir.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    copied = {}
    for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py", "bootstrap.py"):
        raw = (run_dir / name).read_bytes()
        copied[name] = {"sha256": digest(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
    assert copied["execution.json"]["content"]["exit_code"] == 0
    assert tool_result["source_sha256"] == digest(current)
    assert tool_result["svg_references"] == 0
    probe_run = execute([".venv/bin/python", "docs/technical-reviews/artifacts/review-20261019-5-9/verify_resources.py"])
    assert probe_run["exit_code"] == 0
    probe = json.loads(probe_run["stdout"])
    probe_run["result"] = probe
    del probe_run["stdout"]
    cli = execute([".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"])
    assert cli["exit_code"] == 0
    assert "--experiment" in cli["stdout"] and "--device {cpu,cuda}" in cli["stdout"]
    evidence = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-9/recheck-current-01.py",
        "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(original_raw)},
        "original_issues": original["issues"],
        "source_checks": source_checks,
        "original_artifact_checks": artifact_checks,
        "section_comparison": {"source_sha256": digest(current), "current_equals_frozen": current == frozen, "figures_sha256": {}, "changed_sentences": []},
        "source_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path.read_bytes())},
        "training_diff": diff,
        "training_dependency": {"versions": versions, "current_file_sha256": digest(training_raw), "current_section_sha256": digest(training_section), "recipe": recipe, "current_runner_device_argument": parser_excerpt, "relevant_section_unchanged": True, "details": "差異只在入口的裝置說明、多模態示例與量化示例；引用的文字固定實驗配方區塊逐位元組相同。"},
        "section_execution": {"invocation": invocation, "tool_result": tool_result, "copied_files": copied},
        "resource_probe_execution": probe_run,
        "recipe_interface_execution": cli,
        "environment": probe["environment"],
        "result": "全部斷言通過；既有主張的原文與實質證據未變，重跑原始 fence、資源探針及入口 --help 均成功。"
    }
    for source in source_checks:
        assert digest((ROOT/source["path"]).read_bytes()) == source["current_sha256"]
    for artifact in artifact_checks:
        assert digest((ROOT/artifact["path"]).read_bytes()) == artifact["observed_sha256"]
    assert digest(ORIGINAL.read_bytes()) == ORIGINAL_SHA
    with open(DESTINATION, "x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(DESTINATION.relative_to(ROOT)), "sha256": digest(DESTINATION.read_bytes()), "source_changes": [s["path"] for s in source_checks if not s["unchanged"]], "original_artifacts_unchanged": len(artifact_checks), "section_stdout": copied["stdout.txt"]["content"], "resource_totals": probe["totals"], "effective_targets": probe["denominator_replay"]["recomputed_effective_tokens"], "result": evidence["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
