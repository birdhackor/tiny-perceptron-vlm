# 保存工具差異、版本檢查邊界與 CPU 短流程；只讀舊報告，所有新輸出用 exclusive create。
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_readme/technical-readme-current-01.json"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command):
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=90)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    import torch
    previous = json.loads(PREVIOUS.read_bytes())
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
              "previous_report": {"path": PREVIOUS.relative_to(ROOT).as_posix(), "sha256": sha(PREVIOUS)}, "sources": [], "artifacts": [], "commands": [], "validator_cases": []}
    for group in ("sources", "artifacts"):
        for entry in previous[group]:
            if "path" in entry:
                observed = sha(ROOT / entry["path"])
                result[group].append({"id": entry["id"], "path": entry["path"], "expected": entry["sha256"], "observed": observed, "unchanged": observed == entry["sha256"]})
    changed = [r["path"] for r in result["sources"] if not r["unchanged"]]
    assert changed == ["scripts/check_technical_reviews.py"], changed
    assert all(r["unchanged"] for r in result["artifacts"])
    for name in ("README.md", "docs/course-review-20261019/freeze-05/sources/readme.md", "docs/course-review-20261019/freeze-06.json", "course/figures/rewrite-01-character-ids.svg", "docs/technical-review-guide.md", "docs/review-tools/incremental_revision.py"):
        result.setdefault("hashes", {})[name] = sha(ROOT / name)
    assert result["hashes"]["README.md"] == previous["source_sha256"]
    assert result["hashes"]["README.md"] == result["hashes"]["docs/course-review-20261019/freeze-05/sources/readme.md"]
    result["tool_diff"] = run(["git", "diff", "b2464df6d2e1", "d0ef95d5e636", "--", "scripts/check_technical_reviews.py"])
    result["tool_commits"] = []
    for revision in ("b2464df6d2e1", "d0ef95d5e636"):
        process = subprocess.run(["git", "show", revision + ":scripts/check_technical_reviews.py"], cwd=ROOT, env=ENV, capture_output=True, check=True)
        result["tool_commits"].append({"command": ["git", "show", revision + ":scripts/check_technical_reviews.py"], "sha256": hashlib.sha256(process.stdout).hexdigest()})
    validator = load_module("readme_current_validator", ROOT / "scripts/check_technical_reviews.py")
    body = (ROOT / "docs/course-review-20261019/freeze-05/sources/readme.md").read_text()
    reviewer = previous["reviewer_task"]
    owner = {reviewer: ["readme"]}
    reader_tasks = {"/root/rv1019_reader_entry"}
    def validate(report, earlier=None, owners=None, readers=None):
        return validator._validate(ROOT, ROOT/"README.md", "readme", body, report, owner if owners is None else owners, reader_tasks if readers is None else readers, earlier_versions=earlier)
    baseline = copy.deepcopy(previous)
    source = next(s for s in baseline["sources"] if s.get("path") == "scripts/check_technical_reviews.py")
    source["sha256"] = sha(ROOT / source["path"])
    def case(name, report, expected, earlier=None, owners=None, readers=None):
        errors = validate(report, earlier, owners, readers)
        passed = errors == expected
        result["validator_cases"].append({"name": name, "expected": expected, "observed": errors, "passed": passed})
        assert passed, name + ": " + repr(errors)
    case("repo來源仍記舊版時預設拒絕", previous, ["source repo_48: 證據版本已改變：scripts/check_technical_reviews.py"])
    case("所有來源記現行版本時通過", baseline, [])
    stale_body = copy.deepcopy(baseline)
    stale_body["source_sha256"] = "1" * 64
    case("正文SHA不符預設拒絕", stale_body, ["事實審閱不是目前小節原文"])
    case("只有明確列出的正文SHA可另行接受", stale_body, [], {"source_sha256": ["1" * 64]})
    case("未列出的正文SHA仍拒絕", stale_body, ["事實審閱不是目前小節原文"], {"source_sha256": ["2" * 64]})
    old_tool = next(s for s in previous["sources"] if s.get("path") == "scripts/check_technical_reviews.py")["sha256"]
    case("較早來源只在相同路徑允許", previous, [], {"files": {"scripts/check_technical_reviews.py": [old_tool]}})
    case("別的路徑不能授權本來源", previous, ["source repo_48: 證據版本已改變：scripts/check_technical_reviews.py"], {"files": {"scripts/check_course_reviews.py": [old_tool]}})
    stale_artifact = copy.deepcopy(baseline)
    artifact = stale_artifact["artifacts"][0]
    artifact["sha256"] = "2" * 64
    case("較早版本選項不放寬artifact", stale_artifact, ["artifact audit_program: 證據版本已改變：" + artifact["path"]], {"files": {artifact["path"]: ["2" * 64]}})
    stale_figure = copy.deepcopy(baseline)
    figure = "course/figures/rewrite-01-character-ids.svg"
    stale_figure["figure_sha256"][figure] = "3" * 64
    case("較早版本選項不放寬圖", stale_figure, ["figure: 證據版本已改變：" + figure], {"files": {figure: ["3" * 64]}})
    author = copy.deepcopy(baseline)
    author["author_tasks"].append(reviewer)
    case("作者不能審自己的內容", author, ["作者不能自行核實自己的版本與證據"], {"source_sha256": [baseline["source_sha256"]]})
    case("讀者身分不能兼技術審閱", baseline, ["不能重用reader審閱task，兩階段需分開"], readers=reader_tasks | {reviewer})
    case("不同小節不能重用審閱task", baseline, ["reviewer_task被其他小節重用，每節需要新審閱者"], owners={reviewer: ["readme", "another-page"]})
    unresolved = copy.deepcopy(baseline)
    unresolved["claims"][0]["status"] = "unresolved"
    case("版本允許選項不會核准未核實主張", unresolved, ["claim website: 未核實／矛盾主張不能通過，應維持revise"], {"source_sha256": [baseline["source_sha256"]]})
    missing = []
    validator._hash_file(ROOT, "docs/technical-reviews/artifacts/review-20261019-readme/nonexistent-current-fixture", "4"*64, missing, "fixture", ["4"*64])
    result["missing_evidence"] = {"expected_rejected": True, "errors": missing, "rejected": bool(missing)}
    assert missing
    incremental = load_module("readme_patch_functions", ROOT / "docs/review-tools/incremental_revision.py")
    patch_record = {"id": "local-text-change", "path": "sample.md", "old": "較早文字", "new": "目前文字", "requested_by": "author", "requested_on": "2026-10-20", "reason": "僅供驗證版本倒退的合成資料"}
    versions = incremental.undo_patches("前文。目前文字。後文。", [(0, patch_record)], "合成文字")
    assert versions[0][0] == "前文。較早文字。後文。"
    rejected = {}
    for name, text in (("重複出現", "目前文字。目前文字。"), ("程式圍欄內", "```text\n目前文字\n```"), ("未列入的改動", "不同文字。")):
        try:
            incremental.undo_patches(text, [(0, patch_record)], "合成文字")
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise AssertionError("合成倒退條件應拒絕：" + name)
    result["patch_reversal"] = {"fixture": patch_record, "observed": versions, "rejections": rejected, "scope": "只執行純函式合成資料；未讀實際作者補丁清單、其他審閱或manifest。呼叫端由file_versions/page_versions產生可承認的hash，程式引文另見來源。"}
    uv = ["uv", "run", "--no-sync", "python"]
    for args in (["scripts/build_course.py", "--check"], ["scripts/fetch_training_assets.py", "--list"], ["scripts/fetch_course_models.py", "--list"]):
        result["commands"].append(run(uv + args))
    with tempfile.TemporaryDirectory(prefix="scratch-current-", dir=HERE) as tmp:
        tmp = Path(tmp)
        result["commands"].append(run(uv + ["scripts/prepare_data.py", "--kind", "toy-text", "--output", str(tmp/"data")]))
        data = tmp / "data/toy-text/train.jsonl"
        destination = tmp / "unexpected.pt"
        command = uv + ["scripts/train.py", "--task", "text", "--data", str(data), "--device", "cpu", "--output", str(destination)]
        result["commands"].append(run(command))
        result["inputs"] = {p.name: p.read_text() for p in sorted(data.parent.iterdir())}
        from scripts import train
        actual = torch.optim.AdamW
        optimizers = []
        class CheckedAdamW(actual):
            def __init__(self, params, *args, **kwargs):
                params = list(params)
                self.checked = params
                self.before = [p.detach().clone() for p in params]
                optimizers.append(self)
                super().__init__(params, *args, **kwargs)
            def step(self, *args, **kwargs):
                raise AssertionError("dry-run不應更新權重")
        output = io.StringIO()
        argv = command[4:]
        with patch.object(torch.optim, "AdamW", CheckedAdamW), patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            train.main()
        result["dry_run"] = {"argv": argv, "stdout": output.getvalue(), "parameters_unchanged": all(torch.equal(p,b) for o in optimizers for p,b in zip(o.checked,o.before)), "checkpoint_exists": destination.exists()}
        assert result["dry_run"]["parameters_unchanged"] and not destination.exists()
    assert all(r["returncode"] == 0 for r in result["commands"])
    result["figure"] = {"viewed_png": "outputs/course-review-20261019/figure-renders-04/61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad.png", "visual_inspection": "已用view_image觀看；上排後下排，雙向字典標籤、箭頭方向與底部限制一致；SVG雜湊不變。"}
    target = HERE / "check-current-02.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target),"changed_sources":changed,"inherited_artifacts":len(result["artifacts"]),"validator_cases":len(result["validator_cases"]),"case_results":[(r["name"],r["passed"]) for r in result["validator_cases"]],"dry_run":result["dry_run"]},ensure_ascii=False))

if __name__ == "__main__":
    main()
