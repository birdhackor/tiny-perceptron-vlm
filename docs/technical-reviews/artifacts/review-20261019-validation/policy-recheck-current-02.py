"""核對現行規約的作者補丁邊界及沿用證據的完整性。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-current-01.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    previous = json.loads(PREVIOUS.read_bytes())
    evidence = json.loads((HERE / "evidence-results-current-01.json").read_bytes())
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
        "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()},
        "previous_report": {"path": str(PREVIOUS.relative_to(ROOT)), "sha256": sha(PREVIOUS)},
        "guide_diff": command(["git", "diff", "6a683dc4d249", "41651edb", "--", "docs/technical-review-guide.md"]),
        "current_guide_sha256": sha(ROOT / "docs/technical-review-guide.md"),
        "sources": [], "artifacts": [], "previous_input_files": [],
    }
    for source in previous["sources"]:
        if source["kind"] == "repository_code":
            observed = sha(ROOT / source["path"])
            result["sources"].append({"id": source["id"], "path": source["path"],
                "previous_sha256": source["sha256"], "current_sha256": observed,
                "unchanged": observed == source["sha256"]})
    for artifact in previous["artifacts"]:
        observed = sha(ROOT / artifact["path"])
        result["artifacts"].append({"id": artifact["id"], "path": artifact["path"], "sha256": observed,
                                    "unchanged": observed == artifact["sha256"]})
    for name, expected in evidence["input_files"].items():
        if name.startswith("outputs/"):
            assert name == "outputs/course-review-20261019/executed-05/validation-kernel.json"
        if name.startswith("docs/lesson-products/reviews/"):
            assert Path(name).name in {
                "engineering-final-b-mechanisms-original.json", "engineering-final-b-mechanisms-execution.json",
                "engineering-final-c-original.json", "engineering-final-c-execution.json"}
        observed = sha(ROOT / name)
        result["previous_input_files"].append({"path": name, "previous_sha256": expected["sha256"],
                                               "current_sha256": observed, "unchanged": observed == expected["sha256"]})
    result["source_sha256"] = sha(ROOT / "docs/validation.md")
    result["frozen_source_sha256"] = sha(ROOT / "docs/course-review-20261019/freeze-05/sources/validation.md")
    result["source_freeze_sha256"] = sha(ROOT / "docs/course-review-20261019/freeze-06.json")
    module_path = ROOT / "docs/review-tools/incremental_revision.py"
    spec = importlib.util.spec_from_file_location("author_patch_boundary", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result["policy_implementation"] = {
        "path": str(module_path.relative_to(ROOT)), "sha256": sha(module_path),
        "identifiers": ["author_patches", "undo_patches", "fence_blocks", "PATCH_LIMIT", "PATCH_FORBIDDEN"],
        "patch_limit": module.PATCH_LIMIT, "forbidden_literal_tokens": module.PATCH_FORBIDDEN}
    bt = chr(96)
    cases = [
        ("plain_text", "一般說明：新版。\n", "舊版", "新版", True),
        ("inline_code", "呼叫 " + bt + "new_name()" + bt + "。\n", "old_name", "new_name", True),
        ("math", "結果 $x=2$。\n", "x=1", "x=2", True),
        ("link_label", "[新版](https://example.invalid/)\n", "舊版", "新版", True),
        ("indented_code", "    value = 2\n", "1", "2", True),
        ("html_pre", "<pre>value = 2</pre>\n", "1", "2", True),
        ("fenced_code", bt * 3 + "python\nvalue = 2\n" + bt * 3 + "\n", "1", "2", False),
        ("nonplain_fence", "新版\n" + bt * 4 + "python\nvalue = 2\n" + bt * 4 + "\n", "舊版", "新版", False),
        ("multiline_patch", "新版\n", "舊\n版", "新版", False),
        ("oversize_patch", "新版\n", "甲" * 81, "新版", False),
        ("duplicate_target", "新版 新版\n", "舊版", "新版", False),
    ]
    result["synthetic_patch_cases"] = []
    with tempfile.TemporaryDirectory(prefix="policy-fixtures-", dir=HERE) as temporary:
        fixture = Path(temporary)
        for name, markdown, old, new, expected in cases:
            case_root = fixture / name
            case_root.mkdir()
            patch = {"id": name, "path": "lesson.md", "old": old, "new": new,
                     "requested_by": "author", "requested_on": "2026-10-20",
                     "reason": "合成資料，只驗證解析器邊界；不是實際作者要求或發布補丁。"}
            value = {"schema_version": 1, "kind": module.PATCHES_KIND,
                     "statement": module.PATCHES_STATEMENT, "patches": [patch]}
            patch_path = case_root / "patches.json"
            dump_new(patch_path, value)
            manifest = {"author_patches": {"path": "patches.json", "sha256": sha(patch_path)}}
            live = {"sample": {"source": "lesson.md"}}
            entry = {"name": name, "markdown": markdown, "patch": patch,
                     "expected_accepted": expected, "phase": "author_patches"}
            try:
                parsed = module.author_patches(case_root, manifest, live)
                entry["phase"] = "undo_patches"
                restored = module.undo_patches(markdown, list(enumerate(parsed)), "lesson.md")
                entry.update(accepted=True, restored=restored)
                assert restored[-1][0] == markdown.replace(new, old)
            except ValueError as error:
                entry.update(accepted=False, error=str(error))
            entry["matches_expected"] = entry["accepted"] == expected
            result["synthetic_patch_cases"].append(entry)
    result["interpretation"] = [
        "一般文字、行內程式碼、數學式、連結標籤、縮排程式與HTML範例可通過字面補丁核對；不代表其內容已被審對。",
        "圍欄內、非單純圍欄、跨行、超長與非唯一目標範例被拒；這些是具體格式邊界，不是完整Markdown語意解析。",
        "現行規約將author_patches與undo_patches視為機械條件來源，另明記內容由作者負責且未經AI審閱。",
        "本頁未變；理解／技術審閱分工、一般修訂的版本回查與單環境不可外推仍成立。"]
    result["metadata_correction_basis"] = {
        "affected_claim_ids": ["c57", "c58", "c59", "c62"],
        "details": "前報告有scope說明錯置：語音診斷附上LoRA限制、審閱流程附上語音判準、估時附上作者補丁例外；重跑流程仍有較早的Notebook依賴不足觀察。新報告按statement恢復對應範圍，並註明先前Notebook探針已通過。不改頁面、舊報告、主張判定或數值。"}
    drift = [s["path"] for s in result["sources"] if not s["unchanged"]]
    inputs_drift = [s["path"] for s in result["previous_input_files"] if not s["unchanged"]]
    result["checks"] = {
        "previous_report_hash": result["previous_report"]["sha256"] == "5f29a0d97d534a42532749af11667dfba86f305f08cf03d86d17476ce5ceff2f",
        "page_and_freeze_unchanged": result["source_sha256"] == result["frozen_source_sha256"] == "3d4456fda1dc81edb0ac2b4a7314321a4a3cbd98914a99fbc7db5af16bd79b9c"
                                    and result["source_freeze_sha256"] == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9",
        "only_declared_source_changed": drift == ["docs/technical-review-guide.md"],
        "only_declared_input_changed": inputs_drift == ["docs/technical-review-guide.md"],
        "all_prior_artifacts_unchanged": all(s["unchanged"] for s in result["artifacts"]),
        "all_patch_cases_match": all(s["matches_expected"] for s in result["synthetic_patch_cases"]),
    }
    dump_new(HERE / "policy-recheck-results-current-02.json", result)
    print(json.dumps({"checks": result["checks"], "case_count": len(cases),
                      "guide_sha256": result["current_guide_sha256"]}, ensure_ascii=False))
    if not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
