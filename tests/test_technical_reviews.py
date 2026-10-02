"""事實審閱不能沿用過期證據、作者身分或未核實的主張。"""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from scripts import check_course_reviews as readers
from scripts import check_technical_reviews as technical


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_report(root, report):
    path = root / "docs/technical-reviews" / f"{report['lesson_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def reviewed_course(tmp_path):
    (tmp_path / "course/chapters").mkdir(parents=True)
    (tmp_path / "course/figures").mkdir()
    (tmp_path / "course/figures/sample.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    for name in technical.FRONT_MATTER:
        (tmp_path / "course" / name).write_text("# 前置\n", encoding="utf-8")
    source = tmp_path / "course/chapters/14.md"
    source.write_text(
        "# 範例\n\n## 14.1 旋轉與算例\n\n共同旋轉保留點積；2+2=4。\n"
        "![圖](../figures/sample.svg)\n\n```python\nprint(2 + 2)\n```\n\n"
        "## 14.2 算例\n\n2+2=4。\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "docs/technical-reviews/artifacts/output.txt"
    artifact.parent.mkdir(parents=True)
    executed = subprocess.run([sys.executable, "-c", "print(2 + 2)"], capture_output=True, text=True, check=True)
    artifact.write_text(executed.stdout, encoding="utf-8")
    reports = {}
    for lesson_id, body in technical.sections(source):
        has_figure = lesson_id == "14.1"
        report = {
            "schema_version": 1,
            "review_stage": "technical",
            "lesson_id": lesson_id,
            "source": f"course/chapters/14.md#{lesson_id}",
            "reviewer_task": f"/root/technical_{lesson_id.replace('.', '_')}",
            "reviewer_context": "fresh",
            "source_sha256": digest(body.encode()),
            "figure_sha256": {
                "course/figures/sample.svg": digest((tmp_path / "course/figures/sample.svg").read_bytes())
            }
            if has_figure
            else {},
            "verdict": "pass",
            "claims": [
                {
                    "id": "c1",
                    "kind": "concept",
                    "statement": "共同旋轉保留點積",
                    "location": "第一段",
                    "status": "verified",
                    "scope": "旋轉恆等式；不推論模型長文品質",
                    "evidence": [{"source_id": "s1", "locator": "Section 3.2", "supports": "共同旋轉點積恆等式"}],
                    "artifact_ids": [],
                },
                {
                    "id": "c2",
                    "kind": "numeric",
                    "statement": "2+2=4",
                    "location": "第一段",
                    "status": "verified",
                    "scope": "透明整數算例",
                    "evidence": [{"source_id": "s2", "locator": "相加步驟", "supports": "兩次增加2得到4"}],
                    "artifact_ids": [],
                    "verification": {
                        "method": "hand_calculation",
                        "expected": "4",
                        "observed": "4",
                        "tolerance": "整數精確相等",
                        "details": "從2向後數兩次，3、4",
                    },
                },
                {
                    "id": "c3",
                    "kind": "software",
                    "statement": "print(2+2)輸出4",
                    "location": "程式區塊",
                    "status": "verified",
                    "scope": "當次Python版本與此命令",
                    "evidence": [{"source_id": "s3", "locator": "stdout", "supports": "實際輸出4"}],
                    "artifact_ids": ["a1"],
                    "verification": {
                        "method": "executed",
                        "expected": "4",
                        "observed": "4",
                        "details": "在獨立Python程序執行原樣整數算例",
                    },
                },
            ],
            "sources": [
                {
                    "id": "s1",
                    "kind": "paper",
                    "title": "RoFormer",
                    "url": "https://arxiv.org/abs/2104.09864",
                    "version": "測試來源版本v1",
                    "verified": True,
                    "checked_original": True,
                    "accessed_on": "2026-10-02",
                    "authority_reason": "方法原始論文",
                    "inspection_note": "測試格式樣本",
                },
                {"id": "s2", "kind": "derivation", "title": "整數相加", "verified": True, "details": "2+1+1=4"},
                {"id": "s3", "kind": "execution", "title": "Python實際輸出", "verified": True, "artifact_id": "a1"},
            ],
            "artifacts": [
                {
                    "id": "a1",
                    "kind": "execution",
                    "path": artifact.relative_to(tmp_path).as_posix(),
                    "sha256": digest(artifact.read_bytes()),
                    "description": "核對整數算例的實際stdout",
                    "command": f"{sys.executable} -c 'print(2 + 2)'",
                    "result": executed.stdout.strip(),
                    "environment": {"python": sys.version.split()[0], "device": "cpu"},
                }
            ],
            "issues": [],
            "checks": {
                name: {"status": "pass", "details": "測試格式的具體核對記錄", "claim_ids": ["c1", "c2", "c3"]}
                for name in technical.CHECKS
            },
        }
        if not has_figure:
            report["checks"]["figure_consistency"] = {
                "status": "not_applicable",
                "details": "本節沒有圖",
                "claim_ids": [],
            }
        reports[lesson_id] = report
        write_report(tmp_path, report)
    return tmp_path, reports


def assert_failure(root, text, lessons=None):
    result = technical.check(root, lessons)
    assert any(text in failure for failure in result["failures"]), result


def test_pass_and_exact_reader_section_boundaries(reviewed_course):
    root, _ = reviewed_course
    result = technical.check(root)
    assert result["failures"] == []
    assert result["passed_sections"] == 2
    assert technical.FRONT_MATTER == readers.FRONT_MATTER
    assert technical.HEADINGS.pattern == readers.HEADINGS.pattern
    source = root / "course/chapters/14.md"
    assert list(technical.sections(source)) == list(readers.sections(source))


@pytest.mark.parametrize("target", ["body", "figure", "artifact"])
def test_changed_content_requires_reaudit(reviewed_course, target):
    root, _ = reviewed_course
    paths = {
        "body": "course/chapters/14.md",
        "figure": "course/figures/sample.svg",
        "artifact": "docs/technical-reviews/artifacts/output.txt",
    }
    with (root / paths[target]).open("a") as stream:
        stream.write("\nchanged\n")
    assert_failure(root, "目前小節" if target == "body" else "證據版本已改變")


def test_missing_report_and_unknown_section_fail(reviewed_course):
    root, _ = reviewed_course
    (root / "docs/technical-reviews/14.2.json").unlink()
    assert_failure(root, "尚未有獨立事實審閱")
    assert_failure(root, "沒有這個小節", ["99.1"])


@pytest.mark.parametrize("status", ["unresolved", "contradicted"])
def test_unverified_claim_cannot_be_certified(reviewed_course, status):
    root, reports = reviewed_course
    reports["14.1"]["claims"][0]["status"] = status
    write_report(root, reports["14.1"])
    assert_failure(root, "應維持revise")


def test_reader_identity_author_and_duplicate_task_are_rejected(reviewed_course):
    root, reports = reviewed_course
    report = reports["14.1"]
    report["author_tasks"] = [report["reviewer_task"]]
    write_report(root, report)
    assert_failure(root, "作者不能")
    report.pop("author_tasks")
    reports["14.2"]["reviewer_task"] = report["reviewer_task"]
    write_report(root, report)
    write_report(root, reports["14.2"])
    assert_failure(root, "被其他小節重用", ["14.1"])
    reports["14.2"]["reviewer_task"] = "/root/other_technical"
    write_report(root, reports["14.2"])
    reader = root / "docs/reader-reviews/14.2.json"
    reader.parent.mkdir()
    reader.write_text(json.dumps({"reviewer_task": report["reviewer_task"]}))
    assert_failure(root, "不能重用reader")


@pytest.mark.parametrize("field", ["checked_original", "version", "locator", "authority_reason"])
def test_original_source_and_precise_citation_required(reviewed_course, field):
    root, reports = reviewed_course
    report = reports["14.1"]
    if field == "locator":
        report["claims"][0]["evidence"][0][field] = ""
    elif field == "checked_original":
        report["sources"][0][field] = False
    else:
        report["sources"][0][field] = ""
    write_report(root, report)
    assert technical.check(root)["failures"]


def test_candidate_bank_cannot_act_as_authority(reviewed_course):
    root, reports = reviewed_course
    source = reports["14.1"]["sources"][0]
    source["kind"] = "repository_code"
    path = root / "docs/technical-sources/bank.json"
    path.parent.mkdir()
    path.write_text('{"summary":"source candidate"}')
    source.update(path=path.relative_to(root).as_posix(), sha256=digest(path.read_bytes()))
    write_report(root, reports["14.1"])
    assert_failure(root, "候選庫")
    assert_failure(root, "方法／概念主張")


def test_software_execution_cannot_be_replaced_by_handwaving(reviewed_course):
    root, reports = reviewed_course
    reports["14.1"]["claims"][2]["verification"]["method"] = "hand_calculation"
    write_report(root, reports["14.1"])
    assert_failure(root, "軟體／實測主張要有實際執行")
    reports["14.1"]["claims"][2]["verification"]["method"] = "executed"
    reports["14.1"]["claims"][2]["artifact_ids"] = []
    write_report(root, reports["14.1"])
    assert_failure(root, "executed需引用")


def test_empirical_claim_requires_denominators(reviewed_course):
    root, reports = reviewed_course
    claim = reports["14.1"]["claims"][2]
    claim["kind"] = "empirical"
    write_report(root, reports["14.1"])
    assert_failure(root, "需列資料／token／樣本／步數等分母")
    claim["verification"]["denominators"] = {"executions": 1, "input_integers": 2}
    write_report(root, reports["14.1"])
    assert technical.check(root)["failures"] == []


@pytest.mark.parametrize("mutation", ["missing_check", "empty_details", "figure_na", "numeric_na", "unresolved_issue"])
def test_blanket_pass_and_na_do_not_skip_real_work(reviewed_course, mutation):
    root, reports = reviewed_course
    report = reports["14.1"]
    if mutation == "missing_check":
        report["checks"].pop("factual_accuracy")
    elif mutation == "empty_details":
        report["checks"]["source_verification"]["details"] = "  "
    elif mutation == "figure_na":
        report["checks"]["figure_consistency"]["status"] = "not_applicable"
    elif mutation == "numeric_na":
        report["checks"]["numeric_verification"]["status"] = "not_applicable"
    else:
        report["issues"] = [{"status": "unresolved", "details": "still unknown"}]
    write_report(root, report)
    assert technical.check(root)["failures"]


def test_navigation_only_na_needs_reason_and_no_code(reviewed_course):
    root, reports = reviewed_course
    source = root / "course/chapters/14.md"
    source.write_text("# 導覽\n\n## 14.1 入口\n\n請依目錄開啟下一節。\n")
    report = copy.deepcopy(reports["14.1"])
    report.update(
        source_sha256=digest(next(technical.sections(source))[1].encode()),
        figure_sha256={},
        claims=[],
        sources=[],
        artifacts=[],
        applicability={"substantive_claims": False, "reason": "只提供下一節入口"},
    )
    report["checks"] = {
        name: {"status": "not_applicable", "details": "純導覽，沒有相關實質主張"} for name in technical.CHECKS
    }
    write_report(root, report)
    (root / "docs/technical-reviews/14.2.json").unlink()
    assert technical.check(root)["failures"] == []
    report["applicability"]["reason"] = ""
    write_report(root, report)
    assert_failure(root, "applicability=false與理由")
    report["applicability"]["reason"] = "純導覽"
    with source.open("a") as stream:
        stream.write("\n```bash\npython --version\n```\n")
    report["source_sha256"] = digest(next(technical.sections(source))[1].encode())
    write_report(root, report)
    assert_failure(root, "不能整節NA")


def test_front_sections_are_included(reviewed_course):
    root, _ = reviewed_course
    (root / "course/first-steps.md").write_text("# 暖身\n\n## W.1 入口\n\n開啟教材。\n")
    result = technical.check(root)
    assert result["total_sections"] == 3
    assert result["front_sections"] == 1
    assert_failure(root, "W.1: 尚未")


def test_malformed_report_fails_cleanly(reviewed_course):
    root, reports = reviewed_course
    reports["14.1"]["sources"][0].update(kind=["paper"], url="https://[invalid")
    reports["14.1"]["claims"][0]["evidence"][0]["source_id"] = ["s1"]
    reports["14.1"]["artifacts"][0]["kind"] = {}
    write_report(root, reports["14.1"])
    assert technical.check(root)["failures"]
