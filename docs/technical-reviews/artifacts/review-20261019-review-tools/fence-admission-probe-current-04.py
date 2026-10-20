"""用受控完整審閱 fixture 對照圍欄邊界；印出實際准入結果與渲染內容。"""

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    repo, area = args.repo.resolve(), args.area.resolve()
    area.mkdir()
    sys.path[:0] = [str(args.markdown_root), str(args.pymdownx_root)]
    import markdown
    import pymdownx.superfences
    import pymdownx.__meta__

    unit = load("fixture_support", repo / "tests/test_incremental_revision_review.py")
    gate = load("reviewed_incremental_gate", repo / "docs/review-tools/incremental_revision.py")
    preflight = load("reviewed_preflight", repo / "docs/course-review-20261019/tools/preflight.py")
    inventory = load("reviewed_issue_inventory", repo / "docs/course-review-20261019/tools/issue_inventory.py")
    history = area / "incremental_before.py"
    old_raw = subprocess.check_output(["git", "show", "b88b4b63:docs/review-tools/incremental_revision.py"], cwd=repo)
    with history.open("xb") as stream:
        stream.write(old_raw)
    before_gate = load("incremental_before", history)
    original_chapter = unit.CHAPTER
    heading = "## 99.1 範例\n\n"
    rows = []

    def build(root, body, *, listed=True, current=False, patch_changes=None, extra="", source_python=False, empty=False):
        old_body = body.replace("new-token", "old-token")
        reviewed = heading + (body if current else old_body)
        live = heading + body + extra
        unit.CHAPTER = "scripts/example.py" if source_python else original_chapter
        unit.PATCH_OLD, unit.PATCH_NEW = "old-token", "new-token"
        unit.REVIEWED, unit.PATCHED = reviewed, live
        unit.FILE_REVIEWED = unit.INTRO + reviewed
        unit.FILE_PATCHED = unit.INTRO + live
        patch = unit.patch_entry(id="probe", **(patch_changes or {}))

        def raw_save(base, name, value, raw=False):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = value if raw else json.dumps(value, ensure_ascii=False).encode()
            with path.open("xb") as stream:
                stream.write(data)
            return {"path": name, "sha256": sha(data)}

        def save(base, name, value, raw=False):
            if name == "docs/review-tools/incremental_revision.py":
                value, raw = (repo / name).read_bytes(), True
            if name == "reviews/technical.json":
                value = copy.deepcopy(value)
                value["sources"].append({"id": "derivation", "kind": "derivation", "title": "虛構透明算例",
                                         "verified": True, "details": "此 fixture 用 1 等於 1 的恆等式測資料契約。"})
                value["claims"] = [{"id": "literal", "kind": "numeric", "statement": "虛構透明算例 1 等於 1",
                    "location": "fixture", "scope": "只測審閱資料契約", "status": "verified",
                    "evidence": [{"source_id": "derivation", "locator": "恆等式", "supports": "兩邊都是 1"}],
                    "artifact_ids": [], "verification": {"method": "hand_calculation", "expected": "1", "observed": "1",
                    "tolerance": "精確相等", "details": "兩邊逐字都是 1。"}}]
                value["checks"] = {key: {"status": "pass", "details": "虛構已完成核對"} for key in unit.TECHNICAL_CHECKS}
            if name == "sessions/s1/opening.json":
                value = copy.deepcopy(value)
                frozen = raw_save(base, "sessions/s1/freeze.json", {"pages": ["99.1"]})
                plan = raw_save(base, "sessions/s1/plan.json", {"main_pages": ["99.1"], "source_freeze": frozen})
                raw_save(base, "sessions/s1/state.json", {"page_id": "99.1", "reading_plan": plan})
                value.update(reading_plan=plan, source_freeze=frozen)
            if name == "round/manifest.json":
                value = copy.deepcopy(value)
                task = "/root/fixture_continuity"
                report = raw_save(base, "reviews/continuity.json", {"reviewer_task": task, "pages": [{"page_id": "99.1",
                    "source_sha256": sha(reviewed.encode()), "figures_sha256": {}, "verdict": "pass"}]})
                dependency = raw_save(base, "reviews/dependency.json", {"reviewer_task": task, "verdict": "pass",
                    "inventory_page_ids": ["99.1", "plain"]})
                seal = raw_save(base, "round/seal-continuity.json", {"role": "continuity", "status": "completed",
                    "started_at": "2026-10-16T04:00:00+00:00", "completed_at": "2026-10-16T04:30:00+00:00",
                    "report_sha256s": [report["sha256"], dependency["sha256"]]})
                value["review_waves"][0]["seals"]["continuity"] = seal
                assignments = json.loads((base / value["assignments"]["path"]).read_bytes())
                assignments["tasks"].append({"task": task, "role": "continuity", "wave": unit.PATCH_WAVE,
                    "context": "fresh", "pages": ["99.1"], "dispatch_window": {
                    "not_before": "2026-10-16T04:01:00+00:00", "not_after": "2026-10-16T04:02:00+00:00"}})
                value["assignments"] = raw_save(base, "round/assignments-with-continuity.json", assignments)
                value["pages"][0]["continuity"] = {"mode": "review", "wave": unit.PATCH_WAVE, "report": report,
                    "json_pointer": "/pages/0", "original_report": report, "original_json_pointer": "/pages/0"}
                value["dependency_audit"] = dependency
                value["issue_reports"] += [report, dependency]
            return raw_save(base, name, value, raw)

        unit.save = save
        manifest_name = unit.gate_round(root, listed=listed, patches=[] if empty else [patch], reader_on=reviewed,
                                       technical_on=reviewed, chapter_cited=unit.INTRO + reviewed)
        path = root / manifest_name
        return path, json.loads(path.read_bytes()), reviewed, live, patch

    cases = [('nbsp_backtick', '```python\n```\xa0\nprint("new-token")\n```\n', False, {}), ('nbsp_tilde', '~~~python\n~~~\xa0\nprint("new-token")\n~~~\n', False, {}), ('emspace_backtick', '```python\n```\u2003\nprint("new-token")\n```\n', False, {}), ('invalid_backtick_info', '```bad`info\n```\nprint("new-token")\n```\n', False, {}), ('invalid_superfences_options', '```python invalid\n```\nprint("new-token")\n```\n', False, {}), ('ascii_space_control', '```python\n``` \t\n文字 new-token。\n', True, {}), ('two_space_false_close_control', '```python\n  ```\nprint("new-token")\n```\n', False, {}), ('notebook_reopens_control', '```\na\n````\nb\n```\nprint("new-token")\n```\n', False, {})]
    for name, body, should_pass, options in cases:
        root = area / name
        path, manifest, reviewed, live, patch = build(root, body, **options)
        result = gate.audit_manifest(root, path, "all")
        inventory_pages = {page["page_id"]: page for page in json.loads((root / "inventory.json").read_bytes())["pages"]}
        pre = preflight.check_draft(root, manifest, inventory_pages, "sealed", "all")
        issues = inventory.inventory(root, manifest, inventory_pages, "sealed", "all")
        html = markdown.markdown(live, extensions=["pymdownx.superfences"])
        row = {"case": name, "input": {"reviewed_body": reviewed, "current_body": live, "patch": patch,
               "listed": options.get("listed", True), "manifest_path": str(path)}, "expected_pass": should_pass,
               "gate": result, "preflight_failing": pre["failing"], "preflight_pages": pre["pages"],
               "issue_inventory_failing": issues["failing"], "issue_inventory": issues,
               "rendered_html": html, "guard_fenced_at_new": gate.fenced(live, live.find("new-token")),
               "violation": not should_pass and not result["failures"]}
        if not options.get("listed", True):
            row["before_gate"] = before_gate.audit_manifest(root, path, "all")
            row["absent_field_results_equal"] = row["before_gate"] == result
        rows.append(row)

    sources = {}
    for path in [Path(markdown.__file__), Path(pymdownx.superfences.__file__), repo / "zensical.toml", repo / "uv.lock"]:
        sources[str(path)] = sha(path.read_bytes())
    print(json.dumps({"cases": rows, "renderer": {"markdown": markdown.__version__, "pymdownx": pymdownx.__meta__.__version__,
           "source_sha256": sources, "cache_usage": "只讀取既有 uv 套件快取；沒有安裝或修改依賴。"},
           "fixture": {"factory": "tests/test_incremental_revision_review.py:gate_round", "stage": "all",
           "real_components": ["incremental_revision.audit_manifest", "scripts/check_technical_reviews._validate", "preflight.check_draft", "issue_inventory.inventory"],
           "standins": ["baseline_replay.replay", "composite_review.current_inventory", "scripts/build_course.build", "reader.units"],
           "purpose": "受控資料契約反例，不宣稱正式全書候選 manifest 已通過。", "before_gate_sha256": sha(old_raw)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--markdown-root", type=Path, required=True)
    parser.add_argument("--pymdownx-root", type=Path, required=True)
    main(parser.parse_args())
