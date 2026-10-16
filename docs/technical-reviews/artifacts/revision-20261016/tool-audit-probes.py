"""獨立審核收據：只改暫存證據，不改工具、正文或既有報告。

在 repo 根目錄以 uv run python <本檔> 重跑。mutation 沿用同一次已實際
重播成功的 baseline 結果；其他 inventory、來源、角色、處置檢查照原工具執行。
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
MANIFEST = ROOT / "docs/course-revision-20261016/manifest-reader-01.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(ref):
    return json.loads((ROOT / ref["path"]).read_bytes())


def main():
    started = datetime.now(UTC).isoformat()
    original = json.loads(MANIFEST.read_bytes())
    baseline_module = load_module("independent_baseline", ROOT / "docs/review-tools/baseline_replay.py")
    freeze = json.loads((ROOT / "docs/course-publish-20261008/current-source-freeze.json").read_bytes())
    overlay = json.loads((ROOT / "docs/course-revision-20261016/baseline-overlay.json").read_bytes())
    entries = [ref for kind in ("source_files", "criteria", "figures", "notebooks") for ref in freeze[kind].values()]
    entries += overlay["files"]
    watched = sorted({ref[key] for ref in entries for key in ("path", "live_path")})
    before = {name: digest(ROOT / name) for name in watched if (ROOT / name).is_file()}
    baseline = baseline_module.replay(ROOT)
    after = {name: digest(ROOT / name) for name in before}
    assert before == after, "baseline replay changed live or preserved bytes"
    assert baseline["failures"] == [], baseline["failures"]
    review = load_module("independent_incremental", ROOT / "docs/review-tools/incremental_revision.py")
    real_module = review.module
    review.module = lambda name, path: (
        SimpleNamespace(replay=lambda root: baseline)
        if Path(path).name == "baseline_replay.py"
        else real_module(name, path)
    )
    results = []
    cache = ROOT / "outputs/review-tool-audit-probes"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache, prefix="independent-") as temporary:
        work = Path(temporary)
        counter = 0

        def save(value):
            nonlocal counter
            counter += 1
            path = work / f"proof-{counter:03}.json"
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}

        def page(manifest, pid="7.23"):
            return next(row for row in manifest["pages"] if row["page_id"] == pid)

        def run(name, mutate, expected, details, stage="reader"):
            manifest = copy.deepcopy(original)
            mutate(manifest)
            ref = save(manifest)
            result = review.audit_manifest(ROOT, ROOT / ref["path"], stage)
            results.append({
                "id": name,
                "stage": stage,
                "expected_contract_result": expected,
                "actual_status": result["status"],
                "record_count": len(result["records"]),
                "failures": result["failures"],
                "details": details,
            })

        run("real_reader_control", lambda m: None, "pass", "真實 manifest 原樣核對，baseline 已在同次執行完整重播。")
        run("real_all_is_incomplete", lambda m: None, "reject", "本階段確實尚未有 technical/continuity seals，不應通過 all。", "all")
        run("missing_canonical_page", lambda m: m["pages"].pop(), "reject", "刪除一個 canonical row。")

        def stale_source(m):
            frozen = read(m["source_freeze"])
            frozen["pages"][0]["source_sha256"] = "0" * 64
            m["source_freeze"] = save(frozen)

        run("stale_source_freeze", stale_source, "reject", "目前來源與 source freeze 不符。")

        def omitted_note(m):
            page(m)["reader"]["sessions"][0]["notes"].pop()

        run("missing_checkpoint", omitted_note, "reject", "少一筆首次閱讀原始 checkpoint。")

        def author_identity(m):
            assignments = read(m["assignments"])
            assignments["tasks"][0]["task"] = "/root"
            m["assignments"] = save(assignments)

        run("author_as_reviewer", author_identity, "reject", "作者本身列為 reviewer task。")

        def duplicate(m):
            m["pages"].append(copy.deepcopy(page(m)))

        run("duplicate_catalog_row", duplicate, "reject", "重複 7.23 row；dict 覆蓋後 len(catalog) 仍為 350，原始 list 長度未驗。")

        def future_note(m):
            session = page(m)["reader"]["sessions"][0]
            note = read(session["notes"][-1])
            note["saved_at"] = "2100-01-01T00:00:00+00:00"
            session["notes"][-1] = save(note)

        run("checkpoint_after_reader_seal", future_note, "reject", "最後一筆 note 明確晚於 2026 reader seal；原工具只檢查 session 內時間遞增。")

        def empty_assignment_scope(m):
            assignments = read(m["assignments"])
            for assignment in assignments["tasks"]:
                assignment["pages"] = []
            m["assignments"] = save(assignments)

        run("empty_actual_assignment_scope", empty_assignment_scope, "reject", "所有派工 pages 變成空清單，報告仍接受；頁面與 task/wave 的派工範圍未連結。")

        def wrong_original(m):
            fake = save({"reviewer_task": "/root", "page_id": "unrelated", "issues": []})
            page(m)["reader"]["original_report"] = fake
            m["issue_reports"].append(fake)

        run("unrelated_author_original_report", wrong_original, "reject", "某頁 original_report 改指作者寫的無關頁，只要 hash 與 issue inventory 有引用便接受。")

        def fake_verification(m):
            dispositions = read(m["dispositions"])
            unrelated = {"path": "AGENTS.md", "sha256": digest(ROOT / "AGENTS.md")}
            for decision in dispositions["items"]:
                if decision["action"] == "fixed":
                    decision["verification"] = [unrelated]
            m["dispositions"] = save(dispositions)

        run("arbitrary_bytes_close_fixes", fake_verification, "reject", "全部 fixed 的複查引用換成 AGENTS.md；未驗 reviewer/page/source/verdict/stage，仍可結案。")

        def future_dispositions(m):
            dispositions = read(m["dispositions"])
            dispositions["at"] = "2100-01-01T00:00:00+00:00"
            m["dispositions"] = save(dispositions)

        run("dispositions_after_reader_seal", future_dispositions, "reject", "必要缺口處置完成時間晚於已完成 reader seal，仍可通過 reader。")

        def replace_report(m, old, new):
            def visit(value):
                if isinstance(value, dict):
                    if value.get("path") == old["path"] and value.get("sha256") == old["sha256"]:
                        value.update(new)
                    else:
                        for item in value.values():
                            visit(item)
                elif isinstance(value, list):
                    for item in value:
                        visit(item)
            visit(m)
            for wave in m["review_waves"]:
                seal = read(wave["seals"]["reader"])
                seal["report_sha256s"] = [new["sha256"] if x == old["sha256"] else x for x in seal["report_sha256s"]]
                wave["seals"]["reader"] = save(seal)

        def no_current_basis(m):
            proof = page(m, "7.25")["reader"]
            old = copy.deepcopy(proof["report"])
            current = read(old)
            for row in current["pages"]:
                # 只留下 gate 必查四欄，移除同讀者確認、估時、複查與 basis 鏈。
                for key in list(row):
                    if key not in {"page_id", "source_sha256", "figures_sha256", "verdict"}:
                        del row[key]
            current = {"reviewer_task": current["reviewer_task"], "pages": current["pages"]}
            replace_report(m, old, save(current))

        run("changed_source_without_current_basis", no_current_basis, "reject", "7.25 所在整組 current report 只留身分及逐頁 ID/hash/figures/pass；原始 session 是舊稿，未要求同 reader 新稿複查／估時與原證據鏈。")

        def untracked_checkpoint_gap(m):
            session = page(m)["reader"]["sessions"][0]
            note = read(session["notes"][-1])
            note["concept_checks"][0]["verdict"] = "需補說明"
            note["concept_checks"][0]["need"] = "必要機制仍不能理解，尚無本人後補複查。"
            note["issues"] = [{"id": "AUDIT-UNRESOLVED", "necessity": "necessary", "status": "unresolved"}]
            session["notes"][-1] = save(note)

        run("raw_checkpoint_issue_omitted_from_dispositions", untracked_checkpoint_gap, "reject", "原始 checkpoint 新增明列必要未解 issue，未增加任何處置或收束引用；原始判斷無須改成pass，但必須連到後續處置。")

        def implementation_omission(m):
            m["implementation"] = {"AGENTS.md": digest(ROOT / "AGENTS.md")}

        run("implementation_inventory_can_be_omitted", implementation_omission, "reject", "刪除全部新增工具與既定 implementation hash，只留無關非空 entry，完整性未受檢。")

        def stage_overlap(m):
            seal = read(m["review_waves"][0]["seals"]["reader"])
            technical = {"role": "technical", "status": "completed", "started_at": seal["started_at"], "completed_at": seal["completed_at"], "report_sha256s": []}
            m["review_waves"][0]["seals"]["technical"] = save(technical)

        run("overlapping_stage_seals", stage_overlap, "reject", "technical seal 刻意與 reader 重疊；順序檢查應拒絕。", "technical")

        supplement = subprocess.run(
            [sys.executable, "docs/course-revision-20261016/tools/reader.py", "supplement", "7.23", "--reviewer", "/root/tool_audit_unstarted"],
            cwd=ROOT, capture_output=True, check=False,
        )
        freeze01 = json.loads((ROOT / "docs/course-revision-20261016/freeze-01.json").read_bytes())
        supplement_page = next(p for p in freeze01["pages"] if p["page_id"] == "7.23")
        supplement_raw = (ROOT / supplement_page["snapshot"]).read_bytes()
        results.append({
            "id": "supplement_before_any_checkpoint",
            "expected_contract_result": "reject",
            "actual_status": "accepted_full_page" if supplement.returncode == 0 and supplement.stdout == supplement_raw + b"\n" else "rejected_or_different",
            "failures": [] if supplement.returncode == 0 else [supplement.stderr.decode()],
            "returncode": supplement.returncode,
            "stdout_bytes": len(supplement.stdout),
            "stdout_sha256": hashlib.sha256(supplement.stdout).hexdigest(),
            "full_source_sha256": supplement_page["source_sha256"],
            "details": "沒有此 reviewer 的 session 或 checkpoint，supplement 仍直接輸出完整正文與折疊選讀；整組主文完成條件未執行。未持久保存教材全文。",
        })

        # 只用硬連結建立暫存副本；任何修改先 unlink，不接觸原始 bytes。
        clone = work / "repository"
        clone.mkdir()
        for name in ("docs", "course", "notebooks", "scripts", ".agents", "assets"):
            shutil.copytree(ROOT / name, clone / name, copy_function=os.link, symlinks=True)
        for source in ROOT.iterdir():
            if source.is_file() and not source.is_symlink():
                os.link(source, clone / source.name)
        notebook_manifest = copy.deepcopy(original)
        notebook_freeze = read(notebook_manifest["source_freeze"])
        frozen_page = next(p for p in notebook_freeze["pages"] if p["page_id"] == "7.23")
        notebook_path = clone / frozen_page["notebook"]
        notebook_before = digest(notebook_path)
        notebook_data = json.loads(notebook_path.read_bytes())
        notebook_data["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["raise RuntimeError('audit synthetic changed Notebook')\n"]})
        notebook_path.unlink()
        notebook_path.write_text(json.dumps(notebook_data) + "\n", encoding="utf-8")
        frozen_page["notebook_sha256"] = digest(notebook_path)
        freeze_path = clone / "docs/course-revision-20261016/tool-audit-synthetic-freeze.json"
        freeze_path.write_text(json.dumps(notebook_freeze), encoding="utf-8")
        notebook_manifest["source_freeze"] = {"path": freeze_path.relative_to(clone).as_posix(), "sha256": digest(freeze_path)}
        manifest_path = clone / "docs/course-revision-20261016/tool-audit-synthetic-manifest.json"
        manifest_path.write_text(json.dumps(notebook_manifest), encoding="utf-8")
        notebook_result = review.audit_manifest(clone, manifest_path, "reader")
        results.append({
            "id": "reviewed_page_notebook_changes_without_role_review",
            "expected_contract_result": "reject",
            "actual_status": notebook_result["status"],
            "failures": notebook_result["failures"],
            "record_count": len(notebook_result["records"]),
            "notebook": frozen_page["notebook"],
            "notebook_before_sha256": notebook_before,
            "notebook_after_sha256": frozen_page["notebook_sha256"],
            "original_notebook_unchanged": digest(ROOT / frozen_page["notebook"]) == notebook_before,
            "details": "只在暫存副本為新增 7.23 Notebook 加一個必定 raise 的 code cell，作者更新 source freeze notebook_sha256，未增加任何角色複查；既有正文與 reader report 不變仍通過。此新 Notebook 不屬舊 baseline，沿用已實際重播結果。",
        })

        reader = load_module("independent_reader_display", ROOT / "docs/course-revision-20261016/tools/reader.py")
        reader.ROOT = clone
        body = "## 99.1 圖版本 fixture\n\n![fixture](../figures/audit-figure.svg)\n"
        fixture_source = clone / "docs/course-revision-20261016/tool-audit-figure-source.md"
        fixture_source.write_text(body, encoding="utf-8")
        fixture_figure = clone / "course/figures/audit-figure.svg"
        fixture_figure.write_text("<svg>changed live</svg>\n", encoding="utf-8")
        frozen_figure = clone / "docs/course-revision-20261016/audit-frozen-figure.svg"
        frozen_figure.write_text("<svg>original frozen</svg>\n", encoding="utf-8")
        fixture_freeze = clone / "docs/course-revision-20261016/tool-audit-figure-freeze.json"
        fixture_freeze.write_text(json.dumps({
            "pages": [{"page_id": "99.1", "source": "course/chapters/99.md", "snapshot": fixture_source.relative_to(clone).as_posix(), "source_sha256": digest(fixture_source), "figures_sha256": {"course/figures/audit-figure.svg": digest(frozen_figure)}}],
            "figure_snapshots": {"course/figures/audit-figure.svg": {"path": frozen_figure.relative_to(clone).as_posix(), "sha256": digest(frozen_figure)}},
        }), encoding="utf-8")
        state = {"session": "fictional", "page_id": "99.1", "next_index": 0, "source_freeze": {"path": fixture_freeze.relative_to(clone).as_posix(), "sha256": digest(fixture_freeze)}}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            reader.display(state)
        display_output = buffer.getvalue()
        results.append({
            "id": "reader_displays_live_instead_of_frozen_figure",
            "expected_contract_result": "reject",
            "actual_status": "accepted_live_figure_path" if "本段圖：" in display_output and "course/chapters/../figures/audit-figure.svg" in display_output else "different",
            "failures": [],
            "display_stdout_sha256": hashlib.sha256(display_output.encode()).hexdigest(),
            "live_figure_sha256": digest(fixture_figure),
            "frozen_figure_sha256": digest(frozen_figure),
            "details": "來源 freeze 與凍結圖 reference 正確，但 display 未驗 live 圖 hash，仍指示 reader 開啟不同 bytes 的 live 圖；未指向 freeze.figure_snapshots。此 probe 不聲稱實看圖。",
        })

        generated = clone / "docs/course-revision-20261016/tool-audit-generated-freeze.json"
        freeze_command = [sys.executable, str(clone / "docs/course-revision-20261016/tools/freeze_sources.py"), "--previous", str(clone / original["source_freeze"]["path"]), "--output", str(generated)]
        frozen = subprocess.run(freeze_command, cwd=clone, capture_output=True, text=True, check=False)
        generated_value = json.loads(generated.read_bytes()) if frozen.returncode == 0 else {}
        image_refs = generated_value.get("figure_snapshots", {})
        image_valid = all(digest(clone / ref["path"]) == ref["sha256"] for ref in image_refs.values())
        results.append({
            "id": "freeze_sources_generation_control",
            "expected_contract_result": "pass",
            "actual_status": "passed" if frozen.returncode == 0 and len(generated_value.get("pages", [])) == 350 and image_valid else "failed",
            "failures": [] if frozen.returncode == 0 and image_valid else [frozen.stderr],
            "command": "python <temporary repository>/docs/course-revision-20261016/tools/freeze_sources.py --previous <temporary repository>/docs/course-revision-20261016/freeze-08.json --output <new temporary freeze>",
            "returncode": frozen.returncode,
            "pages": len(generated_value.get("pages", [])),
            "figure_snapshots": len(image_refs),
            "figure_snapshot_hashes_valid": image_valid,
            "details": "在暫存副本實際執行 freeze，驗證 canonical 數與所有圖快照 bytes；無訓練。",
        })
        duplicate_freeze = subprocess.run(freeze_command, cwd=clone, capture_output=True, text=True, check=False)
        results.append({
            "id": "freeze_existing_output_is_immutable",
            "expected_contract_result": "reject",
            "actual_status": "rejected" if duplicate_freeze.returncode != 0 else "accepted",
            "failures": [duplicate_freeze.stderr] if duplicate_freeze.returncode else [],
            "returncode": duplicate_freeze.returncode,
            "details": "相同 freeze output 第二次執行確實拒絕覆寫。",
        })

        active_path = clone / "docs/course-reviews-active.json"
        if active_path.exists():
            active_path.unlink()
        active_path.write_text(json.dumps({"schema_version": 2, "review_policy": "tutorial_incremental_v2", "manifest": "docs/missing-incremental-manifest.json", "manifest_sha256": "0" * 64}), encoding="utf-8")
        phase7 = load_module("independent_phase7_route", ROOT / "docs/review-tools/phase7_review.py")
        routed = phase7.check_active(clone, "reader")
        results.append({
            "id": "phase7_new_policy_route_fails_closed",
            "expected_contract_result": "reject",
            "actual_status": routed["status"],
            "failures": routed["failures"],
            "incremental_scope_returned": routed.get("scope_note") == review.SCOPE,
            "details": "暫存 active pointer 指向新 policy 的缺檔 manifest；phase7 確實走 incremental 路由並 fail closed，未切換真實發布 pointer。",
        })

    output = {
        "schema_version": 1,
        "reviewer_task": "/root/tech_20261016_review_tools",
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; metadata/file checks only"},
        "command": "uv run python docs/technical-reviews/artifacts/revision-20261016/tool-audit-probes.py",
        "inputs_sha256": {str(MANIFEST.relative_to(ROOT)): digest(MANIFEST), "docs/review-tools/incremental_revision.py": digest(ROOT / "docs/review-tools/incremental_revision.py"), "docs/review-tools/baseline_replay.py": digest(ROOT / "docs/review-tools/baseline_replay.py")},
        "baseline": {"status": baseline["status"], "failures": baseline["failures"], "before_after_equal": before == after, "watched_files": len(before), "restoration_entries": len(entries)},
        "mutation_baseline_note": "每個 mutation 沿用同次已實際完整重播成功之 baseline；無略過其他 validator，暫存證據用完刪除。",
        "results": results,
        "unexpected_acceptances": [r["id"] for r in results if r["expected_contract_result"] == "reject" and not r["failures"]],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    target = ARTIFACTS / "tool-audit-probes-results.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": len(results), "unexpected_acceptances": output["unexpected_acceptances"], "baseline": output["baseline"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
