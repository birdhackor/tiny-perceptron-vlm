"""沿用經重播的舊證據，核對新增／受影響頁的實際獨立審閱。

只驗證保存的來源、身分聲明、逐段紀錄與證據契約；不產生審閱答案。
"""

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = "tutorial_incremental_v2"
ROLES = ("reader", "technical", "continuity")
SCOPE = (
    "Preserved baseline replay plus actual bounded independent reviews of changed sources and affected routes. "
    "Not a new full-course blind review, human test, proof of comprehension, scientific truth, physical isolation, "
    "image viewing, kernel execution, training success or deployment."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def local(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "expected relative artifact path")
    path = (root / name).resolve()
    require(path.is_relative_to(root), "artifact escapes repository")
    return path


def artifact(root, reference):
    require(isinstance(reference, dict), "missing artifact reference")
    path = local(root, reference.get("path"))
    raw = path.read_bytes()
    require(sha(raw) == reference.get("sha256"), "artifact changed: " + reference["path"])
    return raw


def document(root, reference):
    return json.loads(artifact(root, reference))


def node(data, pointer):
    require(isinstance(pointer, str) and (not pointer or pointer.startswith("/")), "invalid JSON pointer")
    for token in pointer.split("/")[1:] if pointer else []:
        key = token.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def at(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp requires timezone")
    return result


def owner(report):
    return report.get("reviewer_task", report.get("reviewer"))


def reader_session(root, reference, expected_owner):
    opening = document(root, reference["opening"])
    require(opening.get("reviewer_task") == expected_owner, "reader trace owner mismatch")
    page_raw = artifact(root, reference["source"])
    require(sha(page_raw) == opening.get("source_sha256"), "reader trace source mismatch")
    figures = reference.get("figures", {})
    require(set(figures) == set(opening.get("figures_sha256", {})), "reader figure provenance incomplete")
    for name, digest in opening.get("figures_sha256", {}).items():
        require(sha(artifact(root, figures[name])) == digest, "reader original figure changed")
    tool = module("revision_reader_units", root / "docs/course-revision-20261016/tools/reader.py")
    chunks = tool.units(page_raw.decode())
    notes = reference.get("notes", [])
    require(len(notes) == len(chunks), "missing sequential checkpoint")
    last = at(opening["started_at"])
    for i, (chunk, ref) in enumerate(zip(chunks, notes)):
        note = document(root, ref)
        require(note.get("unit_index") == i and note.get("page_id") == opening["page_id"], "wrong checkpoint unit")
        require(note.get("reviewer_task") == expected_owner, "checkpoint owner mismatch")
        require(note.get("unit_sha256") == sha(chunk.encode()), "checkpoint visible text mismatch")
        saved = at(note["saved_at"])
        require(saved >= last, "checkpoint chronology reversed")
        last = saved
        for field in (
            "understanding",
            "materials_and_labels",
            "expected_change",
            "confusion_and_quote",
            "missing_visuals",
        ):
            require(isinstance(note.get(field), str) and note[field].strip(), "empty reader observation")
        checks = note.get("concept_checks")
        require(isinstance(checks, list) and checks, "missing concept-level four questions")
        for check in checks:
            for field in (
                "concept",
                "identity",
                "mechanism",
                "need",
                "usage",
                "example_scope",
                "source_quote",
                "verdict",
            ):
                require(isinstance(check.get(field), str) and check[field].strip(), "incomplete concept observation")
    return opening["page_id"], last


def technical_reports(manifest, root):
    """讀取實際逐節報告；讓舊validator核對來源、算例及獨立身分。"""
    reports = {}
    owners = {}
    for page in manifest["pages"]:
        proof = page.get("technical", {})
        references = (
            proof.get("section_reports", [])
            if proof.get("mode") == "section-composite"
            else ([proof["report"]] if proof.get("mode") == "review" else [])
        )
        for reference in references:
            report = document(root, reference)
            key = report["lesson_id"]
            require(key not in reports, "duplicate technical section")
            reports[key] = report
            owners.setdefault(owner(report), []).append(key)
    return reports, owners


def reported_issues(value, pointer=""):
    """列出報告真正保存的問題位置，不靠作者的完整性布林值。"""
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key in {"issues", "new_issues"} and isinstance(item, list):
                result.extend(location + "/" + str(i) for i, issue in enumerate(item) if isinstance(issue, dict))
            else:
                result.extend(reported_issues(item, location))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            result.extend(reported_issues(item, pointer + "/" + str(i)))
    return result


def check_dispositions(root, manifest, selected):
    dispositions = document(root, manifest["dispositions"])
    require(dispositions.get("pending_count") == 0, "unresolved author dispositions")
    required = {}
    for page in manifest["pages"]:
        for role in selected:
            proof = page.get(role, {})
            for key in ("report", "original_report"):
                if key in proof:
                    required[proof[key]["sha256"]] = proof[key]
            for key in ("section_reports", "original_reports"):
                for ref in proof.get(key, []):
                    required[ref["sha256"]] = ref
    listed = {ref["sha256"]: ref for ref in manifest.get("issue_reports", [])}
    require(set(required) <= set(listed), "issue inventory omits a current or original report")
    expected = set()
    for digest, ref in listed.items():
        report = document(root, ref)
        for pointer in reported_issues(report):
            expected.add((digest, pointer))
    for extra in manifest.get("additional_issue_inventories", []):
        report = document(root, extra["report"])
        items = node(report, extra["json_pointer"])
        require(isinstance(items, list), "additional issues must be a list")
        for i in range(len(items)):
            expected.add((extra["report"]["sha256"], extra["json_pointer"] + "/" + str(i)))
    actual = set()
    for decision in dispositions.get("items", []):
        key = (decision["report"]["sha256"], decision["json_pointer"])
        require(key not in actual, "duplicate issue disposition")
        actual.add(key)
        node(document(root, decision["report"]), decision["json_pointer"])
        for field in ("source_support", "learning_benefit", "change_burden", "reason"):
            require(isinstance(decision.get(field), str) and decision[field].strip(), "incomplete author adjudication")
        require(decision.get("final_necessity") in {"necessary", "optional"}, "issue necessity not adjudicated")
        require(decision.get("action") in {"fixed", "keep", "defer", "reject"}, "issue still pending")
        require(decision["final_necessity"] != "necessary" or decision["action"] == "fixed", "necessary gap unresolved")
        if decision["action"] == "fixed":
            require(decision.get("verification"), "fix requires actual verification references")
        for reference in decision.get("verification", []):
            artifact(root, reference)
    require(actual == expected, "reported issues and dispositions differ")


def audit_manifest(root, path, stage="all"):
    root = Path(root).resolve()
    failures, records = [], []
    try:
        require(stage in {"reader", "technical", "all"}, "unknown review stage")
        manifest = json.loads(Path(path).read_bytes())
        require(manifest.get("review_policy") == POLICY and manifest.get("schema_version") == 2, "wrong policy")
        baseline = module("revision_baseline_replay", root / "docs/review-tools/baseline_replay.py").replay(root)
        require(not baseline["failures"], "preserved baseline failed: " + repr(baseline["failures"]))
        original = document(root, manifest["baseline_freeze"])
        original_policy = json.loads((root / "docs/course-publish-20261008/review-manifest.json").read_bytes())
        require(
            artifact(root, manifest["baseline_freeze"]) == artifact(root, original_policy["source_freeze"]),
            "baseline identity is not the replayed source freeze",
        )
        freeze = document(root, manifest["source_freeze"])
        inventory = module("revision_inventory", root / "docs/review-tools/composite_review.py").current_inventory(root)
        old = {p["page_id"]: p for p in original["pages"]}
        live = {p["page_id"]: p for p in inventory["pages"]}
        frozen = {p["page_id"]: p for p in freeze["pages"]}
        catalog = {p["page_id"]: p for p in manifest["pages"]}
        require(
            len(live) == len(inventory["pages"]) == len(frozen) == len(catalog), "duplicate or missing canonical page"
        )
        require(
            set(live) == set(frozen) == set(catalog) and set(old) <= set(live),
            "canonical scope lost or added without evidence",
        )
        for kind in ("criteria", "implementation"):
            entries = manifest.get(kind, {})
            require(entries, "missing " + kind + " freeze")
            if kind == "criteria":
                required = {".agents/skills/clear-tutorial/" + name for name in original["criteria"]}
                require(set(entries) == required, "incomplete tutorial criteria")
            for name, digest in entries.items():
                require(sha(local(root, name).read_bytes()) == digest, kind + " drift: " + name)
        authors = set(manifest.get("author_tasks", []))
        require("/root" in authors, "missing actual author exclusions")
        assignments = document(root, manifest["assignments"])
        task_roles = {}
        for assignment in assignments["tasks"]:
            task, role = assignment["task"], assignment["role"]
            require(task.startswith("/root/") and task not in authors, "invalid independent task")
            require(role in ROLES and task not in task_roles, "task reused across assignments")
            require(assignment.get("context") == "fresh", "review task context is not fresh")
            task_roles[task] = role
        reports, technical_owners = technical_reports(manifest, root)
        technical = module("revision_factual_validator", root / "scripts/check_technical_reviews.py")
        reader_tasks = {task for task, role in task_roles.items() if role == "reader"}
        selected = ROLES[:1] if stage == "reader" else ROLES[:2] if stage == "technical" else ROLES
        waves = manifest.get("review_waves", [])
        require(waves, "missing actual review chronology")
        sealed = {}
        preceding_wave_end = None
        for wave in waves:
            require(wave["id"] not in sealed, "duplicate review wave")
            sealed[wave["id"]] = {}
            previous_end = preceding_wave_end
            for role in selected:
                seal = document(root, wave["seals"][role])
                require(seal.get("role") == role and seal.get("status") == "completed", "stage is incomplete")
                require(at(seal["started_at"]) <= at(seal["completed_at"]), "stage duration reversed")
                if previous_end:
                    require(at(seal["started_at"]) >= previous_end, "review stages overlapped")
                previous_end = at(seal["completed_at"])
                sealed[wave["id"]][role] = seal
            preceding_wave_end = previous_end
        for assignment in assignments["tasks"]:
            if assignment["role"] not in selected:
                continue
            seal = sealed[assignment["wave"]][assignment["role"]]
            window = assignment["dispatch_window"]
            require(
                at(seal["started_at"])
                <= at(window["not_before"])
                <= at(window["not_after"])
                <= at(seal["completed_at"]),
                "task dispatch window outside stage interval",
            )
        for pid, page in live.items():
            for field in ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256"):
                require(
                    page.get(field) == frozen[pid].get(field), "current source/figure/notebook freeze stale: " + pid
                )
            require(
                sha(artifact(root, {"path": frozen[pid]["snapshot"], "sha256": page["source_sha256"]}))
                == page["source_sha256"],
                "missing source snapshot",
            )
            inherited_identity = pid in old and all(
                page.get(f) == old[pid].get(f) for f in ("source", "selector", "source_sha256", "figures_sha256")
            )
            row = catalog[pid]
            for role in selected:
                proof = row.get(role, {})
                if proof.get("mode") == "inherit":
                    require(inherited_identity, "changed source cannot inherit old verdict: " + pid)
                    if page.get("notebook"):
                        old_notebook = old[pid].get("notebook", {})
                        require(
                            old_notebook.get("live_sha256") == page["notebook_sha256"],
                            "changed Notebook cannot inherit",
                        )
                    require(proof.get("baseline_page_id") == pid, "inherited page mismatch")
                    require(
                        isinstance(proof.get("dependency_note"), str) and proof["dependency_note"].strip(),
                        "inheritance needs scope/route disposition",
                    )
                    continue
                if role == "technical" and proof.get("mode") == "section-composite":
                    require(
                        pid in old and page["selector"] == "whole-file", "section composite needs original whole page"
                    )
                    from scripts import reading_time

                    old_text = local(root, old[pid]["snapshot"]).read_text()
                    live_text = local(root, frozen[pid]["snapshot"]).read_text()
                    old_sections = reading_time.lesson_slices(old_text)
                    live_sections = reading_time.lesson_slices(live_text)
                    require(set(old_sections) <= set(live_sections), "original numbered section removed")
                    require(
                        reading_time.chapter_introduction(old_text) == reading_time.chapter_introduction(live_text),
                        "changed page introduction needs separate factual review",
                    )
                    changed = {key for key, body in live_sections.items() if body != old_sections.get(key)}
                    supplied = {document(root, ref)["lesson_id"] for ref in proof["section_reports"]}
                    require(changed == supplied, "section factual reviews do not cover exact changed sections")
                    for key in changed:
                        report = reports[key]
                        require(task_roles.get(owner(report)) == role, "wrong section reviewer role")
                        errors = technical._validate(
                            root,
                            local(root, page["source"]),
                            key,
                            live_sections[key],
                            report,
                            technical_owners,
                            reader_tasks,
                        )
                        require(not errors, key + ": " + repr(errors))
                    seal = sealed[proof["wave"]][role]
                    require(
                        all(ref["sha256"] in seal["report_sha256s"] for ref in proof["section_reports"]),
                        "section reviews were not collected before stage closure",
                    )
                    continue
                require(proof.get("mode") == "review", "missing current role evidence: " + pid + "/" + role)
                report = document(root, proof["report"])
                require(
                    proof["report"]["sha256"] in sealed[proof["wave"]][role]["report_sha256s"],
                    "report was not collected before stage closure",
                )
                task = owner(report)
                require(task_roles.get(task) == role, "undeclared or wrong-role reviewer")
                current = node(report, proof.get("json_pointer", ""))
                require(current.get("page_id", current.get("lesson_id")) == pid, "report page membership mismatch")
                require(current.get("source_sha256") == page["source_sha256"], "report source is stale: " + pid)
                require(current.get("verdict") in {"pass", "已足夠"}, "reviewer requests revision: " + pid)
                figures = current.get("figures_sha256", current.get("figure_sha256"))
                require(figures == page["figures_sha256"], "review figure identity incomplete: " + pid)
                require(proof.get("original_report"), "original judgment must be retained")
                artifact(root, proof["original_report"])
                if role == "reader":
                    sessions = proof.get("sessions", [])
                    require(sessions, "no actual sequential first-read trace")
                    for reference in sessions:
                        read_pid, _ = reader_session(root, reference, task)
                        require(read_pid == pid, "reader session belongs to another page")
                if role == "technical":
                    body = local(root, frozen[pid]["snapshot"]).read_text()
                    errors = technical._validate(
                        root, local(root, page["source"]), pid, body, report, technical_owners, reader_tasks
                    )
                    require(not errors, pid + ": " + repr(errors))
            records.append({"page_id": pid, "source_sha256": page["source_sha256"]})
        check_dispositions(root, manifest, selected)
        if stage == "all":
            dependency = document(root, manifest["dependency_audit"])
            require(
                dependency.get("verdict") == "pass" and task_roles.get(owner(dependency)) == "continuity",
                "missing independent dependency audit",
            )
            require(set(dependency.get("inventory_page_ids", [])) == set(live), "dependency audit inventory incomplete")
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
        failures.append(str(error))
    return {
        "schema_version": 2,
        "review_policy": POLICY,
        "stage": stage,
        "status": "failed" if failures else "passed_metadata_checks",
        "records": records,
        "failures": failures,
        "scope_note": SCOPE,
    }


def check_active(root, active, stage="all"):
    try:
        require(active.get("review_policy") == POLICY and active.get("schema_version") == 2, "invalid active policy")
        artifact(Path(root).resolve(), {"path": active.get("manifest"), "sha256": active.get("manifest_sha256")})
        return audit_manifest(root, Path(root) / active["manifest"], stage)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"status": "failed", "failures": [str(error)], "scope_note": SCOPE}
