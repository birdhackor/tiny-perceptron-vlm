"""以實際新審承接已結案的歷史修正，不改寫舊處置或舊 verification。

只核對保存的身分、時序、版本、來源與明示判斷；不能證明代理真的閱讀、
工具事件真的發生或語意判斷正確。完整技術報告仍交給既有 _validate。
沒有 sidecar 時，呼叫者保留原來的嚴格失敗。
"""

from datetime import UTC, datetime
from pathlib import Path


def identity(reference):
    return reference["path"], reference["sha256"]


def issue_key(decision):
    return (*identity(decision["report"]), decision["json_pointer"])


def verification_key(verification):
    return (*identity(verification["report"]), verification.get("json_pointer", ""), verification.get("kind", "page"))


def key(decision, verification):
    return issue_key(decision), verification_key(verification)


class Rechecks:
    def __init__(self, root, manifest, live, tasks, sealed, api):
        self.root, self.manifest, self.live = Path(root).resolve(), manifest, live
        self.tasks, self.sealed, self.api = tasks, sealed, api
        self.entries, self.used, self.reports = {}, set(), []
        self.checked_readers = set()
        sidecar = self.doc(manifest["historical_fix_rechecks"])
        self.need(type(sidecar.get("schema_version")) is int and sidecar["schema_version"] == 1, "wrong sidecar schema")
        self.prior(sidecar["prior_active_pointer"])
        assignments = self.doc(manifest["assignments"])["tasks"]
        self.assignments = {item["task"]: item for item in assignments}
        self.need(len(self.assignments) == len(assignments), "duplicate current assignment")
        current = self.doc(manifest["dispositions"])
        self.decisions = {issue_key(item): item for item in current["items"]}
        self.need(len(self.decisions) == len(current["items"]), "duplicate current disposition")
        self.frozen = {p["page_id"]: p for p in self.doc(manifest["source_freeze"])["pages"]}
        groups = sidecar.get("rechecks")
        self.need(isinstance(groups, list) and groups, "empty rechecks")
        for group in groups:
            self.group(group)

    def need(self, condition, message):
        self.api.require(condition, "historical recheck: " + message)

    def doc(self, ref):
        value = self.api.document(self.root, ref)
        self.need(isinstance(value, dict), "expected JSON object artifact")
        return value

    def text(self, value, label):
        self.need(isinstance(value, str) and value.strip(), "missing " + label)

    def mapping(self, rows, field, label):
        self.need(isinstance(rows, list), label + " must be a list")
        result = {row[field]: row for row in rows}
        self.need(len(result) == len(rows), "duplicate " + label)
        return result

    def prior(self, pointer_ref):
        # 由目前 active 的真實 bytes / previous_pointer 鏈認證保存的 pointer；
        # activation 之後仍可沿同一條歷史鏈核對，不把新 active 誤當舊 pointer。
        self.need(pointer_ref["path"] != "docs/course-reviews-active.json", "prior pointer must be an immutable backup")
        saved = self.api.artifact(self.root, pointer_ref)
        raw = (self.root / "docs/course-reviews-active.json").read_bytes()
        seen = set()
        for _ in range(100):
            if raw == saved:
                break
            digest = self.api.sha(raw)
            self.need(digest not in seen, "cyclic active pointer history")
            seen.add(digest)
            import json

            pointer = json.loads(raw)
            self.need(pointer.get("previous_pointer"), "prior backup is not in active pointer history")
            raw = self.api.artifact(self.root, pointer["previous_pointer"])
        else:
            self.need(False, "active pointer history too deep")
        pointer = self.doc(pointer_ref)
        self.need(
            pointer.get("schema_version") == 2 and pointer.get("review_policy") == self.api.POLICY,
            "invalid prior pointer",
        )
        self.prior_activated = self.api.at(pointer["activated_at"])
        self.prior_manifest = self.doc({"path": pointer["manifest"], "sha256": pointer["manifest_sha256"]})
        self.need(
            self.prior_manifest.get("schema_version") == 2
            and self.prior_manifest.get("review_policy") == self.api.POLICY,
            "invalid prior manifest",
        )
        gate = self.doc(pointer["actual_completed_gate"])
        self.need(
            gate.get("schema_version") == 2
            and gate.get("review_policy") == self.api.POLICY
            and gate.get("stage") == "all"
            and gate.get("status") == "passed_metadata_checks"
            and gate.get("failures") == [],
            "prior gate did not complete",
        )
        freeze = self.doc(self.prior_manifest["source_freeze"])
        self.old_pages = self.mapping(freeze["pages"], "page_id", "prior pages")
        records = self.mapping(gate["records"], "page_id", "prior gate records")
        self.need(set(records) == set(self.old_pages), "prior gate inventory differs from freeze")
        self.need(
            all(records[pid].get("source_sha256") == row["source_sha256"] for pid, row in self.old_pages.items()),
            "prior gate source differs from freeze",
        )
        self.old_figures = freeze.get("figure_snapshots", {})
        self.old_disposition_ref = self.prior_manifest["dispositions"]
        self.old_dispositions = self.doc(self.old_disposition_ref)
        self.need(self.old_dispositions.get("pending_count") == 0, "prior dispositions were not closed")
        self.old_tasks = self.mapping(
            self.doc(self.prior_manifest["assignments"])["tasks"], "task", "prior assignments"
        )

    def collection(self, proof, references):
        role, wave = proof["collection_role"], proof["wave"]
        self.need(role in {"technical", "continuity"}, "recheck collected before technical phase")
        phase = self.sealed[wave][role]
        self.need(phase.get("status") == "completed", "collection phase incomplete")
        received = self.api.at(proof["received_at"])
        self.need(
            self.api.at(phase["started_at"]) <= received <= self.api.at(phase["completed_at"]) <= datetime.now(UTC),
            "collection outside actual phase",
        )
        entries = phase.get("collection_index", [])
        for ref in references:
            self.api.artifact(self.root, ref)
            self.need(ref["sha256"] in phase["report_sha256s"], "report not sealed")
            matches = [item for item in entries if identity(item) == identity(ref)]
            self.need(len(matches) == 1, "report missing or duplicated in actual collection_index")
            when = self.api.at(matches[0]["received_at"])
            self.need(when == received, "claimed appendix receipt differs from collection_index")
        return received

    def unavailable(self, evidence, old_task, appendix_at):
        document = self.doc(evidence["report"])
        self.need(
            document.get("author_task") in self.manifest["author_tasks"], "unavailability recorder is not declared"
        )
        if "prior_observations" in document:
            self.api.artifact(self.root, document["prior_observations"])
        record = self.api.node(document, evidence["json_pointer"])
        self.need(isinstance(record, dict), "unavailability record must be an object")
        self.need(
            record.get("reviewer_task") == old_task and record.get("unavailable") is True,
            "original reviewer not explicitly unavailable",
        )
        self.text(record.get("reason"), "unavailability reason")
        recorded = self.api.at(record["recorded_at"])
        self.need(self.prior_activated <= recorded <= appendix_at, "unavailability receipt chronology invalid")
        events = record.get("events", [])
        self.need(
            isinstance(events, list) and all(isinstance(event, dict) for event in events),
            "invalid unavailability events",
        )
        followups = [
            event
            for event in events
            if event.get("tool") == "collaboration.followup_task" and event.get("request", {}).get("target") == old_task
        ]
        self.need(len(followups) == 1, "missing exact original-task followup receipt")
        response = followups[0].get("response")
        path_missing = isinstance(response, str) and old_task in response and "not found" in response
        name_rejected = response == "agent_name must use only lowercase letters, digits, and underscores"
        self.need(path_missing or name_rejected, "followup receipt does not establish the declared failure")
        # invalid-name 並不等於 path-not-found；兩者都另要求實際的空 task-list 回應。
        listings = [event for event in events if event.get("tool") == "collaboration.list_agents"]
        self.need(
            any(
                isinstance(event.get("request", {}).get("path_prefix"), str)
                and event["request"]["path_prefix"]
                and old_task.startswith(event["request"]["path_prefix"])
                and isinstance(event.get("response"), dict)
                and event["response"].get("agents") == []
                for event in listings
            ),
            "missing empty original-task-prefix listing",
        )

    def group(self, group):
        kind = group["kind"]
        self.need(kind in {"page", "tool_audit"}, "unknown recheck kind")
        full_ref, appendix_ref = group["full_review"], group["appendix"]
        full, appendix = self.doc(full_ref), self.doc(appendix_ref)
        self.reports.extend([full_ref, appendix_ref])
        task = self.api.owner(full)
        self.need(task == self.api.owner(appendix), "full review and appendix owner differ")
        assignment = self.assignments[task]
        self.need(
            isinstance(task, str)
            and task.startswith("/root/")
            and task not in self.manifest["author_tasks"]
            and self.tasks.get(task) == "technical"
            and assignment.get("role") == "technical"
            and assignment.get("context") == "fresh",
            "replacement reviewer is not independent fresh technical",
        )
        collection = group["collection"]
        self.need(collection["wave"] == assignment["wave"], "recheck outside actual assignment wave")
        received = self.collection(collection, [appendix_ref])
        phase = self.sealed[assignment["wave"]]["technical"]
        window = assignment["dispatch_window"]
        lo, hi = self.api.at(window["not_before"]), self.api.at(window["not_after"])
        self.need(
            self.api.at(phase["started_at"]) <= lo <= hi <= self.api.at(phase["completed_at"]),
            "replacement dispatch outside technical phase",
        )
        appendix_at = self.api.at(appendix.get("created_at", appendix.get("at")))
        self.need(lo <= appendix_at <= received, "appendix chronology invalid")
        self.need(full.get("verdict") == "pass" and appendix.get("verdict") == "pass", "current recheck does not pass")
        rows = appendix.get("historical_issue_rechecks")
        self.need(isinstance(rows, list) and rows, "appendix lacks historical issue rechecks")
        if kind == "page":
            pid = group["page_id"]
            catalog = self.mapping(self.manifest["pages"], "page_id", "current pages")
            proof = catalog[pid]["technical"]
            self.need(
                proof.get("mode") == "review"
                and identity(proof["report"]) == identity(full_ref)
                and proof.get("json_pointer", "") == "",
                "full review is not current whole-page technical proof",
            )
            self.need(
                identity(full_ref) != identity(appendix_ref) and not full.get("historical_issue_rechecks"),
                "historical appendix cannot replace full review",
            )
            self.need(
                assignment.get("lesson_id") == pid and self.api.owner(self.doc(proof["original_report"])) == task,
                "full review lacks own original page review",
            )
            original_ref = proof["original_report"]
            original = self.doc(original_ref)
            self.need(
                identity(proof["original_report"]) != identity(appendix_ref)
                and not original.get("historical_issue_rechecks")
                and original.get("schema_version") == 1
                and original.get("review_stage") == "technical"
                and original.get("reviewer_context") == "fresh"
                and original.get("lesson_id") == pid
                and original.get("source_sha256"),
                "historical appendix cannot be original full-page review",
            )
            self.need(proof["wave"] == assignment["wave"], "full review assignment wave mismatch")
            full_phase = self.sealed[proof["wave"]][proof.get("collection_role", "technical")]
            self.need(
                full_ref["sha256"] in full_phase["report_sha256s"]
                and proof["original_report"]["sha256"] in self.sealed[proof["wave"]]["technical"]["report_sha256s"],
                "full/original review not sealed",
            )
            self.validate_page(pid, task, full, appendix)
        else:
            pid = None
            proof = self.manifest["tool_audit"]
            self.need(
                identity(proof["report"]) == identity(full_ref) and proof["wave"] == assignment["wave"],
                "full audit is not current tool audit",
            )
            self.need(assignment.get("lesson_id") == "__review_tools__", "tool reviewer lacks actual tool assignment")
            self.need(
                identity(full_ref) != identity(appendix_ref) and not full.get("historical_issue_rechecks"),
                "historical appendix cannot replace full tool audit",
            )
            self.need(group.get("original_review"), "missing original tool audit")
            original_ref = group["original_review"]
            original = self.doc(original_ref)
            self.reports.append(original_ref)
            self.need(
                identity(original_ref) != identity(appendix_ref)
                and not original.get("historical_issue_rechecks")
                and self.api.owner(original) == task
                and original.get("verdict") in {"pass", "revise"},
                "historical appendix cannot be original full tool audit",
            )
            original_files = self.mapping(original.get("checked_files"), "path", "original tool checked files")
            full_files = self.mapping(full.get("checked_files"), "path", "full tool checked files")
            self.need(
                full_files and set(full_files) <= set(original_files), "original tool audit lacks complete file scope"
            )
            self.api.check_tool_audit(self.root, self.manifest, self.tasks, self.sealed)
        full_phase = self.sealed[proof["wave"]][proof.get("collection_role", "technical")]
        full_receipts = [
            item for item in full_phase.get("collection_index", []) if identity(item) == identity(full_ref)
        ]
        self.need(
            len(full_receipts) == 1 and self.api.at(full_receipts[0]["received_at"]) <= received,
            "missing full-review receipt before appendix receipt",
        )
        full_received = self.api.at(full_receipts[0]["received_at"])
        self.need(
            self.api.at(full_phase["started_at"]) <= full_received <= self.api.at(full_phase["completed_at"])
            and lo <= full_received,
            "full review receipt outside actual phase/dispatch",
        )
        original_receipts = [
            item for item in phase.get("collection_index", []) if identity(item) == identity(original_ref)
        ]
        self.need(
            original_ref["sha256"] in phase["report_sha256s"]
            and len(original_receipts) == 1
            and lo
            <= self.api.at(original_receipts[0]["received_at"])
            <= min(full_received, self.api.at(phase["completed_at"])),
            "original full review missing/belated receipt",
        )
        for field in ("created_at", "at", "confirmed_at"):
            if field in full:
                self.need(self.api.at(full[field]) <= appendix_at, "appendix predates full review")
        for row in rows:
            self.closed_item(group, row, full, appendix, task, pid, appendix_at)

    def validate_page(self, pid, task, full, appendix):
        live, frozen = self.live[pid], self.frozen[pid]
        self.need(
            all(
                live.get(field) == frozen.get(field)
                for field in ("source", "selector", "source_sha256", "figures_sha256")
            ),
            "current page freeze stale",
        )
        body = self.api.artifact(self.root, {"path": frozen["snapshot"], "sha256": live["source_sha256"]}).decode()
        technical = self.api.module("historical_factual_validator", self.root / "scripts/check_technical_reviews.py")
        readers = {task for task, role in self.tasks.items() if role == "reader"}
        for report in (full, appendix):
            self.need(report.get("figure_sha256") == live["figures_sha256"], "current figure identity differs")
            errors = technical._validate(
                self.root, self.api.local(self.root, live["source"]), pid, body, report, {task: [pid]}, readers
            )
            self.need(not errors, "current full/appendix technical validation failed: " + repr(errors))

    def validate_reader(self, pid, technical_task, wave):
        """舊 reader 的修正另需同波真正的完整 fresh reader；沿用現有 trace/closure 契約。"""
        if (pid, wave) in self.checked_readers:
            return
        catalog = self.mapping(self.manifest["pages"], "page_id", "current pages")
        proof = catalog[pid].get("reader", {})
        self.need(proof.get("mode") == "review" and proof.get("wave") == wave, "missing same-wave fresh reader proof")
        report = self.doc(proof["report"])
        task = self.api.owner(report)
        assignment = self.assignments.get(task, {})
        self.need(
            isinstance(task, str)
            and task.startswith("/root/")
            and task not in self.old_tasks
            and task not in self.manifest["author_tasks"]
            and task != technical_task
            and self.tasks.get(task) == "reader"
            and assignment.get("role") == "reader"
            and assignment.get("context") == "fresh"
            and assignment.get("wave") == wave
            and pid in assignment.get("pages", []),
            "current reader is not independent fresh assigned reader",
        )
        reader = self.api.module("historical_reader_validator", self.root / "docs/review-tools/incremental_revision.py")
        current = reader.page_member(report, proof.get("json_pointer", ""), pid, task)
        self.need(
            current.get("verdict") in {"pass", "已足夠"}
            and current.get("source_sha256") == self.live[pid]["source_sha256"]
            and current.get("figures_sha256") == self.live[pid]["figures_sha256"],
            "fresh reader current source/figure/verdict stale",
        )
        phase = self.sealed[wave]["reader"]
        window = assignment["dispatch_window"]
        lo, hi = self.api.at(window["not_before"]), self.api.at(window["not_after"])
        self.need(
            self.api.at(phase["started_at"]) <= lo <= hi <= self.api.at(phase["completed_at"]),
            "fresh reader dispatch outside reader phase",
        )

        def receipt(ref, role):
            self.need(role in {"reader", "technical", "continuity"}, "unknown fresh reader collection role")
            seal = self.sealed[wave][role]
            entries = [entry for entry in seal.get("collection_index", []) if identity(entry) == identity(ref)]
            self.need(
                seal.get("status") == "completed" and ref["sha256"] in seal["report_sha256s"] and len(entries) == 1,
                "fresh reader report not actually sealed/collected",
            )
            when = self.api.at(entries[0]["received_at"])
            self.need(
                self.api.at(seal["started_at"]) <= when <= self.api.at(seal["completed_at"]) <= datetime.now(UTC)
                and lo <= when,
                "fresh reader receipt outside phase/dispatch",
            )
            data = self.doc(ref)
            self.need(self.api.owner(data) == task, "fresh reader original/basis/closure owner differs")
            for field in ("created_at", "confirmed_at"):
                if field in data:
                    self.need(self.api.at(data[field]) <= when, "fresh reader report predates actual creation")
            self.reports.append(ref)
            return when

        original = self.doc(proof["original_report"])
        reader.page_member(original, proof.get("original_json_pointer", ""), pid, task)
        original_received = receipt(proof["original_report"], "reader")
        current_received = receipt(proof["report"], proof.get("collection_role", "reader"))
        self.need(original_received <= current_received, "fresh reader current received before original")
        basis_ref = proof["basis"]["report"]
        basis = reader.page_member(self.doc(basis_ref), proof["basis"]["json_pointer"], pid, task)
        # basis 可在較早階段收件；不用目前報告的階段捏造第二次收件。
        basis_roles = [
            role
            for role, seal in self.sealed[wave].items()
            if any(identity(entry) == identity(basis_ref) for entry in seal.get("collection_index", []))
        ]
        self.need(len(basis_roles) == 1, "fresh reader basis lacks unique actual collection")
        basis_received = receipt(basis_ref, basis_roles[0])
        self.need(basis_received <= current_received, "fresh reader current received before basis")
        sessions = proof.get("sessions", [])
        self.need(isinstance(sessions, list) and sessions, "fresh reader lacks full sequential trace")
        complete_basis = False
        for session in sessions:
            self.need(session.get("wave", wave) == wave, "fresh reader trace belongs to another wave")
            opening = self.doc(session["opening"])
            observed_pid, completed = reader.reader_session(self.root, session, task)
            started = self.api.at(opening["started_at"])
            self.need(observed_pid == pid and lo <= started, "fresh reader trace page/dispatch differs")
            plan = self.doc(opening["reading_plan"])
            freeze_ref = opening["source_freeze"]
            freeze = self.doc(freeze_ref)
            frozen = self.mapping(freeze["pages"], "page_id", "reader session frozen pages")[pid]
            self.need(
                plan.get("reviewer_task") == task
                and plan.get("author_task") in self.manifest["author_tasks"]
                and pid in plan.get("main_pages", [])
                and plan.get("source_freeze") == freeze_ref,
                "fresh reader plan owner/page/freeze differs",
            )
            self.need(
                self.api.at(plan["created_at"]) <= started and self.api.at(freeze["created_at"]) <= started,
                "fresh reader plan/freeze predates creation",
            )
            self.need(
                session["source"] == {"path": frozen["snapshot"], "sha256": frozen["source_sha256"]}
                and opening.get("figures_sha256") == frozen["figures_sha256"],
                "fresh reader trace freeze identity differs",
            )
            self.need(
                session.get("figures", {})
                == {name: freeze["figure_snapshots"][name] for name in frozen["figures_sha256"]},
                "fresh reader trace figure provenance differs",
            )
            if (
                opening["source_sha256"] == basis["source_sha256"]
                and opening["figures_sha256"] == basis.get("figures_sha256")
                and completed <= basis_received
            ):
                complete_basis = True
            for ref in session["notes"]:
                if reader.reported_issues(self.doc(ref)):
                    self.reports.append(ref)
        self.need(complete_basis, "fresh reader basis lacks completed current-source trace")
        if "checkpoint_closure" in proof:
            closure = proof["checkpoint_closure"]
            self.need(closure["wave"] == wave, "fresh reader closure belongs to another wave")
            self.need(
                receipt(closure["report"], closure["collection_role"]) == self.api.at(closure["received_at"]),
                "fresh reader closure receipt differs",
            )
        reader.reader_closure(self.root, proof, current, task, pid, phase, self.sealed)
        self.checked_readers.add((pid, wave))

    def closed_item(self, group, row, full, appendix, task, pid, appendix_at):
        ref = row["disposition_ref"]
        self.need(
            identity(ref) == identity(self.old_disposition_ref), "disposition is not from accepted prior manifest"
        )
        old = self.api.node(self.old_dispositions, ref["json_pointer"])
        self.need(
            old.get("action") == "fixed" and old.get("final_necessity") in {"necessary", "optional"},
            "prior issue was not fixed",
        )
        self.need(self.decisions.get(issue_key(old)) == old, "original closed disposition was changed or omitted")
        self.need(
            identity(row["original_report"]) == identity(old["report"]) and row["json_pointer"] == old["json_pointer"],
            "historical issue identity differs",
        )
        verification = row["historical_fix_verification"]
        self.need(verification in old.get("verification", []), "historical verification is not original")
        issue_report = self.doc(old["report"])
        self.api.node(issue_report, old["json_pointer"])
        old_report = self.doc(verification["report"])
        old_task = self.api.owner(issue_report)
        old_role = self.old_tasks[old_task]["role"]
        allowed_roles = {"technical", "reader"} if group["kind"] == "page" else {"technical"}
        self.need(
            self.api.owner(old_report) == old_task and old_role in allowed_roles,
            "original closure owner/role invalid",
        )
        if old_role == "reader":
            self.validate_reader(pid, task, group["collection"]["wave"])
        self.need(task != old_task and task not in self.old_tasks, "replacement is not a new reviewer")
        self.unavailable(group["unavailability"], old_task, appendix_at)
        self.need(
            row.get("verdict") == "pass" and row.get("status") == "still_fixed",
            "historical issue not currently still_fixed",
        )
        self.need(row.get("issue_id") == old.get("issue_id"), "historical issue id differs")
        self.text(row.get("reason"), "issue-specific reason")
        for field in ("owner", "reviewer_task"):
            if field in row:
                self.need(row[field] == task, "issue recheck owner differs")
        if "necessity" in row:
            self.need(row["necessity"] == old["final_necessity"], "historical necessity was downgraded")
        if group["kind"] == "page":
            self.need(
                verification.get("kind", "page") == "page"
                and old.get("affected_pages") == [pid]
                and not old.get("affected_files"),
                "page recheck scope differs",
            )
            previous = self.api.node(old_report, verification["json_pointer"])
            frozen = self.old_pages[pid]
            self.need(
                previous.get("page_id", previous.get("lesson_id")) == pid
                and previous.get("verdict") == "pass"
                and previous.get("source_sha256") == frozen["source_sha256"]
                and previous.get("figures_sha256", previous.get("figure_sha256")) == frozen["figures_sha256"],
                "original fix was not verified on prior frozen page",
            )
            self.api.artifact(self.root, {"path": frozen["snapshot"], "sha256": frozen["source_sha256"]})
            for name, expected in frozen["figures_sha256"].items():
                self.need(
                    self.api.sha(self.api.artifact(self.root, self.old_figures[name])) == expected,
                    "prior figure snapshot differs",
                )
            self.need(
                row.get("current_source_sha256") == self.live[pid]["source_sha256"]
                and row.get("current_figure_sha256") == self.live[pid]["figures_sha256"],
                "issue-specific current source/figure stale",
            )
            sources = {source["id"] for source in appendix["sources"]}
            artifacts = {artifact["id"] for artifact in appendix["artifacts"]}
            evidence, artifact_ids = row.get("current_evidence"), row.get("artifact_ids")
            self.need(
                isinstance(evidence, list)
                and evidence
                and isinstance(artifact_ids, list)
                and artifact_ids
                and set(artifact_ids) <= artifacts,
                "missing/unknown issue-specific evidence",
            )
            for item in evidence:
                self.need(item.get("source_id") in sources, "unknown current evidence source")
                self.text(item.get("locator"), "current evidence locator")
                self.text(item.get("supports"), "current evidence support")
            result = {"kind": "page", "page_id": pid}
        else:
            files = old.get("affected_files", [])
            self.need(
                verification.get("kind") == "tool_audit" and files and not old.get("affected_pages"),
                "tool recheck scope differs",
            )
            self.need(
                old_report.get("verdict") == "pass" and old["issue_id"] in old_report.get("resolved_issue_ids", []),
                "original tool issue was not closed",
            )
            old_checked = self.mapping(old_report.get("checked_files"), "path", "prior checked files")
            current_checked = self.mapping(full.get("checked_files"), "path", "current checked files")
            self.need(
                old["issue_id"] in full.get("resolved_issue_ids", []),
                "current tool audit omits resolved historical issue",
            )
            actual = {}
            for name in files:
                self.need(
                    old_checked[name]["sha256"] == self.prior_manifest["implementation"][name],
                    "original tool closure differs from prior implementation",
                )
                actual[name] = self.api.sha(self.api.local(self.root, name).read_bytes())
                self.need(current_checked[name]["sha256"] == actual[name], "current tool audit file stale")
            self.need(row.get("current_files_sha256") == actual, "issue-specific current files differ")
            evidence = row.get("current_evidence")
            self.need(isinstance(evidence, list) and evidence, "tool issue lacks actual test evidence")
            for ref in evidence:
                self.api.artifact(self.root, ref)
            result = {"kind": "tool_audit", "files": set(files)}
        entry_key = key(old, verification)
        self.need(entry_key not in self.entries, "duplicate historical replacement")
        self.entries[entry_key] = result

    def page(self, decision, verification, pid):
        entry_key = key(decision, verification)
        entry = self.entries.get(entry_key)
        if entry is None:
            return False
        self.need(entry["kind"] == "page" and entry["page_id"] == pid, "replacement page mismatch")
        self.used.add(entry_key)
        return True

    def files(self, decision, verification):
        entry_key = key(decision, verification)
        entry = self.entries.get(entry_key)
        if entry is None:
            return set()
        self.need(entry["kind"] == "tool_audit", "replacement tool kind mismatch")
        self.used.add(entry_key)
        return entry["files"]

    def finish(self):
        self.need(self.used == set(self.entries), "unused historical replacement")
