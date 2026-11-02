"""保留多代真實處置，並接續原連貫性角色已不可用的修正。

舊schema=1入口原樣保留；新schema=2子鏈對page原判補完整來源與時序核對。新增的continuity
接續要求：不可變舊處置、真不可用收據、受改頁的完整fresh reader和
technical、其後fresh continuity的原判/current、逐頁引文和真收件。
本工具不產生任何審閱判定，不證明AI確實閱讀或語意充分。
"""

import copy
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "historical_recheck_preserved_v1", Path(__file__).with_name("historical_fix_rechecks.py")
)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)


class Rechecks(_legacy.Rechecks):
    def __init__(self, root, manifest, live, tasks, sealed, api):
        sidecar = api.document(root, manifest["historical_fix_rechecks"])
        version = sidecar.get("schema_version")
        api.require(type(version) is int and version in {1, 2}, "historical recheck: wrong sidecar schema")
        if version == 1:
            super().__init__(root, manifest, live, tasks, sealed, api)
            return
        self.root, self.manifest, self.live = Path(root).resolve(), manifest, live
        self.tasks, self.sealed, self.api = tasks, sealed, api
        self.entries, self.used, self.reports = {}, set(), []
        chains = sidecar.get("chains")
        self.need(isinstance(chains, list) and chains, "empty historical chains")
        seen = set()
        for reference in chains:
            identity = _legacy.identity(reference)
            self.need(identity not in seen, "duplicate historical chain")
            seen.add(identity)
            chain = self.doc(reference)
            self.need(type(chain.get("schema_version")) is int and chain["schema_version"] == 1, "nested/non-v1 chain")
            child_manifest = copy.deepcopy(manifest)
            child_manifest["historical_fix_rechecks"] = reference
            child = Rechecks(root, child_manifest, live, tasks, sealed, api)
            self.need(
                not set(self.entries).intersection(child.entries), "duplicate historical replacement across chains"
            )
            self.entries.update(child.entries)
            self.reports.extend(child.reports)
        self.need(self.entries, "empty historical replacements")

    def group(self, group):
        if group.get("kind") == "continuity":
            self.continuity_group(group)
        else:
            super().group(group)
            if group.get("kind") == "page":
                self.page_original(group)

    def page_original(self, group):
        """v2 page不能以只有header的空殼代替真原判；原判可誠實保留revise。"""
        pid = group["page_id"]
        catalog = self.mapping(self.manifest["pages"], "page_id", "current pages")
        proof = catalog[pid]["technical"]
        original = self.doc(proof["original_report"])
        task = self.api.owner(original)
        assignment = self.assignments[task]
        original_freeze = proof.get("original_source_freeze")
        self.need(isinstance(original_freeze, dict), "missing page original freeze")
        self.need(not proof.get("original_json_pointer"), "page original must be a whole report")
        self.need(isinstance(original.get("created_at"), str), "page original lacks actual creation time")
        setup = self.api.module("history_v2_page_original", self.root / "scripts/omni_revision/review_setup.py")
        technical = self.api.module("history_v2_page_factual", self.root / "scripts/check_technical_reviews.py")
        setup.check_original(
            original,
            original_freeze,
            pid,
            pid,
            task,
            technical,
            self.api,
            original_source_snapshots=proof.get("original_source_snapshots", []),
        )
        wave, window = assignment["wave"], assignment["dispatch_window"]
        _, original_created, original_at = self.phase_receipt(proof["original_report"], task, wave, "technical", window)
        _, current_created, current_at = self.phase_receipt(proof["report"], task, wave, "technical", window)
        appendix = self.doc(group["appendix"])
        appendix_created = self.api.at(appendix.get("created_at", appendix.get("at")))
        appendix_at = self.api.at(group["collection"]["received_at"])
        self.need(
            original_created <= current_created <= appendix_created and original_at <= current_at <= appendix_at,
            "page original/current/appendix chronology reversed",
        )

    def phase_receipt(self, reference, task, wave, role, window):
        """核真報告、唯一收件、角色、派工窗及已完成的實際stage。"""
        report = self.doc(reference)
        self.need(self.api.owner(report) == task, "receipt owner differs")
        phase = self.sealed[wave][role]
        self.need(phase.get("role") == role and phase.get("status") == "completed", "collection phase incomplete")
        lo, hi = self.api.at(window["not_before"]), self.api.at(window["not_after"])
        start, end = self.api.at(phase["started_at"]), self.api.at(phase["completed_at"])
        self.need(start <= lo <= hi <= end <= datetime.now(UTC), "dispatch outside actual phase")
        matches = [r for r in phase.get("collection_index", []) if _legacy.identity(r) == _legacy.identity(reference)]
        self.need(
            len(matches) == 1 and reference["sha256"] in phase["report_sha256s"], "report lacks unique sealed receipt"
        )
        self.need(isinstance(report.get("created_at"), str), "report lacks actual creation time")
        created, received = self.api.at(report["created_at"]), self.api.at(matches[0]["received_at"])
        self.need(lo <= created <= received <= end, "report creation/receipt chronology invalid")
        self.reports.append(reference)
        return report, created, received

    def current_page(self, report, pointer, pid, task):
        current = self.api.node(report, pointer)
        live = self.live[pid]
        self.need(self.api.owner(report) == task, "current page owner differs")
        self.need(current.get("page_id", current.get("lesson_id")) == pid, "current page membership differs")
        self.need(current.get("verdict") in {"pass", "已足夠"}, "current page still requests revision")
        self.need(current.get("source_sha256") == live["source_sha256"], "current page source stale")
        self.need(
            current.get("figures_sha256", current.get("figure_sha256")) == live["figures_sha256"],
            "current page figures stale",
        )
        return current

    def changed_page_prior_stages(self, pid, continuity_task, wave):
        """受改頁要有本wave完整fresh reader及完整fresh technical，不能只驗索引。"""
        live, prior = self.live[pid], self.old_pages[pid]
        if live["source_sha256"] == prior["source_sha256"] and live["figures_sha256"] == prior["figures_sha256"]:
            return
        catalog = self.mapping(self.manifest["pages"], "page_id", "current pages")
        proof = catalog[pid]["technical"]
        self.need(
            proof.get("mode") == "review" and proof.get("wave") == wave, "changed page lacks same-wave whole technical"
        )
        self.need(
            proof.get("collection_role", "technical") == "technical", "technical review did not precede continuity"
        )
        full, original = self.doc(proof["report"]), self.doc(proof["original_report"])
        task = self.api.owner(full)
        assignment = self.assignments[task]
        self.need(
            task not in self.old_tasks
            and task not in self.manifest["author_tasks"]
            and task != continuity_task
            and self.tasks.get(task) == "technical"
            and assignment.get("role") == "technical"
            and assignment.get("context") == "fresh"
            and assignment.get("lesson_id") == pid
            and assignment.get("wave") == wave
            and self.api.owner(original) == task,
            "changed-page technical is not independent fresh whole-page review",
        )
        self.need(
            not proof.get("json_pointer") and not original.get("historical_issue_rechecks"),
            "technical appendix cannot replace whole review",
        )
        self.current_page(full, "", pid, task)
        original_freeze_ref = proof["original_source_freeze"]
        setup = self.api.module("history_v2_original_validator", self.root / "scripts/omni_revision/review_setup.py")
        technical = self.api.module("history_v2_factual", self.root / "scripts/check_technical_reviews.py")
        setup.check_original(
            original,
            original_freeze_ref,
            pid,
            pid,
            task,
            technical,
            self.api,
            original_source_snapshots=proof.get("original_source_snapshots", []),
        )
        _, original_created, original_at = self.phase_receipt(
            proof["original_report"], task, wave, "technical", assignment["dispatch_window"]
        )
        _, full_created, full_at = self.phase_receipt(
            proof["report"], task, wave, "technical", assignment["dispatch_window"]
        )
        self.need(original_created <= full_created and original_at <= full_at, "technical current precedes original")
        frozen = self.frozen[pid]
        body = self.api.artifact(self.root, {"path": frozen["snapshot"], "sha256": live["source_sha256"]}).decode()
        readers = {t for t, role in self.tasks.items() if role == "reader"}
        errors = technical._validate(
            self.root, self.api.local(self.root, live["source"]), pid, body, full, {task: [pid]}, readers
        )
        self.need(not errors, "whole technical validation failed: " + repr(errors))
        self.validate_reader(pid, task, wave)

    def continuity_group(self, group):
        full_ref, original_ref, appendix_ref = group["full_review"], group["original_review"], group["appendix"]
        self.need(
            _legacy.identity(appendix_ref) not in {_legacy.identity(full_ref), _legacy.identity(original_ref)},
            "appendix replaces full/original continuity",
        )
        full, original, appendix = self.doc(full_ref), self.doc(original_ref), self.doc(appendix_ref)
        task = self.api.owner(full)
        assignment = self.assignments[task]
        self.need(
            task not in self.old_tasks
            and task not in self.manifest["author_tasks"]
            and self.tasks.get(task) == "continuity"
            and assignment.get("role") == "continuity"
            and assignment.get("context") == "fresh"
            and self.api.owner(original) == task
            and self.api.owner(appendix) == task,
            "replacement continuity is not a new independent fresh reviewer",
        )
        collection = group["collection"]
        wave = assignment["wave"]
        self.need(
            collection.get("wave") == wave and collection.get("collection_role") == "continuity",
            "continuity collected in wrong phase",
        )
        phases = self.sealed[wave]
        preceding_end = None
        for role in ("reader", "technical", "continuity"):
            phase = phases[role]
            start, end = self.api.at(phase["started_at"]), self.api.at(phase["completed_at"])
            self.need(
                phase.get("role") == role and phase.get("status") == "completed" and start <= end,
                "required stage incomplete/reversed",
            )
            self.need(preceding_end is None or preceding_end <= start, "three review stages overlapped")
            preceding_end = end
        self.need(
            full.get("verdict") == "pass" and appendix.get("verdict") == "pass",
            "current continuity/appendix does not pass",
        )
        self.need(
            not full.get("historical_issue_rechecks") and not original.get("historical_issue_rechecks"),
            "appendix cannot replace full continuity",
        )
        window = assignment["dispatch_window"]
        self.need(
            type(original.get("schema_version")) is int
            and original["schema_version"] == 1
            and original.get("review_stage") == "continuity"
            and original.get("reviewer_context") == "fresh"
            and original.get("verdict") in {"pass", "revise"},
            "missing original continuity schema/context",
        )
        original_freeze_ref = group["original_source_freeze"]
        self.need(original.get("source_freeze") == original_freeze_ref, "original continuity freeze differs")
        original_freeze = self.doc(original_freeze_ref)
        original_pages = self.mapping(original_freeze["pages"], "page_id", "original continuity freeze pages")
        _, original_created, original_at = self.phase_receipt(original_ref, task, wave, "continuity", window)
        _, full_created, full_at = self.phase_receipt(full_ref, task, wave, "continuity", window)
        _, appendix_created, appendix_at = self.phase_receipt(appendix_ref, task, wave, "continuity", window)
        self.need(
            original_created <= full_created <= appendix_created and original_at <= full_at <= appendix_at,
            "continuity original/current/appendix reversed",
        )
        self.need(self.api.at(collection["received_at"]) == appendix_at, "appendix collection time differs")
        rows = appendix.get("historical_issue_rechecks")
        self.need(isinstance(rows, list) and rows, "empty continuity issue rechecks")
        catalog = self.mapping(self.manifest["pages"], "page_id", "current pages")
        checked = set()
        for row in rows:
            ref = row["disposition_ref"]
            self.need(
                _legacy.identity(ref) == _legacy.identity(self.old_disposition_ref),
                "disposition not from authenticated prior manifest",
            )
            old = self.api.node(self.old_dispositions, ref["json_pointer"])
            self.need(self.decisions.get(_legacy.issue_key(old)) == old, "original closed disposition changed/omitted")
            self.need(
                old.get("action") == "fixed" and old.get("final_necessity") in {"necessary", "optional"},
                "prior issue not closed",
            )
            self.need(
                _legacy.identity(row["original_report"]) == _legacy.identity(old["report"])
                and row["json_pointer"] == old["json_pointer"],
                "original issue identity differs",
            )
            issue_report = self.doc(old["report"])
            self.api.node(issue_report, old["json_pointer"])
            old_task = self.api.owner(issue_report)
            self.need(self.old_tasks[old_task]["role"] == "continuity", "historical issue is not a continuity issue")
            verification = row["historical_fix_verification"]
            self.need(verification in old.get("verification", []), "historical verification was replaced")
            old_report = self.doc(verification["report"])
            self.need(self.api.owner(old_report) == old_task, "original closure owner differs")
            previous = self.api.node(old_report, verification["json_pointer"])
            pid = previous.get("page_id", previous.get("lesson_id"))
            affected = old.get("affected_pages")
            self.need(
                isinstance(affected, list)
                and affected
                and len(set(affected)) == len(affected)
                and pid in affected
                and not old.get("affected_files"),
                "continuity affected-page scope differs",
            )
            prior = self.old_pages[pid]
            self.need(
                previous.get("verdict") == "pass"
                and previous.get("source_sha256") == prior["source_sha256"]
                and previous.get("figures_sha256", previous.get("figure_sha256")) == prior["figures_sha256"],
                "original fix not verified on authenticated prior page",
            )
            self.api.artifact(self.root, {"path": prior["snapshot"], "sha256": prior["source_sha256"]})
            for name, expected in prior["figures_sha256"].items():
                self.need(
                    self.api.sha(self.api.artifact(self.root, self.old_figures[name])) == expected,
                    "prior figure snapshot differs",
                )
            self.unavailable(group["unavailability"], old_task, appendix_created)
            self.need(
                row.get("status") == "still_fixed"
                and row.get("verdict") == "pass"
                and row.get("issue_id") == old.get("issue_id"),
                "historical continuity not still_fixed",
            )
            self.text(row.get("reason"), "issue-specific reason")
            self.need(row.get("necessity") == old["final_necessity"], "historical continuity necessity changed")
            page_evidence = row.get("pages")
            self.need(
                isinstance(page_evidence, dict) and set(page_evidence) == set(affected),
                "missing affected-page evidence",
            )
            for affected_pid in affected:
                proof = catalog[affected_pid]["continuity"]
                self.need(
                    proof.get("mode") == "review"
                    and proof.get("wave") == wave
                    and proof.get("collection_role", "continuity") == "continuity",
                    "missing actual full current continuity proof",
                )
                self.need(
                    _legacy.identity(proof["report"]) == _legacy.identity(full_ref)
                    and _legacy.identity(proof["original_report"]) == _legacy.identity(original_ref),
                    "full/original continuity proof differs",
                )
                self.need(affected_pid in assignment.get("pages", []), "affected page outside actual assignment")
                self.current_page(full, proof.get("json_pointer", ""), affected_pid, task)
                original_page = self.api.node(original, proof.get("original_json_pointer", ""))
                original_frozen = original_pages[affected_pid]
                self.need(
                    original_page.get("page_id", original_page.get("lesson_id")) == affected_pid
                    and original_page.get("source_sha256") == original_frozen["source_sha256"]
                    and original_page.get("figures_sha256", original_page.get("figure_sha256"))
                    == original_frozen["figures_sha256"]
                    and isinstance(original_page.get("checks"), (dict, list))
                    and original_page["checks"]
                    and isinstance(original_page.get("issues"), list),
                    "original continuity page missing",
                )
                original_body = self.api.artifact(
                    self.root, {"path": original_frozen["snapshot"], "sha256": original_frozen["source_sha256"]}
                ).decode()
                original_quote = original_page.get("source_quote")
                self.text(original_quote, "original continuity quote")
                self.need(original_quote in original_body, "original quote absent from original frozen page")
                for name, expected in original_frozen["figures_sha256"].items():
                    self.need(
                        self.api.sha(self.api.artifact(self.root, original_freeze["figure_snapshots"][name]))
                        == expected,
                        "original continuity figure snapshot differs",
                    )
                evidence = page_evidence[affected_pid]
                live, frozen = self.live[affected_pid], self.frozen[affected_pid]
                self.need(
                    all(
                        live.get(field) == frozen.get(field)
                        for field in ("source", "selector", "source_sha256", "figures_sha256")
                    ),
                    "current page freeze stale",
                )
                self.need(
                    evidence.get("source_sha256") == live["source_sha256"]
                    and evidence.get("figures_sha256") == live["figures_sha256"],
                    "issue-specific page identity stale",
                )
                quote = evidence.get("source_quote")
                self.text(quote, "current source quote")
                body = self.api.artifact(
                    self.root, {"path": frozen["snapshot"], "sha256": live["source_sha256"]}
                ).decode()
                self.need(quote in body, "current evidence quote absent from frozen page")
                self.text(evidence.get("reason"), "current page closure reason")
                if affected_pid not in checked:
                    self.changed_page_prior_stages(affected_pid, task, wave)
                    checked.add(affected_pid)
            receipts = row.get("current_evidence")
            self.need(isinstance(receipts, list) and receipts, "continuity issue lacks evidence artifacts")
            for reference in receipts:
                self.api.artifact(self.root, reference)
            entry_key = _legacy.key(old, verification)
            self.need(entry_key not in self.entries, "duplicate historical replacement")
            self.entries[entry_key] = {"kind": "page", "page_id": pid}
