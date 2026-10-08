"""Validate preserved review evidence; never manufacture reader answers or verdicts.

The v1 contract combines an earlier same-criteria diagnostic audit with actual
bounded rereads and informed callbacks. Historical orchestration declarations
are explicitly coordinator attestations, not reconstructed API receipts.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import export_course, reading_time  # noqa: E402

POLICY = "tutorial_composite_v1"
EXPECTED_CANONICAL = 325
ROLES = ("reader", "technical", "continuity")
CRITERIA = (
    "SKILL.md",
    "references/review-protocol.md",
    "references/calibration.md",
    "references/project-context.md",
    "references/writing-prompt.md",
    "README.md",
    "UPSTREAM.json",
)
CORE_CRITERIA = CRITERIA[:4]
EXPOSED = {"5.4", "10.11", "19.12"}
FIELDS = ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256")
NOTE_FIELDS = ("understanding", "materials_and_labels", "expected_change", "confusion_and_quote", "missing_visuals")
RUFF_ARCHIVE_EXCLUDES = (
    "docs/course-repair-20261008/reviews",
    "docs/course-publish-20261008/audit-original",
    "docs/course-publish-20261008/additional-original",
    "docs/course-publish-20261008/freeze",
)
SCOPE_NOTE = (
    "Same-criteria diagnostic baseline plus actual bounded rechecks and informed callbacks; "
    "not a new 325-page blind review. Only archived bytes, source versions, recorded page/role evidence, "
    "chronology and dispositions are checked. Historical orchestration is coordinator-attested where disclosed. "
    "This does not prove comprehension, personal image viewing, physical isolation, human testing or execution."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip())


def timestamp(value):
    require(text(value), "missing actual timestamp")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp needs timezone")
    return result


def local(root, name):
    require(text(name) and not Path(name).is_absolute(), "expected repository-relative artifact path")
    result = (root / name).resolve()
    require(result.is_relative_to(root), "artifact escapes repository")
    return result


def pointer(data, value):
    require(isinstance(value, str) and (value == "" or value.startswith("/")), "invalid JSON pointer")
    if not value:
        return data
    for token in value[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(data, list):
            require(token.isdecimal(), "non-index JSON pointer into list")
            data = data[int(token)]
        else:
            data = data[token]
    return data


class Artifacts:
    """Resolve old absolute paths only through an immutable, explicit sidecar."""

    def __init__(self, root, index_reference):
        self.root = root
        self.bytes_cache = {}
        self.json_cache = {}
        self.aliases = {}
        self.external = {}
        self.external_paths = set()
        index = self.json(index_reference)
        require(index.get("schema_version") == 1, "invalid archive index schema")
        entries = index.get("artifacts")
        require(isinstance(entries, list) and entries, "archive index is empty")
        for entry in entries:
            original = entry.get("original_path")
            require(text(original) and original not in self.aliases, "duplicate or absent original archive path")
            target = local(root, entry.get("path"))
            self.aliases[original] = (target, entry.get("sha256"))
            if entry.get("kind") == "external_paper_receipt":
                self.external[original] = entry
                self.external_paths.add(target)
        for name, (_, digest) in self.aliases.items():
            if name in self.external:
                self.paper(name, self.external[name]["source_sha256"])
            else:
                self.raw({"path": name, "sha256": digest})

    def paper(self, original, source_digest):
        entry = self.external[original]
        require(source_digest == entry.get("source_sha256"), "external paper original hash changed")
        require(
            Path(original).suffix in {".pdf", ".txt"}
            and (
                any(
                    part in original
                    for part in (
                        "/original-papers/",
                        "/technical-modalities-work/originals/",
                        "/technical-architecture/yarn-",
                        "/evidence/technical-integration/graves-ctc-2006.pdf",
                        "/evidence/technical-modalities/original-sources/",
                    )
                )
                or re.search(
                    r"/reviewer-work/technical-learning/evidence/(?:flan|guo|ifeval|instructgpt|mtp|r1|v3)\.(?:pdf|txt)$",
                    original,
                )
            ),
            "paper receipt cannot replace curriculum, code, figures, notes or execution evidence",
        )
        target = self.aliases[original][0]
        raw = target.read_bytes()
        require(sha(raw) == entry["sha256"], "external paper receipt artifact tampered")
        receipt = json.loads(raw)
        require(
            receipt.get("kind") == "external_paper_receipt" and receipt.get("source_sha256") == source_digest,
            "external paper receipt does not bind original source hash",
        )
        parsed = urlparse(receipt.get("url", ""))
        arxiv = parsed.hostname in {"arxiv.org", "export.arxiv.org"} and re.fullmatch(
            r"/(?:abs|pdf)/\d{4}\.\d{4,5}v\d+(?:\.pdf)?", parsed.path
        )
        publisher = parsed.hostname in {
            "proceedings.neurips.cc",
            "proceedings.mlr.press",
            "aclanthology.org",
            "openaccess.thecvf.com",
            "dl.acm.org",
            "ieeexplore.ieee.org",
            "www.nature.com",
            "www.science.org",
        } and bool(parsed.path.strip("/"))
        author_paper = receipt.get("url") == "https://www.cs.toronto.edu/~graves/icml_2006.pdf"
        require(
            parsed.scheme == "https" and (arxiv or publisher or author_paper),
            "external paper URL not fixed published source",
        )
        require(
            text(receipt.get("version")) and receipt.get("locators") and receipt.get("limitations"),
            "paper receipt lacks source scope",
        )
        timestamp(receipt.get("verified_at"))
        # These bytes are deliberately the receipt, never the absent paper. No
        # caller may treat this leaf as a frozen source/trace/image artifact.
        return raw

    def resolve(self, name):
        require(text(name), "missing artifact path")
        if name in self.aliases:
            return self.aliases[name][0]
        return local(self.root, name)

    def raw(self, reference, *, allow_external=False):
        require(isinstance(reference, dict), "missing artifact reference")
        name, digest = reference.get("path"), reference.get("sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest), "invalid artifact SHA-256")
        if name in self.external:
            require(allow_external, "paper receipt cannot substitute for review evidence")
            return self.paper(name, digest)
        target = self.resolve(name)
        require(target not in self.external_paths, "paper receipt cannot substitute for structured/source evidence")
        if target not in self.bytes_cache:
            self.bytes_cache[target] = target.read_bytes()
        raw = self.bytes_cache[target]
        require(sha(raw) == digest, f"artifact missing or tampered: {name}")
        if name in self.aliases:
            require(self.aliases[name][1] == digest, f"reference disagrees with archive sidecar: {name}")
        return raw

    def json(self, reference):
        raw = self.raw(reference)
        key = sha(raw)
        if key not in self.json_cache:
            self.json_cache[key] = json.loads(raw)
        return self.json_cache[key]

    def node(self, reference):
        return pointer(self.json(reference), reference.get("json_pointer", ""))


def current_inventory(root):
    index = json.loads((root / "course/lesson-index.json").read_bytes())
    result = reading_time.build_inventory(
        root, index, export_course.DOCUMENTS, export_course.home_introduction(index, True)
    )
    lessons = {item["id"]: item for item in index}
    for page in result["pages"]:
        if page["page_id"] in lessons:
            page["notebook"] = lessons[page["page_id"]]["notebook"]
            page["notebook_sha256"] = sha(local(root, page["notebook"]).read_bytes())
    return result


def owner(report):
    result = report.get("reviewer", report.get("reviewer_task", report.get("reader")))
    return result.get("reviewer") if isinstance(result, dict) else result


def page_id(record):
    result = record.get("page_id", record.get("page"))
    return result or record.get("current_source", {}).get("page_id")


def source_hash(record):
    if record.get("source_sha256"):
        return record["source_sha256"]
    for key in ("current_source", "source"):
        if isinstance(record.get(key), dict) and record[key].get("source_sha256"):
            return record[key]["source_sha256"]
    return None


def check_candidate(store, candidate, pid, role, digest, authors, identities, figures=None, main_only=False):
    require(isinstance(candidate, dict) and candidate.get("role") == role, f"{pid}: missing {role} page evidence")
    reference = candidate.get("report")
    require(isinstance(reference, dict) and text(reference.get("json_pointer")), f"{pid}/{role}: page pointer required")
    report, record = store.json(reference), store.node(reference)
    require(isinstance(record, dict) and page_id(record) == pid, f"{pid}/{role}: pointer is not this actual page")
    task = owner(report)
    require(text(task) and re.fullmatch(r"/root/[A-Za-z0-9_/-]+", task), f"{pid}/{role}: invalid real task declaration")
    require(task == candidate.get("reviewer") and task not in authors, f"{pid}/{role}: author or invented reviewer")
    require(identities.get(task) == role, f"{pid}/{role}: unrecorded or cross-role reviewer")
    require(source_hash(record) == digest, f"{pid}/{role}: page record has stale source SHA")
    require(candidate.get("recorded_source_sha256") == digest, f"{pid}/{role}: catalog rebound source SHA")
    if figures is not None:
        for node in (record, record.get("source"), record.get("current_source")):
            if isinstance(node, dict) and "figures_sha256" in node:
                recorded_figures = node["figures_sha256"]
                require(
                    isinstance(recorded_figures, dict)
                    and (recorded_figures == figures or (main_only and recorded_figures.items() <= figures.items())),
                    f"{pid}/{role}: actual recorded figure versions stale",
                )
        recorded = record.get("figures")
        if (
            isinstance(recorded, dict)
            and recorded
            and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in recorded.values())
        ):
            require(recorded == figures, f"{pid}/{role}: actual figure hash map stale")
    # A single-page callback may place its answers beside source_and_figures.
    # The precise page/source node must still be genuine, and the full report is
    # byte-bound. Field shape is not a substitute for understanding.
    require(len(record) > 2, f"{pid}/{role}: empty page record")
    return task


def page_map(manifest):
    pages = manifest.get("inventory", {}).get("pages", manifest.get("pages", []))
    result = {p["page_id"]: p for p in pages}
    require(len(result) == len(pages), "duplicate frozen page")
    return result


def main_units(raw):
    def hide(match):
        label = re.search(r"<summary[^>]*>(.*?)</summary>", match[0], re.S)
        value = re.sub("<[^>]+>", "", label[1]).strip() if label else "選讀"
        return "\n[原位置選讀折疊區：" + value + "；正文路線完成後補讀]\n"

    body = re.sub(r"<details\b[^>]*>.*?</details>", hide, raw.decode("utf-8"), flags=re.S)
    spec = importlib.util.spec_from_file_location(
        "composite_unit_scheme", ROOT / "docs/review-tools/incremental_reader.py"
    )
    unit_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unit_reader)
    return unit_reader.units(body) or [""]


def check_traces(store, catalog, batches, identities, authors):
    traces = {}
    scope_pairs = {(stage, group) for stage in ("original", "freeze01") for group in batches[stage]["groups"]}
    entries = catalog.get("trace_validation", [])
    require({(e["stage"], e["group"]) for e in entries} == scope_pairs, "missing original or bounded raw trace")
    require(len(entries) == len(scope_pairs), "duplicate raw trace scope")
    for entry in entries:
        stage, group = entry["stage"], entry["group"]
        metadata = batches[stage]["groups"][group]
        task = metadata["reviewer"]
        require(task not in authors and identities.get(task) == "reader", "trace reader is author/unregistered")
        rows = [json.loads(line) for line in store.raw(entry["ref"]).splitlines()]
        times = [timestamp(row["at"]) for row in rows if "at" in row]
        require(times == sorted(times), "raw trace chronological order changed")
        checks = [row for row in rows if row.get("kind") == "checkpoint" and row.get("phase") == "main"]
        expected = []
        frozen_pages = page_map(batches[stage])
        for pid in metadata["pages"]:
            page = frozen_pages[pid]
            units = main_units(store.raw({"path": page["snapshot"], "sha256": page["source_sha256"]}))
            expected.extend((pid, i, sha(unit.encode()), page["source_sha256"]) for i, unit in enumerate(units))
        actual = [(r["page_id"], r["unit_index"], r["unit_sha256"], r["source_sha256"]) for r in checks]
        require(actual == expected, f"{stage}/{group}: actual raw checkpoint scope/order/version incomplete")
        ends = [r["page_id"] for r in rows if r.get("kind") == "page_end" and r.get("phase") == "main"]
        require(ends == metadata["pages"], f"{stage}/{group}: missing actual page-end records")
        for row in checks:
            note = row.get("note", {})
            require(note.get("reviewer") == task, "trace notes changed real reviewer identity")
            require(
                all(text(note.get(field)) for field in NOTE_FIELDS), "missing saved actual understanding checkpoint"
            )
        traces[entry["ref"]["path"]] = rows
    return traces


def check_sequential(store, page, digest, task, traces):
    evidence = page.get("sequential_evidence")
    require(isinstance(evidence, dict), f"{page['page_id']}: missing sequential evidence")
    store.raw(evidence["trace"])
    rows = traces.get(evidence["trace"]["path"])
    require(rows is not None, "page trace is not a complete original scope")
    lines = evidence.get("raw_checkpoint_lines")
    pid = page["page_id"]
    actual_lines = [
        i + 1
        for i, row in enumerate(rows)
        if row.get("kind") == "checkpoint" and row.get("phase") == "main" and row.get("page_id") == pid
    ]
    require(lines == actual_lines and lines, f"{pid}: skipped or invented checkpoint lines")
    require(all(rows[i - 1]["source_sha256"] == digest for i in lines), "page sequential source is stale")
    require(all(rows[i - 1]["note"]["reviewer"] == task for i in lines), "page sequential reader identity is invented")


def check_current_freeze(store, freeze, current, criteria):
    frozen = page_map(freeze)
    require(set(frozen) == set(current), "source freeze omits canonical page")
    for pid, page in frozen.items():
        live = current[pid]
        require(all(page.get(f) == live.get(f) for f in FIELDS[:4]), f"current source freeze stale: {pid}")
        store.raw({"path": page["snapshot"], "sha256": live["source_sha256"]})
        if live.get("notebook"):
            notebook = page.get("notebook", {})
            require(
                notebook.get("live_path") == live["notebook"] and notebook.get("sha256") == live["notebook_sha256"],
                f"current Notebook freeze stale: {pid}",
            )
            store.raw(notebook)
    for kind in ("source_files", "figures", "criteria", "notebooks"):
        require(isinstance(freeze.get(kind), dict), f"missing current {kind} freeze")
        for name, reference in freeze[kind].items():
            if kind != "criteria":
                require(reference.get("live_path") == name, f"current {kind} file identity rebound")
            store.raw(reference)
            live = local(store.root, reference["live_path"])
            require(sha(live.read_bytes()) == reference["sha256"], f"current {kind} drift: {reference['live_path']}")
    require(set(freeze["criteria"]) == set(CRITERIA), "current freeze lacks complete criteria")
    require(freeze["criteria"] == criteria, "current criteria freeze contract differs")
    require(
        set(freeze["notebooks"]) == {p["notebook"] for p in current.values() if p.get("notebook")},
        "current freeze omits Notebook",
    )
    require(
        {p["source"] for p in current.values()} == set(freeze["source_files"]),
        "current freeze omits canonical source file",
    )
    require(
        {f for p in current.values() for f in p["figures_sha256"]} <= set(freeze["figures"]),
        "current freeze omits figure",
    )


def check_history(store, attestation, stage_order, authors):
    require(attestation.get("kind") == "coordinator_attested_history", "history must disclose coordinator attestation")
    require(attestation.get("coordinator") in authors, "unregistered coordinator")
    timestamp(attestation.get("recorded_at"))
    require(attestation.get("limitations"), "history provenance limits missing")
    identities = {}
    for assignment in attestation.get("reviewer_assignments", []):
        task, role = assignment.get("reviewer_task"), assignment.get("role")
        require(role in ROLES and text(task) and task not in authors, "author or invalid history reviewer")
        require(task not in identities, "duplicate real reviewer assignment; do not rename one reviewer")
        require(assignment.get("evidence_class") == "coordinator_attested_history", "invented historical receipt class")
        require(text(assignment.get("basis")) and assignment.get("limits"), "history assignment has no basis/limits")
        identities[task] = role
    require(identities, "no real reviewer assignments")
    require(isinstance(stage_order, list) and len(stage_order) == 2, "missing original/bounded stage chronology")
    require({row.get("batch") for row in stage_order} == {"original", "freeze01"}, "wrong stage chronology batches")
    for row in stage_order:
        values = [timestamp(row.get(role + "_completed_at")) for role in (*ROLES, "cross")]
        require(values == sorted(values), "initial source-stage global order is invalid")
        require(row.get("independence_before_cross") is True, "initial judgments not sealed before peer cross")
        require(
            text(row.get("chronology_basis")) and row.get("limitations"), "chronology lacks honest provenance scope"
        )
        require(row.get("evidence"), "global stage order lacks actual archived gate/seal evidence")
        actual_times = set()
        for reference in row["evidence"]:
            evidence = store.json(reference)
            for key in ("at", "sealed_at"):
                if key in evidence:
                    actual_times.add(timestamp(evidence[key]))
        require(set(values) <= actual_times, "stage completion times not bound to actual archived gates/seals")
    return identities


def check_latest_continuity(store, value, callback_pages, authors, identities, callback_manifest):
    require(isinstance(value, dict), "missing actual post-technical continuity callback")
    report = store.json(value["report"])
    technical = store.json(value["technical_report"])
    technical_seal = store.json(value["technical_seal"])
    seal = store.json(value["seal"])
    reader_gate = store.json(value["reader_gate"])
    require(technical_seal.get("sha256") == value["technical_report"]["sha256"], "technical callback seal changed")
    require(seal.get("sha256") == value["report"]["sha256"], "post-technical callback seal changed")
    started = timestamp(pointer(report, value["started_at_pointer"]))
    completed = timestamp(pointer(report, value["completed_at_pointer"]))
    technical_completed = timestamp(technical_seal["at"])
    require(started >= technical_completed and completed >= started, "latest continuity began before technical sealed")
    require(timestamp(seal["at"]) >= completed, "post-technical continuity sealed before actual completion")
    require(timestamp(reader_gate["at"]) <= technical_completed, "current technical precedes complete reader gate")
    require(
        text(technical.get("reader_gate"))
        and store.resolve(technical["reader_gate"]) == store.resolve(value["reader_gate"]["path"]),
        "technical callback read a different reader gate",
    )
    require(
        technical.get("manifest_sha256") == callback_manifest["sha256"],
        "technical callback not bound to callback source manifest",
    )
    if "reader_gate_sha256" in technical:
        require(technical["reader_gate_sha256"] == value["reader_gate"]["sha256"], "technical reader gate hash stale")
    task = owner(report)
    require(task not in authors and identities.get(task) == "continuity", "invalid post-technical continuity owner")
    require(
        owner(technical) != task and identities.get(owner(technical)) == "technical", "latest roles not independent"
    )
    require(not report.get("new_necessary_findings"), "latest continuity has unresolved necessary findings")
    require(not report.get("unresolved_necessary_findings"), "latest continuity retains necessary findings")
    require(not technical.get("new_necessary_findings"), "latest technical has unresolved necessary findings")
    require(not technical.get("unresolved_necessary_findings"), "latest technical retains necessary findings")
    require(
        set(callback_pages) <= {page_id(p) for p in report.get("pages", [])}, "post-technical callback missing page"
    )
    return value["report"]


def check_dispositions(store, attestation, catalog, pages, extras):
    dispositions = attestation.get("finding_dispositions")
    require(isinstance(dispositions, list), "missing actual finding dispositions")
    by_id = {d["id"]: d for d in dispositions}
    require(len(by_id) == len(dispositions), "duplicate finding disposition")
    closure_ids = set()
    sources = catalog.get("finding_sources", {})
    require(
        set(sources) == {"adjudication", "original_diagnosis", "bounded_diagnosis"},
        "independent original and bounded finding sources missing",
    )
    adjudication = store.json(sources["adjudication"])
    original_diagnosis = store.json(sources["original_diagnosis"])
    bounded = store.node(sources["bounded_diagnosis"])
    require(isinstance(bounded, list), "bounded diagnosis pointer does not identify actual relation list")
    require(
        bounded == adjudication["new_bounded_review_dispositions"], "bounded diagnosis differs from actual adjudication"
    )
    original_rows = adjudication["original_relation_closures"]
    original_decisions = original_diagnosis["decisions"]
    require(
        {r["id"] for r in original_rows} == {r["id"] for r in original_decisions}
        and len(original_rows) == len(original_decisions),
        "original diagnosis relations omitted from adjudication",
    )
    original_grades = {r["id"]: r["root_grade"] for r in original_decisions}
    require(
        all(r["original_grade"] == original_grades[r["id"]] for r in original_rows), "original finding grade relabelled"
    )
    for closure in catalog.get("necessary_closures", []):
        iid = closure["relation_id"]
        require(iid not in closure_ids, "duplicate necessary relation")
        closure_ids.add(iid)
        record = store.node(closure["coordinator_record"])
        require(record.get("id") == iid, "closure pointer is not original actual relation")
        require(
            store.json(closure["coordinator_record"]) == adjudication,
            "closure belongs to different actual adjudication",
        )
        disposition = by_id.get(iid, {})
        require(
            disposition.get("necessary") is True and disposition.get("status") == "resolved",
            "necessary finding unresolved",
        )
        require(
            text(disposition.get("reason")) and disposition.get("current_evidence"),
            "necessary closure has no actual evidence",
        )
        required_pages = set(closure["pages"])
        require(required_pages, "necessary closure has no required current pages")
        require(
            required_pages == set(disposition.get("pages", [])),
            "necessary closure page scope differs from preserved coordinator disposition",
        )
        if record.get("page_id"):
            require(record["page_id"] in required_pages, "bounded necessary closure omits actual finding page")
        covered = set()
        for candidate in disposition["current_evidence"]:
            reference = candidate.get("report", candidate)
            node = store.node(reference)
            pid = page_id(node)
            require(pid in pages or pid in extras, "closure evidence lacks actual current page")
            role = candidate.get("role")
            selected = (pages[pid] if pid in pages else extras[pid]).get("selected_roles", {}).get(role)
            require(
                role in ROLES
                and selected
                and reference["sha256"] == selected["report"]["sha256"]
                and reference.get("json_pointer") == selected["report"].get("json_pointer")
                and store.resolve(reference["path"]) == store.resolve(selected["report"]["path"]),
                "closure uses author-owned or unselected role evidence",
            )
            digest = (
                pages[pid]["current_source"]["source_sha256"]
                if pid in pages
                else extras[pid]["source"]["source_sha256"]
            )
            require(source_hash(node) == digest, "closure uses stale page evidence")
            covered.add(pid)
        require(required_pages <= covered, f"necessary relation lacks full actual closure: {iid}")
    expected = set()
    for document in [adjudication]:
        expected.update(
            r["id"] for r in document.get("original_relation_closures", []) if r["original_grade"] == "necessary"
        )
        expected.update(
            r["id"] for r in document.get("new_bounded_review_dispositions", []) if r["root_grade"] == "necessary"
        )
        # Every reported optional relation also needs a disposition, not a new
        # automatically generated improvement or forced rewrite.
        all_ids = {r["id"] for r in document.get("original_relation_closures", [])}
        all_ids.update(r["id"] for r in document.get("new_bounded_review_dispositions", []))
        require(all_ids == set(by_id), "reported relation dispositions omitted or invented")
    require(expected == closure_ids and closure_ids, "necessary findings removed from closure catalog")
    for d in dispositions:
        require(d.get("status") in {"resolved", "retained", "deferred_optional"}, "finding remains pending")
        require(text(d.get("reason")), "disposition reason missing")
        require(not d.get("necessary") or d["status"] == "resolved", "necessary finding deferred as optional")


def check_configuration_delta(store, manifest, baseline_digest, current_raw):
    """Only the four byte-preservation archive exclusions may differ from B."""
    delta = manifest.get("configuration_delta", {})
    require(delta.get("path") == "pyproject.toml", "unrecorded configuration drift")
    reference = delta.get("baseline_snapshot", {})
    require(reference.get("sha256") == baseline_digest, "configuration baseline not original sealed implementation")
    baseline = tomllib.loads(store.raw(reference).decode())
    current = tomllib.loads(current_raw.decode())
    require(delta.get("current_sha256") == sha(current_raw), "configuration delta does not bind current bytes")
    require(
        delta.get("allowed_added_ruff_archive_paths") == list(RUFF_ARCHIVE_EXCLUDES),
        "configuration exception permits other paths",
    )
    old = baseline["tool"]["ruff"].get("extend-exclude", [])
    new = current["tool"]["ruff"].get("extend-exclude", [])
    require(
        isinstance(old, list)
        and isinstance(new, list)
        and not set(old) & set(RUFF_ARCHIVE_EXCLUDES)
        and all(new.count(name) == 1 for name in RUFF_ARCHIVE_EXCLUDES)
        and [name for name in new if name not in RUFF_ARCHIVE_EXCLUDES] == old,
        "configuration changed preserved Ruff exclusions",
    )
    current["tool"]["ruff"]["extend-exclude"] = old
    require(current == baseline, "configuration changed outside four archive exclusions")


def audit_manifest(root, manifest_path, stage="all"):
    """Read-only full integrity validation; stage labels never bypass evidence."""
    failures, records, limits = [], [], []
    try:
        root = Path(root).resolve()
        require(stage in {"reader", "technical", "all"}, "invalid review gate stage")
        manifest = json.loads(Path(manifest_path).read_bytes())
        require(
            manifest.get("schema_version") == 1 and manifest.get("review_policy") == POLICY,
            "unsupported composite policy",
        )
        store = Artifacts(root, manifest.get("archive_index"))
        catalog = store.json(manifest["catalog"])
        require(
            catalog.get("review_policy") == POLICY and not catalog.get("pending_catalog_evidence"), "catalog incomplete"
        )
        require(not catalog.get("mechanical_errors"), "catalog retains mechanical errors")
        current_list = current_inventory(root)["pages"]
        current = {page["page_id"]: page for page in current_list}
        pages = {page["page_id"]: page for page in catalog["pages"]}
        require(
            len(current_list) == len(current) == len(catalog["pages"]) == len(pages) == EXPECTED_CANONICAL
            and set(current) == set(pages),
            "missing/duplicate current canonical page",
        )
        authors = set(manifest.get("author_tasks", []))
        require("/root" in authors and all(text(a) for a in authors), "missing actual author exclusions")
        attestation = store.json(manifest["coordinator_attestation"])
        identities = check_history(store, attestation, manifest.get("stage_order"), authors)
        criteria = manifest.get("criteria", {})
        require(set(criteria) == set(CRITERIA), "current complete criteria missing")
        for name, reference in criteria.items():
            store.raw(reference)
            expected_path = ".agents/skills/clear-tutorial/" + name
            require(reference.get("live_path") == expected_path, "criteria bound to different live source")
            require(sha(local(root, expected_path).read_bytes()) == reference["sha256"], "current criteria drift")
        freeze = store.json(manifest["source_freeze"])
        check_current_freeze(store, freeze, current, criteria)
        batches = {name: store.json(ref) for name, ref in catalog["manifest_references"].items()}
        require(set(batches) == {"original", "freeze01", "callback02"}, "missing baseline/recheck/callback chain")
        for name, batch in batches.items():
            hashes = batch.get("criteria_sha256", batch.get("criteria_unchanged", {}))
            require(set(CORE_CRITERIA) <= set(hashes), "baseline criteria missing")
            require(
                all(k in criteria and h == criteria[k]["sha256"] for k, h in hashes.items()),
                "baseline same-criteria chain drift",
            )
            for page in page_map(batch).values():
                store.raw({"path": page["snapshot"], "sha256": page["source_sha256"]})
        implementation = batches["freeze01"].get("implementation_sha256")
        inputs = store.json(manifest.get("current_technical_inputs"))
        for label, hashes in (("implementation", implementation), ("technical input", inputs.get("files_sha256"))):
            require(isinstance(hashes, dict) and hashes, f"missing preserved {label} identity map")
            for name, digest in hashes.items():
                raw = local(root, name).read_bytes()
                if label == "implementation" and name == "pyproject.toml" and sha(raw) != digest:
                    check_configuration_delta(store, manifest, digest, raw)
                else:
                    require(sha(raw) == digest, f"current {label} drift: {name}")
        criteria_entries = catalog.get("criteria", [])
        expected_criteria = {
            (stage_name, name)
            for stage_name, batch in batches.items()
            for name in batch.get("criteria_sha256", batch.get("criteria_unchanged", {}))
        }
        require(
            {(e["stage"], e["name"]) for e in criteria_entries} == expected_criteria
            and len(criteria_entries) == len(expected_criteria),
            "missing actual frozen criteria artifact",
        )
        for entry in criteria_entries:
            expected_digest = criteria[entry["name"]]["sha256"]
            require(entry["expected_sha256"] == expected_digest, "frozen criteria SHA rebound")
            require(entry["frozen"]["sha256"] == expected_digest, "frozen criteria artifact SHA stale")
            store.raw(entry["frozen"])
            store.raw(entry["current"])
        require(set(page_map(batches["original"])) == set(current), "original baseline not complete canonical course")
        for name, metadata in catalog.get("artifact_registry", {}).items():
            store.raw({"path": name, "sha256": metadata["sha256"]}, allow_external=True)
        require(catalog.get("artifact_registry"), "immutable artifact registry missing")
        traces = check_traces(store, catalog, batches, identities, authors)
        frozen_pages = {batch: page_map(value) for batch, value in batches.items()}
        callback_ids = set(frozen_pages["callback02"])
        latest = check_latest_continuity(
            store,
            manifest.get("latest_continuity"),
            callback_ids,
            authors,
            identities,
            catalog["manifest_references"]["callback02"],
        )
        require(
            timestamp(attestation["recorded_at"])
            >= max(
                [timestamp(store.json(manifest["latest_continuity"]["seal"])["at"])]
                + [
                    timestamp(row[role + "_completed_at"])
                    for row in manifest["stage_order"]
                    for role in (*ROLES, "cross")
                ]
            ),
            "historical attestation predates preserved review stages",
        )
        for pid, page in pages.items():
            live = current[pid]
            require(
                all(page["current_source"].get(f) == live.get(f) for f in FIELDS),
                f"catalog current source/Notebook stale: {pid}",
            )
            expected_mode = "callback02"
            for mode in ("original", "freeze01"):
                if all(frozen_pages[mode][pid].get(f) == live.get(f) for f in FIELDS[:4]):
                    expected_mode = mode
                    break
            require(page.get("selected_mode") == expected_mode, f"{pid}: mislabelled evidence reuse")
            require(pid in frozen_pages[expected_mode], "new source has no real callback")
            tasks = []
            for role in ROLES:
                candidate = page.get("selected_roles", {}).get(role)
                tasks.append(
                    check_candidate(
                        store, candidate, pid, role, live["source_sha256"], authors, identities, live["figures_sha256"]
                    )
                )
                if expected_mode == "callback02" and role == "continuity":
                    require(
                        candidate["report"]["sha256"] == latest["sha256"],
                        "current continuity not actual post-technical report",
                    )
                if expected_mode == "callback02" and role == "technical":
                    require(
                        candidate["report"]["sha256"] == manifest["latest_continuity"]["technical_report"]["sha256"],
                        "current technical not sealed callback read by continuity",
                    )
            require(len(set(tasks)) == 3, f"{pid}: same reviewer reused across roles")
            source_mode = "original" if expected_mode == "original" else "freeze01"
            initial = page.get("reader_source_evidence")
            digest = frozen_pages[source_mode][pid]["source_sha256"]
            task = check_candidate(
                store,
                initial,
                pid,
                "reader",
                digest,
                authors,
                identities,
                frozen_pages[source_mode][pid]["figures_sha256"],
                main_only=True,
            )
            check_sequential(store, page, digest, task, traces)
            ending = page.get("page_ending_evidence")
            require(isinstance(ending, dict), f"{pid}: page-ending source evidence missing")
            ending_ref = ending.get("report", ending)
            require(
                ending_ref == initial["report"], f"{pid}: page-ending record differs from actual initial reader page"
            )
            store.node(ending_ref)
            if "raw_main_final_checkpoint_line" in ending:
                require(
                    ending["raw_main_final_checkpoint_line"] == page["sequential_evidence"]["raw_checkpoint_lines"][-1],
                    "page-end final checkpoint rebound",
                )
            if pid in EXPOSED and source_mode == "freeze01":
                replacement = page.get("bounded_role_candidates", {}).get("unprompted_replacement_reader_main")
                require(
                    replacement and initial["report"] == replacement["report"],
                    "exposed original reader used instead of real replacement",
                )
                require(
                    not page["sequential_evidence"].get("early_full_page_exposure"),
                    "replacement source reading was itself exposed",
                )
            original = page.get("original_role_candidates", {})
            for role in ROLES:
                check_candidate(
                    store,
                    original.get(role),
                    pid,
                    role,
                    frozen_pages["original"][pid]["source_sha256"],
                    authors,
                    identities,
                    frozen_pages["original"][pid]["figures_sha256"],
                )
            store.raw(page["original_cross"])
            if source_mode == "freeze01":
                bounded = page.get("bounded_role_candidates", {})
                for role in ("technical", "continuity", "cross"):
                    c = bounded.get(role)
                    if role == "cross":
                        require(
                            c and source_hash(store.node(c["report"])) == digest, "bounded cross page evidence missing"
                        )
                    else:
                        check_candidate(
                            store,
                            c,
                            pid,
                            role,
                            digest,
                            authors,
                            identities,
                            frozen_pages["freeze01"][pid]["figures_sha256"],
                        )
            records.append({"page_id": pid, "evidence_mode": expected_mode, "reviewer_tasks": tasks})
        extras = {}
        for extra in catalog.get("extras", []):
            if extra.get("source_matches_current_file"):
                extras[extra["page_id"]] = extra
                require(
                    sha(local(root, extra["source"]["source"]).read_bytes()) == extra["source"]["source_sha256"],
                    "current supplemental source stale",
                )
                for role in ROLES:
                    candidate = extra.get("selected_roles", {}).get(role)
                    if extra.get("stage") != "callback02" and candidate is None:
                        candidates = extra.get("role_candidates", {}).get(role, [])
                        candidate = candidates[0] if candidates else None
                    check_candidate(
                        store,
                        candidate,
                        extra["page_id"],
                        role,
                        extra["source"]["source_sha256"],
                        authors,
                        identities,
                        extra["source"]["figures_sha256"],
                    )
                    if extra.get("stage") == "callback02" and role in {"continuity", "technical"}:
                        sealed = latest if role == "continuity" else manifest["latest_continuity"]["technical_report"]
                        require(
                            candidate["report"]["sha256"] == sealed["sha256"],
                            "current extra not actual sealed callback",
                        )
        check_dispositions(store, attestation, catalog, pages, extras)
        require(catalog.get("limitations"), "review evidence limitations omitted")
        limits = catalog["limitations"] + attestation["limitations"]
    except (ValueError, KeyError, TypeError, OSError, IndexError, AttributeError) as error:
        failures.append(str(error))
    return {
        "schema_version": 1,
        "review_policy": POLICY,
        "stage": stage,
        "status": "failed" if failures else "passed_metadata_checks",
        "records": records,
        "failures": failures,
        "unverified": limits,
        "scope_note": SCOPE_NOTE,
    }


def check_active(root, active, stage="all"):
    root = Path(root).resolve()
    try:
        require(
            active.get("schema_version") == 1 and active.get("review_policy") == POLICY, "invalid composite pointer"
        )
        path = local(root, active.get("manifest"))
        require(sha(path.read_bytes()) == active.get("manifest_sha256"), "composite active manifest stale")
        return audit_manifest(root, path, stage)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"status": "failed", "stage": stage, "failures": [str(error)], "scope_note": SCOPE_NOTE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", choices=("reader", "technical", "all"), default="all")
    args = parser.parse_args()
    result = audit_manifest(ROOT, args.manifest, args.stage)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
