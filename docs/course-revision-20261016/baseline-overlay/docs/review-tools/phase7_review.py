"""Grouped canonical-page reading and version checks; never generate review answers or verdicts."""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_technical_reviews as technical  # noqa: E402
from scripts import export_course, reading_time  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "grouped_incremental_reader", ROOT / "docs/review-tools/incremental_reader.py"
)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
SPEC = importlib.util.spec_from_file_location(
    "grouped_trace_audit", ROOT / "docs/review-tools/audit_readability_round.py"
)
trace_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_audit)

ACTIVE = Path("docs/course-reviews-active.json")
POLICY = "phase7_grouped"
STAGES = ("reader", "technical", "continuity")
AUTHORS = ("/root", "/root/p7_distillation_sources")
EXCLUDED = ("/root/p7_review_infrastructure_inventory", "/root/p7_mtp_sources_and_capstone")
CRITERIA = (
    ".agents/skills/clear-tutorial/SKILL.md",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
)
SCOPE_NOTE = (
    "Only recorded dispatch declarations, source/figure versions, sequential raw notes and evidence contracts are checked. "
    "This does not prove actual independent orchestration, that an image was viewed, comprehension or scientific truth."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timestamp needs a timezone")
    return parsed


def path_in(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "expected a repository-relative path")
    target = (root / name).resolve()
    require(target.is_relative_to(root.resolve()), "path is outside the repository")
    return target


def relative(root, path):
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_json(path):
    return json.loads(path.read_bytes())


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def artifact(root, reference):
    require(isinstance(reference, dict), "artifact reference needs path and sha256")
    path = path_in(root, reference.get("path"))
    require(path.is_file() and sha(path.read_bytes()) == reference.get("sha256"), f"artifact missing or stale: {path}")
    return path


def current_inventory(root):
    index = read_json(root / "course/lesson-index.json")
    inventory = reading_time.build_inventory(
        root, index, export_course.DOCUMENTS, export_course.home_introduction(index, True)
    )
    for page in inventory["pages"]:
        item = next((item for item in index if item["id"] == page["page_id"]), None)
        if item:
            page["notebook"] = item["notebook"]
            page["notebook_sha256"] = sha(path_in(root, item["notebook"]).read_bytes())
    return inventory


def criteria_hashes(root):
    return {name: sha(path_in(root, name).read_bytes()) for name in CRITERIA}


def page_text(root, page, index):
    if page["kind"] == "home":
        return export_course.home_introduction(index, True)
    text = path_in(root, page["source"]).read_text(encoding="utf-8")
    if page["selector"].startswith("section:"):
        return reading_time.lesson_slices(text)[page["page_id"]]
    return reading_time.chapter_introduction(text) if page["selector"] == "before-first-section" else text


def freeze(
    root, destination, group_plan, *, expected_pages=325, batch_id="phase7", authors=(), excluded=(), previous=None
):
    """Freeze once, after authoring; no dispatch, pointer, estimates or review reports are created."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(
        destination.is_relative_to(root) and not destination.exists(), "freeze destination must be new and inside repo"
    )
    inventory = current_inventory(root)
    require(len(inventory["pages"]) == expected_pages, "canonical inventory does not match expected page count")
    pages = {page["page_id"]: page for page in inventory["pages"]}
    require(len(pages) == expected_pages, "duplicate canonical page")
    plan = read_json(group_plan)
    groups = []
    for proposed in plan["groups"]:
        primary = list(proposed["primary_page_ids"])
        for pending in proposed.get("pending_new_page_ids", []):
            require(pending in pages, f"planned addition is absent: {pending}")
            if pending not in primary:
                predecessor = pending.rsplit(".", 1)[0] + "." + str(int(pending.rsplit(".", 1)[1]) - 1)
                require(predecessor in primary, f"addition needs its preceding lesson: {pending}")
                primary.insert(primary.index(predecessor) + 1, pending)
        groups.append(
            {
                "group": proposed["group"],
                "primary_page_ids": primary,
                "context_page_ids": proposed.get("context_page_ids", proposed.get("boundary_prior_candidates", [])),
            }
        )
    assigned = [pid for group in groups for pid in group["primary_page_ids"]]
    require(
        len(assigned) == len(set(assigned)) and set(assigned) == set(pages),
        "groups must cover every canonical page once",
    )
    require(len({group["group"] for group in groups}) == len(groups), "duplicate group")
    require(all(pid in pages for group in groups for pid in group["context_page_ids"]), "unknown prerequisite page")
    index = read_json(root / "course/lesson-index.json")
    texts = {pid: page_text(root, page, index) for pid, page in pages.items()}
    require(
        all(reading_time.source_sha256(texts[pid]) == page["source_sha256"] for pid, page in pages.items()),
        "source changed while freezing",
    )
    prior_tasks, same_batch_tasks = set(), set()
    for folder in ("docs/reader-reviews", "docs/technical-reviews"):
        for report in (root / folder).glob("*.json"):
            data = read_json(report)
            if isinstance(data, dict) and technical._text(data.get("reviewer_task")):
                if data.get("review_policy") == POLICY and data.get("batch_id") == batch_id:
                    same_batch_tasks.add(data["reviewer_task"])
                else:
                    prior_tasks.add(data["reviewer_task"])
    manifest = {
        "schema_version": 1,
        "review_policy": POLICY,
        "batch_id": batch_id,
        "created_at": now(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "home_variant": "executed_outputs=True",
        "author_tasks": sorted(set(AUTHORS) | set(authors)),
        "excluded_reviewers": sorted(set(EXCLUDED) | set(excluded) | prior_tasks),
        "explicitly_excluded_reviewers": sorted(set(excluded)),
        "unit_scheme_sha256": sha((ROOT / "docs/review-tools/incremental_reader.py").read_bytes()),
        "criteria_sha256": criteria_hashes(root),
        "pages": list(pages.values()),
        "groups": groups,
        "scope_note": SCOPE_NOTE,
    }
    if previous:
        old = load_manifest(root, previous)
        require(
            old["batch_id"] == batch_id and old["groups"] == groups,
            "refresh must preserve batch, primary scopes and original owners",
        )
        require({p["page_id"] for p in old["pages"]} == set(pages), "refresh cannot silently change review scope")
        manifest["previous_manifest"] = {"path": relative(root, previous), "sha256": sha(previous.read_bytes())}
        manifest["author_tasks"] = sorted(set(manifest["author_tasks"]) | set(old["author_tasks"]))
        original_path, original = min(
            versions(root, previous).values(), key=lambda item: timestamp(item[1]["created_at"])
        )
        explicit = set(old.get("explicitly_excluded_reviewers", ())) | set(excluded)
        protected = set(original["excluded_reviewers"]) | set(EXCLUDED) | explicit | prior_tasks
        # A completed report from this batch does not turn its existing owner
        # into a legacy reviewer. Preserve original/explicit exclusions and
        # actual other-batch reviewers; correct only the old self-report scan.
        wrongly_inherited = set(old["excluded_reviewers"]) & same_batch_tasks - protected
        manifest["excluded_reviewers"] = sorted(
            set(manifest["excluded_reviewers"]) | (set(old["excluded_reviewers"]) - wrongly_inherited)
        )
        manifest["explicitly_excluded_reviewers"] = sorted(explicit)
        if wrongly_inherited:
            manifest["exclusion_correction"] = {
                "reason": "Own same-batch reports were previously misclassified as legacy reports on refresh.",
                "reviewer_tasks": sorted(wrongly_inherited),
                "original_manifest_sha256": sha(original_path.read_bytes()),
                "original_and_explicit_exclusions_preserved": True,
            }
    figure_snapshots = {}
    for figure, digest in {
        name: digest for page in pages.values() for name, digest in page["figures_sha256"].items()
    }.items():
        snapshot = destination / "figures" / (digest + Path(figure).suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes(path_in(root, figure).read_bytes())
        figure_snapshots[figure] = {"path": relative(root, snapshot), "sha256": digest}
    manifest["figure_snapshots"] = figure_snapshots
    manifest["criteria_snapshots"] = {}
    for name, digest in manifest["criteria_sha256"].items():
        snapshot = destination / "criteria" / Path(name).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path_in(root, name).read_bytes())
        manifest["criteria_snapshots"][name] = {"path": relative(root, snapshot), "sha256": digest}
    for pid, page in pages.items():
        snapshot = destination / "sources" / (pid + ".md")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(texts[pid].encode("utf-8"))
        page["snapshot"] = relative(root, snapshot)
    write_new(destination / "manifest.json", manifest)
    if previous:
        for name in ("dispatch.json", "collections.jsonl"):
            if (previous.parent / name).is_file():
                shutil.copyfile(previous.parent / name, destination / name)
    return manifest


def load_manifest(root, path):
    manifest = read_json(path)
    require(
        manifest.get("schema_version") == 1 and manifest.get("review_policy") == POLICY, "unsupported grouped manifest"
    )
    require(manifest.get("home_variant") == "executed_outputs=True", "wrong homepage mode")
    require(
        manifest.get("unit_scheme_sha256") == sha((ROOT / "docs/review-tools/incremental_reader.py").read_bytes()),
        "unit scheme changed",
    )
    require(
        manifest.get("criteria_sha256") == criteria_hashes(root),
        "skill/protocol criteria changed; previous results cannot be reused",
    )
    for name, digest in manifest["criteria_sha256"].items():
        reference = manifest.get("criteria_snapshots", {}).get(name)
        require(reference and reference.get("sha256") == digest, "missing frozen review criteria")
        artifact(root, reference)
    require(
        isinstance(manifest.get("author_tasks"), list) and set(AUTHORS) <= set(manifest["author_tasks"]),
        "missing author exclusions",
    )
    require(set(EXCLUDED) <= set(manifest.get("excluded_reviewers", [])), "research/tool authors must be excluded")
    ids = [page["page_id"] for page in manifest["pages"]]
    assigned = [pid for group in manifest["groups"] for pid in group["primary_page_ids"]]
    require(
        len(ids) == len(set(ids)) and len(assigned) == len(set(assigned)) and set(ids) == set(assigned),
        "incomplete or duplicate grouped scope",
    )
    require(
        len({group["group"] for group in manifest["groups"]}) == len(manifest["groups"]), "duplicate group owner slot"
    )
    for page in manifest["pages"]:
        require(
            sha(path_in(root, page["snapshot"]).read_bytes()) == page["source_sha256"], "frozen source bytes changed"
        )
        for figure, digest in page["figures_sha256"].items():
            reference = manifest.get("figure_snapshots", {}).get(figure)
            require(reference and reference["sha256"] == digest, "missing shared frozen figure")
            artifact(root, reference)
    return manifest


def versions(root, manifest_path):
    result = {}
    while manifest_path:
        digest = sha(manifest_path.read_bytes())
        require(digest not in result, "cyclic freeze history")
        manifest = load_manifest(root, manifest_path)
        result[digest] = (manifest_path, manifest)
        reference = manifest.get("previous_manifest")
        manifest_path = artifact(root, reference) if reference else None
    return result


def dispatch(root, manifest_path, manifest, stage, group, task, at=None):
    require(stage in STAGES and technical._task(task), "invalid stage or real reviewer task")
    require(
        task not in set(manifest["author_tasks"]) | set(manifest["excluded_reviewers"]),
        "author/researcher/legacy reviewer cannot review",
    )
    data = read_json(manifest_path.parent / "dispatch.json")
    require(data.get("schema_version") == 1 and data.get("batch_id") == manifest["batch_id"], "wrong dispatch batch")
    matches = [
        item
        for item in data["dispatches"]
        if (item.get("stage"), item.get("group"), item.get("reviewer_task")) == (stage, group, task)
    ]
    require(
        matches and not any(str(item.get("status", "")).startswith("void") for item in matches),
        "missing or void actual dispatch",
    )
    require(
        any(
            item.get("fork_turns") == "none"
            and item.get("spawn_receipt", {}).get("tool") == "collaboration.spawn_agent"
            and item["spawn_receipt"].get("task_name") == task
            and technical._text(item["spawn_receipt"].get("agent_id"))
            for item in matches
        ),
        "missing actual fork-none spawn receipt",
    )
    selected_matches = [item for item in matches if item["dispatched_at"] == at] if at else matches
    require(selected_matches, "trace dispatch declaration disappeared")
    selected = min(selected_matches, key=lambda item: timestamp(item["dispatched_at"]))
    others = [
        item
        for item in data["dispatches"]
        if item.get("reviewer_task") == task and (item.get("stage"), item.get("group")) != (stage, group)
    ]
    require(not others, "reviewer reused across groups or stages")
    return selected


def readiness(root, manifest_path, reference, stage):
    receipt = read_json(artifact(root, reference))
    require(
        receipt.get("status") == "passed_metadata_checks" and receipt.get("stage") == stage,
        "previous stage needs actual complete gate receipt",
    )
    history = versions(root, manifest_path)
    require(receipt.get("manifest_sha256") in history, "readiness refers to another batch/version")
    frozen_path, frozen = history[receipt["manifest_sha256"]]
    records = [record for record in receipt.get("records", []) if record["stage"] == stage]
    require(
        {record["group"] for record in records} == {group["group"] for group in frozen["groups"]},
        "initial readiness omits a group",
    )
    for record in receipt["records"]:
        artifact(root, {"path": record["report_path"], "sha256": record["report_sha256"]})
    timestamp(receipt["sealed_at"])
    return receipt


def followup(root, manifest_path, stage, group, task):
    data = read_json(manifest_path.parent / "dispatch.json")
    matches = [
        item
        for item in data.get("followups", [])
        if (item.get("stage"), item.get("group"), item.get("reviewer_task")) == (stage, group, task)
    ]
    require(matches, "callback needs actual original-owner followup receipt")
    item = max(matches, key=lambda item: timestamp(item["called_at"]))
    receipt = item.get("followup_receipt", {})
    require(
        receipt.get("tool") == "collaboration.followup_task"
        and receipt.get("task_name") == task
        and technical._text(receipt.get("receipt")),
        "missing actual followup receipt",
    )
    return item


def append_event(path, payload):
    raw_lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    event = {
        **payload,
        "event_index": len(raw_lines),
        "recorded_at": now(),
        "previous_sha256": sha(raw_lines[-1]) if raw_lines else None,
    }
    with path.open("ab") as stream:
        stream.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
    return event


def raw_trace(root, reference):
    path = artifact(root, reference) if isinstance(reference, dict) else path_in(root, reference)
    lines = path.read_bytes().splitlines(keepends=True)
    require(lines, "empty trace")
    rows = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        require(
            row.get("event_index") == index
            and row.get("previous_sha256") == (sha(lines[index - 1]) if index else None),
            "trace hash chain/order changed",
        )
        timestamp(row["recorded_at"])
        if rows:
            require(timestamp(rows[-1]["recorded_at"]) <= timestamp(row["recorded_at"]), "trace time order changed")
        rows.append(row)
    require(rows[0].get("event") == "start", "trace needs actual session header")
    return rows


def session_units(root, manifest, order, primary=()):
    pages = {page["page_id"]: page for page in manifest["pages"]}
    units = []
    for pid in order:
        page = pages[pid]
        chunks = reader.units(path_in(root, page["snapshot"]).read_text(encoding="utf-8"))
        ends = [[] for _ in chunks]
        last_section = None
        if page["selector"] == "whole-file":
            for chunk_index, chunk in enumerate(chunks):
                for heading in re.finditer(r"(?m)^## ([A-Z\d]+\.\d+) .+$", chunk):
                    if last_section:
                        end_index = chunk_index if chunk[: heading.start()].strip() else chunk_index - 1
                        ends[max(0, end_index)].append(last_section)
                    last_section = heading[1]
            if last_section:
                ends[-1].append(last_section)
        ends[-1].append(pid)
        for index, chunk in enumerate(chunks):
            references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", chunk)
            references += re.findall(r"<img\b[^>]*\bsrc=[\'\"]([^\'\"]+)[\'\"]", chunk)
            visible = {}
            for reference in references:
                if "://" in reference or reference.startswith("data:"):
                    continue
                figure = relative(root, (path_in(root, page["source"]).parent / reference.split("#", 1)[0]))
                require(figure in page["figures_sha256"], "unit figure missing from frozen source")
                visible[figure] = page["figures_sha256"][figure]
            units.append(
                {
                    "page_id": pid,
                    "primary": pid in primary,
                    "unit_index": index,
                    "unit_sha256": sha(chunk.encode()),
                    "text": chunk,
                    "section_end": bool(ends[index]),
                    "required_section_ends": ends[index],
                    "source_sha256": page["source_sha256"],
                    "figures_sha256": page["figures_sha256"],
                    "visible_figures": visible,
                    "figure_snapshots": {figure: manifest["figure_snapshots"][figure]["path"] for figure in visible},
                }
            )
    return units


def start_session(
    root, manifest_path, stage, group, task, *, context=(), pages=(), recheck_of=(), issues=(), context_only=False
):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    manifest = load_manifest(root, manifest_path)
    declared = dispatch(root, manifest_path, manifest, stage, group, task)
    if stage != "reader" and not recheck_of and not context_only:
        prerequisite = audit_manifest(root, manifest_path, "reader" if stage == "technical" else "technical")
        require(
            not prerequisite["failures"],
            "previous review stage is incomplete: " + "; ".join(prerequisite["failures"][:3]),
        )
    if stage != "reader":
        ready = readiness(
            root, manifest_path, declared.get("ready_receipt"), "reader" if stage == "technical" else "technical"
        )
        require(
            timestamp(declared["dispatched_at"]) >= timestamp(ready["sealed_at"]),
            "initial next-stage spawn precedes prior complete receipt",
        )
    callback = followup(root, manifest_path, stage, group, task) if recheck_of else None
    chosen = next(item for item in manifest["groups"] if item["group"] == group)
    all_ids = {page["page_id"] for page in manifest["pages"]}
    primary = [] if context_only else list(pages or chosen["primary_page_ids"])
    require(set(primary) <= set(chosen["primary_page_ids"]), "page is outside dispatched primary scope")
    require(not pages or recheck_of, "partial primary reading requires an original recheck trace")
    prior = [] if context_only else chosen["context_page_ids"]
    prior = list(dict.fromkeys([*prior, *context]))
    require(set(prior) <= all_ids, "unknown prerequisite")
    prior = [pid for pid in prior if pid not in primary]
    require(primary or prior, "empty session")
    retained = []
    for name in recheck_of:
        trace_path = path_in(root, name)
        rows = raw_trace(root, name)
        require(
            rows[0]["reviewer_task"] == task and rows[0]["stage"] == stage and rows[0]["group"] == group,
            "recheck must use original owner",
        )
        old_issues = {issue["id"] for row in rows[1:] for issue in row.get("issues", [])}
        require(set(issues) <= old_issues, "recheck issue is absent from original trace")
        retained.append({"path": name, "sha256": sha(trace_path.read_bytes()), "issue_ids": list(issues)})
    session = uuid.uuid4().hex
    trace = manifest_path.parent / "traces" / stage / group / (session + ".jsonl")
    trace.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "event": "start",
        "session": session,
        "review_policy": POLICY,
        "batch_id": manifest["batch_id"],
        "manifest": relative(root, manifest_path),
        "manifest_sha256": sha(manifest_path.read_bytes()),
        "stage": stage,
        "group": group,
        "reviewer_task": task,
        "fork_turns": "none",
        "primary_page_ids": primary,
        "context_page_ids": prior,
        "order": prior + primary,
        "recheck_of": retained,
        "dispatch_at": declared["dispatched_at"],
        "followup": callback,
    }
    append_event(trace, header)
    state = {
        **header,
        "trace_file": relative(root, trace),
        "cursor": 0,
        "units": session_units(root, manifest, prior + primary, primary),
    }
    write_new(root / "outputs/grouped-review" / session / "state.json", state)
    return display(state, root)


def display(state, root):
    if state["cursor"] == len(state["units"]):
        return {"complete": True, "session": state["session"], "trace_file": state["trace_file"]}
    unit = state["units"][state["cursor"]]
    prior_figures = {figure for previous in state["units"][: state["cursor"]] for figure in previous["visible_figures"]}
    return {
        "session": state["session"],
        "trace_file": state["trace_file"],
        "page_id": unit["page_id"],
        "stage": state["stage"],
        "primary": unit["primary"],
        "four_questions_required": state["stage"] == "reader" and unit["primary"] and unit["section_end"],
        "unit_index": unit["unit_index"],
        "unit_sha256": unit["unit_sha256"],
        "section_end": unit["section_end"],
        "required_section_ends": unit["required_section_ends"],
        "text": unit["text"],
        "current_figures": {
            name: str(path_in(root, unit["figure_snapshots"][name])) for name in unit["visible_figures"]
        },
        "already_read_figures": sorted(prior_figures),
    }


def evidence(root, claim, unit, header, trace_file, event_index):
    require(isinstance(claim, dict) and technical._text(claim.get("claim")), "core claim needs its own statement")
    status = claim.get("status")
    require(status in {"explicit", "prior", "inferred", "unknown"}, "unknown evidence source status")
    if status == "unknown":
        require(
            type(claim.get("needed_now")) is bool and technical._text(claim.get("reason")),
            "unknown claim needs what is missing and whether needed now",
        )
        if claim["needed_now"]:
            require(technical._text(claim.get("issue_id")), "necessary unknown must enter issue list")
        return
    require(technical._text(claim.get("quote")), "each core claim needs an actual quotation")
    if status == "inferred":
        require(technical._text(claim.get("reason")), "inference needs its relation to read text")
    cited_trace = claim.get("trace_file", trace_file)
    cited_event = claim.get("event_index", event_index)
    if cited_trace == trace_file and cited_event == event_index:
        require(status != "prior", "current unit cannot be called prior reading")
        require(claim.get("page_id", unit["page_id"]) == unit["page_id"], "wrong current source page")
        cited = unit
        if not claim.get("figure"):
            require(claim["quote"] in unit["text"], "current quotation is absent")
            return
    else:
        rows = raw_trace(root, cited_trace)
        require(
            (rows[0]["reviewer_task"], rows[0]["batch_id"], rows[0]["stage"], rows[0]["group"])
            == (header["reviewer_task"], header["batch_id"], header["stage"], header["group"]),
            "prerequisite must actually be read by this owner in this stage",
        )
        require(
            type(cited_event) is int and 0 < cited_event < len(rows), "prior quotation references missing checkpoint"
        )
        row = rows[cited_event]
        require(row["event"] == "checkpoint", "prior reference is not an actual reading checkpoint")
        if cited_trace == trace_file:
            require(cited_event < event_index, "unread future checkpoint cited")
        else:
            cutoff = header.get("checking_at", now())
            require(
                timestamp(row["recorded_at"]) <= timestamp(cutoff), "prerequisite read occurs after this checkpoint"
            )
        frozen_path = path_in(root, rows[0]["manifest"])
        require(sha(frozen_path.read_bytes()) == rows[0]["manifest_sha256"], "prior source manifest changed")
        frozen = load_manifest(root, frozen_path)
        dispatch(
            root,
            frozen_path,
            frozen,
            rows[0]["stage"],
            rows[0]["group"],
            rows[0]["reviewer_task"],
            at=rows[0]["dispatch_at"],
        )
        cited = next(
            candidate
            for candidate in session_units(root, frozen, [row["page_id"]])
            if candidate["unit_index"] == row["unit_index"]
        )
        require(
            row["unit_sha256"] == cited["unit_sha256"] and row["source_sha256"] == cited["source_sha256"],
            "prior checkpoint version mismatch",
        )
        require(claim.get("page_id", row["page_id"]) == row["page_id"], "prior quotation cites wrong page")
        if not claim.get("figure"):
            require(claim["quote"] in cited["text"], "prior quotation not found in actually read unit")
            return
    figure = claim["figure"]
    require(
        figure in cited["visible_figures"]
        and claim.get("figure_sha256") == cited["visible_figures"][figure]
        and technical._text(claim.get("locator")),
        "graphical evidence must refer to a currently/past revealed figure, its SHA and actual location",
    )
    receipt = read_json(artifact(root, claim.get("visual_receipt")))
    require(
        receipt.get("reviewer_task") == header["reviewer_task"]
        and receipt.get("source_sha256") == claim["figure_sha256"],
        "graphical quotation lacks actual owner/source view receipt",
    )
    require(
        receipt.get("tool") in {"view_image", "functions.view_image", "tools.view_image", "playwright"},
        "graphical citation is not an actual visual receipt",
    )
    require(
        timestamp(receipt["received_at"]) <= timestamp(header.get("checking_at", now())),
        "graphical view occurs after checkpoint",
    )
    # Graphical relationships cannot be checked by a string search in SVG markup.
    # The quote/locator is the reader's recorded observation, never an automatic scientific judgment.


def timings(check):
    value = check.get("at")
    values = [value] if isinstance(value, str) else value
    require(
        isinstance(values, list) and values and set(values) <= {"first_use", "section_end"}, "invalid question timing"
    )
    return values


def checkpoint_errors(root, note, unit, header, trace_file, event_index):
    require(
        note.get("page_id") == unit["page_id"] and note.get("unit_index") == unit["unit_index"],
        "checkpoint must describe current unit",
    )
    for field in reader.FIELDS:
        require(technical._text(note.get(field)), "missing actual reading note: " + field)
    checks = note.get("four_questions", [])
    require(isinstance(checks, list), "four_questions must be a list")
    formal_reader = header["stage"] == "reader" and unit["primary"]
    if formal_reader and unit["section_end"]:
        scopes = {
            scope
            for check in checks
            if "section_end" in timings(check)
            for scope in (
                [check.get("scope", unit["page_id"])]
                if isinstance(check.get("scope", unit["page_id"]), str)
                else check["scope"]
            )
        }
        require(set(unit["required_section_ends"]) <= scopes, "four questions required at each section end")
    introduced = note.get("introduced_methods", [])
    require(
        isinstance(introduced, list) and all(technical._text(name) for name in introduced),
        "reader must identify introduced methods, or explicitly record []",
    )
    for check in checks:
        timings(check)
        require(technical._text(check.get("subject")), "four questions need timing and main subject")
        questions = check.get("questions", [])
        require(
            [question.get("number") for question in questions] == [1, 2, 3, 4],
            "four questions must be individually answered in order",
        )
        for question in questions:
            require(
                technical._text(question.get("answer"))
                and isinstance(question.get("claims"), list)
                and question["claims"],
                "each question needs reader answer and core-claim evidence",
            )
            for claim in question["claims"]:
                evidence(root, claim, unit, header, trace_file, event_index)
    methods = note.get("method_introductions", [])
    require(isinstance(methods, list), "method_introductions must be a list")
    if formal_reader:
        require(
            {method.get("name") for method in methods} == set(introduced),
            "each introduced method needs its own original-sentence comparison",
        )
    for method in methods:
        if formal_reader:
            require(
                any("first_use" in timings(check) and check["subject"] == method["name"] for check in checks),
                "new named method lacks four first-use answers",
            )
        for field in ("identity", "purpose", "step_relation"):
            item = method.get(field, {})
            require(
                item.get("assessment") in {"sufficient", "needs_explanation", "not_yet_needed"}
                and technical._text(item.get("reason")),
                "method introduction needs three separate assessments and reasons",
            )
            require(
                isinstance(item.get("evidence"), list) and item["evidence"],
                "method comparison needs original-sentence evidence",
            )
            for claim in item["evidence"]:
                evidence(root, claim, unit, header, trace_file, event_index)
            if item["assessment"] == "needs_explanation":
                require(technical._text(item.get("issue_id")), "missing method connection must enter issues")
    issues = note.get("issues", [])
    require(
        isinstance(issues, list) and len({issue.get("id") for issue in issues}) == len(issues),
        "issues must have distinct IDs",
    )
    for issue in issues:
        require(
            all(
                technical._text(issue.get(field))
                for field in ("id", "type", "severity", "location", "quote", "understanding", "missing")
            ),
            "issue needs original location, quote, understanding and missing connection",
        )
        require(issue["severity"] in {"blocker", "burden", "optional"}, "unknown issue severity")
        require(issue["quote"] in unit["text"], "issue quote is absent from current unit")
    claims = note.get("claims", [])
    require(isinstance(claims, list), "checkpoint claims must be a list")
    for claim in claims:
        evidence(root, claim, unit, header, trace_file, event_index)
    required_issues = {
        claim["issue_id"]
        for check in checks
        for q in check["questions"]
        for claim in q["claims"]
        if claim["status"] == "unknown" and claim["needed_now"]
    }
    required_issues.update(
        claim["issue_id"] for claim in claims if claim["status"] == "unknown" and claim["needed_now"]
    )
    required_issues.update(
        item["issue_id"]
        for method in methods
        for item in (method["identity"], method["purpose"], method["step_relation"])
        if item["assessment"] == "needs_explanation"
    )
    required_issues.update(
        claim["issue_id"]
        for method in methods
        for item in (method["identity"], method["purpose"], method["step_relation"])
        for claim in item["evidence"]
        if claim["status"] == "unknown" and claim["needed_now"]
    )
    require(
        required_issues <= {issue["id"] for issue in issues if issue["severity"] in {"blocker", "burden"}},
        "necessary introduction gap is missing or misclassified as optional",
    )
    require(
        isinstance(note.get("visual_checks", []), list) and isinstance(note.get("rechecks", []), list),
        "visual status and issue recheck IDs must be lists",
    )
    allowed_rechecks = {iid for reference in header.get("recheck_of", []) for iid in reference["issue_ids"]}
    require(
        set(note.get("rechecks", [])) <= allowed_rechecks, "recheck ID is not in retained original issue references"
    )


def next_unit(root, session, note):
    require(re.fullmatch(r"[0-9a-f]{32}", session), "unknown session")
    state_path = root / "outputs/grouped-review" / session / "state.json"
    state = read_json(state_path)
    require(state["cursor"] < len(state["units"]), "session already complete")
    trace_path = path_in(root, state["trace_file"])
    rows = raw_trace(root, state["trace_file"])
    require(len(rows) == state["cursor"] + 1, "state/trace mismatch; do not replay or overwrite a checkpoint")
    unit = state["units"][state["cursor"]]
    checkpoint_errors(root, note, unit, state, state["trace_file"], len(rows))
    fields = (
        *reader.FIELDS,
        "claims",
        "four_questions",
        "introduced_methods",
        "method_introductions",
        "issues",
        "visual_checks",
        "rechecks",
    )
    append_event(
        trace_path,
        {
            "event": "checkpoint",
            **{key: unit[key] for key in ("page_id", "unit_index", "unit_sha256", "source_sha256", "figures_sha256")},
            **{field: note.get(field, []) for field in fields},
        },
    )
    state["cursor"] += 1
    # Only mutable ignored session state is rewritten. The durable trace never is.
    with state_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    if state["cursor"] == len(state["units"]):
        append_event(trace_path, {"event": "complete", "session": session})
    return display(state, root)


def verify_trace(root, reference, task, stage, group, batch_id):
    rows = raw_trace(root, reference)
    header = rows[0]
    require(
        (header["reviewer_task"], header["stage"], header["group"], header["batch_id"])
        == (task, stage, group, batch_id),
        "trace owner/stage/group mismatch",
    )
    require(rows[-1].get("event") == "complete", "trace is incomplete")
    manifest_path = path_in(root, header["manifest"])
    require(sha(manifest_path.read_bytes()) == header["manifest_sha256"], "trace frozen manifest changed")
    frozen = load_manifest(root, manifest_path)
    declared = dispatch(root, manifest_path, frozen, stage, group, task, at=header["dispatch_at"])
    require(
        timestamp(header["recorded_at"]) >= timestamp(header["dispatch_at"]) >= timestamp(declared["dispatched_at"]),
        "reading precedes recorded dispatch",
    )
    assigned = next(item for item in frozen["groups"] if item["group"] == group)
    require(
        set(header["primary_page_ids"]) <= set(assigned["primary_page_ids"]), "trace claims undispatched primary pages"
    )
    require(
        header["order"] == header["context_page_ids"] + header["primary_page_ids"]
        and len(set(header["order"])) == len(header["order"]),
        "trace context/primary reading order changed",
    )
    if header["primary_page_ids"] and not header.get("recheck_of"):
        require(
            header["primary_page_ids"] == assigned["primary_page_ids"], "initial reading skipped assigned primary order"
        )
    units = session_units(root, frozen, header["order"], header["primary_page_ids"])
    notes = rows[1:-1]
    require(len(units) == len(notes), "actual trace unit count mismatch")
    for unit, note in zip(units, notes, strict=True):
        require(
            note.get("event") == "checkpoint" and note.get("unit_sha256") == unit["unit_sha256"],
            "revealed unit hash changed",
        )
        require(
            note.get("source_sha256") == unit["source_sha256"] and note.get("figures_sha256") == unit["figures_sha256"],
            "checkpoint source/figure version mismatch",
        )
        checkpoint_errors(
            root, note, unit, {**header, "checking_at": note["recorded_at"]}, reference["path"], note["event_index"]
        )
    for pid in header["order"]:
        selected = [note for note in notes if note["page_id"] == pid]
        expected = [unit for unit in units if unit["page_id"] == pid]
        source = next(page for page in frozen["pages"] if page["page_id"] == pid)
        errors = trace_audit.trace_errors(
            {
                "reviewer_task": task,
                "source": source["source"],
                "source_sha256": selected[0]["source_sha256"],
                "figure_sha256": selected[0]["figures_sha256"],
            },
            selected,
            task=task,
            source=source["source"],
            body=path_in(root, source["snapshot"]).read_bytes(),
            intro=b"",
            figures=source["figures_sha256"],
            units=[unit["text"] for unit in expected],
            fields=reader.FIELDS,
        )
        require(not errors, "; ".join(errors))
    for previous in header.get("recheck_of", []):
        original = raw_trace(root, previous)
        require(
            original[0]["reviewer_task"] == task and original[0]["session"] != header["session"],
            "recheck is not a new original-owner session",
        )
        require(
            timestamp(original[-1]["recorded_at"]) < timestamp(header["recorded_at"]), "recheck precedes original trace"
        )
        old_issues = {issue["id"] for row in original[1:-1] for issue in row["issues"]}
        require(set(previous["issue_ids"]) <= old_issues, "callback refers to invented original issue")
    if header.get("recheck_of"):
        callback = header.get("followup", {})
        receipt = callback.get("followup_receipt", {})
        require(
            receipt.get("tool") == "collaboration.followup_task"
            and receipt.get("task_name") == task
            and technical._text(receipt.get("receipt")),
            "callback lacks real followup authorization",
        )
        require(
            timestamp(header["recorded_at"]) >= timestamp(callback["called_at"]), "callback reading precedes followup"
        )
    return rows


def visual_check(root, value, task, *, figure=None):
    require(
        isinstance(value, dict)
        and value.get("status") in {"verified", "unverified"}
        and technical._text(value.get("details")),
        "visual check must explicitly state verified/unverified and actual scope",
    )
    if value["status"] == "unverified":
        return False
    require(technical._text(value.get("observation")), "verified visual check needs actual observation")
    receipt = read_json(artifact(root, value.get("receipt")))
    require(
        receipt.get("reviewer_task") == task
        and receipt.get("tool") in {"view_image", "functions.view_image", "tools.view_image", "playwright"},
        "visual receipt does not belong to actual reviewer/tool",
    )
    timestamp(receipt["received_at"])
    artifacts = value.get("artifacts", [])
    require(artifacts, "verified visual check needs preserved actual images")
    for reference in artifacts:
        target = path_in(root, reference.get("path"))
        require(technical.HASH.fullmatch(reference.get("sha256", "")), "rendered image needs SHA-256")
        if target.is_file():
            require(sha(target.read_bytes()) == reference["sha256"], "local rendered image changed")
        # PNGs may stay in outputs. The small durable receipt preserves their exact hashes and view scope.
        require(
            any(
                item.get("path") == reference["path"] and item.get("sha256") == reference["sha256"]
                for item in receipt.get("artifacts", [])
            ),
            "view receipt does not bind rendered image path/hash",
        )
    if figure:
        require(
            value.get("figure") == figure and value.get("source_sha256") == sha(path_in(root, figure).read_bytes()),
            "visual source figure version mismatch",
        )
        require(receipt.get("source_sha256") == value["source_sha256"], "visual receipt source figure SHA mismatch")
        if figure.endswith(".svg"):
            require({item.get("width") for item in artifacts} >= {640, 360}, "SVG needs actual 640 and360 width views")
    else:
        require(
            technical.HASH.fullmatch(value.get("source_sha256", ""))
            and receipt.get("source_sha256") == value["source_sha256"],
            "page view receipt needs canonical source SHA",
        )
        require(
            {tuple(item.get("viewport", [])) for item in artifacts} >= {(1280, 800), (390, 844)},
            "actual page needs desktop and mobile views",
        )
    return True


def technical_page(root, page, body):
    errors = []
    artifacts = technical._artifacts(root, page.get("artifacts"), errors)
    sources = technical._sources(root, page.get("sources"), artifacts, errors)
    claims = technical._claims(page.get("claims"), sources, artifacts, errors)
    if not claims:
        applicability = page.get("applicability", {})
        if (
            applicability.get("substantive_claims") is not False
            or not technical._text(applicability.get("reason"))
            or "```" in body
        ):
            errors.append("no substantive claims needs explicit applicability reason and no program/command fences")
    checks = page.get("checks", {})
    for name in technical.CHECKS:
        value = checks.get(name, {})
        allowed_na = (
            (name in {"factual_accuracy", "source_verification", "limitations"} and not claims)
            or (
                name == "numeric_verification"
                and not any(c["kind"] in {"numeric", "empirical"} for c in claims.values())
            )
            or (name == "figure_consistency" and not page["figures_sha256"])
        )
        if not technical._text(value.get("details")) or (
            value.get("status") != "pass" and not (value.get("status") == "not_applicable" and allowed_na)
        ):
            errors.append("technical check missing or failed: " + name)
        technical._references(value.get("claim_ids", []), claims, errors, name)
    require(not errors, "; ".join(errors))


def page_issues(root, page, traces, task):
    original = {}
    for rows in traces.values():
        for row in rows[1:-1]:
            if row.get("page_id") == page["page_id"]:
                for issue in row["issues"]:
                    original.setdefault(issue["id"], (issue, rows, row))
    final = {issue["id"]: issue for issue in page["issues"]}
    require(
        len(final) == len(page["issues"]) and set(original) <= set(final), "early issue was removed from final report"
    )
    for iid, issue in final.items():
        require(issue.get("severity") in {"blocker", "burden", "optional"}, "report issue lacks severity")
        if iid in original:
            require(issue["severity"] == original[iid][0]["severity"], "early severity cannot be silently rewritten")
        if issue["severity"] == "optional" and issue.get("status") in {"optional", "clarified_later"}:
            require(technical._text(issue.get("resolution")), "optional issue needs disposition")
            continue
        require(
            issue.get("status") == "resolved" and technical._text(issue.get("resolution")),
            "necessary issue remains unresolved",
        )
        ref = issue.get("recheck", {})
        rows = traces.get(ref.get("trace_file"))
        require(rows and rows[0]["reviewer_task"] == task, "missing original-owner recheck trace")
        event = ref.get("event_index")
        require(
            type(event) is int and 0 < event < len(rows) - 1 and iid in rows[event]["rechecks"],
            "issue recheck is not actually recorded",
        )
        require(rows[event].get("page_id") == page["page_id"], "issue recheck is on another page")
        previous = rows[0].get("recheck_of", [])
        require(
            any(
                iid in item["issue_ids"]
                and item["path"] in traces
                and traces[item["path"]][0]["session"] != rows[0]["session"]
                for item in previous
            ),
            "recheck must reference preserved original issue in another session",
        )


def group_report(root, manifest_path, manifest, group, stage, entry, unverified):
    report_path = artifact(root, entry)
    report = read_json(report_path)
    task = report.get("reviewer_task")
    require(
        (
            report.get("schema_version"),
            report.get("review_policy"),
            report.get("batch_id"),
            report.get("stage"),
            report.get("group"),
        )
        == (1, POLICY, manifest["batch_id"], stage, group["group"]),
        "wrong report batch/stage/group",
    )
    history = versions(root, manifest_path)
    require(
        report.get("reviewer_context") == "fresh" and report.get("manifest_sha256") in history,
        "report is from another batch or not a fresh reviewer",
    )
    declared = dispatch(root, manifest_path, manifest, stage, group["group"], task)
    require(timestamp(entry["recorded_at"]) >= timestamp(declared["dispatched_at"]), "collection precedes dispatch")
    require(report.get("verdict") == "pass", "group still requests revision")
    refs = report.get("trace_files", [])
    require(refs and len({ref["path"] for ref in refs}) == len(refs), "report needs distinct actual trace references")
    previous_reports = [
        json.loads(line) for line in (manifest_path.parent / "collections.jsonl").read_bytes().splitlines()
    ]
    retained_paths = set()
    for previous in previous_reports:
        if (previous["stage"], previous["group"]) == (stage, group["group"]) and previous["path"] != entry["path"]:
            retained = read_json(artifact(root, previous))
            require(retained["reviewer_task"] == task, "current owner replaced actual original group owner")
            retained_paths.update(reference["path"] for reference in retained.get("trace_files", []))
    require(
        retained_paths <= {reference["path"] for reference in refs},
        "new report omitted original immutable reading traces",
    )
    traces = {ref["path"]: verify_trace(root, ref, task, stage, group["group"], manifest["batch_id"]) for ref in refs}
    reported = report.get("pages", [])
    require(
        [page["page_id"] for page in reported] == group["primary_page_ids"],
        "group report does not cover exact primary order",
    )
    frozen = {page["page_id"]: page for page in manifest["pages"]}
    actual_context = {pid for rows in traces.values() for pid in rows[0]["context_page_ids"]}
    require(set(group["context_page_ids"]) <= actual_context, "assigned prerequisite pages were not actually read")
    for page in reported:
        source = frozen[page["page_id"]]
        require(
            page.get("source_sha256") == source["source_sha256"]
            and page.get("figures_sha256") == source["figures_sha256"],
            "reported source/figure hashes are stale",
        )
        require(
            page.get("verdict") == "pass" and technical._text(page.get("summary")),
            "page needs own actual understanding/verdict",
        )
        rows = traces.get(page.get("trace_file"))
        require(rows and page["page_id"] in rows[0]["primary_page_ids"], "page lacks current primary reading trace")
        notes = [row for row in rows[1:-1] if row["page_id"] == page["page_id"]]
        require(
            notes and all(row["source_sha256"] == source["source_sha256"] for row in notes),
            "current page was not actually read",
        )
        expected_refs = [row["event_index"] for row in notes if row["four_questions"]]
        require(page.get("question_refs") == expected_refs, "report omits or invents first-use/end question references")
        require(
            isinstance(page.get("issues"), list) and technical._text(page.get("missing_visuals")),
            "page needs issue list and actual missing-visual judgment",
        )
        page_issues(root, page, traces, task)
        # Current reports may add corrected receipt metadata without rewriting
        # immutable first-reading traces. Validate that explicit evidence first;
        # the same owner/source/render requirements still apply to each candidate.
        visuals = page.get("visual_checks", []) + [item for row in notes for item in row["visual_checks"]]
        for figure in source["figures_sha256"]:
            checked = [item for item in visuals if item.get("figure") == figure]
            require(
                checked and any(visual_check(root, item, task, figure=figure) for item in checked),
                "necessary figure not actually verified: " + figure,
            )
        placement = page.get("page_visual_check", {})
        if placement.get("status") == "verified":
            require(placement.get("source_sha256") == source["source_sha256"], "page visual source version mismatch")
        verified = visual_check(root, placement, task)
        if not verified:
            unverified.append(f"{stage}/{group['group']}/{page['page_id']}: actual page layout unverified")
            require(
                not source["figures_sha256"] and placement.get("required") is False,
                "necessary desktop/mobile page check remains unverified",
            )
        if stage == "reader":
            variation = page.get("variation", {})
            require(
                (variation.get("not_applicable") is True and technical._text(variation.get("reason")))
                or all(technical._text(variation.get(field)) for field in ("change", "prediction", "reason")),
                "actual variation prediction or explicit nonapplicability needed",
            )
        elif stage == "technical":
            technical_page(root, page, path_in(root, source["snapshot"]).read_text(encoding="utf-8"))
        else:
            checks = page.get("continuity_checks", {})
            require(
                all(
                    technical._text(checks.get(field))
                    for field in ("prior_relation", "transition", "new_prerequisites", "terminology", "example_change")
                ),
                "each continuity page needs actual transition notes",
            )
    return task, declared


def collection_entries(root, manifest_path):
    path = manifest_path.parent / "collections.jsonl"
    require(path.is_file(), "no actual reports collected for this batch")
    entries = raw_collection = [json.loads(line) for line in path.read_bytes().splitlines()]
    latest, seen_paths = {}, {}
    for entry in raw_collection:
        require(entry.get("stage") in STAGES and technical._text(entry.get("group")), "invalid collection slot")
        artifact(root, entry)
        if entry["path"] in seen_paths:
            require(seen_paths[entry["path"]] == entry["sha256"], "previous report path overwritten")
        seen_paths[entry["path"]] = entry["sha256"]
        latest[(entry["stage"], entry["group"])] = entry
    require(entries, "empty report collection cannot pass")
    return latest


def collect(root, manifest_path, report_path, receipt_path):
    manifest = load_manifest(root, manifest_path)
    report, receipt = read_json(report_path), read_json(receipt_path)
    stage, group, task = report["stage"], report["group"], report["reviewer_task"]
    dispatch(root, manifest_path, manifest, stage, group, task)
    require(
        receipt.get("final_received") is True
        and receipt.get("reviewer_task") == task
        and technical._text(receipt.get("receipt")),
        "collection needs actual original-owner final receipt",
    )
    timestamp(receipt["received_at"])
    entry = {
        "stage": stage,
        "group": group,
        "reviewer_task": task,
        "path": relative(root, report_path),
        "sha256": sha(report_path.read_bytes()),
        "final_receipt": {"path": relative(root, receipt_path), "sha256": sha(receipt_path.read_bytes())},
        "recorded_at": now(),
    }
    path = manifest_path.parent / "collections.jsonl"
    if path.exists():
        previous = collection_entries(root, manifest_path)
        require(
            not any(item["path"] == entry["path"] for item in previous.values()),
            "report path already collected; use a new immutable report",
        )
    with path.open("ab") as stream:
        stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
    return entry


def audit_manifest(root, manifest_path, stage="all"):
    failures, unverified, owners, records = [], [], {}, []
    try:
        require(stage in {"reader", "technical", "all"}, "unknown gate stage")
        manifest = load_manifest(root, manifest_path)
        current = current_inventory(root)
        current_pages = {page["page_id"]: page for page in current["pages"]}
        require(
            set(current_pages) == {page["page_id"] for page in manifest["pages"]},
            "active batch does not cover complete current canonical inventory",
        )
        for page in manifest["pages"]:
            candidate = current_pages[page["page_id"]]
            require(
                all(
                    page.get(field) == candidate.get(field)
                    for field in (
                        "source",
                        "selector",
                        "source_sha256",
                        "figures_sha256",
                        "notebook",
                        "notebook_sha256",
                    )
                ),
                "active source/notebook version is stale: " + page["page_id"],
            )
        collection = collection_entries(root, manifest_path)
        selected = STAGES[:1] if stage == "reader" else STAGES[:2] if stage == "technical" else STAGES
        for selected_stage in selected:
            received = []
            for group in manifest["groups"]:
                try:
                    entry = collection.get((selected_stage, group["group"]))
                    require(entry, "stage/group not collected")
                    final = read_json(artifact(root, entry.get("final_receipt")))
                    require(
                        final.get("final_received") is True
                        and final.get("reviewer_task") == entry["reviewer_task"]
                        and technical._text(final.get("receipt")),
                        "missing actual final receipt",
                    )
                    task, declared = group_report(
                        root, manifest_path, manifest, group, selected_stage, entry, unverified
                    )
                    require(
                        task == entry["reviewer_task"] and task not in owners, "same owner reused across groups/stages"
                    )
                    if selected_stage != "reader":
                        ready = readiness(
                            root,
                            manifest_path,
                            declared.get("ready_receipt"),
                            "reader" if selected_stage == "technical" else "technical",
                        )
                        require(
                            timestamp(declared["dispatched_at"]) >= timestamp(ready["sealed_at"]),
                            "initial stage spawn before prior complete receipt",
                        )
                    require(
                        timestamp(final["received_at"]) <= timestamp(entry["recorded_at"]),
                        "collection before final actually received",
                    )
                    owners[task] = (selected_stage, group["group"])
                    received.append(timestamp(entry["recorded_at"]))
                    records.append(
                        {
                            "stage": selected_stage,
                            "group": group["group"],
                            "reviewer_task": task,
                            "report_path": entry["path"],
                            "report_sha256": entry["sha256"],
                        }
                    )
                except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
                    failures.append(f"{selected_stage}/{group['group']}: {error}")
            if len(received) != len(manifest["groups"]):
                require(selected_stage == selected[-1], "previous stage is incomplete")
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
        failures.append(str(error))
    return {
        "schema_version": 1,
        "review_policy": POLICY,
        "stage": stage,
        "status": "failed" if failures else "passed_metadata_checks",
        "records": records,
        "unverified": unverified,
        "failures": failures,
        "scope_note": SCOPE_NOTE,
    }


def check_active(root=ROOT, stage="all"):
    root = Path(root).resolve()
    pointer = root / ACTIVE
    if not pointer.exists():
        return None
    try:
        active = read_json(pointer)
        if active.get("review_policy") == "tutorial_composite_v1":
            spec = importlib.util.spec_from_file_location(
                "active_composite_review", ROOT / "docs/review-tools/composite_review.py"
            )
            composite = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(composite)
            return composite.check_active(root, active, stage)
        require(
            active.get("schema_version") == 1 and active.get("review_policy") == POLICY,
            "invalid active pointer; legacy fallback forbidden",
        )
        manifest_path = artifact(root, {"path": active.get("manifest"), "sha256": active.get("manifest_sha256")})
        return audit_manifest(root, manifest_path, stage)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"status": "failed", "stage": stage, "failures": [str(error)], "scope_note": SCOPE_NOTE}


def route_active(root=ROOT, stage="all", output=None):
    result = check_active(root, stage)
    if result is None:
        return False
    print(json.dumps(result, ensure_ascii=False))
    if result["failures"]:
        raise SystemExit(1)
    if output:
        write_new(output, result)
    return True


def seal_stage(root, manifest_path, stage):
    require(
        stage in {"reader", "technical"}, "only completed reader/technical stages need next-stage readiness receipts"
    )
    result = audit_manifest(root, manifest_path, stage)
    require(not result["failures"], "cannot seal incomplete review stage")
    result.update(
        {
            "manifest": relative(root, manifest_path),
            "manifest_sha256": sha(manifest_path.read_bytes()),
            "sealed_at": now(),
        }
    )
    destination = manifest_path.parent / "readiness" / (stage + "-" + uuid.uuid4().hex + ".json")
    write_new(destination, result)
    return {"path": relative(root, destination), "sha256": sha(destination.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--groups", type=Path, required=True)
    frozen.add_argument("--expected-pages", type=int, default=325)
    frozen.add_argument("--batch-id", default="phase7")
    frozen.add_argument("--author", action="append", default=[])
    frozen.add_argument("--exclude-reviewer", action="append", default=[])
    frozen.add_argument(
        "--previous", type=Path, help="new same-batch freeze; retain prior dispatch and immutable collections"
    )
    start = commands.add_parser("start")
    start.add_argument("group")
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--stage", choices=STAGES, default="reader")
    start.add_argument("--reviewer", required=True)
    start.add_argument("--context", action="append", default=[])
    start.add_argument("--context-only", action="store_true")
    start.add_argument("--page", action="append", default=[])
    start.add_argument("--recheck-of", action="append", default=[])
    start.add_argument("--issue", action="append", default=[])
    advance = commands.add_parser("next")
    advance.add_argument("session")
    advance.add_argument("--checkpoint", type=Path, required=True)
    receive = commands.add_parser("collect")
    receive.add_argument("--manifest", type=Path, required=True)
    receive.add_argument("--report", type=Path, required=True)
    receive.add_argument("--receipt", type=Path, required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--stage", choices=("reader", "technical", "all"), default="all")
    gate.add_argument("--manifest", type=Path, help="check explicit freeze without creating or changing active pointer")
    ready = commands.add_parser("seal-stage")
    ready.add_argument("--stage", choices=("reader", "technical"), required=True)
    ready.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            manifest = freeze(
                ROOT,
                args.output,
                args.groups,
                expected_pages=args.expected_pages,
                batch_id=args.batch_id,
                authors=args.author,
                excluded=args.exclude_reviewer,
                previous=args.previous,
            )
            result = {
                "manifest": relative(ROOT, args.output / "manifest.json"),
                "pages": len(manifest["pages"]),
                "reviewed": False,
                "dispatched": False,
            }
        elif args.command == "start":
            result = start_session(
                ROOT,
                args.manifest,
                args.stage,
                args.group,
                args.reviewer,
                context=args.context,
                pages=args.page,
                recheck_of=args.recheck_of,
                issues=args.issue,
                context_only=args.context_only,
            )
        elif args.command == "next":
            result = next_unit(ROOT, args.session, read_json(args.checkpoint))
        elif args.command == "collect":
            result = collect(ROOT, args.manifest, args.report, args.receipt)
        elif args.command == "seal-stage":
            result = seal_stage(ROOT, args.manifest, args.stage)
        else:
            result = (
                audit_manifest(ROOT, args.manifest, args.stage) if args.manifest else check_active(ROOT, args.stage)
            )
            require(result is not None, "no active grouped batch; existing review CLIs retain legacy mode")
            print(json.dumps(result, ensure_ascii=False))
            raise SystemExit(bool(result["failures"]))
        print(json.dumps(result, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
        parser.exit(1, "Grouped review error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
