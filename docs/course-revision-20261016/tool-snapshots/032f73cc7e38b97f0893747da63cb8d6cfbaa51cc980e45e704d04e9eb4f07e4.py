"""逐次開放一個凍結正文單位，先保存讀者原始判斷才開下一段。"""

import argparse
import hashlib
import importlib.util
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "docs/course-revision-20261016"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_page(page_id, manifest_path=None):
    manifest = json.loads((manifest_path or BATCH / "freeze-01.json").read_bytes())
    page = next(p for p in manifest["pages"] if p["page_id"] == page_id)
    raw = (ROOT / page["snapshot"]).read_bytes()
    if sha(raw) != page["source_sha256"]:
        raise ValueError("凍結原文已改變")
    return page, raw.decode("utf-8")


def load_state_page(state):
    reference = state.get("source_freeze")
    path = ROOT / reference["path"] if reference else None
    if reference and sha(path.read_bytes()) != reference["sha256"]:
        raise ValueError("閱讀中的來源manifest已改變")
    return load_page(state["page_id"], path)


def units(body):
    def hide(match):
        summary = re.search(r"<summary[^>]*>(.*?)</summary>", match[0], re.S)
        label = re.sub("<[^>]+>", "", summary[1]).strip() if summary else "選讀"
        return "\n[原位置折疊選讀：" + label + "；正文路線完成後才補讀]\n"

    main = re.sub(r"<details\b[^>]*>.*?</details>", hide, body, flags=re.S)
    spec = importlib.util.spec_from_file_location("unit_scheme", ROOT / "docs/review-tools/incremental_reader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.units(main)


def reading_plan(path, reviewer, freeze_path):
    if path is None:
        raise ValueError("新的閱讀操作需要作者事先凍結的--plan；舊session原始紀錄仍保留")
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("reading plan需在repo內")
    plan = json.loads(path.read_bytes())
    if plan.get("reviewer_task") != reviewer or plan.get("author_task") != "/root":
        raise ValueError("reading plan身分不符")
    if not plan.get("main_pages") or set(plan["main_pages"]) & set(plan.get("context_pages", [])):
        raise ValueError("正文與先備範圍需明確分開")
    if plan.get("source_freeze") != {
        "path": freeze_path.relative_to(ROOT).as_posix(),
        "sha256": sha(freeze_path.read_bytes()),
    }:
        raise ValueError("reading plan凍結來源不符")
    if datetime.fromisoformat(plan["created_at"]) > datetime.now(UTC):
        raise ValueError("reading plan尚未建立")
    return plan, {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def supplement_allowed(plan, plan_ref, page_id, reviewer):
    if page_id in plan.get("context_pages", []):
        return "explicit_prerequisite_context"
    if page_id not in plan["main_pages"]:
        raise ValueError("不在派定閱讀範圍")
    completed = set()
    for path in (BATCH / "reader-sessions").glob("*/state.json"):
        state = json.loads(path.read_bytes())
        if state.get("reviewer_task") != reviewer or state.get("reading_plan") != plan_ref:
            continue
        _, body = load_state_page(state)
        if state["next_index"] == len(units(body)):
            completed.add(state["page_id"])
    if not set(plan["main_pages"]) <= completed:
        raise ValueError("必須完成整組正文checkpoint後才能補讀選讀")
    return "after_all_group_main_checkpoints"


def frozen_figure(state, page, name):
    freeze_ref = state.get("source_freeze")
    freeze_path = ROOT / freeze_ref["path"] if freeze_ref else BATCH / "freeze-01.json"
    if freeze_ref and sha(freeze_path.read_bytes()) != freeze_ref["sha256"]:
        raise ValueError("來源freeze已改變")
    freeze = json.loads(freeze_path.read_bytes())
    digest = page["figures_sha256"][name]
    reference = freeze.get("figure_snapshots", {}).get(name)
    if reference is None:
        # 早期真實session沒有figure map；使用既已保存的同hash副本，不轉向live圖。
        baseline = json.loads((ROOT / "docs/course-publish-20261008/current-source-freeze.json").read_bytes())
        reference = next((ref for ref in baseline["figures"].values() if ref["sha256"] == digest), None)
        if reference is None:
            reference = {"path": "docs/course-revision-20261016/frozen-figures/" + digest + ".svg", "sha256": digest}
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT) or sha(path.read_bytes()) != digest or reference["sha256"] != digest:
        raise ValueError("凍結圖缺失或已改變")
    return path


def display(state):
    page, raw = load_state_page(state)
    chunks = units(raw)
    i = state["next_index"]
    if i == len(chunks):
        print(json.dumps({"complete": True, "page_id": page["page_id"], "session": state["session"]}))
        return
    print(
        json.dumps(
            {
                "session": state["session"],
                "page_id": page["page_id"],
                "unit_index": i,
                "units_total": len(chunks),
                "unit_sha256": sha(chunks[i].encode()),
            }
        )
    )
    print(chunks[i])
    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", chunks[i]):
        name = ((ROOT / page["source"]).parent / reference).resolve().relative_to(ROOT).as_posix()
        print("本段凍結圖：", frozen_figure(state, page, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    start = sub.add_parser("start")
    start.add_argument("page_id")
    start.add_argument("--reviewer", required=True)
    start.add_argument("--plan", type=Path, required=True)
    start.add_argument("--freeze", type=Path, default=BATCH / "freeze-01.json")
    advance = sub.add_parser("next")
    advance.add_argument("session")
    advance.add_argument("--checkpoint", type=Path, required=True)
    full = sub.add_parser("supplement")
    full.add_argument("page_id")
    full.add_argument("--reviewer", required=True)
    full.add_argument("--plan", type=Path, required=True)
    full.add_argument("--freeze", type=Path, default=BATCH / "freeze-01.json")
    args = parser.parse_args()
    if args.operation in {"start", "supplement"}:
        if not re.fullmatch(r"/root/[a-zA-Z0-9_/-]+", args.reviewer):
            raise ValueError("需要真正的審閱task名稱")
        freeze_path = args.freeze.resolve()
        if not freeze_path.is_relative_to(ROOT):
            raise ValueError("來源manifest需在repository內")
        plan, plan_ref = reading_plan(args.plan, args.reviewer, freeze_path)
        page, raw = load_page(args.page_id, freeze_path)
        if args.operation == "supplement":
            purpose = supplement_allowed(plan, plan_ref, args.page_id, args.reviewer)
            receipt = BATCH / "supplement-access" / (uuid.uuid4().hex + ".json")
            write_new(
                receipt,
                {
                    "reviewer_task": args.reviewer,
                    "page_id": args.page_id,
                    "reading_plan": plan_ref,
                    "purpose": purpose,
                    "accessed_at": datetime.now(UTC).isoformat(),
                    "source_sha256": page["source_sha256"],
                },
            )
            print(raw)
            return
        if args.page_id not in plan["main_pages"]:
            raise ValueError("start需為派定正文頁")
        session = uuid.uuid4().hex
        state = {
            "schema_version": 1,
            "session": session,
            "page_id": args.page_id,
            "source_sha256": page["source_sha256"],
            "figures_sha256": page["figures_sha256"],
            "reviewer_task": args.reviewer,
            "next_index": 0,
            "started_at": datetime.now(UTC).isoformat(),
            "reading_plan": plan_ref,
            "source_freeze": {
                "path": freeze_path.relative_to(ROOT).as_posix(),
                "sha256": sha(freeze_path.read_bytes()),
            },
        }
        write_new(BATCH / "reader-sessions" / session / "state.json", state)
        write_new(BATCH / "reader-sessions" / session / "opening.json", state)
        display(state)
        return
    if not re.fullmatch(r"[a-f0-9]{32}", args.session):
        raise ValueError("無效session")
    folder = BATCH / "reader-sessions" / args.session
    path = folder / "state.json"
    state = json.loads(path.read_bytes())
    page, raw = load_state_page(state)
    if page["source_sha256"] != state["source_sha256"]:
        raise ValueError("閱讀中不可換稿")
    chunks = units(raw)
    i = state["next_index"]
    note = json.loads(args.checkpoint.read_bytes())
    if i >= len(chunks) or note.get("unit_index") != i:
        raise ValueError("checkpoint必須對應當前單位")
    for field in ("understanding", "materials_and_labels", "expected_change", "confusion_and_quote", "missing_visuals"):
        if not isinstance(note.get(field), str) or not note[field].strip():
            raise ValueError("缺少讀者自己的判斷：" + field)
    questions = note.get("concept_checks")
    if not isinstance(questions, list) or not questions:
        raise ValueError("需按當段實際概念保存四題，不以五欄摘要代替")
    for check in questions:
        for field in ("concept", "identity", "mechanism", "need", "usage", "example_scope", "source_quote", "verdict"):
            if not isinstance(check.get(field), str) or not check[field].strip():
                raise ValueError("四題缺欄：" + field)
    note.update(
        page_id=state["page_id"],
        reviewer_task=state["reviewer_task"],
        unit_sha256=sha(chunks[i].encode()),
        saved_at=datetime.now(UTC).isoformat(),
    )
    write_new(folder / f"note-{i:03}.json", note)
    state["next_index"] += 1
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    display(state)


if __name__ == "__main__":
    main()
