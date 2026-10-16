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
        print("本段圖：", (ROOT / page["source"]).parent / reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    start = sub.add_parser("start")
    start.add_argument("page_id")
    start.add_argument("--reviewer", required=True)
    start.add_argument("--freeze", type=Path, default=BATCH / "freeze-01.json")
    advance = sub.add_parser("next")
    advance.add_argument("session")
    advance.add_argument("--checkpoint", type=Path, required=True)
    full = sub.add_parser("supplement")
    full.add_argument("page_id")
    full.add_argument("--reviewer", required=True)
    full.add_argument("--freeze", type=Path, default=BATCH / "freeze-01.json")
    args = parser.parse_args()
    if args.operation in {"start", "supplement"}:
        if not re.fullmatch(r"/root/[a-zA-Z0-9_/-]+", args.reviewer):
            raise ValueError("需要真正的審閱task名稱")
        freeze_path = args.freeze.resolve()
        if not freeze_path.is_relative_to(ROOT):
            raise ValueError("來源manifest需在repository內")
        page, raw = load_page(args.page_id, freeze_path)
        if args.operation == "supplement":
            print(raw)
            return
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
