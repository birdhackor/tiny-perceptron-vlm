"""逐段提供凍結正文；保存讀者當場記錄，不產生判定或審閱摘要。"""

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/incremental-reader"
FIELDS = ("understanding", "materials_and_labels", "expected_change", "confusion_and_quote", "missing_visuals")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory():
    paths = sorted((ROOT / "course/chapters").glob("*.md"))
    paths += [ROOT / "course" / f for f in ("README.md", "first-steps.md", "glossary.md", "training.md")]
    result = {}
    for source in paths:
        raw = source.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^## ([A-Z\d]+\.\d+) .+$", raw, re.M))
        for i, heading in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
            result[heading[1]] = (source, raw[heading.start() : end], raw[: headings[0].start()] if i == 0 else "")
    return result


def blocks(raw):
    current, inside = [], False
    for line in raw.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            inside = not inside
        if not inside and not line.strip() and current:
            yield "".join(current)
            current = []
        else:
            current.append(line)
    if current:
        yield "".join(current)


def units(raw):
    chunks, current = [], []
    attach_next = False
    for block in blocks(raw):
        if block.lstrip().startswith("```"):
            if current:
                chunks.append("\n".join(current))
                current = []
            chunks.append(block)
            attach_next = False
            continue
        if current and sum(len(x) for x in current) >= 650 and not attach_next:
            chunks.append("\n".join(current))
            current = []
        current.append(block)
        attach_next = bool(re.search(r"!\[[^\]]*\]\([^)]+\)", block)) or block.strip().startswith(
            ("<details", "<summary")
        )
    if current:
        chunks.append("\n".join(current))
    return chunks


def figure_records(body, source):
    result = {}
    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", body):
        path = (source.parent / reference).resolve()
        result[path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
    return result


def display(state, index):
    if index >= len(state["units"]):
        print(
            json.dumps(
                {
                    "complete": True,
                    "lesson_id": state["lesson_id"],
                    "source": state["source"],
                    "source_sha256": state["source_sha256"],
                    "figure_sha256": state["figure_sha256"],
                    "intro_sha256": state.get("intro_sha256"),
                    "trace_file": state["trace_file"],
                },
                ensure_ascii=False,
            )
        )
        return
    unit = state["units"][index]
    print(
        json.dumps(
            {
                "session": state["session"],
                "lesson_id": state["lesson_id"],
                "unit_index": index,
                "units_total": len(state["units"]),
                "unit_sha256": sha(unit.encode("utf-8")),
            },
            ensure_ascii=False,
        )
    )
    print(unit)
    local = ROOT / state["source"].split("#")[0]
    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", unit):
        print("VIEW CURRENT FIGURE:", (local.parent / reference).resolve())
    print("Save all five checkpoint fields before requesting another unit:", ", ".join(FIELDS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("lesson_id")
    start.add_argument("--reviewer", required=True)
    advance = commands.add_parser("next")
    advance.add_argument("session")
    advance.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "start":
        if not args.reviewer.startswith("/root/"):
            raise ValueError("Use the actual reviewer task identity")
        source, body, intro = inventory()[args.lesson_id]
        session = uuid.uuid4().hex
        folder = RUNS / session
        folder.mkdir(parents=True)
        trace = (
            ROOT
            / "docs/reader-reviews/traces"
            / args.lesson_id
            / (args.reviewer.rsplit("/", 1)[-1] + "-" + session[:8] + ".jsonl")
        )
        trace.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "session": session,
            "lesson_id": args.lesson_id,
            "source": source.relative_to(ROOT).as_posix() + "#" + args.lesson_id,
            "source_sha256": sha(body.encode("utf-8")),
            "figure_sha256": figure_records(body, source),
            "reviewer_task": args.reviewer,
            "trace_file": trace.relative_to(ROOT).as_posix(),
            "units": units(intro + body),
            "next_index": 0,
        }
        if intro:
            state["intro_sha256"] = sha(intro.encode("utf-8"))
        (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        trace.write_text(
            json.dumps({k: v for k, v in state.items() if k not in {"units", "next_index"}}, ensure_ascii=False) + "\n"
        )
        display(state, 0)
        return
    if not re.fullmatch(r"[a-f0-9]{32}", args.session):
        raise ValueError("Unknown session")
    path = RUNS / args.session / "state.json"
    state = json.loads(path.read_text())
    index = state["next_index"]
    if index >= len(state["units"]):
        raise ValueError("Reading is already complete")
    note = json.loads(args.checkpoint.read_text())
    if note.get("unit_index") != index:
        raise ValueError("Checkpoint must describe the current unit")
    for field in FIELDS:
        if not isinstance(note.get(field), str) or not note[field].strip():
            raise ValueError("Missing actual reader note: " + field)
    note["unit_sha256"] = sha(state["units"][index].encode("utf-8"))
    with (ROOT / state["trace_file"]).open("a") as stream:
        stream.write(json.dumps(note, ensure_ascii=False) + "\n")
    state["next_index"] += 1
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    display(state, state["next_index"])


if __name__ == "__main__":
    main()
