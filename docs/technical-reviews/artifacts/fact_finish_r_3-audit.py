"""Fresh R.3 inventory replay; filesystem counts and one tiny CPU alignment check."""

import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_r_3"
FRONT = ("first-steps.md", "README.md", "training.md", "glossary.md")
NUMBERED = re.compile(r"^## ([A-Z\d]+\.\d+) (.+)$", re.M)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def frozen_snapshot(path, label):
    raw = path.read_bytes()
    snapshot = ARTIFACTS / f"{PREFIX}-{label}"
    if snapshot.exists():
        assert snapshot.read_bytes() == raw, f"Source changed after reading: {path}"
    else:
        snapshot.write_bytes(raw)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest(raw),
        "snapshot": snapshot.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
    }


def main():
    import torch

    from scripts.check_technical_reviews import sections
    from tiny_perceptron.data import ByteTokenizer, render_chat

    start = time.perf_counter()
    torch.set_num_threads(1)
    body = dict(sections(ROOT / "course/README.md"))["R.3"]
    raw_body = body.encode("utf-8")
    saved = ARTIFACTS / f"{PREFIX}-section.raw.txt"
    assert saved.read_bytes() == raw_body, "R.3 changed after the personal reading"
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body)
    chapter_rows = []
    for line in body.splitlines():
        match = re.fullmatch(r"\| \[([^\]]+)\]\((chapters/[^)]+\.md)\) \| (.+) \|", line)
        if match:
            chapter_rows.append({"label": match[1], "target": f"course/{match[2]}", "question": match[3]})
    sources = []
    chapter_files = sorted((ROOT / "course/chapters").glob("*.md"))
    inventory = []
    chapters = []
    for path in chapter_files:
        text = path.read_text(encoding="utf-8")
        matches = list(NUMBERED.finditer(text))
        assert matches, path
        assert len(matches) == len(re.findall(r"^## ", text, re.M)), path
        headings = [{"id": item[1], "title": item[2]} for item in matches]
        chapters.append({"source": path.relative_to(ROOT).as_posix(), "sections": len(headings), "headings": headings})
        sources.append(frozen_snapshot(path, f"chapter-{path.stem}.source.md"))
        inventory.extend({**heading, "source": path.relative_to(ROOT).as_posix()} for heading in headings)
    front = []
    for name in FRONT:
        path = ROOT / "course" / name
        matches = list(NUMBERED.finditer(path.read_text(encoding="utf-8")))
        headings = [{"id": item[1], "title": item[2]} for item in matches]
        front.append({"source": path.relative_to(ROOT).as_posix(), "sections": len(headings), "headings": headings})
        sources.append(frozen_snapshot(path, f"front-{name}"))
    index_path = ROOT / "course/lesson-index.json"
    lessons_path = ROOT / "course/lessons.md"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sources.append(frozen_snapshot(index_path, "lesson-index.snapshot.json"))
    sources.append(frozen_snapshot(lessons_path, "lessons.snapshot.md"))
    by_id = {item["id"]: item for item in inventory}
    assert len(by_id) == len(inventory), "Duplicate chapter section IDs"
    index_ids = [item["id"] for item in index]
    assert len(set(index_ids)) == len(index_ids)
    assert set(index_ids) == set(by_id)
    notebook_records = []
    for item in index:
        raw = (ROOT / item["notebook"]).read_bytes()
        notebook = json.loads(raw)
        origin = by_id[item["id"]]
        assert item["source"] == origin["source"]
        assert item["title"] == origin["title"]
        assert notebook["metadata"]["lesson_id"] == item["id"]
        assert notebook["nbformat"] == 4
        title = "".join(notebook["cells"][0]["source"]).strip()
        assert title == f"# {item['id']} {item['title']}"
        notebook_records.append(
            {
                "id": item["id"],
                "path": item["notebook"],
                "sha256": digest(raw),
                "cells": len(notebook["cells"]),
                "code_cells": sum(cell["cell_type"] == "code" for cell in notebook["cells"]),
                "metadata_and_title_match": True,
            }
        )
    actual_notebooks = {path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").rglob("*.ipynb")}
    indexed_notebooks = {item["notebook"] for item in index}
    assert len(indexed_notebooks) == len(index)
    assert actual_notebooks == indexed_notebooks
    expected_rows = [
        f"| {item['id']} {item['title']} | [閱讀](../{item['source']}#{item['id']}) | [開啟](../{item['notebook']}) |"
        for item in index
    ]
    actual_rows = [line for line in lessons_path.read_text().splitlines() if "[閱讀]" in line]
    assert actual_rows == expected_rows
    targets = {item["target"] for item in chapter_rows}
    assert len(targets) == len(chapter_rows)
    assert targets == {item["source"] for item in chapters}
    resolved_links = []
    for label, target in links:
        path_text, _, fragment = target.partition("#")
        path = (ROOT / "course" / path_text).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), target
        if fragment:
            assert fragment in dict(sections(path)), target
        resolved_links.append({"label": label, "target": target, "exists": True})
    declared = {
        "chapter_sections": int(re.search(r"收錄(\d+)節", body)[1]),
        "front_sections": int(re.search(r"的(\d+)節，共", body)[1]),
        "total_sections": int(re.search(r"共(\d+)個編號小節", body)[1]),
    }
    observed = {
        "chapter_files": len(chapter_files),
        "chapter_rows": len(chapter_rows),
        "chapter_sections": len(inventory),
        "index_entries": len(index),
        "lesson_rows": len(actual_rows),
        "notebooks": len(actual_notebooks),
        "front_sections": sum(item["sections"] for item in front),
        "total_sections": len(inventory) + sum(item["sections"] for item in front),
    }
    for key, value in declared.items():
        assert observed[key] == value, (key, value, observed[key])
    all_ids = list(by_id) + [heading["id"] for item in front for heading in item["headings"]]
    assert len(set(all_ids)) == len(all_ids)
    first_id = next(sections(ROOT / "course/README.md"))[0]
    question = next(item for item in index if item["title"] == "回答第一個 token 在哪個位置被預測？")
    assert question["id"] == "7.4"
    tokenizer = ByteTokenizer()
    alignment = []
    for prompt, answer in [("Q", "A"), ("QQ", "A"), ("QQ", "B")]:
        x, y = render_chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
        first = int((y != -100).nonzero()[0].item())
        assert x.device.type == y.device.type == "cpu"
        assert x[first].item() == tokenizer.assistant_id
        assert y[first].item() == tokenizer.encode(answer)[0]
        alignment.append({"prompt": prompt, "answer": answer, "x": x.tolist(), "y": y.tolist(), "first": first})
    for name in [
        "scripts/check_technical_reviews.py",
        "scripts/build_course.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/attention.py",
        "tiny_perceptron/tokenization.py",
    ]:
        sources.append(frozen_snapshot(ROOT / name, name.replace("/", "-") + ".snapshot.txt"))
    result = {
        "reviewer_task": "/root/fact_finish_r_3",
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_r_3-audit.py",
        "executed_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "CPU",
            "threads": str(torch.get_num_threads()),
            "platform": platform.platform(),
        },
        "source_sha256": digest(saved.read_bytes()),
        "is_first_numbered_section": first_id == "R.3",
        "first_numbered_section": first_id,
        "figure_references_in_r3": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
        "declared": declared,
        "observed": observed,
        "front_counts": {item["source"]: item["sections"] for item in front},
        "chapter_counts": {item["source"]: item["sections"] for item in chapters},
        "chapters": chapters,
        "front_inventory": front,
        "chapter_navigation_rows": chapter_rows,
        "resolved_links": resolved_links,
        "notebook_records": notebook_records,
        "source_snapshots": sources,
        "question_location": question,
        "cpu_alignment_supporting_navigation_example": alignment,
        "scope": "Inventory, link and metadata identity, plus the 7.4 tiny CPU data example. No notebook suite, training, quality or GPU acceleration was measured.",
        "elapsed_seconds": time.perf_counter() - start,
        "result": "All independently derived inventory, identity and CPU alignment assertions passed.",
    }
    output = ARTIFACTS / f"{PREFIX}-audit-output.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["observed", "front_counts", "chapter_counts", "environment", "elapsed_seconds", "result"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert Counter(item["id"] for item in index).most_common(1)[0][1] == 1


if __name__ == "__main__":
    main()
