"""新增來源版本；不覆寫先前凍結檔，也不產生審閱判定。"""

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("凍結版本不可覆寫")
    previous = json.loads(args.previous.read_bytes())
    prior = {p["page_id"]: p for p in previous["pages"]}
    spec = importlib.util.spec_from_file_location("freeze_inventory", ROOT / "docs/review-tools/composite_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = module.current_inventory(ROOT)
    from scripts import export_course, reading_time

    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    folder = args.output.with_suffix("")
    for page in inventory["pages"]:
        pid = page["page_id"]
        old = prior.get(pid)
        if old and old["source_sha256"] == page["source_sha256"]:
            page["snapshot"] = old["snapshot"]
            continue
        text = (ROOT / page["source"]).read_text()
        if page["selector"].startswith("section:"):
            text = reading_time.lesson_slices(text)[pid]
        elif page["selector"] == "before-first-section":
            text = reading_time.chapter_introduction(text)
        elif page["kind"] == "home":
            text = export_course.home_introduction(index, True)
        require_digest = sha(text.encode())
        if require_digest != page["source_sha256"]:
            raise ValueError("來源選取與inventory不同")
        target = folder / "sources" / (pid + ".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(text)
        page["snapshot"] = target.resolve().relative_to(ROOT).as_posix()
    inventory.update(
        schema_version=2,
        review_policy="tutorial_incremental_v2",
        created_at=datetime.now(UTC).isoformat(),
        previous={"path": args.previous.as_posix(), "sha256": sha(args.previous.read_bytes())},
    )
    baseline = json.loads((ROOT / "docs/course-publish-20261008/current-source-freeze.json").read_bytes())
    figure_copies = {ref["sha256"]: ref["path"] for ref in baseline["figures"].values()}
    storage = ROOT / "docs/course-revision-20261016/frozen-figures"
    storage.mkdir(exist_ok=True)
    for target in storage.glob("*.svg"):
        figure_copies[sha(target.read_bytes())] = target.relative_to(ROOT).as_posix()
    figures = {name: digest for page in inventory["pages"] for name, digest in page["figures_sha256"].items()}
    inventory["figure_snapshots"] = {}
    for name, digest in figures.items():
        raw = (ROOT / name).read_bytes()
        if sha(raw) != digest:
            raise ValueError("圖在凍結過程中改變")
        if digest not in figure_copies:
            target = storage / (digest + ".svg")
            with target.open("xb") as stream:
                stream.write(raw)
            figure_copies[digest] = target.relative_to(ROOT).as_posix()
        inventory["figure_snapshots"][name] = {"path": figure_copies[digest], "sha256": digest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pages": len(inventory["pages"]), "snapshot": str(args.output)}))


if __name__ == "__main__":
    main()
