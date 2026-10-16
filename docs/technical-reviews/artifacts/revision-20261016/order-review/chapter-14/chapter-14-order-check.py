"""Chapter-14 owner callback: byte identity and local navigation only."""

import difflib
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path.cwd().resolve()
OUT = Path(__file__).with_name("chapter-14-order-check-output.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    reading = load_module("order_reading_time", "scripts/reading_time.py")
    exporter = load_module("order_export_course", "scripts/export_course.py")
    baseline_path = ROOT / "docs/course-revision-20261016/freeze-16.json"
    current_path = ROOT / "docs/course-revision-20261016/freeze-18.json"
    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())
    old_pages = {p["page_id"]: p for p in baseline["pages"]}
    pages = {p["page_id"]: p for p in current["pages"]}
    source = ROOT / "course/chapters/14.md"
    content = source.read_text(encoding="utf-8")
    body = reading.chapter_introduction(content)
    slices = reading.lesson_slices(content)
    ids = list(slices)
    expected = [f"14.{number}" for number in range(1, 19)]
    failures = []

    def check(ok, label):
        if not ok:
            failures.append(label)
        return ok

    snapshot = ROOT / pages["chapter-14"]["snapshot"]
    original_snapshot = ROOT / old_pages["chapter-14"]["snapshot"]
    intro_checks = {
        "current_body_matches_frozen_bytes": check(body.encode() == snapshot.read_bytes(), "intro frozen bytes"),
        "body_matches_freeze18_sha256": check(reading.source_sha256(body) == pages["chapter-14"]["source_sha256"], "intro SHA"),
        "no_fenced_code": check("```" not in body and "~~~" not in body, "unexpected fenced code"),
        "no_figure_references": check(reading.figure_hashes(ROOT, source, body) == {}, "unexpected intro figure"),
        "ascending_source_order": check(ids == expected, "source order"),
    }

    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    indexed = [row for row in index if row["source"] == "course/chapters/14.md"]
    indexed_ids = [row["id"] for row in indexed]
    check(indexed_ids == expected, "lesson index order")
    settings = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    title = content.splitlines()[0].removeprefix("# ")

    def find_nav(items):
        for item in items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key == title:
                    return value
                if isinstance(value, list):
                    match = find_nav(value)
                    if match is not None:
                        return match
        return None

    nav = find_nav(settings["nav"])
    nav_paths = [next(iter(row.values())) for row in nav] if nav else []
    check(nav_paths == ["chapter-14.md"] + [item + ".md" for item in expected], "Zensical chapter nav order")

    targets = {(ROOT / row["source"]).resolve(): "chapter-" + Path(row["source"]).stem + ".md" for row in index}
    lesson_targets = {}
    for row in index:
        lesson_targets.setdefault((ROOT / row["source"]).resolve(), {})[row["id"]] = row["id"] + ".md"
    rewritten = exporter.reading_markdown(body, source, targets, lesson_targets, "order-owner-callback")
    direct_links = []
    for label, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
        path_text, _, fragment = link.partition("#")
        target = (source.parent / path_text).resolve()
        target_slices = reading.lesson_slices(target.read_text())
        exists = check(fragment in target_slices, f"intro link {link}")
        mapped = lesson_targets.get(target, {}).get(fragment)
        direct_links.append({"label": label, "source_link": link, "exists": exists, "published_link": mapped,
                             "rewrite_verified": check(f"]({mapped})" in rewritten, f"rewritten link {link}")})

    numbered = []
    for identifier, text in slices.items():
        old = old_pages[identifier]
        page = pages[identifier]
        sha = reading.source_sha256(text)
        figure_checks = {}
        for path, wanted in page["figures_sha256"].items():
            figure_checks[path] = check(digest(ROOT / path) == wanted == old["figures_sha256"][path], f"figure {path}")
        numbered.append({
            "id": identifier,
            "source_sha256": sha,
            "same_as_freeze16_and_freeze18": check(sha == old["source_sha256"] == page["source_sha256"], f"slice {identifier}"),
            "snapshot_bytes_match": check(text.encode() == (ROOT / page["snapshot"]).read_bytes(), f"snapshot {identifier}"),
            "notebook_sha256": digest(ROOT / page["notebook"]),
            "notebook_unchanged": check(digest(ROOT / page["notebook"]) == page["notebook_sha256"] == old["notebook_sha256"], f"notebook {identifier}"),
            "figure_byte_checks_only": figure_checks,
        })

    intra_chapter = []
    for identifier, text in slices.items():
        for fragment in re.findall(r"\]\(14\.md#([^)]*)\)", text):
            exists = check(fragment in slices, f"chapter link {identifier} -> {fragment}")
            intra_chapter.append({"from": identifier, "to": fragment, "exists": exists,
                                  "target_is_earlier": exists and ids.index(fragment) < ids.index(identifier)})

    routes = json.loads((ROOT / "course/reading-time-routes.json").read_text())["routes"]
    route_refs = [{"id": route["id"], "chapter14_page_ids": [p for p in route["page_ids"] if p == "chapter-14" or p.startswith("14.")]}
                  for route in routes]
    route_refs = [route for route in route_refs if route["chapter14_page_ids"]]
    for route in route_refs:
        selected = route["chapter14_page_ids"]
        check(all(p == "chapter-14" or p in slices for p in selected), f"route references {route['id']}")

    original_report = ROOT / "docs/course-revision-20261016/reviews/technical-chapter-14-original.json"
    pinned_files = [source, ROOT / "course/chapters/04.md", ROOT / "course/lesson-index.json",
                    ROOT / "course/README.md", ROOT / "course/reading-time-routes.json", ROOT / "zensical.toml",
                    ROOT / "scripts/reading_time.py", ROOT / "scripts/export_course.py", ROOT / "scripts/check_technical_reviews.py",
                    baseline_path, current_path, original_report]
    result = {
        "scope": "Entire current chapter-14 introduction; chapter-local ordering, source slices and link dependencies only. No model execution, formal tests, download or visual review of subsection figures.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU/static text and link checks"},
        "intro_bytes": len(body.encode()), "source_sha256": reading.source_sha256(body),
        "original_report_sha256": digest(original_report),
        "original_report_preserved": check(digest(original_report) == "d7bd164ee9608c8993be9d1e578d6e729ef5e8dec10b86122a658fd1bd37babe", "owner original report changed"),
        "intro_checks": intro_checks,
        "intro_diff_from_original": "".join(difflib.unified_diff(original_snapshot.read_text().splitlines(keepends=True), body.splitlines(keepends=True), fromfile="freeze16/original", tofile="freeze18/current")),
        "direct_intro_links": direct_links,
        "published_intro_markdown": rewritten,
        "current_source_order": ids, "lesson_index_order": indexed_ids, "zensical_chapter_nav": nav_paths,
        "chapter_local_source_dependencies": intra_chapter,
        "reading_route_references": route_refs,
        "numbered_source_identity_checks_only": numbered,
        "current_file_pins": {p.relative_to(ROOT).as_posix(): digest(p) for p in pinned_files},
        "failures": failures,
    }
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": OUT.relative_to(ROOT).as_posix(), "sha256": digest(OUT), "failures": failures,
                      "intro_source_sha256": result["source_sha256"], "numbered_identity_checks": len(numbered)}, ensure_ascii=False))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
