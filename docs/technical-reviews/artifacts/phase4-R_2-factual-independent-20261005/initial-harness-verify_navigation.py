"""R.2 independent, bounded source/link contract check; no build or model execution."""
import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
FROZEN = ART / "frozen-input"
TASK = "/root/phase4_factual_coordinator/factual_r_2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(relative):
    original = ROOT / relative
    raw = original.read_bytes()
    target = FROZEN / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == raw, f"frozen input changed: {relative}"
    else:
        target.write_bytes(raw)
    assert sha(target.read_bytes()) == sha(original.read_bytes())
    return {"original_path": relative, "snapshot": str(target.relative_to(ROOT)), "sha256": sha(raw), "bytes": len(raw)}


def sections(path):
    raw = path.read_bytes()
    raw.decode("utf-8")
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    result = {}
    for i, h in enumerate(headings):
        match = re.match(rb"^## ([A-Z\d]+\.\d+) (.+)$", h[0])
        assert match, (path, h[0])
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        result[match[1].decode()] = {"title": match[2].decode(), "body": raw[h.start():end], "line": raw[:h.start()].count(b"\n") + 1}
    return result


def selected_code(relative, names, constants=()):
    raw = (ROOT / relative).read_bytes()
    tree = ast.parse(raw)
    selected = []
    locators = []
    for node in tree.body:
        include = isinstance(node, ast.FunctionDef) and node.name in names
        if isinstance(node, ast.Assign):
            include |= any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        if include:
            selected.append(node)
            locators.append({"name": getattr(node, "name", next((t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)), "")), "lines": [node.lineno, node.end_lineno]})
    namespace = {"re": re, "Path": Path, "ROOT": ROOT}
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), namespace)
    return namespace, {"path": relative, "sha256": sha(raw), "executed_ast_nodes": locators}


def main():
    source = ROOT / "course/README.md"
    current = sections(source)["R.2"]
    raw = current["body"]
    artifact_section = ART / "section.md"
    artifact_section.write_bytes(raw)
    assert artifact_section.read_bytes() == raw
    assert raw == (ROOT / "outputs/reviewer-tools/runs/phase4-r2-factual-independent-20261005/section.md").read_bytes()
    body = raw.decode("utf-8")
    assert "```" not in body and not re.search(r"!\[|<(?:img|svg)\b", body)
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body)
    assert len(links) == 20
    contract = json.loads((ROOT / "course/lesson-contract.json").read_bytes())
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    assert contract["schema_version"] == 1
    ids = contract["lesson_ids"]
    assert len(ids) == len(set(ids))
    indexed = {row["id"]: row for row in index}
    assert set(ids) == set(indexed)
    assert len(indexed) == len(index)
    inventory = {}
    for path in sorted((ROOT / "course/chapters").glob("*.md")):
        for lesson, record in sections(path).items():
            assert lesson not in inventory
            row = indexed[lesson]
            assert row["source"] == str(path.relative_to(ROOT))
            assert row["title"] == record["title"]
            assert (ROOT / row["notebook"]).is_file()
            inventory[lesson] = {"source": str(path.relative_to(ROOT)), "title": record["title"], "line": record["line"], "notebook": row["notebook"]}
    assert set(inventory) == set(ids)
    bc, bc_execution = selected_code("scripts/build_course.py", ("lesson_anchor_aliases", "notebook_reading_links"), ("COURSE_URL", "READING_PAGES"))
    ex, ex_execution = selected_code("scripts/export_course.py", ("reading_markdown",))
    ex["lesson_anchor_aliases"] = bc["lesson_anchor_aliases"]
    targets = {}
    lesson_targets = {}
    for row in index:
        path = (ROOT / row["source"]).resolve()
        targets[path] = "chapter-" + path.stem + ".md"
        lesson_targets.setdefault(path, {})[row["id"]] = row["id"] + ".md"
        targets[(ROOT / row["notebook"]).resolve()] = row["id"] + ".md"
    transformed = ex["reading_markdown"](body, source, targets, lesson_targets, "main")
    transformed_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", transformed)
    assert len(transformed_links) == len(links)
    result_links = []
    for position, ((label, target), (out_label, out_target)) in enumerate(zip(links, transformed_links, strict=True)):
        path_text, _, fragment = target.partition("#")
        path = (source.parent / path_text).resolve()
        assert path.is_file() and path.parent == ROOT / "course/chapters"
        if fragment:
            assert fragment in lesson_targets[path]
            expected_page = fragment + ".md"
            record = inventory[fragment]
            notebook = json.loads((ROOT / record["notebook"]).read_bytes())
            assert notebook["metadata"]["lesson_id"] == fragment
            assert notebook["cells"][0]["cell_type"] == "markdown"
            assert "".join(notebook["cells"][0]["source"]).strip() == "# " + fragment + " " + record["title"]
            expected_notebook_link = bc["COURSE_URL"] + fragment + ".html"
        else:
            expected_page = "chapter-" + path.stem + ".md"
            record = {"chapter_heading": path.read_text(encoding="utf-8").splitlines()[0], "section_ids": [i for i,r in inventory.items() if r["source"] == str(path.relative_to(ROOT))]}
            assert record["section_ids"]
            expected_notebook_link = bc["COURSE_URL"] + "chapter-" + path.stem + ".html"
        assert out_label == label and out_target == expected_page
        converted = bc["notebook_reading_links"]("[" + label + "](" + target + ")", source)
        assert converted == "[" + label + "](" + expected_notebook_link + ")"
        result_links.append({"index": position, "label": label, "source_target": target, "reading_target": out_target, "notebook_link_conversion": expected_notebook_link, "target": record})
    # Small contract variants: numbered fragment, preserved full heading slug,
    # chapter-only link, current-document numbered anchor, and external URL.
    seven = ROOT / "course/chapters/07.md"
    aliases = bc["lesson_anchor_aliases"](seven)
    full_slug = next(k for k,v in aliases.items() if v == "7.11")
    cases = [("[x](chapters/07.md#7.11)", "[x](7.11.md)"), ("[x](chapters/07.md#" + full_slug + ")", "[x](7.11.md)"), ("[x](chapters/07.md)", "[x](chapter-07.md)"), ("[x](#R.2)", "[x](#R.2)"), ("[x](https://example.com/a#b)", "[x](https://example.com/a#b)")]
    for input_text, expected in cases:
        observed = ex["reading_markdown"](input_text, source, targets, lesson_targets, "main").strip()
        assert observed == expected, (input_text, observed, expected)
    snapshots = [freeze(p) for p in ("course/README.md", "course/lesson-contract.json", "course/lesson-index.json", "scripts/build_course.py", "scripts/export_course.py", "docs/course-revision-20261005/outline.md", "docs/course-revision-20261005/rewrite-contract.md", "docs/review-tools/factual-reviewer-instructions.md", "docs/review-tools/section_facts.py", "scripts/check_technical_reviews.py", ".agents/skills/clear-tutorial/SKILL.md", ".agents/skills/clear-tutorial/references/review-protocol.md")]
    relevant = sorted({str((source.parent / t.split("#")[0]).resolve().relative_to(ROOT)) for _,t in links} | {"course/chapters/02.md", "course/chapters/03.md", "course/chapters/04.md", "course/chapters/13.md"})
    snapshots.extend(freeze(p) for p in relevant)
    inspected_notebooks = sorted({inventory[t.split("#")[1]]["notebook"] for _,t in links if "#" in t})
    snapshots.extend(freeze(p) for p in inspected_notebooks)
    evidence = {
        "reviewer_task": TASK,
        "source": "course/README.md#R.2",
        "source_sha256": sha(raw),
        "section_first_line": current["line"],
        "newline_policy": "original UTF-8 bytes, no stripping or normalization",
        "whole_markdown_input": {"meaning": "frozen whole-file input at this check, not a continuing claim about the current complete guide", "path": str((FROZEN / "course/README.md").relative_to(ROOT)), "sha256": sha(source.read_bytes())},
        "environment": {"python": sys.version, "python_executable": sys.executable, "device": "CPU; no tensor/model computation", "platform": platform.platform()},
        "checked_pointers": {"course/lesson-contract.json": ["/schema_version", "/lesson_ids"], "course/lesson-index.json": ["/{row}/id", "/{row}/title", "/{row}/source", "/{row}/notebook"], "linked_section_notebooks": ["/metadata/lesson_id", "/cells/0/cell_type", "/cells/0/source"]},
        "original_code_execution": [bc_execution, ex_execution],
        "navigation_links": result_links,
        "bounded_conversion_cases": [{"input": inp, "expected": exp, "observed": exp} for inp,exp in cases],
        "coverage": {"r2_link_count": len(links), "section_link_count": sum('#' in t for _,t in links), "chapter_link_count": sum('#' not in t for _,t in links), "row_count": 10, "inventory_count": len(inventory), "images": 0, "code_fences": 0},
        "snapshots": snapshots,
        "result": "All 20 R.2 source links, 10 question rows, target notebooks, and bounded link-conversion cases matched the original code contract. No build, training, data/model download, inference, or published-site access was performed.",
    }
    (ART / "navigation-check.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ART / "r2-rendered-markdown.md").write_text(transformed, encoding="utf-8")
    print(json.dumps({"source_sha256": sha(raw), "coverage": evidence["coverage"], "result": evidence["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
