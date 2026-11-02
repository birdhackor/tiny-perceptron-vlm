"""Portable read-only source, link, notebook, and figure inspection receipt."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

from scripts.export_course import checked_notebook

ROOT = Path(__file__).resolve().parents[4]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
body_path = ROOT / "course/first-steps.md"
body = body_path.read_text()
headings = list(re.finditer(r"^## (W\.\d+) (.+)$", body, re.M))
units = {"introduction": body[:headings[0].start()]}
for i, heading in enumerate(headings):
    units[heading[1]] = body[heading.start():headings[i+1].start() if i+1 < len(headings) else len(body)]

index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
freeze = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v4.json").read_bytes())
frozen_notebooks = {x["page_id"]: x for x in freeze["pages"] if "notebook" in x}
execution_rows = []
bundle = ROOT / "outputs/omni-P5-current-kernel-bundle-v6"
for item in index:
    original = ROOT / item["notebook"]
    execution = bundle / Path(item["notebook"]).relative_to("notebooks")
    row = {"id": item["id"], "notebook": item["notebook"], "notebook_sha256": digest(original), "preserved_execution_sha256": digest(execution)}
    restored = json.loads(execution.read_bytes())
    for cell in restored["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    restored_bytes = (json.dumps(restored, ensure_ascii=False, indent=1) + "\n").encode()
    restored_sha = hashlib.sha256(restored_bytes).hexdigest()
    row["restored_unexecuted_notebook_sha256"] = restored_sha
    row["canonical_v4_notebook_sha256"] = frozen_notebooks[item["id"]]["notebook_sha256"]
    row["canonical_v4_exact_match"] = restored_sha == row["canonical_v4_notebook_sha256"]
    assert row["canonical_v4_exact_match"]
    checked_notebook(restored, execution)
    try:
        notebook = checked_notebook(json.loads(original.read_bytes()), execution)
    except ValueError as error:
        row["status"] = "source_mismatch"
        row["error"] = str(error)
        current = json.loads(original.read_bytes())
        preserved = json.loads(execution.read_bytes())
        row["different_cell_indexes"] = [i for i, (a, b) in enumerate(zip(current["cells"], preserved["cells"])) if a["cell_type"] != b["cell_type"] or a["source"] != b["source"]]
        row["different_code_cell_indexes"] = [i for i in row["different_cell_indexes"] if current["cells"][i]["cell_type"] == "code"]
    else:
        row.update(status="source_match", code_cells=sum(c["cell_type"] == "code" for c in notebook["cells"]), output_items=sum(len(c.get("outputs", [])) for c in notebook["cells"]))
    execution_rows.append(row)

sections = {}
for p in (ROOT / "course/chapters").glob("*.md"):
    text = p.read_text()
    hs = list(re.finditer(r"^## ([A-Z0-9]+\.\d+) (.+)$", text, re.M))
    for i, h in enumerate(hs):
        sections[h[1]] = (p, text[h.start():hs[i+1].start() if i+1 < len(hs) else len(text)])
navigation_rows = []
for line in units["W.7"].splitlines():
    if not line.startswith("| `tiny_perceptron/") or "chapters/" not in line:
        continue
    name = re.search(r"`([^`]+)`", line)[1]
    suffix = name.removeprefix("tiny_perceptron/").removesuffix(".py").rstrip("/")
    references = []
    for match in re.finditer(r"\[([^]]+)\]\(([^)]+)#([A-Z0-9]+\.\d+)\)(?:–([A-Z0-9]+\.\d+))?", line):
        start, end = match[3], match[4]
        ids = [start]
        if end:
            ids = [start.split(".")[0] + "." + str(j) for j in range(int(start.split(".")[1]), int(end.split(".")[1]) + 1)]
        for ident in ids:
            p, section = sections[ident]
            hits = [x for x in section.splitlines() if name in x or ("tiny_perceptron." + suffix) in x or (suffix + ".py") in x]
            references.append({"section": ident, "source": p.relative_to(ROOT).as_posix(), "section_sha256": hashlib.sha256(section.encode()).hexdigest(), "matching_line_count": len(hits), "short_support": next((x[:420] for x in hits if not x.startswith("from ")), hits[0][:420] if hits else "NO EXPLICIT FILE REFERENCE")})
    navigation_rows.append({"module": name, "exists": (ROOT / name).exists(), "references": references})
assert all(r["exists"] for r in navigation_rows)
assert all(x["matching_line_count"] > 0 for r in navigation_rows for x in r["references"])

links = []
for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", body):
    if "://" in target:
        continue
    rel, _, fragment = target.partition("#")
    p = (body_path.parent / rel).resolve() if rel else body_path
    valid = p.is_file()
    if valid and fragment:
        valid = bool(re.search(r"^## " + re.escape(fragment) + r"(?: |$)", p.read_text(), re.M))
    links.append({"target": target, "exists": valid})
assert all(x["exists"] for x in links)

svg_path = ROOT / "course/figures/rewrite-W-6-gradient-step.svg"
svg = ET.fromstring(svg_path.read_bytes())
circles = [{"x": float(x.attrib["cx"]), "y": float(x.attrib["cy"])} for x in svg if x.tag.endswith("circle")]
expected = [(1., 4.), (1.4, 2.56), (3., 0.)]
for point, (w, loss) in zip(circles, expected, strict=True):
    assert abs(point["x"] - (75 + 123 * w)) < 1e-9
    assert abs(point["y"] - (430 - 34 * loss)) < 1e-9
curve = next(x.attrib["d"] for x in svg if x.tag.endswith("path") and x.attrib.get("stroke-width") == "3" and x.attrib.get("stroke") == "#64748b")
curve_points = [tuple(map(float, p)) for p in re.findall(r"[ML]([\d.]+) ([\d.]+)", curve)]
curve_residual = max(abs(y - (430 - 34 * (((x - 75) / 123) - 3)**2)) for x, y in curve_points)
assert curve_residual < .03
print(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(body_path), "coverage": {k: {"utf8_bytes": len(v.encode()), "sha256": hashlib.sha256(v.encode()).hexdigest()} for k, v in units.items()}, "repository_paths": {k: {"exists_or_ignored_output_directory": (ROOT / k).exists(), "gitignore_rule": rule} for k, rule in [("course/chapters", "tracked source"), ("notebooks", "tracked source"), ("tiny_perceptron", "tracked source"), ("scripts", "tracked source"), ("data", "/data/"), ("checkpoints", "/checkpoints/"), ("outputs", "/outputs/"), (".venv", ".venv/")]}, "w7_navigation_rows": navigation_rows, "local_links": links, "gradient_figure_geometry": {"mapping": "x=75+123*w; y=430-34*L", "circles": circles, "expected_w_loss": expected, "curve_points": len(curve_points), "max_pixel_residual": curve_residual, "note": "The orange connector is an update direction between finite points, not a tangent; the SVG description states this explicitly."}, "preserved_notebook_output_inspection": {"count": len(execution_rows), "canonical_v4_all_exact_and_nonerror_executed_cells": all(x["canonical_v4_exact_match"] for x in execution_rows), "live_all_exact_source_and_nonerror_executed_cells": all(x["status"] == "source_match" for x in execution_rows), "restoration_method": "Clear only code execution_count and outputs; serialize with ensure_ascii=False, indent=1, trailing newline, matching scripts/build_course.py:194. Exact SHA equality then authenticates all other notebook bytes against the immutable canonical v4 freeze. checked_notebook checks all restored source cells and preserved output errors/counts.", "rows": execution_rows, "scope": "Read preserved author CPU output cells with existing checked_notebook; no formal neural inference was repeated. This validates byte/source association and recorded output presence, not independent re-execution or every output's semantics. Four live notebook versions changed after canonical v4; their differences are retained and do not invalidate the exact frozen v4 association."}, "limits": "Only the W.7-linked sections were inspected for file-to-lesson navigation; this is not a technical review of those other lessons. No author dispositions, other technical reports, reader answers, or section sidecar read."}, ensure_ascii=False, indent=2))
