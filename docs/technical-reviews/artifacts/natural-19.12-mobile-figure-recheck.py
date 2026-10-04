"""Original owner CPU recheck after mobile re-layout of two prerequisite SVGs."""
from pathlib import Path
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
from PIL import Image
from tiny_perceptron.natural_concepts import picture_order_report, thin_stroke_report, swapped_picture

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("docs/technical-reviews/artifacts")
PRIOR_SHA = "fe8c1ebca38a7079b5030f57b8c2df45f3437499563e69bd534ea6fa96bb79be"
HISTORY = OUT / "natural-19.12-history" / PRIOR_SHA
OLD_FIG_HISTORY = OUT / "natural-19.12-history/4bd8f69add2ff4ddc73e1966716df6dc9d9f50993cd2f588340fd95ce1890890/registered-files"
NEW_FIGURES = {
    "course/figures/practical_order.svg": "6d4a0e684aedb9196b1c61bc2fd5a78fb3ce11663d2875e5d3b04532030ed523",
    "course/figures/practical_stroke.svg": "3e4a4ddedd9558254a44c3a87f09a45f0f6abba45d565f6f67a910869a75901f",
}


def read(path):
    return (ROOT / path).read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == data, str(path)
    else:
        target.write_bytes(data)


def section(path, lesson):
    data = read(path)
    start = re.search(rb"(?m)^## " + re.escape(lesson.encode()) + rb" [^\n]*\n", data).start()
    following = re.search(rb"(?m)^## ", data[start + 1:])
    return data[start:start + 1 + following.start()] if following else data[start:]


prior_bytes = read(HISTORY / "report.json") if (ROOT / HISTORY / "report.json").exists() else read("docs/technical-reviews/19.12.json")
assert sha(prior_bytes) == PRIOR_SHA
prior = json.loads(prior_bytes)
assert prior["reviewer_task"] == "/root/natural_factual_19_12" and prior["verdict"] == "pass"
assert len(prior["claims"]) == 86 and all(c["status"] == "verified" for c in prior["claims"])
assert sha(section("course/chapters/19.md", "19.12")) == prior["source_sha256"]
save(HISTORY / "report.json", prior_bytes)
entries = {}
for item in prior["sources"] + prior["artifacts"]:
    if "path" not in item:
        continue
    data = read(item["path"])
    assert sha(data) == item["sha256"], item["path"]
    archived = HISTORY / "registered-files" / item["path"]
    save(archived, data)
    entry = entries.setdefault(item["path"], {"path": item["path"], "sha256": sha(data), "archive_path": str(archived), "registrations": []})
    entry["registrations"].append(item["id"])
figure_changes = []
for path, old_sha in prior["figure_sha256"].items():
    old = read(OLD_FIG_HISTORY / path) if path in NEW_FIGURES else read(path)
    assert sha(old) == old_sha, path
    archived = HISTORY / "registered-files" / path
    save(archived, old)
    entries[path] = {"path": path, "sha256": sha(old), "archive_path": str(archived), "registrations": ["figure_sha256"],
        "recovered_from": str(OLD_FIG_HISTORY / path) if path in NEW_FIGURES else "current exact matching unchanged bytes"}
    if path in NEW_FIGURES:
        new = read(path)
        assert sha(new) == NEW_FIGURES[path] and new != old
        name = Path(path).stem
        save(OUT / f"natural-19.12-mobile-{name}-current.svg", new)
        diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True), fromfile=f"true archived old {path}", tofile=f"current {path}"))
        save(OUT / f"natural-19.12-mobile-{name}-source-diff.patch", diff.encode())
        figure_changes.append({"path": path, "old_sha256": old_sha, "new_sha256": sha(new)})
save(HISTORY / "archive-manifest.json", (json.dumps({"reviewer_task": prior["reviewer_task"], "prior_report_sha256": PRIOR_SHA,
    "files": list(entries.values()), "old_figure_recovery": "True previous own archived SVG bytes matching exact prior report hashes", "initial_b821_full_source_still_unavailable": True}, ensure_ascii=False, indent=2) + "\n").encode())

lesson_outputs = {}
section_hashes = {}
for path, ids in [("course/chapters/11.md", ["11.14", "11.15"]), ("course/chapters/20.md", ["20.1", "20.2"])]:
    for lesson in ids:
        data = section(path, lesson)
        section_hashes[lesson] = sha(data)
        save(OUT / f"natural-19.12-mobile-section-{lesson}.md", data)
        python_blocks = re.findall(rb"(?ms)^```python\n(.*?)^```[ \t]*$", data)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for block in python_blocks:
                exec(compile(block.decode(), f"{path}#{lesson}", "exec"), {})
        lesson_outputs[lesson] = output.getvalue()
order = picture_order_report()
stroke = thin_stroke_report()
assert order == {"different_pixels": 128, "same_4_by_4_summary": True, "same_pixel_sequence": False}
assert stroke == {"original_shape": [32, 32], "small_shape": [4, 4], "original_brightest": 1.0, "small_brightest": 0.125,
    "pixels_at_least_half_bright_before": 32, "pixels_at_least_half_bright_after": 0}
first, second = swapped_picture()
assert str(first.device) == "cpu" and int((first != second).any(dim=0).sum()) == 64
moved = first.clone()
moved[0, 8:16, :4] = 0
moved[0, 8:16, 8:12] = 1
pooled_first, pooled_moved = [F.adaptive_avg_pool2d(t[None], (4, 4)) for t in (first, moved)]
assert not torch.equal(pooled_first, pooled_moved)
stroke_a = torch.zeros(1, 1, 32, 32); stroke_a[:, :, :, 15] = 1
stroke_b = torch.zeros_like(stroke_a); stroke_b[:, :, :, 8] = 1
assert torch.equal(F.interpolate(stroke_a, (4, 4), mode="area"), F.interpolate(stroke_b, (4, 4), mode="area"))

# Numeric/geometry consistency supplements the actual visual inspection.
ns = {"s": "http://www.w3.org/2000/svg"}
order_xml = ET.fromstring(read("course/figures/practical_order.svg"))
bars = [r.attrib for r in order_xml.findall(".//s:rect", ns) if r.attrib.get("fill") in {"#dc2626", "#2563eb"}]
assert [(b["x"], b["y"], b["width"], b["height"], b["fill"]) for b in bars] == [
    ("84", "258", "64", "98", "#dc2626"), ("148", "258", "64", "98", "#2563eb"),
    ("84", "538", "64", "98", "#2563eb"), ("148", "538", "64", "98", "#dc2626")]
stroke_xml = ET.fromstring(read("course/figures/practical_stroke.svg"))
stroke_rects = stroke_xml.findall(".//s:rect", ns)
bright = next(r for r in stroke_rects if r.attrib.get("fill") == "#fff")
base = next(r for r in stroke_rects if r.attrib.get("fill") == "#111827")
assert float(bright.attrib["width"]) / float(base.attrib["width"]) == 1 / 8
assert round(0.125 * 255) == int("20", 16)
renders = []
for name in ["order", "stroke"]:
    for width in [688, 311]:
        path = OUT / f"natural-19.12-mobile-{name}-{width}.png"
        with Image.open(ROOT / path) as picture:
            size = list(picture.size)
        assert size[0] == width
        renders.append({"path": str(path), "sha256": sha(read(path)), "size": size,
            "command": f"inkscape course/figures/practical_{name}.svg --export-width={width} --export-filename={path}", "exit_code": 0,
            "actually_viewed": True, "view_tool": "view_image(detail=original)",
            "inspection": "Labels, numbers and downward arrow visible without clipping at both widths; panel arrangement and limits match the read source."})
result = {"reviewer_task": prior["reviewer_task"], "command": ".venv/bin/python docs/technical-reviews/artifacts/natural-19.12-mobile-figure-recheck.py",
    "prior_report_sha256": PRIOR_SHA, "prior_report_archive": str(HISTORY / "report.json"), "archive_manifest": str(HISTORY / "archive-manifest.json"),
    "actual_read_scope": "Full19.12,full11.14/11.15 and source functions; full20.1/20.2 bridge; current SVGs; all four own688/311 rendered PNGs viewed directly. No other reader or technical report read.",
    "section_sha256": {"19.12": prior["source_sha256"], **section_hashes}, "figure_changes": figure_changes,
    "all_other_registered_source_evidence_paths_matched": len(entries) - 2, "total_exact_prior_archived_paths": len(entries),
    "original_lesson_code_outputs": lesson_outputs, "picture_order": order, "thin_stroke": stroke,
    "exercise_checks": {"changed_RGB_channel_values": 128, "changed_spatial_positions": 64,
        "first_cell_original_RGB_mean": pooled_first[0, :, 1, 0].tolist(), "moving_red_to_next_cell_changes_summary": True,
        "first_cell_moved_RGB_mean": pooled_moved[0, :, 1, 0].tolist(), "next_cell_moved_RGB_mean": pooled_moved[0, :, 1, 1].tolist(),
        "different_stroke_columns8_and15_produce_identical_area_summary": True},
    "visual_inspection": {"renderer": "Inkscape1.4(e7c3feb100,2024-10-09)", "render_warnings": "PangoFT2FontMap/GtkRecentManager wrapping warnings; all exports exited0 and were directly viewed",
        "order": "Both equal-area red/blue bars lie within their same dashed average region and reverse horizontal order. Arrow leads to identical-summary conclusion. Footer declares a magnified local schematic and actual32x32/8x8 program, avoiding exact-scale claims.",
        "stroke": "Eight equal-width columns, one white column; numeric labels1/0 and1÷8→0.125 agree with area averaging. Output dark gray represents rounded8-bit display of0.125, with explicitly magnified schematic/no ChineseOCR test; colors are illustrative, not sampled model inputs.",
        "renders": renders},
    "claim_impact": [{"id": c["id"], "status": c["status"], "statement_sha256": sha(c["statement"].encode()),
        "scope_check": "c43/c44 information-loss scope rechecked against current source/CPU/current diagrams; c45 bridge unchanged, no natural-quality guarantee. Remaining original statements/evidence unchanged."} for c in prior["claims"]],
    "limitations": "Non-injective input examples do not evaluate recognition or natural competence. A deterministic larger downstream model cannot uniquely distinguish identical summaries; retaining detail alone supplies no learned mapping or success guarantee. Original chapter19 synthetic tests remain separate from upstreamDense+LoRA+ASR evidence; no GPU/model inference ran.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "renderer": "Inkscape1.4"},
    "result": "Passed actual prerequisite Python examples/exercises, exact old-history/source hashes and current SVG numeric/geometric checks; owner actually rendered/viewed both figures at688/311 and checked19.12/20.1/20.2 scopes"}
(ROOT / OUT / "natural-19.12-mobile-figure-recheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("result", "total_exact_prior_archived_paths", "picture_order", "thin_stroke", "exercise_checks", "environment")}, ensure_ascii=False, indent=2))
