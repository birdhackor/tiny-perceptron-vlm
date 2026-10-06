"""Render current figures and inspect actual local section page; no model execution."""
import difflib
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "figure-reinspection"
OUT.mkdir(exist_ok=True)
PREVIOUS = Path(__file__).resolve().parent
PRIOR_SHA = "4b8dc50adda26f0900317fa14438d6bc04bc42459ddb324f62902711ba8c8446"
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name, obj): (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
report_path = ROOT / "docs/technical-reviews/6.5.json"
prior_raw = report_path.read_bytes()
assert sha(prior_raw) == PRIOR_SHA
history = ROOT / ("docs/technical-reviews/history/phase4-6_5-own-prior-before-figure-reinspection-" + PRIOR_SHA + ".json")
if history.exists():
    assert history.read_bytes() == prior_raw
else:
    history.write_bytes(prior_raw)
assert history.read_bytes() == prior_raw
prior = json.loads(prior_raw)
spec = importlib.util.spec_from_file_location("original_section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
section, _, first_line = helper.original_section(ROOT / "course/chapters/06.md", "6.5")
assert sha(section) == prior["source_sha256"] == "57a909b96283fd0b916af68c6e7dda9e1d0dc9085d11f0d8cbf60ec54c385b91"
assert section == (PREVIOUS / "recheck/section.md").read_bytes()
(OUT / "section.md").write_bytes(section)
fences = helper.fences(section, first_line)
assert len(fences) == 1 and fences[0]["raw"] == (PREVIOUS / "fence-1.py").read_bytes()
unchanged_artifacts = []
for art in prior["artifacts"]:
    observed = sha((ROOT / art["path"]).read_bytes())
    assert observed == art["sha256"], art["id"]
    unchanged_artifacts.append({"id": art["id"], "path": art["path"], "sha256": observed})
figures = {}
render_receipts = []
for short, filename in [("shared-denominator", "rewrite-06-05-shared-denominator.svg"), ("common-scale", "tokenizer_common_scale.svg")]:
    path = ROOT / "course/figures" / filename
    raw = path.read_bytes()
    (OUT / filename).write_bytes(raw)
    old = (PREVIOUS / "inputs/course/figures" / filename).read_bytes()
    (OUT / (short + ".diff")).write_text("".join(difflib.unified_diff(old.decode().splitlines(True), raw.decode().splitlines(True), fromfile="own-original-frozen-figure", tofile="personally-inspected-current-figure")))
    png = OUT / (short + ".png")
    argv = ["inkscape", str(path), "--export-type=png", "--export-filename=" + str(png)]
    run = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    (OUT / (short + "-render.stdout.txt")).write_text(run.stdout)
    (OUT / (short + "-render.stderr.txt")).write_text(run.stderr)
    assert run.returncode == 0 and png.is_file()
    figures[str(path.relative_to(ROOT))] = {"sha256": sha(raw), "initial_sha256": sha(old), "changed": raw != old, "snapshot": str((OUT / filename).relative_to(ROOT)), "render": str(png.relative_to(ROOT)), "png_sha256": sha(png.read_bytes())}
    render_receipts.append({"command_argv": argv, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
assert figures["course/figures/tokenizer_common_scale.svg"]["sha256"] == "c56a5b45a970c1f68b11f96b7faf7974e0893fa517cf36f26a903a72ecef5180"
assert not figures["course/figures/rewrite-06-05-shared-denominator.svg"]["changed"]
# Only read the needed original measurement pointers, avoiding unrelated author's result notes.
result_raw = (ROOT / "docs/course-experiments/results/tokenizer.json").read_bytes()
assert result_raw == (PREVIOUS / "inputs/docs/course-experiments/results/tokenizer.json").read_bytes()
record = json.loads(result_raw)
test_byte = record["results"]["runs"]["byte256"]["after"]["test"]
test_bpe = record["results"]["runs"]["bpe512"]["after"]["test"]
values = [test_byte["mean_token_nll"], test_bpe["mean_token_nll"], test_byte["bpb_including_eos_boundary_targets"], test_bpe["bpb_including_eos_boundary_targets"]]
expected_strings = [f"{v:.5f}" for v in values]
svg = ET.parse(OUT / "tokenizer_common_scale.svg").getroot()
ns = {"s": "http://www.w3.org/2000/svg"}
texts = [e.text for e in svg.findall("s:text", ns)]
assert all(s in texts for s in expected_strings)
bars = [e for e in svg.findall("s:rect", ns) if e.get("fill") in ["#d97706", "#2563eb"]]
assert len(bars) == 4
bar_checks = []
for elem, value, displayed in zip(bars, values, expected_strings, strict=True):
    width = float(elem.get("width"))
    err_raw = abs(width - 120 * value)
    err_displayed = abs(width - 120 * float(displayed))
    assert elem.get("x") == "32" and err_raw < 0.0006 and err_displayed < 1e-10
    bar_checks.append({"x":32, "y":float(elem.get("y")), "width":width, "original_record_value":value, "displayed_value":displayed, "px_per_unit":120, "raw_record_width_error_px":err_raw, "rounded_label_width_error_px":err_displayed})
assert test_byte["records"] == test_bpe["records"] == 20
assert test_byte["raw_utf8_bytes"] == test_bpe["raw_utf8_bytes"] == 4193
assert test_byte["effective_tokens"] == 4213 and test_bpe["effective_tokens"] == 2503
assert "逐byte 4,213；BPE 2,503" in texts
assert "共4,193個原文UTF-8 bytes" in texts
assert "原文bytes不含EOS或結構標記。" in texts
pages = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    for name, viewport in [("desktop", {"width":1280,"height":800}), ("mobile", {"width":390,"height":844})]:
        context = browser.new_context(viewport=viewport, device_scale_factor=1)
        page = context.new_page()
        response = page.goto("http://127.0.0.1:8765/6.5.html", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_function("Array.from(document.images).every(img=>img.complete)", timeout=20000)
        page.screenshot(path=str(OUT / (name + "-page-opening.png")))
        page.locator("details").evaluate_all("els => els.forEach(e => e.open = true)")
        image = page.locator('img[src*="tokenizer_common_scale.svg"]').first
        image.wait_for(state="visible", timeout=10000)
        image.scroll_into_view_if_needed(timeout=10000)
        image.screenshot(path=str(OUT / (name + "-empirical-figure-in-page.png")), timeout=15000)
        image.evaluate("e => window.scrollTo(0,e.getBoundingClientRect().top+window.scrollY-90)")
        page.screenshot(path=str(OUT / (name + "-empirical-page-top.png")))
        image.evaluate("e => window.scrollTo(0,e.getBoundingClientRect().bottom+window.scrollY-window.innerHeight+50)")
        page.screenshot(path=str(OUT / (name + "-empirical-page-bottom.png")))
        metadata = page.evaluate("""() => ({title:document.title, viewport:{width:window.innerWidth,height:window.innerHeight}, documentWidth:document.documentElement.scrollWidth, figures:Array.from(document.images).map(e=>({src:e.getAttribute('src'),complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}))})""")
        figures_response = page.request.get(image.get_attribute("src"), timeout=10000)
        fetched_svg = figures_response.body()
        assert sha(fetched_svg) == sha((OUT / "tokenizer_common_scale.svg").read_bytes())
        metadata.update({"name": name, "response_status": response.status, "url": page.url, "details_opened":True,"empirical_svg_response_sha256":sha(fetched_svg),"browser_version":browser.version})
        (OUT / (name + "-page.html")).write_text(page.content())
        pages.append(metadata)
        context.close()
    browser.close()
environment = {"python": sys.version, "python_executable":sys.executable,"device":"cpu","platform":platform.platform(),
               "inkscape":subprocess.check_output(["inkscape","--version"],text=True).strip(),"chromium":subprocess.check_output(["chromium","--version"],text=True).strip(),"playwright":"installed repo .venv package"}
write("environment.json", environment)
write("results.json", {"reviewer_task":prior["reviewer_task"],"kind":"own_figure_support_reinspection","prior_history_path":str(history.relative_to(ROOT)),"prior_history_sha256":PRIOR_SHA,
      "current_section_sha256":sha(section),"current_section_unchanged":True,"original_fence_sha256":sha(fences[0]["raw"]),"unchanged_formal_artifacts":unchanged_artifacts,
      "figures":figures,"render_receipts":render_receipts,"page_receipts":pages,"raw_result_sha256":sha(result_raw),
      "raw_measurement_pointers_read":["/results/runs/byte256/after/test/records","/results/runs/byte256/after/test/effective_tokens","/results/runs/byte256/after/test/raw_utf8_bytes","/results/runs/byte256/after/test/mean_token_nll","/results/runs/byte256/after/test/bpb_including_eos_boundary_targets","same five test fields for bpe512"],
      "bar_checks":bar_checks,"support_scope":"Twenty held-out excerpts,4193 original UTF-8bytes. Mean token NLL per4213/2503 EOS-inclusive targets versus BPB per shared raw bytes, including EOS in numerator only. New bars startx32 and use120px per displayed numeric unit instead of oldx140/80px. This alters layout/geometry, not metric units/denominators/ranking or generalization scope.",
      "reuse_scope":"All prior formal source/code/CPU/measurement artifacts hash checked unchanged and reused, no rerun of original CPU or model evaluation. Only current figures and actual localhost page were newly rendered.",
      "requires_actual_view_before_final_verdict":True,"whole_current_section_personally_read":True,"unresolved_after_numeric_checks":[]})
print(json.dumps({"history_path":str(history.relative_to(ROOT)),"prior_history_sha256":PRIOR_SHA,"section_sha256":sha(section),"current_figure_sha256":figures["course/figures/tokenizer_common_scale.svg"]["sha256"],"renders":len(render_receipts),"pages":len(pages),"bars":bar_checks,"actual_view_pending":True},ensure_ascii=False,indent=2))
