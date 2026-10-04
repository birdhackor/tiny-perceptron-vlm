"""Fresh, bounded R.1 audit: source, local generated site, CPU example, links."""
import contextlib
import datetime
import functools
import hashlib
import http.server
import importlib.metadata
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import nbformat
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_course
import export_course

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def section(path, lesson):
    raw = path.read_text()
    headings = list(re.finditer(r"^## .+$", raw, re.M))
    for i, heading in enumerate(headings):
        if heading.group().startswith("## " + lesson + " "):
            return raw[heading.start():headings[i+1].start() if i+1 < len(headings) else len(raw)]
    raise ValueError(lesson)

def execute(source):
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(source, "<original-course-code>", "exec"), namespace)
    return output.getvalue(), namespace

failures = []
report = {"reviewer_task": "/root/technical_dep_r1", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-r1-audit.py", "environment": {"python": sys.version.split()[0], "device": "CPU; local Chromium headless", **{package: importlib.metadata.version(package) for package in ["torch", "playwright", "nbformat", "zensical"]}}}
index = json.loads((ROOT / "course/lesson-index.json").read_text())
site = ROOT / "outputs/site"
info = json.loads((site / "build-info.json").read_text())
report["build_info"] = info
report["input_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in [ROOT/"scripts/build_course.py", ROOT/"scripts/export_course.py", ROOT/"course/lesson-index.json", ROOT/"zensical.toml", ROOT/"course/README.md", ROOT/"course/first-steps.md", ROOT/"course/chapters/01.md"]}
raw = (ROOT / "course/README.md").read_text()
intro = raw[:re.search(r"^## ", raw, re.M).start()]
report["intro_sha256"] = hashlib.sha256(intro.encode()).hexdigest()
report["source_sha256"] = hashlib.sha256(section(ROOT/"course/README.md", "R.1").encode()).hexdigest()
report["prerequisite_sha256"] = {lesson: hashlib.sha256(section(ROOT/path, lesson).encode()).hexdigest() for lesson, path in [("1.1", "course/chapters/01.md"), ("W.1", "course/first-steps.md"), ("W.2", "course/first-steps.md")]}

one = section(ROOT / "course/chapters/01.md", "1.1")
one_code = re.search(r"```python\n(.*?)```", one, re.S)[1]
stdout, namespace = execute(one_code)
expected_stdout = "['。', '狗', '看', '貓', '，']\n[3, 2, 1, 4, 1, 2, 3, 0]\n貓看狗，狗看貓。\n"
assert stdout == expected_stdout
assert namespace["ids"][0] == namespace["ids"][6] == 3
bird_stdout, bird = execute(one_code.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"'))
assert bird["restored"] == "鳥看狗，狗看鳥。"
warm_code = re.search(r"```python\n(.*?)```", section(ROOT/"course/first-steps.md", "W.2"), re.S)[1]
warm_stdout, _ = execute(warm_code)
assert warm_stdout == "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n"
report["cpu_example"] = {"source": "course/chapters/01.md#1.1 first original Python fence, no bootstrap or model training", "stdout": stdout, "expected": expected_stdout, "equal": stdout == expected_stdout, "cat_ids": [namespace["ids"][0], namespace["ids"][6]], "restored": namespace["restored"], "W1_bird_exercise_stdout": bird_stdout, "W2_original_stdout": warm_stdout}

rows = []
document_targets = {(ROOT/path).resolve(): name+".md" for name, path in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    document_targets[(ROOT/item["source"]).resolve()] = "chapter-"+Path(item["source"]).stem+".md"
    document_targets[(ROOT/item["notebook"]).resolve()] = item["id"]+".md"
    lesson_targets.setdefault((ROOT/item["source"]).resolve(), {})[item["id"]] = item["id"]+".md"
for item in index:
    source = ROOT/item["source"]
    body = section(source, item["id"])
    heading, content = body.split("\n", 1)
    title = heading.split(" ", 2)[2]
    generated = build_course.notebook(item["id"], title, content, source)
    original_path = ROOT/item["notebook"]
    original = json.loads(original_path.read_text())
    row = {"lesson": item["id"], "source_to_notebook_equal": generated == original, "site_download_equal": (site/item["notebook"]).read_bytes() == original_path.read_bytes()}
    exported = (ROOT/"outputs/zensical/docs"/(item["id"]+".md")).read_text()
    row["exported_prose_current"] = all(export_course.reading_markdown(export_course.joined(cell["source"]), source, document_targets, lesson_targets, info["revision"]) in exported for cell in original["cells"][1:] if cell["cell_type"] == "markdown")
    nbformat.validate(nbformat.from_dict(original))
    page = BeautifulSoup((site/(item["id"]+".html")).read_text(), "html.parser")
    action = page.select_one(".actions")
    colab = action.find("a", string=re.compile("在 Colab")) if action else None
    download = action.find("a", attrs={"download": True}) if action else None
    expected_url = f'https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/{info["revision"]}/{item["notebook"]}'
    row.update({"colab_url": colab.get("href") if colab else None, "colab_equal": colab is not None and colab["href"] == expected_url, "download_link_equal": download is not None and download["href"] == item["notebook"], "warmup_nav": bool(page.select('nav a[href$="first-steps.html"]'))})
    try:
        executed = export_course.checked_notebook(original, ROOT/"outputs/tool-choice-site-kernels"/Path(item["notebook"]).relative_to("notebooks"))
        row["executed_copy_current"] = True
    except ValueError as error:
        executed = original
        row["executed_copy_current"] = False
        row["executed_error"] = str(error)
    code_cells = [c for c in executed["cells"] if c["cell_type"] == "code" and not any(k in c["metadata"] for k in ["course_setup", "course_figure"])]
    row["program_cells"] = len(code_cells)
    row["output_cells"] = len(page.select("div.output"))
    python_blocks = page.select(".language-python.highlight pre code")
    row["output_alignment"] = len(code_cells) == len(python_blocks) and all(export_course.joined(c["source"]).rstrip() == block.get_text().rstrip() for c, block in zip(code_cells, python_blocks))
    output_expected = [export_course.output_html(c.get("outputs", [])) for c in code_cells]
    expected_text = [BeautifulSoup(html, "html.parser").get_text("", strip=False) for html in output_expected if html]
    actual_text = [node.get_text("", strip=False) for node in page.select("div.output")]
    expected_media = [image["src"] for html in output_expected if html for image in BeautifulSoup(html, "html.parser").select("img")]
    actual_media = [image["src"] for output in page.select("div.output") for image in output.select("img")]
    row["printed_outputs_equal"] = row["executed_copy_current"] and expected_text == actual_text and expected_media == actual_media
    for field in ["source_to_notebook_equal", "site_download_equal", "exported_prose_current", "colab_equal", "download_link_equal", "warmup_nav", "executed_copy_current", "output_alignment", "printed_outputs_equal"]:
        if not row[field]:
            failures.append({"lesson": item["id"], "failed_check": field})
    rows.append(row)
report["all_lesson_checks"] = {"lesson_count": len(rows), "passed_by_check": {field: sum(bool(row[field]) for row in rows) for field in ["source_to_notebook_equal", "site_download_equal", "exported_prose_current", "colab_equal", "download_link_equal", "warmup_nav", "executed_copy_current", "output_alignment", "printed_outputs_equal"]}, "failed_rows": [row for row in rows if any(f["lesson"] == row["lesson"] for f in failures)], "sampled_rows": [row for row in rows if row["lesson"] in ["1.1", "B.4", "B.5", "B.6", "B.7", "B.8"]]}
all_html = list(site.glob("*.html"))
missing_nav = [p.name for p in all_html if not BeautifulSoup(p.read_text(), "html.parser").select('nav a[href$="first-steps.html"]')]
report["all_page_navigation"] = {"pages_checked": len(all_html), "missing_warmup_navigation": missing_nav}
failures.extend({"page": p, "failed_check": "warmup_nav"} for p in missing_nav)
report["executed_input_directory"] = "outputs/tool-choice-site-kernels (site export input; validation does not imply rerunning all kernels)"
report["browser_checks"] = {"completed": False}
report["failures"] = failures
(DEST/"tool-choice-dep-r1-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print("Source, notebook, site and published output comparison completed.", flush=True)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(site)))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_port}/"
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        report["environment"]["chromium"] = browser.version
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page.set_default_timeout(10000)
        page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        response = page.goto(base+"course.html#R.1", wait_until="networkidle")
        assert response.status == 200
        assert page.locator('[id="R.1"]').count() == 1
        page.get_by_role("link", name="第一節：文字怎麼變成數字？", exact=True).click()
        page.wait_for_url("**/1.1.html")
        assert "實際執行結果 · CPU" in page.locator("article").inner_text()
        assert page.locator("article .output").first.inner_text().rstrip().endswith(stdout.rstrip())
        notebook = json.loads((ROOT/"notebooks/01/1.1.ipynb").read_text())
        paragraphs = [node.get_text(" ", strip=True) for cell in notebook["cells"] if cell["cell_type"] == "markdown" for node in BeautifulSoup(markdown.markdown(export_course.joined(cell["source"])), "html.parser").find_all("p")]
        live_paragraphs = page.locator("article p").all_inner_texts()
        normalize = lambda text: "".join(text.split())
        assert all(normalize(paragraph) in [normalize(text) for text in live_paragraphs] for paragraph in paragraphs)
        with page.expect_download() as download_info:
            page.locator("article .actions a[download]").click()
        downloaded = Path(download_info.value.path())
        download_sha = digest(downloaded)
        assert download_sha == digest(ROOT/"notebooks/01/1.1.ipynb")
        visible_colab = page.locator('article .actions a[href^="https://colab.research.google.com/"]').get_attribute("href")
        page.locator('article a[href$="first-steps.html#W.2"]').first.click()
        page.wait_for_url("**/first-steps.html#W.2")
        assert page.locator('[id="W.2"]').count() == 1
        assert page.locator('[id="W.1"]').count() == 1
        browser_checks = {"completed": True, "course_R1_http": response.status, "first_lesson_clicked": True, "CPU_output_visible_and_matches_fresh_run": True, "first_lesson_prose_paragraphs_compared": len(paragraphs), "first_lesson_prose_matches_Notebook": True, "download_clicked_and_sha256": download_sha, "colab_visible_url": visible_colab, "W2_clicked_and_anchor_exists": True, "W1_anchor_exists": True, "desktop_warmup_navigation": page.locator('nav a[href$="first-steps.html"]:visible').count()}
        print("Desktop R.1/1.1/W.2 navigation and notebook download completed.", flush=True)
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.set_default_timeout(10000)
        mobile.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        mobile.goto(base+"B.8.html", wait_until="networkidle")
        mobile.locator('label.md-header__button[for="__drawer"]').click()
        mobile.locator('label[id="__nav_2_label"]').click()
        mobile.locator('nav a[href$="first-steps.html"]').filter(has_text="基礎暖身").first.click()
        mobile.wait_for_url("**/first-steps.html")
        browser_checks["mobile_B8_menu_to_warmup_clicked"] = True
        page.goto(base+"figures/character_ids.svg", wait_until="load")
        page.locator("svg").screenshot(path=str(DEST/"tool-choice-dep-r1-character-ids-render.png"))
        browser_checks["prerequisite_figure_rendered"] = "course/figures/character_ids.svg"
        report["browser_checks"] = browser_checks
        browser.close()
finally:
    server.shutdown()
    server.server_close()
report["failures"] = failures
report["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
report["scope"] = "Local current-source/site contracts for R.1; no authenticated Google Colab runtime, installation, GPU training or inference quality claim. nbformat validation and body/code alignment cover all chapter lesson downloads."
(DEST/"tool-choice-dep-r1-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({key: value for key, value in report.items() if key != "all_lesson_checks"}, ensure_ascii=False, indent=2))
print(f"AUDITED {len(rows)} chapter lessons; {len(failures)} failed checks")
