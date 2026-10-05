"""Bounded owner recheck of the changed 20.7 prerequisite; no model downloads."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from playwright.sync_api import sync_playwright
from tiny_perceptron.data import render_chat

OUT = ROOT / "docs/technical-reviews/artifacts/natural-v4-factual/R.3/corrective-001"
BASE = "http://127.0.0.1:8790/"
torch.set_num_threads(1)
result = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available()}, "alignment": []}
for answer, expected_input_length, expected_targets in [("A", 6, [73, 2]), ("AB", 7, [73, 74, 2])]:
    x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": answer}])
    active = y != -100
    first = active.nonzero()[0].item()
    assert len(x) == expected_input_length
    assert first == 4 and x[first].item() == 4
    assert y[active].tolist() == expected_targets
    result["alignment"].append({"question": "Q", "answer": answer, "x": x.tolist(), "y": y.tolist(), "input_length": len(x), "effective_targets": active.sum().item(), "first_effective_target_position": first, "input_at_first": x[first].item(), "target_ids": y[active].tolist()})
current_chapter = (ROOT / "course/chapters/20.md").read_text()
start = current_chapter.index("## 20.7 ")
end = current_chapter.find("\n## ", start + 1)
section = current_chapter[start:end + 1 if end >= 0 else None]
notebook = json.loads((ROOT / "notebooks/20/20.7.ipynb").read_text())
code_blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", section, re.S)]
notebook_blocks = ["".join(c["source"]).strip() for c in notebook["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and not c.get("metadata", {}).get("course_figure")]
assert code_blocks == notebook_blocks
result["changed_section_notebook_python_parity"] = True
result["source_sha256"] = {"chapter20": hashlib.sha256((ROOT / "course/chapters/20.md").read_bytes()).hexdigest(), "20.7": hashlib.sha256(section.encode()).hexdigest()}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    result["environment"]["chromium"] = browser.version
    page = browser.new_page(viewport={"width": 1360, "height": 1000}, device_scale_factor=1)
    response = page.goto(BASE + "course.html#R.3", wait_until="networkidle")
    assert response.status == 200
    h = page.locator('[id="R.3"]')
    links = h.evaluate('el=>{let links=[];for(let n=el.nextElementSibling;n&&!/^H[12]$/.test(n.tagName);n=n.nextElementSibling){links.push(...Array.from(n.querySelectorAll("a")).map(a=>({text:a.innerText,url:a.href})));}return links;}')
    chapter20_link = next(link for link in links if link["url"].endswith("chapter-20.html"))
    response = page.goto(chapter20_link["url"], wait_until="networkidle")
    assert response.status == 200
    entry_links = page.locator("article a").evaluate_all('els=>els.map(a=>({text:a.innerText,url:a.href}))')
    for identifier in ["20.6", "20.7", "20.8"]:
        assert any(link["url"] == BASE + identifier + ".html" for link in entry_links)
    result["current_chapter20_route"] = {"from": BASE + "course.html#R.3", "chapter_entry": page.url, "status": response.status, "20.6_to_20.8_links_present": True}
    page.locator('article a[href$="/20.7.html"]').click()
    page.wait_for_load_state("networkidle")
    article_text = page.locator("article").inner_text()
    assert "輸入與下一項目標的對齊方式" in article_text
    assert "不是把完整Qwen圖片輸入假定成六格" in article_text
    notebook_links = page.locator('article a[href$="20.7.ipynb"]').evaluate_all('els=>els.map(a=>({text:a.innerText,url:a.href}))')
    assert notebook_links
    figure = page.locator('article img[src*="natural-v4-answer-mask.svg"]')
    assert figure.count() == 1
    figure_url = figure.get_attribute("src")
    image_response = page.request.get(figure_url)
    assert image_response.status == 200
    served_svg = image_response.body()
    local_svg = (ROOT / "course/figures/natural-v4-answer-mask.svg").read_bytes()
    assert served_svg == local_svg
    render = OUT / "answer-mask-current.png"
    figure.screenshot(path=str(render))
    result["current_20.7_preview"] = {"url": page.url, "heading": page.locator("article h1").inner_text(), "updated_alignment_phrase_present": True, "tiny_example_scope_present": True, "notebook_links": notebook_links}
    result["figure"] = {"path": "course/figures/natural-v4-answer-mask.svg", "sha256": hashlib.sha256(local_svg).hexdigest(), "served_url": figure_url, "served_bytes_equal_local": True, "render": str(render.relative_to(ROOT)), "render_sha256": hashlib.sha256(render.read_bytes()).hexdigest(), "render_method": "Actual current 8790 section image rendered by Chromium and element screenshot; no figure transformation"}
    result["scope"] = "Two deterministic byte-encoding examples and one changed chapter/section route; existing genuine R.3 original-authority/CPU records reused after byte equality. No whole-book rerun, training, GPU, ASR/Qwen inference or release-score audit."
    browser.close()
print(json.dumps(result, ensure_ascii=False, indent=2))
