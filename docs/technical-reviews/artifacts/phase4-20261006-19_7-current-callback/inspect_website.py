"""Inspect only the exact current own-page locator and changed paragraph."""
import hashlib
import json
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
local = ROOT / "outputs/site/19.7.html"
local_raw = local.read_bytes()
url = "http://127.0.0.1:8765/19.7.html"
with urllib.request.urlopen(url, timeout=10) as response:
    response_raw = response.read()
    status = response.status
    content_type = response.headers.get("Content-Type")
assert status == 200
assert response_raw == local_raw
(ART / "inputs/19.7-current-own-page.html").write_bytes(local_raw)
(ART / "inputs/19.7-current-http-response.html").write_bytes(response_raw)


class OwnParagraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_p = False
        self.pending = []
        self.paragraphs = []
        self.images = []
    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_p = True
            self.pending = []
        if tag == "img":
            self.images.append(dict(attrs))
    def handle_data(self, data):
        if self.in_p:
            self.pending.append(data)
    def handle_endtag(self, tag):
        if tag == "p" and self.in_p:
            self.paragraphs.append("".join(self.pending))
            self.in_p = False


parser = OwnParagraphs()
parser.feed(response_raw.decode("utf-8"))
changed = [p for p in parser.paragraphs if "所以不是只差一個結束符號" in p]
assert len(changed) == 1
assert "兩次最後生成都有EOS" in changed[0]
assert "只答對10／12" in changed[0]
assert "不是隻差" not in response_raw.decode("utf-8")
figures = [i for i in parser.images if "rewrite-19-tool-roundtrip.svg" in i.get("src", "")]
assert len(figures) == 1
receipt = {
    "own_page_path": str(local.relative_to(ROOT)),
    "own_page_sha256": hashlib.sha256(local_raw).hexdigest(),
    "snapshot_path": str((ART / "inputs/19.7-current-own-page.html").relative_to(ROOT)),
    "url": url,
    "status": status,
    "content_type": content_type,
    "response_sha256": hashlib.sha256(response_raw).hexdigest(),
    "response_equals_own_file": True,
    "response_snapshot_path": str((ART / "inputs/19.7-current-http-response.html").relative_to(ROOT)),
    "actual_semantic_inspection": "Only the paragraph containing the changed spelling and the exact referenced SVG img element; no unrelated page/full-site parity or new visual assertion.",
    "changed_paragraph": changed[0],
    "own_svg_img": figures[0],
    "visual_render_repeated": False,
    "reason": "SVG and all visual claims are byte-identical to personally rendered/viewed original evidence; spelling change introduces no new visual statement.",
    "environment": {"python": sys.version, "device": "CPU; local own-page HTTP GET and HTML parsing only"},
}
(ART / "own-website-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
inspection = json.loads((ART / "current-inspection.json").read_bytes())
inspection["website"] = receipt
(ART / "current-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"own_page_path": receipt["own_page_path"], "own_page_sha256": receipt["own_page_sha256"], "status": status, "response_sha256": receipt["response_sha256"], "changed_paragraph": changed[0], "svg_src": figures[0].get("src"), "new_render": False}, ensure_ascii=False, indent=2))
