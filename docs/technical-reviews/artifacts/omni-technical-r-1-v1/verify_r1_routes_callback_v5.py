"""Read the actual source15 R.1/warmup/glossary HTTP routes; do not mutate site."""

from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[4]
REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"
SITE = ROOT / "outputs/omni-review-site-technical-fix15"
BASE = "http://127.0.0.1:8776/"


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids, self.links, self.text = set(), [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))

    def handle_data(self, data):
        self.text.append(data)


records = []
for name, anchors, phrases, links in [
    ("course", ["R.1"], ["理解需要的程式碼、關鍵輸出與解說直接放在正文", "Colab需要Google帳號，雲端執行尚未實測"], ["1.1.html", "first-steps.html#W.1", "first-steps.html#W.2", "first-steps.html#W.3", "glossary.html"]),
    ("first-steps", ["W.1", "W.2", "W.3"], ["本課還沒有實測過 Colab 的雲端執行"], []),
    ("glossary", ["G.1"], [], []),
]:
    with urlopen(BASE + name + ".html", timeout=10) as response:
        raw = response.read(); assert response.status == 200
    assert raw == (SITE / (name + ".html")).read_bytes()
    page = Page(); page.feed(raw.decode()); text = " ".join(page.text)
    assert all(a in page.ids for a in anchors)
    assert all(p in text for p in phrases)
    assert all(link in page.links for link in links)
    records.append({"page_id": name, "url": BASE + name + ".html", "http_status": 200, "html_sha256": sha256(raw).hexdigest(), "required_anchors": anchors, "exact_phrases_checked": phrases, "required_links": links})

current = json.loads((REVIEWS / "numbered-freeze-v13.json").read_text())
previous = json.loads((REVIEWS / "numbered-freeze-v12.json").read_text())
identities = lambda value: [(p["page_id"], p["source_sha256"], p["figures_sha256"]) for p in value["pages"]]
assert identities(current) == identities(previous) and len(current["pages"]) == 26
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_r1_routes_callback_v5.py",
    "environment": {"python": sys.version, "device": "CPU; bounded actual local HTTP and source identity reading only"},
    "numbered_freeze": {"path": str((REVIEWS / "numbered-freeze-v13.json").relative_to(ROOT)), "sha256": sha256((REVIEWS / "numbered-freeze-v13.json").read_bytes()).hexdigest()},
    "canonical_parent": current["canonical_source_freeze"], "all26_numbered_source_and_figure_identities_equal_v12": True,
    "actual_HTTP_routes": records, "scope": "R.1 explicit reader routes on new source15 preview, not deployment or a whole-file collector pass. No Colab navigation, login, cell, training, or global tool approval."}
with Path(__file__).with_name("r1-routes-callback-v5.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps({"routes_checked": [r["page_id"] for r in records], "numbered_sections_unchanged": 26, "canonical_parent": current["canonical_source_freeze"]}, ensure_ascii=False, indent=2))
