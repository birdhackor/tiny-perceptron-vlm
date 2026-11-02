"""核對 chapter-17 導讀的精確範圍與明連內容；不執行連結頁程式。"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = Path(__file__).resolve().parent
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix12.json"
COMMAND = (
    "uv run --extra cpu python "
    "docs/technical-reviews/artifacts/omni-technical-chapter-17-v1/inspect_scope.py"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def slice_for(raw, selector):
    text = raw.decode("utf-8")
    headings = list(re.finditer(r"(?m)^## .+$", text))
    if selector == "before-first-section":
        return text[: headings[0].start()].encode(), 1, text[: headings[0].start()].count("\n")
    identifier = selector.removeprefix("section:")
    match = next(h for h in headings if h.group().startswith("## " + identifier + " "))
    next_start = next((h.start() for h in headings if h.start() > match.start()), len(text))
    selected = text[match.start() : next_start]
    first = text[: match.start()].count("\n") + 1
    return selected.encode(), first, first + selected.count("\n") - 1


freeze_raw = (ROOT / FREEZE).read_bytes()
freeze = json.loads(freeze_raw)
pages = {p["page_id"]: p for p in freeze["pages"]}
scope = []
for identifier in ("chapter-17", "12.18", "17.11", "17.16", "17.17"):
    page = pages[identifier]
    frozen = (ROOT / page["snapshot"]).read_bytes()
    current, first, last = slice_for((ROOT / page["source"]).read_bytes(), page["selector"])
    assert sha(frozen) == page["source_sha256"]
    assert current == frozen, identifier
    scope.append(
        {
            "page_id": identifier,
            "source": page["source"],
            "selector": page["selector"],
            "current_lines_inclusive": [first, last],
            "snapshot": page["snapshot"],
            "slice_sha256": sha(frozen),
            "current_matches_snapshot": True,
            "review_scope": (
                "完整導讀；逐項技術核對"
                if identifier == "chapter-17"
                else "明連內容的承接意義；不宣稱本 task 逐項重審其算例、程式、圖或品質"
            ),
        }
    )

page = pages["chapter-17"]
assigned = (ROOT / page["snapshot"]).read_text()
assert page["figures_sha256"] == {}
assert not re.search(r"!\[[^\]]*\]\(", assigned)
assert "```" not in assigned
links = re.findall(r"\[([^]]+)\]\(([^)]+)\)", assigned)
assert {label for label, _ in links} == {"17.11", "名詞頁", "12.18", "17.16", "17.17"}

glossary_page = pages["glossary"]
glossary_raw = (ROOT / glossary_page["snapshot"]).read_bytes()
assert sha(glossary_raw) == glossary_page["source_sha256"]
assert glossary_raw == (ROOT / glossary_page["source"]).read_bytes()
terms = (
    "| 中間特徵（activation）",
    "| 聲音codec、碼本（codebook）",
    "| 量化（quantization）、碼、刻度（scale）、還原值",
    "| 向量量化（Vector Quantization，VQ）",
    "| 直通估計（straight-through estimator，STE）、commitment loss",
    "| 殘差向量量化（Residual Vector Quantization，RVQ）",
)
rows = []
for number, line in enumerate(glossary_raw.decode().splitlines(keepends=True), 1):
    if line.startswith(terms):
        rows.append({"line": number, "row": line.rstrip("\n"), "row_sha256_including_newline": sha(line.encode())})
assert len(rows) == len(terms)

headings = [line for line in (ROOT / "course/chapters/17.md").read_text().splitlines() if line.startswith("## ")]
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": COMMAND,
    "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "CPU (metadata / byte comparison only)", "platform": platform.platform()},
    "source_freeze": {"path": FREEZE, "sha256": sha(freeze_raw)},
    "scope": scope,
    "assigned_content": {"paragraphs": 3, "embedded_code_fences": 0, "image_references": 0, "figures_sha256": {}, "links": links},
    "glossary": {"source": glossary_page["source"], "snapshot": glossary_page["snapshot"], "sha256": sha(glossary_raw), "checked_rows_only": rows},
    "roadmap_headings_only": headings,
    "result": "PASS: assigned intro and four linked section slices match fix12 byte-for-byte; required anchors and six relevant glossary rows exist. No embedded code or figure in assigned intro. Linked-page numerical examples and diagrams are outside this task's technical verdict.",
}
output = ARTIFACT_DIR / "scope-check.json"
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "sha256": sha(output.read_bytes()), "result": result["result"]}, ensure_ascii=False))
