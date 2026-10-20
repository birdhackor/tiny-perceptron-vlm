"""保存凍結頁面、導覽與原始論文讀取的最小收據。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).with_name("evidence.json")
COMMAND = "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python " + Path(__file__).relative_to(ROOT).as_posix()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_original(url, version, excerpts):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(result.stdout.decode("utf-8"))
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    assert version in text, "原始來源的版本標記不符"
    for excerpt in excerpts:
        assert excerpt in text, "指定原文摘錄未出現在讀取內容"
    return {
        "url": url,
        "version": version,
        "accessed_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "returncode": result.returncode,
        "stderr": result.stderr.decode("utf-8"),
        "retrieved_bytes": len(result.stdout),
        "retrieved_sha256": digest(result.stdout),
        "excerpts": excerpts,
        "excerpt_matches": True,
        "full_text_saved": False,
    }


freeze_path = Path("docs/course-review-20261019/freeze-04.json")
freeze_raw = (ROOT / freeze_path).read_bytes()
freeze = json.loads(freeze_raw)
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-08")
raw = (ROOT / page["snapshot"]).read_bytes()
body = raw.decode("utf-8")
assert digest(raw) == page["source_sha256"], "凍結頁面內容與 manifest 不符"
live_raw = (ROOT / page["source"]).read_bytes()
live_intro = live_raw.split(b"\n## ", 1)[0] + b"\n"
assert live_intro == raw, "目前章首與凍結頁面不同"
sections = []
for item in freeze["pages"]:
    if item["source"] != page["source"] or item["page_id"] == page["page_id"]:
        continue
    section_raw = (ROOT / item["snapshot"]).read_bytes()
    assert digest(section_raw) == item["source_sha256"], "導覽目的頁內容與 manifest 不符"
    sections.append({
        "title": item["title"],
        "page_id": item["page_id"],
        "path": item["snapshot"],
        "sha256": digest(section_raw),
        "first_paragraph": section_raw.decode("utf-8").split("\n\n")[1],
    })
links = []
for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", body):
    path = (ROOT / page["source"]).parent / target
    assert path.is_file(), "頁面連結的目標不存在"
    links.append({"target": target, "resolved_path": path.resolve().relative_to(ROOT).as_posix(), "exists": True})
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "device": "CPU；僅讀取文字與計算 SHA-256，未訓練或推論模型",
    "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
}
evidence = {
    "command": COMMAND,
    "environment": environment,
    "local": {
        "freeze": {"path": freeze_path.as_posix(), "sha256": digest(freeze_raw)},
        "page": {"path": page["snapshot"], "sha256": digest(raw), "matches_manifest": True},
        "live_intro_matches_freeze": True,
        "images": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
        "code_fences": body.count("```"),
        "sections": sections,
        "links": links,
    },
    "originals": [
        read_original(
            "https://arxiv.org/html/2311.07911v1", "2311.07911v1",
            ["Entire output should be wrapped in JSON format.", "Do not include keywords {forbidden words} in the response."],
        ),
        read_original(
            "https://arxiv.org/html/2203.02155v1", "2203.02155v1",
            ["given some examples of the task as input", "We then collect a dataset of human-written demonstrations of the desired output behavior"],
        ),
        read_original(
            "https://arxiv.org/html/2106.09685v2", "2106.09685v2",
            ["we constrain its update by representing the latter with a low-rank decomposition"],
        ),
    ],
    "result": "凍結內容、導覽目的頁、名詞頁連結與指定版本原文摘錄均核對成功。",
}
serialized = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
with DESTINATION.open("x", encoding="utf-8") as handle:
    handle.write(serialized)
print(serialized, end="")
