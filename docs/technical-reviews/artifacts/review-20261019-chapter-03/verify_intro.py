"""核對導言的資料前提、後文對應與注意力原始來源。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


def lesson(text, heading):
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


chapter = ROOT / "course/chapters/03.md"
previous = ROOT / "course/chapters/02.md"
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/chapter-03.md"
library = ROOT / "tiny_perceptron/attention.py"
chapter_text = chapter.read_text()
intro = chapter_text[: chapter_text.index("## ")]
assert intro.encode() == frozen.read_bytes(), "目前導言應與凍結原文逐位元組相同"
assert digest(frozen) == "dd82f6c23f00e3f5f17fc0da00d318493651349067498f51bb857ae690223c36"
assert not re.search(r"!\[.*?\]\(", intro), "導言不應包含圖引用"

fixture_block = next(
    block for block in blocks(previous.read_text()) if block.startswith('examples = [("紅色物體是"')
)
fixture_tree = ast.parse(fixture_block)
examples = ast.literal_eval(fixture_tree.body[0].value)
assert examples == [("紅色物體是", "圓"), ("藍色物體是", "方")], "玩具資料的前文與答案應符合文字說明"
window_checks = {}
for length in range(1, 6):
    visible = [prefix[-length:] for prefix, _ in examples]
    same = visible[0] == visible[1]
    assert same == (length < 5), "只有包含顏色線索時，兩筆前文才能區分"
    window_checks[str(length)] = {"visible": visible, "same": same}

masked_section = lesson(chapter_text, "## 3.6 整段一起算，怎樣禁止讀後面的答案？")
heads_section = lesson(chapter_text, "## 3.7 幾種讀法怎樣並行，再交回同一位置？")
manual_tree = next(ast.parse(block) for block in blocks(masked_section) if "def manual_attention(" in block)
chapter_manual = next(node for node in manual_tree.body if isinstance(node, ast.FunctionDef))
library_manual = next(
    node for node in ast.parse(library.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "manual_attention"
)
assert ast.dump(chapter_manual) == ast.dump(library_manual), "課文節錄應與函式庫函式相同"

section_outputs = {}
for title, text in [("因果遮罩", masked_section), ("多頭注意力", heads_section)]:
    namespace = {"__name__": "__review_execution__"}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), torch.no_grad():
        for number, block in enumerate(blocks(text)):
            exec(compile(block, f"{chapter.name}:{title}:block-{number}", "exec"), namespace)
    section_outputs[title] = captured.getvalue()

glossary = (chapter.parent / "../glossary.md").resolve()
assert glossary.is_file(), "名詞頁連結應指向現有檔案"

url = "https://arxiv.org/html/1706.03762v7"
curl_command = ["curl", "-fLsS", "--max-time", "45", url]
download = subprocess.run(curl_command, capture_output=True, check=True)
parser = PlainText()
parser.feed(download.stdout.decode())
plain = " ".join(parser.parts)
assert "arXiv:1706.03762v7" in plain, "應讀取明確指定的論文版本"
start = plain.index("An attention function")
end = plain.index("3.3 Position-wise Feed-Forward Networks", start)
passage = plain[start:end]
quotes = [
    {"locator": "Attention", "text": "a weighted sum of the values"},
    {"locator": "Multi-Head Attention", "text": "perform the attention function in parallel"},
    {"locator": "Applications of Attention in our Model", "text": "up to and including that position"},
]
for quote in quotes:
    assert quote["text"] in passage, "短摘錄應能在當次讀取的原文中找到"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-03/verify_intro.py",
    "accessed_at": datetime.now().astimezone().isoformat(),
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": "cpu",
    },
    "source_files": {
        str(path.relative_to(ROOT)): digest(path)
        for path in [frozen, chapter, previous, library, glossary]
    },
    "intro_matches_freeze": True,
    "intro_has_figures": False,
    "examples": examples,
    "window_checks": window_checks,
    "manual_attention_ast_matches_library": True,
    "section_code_outputs": section_outputs,
    "glossary_target": str(glossary.relative_to(ROOT)),
    "original_source": {
        "url": url,
        "version": "arXiv:1706.03762v7, 2023-08-02",
        "command": curl_command,
        "curl_exit": download.returncode,
        "response_sha256": hashlib.sha256(download.stdout).hexdigest(),
        "response_bytes": len(download.stdout),
        "read_scope": [
            "Attention",
            "Scaled Dot-Product Attention, equation (1)",
            "Multi-Head Attention, projected heads and concatenation",
            "Applications of Attention in our Model, decoder masking",
        ],
        "short_excerpts": quotes,
        "full_text_retained": False,
    },
    "result": "所有離線資料、節錄對應與課文程式斷言均通過；已用 curl 讀取指定論文原文，全文僅在記憶體處理。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
