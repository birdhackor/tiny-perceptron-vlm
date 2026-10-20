"""用 curl 讀原始論文，只保存來源收據與支持主張的短摘錄。"""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class Sections(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.text = {}

    def handle_starttag(self, tag, attrs):
        if tag == "section":
            identifier = dict(attrs).get("id", "")
            self.stack.append(identifier)
            self.text.setdefault(identifier, [])

    def handle_endtag(self, tag):
        if tag == "section" and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        for identifier in self.stack:
            self.text[identifier].append(data)


specs = [
    {
        "url": "https://arxiv.org/html/2310.13548v4",
        "title": "Towards Understanding Sycophancy in Language Models",
        "version": "arXiv:2310.13548v4，2025-05-10",
        "inspected_sections": ["Abstract", "Background: AI Assistants and Sycophancy", "What Behavior Is Incentivized By Human Preference Data?", "How Often Do Humans and Preference Models Prefer Truthful Responses?", "Prompt Details；Non-sycophantic PM prompt"],
        "excerpts": [
            {"locator": "S4.SS3", "quote": "prefer convincingly-written sycophantic responses over correct responses"},
            {"locator": "A4.SS2", "quote": "If the statement is factually accurate, please confirm."},
            {"locator": "A4.SS2", "quote": "please provide a clear, polite correction."},
        ],
        "inspection_note": "已讀定義、偏好資料分析及正答／附和答比較；附錄提示明列正確時確認、錯誤時禮貌修正。偏好結果受資料與評分設定限制，並非所有人必然喜歡附和，也未驗證本課示例的訓練效果。",
    },
    {
        "url": "https://arxiv.org/html/2308.03958v2",
        "title": "Simple synthetic data reduces sycophancy in large language models",
        "version": "arXiv:2308.03958v2，2024-02-15",
        "inspected_sections": ["Model scaling and instruction tuning increases sycophancy", "Models are sycophantic for objectively-wrong answers", "Data generation and filtration", "Synthetic-data intervention reduces sycophancy", "Intervention requires filtering prompts containing claims the model does not know the answer to", "Prompt template discussion"],
        "excerpts": [
            {"locator": "S4.SS1", "quote": "Once we construct a true or false claim, we add a user opinion that agrees or disagrees with the claim"},
        ],
        "inspection_note": "已讀算術附和測試、真假主張與同意／不同意的資料生成，以及未見任務、措辭泛化與小模型失敗的限制。支持保持事實與使用者意見獨立的對照設計；不能推論一組人工偏好對足以改善模型。",
    },
]
records = []
for spec in specs:
    command = ["curl", "-fsSL", "--max-time", "45", spec["url"]]
    result = subprocess.run(command, check=True, capture_output=True)
    parser = Sections()
    parser.feed(result.stdout.decode("utf-8"))
    for excerpt in spec["excerpts"]:
        section_text = " ".join(" ".join(parser.text[excerpt["locator"]]).split())
        assert excerpt["quote"] in section_text
    assert sum(len(item["quote"].split()) for item in spec["excerpts"]) <= 25
    records.append({
        **spec,
        "command_argv": command,
        "exit_code": result.returncode,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "full_response_bytes": len(result.stdout),
        "checked_original": True,
        "excerpt_match": True,
        "storage_policy": "全文僅在記憶體讀取；本檔只保存收據、定位與短摘錄。",
    })
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-8/inspect_sources.py",
    "environment": {"python": platform.python_version(), "curl": subprocess.run(["curl", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]},
    "sources": records,
}, ensure_ascii=False, indent=2))
