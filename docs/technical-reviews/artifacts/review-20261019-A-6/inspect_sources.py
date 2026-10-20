"""以 curl 讀取原始來源，僅保存短摘錄與取得收據。"""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


specs = [
    {
        "id": "owasp",
        "url": "https://genai.owasp.org/llmrisk/llm01-prompt-injection/",
        "version": "LLM01:2025 Prompt Injection；本次取得的 HTML",
        "quotes": [
            "Indirect prompt injections occur when an LLM accepts input from external sources",
            "Segregate and identify external content",
        ],
        "locators": ["Types of Prompt Injection / Indirect Prompt Injections", "Prevention and Mitigation Strategies / Segregate and identify external content"],
    },
    {
        "id": "dcmi",
        "url": "https://www.dublincore.org/specifications/dublin-core/dcmi-terms/",
        "version": "DCMI Metadata Terms，頁首 Date Issued 2020-01-20",
        "quotes": [
            "Date of formal issuance of the resource.",
            "Date (often a range) of validity of a resource.",
        ],
        "locators": ["http://purl.org/dc/terms/issued / Definition", "http://purl.org/dc/terms/valid / Definition"],
    },
    {
        "id": "squad",
        "url": "https://aclanthology.org/P18-2124/",
        "version": "ACL 2018 short paper P18-2124，出版者網頁 Abstract",
        "quotes": ["determine when no answer is supported by the paragraph and abstain from answering"],
        "locators": ["Abstract / SQuADRUn combines answerable and unanswerable questions"],
    },
]

receipts = []
for spec in specs:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", spec["url"]]
    response = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(response.stdout.decode("utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    for quote in spec["quotes"]:
        assert quote in text, "來源內容缺少預期短摘錄"
    assert len(" ".join(spec["quotes"]).split()) <= 25
    receipts.append({
        **spec,
        "command_argv": command,
        "exit_code": response.returncode,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_response_sha256": hashlib.sha256(response.stdout).hexdigest(),
        "full_response_bytes": len(response.stdout),
        "checked_original": True,
        "storage_scope": "完整 HTML 僅在記憶體解析；本檔只存短摘錄與取得資訊。",
    })

print(json.dumps({"python": platform.python_version(), "sources": receipts}, ensure_ascii=False, indent=2))
