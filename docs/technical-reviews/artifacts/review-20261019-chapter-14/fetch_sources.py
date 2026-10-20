"""以 curl 讀取原始來源，只保存短摘錄與取得收據，不保存第三方全文。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.depth = 0
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.current = []
        if self.current is not None and tag in ("math",):
            attributes = dict(attrs)
            if attributes.get("alttext"):
                self.current.append(attributes["alttext"])

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            self.paragraphs.append(re.sub(r"\s+", " ", "".join(self.current)).strip())
            self.current = None


specifications = [
    ("ud_subject", "https://universaldependencies.org/zh/dep/nsubj.html", "2026-10-20 讀取的官方中文依存語法頁面；以取得的原文 SHA-256 識別版本", [
        ("nominal subject 定義", "The nsubj relation links", "The nsubj relation links a subject to its predicate, and this subject must be nominal."),
    ]),
    ("ud_object", "https://universaldependencies.org/zh/dep/obj.html", "2026-10-20 讀取的官方中文依存語法頁面；以取得的原文 SHA-256 識別版本", [
        ("object 定義", "The object of a transitive verb", "The object of a transitive verb is its second core argument after its subject."),
    ]),
    ("position_interpolation", "https://arxiv.org/html/2306.15595v2", "arXiv:2306.15595v2", [
        ("Direct Extrapolation", "While the attention score in RoPE only depends", "While the attention score in RoPE only depends on the relative positions, which is what we want, its extrapolation performance is not great"),
        ("Proposed approach: Position Interpolation", "In Fig.", "\\tilde{a}(s)=a(Ls/L^{\\prime})"),
        ("Fine-tuning", "Fine-tuning.", "We can further fine-tune the interpolated model using the next token prediction task with interpolated position encodings on the extended context window size"),
    ]),
    ("universal_transformer", "https://arxiv.org/html/1807.03819v3", "arXiv:1807.03819v3，2019-03-05", [
        ("The Universal Transformer：更新軸", "However, in contrast to most applications", "However, in contrast to most applications of recurrent neural networks to sequential data, the UT does not recur over positions in the sequence, but over consecutive revisions of the vector representations of each position (i.e., over “depth”)."),
        ("The Universal Transformer：共享 transition", "Then, by applying a transition function", "Then, by applying a transition function (shared across position and time) to the outputs of the self-attention mechanism, independently at each position."),
    ]),
]
receipts = []
for identifier, url, version, selectors in specifications:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    parser = Paragraphs()
    parser.feed(raw.decode("utf-8"))
    excerpts = []
    for locator, needle, exact in selectors:
        candidates = [p for p in parser.paragraphs if needle in p]
        if identifier == "position_interpolation" and locator == "Proposed approach: Position Interpolation":
            candidates = [p for p in candidates if "interpolation is much more stable" in p]
        if len(candidates) != 1:
            raise ValueError(f"來源定位需要唯一段落：{identifier} {locator}；找到 {len(candidates)} 段")
        paragraph = candidates[0]
        if exact:
            if exact not in paragraph:
                raise ValueError(f"來源摘錄不符：{identifier} {locator}")
            excerpt = exact
        else:
            excerpt = paragraph[:1800]
        excerpts.append({"locator": locator, "excerpt": excerpt})
    receipts.append({
        "id": identifier,
        "url": url,
        "version": version,
        "accessed_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "returncode": completed.returncode,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpts": excerpts,
        "full_text_saved": False,
    })
print(json.dumps({
    "environment": {"python": platform.python_version(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
    "sources": receipts,
}, ensure_ascii=False, indent=2))
