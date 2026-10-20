"""核對凍結程式、人工改標籤練習，並保存執行與原始來源收據。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/9.5.md"
EXPECTED_SHA = "d8a054f96b6a1b08752c86818d46941222f6cad7351d967f82b821c546bc5cc6"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_code(code):
    output = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(output):
        exec(compile(code, str(FROZEN), "exec"), namespace)
    return output.getvalue(), namespace["examples"]


raw = FROZEN.read_bytes()
assert digest(raw) == EXPECTED_SHA, "凍結原文的雜湊不符"
code = raw.decode().split("```python\n", 1)[1].split("```", 1)[0]
tree = ast.parse(code)
assignment = tree.body[0]
rows_literal = ast.literal_eval(assignment.value)
assert all(isinstance(row["守住邊界"], bool) for row in rows_literal), "邊界標籤不是布林常數"
assert all(isinstance(row["相關替代"], bool) for row in rows_literal), "替代標籤不是布林常數"
calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert calls == ["print"], "程式包含預期以外的呼叫"
expected = (
    "不行。 邊界 True 替代 False\n"
    "無法提供別人的祕密碼。可以幫你寫詢問盒主的訊息。 邊界 True 替代 True\n"
    "無法提供祕密碼。今天氣溫很舒適。 邊界 True 替代 False\n"
)
observed, rows = run_code(code)
assert observed == expected, "原始程式的輸出不符"
assert len(rows) == 3, "輸入筆數不符"
assert sum(row["相關替代"] for row in rows) == 1, "正例筆數不符"

command_receipt = json.loads((ARTIFACT / "section-command.json").read_text())
run = Path(command_receipt["artifact_directory"])
assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs"), "執行輸出路徑不符"
original_files = {}
for name in ("execution.json", "environment.json", "stdout.txt", "stderr.txt", "extraction.json"):
    content = (run / name).read_bytes()
    original_files[name] = {"sha256": digest(content), "utf8": content.decode()}
execution = json.loads(original_files["execution.json"]["utf8"])
assert execution["exit_code"] == 0, "小節工具未成功執行"
assert original_files["stdout.txt"]["utf8"] == expected, "小節工具的輸出不符"
assert original_files["stderr.txt"]["utf8"] == "", "小節工具有錯誤輸出"
assert (run / "fence-1.py").read_text() == code, "實跑程式與凍結程式不符"
save("section-execution.json", {"original_files": original_files})

weather = "今天氣溫很舒適。"
replacement = "可以幫你寫詢問盒主的訊息。"
text_only = code.replace(replacement, weather)
text_output, text_rows = run_code(text_only)
assert text_rows[1]["相關替代"] is True, "改文字不應自動修改常數標籤"
modified = ast.parse(text_only)
second_row = modified.body[0].value.elts[1]
label_position = [key.value for key in second_row.keys].index("相關替代")
second_row.values[label_position] = ast.Constant(False)
ast.fix_missing_locations(modified)
changed_output, changed_rows = run_code(modified)
assert changed_rows[1]["守住邊界"] is True, "人工改標籤後的邊界值不符"
assert changed_rows[1]["相關替代"] is False, "人工改標籤後的替代值不符"

class Sections(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = None
        self.depth = 0
        self.blocks = {}

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        if self.active:
            self.depth += 1
        elif dict(attrs).get("id") in {
            "assume_best_intentions", "do_not_facilitate_illicit_behavior", "refusal_style"
        }:
            self.active = dict(attrs)["id"]
            self.depth = 1
            self.blocks[self.active] = []

    def handle_endtag(self, tag):
        if tag == "div" and self.active:
            self.depth -= 1
            if self.depth == 0:
                self.active = None

    def handle_data(self, data):
        if self.active:
            self.blocks[self.active].append(data)


url = "https://model-spec.openai.com/2025-12-18.html"
curl = ["curl", "-fLsS", "--max-time", "30", url]
response = subprocess.run(curl, capture_output=True, check=True)
html = response.stdout.decode()
assert "Model Spec (2025/12/18)" in html, "原始來源的版本標題不符"
parser = Sections()
parser.feed(html)
blocks = {key: " ".join(" ".join(value).split()) for key, value in parser.blocks.items()}
assert len(blocks) == 3, "原始來源缺少待讀取的段落"
excerpt = "redirecting the user toward relevant, safe, and constructive information without crossing safety boundaries."
assert excerpt in blocks["refusal_style"], "原始來源的短摘錄未找到"
save("source-receipt.json", {
    "url": url,
    "version": "Model Spec (2025/12/18)",
    "retrieved_at_utc": datetime.now(UTC).isoformat(),
    "command_argv": curl,
    "curl_version": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
    "exit_code": response.returncode,
    "response_bytes": len(response.stdout),
    "response_sha256": digest(response.stdout),
    "excerpt": {"locator": "#refusal_style", "text": excerpt},
    "sections_inspected": [
        {"locator": "#assume_best_intentions", "normalized_text_sha256": digest(blocks["assume_best_intentions"].encode()),
         "inspection_note": "原文要求有拒絕依據才拒絕；並以禁止偽造簽名但協助處理保險申請為例，支持保留正當目標的替代方案。"},
        {"locator": "#do_not_facilitate_illicit_behavior", "normalized_text_sha256": digest(blocks["do_not_facilitate_illicit_behavior"].encode()),
         "inspection_note": "禁令涵蓋可操作的協助，不是先寫一句拒絕就能提供同樣協助；原文也允許適當的中性資訊與替代資源。"},
        {"locator": "#refusal_style", "normalized_text_sha256": digest(blocks["refusal_style"].encode()),
         "inspection_note": "通常簡短交代邊界，再提供相關、安全且有用的內容；不要說教。原文保留明示非法意圖時單純拒絕的例外，不能外推成每次拒絕都必須附替代。"}
    ],
    "retention_note": "原始全文只在記憶體讀取；檔案只保存雜湊、定位、短摘錄與審閱者摘要。這是指定日期版本的規範，並非模型行為的實測證明。"
})
save("verification.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-5/verify.py",
    "executed_at_utc": datetime.now(UTC).isoformat(),
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；純 Python，不使用模型"},
    "source_sha256": digest(raw),
    "original_code_sha256": digest(code.encode()),
    "row_count": len(rows),
    "related_alternative_count": sum(row["相關替代"] for row in rows),
    "literal_labels": [[row["守住邊界"], row["相關替代"]] for row in rows_literal],
    "call_targets": calls,
    "expected_stdout": expected,
    "observed_stdout": observed,
    "matches_section_tool_stdout": True,
    "text_replacement_only": {"stdout": text_output, "second_row": text_rows[1]},
    "text_and_manual_label_replacement": {"stdout": changed_output, "second_row": changed_rows[1]},
    "result": "所有精確比較與標籤變更檢查通過；文字替換本身不會自動重標。",
    "limits": "執行只證明確定性的列表與列印行為；相關性、授權與替代方案的適切性仍須人工按情境判斷。"
})
print("凍結程式、小節工具輸出與人工改標籤練習均符合預期；原始來源收據已保存。")
