"""以 curl 讀官方原文，只保留讀取收據與支持主張的小段摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class TextReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


SOURCES = [
    {
        "url": "https://docs.python.org/3.13/library/json.html",
        "locator": "json.loads、json.load 的 object_pairs_hook；Repeated Names Within an Object",
        "version_marker": "Python 3.13.16 documentation",
        "excerpts": ["ordered list of pairs", "it ignores all but the last name-value pair for a given name"],
        "inspection_note": "讀取 loads/load 解析介面、JSONDecodeError、JSON 轉 Python 型別表與重複欄位段落；hook 能取得未被覆蓋的成對欄位。本課 _unique_object 自己拒絕重複欄位；標準 loads 預設並不拒絕。",
    },
    {
        "url": "https://docs.python.org/3.13/reference/compound_stmts.html",
        "locator": "Function definitions 的 annotations 段落；The try statement",
        "version_marker": "Python 3.13.16 documentation",
        "excerpts": ["Otherwise, they are evaluated when the function definition is executed."],
        "inspection_note": "本專案 .python-version 為 3.13；教材函式沒有 future annotations。在這個條件下定義時會解析回傳註記的名稱。except 在相符例外發生時執行。",
    },
    {
        "url": "https://docs.python.org/3.13/library/typing.html",
        "locator": "頁首 Note：runtime enforcement",
        "version_marker": "Python 3.13.16 documentation",
        "excerpts": ["The Python runtime does not enforce function and variable type annotations."],
        "inspection_note": "讀頁首 Note 與函式範例：型別註記本身不執行參數型別驗證，教材的安全限制來自實際 if 檢查。",
    },
    {
        "url": "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview",
        "locator": "Tool use with Claude；How tool use works 的 client tool 往返說明",
        "version_marker": "Tool use with Claude",
        "excerpts": ["Claude determines when to call a tool", "your code runs the lookup"],
        "inspection_note": "核對 client tool 的一般分工：模型提出結構化呼叫，應用程式執行並送回 tool_result，模型再生成回答；工具描述與使用者請求影響選擇。只支持這個方法分工，不把 Claude API 格式或品質推論到本專案。",
    },
    {
        "url": "https://docs.python.org/3.14/reference/compound_stmts.html",
        "locator": "Annotations 的 Changed in version 3.14",
        "version_marker": "Python 3.14.8 documentation",
        "excerpts": ["Annotations are now lazily evaluated by default."],
        "inspection_note": "確認定義時求值不能外推到 Python 3.14；此節以專案釘選的 Python 3.13 為成立範圍，沒有把這一行當跨版本語言契約。",
    },
]


def main():
    receipts = []
    for source in SOURCES:
        command = ["curl", "-f", "-s", "-S", "-L", "--max-time", "40", source["url"]]
        result = subprocess.run(command, capture_output=True, check=True)
        parser = TextReader()
        parser.feed(result.stdout.decode("utf-8"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        assert source["version_marker"] in text, source["url"]
        for excerpt in source["excerpts"]:
            assert excerpt in text, excerpt
        receipts.append({
            **source, "command_argv": command, "returncode": result.returncode,
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "original_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "original_bytes": len(result.stdout), "checked_original": True,
            "full_original_saved": False,
        })
    path = Path(__file__).with_name("official-source-receipts.json")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sources_read": len(receipts), "full_originals_saved": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
