"""以 curl 讀官方原文，只保存短摘錄與取得收據。"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


sources = [
    ("json", "https://docs.python.org/3.13/library/json.html", "Python 3.13.16", ["Performs the following translations", "parse_float is an optional", "By default, this module does not raise an exception"], "Subclass of ValueError"),
    ("types", "https://docs.python.org/3.13/library/stdtypes.html", "Python 3.13.16", ["Booleans are a subtype", "iter(d)", "dict.values()"], "Booleans are a subtype of integers."),
    ("functions", "https://docs.python.org/3.13/library/functions.html", "Python 3.13.16", ["any ( iterable", "isinstance ( object", "type ( object"], "Return True if any element of the iterable is true."),
    ("exceptions", "https://docs.python.org/3.13/tutorial/errors.html", "Python 3.13.16", ["First, the try clause", "The sole argument to raise"], "If no exception occurs, the except clause is skipped"),
    ("math", "https://docs.python.org/3.13/library/math.html", "Python 3.13.16", ["neither an infinity nor a NaN", "Return True if x is neither"], "Check if x is neither an infinity nor a NaN"),
    ("schema", "https://json-schema.org/understanding-json-schema/reference/object", "Understanding JSON Schema，讀取當日網站版本", ["properties keyword", "The required keyword", "The additionalProperties keyword"], "The value of properties is an object"),
    ("allowlist", "https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html", "OWASP Cheat Sheet Series，讀取當日網站版本", ["Allowlist vs Denylist", "Parse Safely, Then Validate", "For JSON, explicitly configure"], "listing a property alone neither requires it nor rejects unknown fields"),
]
receipts = []
for identifier, url, version, needles, quote in sources:
    command = ["curl", "-fsSL", "--max-time", "30", url]
    result = subprocess.run(command, check=True, capture_output=True)
    parser = PlainText()
    parser.feed(result.stdout.decode())
    plain = " ".join(" ".join(parser.parts).split())
    contexts = []
    for needle in needles:
        position = plain.rfind(needle) if needle in ("Allowlist vs Denylist", "Parse Safely, Then Validate") else plain.find(needle)
        if position < 0:
            raise ValueError(f"官方原文找不到定位文字：{identifier}: {needle}")
        contexts.append({"locator": needle, "text": plain[max(0, position - 100):position + 800]})
    if quote not in plain:
        raise ValueError(f"官方原文找不到短摘錄：{identifier}")
    receipts.append({"id": identifier, "url": url, "command_argv": command, "version": version, "accessed_at": datetime.now(timezone.utc).isoformat(), "original_sha256": hashlib.sha256(result.stdout).hexdigest(), "locators": needles, "short_excerpt": quote, "checked_original": True})
    print(json.dumps({"id": identifier, "contexts": contexts}, ensure_ascii=False))
Path(__file__).with_name("official-source-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
