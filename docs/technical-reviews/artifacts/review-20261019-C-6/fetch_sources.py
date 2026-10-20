"""以 curl 讀取原始來源，只保存定位附近的短摘錄與取得收據。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
REQUESTS = [
    (
        "python_re",
        "https://docs.python.org/3.13/library/re.html",
        [
            ("re.fullmatch", "If the whole string matches", 265),
            ("Match objects", "Match objects always have a boolean value", 200),
            ("Regular Expression Syntax: +", "Causes the resulting RE to match 1 or more", 160),
            ("Regular Expression Syntax: ?", "Causes the resulting RE to match 0 or 1", 160),
            ("Regular Expression Syntax: []", "Ranges of characters can be indicated", 235),
            ("raw string notation", "backslashes are not handled in any special way", 215),
        ],
    ),
    (
        "python_types",
        "https://docs.python.org/3.13/library/stdtypes.html",
        [
            ("str.strip", "Return a copy of the string with the leading and trailing characters removed", 365),
            ("str.split", "If sep is not specified or is None", 390),
            ("Boolean Type", "Booleans are a subtype of integers", 180),
        ],
    ),
    (
        "python_builtins",
        "https://docs.python.org/3.13/library/functions.html",
        [
            ("float", "If the argument is a string, it should contain", 180),
            ("int", "Return an integer object constructed from", 360),
            ("repr", "Return a string containing a printable representation", 220),
            ("str", "class str (", 220),
        ],
    ),
    (
        "python_conditional",
        "https://docs.python.org/3.13/reference/expressions.html",
        [("Conditional expressions", "The expression x if C else y", 380)],
    ),
    (
        "rule_rewards",
        "https://arxiv.org/html/2501.12948v1",
        [
            ("Reward Modeling", "The reward is the source of the training signal", 210),
            ("Accuracy rewards", "in the case of math problems with deterministic results", 275),
            ("Format rewards", "Format rewards :", 225),
            ("Reward Modeling", "We do not apply the outcome or process neural reward model", 170),
        ],
    ),
    (
        "specification_gaming",
        "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
        [
            ("Definition", "Specification gaming is a behaviour", 155),
            ("RL task example", "a reinforcement learning agent can find a shortcut", 180),
        ],
    ),
]


def plain(raw):
    raw = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


receipts = []
for identifier, url, queries in REQUESTS:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = plain(raw.decode("utf-8"))
    snippets = []
    for locator, phrase, length in queries:
        position = text.find(phrase)
        if position < 0:
            raise RuntimeError(f"來源定位失敗：{identifier} / {locator}")
        snippets.append({"locator": locator, "excerpt": text[position : position + length]})
    title = re.search(r"<title[^>]*>(.*?)</title>", raw.decode("utf-8"), flags=re.S | re.I)
    receipts.append(
        {
            "id": identifier,
            "url": url,
            "command_argv": command,
            "exit_code": completed.returncode,
            "accessed_at_utc": datetime.now(UTC).isoformat(),
            "title": plain(title[1]) if title else identifier,
            "retrieved_bytes": len(raw),
            "retrieved_sha256": hashlib.sha256(raw).hexdigest(),
            "snippets": snippets,
        }
    )

output = DESTINATION / "source-excerpts.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump({"retrieval": "curl；第三方全文僅在記憶體解析，未寫入 repo", "sources": receipts}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"sources": len(receipts), "output": str(output.relative_to(Path.cwd()))}, ensure_ascii=False))
