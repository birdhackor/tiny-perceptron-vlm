"""讀取原始來源，只保存收據與少量支持主張的摘錄。"""

import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


SOURCES = [
    ("brown", "https://arxiv.org/html/2005.14165v4", "arXiv:2005.14165v4，2020-07-22", "Approach", "no gradient updates", "no gradient updates"),
    ("order", "https://aclanthology.org/2022.acl-long.556/", "ACL 2022，正式出版版", "Abstract", "the order in which", "the order in which the samples are provided"),
    ("utf8", "https://www.rfc-editor.org/rfc/rfc3629.txt", "RFC 3629，2003-11", "UTF-8 definition", "0000 0000-0000 007F", "0000 0000-0000 007F | 0xxxxxxx"),
    ("types", "https://docs.python.org/3.13/library/stdtypes.html", "Python 3.13.16", "str.join / str.endswith", "separator between elements", "The separator between elements is the string providing this method."),
    ("all", "https://docs.python.org/3.13/library/functions.html", "Python 3.13.16", "all", "all elements of the iterable", "Return True if all elements of the iterable are true"),
    ("expressions", "https://docs.python.org/3.13/reference/expressions.html", "Python 3.13.16", "Generator expressions", "generator expression yields", "A generator expression yields a new generator object."),
]


def fetch(item):
    key, url, version, locator, needle, excerpt = item
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    decoded = raw.decode("utf-8")
    text = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    text = re.sub(r"\s+", " ", text)
    # 定位失敗會停止；不以猜測摘錄替代實讀原文。
    index = text.casefold().find(needle.casefold())
    if index < 0:
        raise ValueError("找不到原始來源定位：" + key)
    print(json.dumps({"source": key, "version": version, "context": text[max(0, index-200):index+1700]}, ensure_ascii=False))
    return {
        "id": key,
        "url": url,
        "version": version,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "accessed_on": "2026-10-20",
        "locator": locator,
        "command_argv": command,
        "exit_code": result.returncode,
        "original_bytes": len(raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpt": excerpt,
        "full_text_saved": False,
    }


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, SOURCES))
    target = Path(__file__).with_name("source-receipts.json")
    with target.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
