"""以 curl 讀取原始來源，只保存短摘錄與取得收據。"""

import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from datetime import UTC, datetime


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(raw):
    parser = Text()
    parser.feed(raw)
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


SPECS = [
    {
        "id": "composition",
        "url": "https://proceedings.mlr.press/v80/lake18a.html",
        "locator": "Abstract",
        "start": '<div id="abstract" class="abstract">',
        "end": "</div>",
        "quote": 'Once a person learns the meaning of a new verb "dax,"',
        "note": "閱讀出版者頁面上的完整原始摘要：已知部分在新組合中的使用是要測的能力；摘要不替此屬性抽取任務提供品質證據。",
    },
    {
        "id": "comprehensions",
        "url": "https://docs.python.org/3.13/tutorial/datastructures.html",
        "locator": "list-comprehensions、sets",
        "start": '<section id="list-comprehensions">',
        "end": '<section id="nested-list-comprehensions">',
        "quote": "List comprehensions provide a concise way to create lists.",
        "note": "閱讀 List Comprehensions 的雙 for、if 篩選與等價巢狀迴圈，另讀 Sets 的成員判定、無序與 & 交集範例。",
    },
    {
        "id": "assert",
        "url": "https://docs.python.org/3.13/reference/simple_stmts.html",
        "locator": "the-assert-statement",
        "start": '<section id="the-assert-statement">',
        "end": '<section id="the-pass-statement">',
        "quote": "Assert statements are a convenient way to insert debugging assertions into a program:",
        "note": "閱讀 assert 與 __debug__ 的等價條件及 -O 限制；本次一般解譯器沒有開啟最佳化。",
    },
    {
        "id": "grouped-data",
        "url": "https://scikit-learn.org/stable/modules/cross_validation.html",
        "locator": "cross-validation-iterators-for-grouped-data、group-k-fold",
        "start": '<section id="cross-validation-iterators-for-grouped-data">',
        "end": '<section id="stratifiedgroupkfold">',
        "quote": "all the samples in the validation fold come from groups that are not represented at all in the paired training fold.",
        "note": "閱讀相依樣本以 groups 隔開及 GroupKFold 的說明；同屬性組合的措辭變體應共用 group，是本節測量目標對此原則的具體應用。",
    },
]


def main():
    receipt = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-12/read-original-sources.py",
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "sources": [],
    }
    for spec in SPECS:
        command = ["curl", "-fLsS", "--proto", "=https", spec["url"]]
        raw = subprocess.check_output(command)
        doc = raw.decode("utf-8")
        start = doc.index(spec["start"])
        end = doc.index(spec["end"], start)
        content = plain(doc[start:end])
        if spec["id"] == "comprehensions":
            a = doc.index('<section id="sets">')
            b = doc.index('<section id="dictionaries">', a)
            content += "\n" + plain(doc[a:b])
        assert spec["quote"] in content
        version = plain(re.search(r"<title>(.*?)</title>", doc, re.S)[1])
        item = {
            "id": spec["id"],
            "url": spec["url"],
            "command_argv": command,
            "exit_code": 0,
            "document_sha256": hashlib.sha256(raw).hexdigest(),
            "document_bytes": len(raw),
            "page_title": version,
            "locator": spec["locator"],
            "excerpt": spec["quote"],
            "inspection_note": spec["note"],
        }
        receipt["sources"].append(item)
        print(spec["id"], version)
        print(content[:10000])
    result_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/main/docs/course-experiments/results/sft.json"
    result_command = ["curl", "-fLsS", "--proto", "=https", result_url]
    result_raw = subprocess.check_output(result_command)
    local_raw = Path("docs/course-experiments/results/sft.json").read_bytes()
    assert result_raw == local_raw
    result = json.loads(result_raw)
    receipt["linked_experiment_record"] = {
        "url": result_url, "command_argv": result_command, "exit_code": 0,
        "sha256": hashlib.sha256(result_raw).hexdigest(), "matches_local_bytes": True,
        "selected_output": {k: result[k] for k in ("experiment_id", "revision", "seed", "gpu", "elapsed_seconds", "timing_scope")},
    }
    print(json.dumps(receipt["linked_experiment_record"], ensure_ascii=False))
    output = Path(__file__).with_name("original-sources.json")
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("短摘錄與取得收據已保存", output.as_posix())


if __name__ == "__main__":
    main()
