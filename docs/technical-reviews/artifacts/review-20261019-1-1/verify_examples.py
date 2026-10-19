"""核對字表、可還原性、未知鍵、圖中文字與凍結版本。"""

import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET


ARTIFACTS = Path(__file__).resolve().parent
ROOT = ARTIFACTS.parents[3]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, identifier):
    raw = (ROOT / path).read_bytes()
    match = re.search(rb"(?m)^## " + re.escape(identifier.encode()) + rb" ", raw)
    end = re.search(rb"(?m)^## ", raw[match.end():])
    return raw[match.start():match.end() + end.start()] if end else raw[match.start():]


frozen_path = "docs/course-review-20261019/freeze-01/sources/1.1.md"
frozen = (ROOT / frozen_path).read_bytes()
assert hashlib.sha256(frozen).hexdigest() == "bb05afdf5bf320d19628e7f0a3f9a14076e9ae28b40c5fde924d441ff87e8481"
assert section("course/chapters/01.md", "1.1") == frozen
assert digest("docs/course-review-20261019/freeze-01.json") == "71bd28f633f6b1ae3fce6e6279239f261893b42055e70974b142d25b9a45b5c0"
figure = "course/figures/rewrite-01-character-ids.svg"
assert digest(figure) == "61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad"
png = "outputs/course-review-20261019/figure-renders/61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad.png"
assert digest(png) == "5fc177e37fd86b70bf1ad336b07c6d7c95ad5440e6bf74f01d5707d264ed7cfb"
code = re.search(r"```python\n(.*?)```", frozen.decode(), re.S).group(1)


def execute(source):
    scope = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(source, frozen_path, "exec"), scope)
    return scope, stdout.getvalue()


scope, stdout = execute(code)
expected_chars = ["。", "狗", "看", "貓", "，"]
expected_ids = [3, 2, 1, 4, 1, 2, 3, 0]
assert scope["chars"] == expected_chars
assert scope["ids"] == expected_ids
expected_stdout = str(expected_chars) + "\n" + str(expected_ids) + "\n貓看狗，狗看貓。\n"
assert stdout == expected_stdout
pairs = list(enumerate(scope["chars"]))
assert pairs == [(0, "。"), (1, "狗"), (2, "看"), (3, "貓"), (4, "，")]
reverse_code = code.replace("chars = sorted(set(text))", "chars = sorted(set(text), reverse=True)")
reverse_scope, reverse_stdout = execute(reverse_code)
assert reverse_scope["chars"] == ["，", "貓", "看", "狗", "。"]
assert reverse_scope["ids"] == [1, 2, 3, 0, 3, 2, 1, 4]
assert reverse_scope["restored"] == scope["text"]
try:
    unknown = [scope["to_id"][char] for char in "貓看鳥"]
except KeyError as error:
    unknown_result = {"type": type(error).__name__, "args": error.args, "message": str(error)}
else:
    raise AssertionError("未收錄的字應觸發 KeyError")
assert unknown_result["args"] == ("鳥",)
ord_values = {char: ord(char) for char in expected_chars}
assert list(ord_values.values()) == [12290, 29399, 30475, 35987, 65292]
permutations = 0
for chars in itertools.permutations(expected_chars):
    to_id = {char: i for i, char in enumerate(chars)}
    ids = [to_id[char] for char in scope["text"]]
    assert "".join(chars[i] for i in ids) == scope["text"]
    permutations += 1
assert permutations == 120
wrong_table_result = "".join(reverse_scope["chars"][i] for i in expected_ids)
assert wrong_table_result != scope["text"]
silent = io.StringIO()
with contextlib.redirect_stdout(silent):
    exec("assert '貓' == '貓'")
assert silent.getvalue() == ""
try:
    exec("assert '貓' == '狗'")
except AssertionError:
    assert_failure = "AssertionError"
else:
    raise AssertionError("未最佳化執行的錯誤斷言應失敗")
svg = ET.fromstring((ROOT / figure).read_bytes())
texts = [{"x": node.get("x"), "y": node.get("y"), "text": node.text} for node in svg.iter("{http://www.w3.org/2000/svg}text")]
chars = [item["text"] for item in texts if item["y"] in ("181", "401")]
ids = [int(item["text"]) for item in texts if item["y"] in ("314", "534")]
assert chars == list(scope["text"])
assert ids == expected_ids
assert sum(item["text"] == "↕" for item in texts) == 8
course_receipts = {}
for path in ("course/chapters/01.md", "course/chapters/02.md", "course/chapters/06.md", "course/first-steps.md"):
    course_receipts[path] = {"sha256": digest(path)}
for path, identifier, needles in (
    ("course/chapters/01.md", "1.5", ("字表為",)),
    ("course/chapters/01.md", "1.6", ("重新約定 ID",)),
    ("course/chapters/01.md", "1.14", ("字表直接寫成",)),
    ("course/chapters/02.md", "2.1", ("新的列順序",)),
    ("course/chapters/02.md", "2.2", ("先約定 ID",)),
    ("course/chapters/06.md", "6.1", ("第 6 章", "另一邊是字表")),
    ("course/chapters/06.md", "6.3", ("initial_alphabet=", "保存 tokenizer 時")),
    ("course/first-steps.md", "W.2", ("`set(...)`", "`enumerate(...)`", "`\"\".join(...)`")),
    ("course/first-steps.md", "W.7", ("`KeyError` 是",)),
):
    body = section(path, identifier).decode()
    selected = [line for line in body.splitlines() if any(needle in line for needle in needles)]
    assert selected, path + "#" + identifier
    course_receipts[path].setdefault("checked_sections", {})[identifier] = selected
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；Python 內建運算", "unicode_database": unicodedata.unidata_version, "optimization": str(sys.flags.optimize)},
    "source_sha256": hashlib.sha256(frozen).hexdigest(),
    "freeze_and_current_equal": True,
    "stdout": stdout,
    "chars": scope["chars"], "to_id": scope["to_id"], "enumerate": pairs,
    "ids": scope["ids"], "restored": scope["restored"],
    "ord_values": ord_values,
    "unicode_names": {char: unicodedata.name(char) for char in expected_chars},
    "reverse": {"chars": reverse_scope["chars"], "ids": reverse_scope["ids"], "restored": reverse_scope["restored"], "stdout": reverse_stdout},
    "unknown": unknown_result,
    "all_bijections_tested": permutations,
    "wrong_table_decoding": wrong_table_result,
    "assert_success_stdout": silent.getvalue(), "assert_failure": assert_failure,
    "figure": {"path": figure, "sha256": digest(figure), "character_sequence": chars, "id_sequence": ids, "bidirectional_arrows": 8, "png_path": png, "png_sha256": digest(png), "text_elements": texts},
    "course_receipts": course_receipts,
    "result": "所有精確相等檢查通過；未執行模型訓練。",
}
with (ARTIFACTS / "example-verification.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
