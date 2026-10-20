"""核對文字指紋、改字練習、固定資料包與原始來源收據。"""

import ast
import hashlib
import html
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tiny_perceptron.data import SPECIALS, fingerprint
from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _deduplicate_text
import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def curl(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    return subprocess.check_output(command)


def write(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def function_source(path, name):
    text = (ROOT / path).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


docs = ["貓  看狗", "貓 看狗", "狗 看貓"]
normalized = [" ".join(text.split()) for text in docs]
hashes = [fingerprint(text) for text in docs]
assert normalized == ["貓 看狗", "貓 看狗", "狗 看貓"]
assert len(set(hashes)) == 2 and hashes[0] == hashes[1]
assert all(len(h) == 64 and re.fullmatch("[0-9a-f]{64}", h) for h in hashes)
changed = [docs[0], "鳥 看狗", docs[2]]
changed_hashes = [fingerprint(text) for text in changed]
try:
    assert changed_hashes[0] == changed_hashes[1]
except AssertionError:
    old_assert_failed = True
else:
    old_assert_failed = False
assert old_assert_failed and len(set(changed_hashes)) == 3
assert changed_hashes[0] != changed_hashes[1]

encoded = "貓 看狗".encode()
api_cases = {}
for value in [encoded, bytearray(encoded), memoryview(encoded), "貓 看狗"]:
    try:
        digest = hashlib.sha256(value).hexdigest()
        api_cases[type(value).__name__] = {"accepted": True, "digest": digest}
    except TypeError as error:
        api_cases[type(value).__name__] = {"accepted": False, "error": str(error)}
assert all(api_cases[t]["accepted"] for t in ["bytes", "bytearray", "memoryview"])
assert not api_cases["str"]["accepted"]

custom = [{"text": "  貓\t看狗\n"}, {"text": "貓 看狗"}, {"text": "狗 看貓"}, {"text": "鳥 看狗"}]
dedup = _deduplicate_text(custom)
parts = split_records(dedup, seed=42)
assert dedup[0]["text"] == custom[0]["text"]
assert len(dedup) == 3 and all(row["family"] == fingerprint(row["text"]) for row in dedup)
assert len({row["family"] for rows in parts.values() for row in rows}) == 3
examples = text_examples([dedup[0]], mode="text", max_length=128)
input_ids, labels = examples[0]
assert [token - len(SPECIALS) for token in labels.tolist()[:-1]] == list(custom[0]["text"].encode())

manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
recorded = json.loads((ROOT / "docs/course-experiments/results/real_text.json").read_bytes())
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
assets = {}
for identifier, basename in [("tinystories", "tinystories-train-512.jsonl"), ("chinese-poetry", "chinese-classical-train-365.jsonl")]:
    asset = next(a for a in manifest["assets"] if a["id"] == identifier)
    url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{revision}/{asset['archive']}"
    raw = curl(url)
    assert sha(raw) == asset["archive_sha256"] and len(raw) == asset["archive_bytes"]
    entry = next(e for e in asset["files"] if e["path"].endswith('/' + basename))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith('/' + basename) or m.name == basename]
        assert len(members) == 1
        content = archive.extractfile(members[0]).read()
    assert sha(content) == entry["sha256"] and len(content) == entry["bytes"]
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    unique = _deduplicate_text(rows)
    split = split_records(unique, seed=42)
    reference = recorded["results"]["runs"][identifier]
    assert len(rows) == reference["source_records"]
    assert len(unique) == reference["deduplicated_records"]
    assert [r["text"] for r in unique] == [r["text"] for r in rows]
    actual_splits = {}
    for name, group in split.items():
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group).encode()
        assert sha(data) == reference["data"][name]["sha256"]
        actual_splits[name] = {"records": len(group), "sha256": sha(data)}
    assets[identifier] = {"url": url, "archive_sha256": sha(raw), "archive_bytes": len(raw),
                          "jsonl_sha256": sha(content), "source_records": len(rows),
                          "deduplicated_records": len(unique), "original_text_preserved": True,
                          "split_manifest_matches": actual_splits}

source_specs = [
    ("python-hashlib", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/library/hashlib.rst", "CPython v3.13.16", "Hash algorithms; Constructors; hash.hexdigest", "bytes-like objects <bytes-like object>", "建構子與 update 接收 bytes-like；hexdigest 的長度為 digest 的兩倍。"),
    ("python-types", "https://docs.python.org/3.13/library/stdtypes.html", "Python 3.13.16 documentation", "str.split; str.join; str.encode; Set Types", "runs of consecutive whitespace are regarded as a single separator", "split(None) 合併連續空白並移除邊緣空白；join 插入指定分隔；encode 預設 UTF-8；set 儲存相異值。"),
    ("python-lexical", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/reference/lexical_analysis.rst", "CPython v3.13.16", "Indentation; String and Bytes literals", "Indentation cannot be split over multiple physical lines", "縮排決定 statement 分組，字串字面值包含自己的內容；不可把全文空白合併當成普遍保義規則。"),
    ("nist-sha", "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf", "FIPS PUB 180-4, August 2015", "Explanation; Introduction; Bit Strings and Integers; SHA-256; Truncation of a Message Digest", "Any change to a message will, with a very high probability, result in a different message digest.", "使用 curl 取得 PDF bytes 並記 hash；另以 web PDF 文字檢視讀取 Explanation、Introduction、Bit Strings and Integers。雜湊是單向摘要，SHA-256 輸出 256 bits，hex 每字元表示 4 bits。"),
    ("dedup-paper", "https://aclanthology.org/2022.acl-long.577.pdf", "ACL 2022 final, pp. 8424–8445, DOI 10.18653/v1/2022.acl-long.577", "Introduction; Train / Test Set Leakage", "This train-test set overlap not only causes researchers to over-estimate model accuracy", "使用 curl 取得 PDF bytes 並記 hash；另以 web PDF 文字檢視讀取 Introduction、Train / Test Set Leakage。論文支持重複跨訓練與驗證集合使評估偏高，不替本課資料去重結果作證。"),
    ("layernorm-paper", "https://arxiv.org/abs/1607.06450", "arXiv:1607.06450v1, 21 July 2016", "Abstract", "computing the mean and variance used for normalization", "原始摘要以同一層各單元的數值計算 mean/variance，與文字空白轉換是不同運算。"),
]
sources = []
for identifier, url, version, locator, excerpt, note in source_specs:
    raw = curl(url)
    text = raw.decode(errors="replace")
    if not raw.startswith(b"%PDF"):
        if identifier == "python-types":
            text = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", text)))
        assert excerpt in text, identifier
    sources.append({"id": identifier, "url": url, "version": version, "accessed_on": "2026-10-20",
                    "retrieval_command": ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url],
                    "full_source_sha256": sha(raw), "full_source_bytes": len(raw),
                    "locator": locator, "excerpt": excerpt, "inspection_note": note})

write("source-receipts.json", {"sources": sources, "full_text_storage": "原文僅在記憶體，不把第三方全文寫入 repo。"})
cli_command = [sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", "real_text"]
cli_output = subprocess.check_output(cli_command, cwd=ROOT, text=True)
assert set(cli_output.splitlines()) == {"assets/training/tinystories-v1.tar.gz", "assets/training/chinese-poetry-v1.tar.gz"}
repository_paths = ["tiny_perceptron/data.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py", "docs/course-experiments/plan.json", "docs/course-experiments/results/real_text.json", "assets/training/manifest.json", "course/training.md"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-11/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "normalized": normalized, "hashes": hashes,
    "changed": {"docs": changed, "hashes": changed_hashes, "distinct": len(set(changed_hashes)), "old_assert_failed": old_assert_failed, "new_assert_passed": True},
    "api_cases": api_cases, "default_encode_equals_utf8": encoded == "貓 看狗".encode("utf-8"),
    "whitespace_boundary_case": {"input": custom[0]["text"], "normalized": " ".join(custom[0]["text"].split()), "original_preserved_in_dedup": dedup[0]["text"], "training_labels": labels.tolist(), "original_bytes": list(custom[0]["text"].encode())},
    "assets": assets,
    "cli": {"command_argv": cli_command, "stdout": cli_output, "exit_code": 0},
    "derivations": {"hex_length": "SHA-256 為 256 bits；hex 每字元表示 4 bits，256/4=64；0–9 加 a–f 共 10+6=16 種。", "duplicate_sampling": "均勻抽記錄時，三筆中的『貓 看狗』機率 2/3；完整去重剩兩筆時為 1/2。增加副本改變內容的抽樣分配；非均勻 sampler 另論。", "truncation": "只保留 k 個 hex 字元的值域為 16**k；完整值域為 16**64。前綴相同的條件較弱，不能保證唯一。"},
    "repo_revision": revision,
    "repository_sha256": {name: sha((ROOT / name).read_bytes()) for name in repository_paths},
    "inspected_functions": {path + ':' + name: function_source(path, name) for path, name in [("tiny_perceptron/data.py", "fingerprint"), ("scripts/course_experiments/text.py", "_deduplicate_text"), ("scripts/course_experiments/text.py", "run_real_text"), ("scripts/course_experiments/common.py", "split_records"), ("scripts/course_experiments/common.py", "text_examples")]},
    "result": "所有執行斷言通過；sha256 接受 bytes、bytearray、memoryview，故『只接受 bytes』被反例推翻。固定資料去重與切分 hash 符合已存紀錄；沒有執行訓練。"
}
write("verification.json", result)
print(json.dumps(result, ensure_ascii=False, indent=2))
