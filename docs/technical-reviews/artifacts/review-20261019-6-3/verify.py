"""核對 tokenizer 的設定、算例、空白分段及固定實驗的資料準備。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _BPE, _deduplicate_text, _utf8_prefix
from tiny_perceptron.data import SPECIALS, ByteTokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_tokenizer(texts, size=280, prefix=False, specials=()):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=prefix)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(specials), show_progress=False,
    ))
    return tok


def sample(tok, text):
    encoded = tok.encode(text)
    return {
        "text": text, "utf8_bytes": len(text.encode("utf-8")),
        "token_count": len(encoded.ids), "ids": encoded.ids,
        "tokens": encoded.tokens, "restored": tok.decode(encoded.ids),
        "same": tok.decode(encoded.ids) == text,
    }


def main():
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-3/verify.py",
        "environment": {
            "python": sys.version, "tokenizers": tokenizers.__version__,
            "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(),
        },
    }
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/6.3.md"
    body = frozen.read_text()
    code = body.split("```python\n", 1)[1].split("```", 1)[0]
    chapter = (ROOT / "course/chapters/06.md").read_text()
    preceding = re.search(r"(?ms)^## 6\.2 .*?(?=^## )", chapter)[0]
    preceding_code = preceding.split("```python\n", 1)[1].split("```", 1)[0]
    preceding_stdout = io.StringIO()
    with contextlib.redirect_stdout(preceding_stdout):
        exec(preceding_code, {})
    result["preceding_handwritten_example"] = {
        "section_sha256": sha(preceding.encode()), "code": preceding_code,
        "stdout": preceding_stdout.getvalue(),
    }
    tok = make_tokenizer(["貓看狗。", "a cat sees a dog."])
    state = json.loads(tok.to_str())
    result["short_example"] = {
        "normalizer": state["normalizer"], "vocab_size": tok.get_vocab_size(),
        "alphabet_size": len(pre_tokenizers.ByteLevel.alphabet()),
        "merge_count": len(state["model"]["merges"]),
        "samples": [sample(tok, x) for x in ["貓看狗。", "小鳥🙂new", "未見字🦊"]],
        "training_chunks_token_counts": {
            text: [len(tok.encode(tok.decoder.decode([chunk])).ids)
                   for chunk, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
            for text in ["貓看狗。", "a cat sees a dog."]
        },
    }
    result["pretokenizer"] = []
    for text in ["a cat", "a  cat", "a\ncat", "abc123", " ＡA\t!\n "]:
        pieces = tok.pre_tokenizer.pre_tokenize_str(text)
        result["pretokenizer"].append({
            "text": text, "visible_pieces": pieces,
            "decoded_pieces": [tok.decoder.decode([chunk]) for chunk, _ in pieces],
            "sample": sample(tok, text),
        })
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            exec(code.replace("add_prefix_space=False", "add_prefix_space=True"), {})
    except AssertionError:
        assertion = "AssertionError"
    else:
        assertion = "沒有斷言失敗"
    prefixed = make_tokenizer(["貓看狗。", "a cat sees a dog."], prefix=True)
    result["prefix_exercise"] = {
        "stdout": captured.getvalue(), "exception": assertion,
        "samples": [sample(prefixed, x) for x in ["貓看狗。", "小鳥🙂new", " 已有空白"]],
    }
    result["byte_tokenizer"] = [
        {"text": t, "ids": ByteTokenizer().encode(t),
         "restored": ByteTokenizer().decode(ByteTokenizer().encode(t)),
         "same": t == ByteTokenizer().decode(ByteTokenizer().encode(t))}
        for t in ["", "貓看狗。", "小鳥🙂new", "未見字🦊", " 小鳥🙂\nnew ", "<user>"]
    ]
    course_files = sorted((ROOT / "course/chapters").glob("*.md"))
    result["byte_tokenizer_usage"] = [
        {"path": str(p.relative_to(ROOT)), "sha256": sha(p.read_bytes()),
         "occurrences": p.read_text().count("ByteTokenizer")}
        for p in course_files if "ByteTokenizer" in p.read_text()
    ]
    other = make_tokenizer(["狗看貓。", "a dog sees a cat."], size=277)
    result["vocabulary_binding"] = {
        "sizes": [tok.get_vocab_size(), other.get_vocab_size()],
        "different_id_meanings": [
            {"id": i, "first": tok.id_to_token(i), "retrained": other.id_to_token(i)}
            for i in range(min(tok.get_vocab_size(), other.get_vocab_size()))
            if tok.id_to_token(i) != other.id_to_token(i)
        ],
        "scope": "展示變更訓練語料時相同 ID 可變義；不聲稱同語料、同版本的重跑必定不同。",
    }
    record_path = ROOT / "docs/course-experiments/results/tokenizer.json"
    record = json.loads(record_path.read_bytes())
    result["recorded_experiment"] = {
        "path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path.read_bytes()),
        "revision": record["revision"], "seed": record["seed"],
        "python_version": record["python_version"], "torch_version": record["torch_version"],
        "gpu": record["gpu"], "data": record["results"]["data"],
        "roundtrip": record["results"]["roundtrip"],
        "bpe_vocab_size": record["results"]["runs"]["bpe512"]["vocab_size_including_8_specials"],
        "tokenizer_file": record["results"]["runs"]["bpe512"]["tokenizer_file"],
        "recorded_tokenizer_artifact": next(
            x for x in record["artifacts"] if x["path"] == "tokenizer-bpe512.json"
        ),
        "newline_count": record["results"]["roundtrip"][1]["text"].count("\n"),
        "literal_slash_n_count": record["results"]["roundtrip"][1]["text"].count("\\n"),
    }
    result["experiment_cli_help"] = subprocess.run(
        [sys.executable, "-m", "scripts.course_experiments.run", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
    raw = []
    fetched = []
    for identifier, basename in [
        ("tinystories", "tinystories-train-512.jsonl"),
        ("chinese-poetry", "chinese-classical-train-365.jsonl"),
    ]:
        asset = next(x for x in manifest["assets"] if x["id"] == identifier)
        url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/" + asset["archive"]
        archive = subprocess.run(["curl", "-fsSL", url], capture_output=True, check=True).stdout
        assert len(archive) == asset["archive_bytes"] and sha(archive) == asset["archive_sha256"]
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            entry = next(x for x in asset["files"] if x["path"].endswith("/" + basename))
            content = package.extractfile(entry["path"]).read()
        assert sha(content) == entry["sha256"]
        rows = [json.loads(line) for line in content.splitlines()][:96]
        raw.extend(rows)
        fetched.append({"url": url, "accessed_on": "2026-10-20", "bytes": len(archive),
                        "sha256": sha(archive), "member": entry["path"],
                        "member_sha256": sha(content), "selected_records": len(rows)})
    complete = split_records(_deduplicate_text(raw), seed=42)
    parts = {
        name: [{**row, "text": _utf8_prefix(row["text"], 256),
                "complete_text_sha256": sha(row["text"].encode())} for row in rows]
        for name, rows in complete.items()
    }
    split_receipts = {}
    for name, rows in parts.items():
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        expected = record["results"]["data"][name]
        split_receipts[name] = {
            "records": len(rows), "sha256": sha(payload),
            "matches_record": sha(payload) == expected["sha256"],
            "max_utf8_bytes": max(len(row["text"].encode()) for row in rows),
        }
        assert sha(payload) == expected["sha256"]
    rebuilt = make_tokenizer([row["text"] for row in parts["train"]], 512, specials=SPECIALS)
    saved_path = Path(__file__).with_name("tokenizer-save-check.json")
    rebuilt.save(str(saved_path))
    saved_raw = saved_path.read_bytes()
    saved_state = json.loads(saved_raw)
    saved_path.unlink()
    wrapper = _BPE(rebuilt)
    roundtrips = []
    for row in record["results"]["roundtrip"]:
        text = row["text"]
        ids = wrapper.encode(text)
        restored = wrapper.decode(ids)
        roundtrips.append({"text": text, "ids": ids, "restored": restored,
                           "same": restored == text,
                           "contains_control_id": any(i in wrapper.special_ids for i in ids)})
        assert restored == text
    family_sets = {name: {row["family"] for row in rows} for name, rows in parts.items()}
    result["data_preparation_reproduction"] = {
        "downloads": fetched, "splits": split_receipts,
        "family_intersections": {a + "/" + b: len(family_sets[a] & family_sets[b])
                                 for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]},
        "training_input_records": len(parts["train"]), "vocab_size": rebuilt.get_vocab_size(),
        "saved_json": {
            "sha256": sha(saved_raw), "bytes": len(saved_raw),
            "matches_recorded_tokenizer": sha(saved_raw) == result["recorded_experiment"]["recorded_tokenizer_artifact"]["sha256"],
            "vocab_size": len(saved_state["model"]["vocab"]),
            "merge_count": len(saved_state["model"]["merges"]),
        },
        "special_token_ids": {x: rebuilt.token_to_id(x) for x in SPECIALS},
        "roundtrip": roundtrips,
        "scope": "只重現資料切分與 tokenizer 訓練；沒有執行語言模型訓練或 GPU 效能量測。",
    }
    source = ast.parse((ROOT / "scripts/course_experiments/text.py").read_text())
    function = next(x for x in source.body if isinstance(x, ast.FunctionDef) and x.name == "run_tokenizer")
    iterator_call = next(x for x in ast.walk(function)
                         if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                         and x.func.attr == "train_from_iterator")
    result["training_iterator_source"] = ast.unparse(iterator_call.args[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
