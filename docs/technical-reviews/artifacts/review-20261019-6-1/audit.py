"""取得原始來源收據，並核對文字算例與既存生成紀錄。"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import unicodedata
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    specifications = [
        (
            "unicode",
            "https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/",
            "Unicode 17.0.0",
            ["D7 Abstract character:", "D10 Code point:", "D60 Grapheme cluster:",
             "D76 Unicode scalar value:", "D92 UTF-8 encoding form:",
             "Table 3-6", "Table 3-7"],
            "any UTF-8 byte sequence that would otherwise map to code points U+D800..U+DFFF is ill-formed.",
            "抽象字元、碼點與使用者眼中的字不同；UTF-8 對 scalar value 編碼，排除 surrogate 碼點。ASCII、BMP 中文與補充平面 emoji 的編碼長度可由 bit 分布表核對。",
        ),
        (
            "python_str",
            "https://docs.python.org/3.13/library/stdtypes.html",
            "Python 3.13.16 documentation",
            ["Strings are immutable sequences of Unicode code points.", "str. encode", "bytes. decode"],
            "Strings are immutable sequences of Unicode code points.",
            "str 是碼點序列；標準 sequence 的 len 數元素，encode 產生 bytes，bytes 的 len 數 byte。",
        ),
        (
            "python_codecs",
            "https://docs.python.org/3.13/library/codecs.html",
            "Python 3.13.16 documentation",
            ["Error Handlers", "surrogatepass", "U+FFFD"],
            "Otherwise these codecs treat the presence of surrogate code point in str as an error.",
            "strict 為預設；UTF-8 對 str 裡的 surrogate 拒絕編碼，replace 解碼用 U+FFFD；surrogatepass 是另行指定的非預設行為。",
        ),
        (
            "tokenizers",
            "https://huggingface.co/docs/tokenizers/v0.20.3/en/components",
            "Tokenizers v0.20.3 documentation",
            ["Normalizers", "ByteLevel", "Models", "WordLevel", "BPE", "Decoders"],
            "it simply maps input tokens to IDs.",
            "WordLevel 把 token 對應 ID；BPE 可以組成多字或子字單位；ByteLevel 的初始 alphabet 是 256 個 byte 值，支援未見文字；normalizer 可能改變原文，所以解碼可逆性必須附條件。",
        ),
        (
            "embedding",
            "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html",
            "PyTorch 2.14 documentation; local torch 2.14.1",
            ["class torch.nn. Embedding", "num_embeddings", "embedding_dim", "Variables"],
            "the learnable weights of the module of shape (num_embeddings, embedding_dim)",
            "Embedding.weight 有字表大小乘特徵寬度個元素；這裡計的是該表的參數，並非整個模型參數。",
        ),
        (
            "attention",
            "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html",
            "PyTorch 2.14 documentation; local torch 2.14.1",
            ["L , S = query", "attn_weight = query @ key", "attn_weight = torch . softmax"],
            "attn_weight = query @ key . transpose ( - 2 , - 1 ) * scale_factor",
            "官方等價程式用 query 與 key 長度 L、S 建立配對表；完整 self-attention 的 L=S=T，所以每個樣本與 head 有 T² 格，因果遮罩使部分權重為零。不能把格數比例當整體耗時比例。",
        ),
    ]
    results = []
    for identifier, url, version, locators, excerpt, note in specifications:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
        completed = subprocess.run(command, capture_output=True, check=True)
        text = unescape(re.sub(r"<[^>]+>", " ", completed.stdout.decode("utf-8")))
        text = re.sub(r"\s+", " ", text)
        if excerpt not in text:
            raise ValueError(f"找不到指定原文摘錄：{identifier}")
        matched = {locator: locator in text for locator in locators}
        if not all(matched.values()):
            raise ValueError(f"原文定位不符：{identifier}: {matched}")
        results.append({
            "id": identifier, "url": url, "version": version,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "command_argv": command, "exit_code": completed.returncode,
            "original_response_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "response_bytes": len(completed.stdout), "checked_original": True,
            "locators_found": matched, "excerpt": excerpt, "inspection_note": note,
        })
    return {"method": "curl 原文只在記憶體解析；只保存定位、短摘錄與原始回應 SHA-256", "sources": results}


def bundle(directory):
    directory = directory.resolve()
    names = ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"]
    records = {}
    for name in names:
        path = directory / name
        records[name] = {"sha256": sha(path), "content": path.read_text(encoding="utf-8")}
    return {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/06.md#6.1" --execute',
        "tool_output_directory": str(directory.relative_to(ROOT)),
        "copied_files": records,
    }


def probe():
    import torch
    from torch import nn

    from scripts.course_experiments.common import evaluate_lm, split_records
    from scripts.course_experiments.run import experiment_spec
    from scripts.course_experiments.text import _deduplicate_text
    from tiny_perceptron.attention import manual_attention
    from tiny_perceptron.data import SPECIALS, ByteTokenizer, CharTokenizer

    torch.set_num_threads(1)
    tokenizer = ByteTokenizer()
    counts = []
    for text in ["小小貓", "cat", "🙂", "👍🏽", "cat🙂", "貓🙂"]:
        raw = text.encode("utf-8")
        encoded = tokenizer.encode(text)
        counts.append({"text": text, "codepoints": len(text), "utf8_bytes": len(raw),
                       "ord": [ord(c) for c in text], "names": [unicodedata.name(c) for c in text], "bytes": list(raw),
                       "byte_token_ids": encoded, "roundtrip": tokenizer.decode(encoded) == text})

    boundaries = []
    for value in [0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF]:
        text = chr(value)
        boundaries.append({"codepoint": f"U+{value:04X}", "utf8_bytes": len(text.encode("utf-8")),
                           "roundtrip": tokenizer.decode(tokenizer.encode(text)) == text})
    surrogate = chr(0xD800)
    try:
        surrogate.encode("utf-8")
    except UnicodeEncodeError as error:
        surrogate_result = {"codepoint": "U+D800", "len": len(surrogate),
                            "exception": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("預期預設 UTF-8 拒絕 surrogate 碼點")

    attention = []
    for length in (3, 9):
        q = torch.zeros(1, 1, length, 2, dtype=torch.float32)
        v = torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2)
        mask = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
        _, weights = manual_attention(q, q, v, mask)
        attention.append({"tokens": length, "shape": list(weights.shape), "cells": weights.numel(),
                          "visible_pairs": int(mask.sum()), "dtype": str(weights.dtype)})
    embeddings = []
    for size in (256, 264):
        table = nn.Embedding(size, 8)
        embeddings.append({"vocabulary": size, "width": 8, "weight_shape": list(table.weight.shape),
                           "parameters": sum(p.numel() for p in table.parameters())})
    char_tokenizer = CharTokenizer("cat小小貓")
    unseen_ids = char_tokenizer.encode("狗🙂")

    result_path = ROOT / "docs/course-experiments/results/real_text.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    poetry = result["results"]["runs"]["chinese-poetry"]
    sample = poetry["after"]["validation"]["samples"][7]
    asset = next(item for item in result["assets"] if item["id"] == "chinese-poetry")
    archive = ROOT / asset["archive"]
    archive_raw = archive.read_bytes()
    archive_retrieval = {"local_sha256": sha(archive), "local_bytes": len(archive_raw)}
    if archive_raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{result['revision']}/{asset['archive']}"
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
        downloaded = subprocess.run(command, capture_output=True, check=True)
        archive_raw = downloaded.stdout
        archive_retrieval.update(command_argv=command, exit_code=downloaded.returncode,
                                 accessed_on=datetime.now(UTC).date().isoformat(),
                                 method="LFS pointer 的物件只在記憶體讀取；不修改本機資料包。")
    archive_sha = hashlib.sha256(archive_raw).hexdigest()
    assert archive_sha == asset["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as packed:
        member = next(item for item in packed.getmembers() if item.name.endswith("chinese-classical-train-365.jsonl"))
        raw_rows = [json.loads(line) for line in packed.extractfile(member).read().decode("utf-8").splitlines()]
    reconstructed = split_records(_deduplicate_text(raw_rows), seed=result["seed"])
    validation_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reconstructed["validation"]).encode("utf-8")
    validation_sha = hashlib.sha256(validation_bytes).hexdigest()
    assert validation_sha == poetry["data"]["validation"]["sha256"]
    validation_text = reconstructed["validation"][7]["text"]
    assert validation_text.startswith(sample["prompt"])
    generated_ids = sample["generated_ids"]
    assert all(len(SPECIALS) <= value < tokenizer.vocab_size for value in generated_ids)
    raw = bytes(value - len(SPECIALS) for value in generated_ids)
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        decode_error = {"start": error.start, "end": error.end, "reason": error.reason}
    else:
        raise AssertionError("預期生成結尾存在不完整 UTF-8 字元")

    prefix_size = 1 + len(tokenizer.encode(sample["prompt"]))

    class RecordedLogits(nn.Module):
        """回播已保存的選擇，僅驗證評估與生成流程，不重現模型品質。"""

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(max_length=128)

        def forward(self, ids, valid=None, cache=None):
            logits = torch.zeros(*ids.shape, tokenizer.vocab_size)
            index = min(max(ids.shape[1] - prefix_size, 0), len(generated_ids) - 1)
            logits[:, -1, generated_ids[index]] = 1.0
            return {"logits": logits, "cache": None}

    replay_record = {"text": validation_text}
    replay = evaluate_lm(RecordedLogits(), [replay_record], mode="text", tokens=32)["samples"][0]
    assert replay == sample
    help_command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
    help_run = subprocess.run(help_command, cwd=ROOT, capture_output=True, text=True, check=True)
    sources_checked = {}
    for path in ["tiny_perceptron/data.py", "tiny_perceptron/attention.py", "tiny_perceptron/model.py",
                 "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]:
        current = sha(ROOT / path)
        sources_checked[path] = {"sha256": current, "recorded_sha256": result["code_sha256"][path],
                                 "matches_recorded": current == result["code_sha256"][path]}
        assert sources_checked[path]["matches_recorded"]
    completions = [raw + bytes([value]) for value in range(0x80, 0xC0)]
    assert all(candidate.decode("utf-8").startswith(raw[:-2].decode("utf-8")) for candidate in completions)
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-1/audit.py probe",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": sys.platform},
        "counts": counts, "explicit_codepoints": {"c": ord("c"), "貓": ord("貓"), "🙂": ord("🙂")},
        "utf8_boundaries": boundaries, "surrogate_counterexample": surrogate_result,
        "byte_arithmetic": {"bits_per_byte": 8, "possibilities": 2 ** 8, "min": 0, "max": 2 ** 8 - 1},
        "attention": attention, "embeddings": embeddings,
        "tokenizer_vocabulary": {"byte_values": 256, "specials": list(SPECIALS), "actual_vocabulary": tokenizer.vocab_size},
        "unknown_characters": {"input": "狗🙂", "ids": unseen_ids, "decoded": char_tokenizer.decode(unseen_ids)},
        "recorded_result": {
            "path": str(result_path.relative_to(ROOT)), "sha256": sha(result_path),
            "pointer": "results.runs.chinese-poetry.after.validation.samples[7]", "sample": sample,
            "run_environment": {key: result[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
            "denominators": {"seed": result["seed"], "train_records": poetry["data"]["train"]["records"],
                             "validation_records": poetry["data"]["validation"]["records"], "test_records": poetry["data"]["test"]["records"],
                             "training_steps": poetry["training"]["steps"], "saved_validation_samples": len(poetry["after"]["validation"]["samples"]),
                             "inspected_samples": 1, "zero_based_sample_index": 7},
            "prompt_bytes": len(sample["prompt"].encode("utf-8")), "generated_ids_count": len(generated_ids),
            "validation_reconstruction": {"archive": asset["archive"], "archive_sha256": archive_sha,
                                          "archive_bytes": len(archive_raw), "retrieval": archive_retrieval,
                                          "validation_sha256": validation_sha, "matches_recorded": True,
                                          "sample_text_starts_with_prompt": True,
                                          "next_codepoint_after_prompt": validation_text[len(sample["prompt"])]},
            "generated_bytes": list(raw), "decoded": tokenizer.decode(generated_ids),
            "strict_decode_error": decode_error, "complete_prefix": raw[:-2].decode("utf-8"),
            "complete_prefix_bytes": len(raw[:-2]), "incomplete_suffix_hex": raw[-2:].hex(" "),
            "all_single_continuation_completions": len(completions),
            "existing_text_retained_by_all_completions": True, "replay": replay,
            "replay_scope": "固定 logits 回播原始 ID，核對現行 evaluate_lm、ByteTokenizer 與 generate；沒有重訓或重新生成模型預測。",
            "code_version_matches": sources_checked,
        },
        "budget_derivation": {"ascii_bytes_per_character": 1, "cjk_bytes_per_character": 3,
                              "byte_positions": 32, "ascii_characters": 32, "complete_common_cjk_characters": 32 // 3, "remaining_bytes": 32 % 3},
        "recipe_interface": {"help_command_argv": help_command, "exit_code": help_run.returncode, "stdout": help_run.stdout,
                             "stderr": help_run.stderr, "selected_specification": experiment_spec("real_text"),
                             "scope": "只核對 CLI 與配方選擇；沒有執行 GPU 訓練或下載。"},
        "result": "算例、記錄樣本與固定 logits 回播相符；surrogate 碼點反例推翻對所有碼點的 UTF-8 編碼斷言。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sources", "bundle", "probe"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.mode == "sources":
        result = sources()
    elif args.mode == "bundle":
        result = bundle(args.directory)
    else:
        result = probe()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
