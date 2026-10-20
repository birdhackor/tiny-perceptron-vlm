"""核對章首導讀的版本、BPE 還原性與 token ID 對應。"""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B "
    "docs/technical-reviews/artifacts/review-20261019-chapter-06/verify_intro.py"
)
COMMIT = "88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3"
ACCESSED_ON = datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    return subprocess.check_output(command), command


def save_json(name, value):
    target = ARTIFACTS / name
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    source_path = "docs/course-review-20261019/freeze-01/sources/chapter-06.md"
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    raw = (ROOT / source_path).read_bytes()
    freeze_raw = (ROOT / freeze_path).read_bytes()
    assert digest(raw) == "d681ee049910e848e6891cd1d01f9ec614df8a8870895db2ecdb9578a6ce88f3", "凍結原文的雜湊不符"
    assert digest(freeze_raw) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單的雜湊不符"
    page = next(item for item in json.loads(freeze_raw)["pages"] if item["page_id"] == "chapter-06")
    assert page["snapshot"] == source_path, "凍結清單指向其他原文"
    assert page["source_sha256"] == digest(raw), "凍結清單與原文不一致"
    assert page["figures_sha256"] == {}, "凍結清單含有圖檔"
    text = raw.decode("utf-8")
    assert not re.search(r"!\[[^\]]*\]\([^)]+\)", text), "導讀含有圖片引用"
    assert "```" not in text, "導讀含有程式或命令區塊"
    current_intro = (ROOT / "course/chapters/06.md").read_bytes().split(b"## ", 1)[0]
    assert current_intro == raw, "目前導讀與凍結導讀不同"

    tag_url = "https://api.github.com/repos/huggingface/tokenizers/git/ref/tags/v0.23.2"
    tag_raw, tag_command = fetch(tag_url)
    tag = json.loads(tag_raw)
    assert tag["object"]["type"] == "commit", "版本標籤未直接指向 commit"
    assert tag["object"]["sha"] == COMMIT, "版本標籤與核對的 commit 不符"
    excerpts = {
        "components.mdx": [
            "we can represent anything with 256 tokens",
            "Reverts the ByteLevel PreTokenizer.",
        ],
        "quicktour.mdx": [
            "Similarly, the `ids` attribute will\ncontain the index of each of those tokens in the tokenizer's\nvocabulary:",
        ],
        "pipeline.mdx": [
            "It's also responsible for mapping those tokens to\ntheir corresponding IDs in the vocabulary of the model.",
        ],
    }
    originals = []
    for name, selections in excerpts.items():
        url = f"https://raw.githubusercontent.com/huggingface/tokenizers/{COMMIT}/docs/source-doc-builder/{name}"
        original, curl_command = fetch(url)
        original_text = original.decode("utf-8")
        for selection in selections:
            assert selection in original_text, "來源不含預期摘錄"
        originals.append(
            {
                "url": url,
                "version": f"tokenizers v0.23.2, commit {COMMIT}",
                "accessed_on": ACCESSED_ON,
                "retrieval_command": curl_command,
                "original_sha256": digest(original),
                "excerpts": selections,
            }
        )
    save_json(
        "original-source-excerpts.json",
        {
            "tag_receipt": {
                "url": tag_url,
                "command": tag_command,
                "sha256": digest(tag_raw),
                "ref": tag["ref"],
                "commit": tag["object"]["sha"],
            },
            "originals": originals,
            "保存範圍": "全文只在記憶體讀取；檔案只保存支持主張的短摘錄與原檔雜湊。",
        },
    )

    training_texts = ["貓看狗。", "a cat sees a dog."]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    trainer = trainers.BpeTrainer(vocab_size=280, initial_alphabet=alphabet, show_progress=False)
    tokenizer.train_from_iterator(training_texts, trainer)
    assert len(set(alphabet)) == 256, "byte 字母表未覆蓋全部值"
    assert all(tokenizer.token_to_id(symbol) is not None for symbol in alphabet), "字表缺少 byte 的表示"
    definition = json.loads(tokenizer.to_str())
    assert definition["normalizer"] is None, "設定改寫了原文"
    assert not definition["pre_tokenizer"]["add_prefix_space"], "設定自動加入前綴空白"
    assert definition["added_tokens"] == [], "設定含有特殊 token 解析規則"
    cases = ["貓看狗。", "小鳥🙂new", "小小貓", " a\tB\n", "e\u0301", "👨‍👩‍👧‍👦", ""]
    rows = []
    for value in cases:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        mapping_matches = all(
            tokenizer.id_to_token(token_id) == token
            for token_id, token in zip(encoded.ids, encoded.tokens, strict=True)
        )
        offsets_valid = all(0 <= start < end <= len(value) for start, end in encoded.offsets)
        assert decoded == value, "完整編碼序列無法還原原文"
        assert mapping_matches, "ID 與 token 的字表對應不符"
        assert offsets_valid, "token 的原文範圍無效"
        rows.append(
            {
                "text": value,
                "absent_from_training": value not in training_texts,
                "utf8_bytes": list(value.encode("utf-8")),
                "ids": encoded.ids,
                "tokens": encoded.tokens,
                "offsets": encoded.offsets,
                "decoded": decoded,
                "exact_roundtrip": decoded == value,
                "ids_match_tokens": mapping_matches,
                "offsets_within_original": offsets_valid,
            }
        )
    unseen_characters = sorted(set("小鳥🙂new") - set("".join(training_texts)))
    assert set("小鳥🙂").issubset(unseen_characters), "驗證樣本未包含預期的未見字元"
    result = {
        "command": COMMAND,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "device": "CPU",
        },
        "source_inspection": {
            "snapshot": source_path,
            "source_sha256": digest(raw),
            "source_bytes": len(raw),
            "source_text": text,
            "source_freeze": {"path": freeze_path, "sha256": digest(freeze_raw)},
            "matches_current_intro": current_intro == raw,
            "figures": {},
            "contains_code_fence": False,
        },
        "tokenizer": {
            "training_texts": training_texts,
            "requested_vocab_size": 280,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "alphabet_size": len(alphabet),
            "all_byte_symbols_in_vocabulary": True,
            "normalizer": None,
            "add_prefix_space": False,
            "decoder": "ByteLevel",
            "added_tokens": [],
            "unseen_sample_characters_absent_from_training": unseen_characters,
        },
        "cases": rows,
        "result": "全部樣本逐字元精確還原；每個 ID 皆與字表中的 token 一致。",
        "limits": "只核對導讀的技術含意；有限樣本不是所有 tokenizer、所有 Unicode 輸入或生成結果的保證。",
    }
    save_json("verification-output.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
