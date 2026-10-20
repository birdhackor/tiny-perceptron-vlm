"""核對 UTF-8、切詞與逐步解碼；外部全文只在記憶體讀取。"""

import codecs
import hashlib
import html
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url, version, selectors, note, is_html=False):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    if is_html:
        text = html.unescape(re.sub(r"<[^>]*>", "", text))
    excerpts = []
    for locator, needle in selectors:
        assert needle in text, "來源沒有預期的定位文字"
        start = text.index(needle)
        excerpts.append({"locator": locator, "excerpt": needle,
                         "retrieved_text_line": text[:start].count("\n") + 1})
    return {"url": url, "version": version, "command_argv": command,
            "accessed_at_utc": datetime.now(UTC).isoformat(),
            "response_sha256": digest(raw), "response_bytes": len(raw),
            "exit_code": completed.returncode,
            "stderr": completed.stderr.decode("utf-8"), "excerpts": excerpts,
            "inspection_note": note}


raw = "貓🙂".encode()
assert list(raw) == [232, 178, 147, 240, 159, 153, 130]
assert "貓".encode() == bytes([232, 178, 147])
assert "🙂".encode() == bytes([240, 159, 153, 130])
assert raw.decode("utf-8") == "貓🙂"
prefixes = {str(n): raw[:n].decode("utf-8", errors="replace") for n in (1, 3, 4)}
assert prefixes == {"1": "�", "3": "貓", "4": "貓�"}
try:
    raw[:1].decode("utf-8")
except UnicodeDecodeError as error:
    strict_error = {"type": type(error).__name__, "encoding": error.encoding,
                    "start": error.start, "end": error.end, "reason": error.reason}
else:
    raise AssertionError("不完整的 UTF-8 應觸發解碼錯誤")

decoder_class = codecs.getincrementaldecoder("utf-8")
decoder = decoder_class()
shown, buffers = [], []
for value in raw:
    shown.append(decoder.decode(bytes([value])))
    buffers.append(list(decoder.getstate()[0]))
assert shown == ["", "", "貓", "", "", "", "🙂"]
assert "".join(shown) == "貓🙂"
assert decoder.decode(b"", final=True) == ""
separate = [bytes([value]).decode("utf-8", errors="replace") for value in raw]
assert separate == ["�"] * 7
saved_prefix = prefixes["1"].encode("utf-8")
overwritten = saved_prefix + raw[1:]
assert overwritten != raw
assert "貓" not in overwritten.decode("utf-8", errors="replace")
chunked = codecs.getincrementaldecoder("utf-8")()
split_outputs = [chunked.decode(raw[:2]), chunked.decode(raw[2:], final=True)]
assert split_outputs == ["", "貓🙂"]
unfinished = codecs.getincrementaldecoder("utf-8")()
assert unfinished.decode(raw[:2]) == ""
try:
    unfinished.decode(b"", final=True)
except UnicodeDecodeError as error:
    final_error = type(error).__name__
else:
    raise AssertionError("結束時仍不完整的 UTF-8 應觸發解碼錯誤")

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=280,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
tok.train_from_iterator(["貓看狗。", "a cat sees a dog."], trainer)
bpe = {}
for text in ("貓看狗。", "貓看狗", "小", "小鳥🙂new"):
    encoded = tok.encode(text)
    bpe[text] = {"utf8_bytes": list(text.encode("utf-8")), "ids": encoded.ids,
                 "tokens": encoded.tokens, "decoded": tok.decode(encoded.ids)}
    assert bpe[text]["decoded"] == text
assert len(bpe["貓看狗"]["ids"]) == 1
assert len(bpe["小"]["ids"]) == 3
assert all(len(piece) == 1 for piece in bpe["小"]["tokens"])
assert len(bpe["貓看狗。"]["ids"]) == 2
assert len(bpe["小鳥🙂new"]["ids"]) == 13

experiment_path = "docs/course-experiments/results/real_text.json"
experiment_raw = (ROOT / experiment_path).read_bytes()
experiment = json.loads(experiment_raw)
samples = experiment["results"]["runs"]["chinese-poetry"]["after"]["validation"]["samples"]
sample = samples[7]
byte_tok = ByteTokenizer()
generated = bytes(value - 8 for value in sample["generated_ids"])
assert len(generated) == 32
assert byte_tok.decode(sample["generated_ids"]) == sample["generated"]
sample_decoder = codecs.getincrementaldecoder("utf-8")()
complete_sample = sample_decoder.decode(generated)
pending = sample_decoder.getstate()[0]
assert len(pending) == 2
assert complete_sample + "�" == sample["generated"]
assert sample["generated"].endswith("�")

source_receipts = [
    fetch("https://docs.python.org/3.13/library/stdtypes.html", "Python 3.13.16 官方文件",
          [("str.encode", "str.encode(encoding='utf-8', errors='strict')"),
           ("bytes.decode", "bytes.decode(encoding='utf-8', errors='strict')"),
           ("bytes 建構方式", "From an iterable of integers: bytes(range(20))")],
          "已讀 str.encode、bytes.decode 與 bytes；預設 UTF-8／strict，bytes 可由整數 iterable 建立。", True),
    fetch("https://docs.python.org/3.13/library/codecs.html", "Python 3.13.16 官方文件",
          [("codecs.getincrementaldecoder", "return its incremental decoder\nclass or factory function."),
           ("IncrementalDecoder.decode", "Decodes object (taking the current state of the decoder into account)"),
           ("codecs.replace_errors", "� (U+FFFD")],
          "已讀 getincrementaldecoder、IncrementalDecoder.decode 與 replace_errors；解碼器保留狀態，final=True 必須清空緩衝，未完成輸入會進入錯誤處理，replace 使用 U+FFFD。", True),
    fetch("https://www.rfc-editor.org/rfc/rfc3629.txt", "RFC 3629，November 2003",
          [("UTF-8 encoding table", "0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx"),
           ("UTF-8 encoding table", "0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx")],
          "已讀 UTF-8 編碼與解碼規則及有效位元範圍；232 的二進位為 11101000，需接兩個 continuation byte。"),
    fetch("https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/trainer.rs",
          "Hugging Face tokenizers v0.23.2",
          [("impl Ord for Merge::cmp", "self.count.cmp(&other.count)"),
           ("BpeTrainer::do_train", "self.count_pairs(&words, &counts, &progress);"),
           ("BpeTrainer::do_train", "let Some(mut top) = queue.pop() else {"),
           ("BpeTrainer::do_train", 'let new_token = format!("{part_a}{part_b}");')],
          "已讀 Merge 的排序、count_pairs 與 do_train；頻次決定相鄰對的優先順序，合併把兩個片段串接，沒有 UTF-8 碼點對齊條件。"),
    fetch("https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/pre_tokenizers/byte_level.rs",
          "Hugging Face tokenizers v0.23.2",
          [("bytes_char", "pub(crate) fn bytes_char() -> AHashMap<u8, char> {"),
           ("ByteLevel::pre_tokenize", "s.as_bytes()[i..i + size]"),
           ("ByteLevel::pre_tokenize", ".map(|(i, b)| (BYTES_CHAR[b], isize::from(i > 0))),"),
           ("ByteLevel::decode_chain", "String::from_utf8_lossy(&toks)")],
          "已讀 bytes_char、ByteLevel 的前處理與 decoder；UTF-8 每個 byte 先映成一個可見符號，decoder 把 tokens 拼回 byte 後一起解碼。")
]

paths = ["course/chapters/06.md", "tiny_perceptron/data.py", experiment_path,
         "docs/course-review-20261019/freeze-01/sources/6.7.md",
         "docs/course-review-20261019/freeze-04.json"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-7/verify.py",
    "environment": {"python": sys.version, "tokenizers": tokenizers.__version__,
                    "platform": platform.platform(), "device": "cpu"},
    "input_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
    "utf8": {"raw": list(raw), "cat": list("貓".encode()), "emoji": list("🙂".encode()),
             "lead_binary": format(raw[0], "08b"), "prefix_decodes": prefixes,
             "strict_error": strict_error, "full_decode": raw.decode("utf-8"),
             "decoder_class": decoder_class.__module__ + "." + decoder_class.__name__,
             "factory_returns_class": isinstance(decoder_class, type),
             "shown": shown, "buffer_after_each_byte": buffers,
             "single_byte_decodes": separate, "joined_separate": "".join(separate),
             "replacement_utf8": list(saved_prefix), "overwritten_bytes": list(overwritten),
             "overwritten_decode": overwritten.decode("utf-8", errors="replace"),
             "original_after_observations": list(raw), "chunked_decode": split_outputs,
             "unfinished_final_error": final_error},
    "bpe": {"training_texts": ["貓看狗。", "a cat sees a dog."],
            "vocab_size_target": 280, "initial_alphabet_size": len(pre_tokenizers.ByteLevel.alphabet()),
            "add_prefix_space": False, "actual_vocab_size": tok.get_vocab_size(), "examples": bpe},
    "recorded_generation": {"path": experiment_path,
                            "locator": "results.runs.chinese-poetry.after.validation.samples[7]",
                            "sample": sample, "raw_generated_bytes": list(generated),
                            "complete_prefix": complete_sample, "pending_bytes": list(pending),
                            "prompt_bytes": len(sample["prompt"].encode("utf-8")),
                            "recorded_environment": {key: experiment[key] for key in ("revision", "device", "seed", "torch_version", "python_version")},
                            "training_rerun": False},
    "source_receipts": source_receipts,
    "result": "全部斷言通過；只核對確定性解碼、當前切詞範例與既有生成紀錄，不外推模型品質。"
}
print(json.dumps(result, ensure_ascii=False, indent=2))
