"""以 curl 讀取官方原文，只保存支持本節主張的片段與收據。"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime


TOKENIZERS = "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/"
PYTHON = "https://raw.githubusercontent.com/python/cpython/v3.13.16/"
SPECS = [
    ("whitespace", TOKENIZERS + "tokenizers/src/pre_tokenizers/whitespace.rs", [
        ("impl PreTokenizer for Whitespace {", 10),
    ]),
    ("bytelevel", TOKENIZERS + "tokenizers/src/pre_tokenizers/byte_level.rs", [
        ("pub(crate) fn bytes_char()", 23),
        ("impl PreTokenizer for ByteLevel {", 16),
        ("impl Decoder for ByteLevel {", 23),
    ]),
    ("bpe_trainer", TOKENIZERS + "tokenizers/src/models/bpe/trainer.rs", [
        ("fn add_special_tokens(", 13),
        ("fn compute_alphabet(", 18),
        ("kept.sort_unstable_by_key(|k| *k.0 as u32);", 3),
        ("// Stop as soon as we have a big enough vocabulary", 23),
    ]),
    ("bpe_model", TOKENIZERS + "tokenizers/src/models/bpe/model.rs", [
        ("fn merge_word(&self, w: &str)", 7),
        ("if let Some(unk_token) = &self.unk_token", 30),
    ]),
    ("tokenizer", TOKENIZERS + "tokenizers/src/tokenizer/mod.rs", [
        ("None => pretokenized.tokenize(|normalized| self.tokenize(normalized.get()))", 3),
    ]),
    ("pretokenized", TOKENIZERS + "tokenizers/src/tokenizer/pre_tokenizer.rs", [
        ("pub fn tokenize<F>", 10),
    ]),
    ("template", TOKENIZERS + "tokenizers/src/processors/template.rs", [
        ("//! Provides a way to specify templates", 11),
    ]),
    ("buffered_decoder", PYTHON + "Lib/codecs.py", [
        ("class BufferedIncrementalDecoder(IncrementalDecoder):", 25),
    ]),
    ("utf8_decoder", PYTHON + "Lib/encodings/utf_8.py", [
        ("class IncrementalDecoder(codecs.BufferedIncrementalDecoder):", 2),
    ]),
]


def fetch(spec):
    name, url, selections = spec
    command = ["curl", "-fsSL", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    lines = raw.decode("utf-8").splitlines()
    excerpts = []
    for marker, length in selections:
        start = next(i for i, line in enumerate(lines) if marker in line)
        excerpts.append({
            "symbol_or_marker": marker,
            "first_line": start + 1,
            "text": "\n".join(lines[start:start + length]),
        })
    return {
        "id": name,
        "url": url,
        "version": "v0.23.2" if url.startswith(TOKENIZERS) else "v3.13.16",
        "accessed_at": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": result.returncode,
        "stderr": result.stderr.decode(),
        "original_bytes": len(raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpts": excerpts,
    }


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(fetch, SPECS))
    print(json.dumps({"sources": sources}, ensure_ascii=False, indent=2))
