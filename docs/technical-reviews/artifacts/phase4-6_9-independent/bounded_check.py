"""Bounded CPU checks of 6.9, without model inference or corpus preparation."""
import codecs
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.tokenization import ByteLevelBPE
from tiny_perceptron.data import ByteTokenizer

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids
def info(tok, text):
    e = tok.encode(text, add_special_tokens=False)
    return {"text": text, "ids": e.ids, "tokens": e.tokens, "length": len(e.ids)}

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    original = runpy.run_path(str(OUT / "original-run/fence-1.py"))
tok = original["tok"]
whole = original["whole"]
chunks = original["chunks"]
assert len(whole) == 1 and len(chunks) == 2 and whole != chunks
assert tok.token_to_id("[UNK]") == 0
exercise_whole = encode(tok, "hello hello")
exercise_chunks = encode(tok, "hello ") + encode(tok, "hello")
assert exercise_whole == exercise_chunks == whole + whole
interior = [{"cut": n, "ids": encode(tok, "hello"[:n]) + encode(tok, "hello"[n:])} for n in range(1, 5)]
assert all(row["ids"] != whole for row in interior)

# A restricted safe implementation for this ASCII Whitespace example. It retains
# the pending word until a whitespace separator is available, and flushes at EOF.
# It is deliberately not offered as a universal tokenizer streaming algorithm.
def whitespace_stream(chunks):
    pending, ids = "", []
    for chunk in chunks:
        pending += chunk
        last_space = max((i for i, c in enumerate(pending) if c.isspace()), default=-1)
        if last_space >= 0:
            ids.extend(encode(tok, pending[:last_space + 1]))
            pending = pending[last_space + 1:]
    return ids + encode(tok, pending)
safe_checks = [whitespace_stream(["hello hello"[:n], "hello hello"[n:]]) == exercise_whole for n in range(12)]
assert all(safe_checks)

byte = Tokenizer(models.BPE())
byte.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
byte.decoder = decoders.ByteLevel()
byte.train_from_iterator(["hello hello hello"], trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
byte_whole = encode(byte, "hello hello")
byte_chunks = encode(byte, "hello ") + encode(byte, "hello")
assert byte_whole != byte_chunks
assert byte.decode(byte_whole) == byte.decode(byte_chunks) == "hello hello"
space_rules = {"whitespace_whole": tok.pre_tokenizer.pre_tokenize_str("hello hello"), "whitespace_left": tok.pre_tokenizer.pre_tokenize_str("hello "), "byte_whole": byte.pre_tokenizer.pre_tokenize_str("hello hello"), "byte_left": byte.pre_tokenizer.pre_tokenize_str("hello ")}

text = "貓🙂hello hello"
raw = text.encode("utf-8")
decoder = codecs.getincrementaldecoder("utf-8")("strict")
decoded_pieces, pending_byte_counts = [], []
for i in range(len(raw)):
    decoded_pieces.append(decoder.decode(raw[i:i+1], final=False))
    pending_byte_counts.append(len(decoder.getstate()[0]))
decoded_pieces.append(decoder.decode(b"", final=True))
assert "".join(decoded_pieces) == text
lossy = "".join(raw[i:i+1].decode("utf-8", errors="replace") for i in range(len(raw)))
assert lossy != text and "�" in lossy
utf8_whole = encode(byte, text)
utf8_naive = [i for piece in decoded_pieces for i in encode(byte, piece)]
assert utf8_whole != utf8_naive
assert byte.decode(utf8_naive) == byte.decode(utf8_whole) == text
stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")
text_io_pieces = list(iter(lambda: stream.read(1), ""))
assert "".join(text_io_pieces) == text and "�" not in text_io_pieces

separate = Tokenizer(models.BPE(unk_token="[UNK]"))
separate.pre_tokenizer = pre_tokenizers.Whitespace()
separate.train_from_iterator(["hel", "lo"], trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]"], show_progress=False))
assert separate.token_to_id("hello") is None
assert tok.token_to_id("hello") is not None
roundtrip_limit = {"whitespace_decode_of_original_chunks": tok.decode(chunks), "whitespace_decode_of_whole": tok.decode(whole), "whitespace_unknown_ids": encode(tok, "world"), "whitespace_whitespace_lost": tok.decode(encode(tok, " hello ")), "byte_whole": byte_whole, "byte_chunks": byte_chunks, "byte_both_roundtrip": byte.decode(byte_chunks) == byte.decode(byte_whole) == "hello hello"}
assert roundtrip_limit["whitespace_decode_of_original_chunks"] != "hello"
assert roundtrip_limit["whitespace_whitespace_lost"] != " hello "

saved = OUT / "inputs/original-tokenizer"
original_result = json.loads((OUT / "inputs/current/docs/course-experiments/results/tokenizer.json").read_bytes())
bpe = ByteLevelBPE(saved / "tokenizer-bpe512.json")
result = original_result["results"]
data_audit = {}
families = {}
for split in ("train", "validation", "test"):
    path = saved / "data" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    families[split] = {row["family"] for row in rows}
    assert digest(path) == result["data"][split]["sha256"]
    assert len(rows) == result["data"][split]["records"]
    assert all(len(row["text"].encode()) <= 256 for row in rows)
    bpe_lengths = [len(bpe.encode(row["text"])) for row in rows]
    byte_lengths = [len(ByteTokenizer().encode(row["text"])) for row in rows]
    assert all(bpe.decode(bpe.encode(row["text"])) == row["text"] for row in rows)
    if split == "validation":
        assert bpe_lengths == result["runs"]["bpe512"]["validation_text_token_lengths"]
        assert byte_lengths == result["runs"]["byte256"]["validation_text_token_lengths"]
    data_audit[split] = {"records": len(rows), "families": len(families[split]), "utf8_bytes": sum(byte_lengths), "byte_content_tokens": sum(byte_lengths), "bpe_content_tokens": sum(bpe_lengths), "max_utf8_bytes": max(byte_lengths), "all_saved_text_roundtrip": True, "sha256": digest(path)}
assert all(not families[a] & families[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
for row in result["roundtrip"]:
    assert bpe.encode(row["text"]) == row["ids"]
    assert bpe.decode(row["ids"]) == row["restored"] == row["text"]

# Verify the already saved excerpts against the original fixed source records;
# no new split or training input is created by this audit.
source_records = {}
raw_input_hashes = []
for relative in ("data/training/text-initial/tinystories-train-512.jsonl", "data/training/text-initial/chinese-classical-train-365.jsonl"):
    src = ROOT / relative
    target = OUT / "inputs/original-sources" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    raw_input_hashes.append({"path": relative, "snapshot": str(target.relative_to(ROOT)), "sha256": digest(target), "source_records": len(rows), "selected_by_original_recipe": 96})
    for row in rows[:96]:
        source_records[hashlib.sha256(row["text"].encode()).hexdigest()] = row["text"]
prefix_audit = []
for split in ("train", "validation", "test"):
    rows = [json.loads(line) for line in (saved / "data" / f"{split}.jsonl").read_text().splitlines()]
    for row in rows:
        complete = source_records[row["complete_text_sha256"]]
        assert row["text"] == complete.encode()[:256].decode("utf-8", errors="ignore")
        prefix_audit.append(row["complete_text_sha256"])
assert len(prefix_audit) == 192

report = {"environment": {"python": sys.version, "tokenizers": tokenizers.__version__, "device": "CPU", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), "model_weights_loaded": False}, "original_fence_stdout": capture.getvalue(), "original_example": {"whole": info(tok, "hello"), "left": info(tok, "hel"), "right": info(tok, "lo"), "chunks": chunks, "vocab": tok.get_vocab(), "actual_vocabulary_size": tok.get_vocab_size(), "target_vocabulary_size": 20}, "exercise": {"whole": exercise_whole, "chunks": exercise_chunks, "equal": True}, "interior_cuts": interior, "safe_whitespace_stream": {"checked_cuts": 12, "all_equal": all(safe_checks), "scope": "This exact Whitespace/ASCII example without other transforms; no universal finite-tail claim"}, "pretokenizer_space_rules": space_rules, "utf8_decoder_and_tokenizer_states": {"input": text, "utf8_bytes": len(raw), "chunks": len(raw), "decoded_pieces": decoded_pieces, "pending_byte_counts": pending_byte_counts, "replacement_per_byte_decode": lossy, "incremental_decode_exact": True, "text_io_read_1_exact": True, "whole_ids": utf8_whole, "per_decoded_piece_ids": utf8_naive, "ids_equal": False, "both_byte_bpe_roundtrip": True}, "training_iterator_boundaries": {"whole_trained_hello_id": tok.token_to_id("hello"), "separate_hel_lo_hello_id": separate.token_to_id("hello"), "whole_corpus_vocab_size": tok.get_vocab_size(), "separate_corpus_vocab_size": separate.get_vocab_size(), "scope": "Two small tokenizer training examples, not model training"}, "roundtrip_limitations": roundtrip_limit, "original_measurement_audit": {"result_sha256": digest(OUT / "inputs/current/docs/course-experiments/results/tokenizer.json"), "revision": original_result["revision"], "original_result_sections": result["sections"], "not_a_streaming_measurement": True, "original_model_training_reexecuted": False, "raw_source_input_hashes": raw_input_hashes, "prefixes_matched_to_complete_source": len(prefix_audit), "split_family_overlap": False, "data": data_audit, "saved_roundtrip_ids_exactly_reproduced": True, "saved_tokenizer_hash": bpe.sha256, "boundary_contract": "Original recipe splits full documents first, then takes one UTF-8-complete prefix per document, fits tokenizer on train only, encodes each record separately, and inserts BOS/EOS in code."}}
(OUT / "bounded-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
