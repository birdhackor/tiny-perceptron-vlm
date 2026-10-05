"""Independent CPU checks of section 6.7; no model/weight/data downloads."""
import ast
import codecs
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, SPECIALS
from tiny_perceptron.tokenization import ByteLevelBPE, generation_report

os.environ["TOKENIZERS_PARALLELISM"] = "false"
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment = {"python": sys.version, "python_executable": sys.executable,
               "torch": str(torch.__version__), "tokenizers": tokenizers.__version__,
               "device": "CPU", "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(),
               "cwd": str(Path.cwd()), "recipe_executed": "No training/model/download recipe"}
results = {"environment": environment}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def decode_status(raw):
    try:
        return {"strict": raw.decode("utf-8"), "replace": raw.decode("utf-8", errors="replace")}
    except UnicodeDecodeError as e:
        return {"strict_error": type(e).__name__, "error_start": e.start, "error_end": e.end,
                "reason": e.reason, "replace": raw.decode("utf-8", errors="replace")}

def utf8_from_scalar(cp):
    # RFC 3629 section 3 bit placement, independently implemented.
    if cp < 128:
        return [cp]
    if cp < 2048:
        return [192 | (cp >> 6), 128 | (cp & 63)]
    if cp < 65536:
        return [224 | (cp >> 12), 128 | ((cp >> 6) & 63), 128 | (cp & 63)]
    return [240 | (cp >> 18), 128 | ((cp >> 12) & 63), 128 | ((cp >> 6) & 63), 128 | (cp & 63)]

raw = "貓🙂".encode()
expected = [232,178,147,240,159,153,130]
assert list(raw) == expected
calculations = [{"character": c, "code_point": f"U+{ord(c):04X}",
                 "rfc3629_octets": utf8_from_scalar(ord(c)), "python_octets": list(c.encode())}
                for c in "貓🙂"]
assert all(x["rfc3629_octets"] == x["python_octets"] for x in calculations)
results["numeric"] = {"raw_octets": list(raw), "total_bytes": len(raw), "scalars": calculations,
                       "unit": "integer octets (bytes), no loss axes or dataset denominator"}
results["prefixes"] = [{"prefix_bytes": n, "octets": list(raw[:n]), **decode_status(raw[:n])}
                       for n in (1,3,4,7)]
assert [x["replace"] for x in results["prefixes"]] == ["�", "貓", "貓�", "貓🙂"]
assert results["prefixes"][0]["strict_error"] == "UnicodeDecodeError"
assert results["prefixes"][2]["strict_error"] == "UnicodeDecodeError"

# Execute the unmodified code and the two exact exercise changes in fresh namespaces.
fence = (BASE / "original/fence-1.py").read_text()
for n in (1,3,4):
    changed = fence if n == 1 else fence.replace("raw[:1]", f"raw[:{n}]")
    (BASE / f"code/fence-prefix-{n}.py").write_text(changed, encoding="utf-8")
    print(f"FENCE prefix={n}")
    exec(compile(changed, f"fence-prefix-{n}.py", "exec"), {})
assert raw.decode() == "貓🙂"

single = "".join(bytes([b]).decode("utf-8", errors="replace") for b in raw)
stored = raw[:1].decode("utf-8", errors="replace").encode("utf-8") + raw[1:]
results["replacement_loss"] = {"separate_byte_decodes": single,
    "all_bytes_decode": raw.decode(), "persisted_replacement_octets": list(stored),
    "persisted_replacement_decoded": stored.decode("utf-8", errors="replace"),
    "original_unchanged": list(raw) == expected}
assert single != raw.decode() and stored != raw

inc = codecs.getincrementaldecoder("utf-8")("strict")
steps = []
for b in raw:
    out = inc.decode(bytes([b]), final=False)
    steps.append({"input_octet": b, "emitted": out, "buffered_octets": list(inc.getstate()[0])})
tail = inc.decode(b"", final=True)
assert "".join(s["emitted"] for s in steps) + tail == "貓🙂"
results["incremental"] = {"steps": steps, "final_flush": tail}
finals = []
for errors in ("strict", "replace"):
    decoder = codecs.getincrementaldecoder("utf-8")(errors)
    assert decoder.decode(raw[:1], final=False) == ""
    try:
        value = decoder.decode(b"", final=True)
        finals.append({"errors": errors, "final_output": value})
    except UnicodeDecodeError as e:
        finals.append({"errors": errors, "final_error": type(e).__name__, "reason": e.reason})
assert finals[0]["final_error"] == "UnicodeDecodeError" and finals[1]["final_output"] == "�"
results["final_truncation"] = finals
results["invalid_complete_sequences"] = [{"octets": list(x), **decode_status(x)}
                                         for x in (b"\xff", b"\xc0\x80", b"\xed\xa0\x80", b"\xe8A\xb2\x93")]
assert all(x.get("strict_error") == "UnicodeDecodeError" for x in results["invalid_complete_sequences"])
try:
    codecs.getincrementaldecoder("utf-8")("strict").decode(b"\xff", final=False)
    raise AssertionError("Invalid byte unexpectedly buffered")
except UnicodeDecodeError:
    results["invalid_complete_sequences_immediate_failure"] = True

byte = ByteTokenizer()
ids = byte.encode("貓🙂")
results["byte_tokens"] = {"ids": ids, "full_decode": byte.decode(ids),
    "single_token_decodes": [byte.decode([i]) for i in ids],
    "specials_filtered_by_plain_decode": byte.decode([byte.bos_id] + ids + [byte.eos_id]),
    "invalid_sequence_report": generation_report(byte, [255+8]),
    "incomplete_sequence_report": generation_report(byte, [232+8]),
    "invalid_control_report": generation_report(byte, [232+8,byte.assistant_id,178+8,147+8]),
    "eos_with_incomplete_tail": generation_report(byte, [232+8,byte.eos_id])}
assert results["byte_tokens"]["full_decode"] == "貓🙂"
assert results["byte_tokens"]["invalid_control_report"]["valid_answer_tokens"] is False
assert results["byte_tokens"]["invalid_sequence_report"]["answer"] == "�"

# Reproduce roundtrips/IDs only from hash-verified historical definitions.
summary = json.loads((BASE / "inputs/current/docs/course-experiments/results/tokenizer.json").read_text())
provenance = json.loads((BASE / "input-provenance.json").read_text())
for entry in provenance["checks"]:
    assert sha(ROOT / entry["permanent"]) == entry["expected_sha256"]
bpe = ByteLevelBPE(BASE / "inputs/historical/tokenizer-bpe512.json")
roundtrips = []
for item in summary["results"]["roundtrip"]:
    actual_ids = bpe.encode(item["text"])
    restored = bpe.decode(actual_ids)
    assert actual_ids == item["ids"] and restored == item["restored"] == item["text"]
    roundtrips.append({"text": item["text"], "ids": actual_ids, "restored": restored,
        "contains_control_id": any(i in bpe.special_ids for i in actual_ids)})
bpe_ids = bpe.encode("貓🙂")
assert bpe.decode(bpe_ids) == "貓🙂"
assert "".join(bpe.decode([i]) for i in bpe_ids) != "貓🙂"
results["historical_bpe"] = {"vocab_size": bpe.vocab_size, "input_sha256": bpe.sha256,
    "roundtrip_record_count": len(roundtrips), "roundtrips": roundtrips,
    "cat_emoji_ids": bpe_ids, "cat_emoji_tokens": [bpe.content.id_to_token(i) for i in bpe_ids],
    "single_token_decodes": [bpe.decode([i]) for i in bpe_ids], "full_decode": bpe.decode(bpe_ids)}
split_counts = {}
for name, entry in summary["results"]["data"].items():
    if not isinstance(entry, dict):
        continue
    rows = [json.loads(l) for l in (BASE / "inputs/historical" / entry["path"]).read_text().splitlines()]
    assert len(rows) == entry["records"]
    assert sha(BASE / "inputs/historical" / entry["path"]) == entry["sha256"]
    assert all(byte.decode(byte.encode(r["text"])) == r["text"] and
               bpe.decode(bpe.encode(r["text"])) == r["text"] for r in rows)
    split_counts[name] = {"records": len(rows), "utf8_bytes": sum(len(r["text"].encode()) for r in rows),
                          "roundtrips_checked": len(rows)}
results["historical_splits"] = split_counts

# A bounded self-authored tokenizer demonstration, not a language-model experiment.
toy = Tokenizer(models.BPE())
toy.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
toy.decoder = decoders.ByteLevel()
toy.train_from_iterator(["貓狗"] * 12, trainers.BpeTrainer(vocab_size=280,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=list(SPECIALS), show_progress=False))
toy_ids = toy.encode("貓狗", add_special_tokens=False).ids
assert len(toy_ids) == 1 and toy.decode(toy_ids) == "貓狗"
toy.save(str(BASE / "execution/toy-tokenizer.json"))
results["tiny_bpe_multiple_scalars"] = {"training_strings": ["貓狗"] * 12,
    "vocabulary_budget": 280, "actual_vocab_size": toy.get_vocab_size(), "ids": toy_ids,
    "decoded": toy.decode(toy_ids), "unicode_scalars": len(toy.decode(toy_ids)),
    "scope": "One possible learned byte-level BPE token spans two Chinese code points; not a quality score"}
results["grapheme_scope"] = [{"text": t, "unicode_code_points": [f"U+{ord(c):04X}" for c in t],
                              "python_length_in_code_points": len(t), "utf8_bytes": list(t.encode())}
                             for t in ("G\u0300", "👩\u200d💻")]
results["files_used"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in
    [BASE / "original/fence-1.py", ROOT / "tiny_perceptron/data.py", ROOT / "tiny_perceptron/tokenization.py",
     BASE / "inputs/historical/tokenizer-bpe512.json", BASE / "inputs/current/docs/course-experiments/results/tokenizer.json"]]
(BASE / "execution/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(BASE / "execution/environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
print("All bounded assertions passed; no model weights were read or evaluated.")
