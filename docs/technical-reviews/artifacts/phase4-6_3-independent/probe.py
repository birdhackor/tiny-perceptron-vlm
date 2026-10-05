"""Small tokenizer contracts and read-only reconstruction of existing evidence; no LM training."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tiny_perceptron.data import SPECIALS
from tiny_perceptron.tokenization import ByteLevelBPE

sha = lambda raw: hashlib.sha256(raw).hexdigest()
def dump(name, value):
    (ART / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

namespace = {}
exec(compile((ART / "original/fence-1.py").read_bytes(), "original-fence-1.py", "exec"), namespace)
tok = namespace["tok"]
definition = json.loads(tok.to_str())
inputs = ["未見字🦊", " 小鳥🙂\nnew ", "\t\r\n\x00", "e\u0301 é", "𠮷\U0010ffff", "<user>這只是引用文字", ""]
variations = []
for text in inputs:
    encoding = tok.encode(text)
    restored = tok.decode(encoding.ids)
    variations.append({"input": text, "utf8_hex": text.encode().hex(), "tokens": encoding.tokens, "ids": encoding.ids, "restored": restored, "same": restored == text})
    assert restored == text
alphabet = set(pre_tokenizers.ByteLevel.alphabet())
assert len(alphabet) == 256 and alphabet <= set(tok.get_vocab())
dump("bounded-roundtrips.json", {"cases": variations, "original_target": 280, "actual_vocab": tok.get_vocab_size(), "alphabet_count": len(alphabet), "merge_count": len(definition["model"]["merges"]), "added_tokens": definition["added_tokens"], "normalizer": definition["normalizer"], "pre_tokenizer": definition["pre_tokenizer"], "decoder": definition["decoder"]})

def train(corpus, alphabet_enabled=True, target=280):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=target, initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if alphabet_enabled else [], show_progress=False))
    return t

small = train(["a"])
incomplete = train(["a"], alphabet_enabled=False)
prefix = Tokenizer.from_str(tok.to_str())
prefix.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
lower = Tokenizer.from_str(tok.to_str())
lower.normalizer = normalizers.Lowercase()
changed = train(["zebra mountain river ocean planet orange purple yellow building weather" * 3], target=tok.get_vocab_size())
different_ids = [{"id": i, "original": tok.id_to_token(i), "other": changed.id_to_token(i)} for i in range(min(tok.get_vocab_size(), changed.get_vocab_size())) if tok.id_to_token(i) != changed.id_to_token(i)]
assert small.get_vocab_size() == 256
assert incomplete.decode(incomplete.encode("未見字🦊").ids) != "未見字🦊"
assert prefix.decode(prefix.encode("小鳥").ids) == " 小鳥"
assert lower.decode(lower.encode("NEW").ids) == "new"
assert tok.get_vocab_size() == changed.get_vocab_size() == 277 and different_ids
dump("contract-variations.json", {
    "single_char_corpus_actual_vocab": small.get_vocab_size(), "single_char_target": 280,
    "no_alphabet_vocab": incomplete.get_vocab_size(), "no_alphabet_unseen_encoded": incomplete.encode("未見字🦊").ids, "no_alphabet_unseen_restored": incomplete.decode(incomplete.encode("未見字🦊").ids),
    "prefix_added_input": "小鳥", "prefix_added_restored": prefix.decode(prefix.encode("小鳥").ids),
    "normalizer_input": "NEW", "normalizer_restored": lower.decode(lower.encode("NEW").ids),
    "other_training_input": ["zebra mountain river ocean planet orange purple yellow building weather" * 3],
    "other_actual_vocab": changed.get_vocab_size(), "same_size_changed_id_meanings": different_ids,
})

result_path = ROOT / "docs/course-experiments/results/tokenizer.json"
result = json.loads(result_path.read_text())
results = result["results"]
candidate = ROOT / "outputs/text-behavior-interface-check/tokenizer"
provenance = {"original_result": {"path": str(result_path.relative_to(ROOT)), "sha256": sha(result_path.read_bytes()), "revision": result["revision"], "code_hashes": result["code_sha256"], "environment": {k: result[k] for k in ("device", "seed", "torch_version", "python_version", "gpu", "evidence_status")}}, "candidate_files": [], "asset_files": [], "note": "Only local existing files read. No training assets downloaded or prepared; no tokenizer refit on the 153 records; no language model training or evaluation."}
for split in ("train", "validation", "test"):
    path = candidate / "data" / f"{split}.jsonl"
    expected = results["data"][split]["sha256"]
    actual = sha(path.read_bytes())
    provenance["candidate_files"].append({"path": str(path.relative_to(ROOT)), "sha256": actual, "recorded_sha256": expected, "same": actual == expected})
dump("input-provenance.json", provenance)
print(json.dumps({"candidate_files": provenance["candidate_files"], "original_vocab": tok.get_vocab_size(), "different_ids": len(different_ids)}, ensure_ascii=False))
if not all(item["same"] for item in provenance["candidate_files"]):
    raise RuntimeError("Candidate split bytes do not match original; stop historical checks")

# Preserve the matching original split bytes (read only) and tokenizer JSON as review inputs.
for name in ("data/train.jsonl", "data/validation.jsonl", "data/test.jsonl", "data/manifest.json", "tokenizer-bpe512.json", "tokenizer-byte.json"):
    destination = ART / "historical-inputs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((candidate / name).read_bytes())

expected_tok_sha = next(item["sha256"] for item in result["artifacts"] if item["path"] == "tokenizer-bpe512.json")
bpe_path = ART / "historical-inputs/tokenizer-bpe512.json"
assert sha(bpe_path.read_bytes()) == expected_tok_sha
provenance["candidate_files"].append({"path": str((candidate / "tokenizer-bpe512.json").relative_to(ROOT)), "sha256": sha(bpe_path.read_bytes()), "recorded_sha256": expected_tok_sha, "same": True, "reference": "original result artifacts list"})
bpe = ByteLevelBPE(bpe_path)
past_roundtrips = []
for item in results["roundtrip"]:
    ids = bpe.encode(item["text"])
    restored = bpe.decode(ids)
    observation = {**item, "observed_ids": ids, "observed_restored": restored, "exact_recorded_ids_match": ids == item["ids"], "same": restored == item["text"], "observed_control_id": any(i < 8 for i in ids)}
    assert observation["exact_recorded_ids_match"] and observation["same"] and not observation["observed_control_id"]
    past_roundtrips.append(observation)

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

raw = []
for filename in ("tinystories-train-512.jsonl", "chinese-classical-train-365.jsonl"):
    path = ROOT / "data/training/text-initial" / filename
    records = load_jsonl(path)
    expected = next(f["sha256"] for asset in result["assets"] for f in asset["files"] if Path(f["path"]).name == filename)
    assert sha(path.read_bytes()) == expected
    provenance["asset_files"].append({"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), "recorded_sha256": expected, "same": True, "records": len(records), "selected": 96, "first_record_source_revision": records[0].get("source_revision")})
    raw.extend(records[:96])

# Independently inspect complete-document split and safe-prefix transformation; no data preparation entrypoint.
seen, dedup = set(), []
for row in raw:
    key = " ".join(row["text"].split())
    if key not in seen:
        dedup.append({**row, "family": sha(key.encode())})
        seen.add(key)
groups = {row["family"]: [row] for row in dedup}
keys = sorted(groups)
random.Random(42).shuffle(keys)
a, b = int(len(keys) * .8), int(len(keys) * .9)
selected = {"train": keys[:a], "validation": keys[a:b], "test": keys[b:]}
facts, identities = {}, []
for split in ("train", "validation", "test"):
    rows = load_jsonl(ART / "historical-inputs/data" / f"{split}.jsonl")
    complete = [groups[key][0] for key in selected[split]]
    assert len(rows) == len(complete) == results["data"][split]["records"]
    for i, (row, full) in enumerate(zip(rows, complete, strict=True)):
        expected_text = full["text"].encode()[:256].decode("utf-8", errors="ignore")
        assert row["family"] == full["family"] and row["complete_text_sha256"] == sha(full["text"].encode()) and row["text"] == expected_text
        identities.append({"split": split, "row": i, "family": row["family"], "source": full.get("source"), "source_row": full.get("source_row"), "source_revision": full.get("source_revision"), "complete_text_sha256": sha(full["text"].encode()), "prefix_utf8_sha256": sha(row["text"].encode()), "prefix_utf8_bytes": len(row["text"].encode()), "prefix_characters": len(row["text"]), "prefix_formula_matched": True})
    families = {row["family"] for row in rows}
    lengths = [len(row["text"].encode()) for row in rows]
    facts[split] = {"records": len(rows), "families": len(families), "min_prefix_bytes": min(lengths), "max_prefix_bytes": max(lengths), "raw_utf8_bytes": sum(lengths), "all_complete_document_family_prefixes_matched": True}
assert not (set(selected["train"]) & set(selected["validation"]) | set(selected["train"]) & set(selected["test"]) | set(selected["validation"]) & set(selected["test"]))
defn = json.loads(bpe_path.read_text())
assert bpe.vocab_size == 512 and len(defn["added_tokens"]) == 8 and len(defn["model"]["merges"]) == 248
assert set(pre_tokenizers.ByteLevel.alphabet()) <= set(defn["model"]["vocab"])
assert [defn["model"]["vocab"][s] for s in SPECIALS] == list(range(8))
dump("original-measurement-check.json", {"original_result_revision": result["revision"], "raw_selected": len(raw), "deduplicated": len(dedup), "seed": 42, "split_boundaries": {"a": a, "b": b, "train": a, "validation": b-a, "test": len(keys)-b}, "prefix_rule": "UTF-8 first <=256 bytes, discard only incomplete last code point", "facts": facts, "vocab": {"actual": bpe.vocab_size, "specials": len(defn["added_tokens"]), "byte_alphabet": len(alphabet), "merges": len(defn["model"]["merges"]), "count_identity": "512 = 8 + 256 + 248"}, "roundtrips": past_roundtrips, "language_model_training_or_evaluation_rerun": False})
dump("original-row-provenance.json", identities)
dump("input-provenance.json", provenance)
print(json.dumps({"split_facts": facts, "roundtrip_count": len(past_roundtrips), "vocab_identity": "512 = 8+256+248", "no_lm_training": True}, ensure_ascii=False))
