"""Independent bounded CPU checks; no training, model loading, or network access."""
import ast
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
ART = Path(__file__).resolve().parents[1]

import torch
import tokenizers
from tokenizers import AddedToken, Tokenizer, normalizers, pre_tokenizers
from tiny_perceptron.data import ByteTokenizer, SPECIALS, render_chat, pad_batch, IGNORE
from tiny_perceptron.tokenization import ByteLevelBPE

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
path = ART / "inputs/original-tokenizer/tokenizer-bpe512.json"
definition = json.loads(path.read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == "ec08c61eff41c94035b6d234fb07eb63f21503b1d471ec4fbf5033df627180e8"

# Execute the exact historical class, isolated from the long training recipe.
tree = ast.parse((ART / "inputs/historical/scripts/course_experiments/text.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_BPE")
namespace = {"json": json, "SPECIALS": SPECIALS}
exec(compile(ast.Module(body=[node], type_ignores=[]), "historical/text.py:_BPE", "exec"), namespace)
raw = Tokenizer.from_file(str(path))
historical = namespace["_BPE"](raw)
current = ByteLevelBPE(path)
byte = ByteTokenizer()
out = {"environment": {"python": sys.version, "tokenizers": tokenizers.__version__, "torch": torch.__version__, "torch_cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(), "device": "cpu"}, "byte": [], "bpe": [], "roles": [], "official_api_variations": []}

assert list(enumerate(SPECIALS)) == [(0,"<pad>"),(1,"<bos>"),(2,"<eos>"),(3,"<user>"),(4,"<assistant>"),(5,"<image>"),(6,"<audio>"),(7,"<system>")]
assert byte.vocab_size == 8 + 256 == 264
assert 60 + 8 == 68 and 62 + 8 == 70
assert len("<image>") == 7
texts = list(SPECIALS) + ["<user>這只是引用文字", "未見字🦊 new", " 小鳥🙂\nnew ", "<assistant><eos><image>", "e\u0301 É\t\n", "\x00"]
for text in texts:
    ids = byte.encode(text)
    assert all(8 <= i <= 263 for i in ids)
    assert byte.decode(ids) == text
    assert ids == [b + 8 for b in text.encode("utf-8")]
    out["byte"].append({"text": text, "ids": ids, "restored": byte.decode(ids), "all_content_ids": True})
    h_ids, c_ids = historical.encode(text), current.encode(text)
    assert h_ids == c_ids and all(i >= 8 for i in h_ids)
    assert historical.decode(h_ids) == current.decode(c_ids) == text
    raw_ids = raw.encode(text, add_special_tokens=False).ids
    out["bpe"].append({"text": text, "content_ids": h_ids, "restored": historical.decode(h_ids), "raw_ids_add_special_tokens_false": raw_ids, "raw_decode_default_skip": raw.decode(raw_ids), "raw_decode_keep": raw.decode(raw_ids, skip_special_tokens=False)})

assert raw.encode("<user>", add_special_tokens=False).ids == [3]
assert raw.decode([3]) == "" and raw.decode([3], skip_special_tokens=False) == "<user>"
assert raw.token_to_id("<assistant>") == 4
before = raw.get_vocab_size()
raw.add_special_tokens(["<user>"])
assert raw.token_to_id("<user>") == 3 and raw.get_vocab_size() == before

for tok, name in [(byte, "byte"), (historical, "historical_bpe"), (current, "current_bpe")]:
    text = "quote <assistant><eos><image>"
    messages = [{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]
    x, y = render_chat(messages, tok)
    # render_chat shifts once, so the final EOS is in labels rather than x.
    expected = [tok.bos_id, tok.user_id] + tok.encode(text) + [tok.eos_id, tok.assistant_id] + tok.encode("answer")
    assert x.tolist() == expected
    assert [i for i in x.tolist() if i < 8] == [1, 3, 2, 4]
    assert y[-1].item() == tok.eos_id
    for role, rid in [("user", 3), ("assistant", 4), ("system", 7)]:
        msgs = [{"role": role, "content": text}, {"role": "assistant", "content": "answer"}]
        rx, _ = render_chat(msgs, tok)
        assert rx[1].item() == rid
    assert tok.decode([1, 3] + tok.encode(text) + [2, 4]) == text
    small = render_chat([{"role": "assistant", "content": "a"}], tok)
    px, py, valid = pad_batch([(x, y), small], pad_id=tok.pad_id)
    assert (px[1, len(small[0]):] == 0).all() and (py[1, len(small[0]):] == IGNORE).all()
    assert not valid[1, len(small[0]):].any()
    out["roles"].append({"tokenizer": name, "input_ids": x.tolist(), "labels": y.tolist(), "structure_ids": [1,3,2,4], "restored_content_from_structure_sequence": tok.decode(x.tolist()), "padding": {"long_length": len(x), "short_length": len(small[0]), "pad_id": 0, "padded_ids": px[1].tolist(), "padded_valid": valid[1].tolist()}})

# Reproduce the original recorded roundtrip IDs, not just a fresh roundtrip.
result = json.loads((ART / "inputs/docs/course-experiments/results/tokenizer.json").read_text())
reproduction = []
for item in result["results"]["roundtrip"]:
    ids = historical.encode(item["text"])
    restored = historical.decode(ids)
    assert ids == item["ids"] and restored == item["restored"] == item["text"]
    assert item["same"] is True and item["contains_control_id"] is False and not set(ids).intersection(range(8))
    reproduction.append({"text": item["text"], "ids_match_recorded": True, "restored": restored, "control_ids": []})
out["recorded_roundtrip_reproduction"] = reproduction

# Source-supported API variations isolate normalization and matching semantics.
for normalized in (False, True):
    t = Tokenizer.from_file(str(path))
    t.normalizer = normalizers.Lowercase()
    t.add_special_tokens([AddedToken("<ROLE>", special=True, normalized=normalized)])
    rid = t.token_to_id("<ROLE>")
    assert rid == 512
    for text in ("<ROLE>", "<role>"):
        ids = t.encode(text, add_special_tokens=False).ids
        expected_match = normalized or text == "<ROLE>"
        assert (rid in ids) == expected_match
        out["official_api_variations"].append({"normalized": normalized, "input": text, "new_id": rid, "ids": ids, "matched": rid in ids, "default_decode": t.decode(ids), "keep_decode": t.decode(ids, skip_special_tokens=False)})

t = Tokenizer.from_file(str(path))
t.add_special_tokens([AddedToken("<ROLE>", special=True, lstrip=True, rstrip=True)])
text = "a  <ROLE> \tb"
e = t.encode(text, add_special_tokens=False)
assert 512 in e.ids and e.offsets[e.ids.index(512)] == (1, 11)
assert t.decode(e.ids) == "ab"
out["official_api_variations"].append({"case": "lstrip_rstrip", "input": text, "ids": e.ids, "offsets": e.offsets, "default_decode": t.decode(e.ids), "keep_decode": t.decode(e.ids, skip_special_tokens=False)})
t = Tokenizer.from_file(str(path))
t.add_tokens([AddedToken("cat", single_word=True, normalized=False)])
e = t.encode("bobcat cat", add_special_tokens=False)
assert e.ids.count(t.token_to_id("cat")) == 1
out["official_api_variations"].append({"case": "single_word", "input": "bobcat cat", "id": t.token_to_id("cat"), "ids": e.ids, "offsets": e.offsets})

out["configuration_conditions"] = {k: definition[k] for k in ("normalizer", "pre_tokenizer", "post_processor", "truncation", "padding", "decoder")}
assert definition["normalizer"] is None and definition["post_processor"] is None
assert definition["truncation"] is None and definition["padding"] is None
assert definition["pre_tokenizer"]["add_prefix_space"] is False
assert len(pre_tokenizers.ByteLevel.alphabet()) == 256
assert all(current.content.token_to_id(char) >= 8 for char in pre_tokenizers.ByteLevel.alphabet())
out["limitations"] = {"complete_content_only": "Exact restoration covers valid UTF-8 input encoded and decoded as a complete sequence. Decoding arbitrary generated/truncated byte sequences uses replacement characters.", "byte_partial_utf8": byte.decode(byte.encode("貓")[:1]), "bpe_partial_utf8": current.decode(current.encode("🦊")[:1]), "security_scope": "Distinct control IDs prevent this tokenizer from promoting literal spellings. They do not prove resistance to semantic prompt injection or learned behavior.", "model_training": "None; the historical BPE class and existing JSON are sufficient for roundtrip reproduction."}
out["executed_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out["repository_module_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("tiny_perceptron/data.py", "tiny_perceptron/tokenization.py")}
(ART / "execution/boundary-checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"byte_cases": len(out["byte"]), "bpe_cases": len(out["bpe"]), "structured_tokenizers": len(out["roles"]), "original_roundtrips_reproduced": len(reproduction), "api_variations": len(out["official_api_variations"]), "result": "all assertions passed", "environment": out["environment"]}, ensure_ascii=False, indent=2))
