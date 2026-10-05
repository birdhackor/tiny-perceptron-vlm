"""Bounded CPU checks of the original character-ID contract and its stated limits."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import unicodedata

OUT = Path(__file__).resolve().parent
original = (OUT / "original.py").read_bytes()
original_state = {}
original_stdout = io.StringIO()
with contextlib.redirect_stdout(original_stdout):
    exec(compile(original, "course/chapters/01.md#1.1:original-fence", "exec"), original_state)
expected_chars = ["。", "狗", "看", "貓", "，"]
expected_ids = [3, 2, 1, 4, 1, 2, 3, 0]
assert original_state["chars"] == expected_chars
assert original_state["ids"] == expected_ids
assert original_state["restored"] == "貓看狗，狗看貓。"
assert original_stdout.getvalue() == "['。', '狗', '看', '貓', '，']\n[3, 2, 1, 4, 1, 2, 3, 0]\n貓看狗，狗看貓。\n"

variant = original.replace(b"chars = sorted(set(text))", b"chars = sorted(set(text), reverse=True)")
(OUT / "reverse.py").write_bytes(variant)
reverse_state = {}
reverse_stdout = io.StringIO()
with contextlib.redirect_stdout(reverse_stdout):
    exec(compile(variant, "course/chapters/01.md#1.1:reverse-exercise", "exec"), reverse_state)
assert reverse_state["chars"] == ["，", "貓", "看", "狗", "。"]
assert reverse_state["ids"] == [1, 2, 3, 0, 3, 2, 1, 4]
assert reverse_state["restored"] == original_state["text"]
changed = [c for c in expected_chars if original_state["to_id"][c] != reverse_state["to_id"][c]]
assert changed == ["。", "狗", "貓", "，"]

try:
    [original_state["to_id"][c] for c in "貓看鳥"]
except KeyError as error:
    assert error.args == ("鳥",)
    unknown = {"exception":"KeyError", "args":list(error.args)}
else:
    raise AssertionError("Unknown character did not raise KeyError")

# The mapping has no arithmetic or next-character prediction stage.
# Exhaust all 5**3 sequences to verify the same-table inverse over this alphabet.
import itertools
count = 0
for sequence in itertools.product(expected_chars, repeat=3):
    text = "".join(sequence)
    ids = [original_state["to_id"][c] for c in text]
    restored = "".join(original_state["chars"][i] for i in ids)
    assert restored == text
    count += 1
assert count == 125

# Verify the intended "character" unit is a Python Unicode code point. The
# section's Chinese characters and punctuation each occupy exactly one point.
code_points = [{"char":c,"decimal":ord(c),"hex":f"U+{ord(c):04X}"} for c in expected_chars]
assert [item["decimal"] for item in code_points] == [12290, 29399, 30475, 35987, 65292]
combining = "e\u0301"
assert list(combining) == ["e", "\u0301"]
assert len(combining) == 2
assert len(unicodedata.normalize("NFC",combining)) == 1

result = {
    "source_code_sha256":hashlib.sha256(original).hexdigest(),
    "original":{"chars":original_state["chars"],"to_id":original_state["to_id"],"ids":original_state["ids"],"restored":original_state["restored"],"stdout":original_stdout.getvalue(),"stdout_lines":len(original_stdout.getvalue().splitlines())},
    "reverse":{"chars":reverse_state["chars"],"to_id":reverse_state["to_id"],"ids":reverse_state["ids"],"restored":reverse_state["restored"],"stdout":reverse_stdout.getvalue(),"characters_with_changed_id":changed,"unchanged_id":{"看":2}},
    "unknown":unknown,
    "alphabet_roundtrips_checked":count,
    "code_points":code_points,
    "character_unit_limit":{"text":combining,"points":list(combining),"length":len(combining),"NFC_length":len(unicodedata.normalize("NFC",combining)),"scope":"Contract handles code points, not general grapheme segmentation; this does not contradict the section's five one-point characters."},
    "all_checks_passed":True,
    "python":sys.version,
    "device":"CPU; no model, training, network, or third-party ML imports",
}
(OUT / "probe-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
