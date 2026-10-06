"""Independent bounded CPU check of the exact frozen 1.1 fence and its variants."""
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

A = Path(__file__).resolve().parent
code = (A / "extracted/fence-1.py").read_bytes()
original = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, "course/chapters/01.md:line-24:fence-1", "exec"), original)
expected_chars = ["。", "狗", "看", "貓", "，"]
expected_ids = [3, 2, 1, 4, 1, 2, 3, 0]
assert original["chars"] == expected_chars
assert original["ids"] == expected_ids
assert original["restored"] == "貓看狗，狗看貓。"
assert stdout.getvalue().splitlines() == [str(expected_chars), str(expected_ids), original["text"]]
assert len(original["ids"]) == 8 and len(original["chars"]) == 5
assert list(map(ord, original["chars"])) == [12290, 29399, 30475, 35987, 65292]

reverse_code = code.replace(b"chars = sorted(set(text))", b"chars = sorted(set(text), reverse=True)")
assert reverse_code != code
(A / "reverse-fence.py").write_bytes(reverse_code)
reverse = {}
reverse_stdout = io.StringIO()
with contextlib.redirect_stdout(reverse_stdout):
    exec(compile(reverse_code, "1.1 reverse=True exercise", "exec"), reverse)
assert reverse["chars"] == ["，", "貓", "看", "狗", "。"]
assert reverse["ids"] == [1, 2, 3, 0, 3, 2, 1, 4]
assert reverse["restored"] == original["text"]
cross_decode = "".join(reverse["chars"][i] for i in original["ids"])
assert cross_decode == "狗看貓。貓看狗，" and cross_decode != original["text"]
try:
    [original["to_id"][char] for char in "貓看鳥"]
except KeyError as error:
    unknown = {"class": type(error).__name__, "args": list(error.args)}
else:
    raise AssertionError("The absent character must raise KeyError")
assert unknown == {"class": "KeyError", "args": ["鳥"]}

notebook = json.loads((A / "frozen/notebooks/01/1.1.ipynb").read_text())
notebook_code = "".join(notebook["cells"][5]["source"]).encode("utf-8")
assert code == notebook_code + b"\n"

# Compile the unmodified class only, avoiding unrelated torch/data functions.
data = (A / "frozen/tiny_perceptron/data.py").read_bytes()
tree = ast.parse(data)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CharTokenizer")
implementation = ast.Module(body=[node], type_ignores=[])
repository = {}
exec(compile(implementation, "tiny_perceptron/data.py:CharTokenizer:31-43", "exec"), repository)
tokenizer = repository["CharTokenizer"](original["text"])
assert tokenizer.chars == ["<unk>"] + expected_chars
assert tokenizer.encode(original["text"]) == [4, 3, 2, 5, 2, 3, 4, 1]
assert tokenizer.decode(tokenizer.encode(original["text"])) == original["text"]
assert tokenizer.encode("貓看鳥") == [4, 3, 0]
assert tokenizer.decode(tokenizer.encode("貓看鳥")) == "貓看<unk>"

result = {
    "environment": {"python": sys.version, "python_executable": sys.executable,
                    "platform": platform.platform(), "device": "CPU; stdlib only; no training or GPU", "network": "none in this audit"},
    "fence_sha256": hashlib.sha256(code).hexdigest(),
    "original": {"chars": original["chars"], "ids": original["ids"], "restored": original["restored"],
                 "stdout": stdout.getvalue(), "unicode_code_points": list(map(ord, original["chars"])),
                 "characters": len(original["ids"]), "vocabulary_size": len(original["chars"])},
    "reverse": {"chars": reverse["chars"], "ids": reverse["ids"], "restored": reverse["restored"], "stdout": reverse_stdout.getvalue()},
    "cross_table_decode": cross_decode, "unknown_character": unknown,
    "notebook_cell_5_match": {"terminal_LF_difference_only": True,
                              "fence_bytes": len(code), "notebook_code_bytes": len(notebook_code),
                              "notebook_code_sha256": hashlib.sha256(notebook_code).hexdigest()},
    "repository_contract": {"inspection": "Unmodified CharTokenizer AST, lines 31-43", "chars": tokenizer.chars,
                            "known_ids": tokenizer.encode(original["text"]), "unknown_ids": tokenizer.encode("貓看鳥"),
                            "unknown_decoded": tokenizer.decode(tokenizer.encode("貓看鳥"))},
    "outcome": "All exact assertions passed; this establishes lookup and round-trip behavior, not prediction or semantic understanding."
}
(A / "cpu-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
