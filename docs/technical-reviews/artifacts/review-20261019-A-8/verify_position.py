"""核對原文的排列、UTF-8 長度、練習及接縫切詞的可能性。"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tokenizers
import torch
from tokenizers import Tokenizer, models
from tiny_perceptron.data import ByteTokenizer

snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/A.8.md"
raw = snapshot.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(snapshot), "exec"), namespace)

exercise = io.StringIO()
with contextlib.redirect_stdout(exercise):
    exec(compile(code.replace('(\"中\", 3)', '(\"中\", 5)'), str(snapshot), "exec"), {})

clue, others, question = (namespace[key] for key in ("clue", "others", "question"))
tok = ByteTokenizer()
variants = []
for index in (0, 3, 7, 5):
    records = others[:index] + [clue] + others[index:]
    prompt = "\n".join(records) + "\n" + question
    ids = tok.encode(prompt)
    assert len(records) == len(set(records)) == 8
    assert records.count(clue) == 1
    assert records.index(clue) + 1 == index + 1
    assert [r for r in records if r != clue] == others
    assert len(ids) == len(prompt.encode("utf-8")) == 277
    assert prompt.splitlines()[-1] == question
    assert tok.decode(ids) == prompt
    variants.append({
        "insertion_index": index,
        "record_position": records.index(clue) + 1,
        "record_labels": ["N" if r == clue else "U" + str(others.index(r) + 1) for r in records],
        "byte_tokens": len(ids),
        "same_order": [r for r in records if r != clue] == others,
        "bytes_between_clue_and_question": len(prompt[prompt.index(clue) + len(clue):prompt.index(question)].encode()),
    })

bpe = Tokenizer(models.BPE(vocab={"a": 0, "b": 1, "\n": 2, "\nb": 3}, merges=[("\n", "b")]))
boundary_examples = []
for records in (["a", "b"], ["b", "a"]):
    prompt = "\n".join(records) + "\na"
    encoded = bpe.encode(prompt)
    boundary_examples.append({"records": records, "question": "a", "prompt": prompt, "tokens": encoded.tokens, "count": len(encoded.ids)})
assert [x["count"] for x in boundary_examples] == [4, 5]

result = {
    "environment": {"python": sys.version, "torch": torch.__version__, "tokenizers": tokenizers.__version__, "device": "cpu"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_stdout": stdout.getvalue(),
    "exercise_stdout": exercise.getvalue(),
    "length_derivation": {
        "record_byte_lengths_N_U1_to_U7": [len(r.encode()) for r in [clue] + others],
        "question_bytes": len(question.encode()),
        "newline_bytes": 8,
        "sum": sum(len(r.encode()) for r in [clue] + others) + len(question.encode()) + 8,
    },
    "variants": variants,
    "boundary_bpe": {
        "configuration": "BPE，僅合併換行與 b；無 pre-tokenizer、normalizer 或 post-processor。",
        "examples": boundary_examples,
        "scope": "只證明接縫合併可能讓重排後長度不同；不宣稱每種 tokenizer 或本文八筆紀錄一定如此。",
    },
    "result": "所有斷言通過；沒有模型生成或品質測量。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
