"""Bounded CPU verification of A.8's paper example; no model inference/training."""
import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
ART = OUT.parent
raw = (ART / "inputs/section.md").read_bytes()
section = raw.decode("utf-8")
table = re.findall(r"^\| (T|U[1-7]) \| (.+) \|$", section, re.M)
assert len(table) == 8 and len(dict(table)) == 8
records = dict(table)
distractors = [f"U{i}" for i in range(1, 8)]
question = "攝影社的活動室代碼是什麼？只回答代碼"
orders = {}
for name, index in [("first", 0), ("middle", 3), ("last", 7)]:
    order = distractors.copy()
    order.insert(index, "T")
    assert len(order) == 8 and set(order) == set(records)
    assert [label for label in order if label != "T"] == distractors
    answers = [re.fullmatch(r"攝影社的活動室代碼是 ([A-Z][0-9]{2})。", records[x]).group(1)
               for x in order if re.fullmatch(r"攝影社的活動室代碼是 ([A-Z][0-9]{2})。", records[x])]
    assert answers == ["M17"]
    orders[name] = order

# Execute only the original ByteTokenizer class and SPECIALS assignment found by AST.
# This avoids importing unrelated dataset functions; source bytes remain unchanged.
data_path = ROOT / "tiny_perceptron/data.py"
data_raw = data_path.read_bytes()
tree = ast.parse(data_raw)
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ByteTokenizer"
         or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPECIALS" for t in n.targets)]
assert len(nodes) == 2
ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(data_path), "exec"), ns)
tokenizer = ns["ByteTokenizer"]()
prefix = "資料：\n"
suffix = "\n問題：" + question
measurements = {}
for name, order in orders.items():
    content = prefix + "\n".join(records[x] for x in order) + suffix
    ids = [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode(content) + [tokenizer.assistant_id]
    assert tokenizer.decode(ids) == content
    assert content.endswith(suffix) and all(content.count(records[x]) == 1 for x in order)
    clue_offset = len(tokenizer.encode(content[:content.index(records["T"])])) + 2
    measurements[name] = {"order": order, "clue_record_1_based": order.index("T") + 1,
                          "clue_first_id_0_based": clue_offset, "input_ids": ids,
                          "input_token_count": len(ids), "unicode_character_count": len(content),
                          "fixed_new_token_cap": 8, "paper_answer": "M17"}
assert len({m["input_token_count"] for m in measurements.values()}) == 1
assert len({m["unicode_character_count"] for m in measurements.values()}) == 1
assert [m["clue_record_1_based"] for m in measurements.values()] == [1, 4, 8]
assert tokenizer.encode("攝") == [b + 8 for b in "攝".encode("utf-8")]
assert len("攝") == 1 and len(tokenizer.encode("攝")) == 3
short_content = prefix + records["T"] + suffix
short_ids = [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode(short_content) + [tokenizer.assistant_id]
assert len(short_ids) < measurements["first"]["input_token_count"]

# Reconstruct visible SVG rows from text coordinates, independently of alt text.
svg_path = ART / "inputs/course/figures/rewrite-A-clue-position.svg"
svg = ET.fromstring(svg_path.read_bytes())
visible = list(svg.iter("{http://www.w3.org/2000/svg}text"))
svg_rows = []
for y in [182, 288, 394]:
    labels = sorted((float(e.get("x")), e.text) for e in visible
                    if float(e.get("y", "0")) == y and e.get("class") == "value")
    svg_rows.append(["T" if text == "M17" else text for _, text in labels])
assert svg_rows == list(orders.values())
clue_rects = [e for e in svg.iter("{http://www.w3.org/2000/svg}rect") if e.get("class") == "clue"]
assert [(int(e.get("x")), int(e.get("y"))) for e in clue_rects] == [(104, 140), (377, 246), (741, 352)]
assert len([e for e in svg.iter("{http://www.w3.org/2000/svg}rect")
            if e.get("class") in ["card", "clue"]]) == 24

# Defined diagnostic examples, not generated outputs or a model score.
diagnostics = []
for text, eos, exhausted in [("M17", True, False), ("B04", True, False), ("M", False, True)]:
    diagnostics.append({"illustrative_output": text, "content_correct": text == "M17",
                        "has_eos": eos, "budget_exhausted": exhausted})
assert [d["content_correct"] for d in diagnostics] == [True, False, False]

result = {"scope": "Paper-input construction and deterministic CPU checks only; no measured model output or accuracy.",
          "source_section_sha256": hashlib.sha256(raw).hexdigest(),
          "byte_tokenizer_source_sha256": hashlib.sha256(data_raw).hexdigest(),
          "byte_tokenizer_inspected_lines": "11,14–28; AST-selected original assignment/class execution",
          "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                          "tokenizer": "repository ByteTokenizer, untrained UTF-8 bytes"},
          "counts": {"records_per_version": 8, "controlled_versions": 3, "svg_cards": 24},
          "measurements": measurements, "short_diagnostic_input_tokens": len(short_ids),
          "char_vs_token_example": {"text": "攝", "characters": 1, "byte_token_ids": tokenizer.encode("攝")},
          "svg_rows": svg_rows, "diagnostic_examples": diagnostics,
          "tolerance": "Exact equality for integers, labels, records, original bytes and token IDs.",
          "assertions": "all passed"}
(OUT / "audit-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"assertions": "all passed", "counts": result["counts"],
                  "positions": [m["clue_record_1_based"] for m in measurements.values()],
                  "full_input_tokens": [m["input_token_count"] for m in measurements.values()],
                  "clue_first_ids_0_based": [m["clue_first_id_0_based"] for m in measurements.values()],
                  "short_input_tokens": len(short_ids), "scope": result["scope"]}, ensure_ascii=False))
