"""Independent CPU audit of existing raw outputs; no model generation/training."""
import ast
import hashlib
import json
import platform
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
original = ROOT / "docs/course-experiments/results/rag.json"
retained = BASE / "primary/rag-original.json"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(original) == digest(retained)
record = json.loads(retained.read_bytes())
assert record["experiment_id"] == "rag" and record["seed"] == 42
code = ROOT / "scripts/course_experiments/applications.py"
assert digest(code) == record["code_sha256"]["scripts/course_experiments/applications.py"]
assert digest(ROOT / "tiny_perceptron/data.py") == record["code_sha256"]["tiny_perceptron/data.py"]
source_ast = ast.parse(code.read_bytes())
grounding_ast = next(n for n in source_ast.body if isinstance(n, ast.FunctionDef) and n.name == "_grounding")
namespace = {"re": re}
exec(compile(ast.Module(body=[grounding_ast], type_ignores=[]), str(code) + ":_grounding", "exec"), namespace)

toy_source = {"id": "d2", "text": "書店地址是青街8號"}
toy_cases = [("青街8號", "d2", (True, True)), ("紅街9號", "d2", (True, False)),
             ("青街8號", "d9", (False, False))]
toy = []
for address, citation, expected in toy_cases:
    valid = citation == toy_source["id"]
    result = (valid, valid and address in toy_source["text"])
    assert result == expected
    toy.append({"address": address, "citation": citation, "observed": result})
negative_literal = "青街8號" in "書店地址不是青街8號"
paraphrase_literal = "青街8號" in "書店位於青街八號"
assert negative_literal is True and paraphrase_literal is False

rows = record["results"]["samples"]
selected = [(i, r) for i, r in enumerate(rows) if r["mode"] in ("correct_with_distractor", "distractor_only")]
assert len(selected) == 24
audited = []
for index, row in selected:
    generation = row["generation"]
    assert generation["candidate_count"] == 1 and generation["temperature"] == 0.0
    assert generation["max_new_tokens"] == 16 and len(generation["samples"]) == 1
    sample = generation["samples"][0]
    ids = sample["generated_ids"]
    raw = ids[:-1] if ids and ids[-1] == 2 else ids
    # ByteTokenizer's exact retained version assigns byte b to b+8, EOS to 2.
    text = bytes(i - 8 for i in raw if i >= 8).decode("utf-8", errors="replace")
    assert text == sample["generated"]
    assert sample["invalid_special_tokens"] == [i for i in raw if i < 8]
    assert sample["eos"] == bool(ids and ids[-1] == 2)
    assert sample["generated_tokens"] == len(ids)
    docs = row["documents"]
    assert len({d["id"] for d in docs}) == len(docs)
    structured = {}
    for d in docs:
        key, field = d["text"].split()
        assert field.startswith("address=") and len(field.split("=", 1)[1]) == 2
        structured[d["id"]] = (key, field.split("=", 1)[1])
    prompt = "Docs:\n" + "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    prompt += f"\nFind:{row['fact']['key']}\nReply:address[source] or UNKNOWN"
    assert generation["messages"] == [{"role": "user", "content": prompt}]
    if text == "UNKNOWN":
        address = citation = None
    else:
        if len(text) == 6 and text[2] == "[" and text[-1] == "]" and text[0] in "ABCDE" and text[1] in "0123456789" and text[3] == "D" and text[4] in "0123456789":
            address, citation = text[:2], text[3:5]
        else:
            address = citation = None
    clean = not sample["invalid_special_tokens"]
    valid = clean and citation in structured
    supports = clean and citation in structured and structured[citation][1] == address
    correct_address = clean and address == row["fact"]["address"]
    correct = clean and text == f"{row['fact']['address']}[{row['fact']['source']}]"
    assert row["parsed_address"] == address and row["parsed_citation"] == citation
    assert row["citation_valid"] == valid
    assert row["supported_by_cited_source"] == supports
    assert row["fact_answer_correct"] == correct
    native = namespace["_grounding"](sample, docs, row["expected"])
    assert all(row[k] == v for k, v in native.items())
    audited.append({"original_sample_index": index, "family": row["family"], "mode": row["mode"],
                    "generated": text, "documents": docs, "valid": valid, "supports": supports,
                    "correct_address": correct_address, "correct_with_source": correct})

counts = {}
for mode in ("correct_with_distractor", "distractor_only"):
    group = [r for r in audited if r["mode"] == mode]
    assert len(group) == 12 and len({r["family"] for r in group}) == 12
    counts[mode] = {name: sum(r[name] for r in group) for name in ("valid", "supports", "correct_address", "correct_with_source")}
    counts[mode]["denominator"] = len(group)
    for our, metric in (("valid", "citation_valid"), ("supports", "supported_by_cited_source"), ("correct_with_source", "fact_answer_correct")):
        saved = record["results"]["metrics"][mode][metric]
        assert saved["numerator"] == counts[mode][our] and saved["denominator"] == len(group)
        assert saved["rate"] == counts[mode][our] / len(group)
assert counts["correct_with_distractor"] == {"valid": 12, "supports": 2, "correct_address": 2, "correct_with_source": 2, "denominator": 12}
assert counts["distractor_only"] == {"valid": 5, "supports": 5, "correct_address": 0, "correct_with_source": 0, "denominator": 12}
k003 = next(r for r in audited if r["family"] == "K003" and r["mode"] == "correct_with_distractor")
assert k003["generated"] == "C9[D9]" and k003["valid"] and not k003["supports"]
assert {d["id"]: d["text"] for d in k003["documents"]} == {"D0": "K013 address=C9", "D9": "K003 address=D2"}
summary = {"kind": "independent_existing_raw_audit", "python": platform.python_version(), "device": "cpu",
           "original_result_sha256": digest(original), "retained_result_sha256": digest(retained),
           "original_revision": record["revision"], "original_device": record["device"],
           "original_torch": record["torch_version"], "original_python": record["python_version"],
           "code_sha_matches_original_result": True, "toy": toy,
           "substring_negation_returns_true": negative_literal, "paraphrase_substring_returns_false": paraphrase_literal,
           "counts": counts, "audited_rows": audited,
           "scope": "Existing token IDs, prompt documents and scoring only; no model weights loaded and no generated model answers."}
(BASE / "execution/recompute-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("kind", "python", "device", "original_result_sha256", "counts", "toy", "substring_negation_returns_true", "paraphrase_substring_returns_false", "scope")}, ensure_ascii=False, indent=2))
