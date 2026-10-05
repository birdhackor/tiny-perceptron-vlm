"""Inspect original pointers and reproduce data preparation only, on 100 existing rows."""
import ast
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
ORIGINAL = BASE / "original-records"
ROWS = ORIGINAL / "train-first-100.jsonl"
raw_lines = ROWS.read_bytes().splitlines()
raw = [json.loads(line) for line in raw_lines]
assert len(raw) == 100
source_asset = json.loads((ORIGINAL / "asset.json").read_bytes())
source_api = json.loads((ORIGINAL / "source-api.json").read_bytes())
viewer = json.loads((ORIGINAL / "source-viewer-response.json").read_bytes())
manifest_rows = [json.loads(line) for line in (ORIGINAL / "rows.manifest.jsonl").read_bytes().splitlines()]
raw_result = json.loads((ORIGINAL / "original-dpo-result.json").read_bytes())
dataset_revision = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
assert source_asset["dataset"] == source_api["id"] == "HuggingFaceH4/ultrafeedback_binarized"
assert source_asset["revision"] == source_api["sha"] == dataset_revision
assert source_asset["split"] == "train_prefs"
raw_sha = hashlib.sha256(ROWS.read_bytes()).hexdigest()
assert source_asset["raw_training_file"]["sha256"] == raw_sha
result_asset = raw_result["assets"][0]
assert result_asset["id"] == "ultrafeedback-dpo"
for basename in ["train-first-100.jsonl", "rows.manifest.jsonl", "source-viewer-response.json", "source-api.json", "asset.json"]:
    reported = next(item for item in result_asset["files"] if item["path"].endswith("/" + basename))
    assert reported["sha256"] == hashlib.sha256((ORIGINAL / basename).read_bytes()).hexdigest()
assert len(viewer["rows"]) == len(manifest_rows) == 100
assert viewer["num_rows_total"] == 61135 and viewer["partial"] is False
for i, (row, delivered, provenance) in enumerate(zip(raw, viewer["rows"], manifest_rows)):
    assert delivered["row_idx"] == provenance["source_row_index"] == provenance["local_row"] == i
    assert delivered["row"] == row and delivered["truncated_cells"] == []
    assert provenance["revision"] == dataset_revision
    assert provenance["source_split"] == "train_prefs" and provenance["source_config"] == "default"
    for side in ["chosen", "rejected"]:
        assert row[side][0]["role"] == "user" and row[side][0]["content"] == row["prompt"]
        assert row[side][-1]["role"] == "assistant"

namespace = {"json": json, "hashlib": hashlib, "random": random, "Path": Path}
def load_selected_functions(path, names):
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    compile_tree = ast.Module(body=selected, type_ignores=[])
    exec(compile(compile_tree, str(path), "exec"), namespace)

text_source = BASE / "sources/experiment-original-text.py"
behavior_source = BASE / "sources/experiment-original-behavior.py"
common_source = ROOT / "scripts/course_experiments/common.py"
assert hashlib.sha256(text_source.read_bytes()).hexdigest() == raw_result["code_sha256"]["scripts/course_experiments/text.py"]
assert hashlib.sha256(behavior_source.read_bytes()).hexdigest() == raw_result["code_sha256"]["scripts/course_experiments/behavior.py"]
assert hashlib.sha256(common_source.read_bytes()).hexdigest() == raw_result["code_sha256"]["scripts/course_experiments/common.py"]
load_selected_functions(text_source, ["_json_bytes", "_digest", "_utf8_prefix", "_save_splits"])
load_selected_functions(common_source, ["write_json", "split_records"])
tree = ast.parse(behavior_source.read_bytes())
original_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_natural_dpo_pilot")
preparation = copy.deepcopy(original_fn)
# Execute only original statements 656-678: asset rows, excerpts and family split.
# The SFT and DPO training statements are excluded from the compiled function.
preparation.body = [copy.deepcopy(node) for node in original_fn.body if node.lineno <= 678]
preparation.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id="records", ctx=ast.Load()), ast.Name(id="parts", ctx=ast.Load())], ctx=ast.Load())))
compile_tree = ast.fix_missing_locations(ast.Module(body=[preparation], type_ignores=[]))
namespace["_asset_rows"] = lambda context, identifier, basename: raw
exec(compile(compile_tree, str(behavior_source) + ":data-preparation-only", "exec"), namespace)
context = SimpleNamespace(seed=raw_result["seed"], output=BASE / "reconstructed-ultrafeedback")
records, parts = namespace["_natural_dpo_pilot"](context)
generated_manifest = namespace["_save_splits"](context, parts, name="ultrafeedback-excerpts")
raw_data = raw_result["results"]["ultrafeedback_pilot"]["data"]
assert raw_result["results"]["ultrafeedback_pilot"]["source_records"] == len(raw)
for split, count in [("train", 80), ("validation", 10), ("test", 10)]:
    assert generated_manifest[split]["records"] == generated_manifest[split]["families"] == count
    for key in ["records", "families", "sha256", "path"]:
        assert generated_manifest[split][key] == raw_data[split][key], (split, key)
families = {split: {row["family"] for row in rows} for split, rows in parts.items()}
assert all(not families[a] & families[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
prefix = namespace["_utf8_prefix"]
for text, expected in [("A" * 119 + "你Z", "A" * 119), ("A" * 118 + "🙂", "A" * 118), ("é" * 61, "é" * 60)]:
    assert prefix(text, 120) == expected
    assert len(prefix(text, 120).encode()) <= 120
fields = {}
for side in ["prompt", "chosen", "rejected"]:
    original_texts = [row["prompt"] if side == "prompt" else next(message["content"] for message in reversed(row[side]) if message["role"] == "assistant") for row in raw]
    excerpt_texts = [row[side] for row in records]
    assert all(text == prefix(original, 120) for original, text in zip(original_texts, excerpt_texts))
    assert all(len(text.encode("utf-8")) <= 120 for text in excerpt_texts)
    fields[side] = {"source_max_bytes": max(len(text.encode("utf-8")) for text in original_texts),
                    "excerpt_max_bytes": max(len(text.encode("utf-8")) for text in excerpt_texts),
                    "truncated_records": sum(text != original for original, text in zip(original_texts, excerpt_texts))}
assert len({row["family"] for row in records}) == 100
result = {"dataset": source_asset["dataset"], "revision": dataset_revision, "original_official_split": "train_prefs",
          "raw_rows": len(raw), "families": 100, "raw_sha256": raw_sha,
          "source_viewer_rows_match_jsonl": True, "source_viewer_truncated_cells": 0,
          "split_counts": {name: len(rows) for name, rows in parts.items()},
          "split_sha256_match_original_measurement": True, "family_overlap": 0,
          "fields": fields, "identical_chosen_rejected_excerpts": sum(row["chosen"] == row["rejected"] for row in records),
          "complete_utf8_codepoint_variants": True,
          "read_pointers": ["/results/ultrafeedback_pilot/source_records", "/results/ultrafeedback_pilot/data/{train,validation,test}/{records,families,sha256,path}",
                            "/seed", "/revision", "/assets/0/{id,target,version,files}", "/code_sha256/{scripts/course_experiments/behavior.py,scripts/course_experiments/text.py,scripts/course_experiments/common.py,tiny_perceptron/data.py}",
                            "asset.json: /{id,dataset,revision,config,split,rows,raw_training_file}", "source-api.json: /{id,sha}",
                            "source-viewer-response.json: /{num_rows_total,num_rows_per_page,partial}, /rows/*/{row_idx,row,truncated_cells}",
                            "rows.manifest.jsonl: /{local_row,source_row_index,revision,source_split,source_config}",
                            "train-first-100.jsonl: /{prompt,prompt_id,chosen,rejected}; full records read by original _digest without printing extra values"],
          "scope": "Reproduction of original data preparation and original raw count/hash checks; no downloads, model loading, training or re-evaluation."}
(BASE / "ultrafeedback-verification-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
