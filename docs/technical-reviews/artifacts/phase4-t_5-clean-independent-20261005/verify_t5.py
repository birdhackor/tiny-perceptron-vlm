"""Independent bounded T.5 check: material, CLI, criteria and provenance; no model capability measurement."""
import ast
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import generate_records
from scripts.train import parser, prepare_examples
from scripts.evaluate import answer_sample, parse_limit
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.alignment import LoRALinear
from scripts.course_experiments.common import split_records


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return [{"function": n.name, "lines": [n.lineno, n.end_lineno], "file_sha256": sha(path.read_bytes())} for n in selected]


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    result = {"purpose": __doc__, "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu", "cuda_available": str(torch.cuda.is_available())}, "commands": [], "inspected_ast": []}
    tok = ByteTokenizer()
    with tempfile.TemporaryDirectory(prefix="t5-clean-cpu-") as td:
        work = Path(td)
        (work / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        env = dict(os.environ)
        env.update(PATH=str(ROOT / ".venv/bin") + os.pathsep + env["PATH"], PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for kind in ("style", "safety"):
            # The two safe preparation commands are the exact shell recipe's argv.
            command = ["python", "scripts/prepare_data.py", "--kind", kind]
            completed = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stderr
            result["commands"].append({"argv": command, "cwd": "isolated temporary workspace; scripts symlink to repository", "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
            parts = {s: [json.loads(x) for x in (work / "data/generated" / kind / (s + ".jsonl")).read_text().splitlines()] for s in ("train", "validation", "test")}
            families = {s: {r["family"] for r in rows} for s, rows in parts.items()}
            assert not any(families[a] & families[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
            args = parser().parse_args(["--task", "sft", "--data", str(work / "data/generated" / kind / "train.jsonl"), "--train", "--steps", "500", "--max-length", "256", "--output", "checkpoints/" + kind + ".pt"])
            assert args.train and args.steps == 500 and args.checkpoint is None
            examples = prepare_examples(args, tok)
            assert all(len(x) <= 256 and bool((y != -100).any()) for x, y in examples)
            result[kind + "_recipe"] = {"split_records": {s: len(rows) for s, rows in parts.items()}, "split_families": {s: len(f) for s, f in families.items()}, "family_disjoint": True, "max_input_tokens": max(len(x) for x, y in examples), "supervised_answer_tokens_train": sum(int((y != -100).sum()) for x, y in examples), "independent_random_initialization": args.checkpoint is None}
            # Bounded substitute: one gradient diagnostic, --train removed. No weights are saved.
            command = ["python", "scripts/train.py", "--task", "sft", "--data", "data/generated/" + kind + "/train.jsonl", "--steps", "1", "--max-length", "256", "--device", "cpu", "--output", "checkpoints/" + kind + ".pt"]
            completed = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stderr
            assert not (work / "checkpoints" / (kind + ".pt")).exists()
            result["commands"].append({"argv": command, "bounded_substitution": "500-step weight updating command replaced by one gradient diagnostic without --train", "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "weights_saved": False})
        for command in (["python", "scripts/evaluate.py", "--help"], ["python", "scripts/fetch_training_assets.py", "--help"], ["python", "-m", "scripts.course_experiments.run", "--list-assets", "style"], ["python", "-m", "scripts.course_experiments.run", "--list-assets", "safety"], ["python", "-m", "scripts.course_experiments.run", "--list-assets", "lora"]):
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stderr
            result["commands"].append({"argv": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    assert parse_limit("all") is None
    assert parse_limit("1") == 1
    targets = [generate_records("style")[i]["messages"][-1]["content"] for i in range(3)]
    assert targets[0] == "0" and targets[1].startswith("0，像") and json.loads(targets[2]) == {"answer": 0}
    result["handwritten_example"] = {"sum": 2 + 3, "concise_sentence": "2+3等於5。", "vivid_sentence": "像把2顆星與3顆星收進同一個籃子，合起來是5顆星。", "sentence_count_each": 1, "type": "human-authored criteria example, not generated by a model"}
    variants = {"correct_eos": tok.encode("5") + [tok.eos_id], "wrong_number_eos": tok.encode("6") + [tok.eos_id], "correct_no_eos": tok.encode("5"), "invalid_special_eos": tok.encode("5") + [tok.user_id, tok.eos_id]}
    judged = {name: answer_sample(tok, ids, "5") for name, ids in variants.items()}
    assert judged["correct_eos"]["completed_exact_match"]
    assert not judged["wrong_number_eos"]["exact_match"]
    assert judged["correct_no_eos"]["exact_match"] and not judged["correct_no_eos"]["completed_exact_match"]
    assert not judged["invalid_special_eos"]["exact_match"]
    result["artificial_evaluator_variants"] = {name: {k: x[k] for k in ["exact_match", "completed_exact_match", "eos", "invalid_special_tokens"]} for name, x in judged.items()}
    original = HERE / "original-code/safety/scripts/course_experiments"
    ns = {"json": json, "hashlib": hashlib, "split_records": split_records}
    result["inspected_ast"] += load_functions(original / "text.py", ["_json_bytes", "_digest", "_utf8_prefix", "arithmetic_records"], ns)
    result["inspected_ast"] += load_functions(original / "behavior.py", ["_conversation", "_safety_records", "_style_record"], ns)
    fixed = split_records(ns["_safety_records"](), seed=42)
    fixed_arithmetic = split_records(ns["arithmetic_records"](), seed=42)
    existing = json.loads((HERE / "frozen/docs/course-experiments/results/safety.json").read_text())
    checked_pointers = []
    for split, rows in fixed.items():
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        expected = existing["results"]["data"][split]
        assert len(rows) == expected["records"] and sha(payload) == expected["sha256"]
        checked_pointers.extend(["/results/data/" + split + "/records", "/results/data/" + split + "/sha256"])
    normal = [r for r in fixed["test"] if not r["should_refuse"]]
    denied = [r for r in fixed["test"] if r["should_refuse"]]
    assert all(r["kind"] == "permission" for r in denied)
    assert any(r["kind"] == "unknown" for r in normal)
    result["fixed_safety_material"] = {"rows": {s: len(r) for s, r in fixed.items()}, "test_should_refuse": len(denied), "test_normal_including_clarification": len(normal), "arithmetic_replay_rows": len(fixed_arithmetic["train"]), "fresh_preparation_recipe_is_different": True, "original_data_fingerprints_match": True}
    source = HERE / "raw/pku-train-first-100.jsonl"
    assert sha(source.read_bytes()) == "c4a88d08ef7456669766f1a3b908f82d918be5625a24794d9843d7f6df1d552b"
    raw = [json.loads(x) for x in source.read_text().splitlines()]
    ns["_asset_rows"] = lambda *args: raw
    # Execute only original raw-selection and split statements. Model construction/training/evaluation are absent.
    tree = ast.parse((original / "behavior.py").read_bytes())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pku_pilot"))
    assert isinstance(fn.body[3], ast.Assign) and ast.unparse(fn.body[3].targets[0]) == "parts"
    fn.body = fn.body[:4] + [ast.Return(value=ast.Tuple(elts=[ast.Name(id="records", ctx=ast.Load()), ast.Name(id="parts", ctx=ast.Load())], ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, "original _pku_pilot selection only, lines 455-473", "exec"), ns)
    from types import SimpleNamespace
    selected, parts = ns["_pku_pilot"](SimpleNamespace(seed=42))
    assert len(raw) == existing["results"]["pku_pilot"]["source_records"]
    assert len(selected) == existing["results"]["pku_pilot"]["safe_selected_records"]
    checked_pointers.extend(["/results/pku_pilot/source_records", "/results/pku_pilot/safe_selected_records"])
    for split, rows in parts.items():
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        expected = existing["results"]["pku_pilot"]["data"][split]
        assert len(rows) == expected["records"] and sha(payload) == expected["sha256"]
        checked_pointers.extend(["/results/pku_pilot/data/" + split + "/records", "/results/pku_pilot/data/" + split + "/sha256"])
    assert all(raw[r["source_row"]]["is_response_" + str(r["safer_response_id"]) + "_safe"] for r in selected)
    assert all(len(m["content"].encode()) <= 120 for r in selected for m in r["messages"])
    assert all(r["source_record_sha256"] == ns["_digest"](raw[r["source_row"]]) for r in selected)
    unsafe_both = [i for i, r in enumerate(raw) if not r["is_response_0_safe"] and not r["is_response_1_safe"]]
    assert not set(unsafe_both) & {r["source_row"] for r in selected}
    result["pku_selection"] = {"source_records": len(raw), "selected_safe_records": len(selected), "both_unsafe_rows_discarded": len(unsafe_both), "split_records": {s: len(rows) for s, rows in parts.items()}, "source_fingerprints_and_all_split_hashes_match": True, "max_prompt_or_answer_utf8_bytes": 120, "source_criteria": "ranked safer response AND its individual safety Boolean; not better_response_id", "human_scope": "cropped fragments preserve source-row provenance, not a newly verified safety annotation for each fragment"}
    result["pku_named_fragment_inspection"] = {"source_row": selected[0]["source_row"], "raw_json_pointer": "/" + str(selected[0]["source_row"]), "safer_response_id": selected[0]["safer_response_id"], "individual_safe": raw[selected[0]["source_row"]]["is_response_" + str(selected[0]["safer_response_id"]) + "_safe"], "prompt_excerpt": selected[0]["messages"][0]["content"], "answer_excerpt": selected[0]["messages"][1]["content"]}
    # Test ranking-vs-safety logic on a deliberately both-unsafe pair with no natural content.
    ns["_asset_rows"] = lambda *args: [dict(raw[0], is_response_0_safe=False, is_response_1_safe=False)]
    ns["split_records"] = lambda rows, **kwargs: {"unpartitioned": rows}
    excluded, _ = ns["_pku_pilot"](SimpleNamespace(seed=42))
    assert excluded == []
    result["artificial_both_unsafe_variant"] = "rank remains defined, but zero SFT records are selected"
    for split in ["validation", "test"]:
        keys = existing["results"]["pku_pilot"]["evaluation"][split].keys()
        assert "samples" not in keys
        result.setdefault("public_pku_aggregate_keys", {})[split] = list(keys)
        checked_pointers.append("/results/pku_pilot/evaluation/" + split + " (key names/types only)")
    published = json.loads((HERE / "upstream/published-safety-export-manifest.json").read_text())
    assert {x["output"] for x in published["files"]} == {"safety-only.pt", "model.pt"}
    result["public_safety_exports"] = [x["output"] for x in published["files"]]
    torch.manual_seed(42)
    base = torch.nn.Linear(3, 2)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.tensor([[1., 2., 3.]])
    with torch.no_grad():
        layer.b.fill_(0.25)
        before = layer(x)
        merged = copy.deepcopy(base)
        merged.weight.copy_(layer.merged_weight())
        diff = float((before - merged(x)).abs().max())
        doubled = merged(x) + (x @ layer.a.T @ layer.b.T) * (layer.alpha / layer.rank)
        wrong_diff = float((doubled - before).abs().max())
    assert diff < 1e-6 and wrong_diff > 0
    result["lora_merge_variant"] = {"merged_vs_adapter_max_abs_difference": diff, "reapplying_same_adapter_changes_output": wrong_diff, "tolerance": "1e-6 for equivalence", "no_weights_saved": True}
    result["json_pointers_actually_inspected"] = checked_pointers
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["commands", "inspected_ast"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
