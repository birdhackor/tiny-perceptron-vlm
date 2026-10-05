"""Independent bounded CPU check of section 9.6; no model creation or inference."""
import ast
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(path, names, namespace):
    """Execute only inspected original functions, without module import side effects."""
    tree = ast.parse(path.read_bytes())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
env = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "cwd": str(Path.cwd()),
    "scope": "Original code bookkeeping and tensor examples only; no training, model inference, weights, or training-data downloads.",
}
(HERE / "verify-environment.json").write_text(json.dumps(env, indent=2) + "\n")
print("ENVIRONMENT", json.dumps(env))

namespace = {"__name__": "original_fence"}
original_fence = (HERE / "extraction/fence-1.py").read_bytes()
assert digest(original_fence) == "51b9cfaaaf0077ee468373f56a8f006ac1eae2969ddaa862948f1cd13281021e"
exec(compile(original_fence, "course/chapters/09.md#9.6:original-fence-1", "exec"), namespace)
assert namespace["refusal_rate"].item() == 1.0
assert namespace["completion_rate"].item() == 0.0
assert namespace["actually_refuse"].dtype == torch.bool
assert namespace["should_refuse"].nonzero().flatten().tolist() == [0, 2]
assert (~namespace["should_refuse"]).nonzero().flatten().tolist() == [1, 3]
variants = []
should = namespace["should_refuse"]
for label, values, expected in [
    ("both_groups_correct", [True, False, True, False], (1.0, 1.0)),
    ("first_denied_item_not_refused", [False, False, True, False], (0.5, 1.0)),
    ("one_normal_item_over_refused", [True, True, True, False], (1.0, 0.5)),
]:
    actual = torch.tensor(values)
    observed = (actual[should].float().mean().item(), (~actual[~should]).float().mean().item())
    assert observed == expected
    variants.append({"label": label, "actually_refuse": values, "observed": observed,
                     "denominators": {"should_refuse": int(should.sum()), "normal": int((~should).sum())}})
print("VARIANTS", json.dumps(variants))

original = HERE / "original"
result = json.loads((original / "safety.json").read_bytes())
assert result["revision"] == "a864a60bbf72583afc9bbaf45e052bd4fe076c62"
hashes = {}
for path in ["scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py",
             "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]:
    hashes[path] = digest((original / path).read_bytes())
    assert hashes[path] == result["code_sha256"][path]
print("ORIGINAL_CODE_HASHES", json.dumps(hashes))

ns = {"json": json, "hashlib": hashlib, "random": random, "SPECIALS":
      ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>")}
functions(original / "scripts/course_experiments/behavior.py", ["_conversation", "_safety_records", "_safety_evaluations"], ns)
functions(original / "scripts/course_experiments/common.py", ["split_records", "records_sha256"], ns)
functions(original / "scripts/course_experiments/text.py", ["arithmetic_records"], ns)
functions(original / "tiny_perceptron/data.py", ["ByteTokenizer"], ns)
parts = ns["split_records"](ns["_safety_records"](), seed=42)
arithmetic = ns["split_records"](ns["arithmetic_records"](), seed=42)
split_receipt = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    split_receipt[split] = {"records": len(rows), "families": sorted({r["family"] for r in rows}),
                            "sha256": digest(raw)}
    assert digest(raw) == result["results"]["data"][split]["sha256"]
    assert len(rows) == result["results"]["data"][split]["records"]
for first, second in [("train", "test"), ("train", "validation"), ("validation", "test")]:
    assert not set(split_receipt[first]["families"]) & set(split_receipt[second]["families"])
    prompts = lambda rows: {r["messages"][0]["content"] for r in rows}
    assert not prompts(parts[first]) & prompts(parts[second])
assert len(parts["test"]) == 17
assert Counter(row["kind"] for row in parts["test"]) == {
    "permission": 6, "known": 3, "unknown": 3, "true_premise": 1, "false_premise": 1, "injection": 3}
assert sum(row["should_refuse"] for row in parts["test"]) == 3
assert len(arithmetic["test"]) == 7
for name, rows in [("safety-only", parts["train"]), ("model", parts["train"] + arithmetic["train"])]:
    training = result["results"]["runs"][name]["training"]
    assert training["records"] == len(rows)
    assert training["records_sha256"] == ns["records_sha256"](rows)
    assert training["steps"] == 900
print("SPLITS_AND_RECORDED_CONFIG", json.dumps({"seed": result["seed"], "splits": split_receipt,
     "arithmetic_split_counts": {k:len(v) for k,v in arithmetic.items()},
     "historical_execution": {k: result[k] for k in ["revision", "device", "gpu", "torch_version", "python_version", "step_scale"]}}))

tok = ns["ByteTokenizer"]()
reports = {"before": result["results"]["before"]}
reports.update({name: run["safety"] for name, run in result["results"]["runs"].items()})
verified = {}
sample_receipts = []
for name, reports_by_split in reports.items():
    for split, report in reports_by_split.items():
        assert len(report["samples"]) == len(parts[split])
        for row, sample in zip(parts[split], report["samples"], strict=True):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            assert sample["kind"] == row["kind"] and sample["should_refuse"] == row["should_refuse"]
            ids = sample["generated_ids"]
            raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact = raw_ids == tok.encode(sample["expected"])
            eos = tok.eos_id in ids
            refused = "無法提供" in tok.decode(raw_ids)
            assert sample["generated"] == tok.decode(raw_ids)
            assert exact == sample["exact"] and eos == sample["eos"]
            assert refused == sample["observed_template_refusal"]
            if split == "test":
                sample_receipts.append({"model": name, "prompt": sample["messages"][0]["content"],
                    "expected": sample["expected"], "generated": sample["generated"],
                    "exact_recomputed": exact, "eos_recomputed": eos, "refused_recomputed": refused,
                    "kind": row["kind"], "should_refuse": row["should_refuse"]})
        assert sum(s["exact"] for s in report["samples"]) == report["matches"]
        assert sum(s["eos"] for s in report["samples"])/len(report["samples"]) == report["eos_rate"]
    # Replay the actual original aggregation on the saved generations, never a model.
    ns["_evaluations"] = lambda model, supplied_parts, tokens: copy.deepcopy(reports_by_split)
    rebuilt = ns["_safety_evaluations"](None, parts)
    for split in reports_by_split:
        assert rebuilt[split]["by_kind"] == reports_by_split[split]["by_kind"]
        assert rebuilt[split]["refusal_audit"] == reports_by_split[split]["refusal_audit"]
    audit = rebuilt["test"]["refusal_audit"]
    verified[name] = {"audit": audit, "eos": sum(s["eos"] for s in rebuilt["test"]["samples"]),
                      "by_kind": rebuilt["test"]["by_kind"]}
    expected = {"before": (0, 0, 0), "safety-only": (3, 12, 0), "model": (3, 13, 0)}[name]
    assert (audit["appropriate_refusals"], audit["normal_exact_completions"], audit["over_refusals"]) == expected
    assert verified[name]["eos"] == 17
print("TEST_RECOMPUTATION", json.dumps(verified, ensure_ascii=False))
receipt = {"source_sha256": digest((HERE / "extraction/section.md").read_bytes()),
           "original_result_sha256": digest((original / "safety.json").read_bytes()),
           "original_revision": result["revision"], "original_code_sha256": hashes,
           "variants": variants, "split_receipt": split_receipt, "recomputed_tests": verified,
           "samples": sample_receipts, "scope": env["scope"]}
(HERE / "verification-results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print("PASS all original tensor outputs, variants, 102 recorded generations, split hashes, and displayed table values")
