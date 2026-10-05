"""Own bounded CPU audit: original fence, pure data methods, saved measurement pointers.

No model is loaded, evaluated or trained. Only named AST definitions are compiled.
"""
import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent
ns = {"json": json, "hashlib": hashlib, "random": random}
definitions = [
    ("inputs/scripts/course_experiments/text.py", ["arithmetic_records", "_steps"]),
    ("inputs/scripts/course_experiments/common.py", ["split_records"]),
    ("inputs/behavior-original.py", ["_preference_parts"]),
]
coverage = []
for filename, names in definitions:
    raw = (ART / filename).read_bytes()
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for n in nodes:
        coverage.append({"file": filename, "function": n.name, "lines": [n.lineno, n.end_lineno],
                         "sha256": hashlib.sha256(raw).hexdigest()})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), ns)

rows = ns["arithmetic_records"]()
parts = ns["split_records"](rows, seed=42)
pairs = ns["_preference_parts"](parts)
dpo = json.loads((ART / "inputs/dpo-results-original.json").read_text())
style = json.loads((ART / "inputs/style-results-original.json").read_text())
assert len(rows) == 8 * 8 == 64
assert len({r["family"] for r in rows}) == 8 * 9 // 2 == 36
assert {k: len(v) for k, v in parts.items()} == {"train": 49, "validation": 8, "test": 7}
groups = {k: {r["family"] for r in v} for k, v in parts.items()}
assert not groups["train"] & groups["validation"]
assert not groups["train"] & groups["test"]
assert not groups["validation"] & groups["test"]
assert set.union(*groups.values()) == {r["family"] for r in rows}
prompt_split = {r["prompt"]: split for split, records in pairs.items() for r in records}
assert prompt_split["2+3=?"] == prompt_split["3+2=?"]
all_pairs = [r for v in pairs.values() for r in v]
assert len({r["prompt"] for r in all_pairs}) == 64
for r in all_pairs:
    a, b = map(int, r["prompt"].removesuffix("=?").split("+"))
    assert r["chosen"] == str(a + b)
    assert r["rejected"] == str(a + b + 1)

pointers = ["/revision", "/seed", "/step_scale", "/device", "/torch_version", "/python_version",
            "/code_sha256", "/results/data", "/results/before/validation/records",
            "/results/before/test/records", "/results/before/validation/samples",
            "/results/before/test/samples"]
split_audit = {}
for split, records in pairs.items():
    pair_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    row_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parts[split]).encode()
    pair_sha = hashlib.sha256(pair_bytes).hexdigest()
    row_sha = hashlib.sha256(row_bytes).hexdigest()
    for historical, digest in [(dpo["results"]["data"][split], pair_sha),
                               (style["results"]["arithmetic_data"][split], row_sha)]:
        assert historical["records"] == len(records)
        assert historical["families"] == len(groups[split])
        assert historical["sha256"] == digest
    split_audit[split] = {"records": len(records), "families": len(groups[split]),
                          "pair_jsonl_sha256": pair_sha, "arithmetic_jsonl_sha256": row_sha,
                          "family_ids": sorted(groups[split])}
for label, entry in [("before", dpo["results"]["before"])] + [
        (name, run["preference"]) for name, run in dpo["results"]["runs"].items()]:
    for split in ["validation", "test"]:
        samples = entry[split]["samples"]
        assert entry[split]["records"] == len(samples) == len(pairs[split])
        assert [{k: s[k] for k in ["family", "prompt", "chosen", "rejected"]} for s in samples] == pairs[split]
        if label != "before":
            pointers.extend([f"/results/runs/{label}/preference/{split}/records",
                             f"/results/runs/{label}/preference/{split}/samples"])
settings = {}
for name, beta in [("model", 0.1), ("beta1", 1.0)]:
    run = dpo["results"]["runs"][name]
    assert run["beta"] == beta and run["training"]["steps"] == 250
    settings[name] = {"beta": beta, "saved_updates": run["training"]["steps"]}
    pointers.extend([f"/results/runs/{name}/beta", f"/results/runs/{name}/training/steps"])
assert dpo["seed"] == style["seed"] == 42 and dpo["step_scale"] == 1.0
assert ns["_steps"](type("Context", (), {"step_scale": 1.0})(), 250) == 250

bt = ast.parse((ART / "inputs/behavior-original.py").read_bytes())
run_dpo = next(n for n in bt.body if isinstance(n, ast.FunctionDef) and n.name == "run_dpo")
format_node = next(n for n in run_dpo.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "format_pairs" for t in n.targets))
ns["pairs"] = pairs
exec(compile(ast.Module(body=[format_node], type_ignores=[]), "original-format-pairs", "exec"), ns)
format_audit = {}
for split, records in ns["format_pairs"].items():
    assert all(r["rejected"] == r["chosen"] + "; answer complete" for r in records)
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    sha = hashlib.sha256(raw).hexdigest()
    saved = dpo["results"]["format_only"]["data"][split]
    assert saved["sha256"] == sha and saved["records"] == len(records)
    format_audit[split] = {"records": len(records), "format_pair_jsonl_sha256": sha}
    pointers.append(f"/results/format_only/data/{split}")
coverage.append({"file": "inputs/behavior-original.py", "assignment": "format_pairs",
                 "lines": [format_node.lineno, format_node.end_lineno]})

fence = (ART / "original-execution/fence-1.py").read_bytes()
ft = ast.parse(fence)
assert [ast.unparse(n.func) for n in ast.walk(ft) if isinstance(n, ast.Call)] == ["print", "print", "print"]
out = io.StringIO()
fns = {}
with contextlib.redirect_stdout(out):
    exec(compile(fence, "original-fence-1.py", "exec"), fns)
assert fns["chosen"] == "A"
fns["pair"]["prompt"] = "2+2=?"
assert fns["chosen"] == "A"  # Changing a dictionary value cannot reassign chosen.
variation = {"changed_prompt": fns["pair"]["prompt"], "chosen_before_manual_relabel": fns["chosen"],
             "interpretation": "Removing the constraint requires a fresh human preference; no automatic B selection follows."}
fns["rule"] = "兩者加法內容都正確，這題不指定說法，先記平手"
fns["chosen"] = "tie"
variation["manually_set_rule"] = fns["rule"]
variation["manually_set_label"] = fns["chosen"]
result = {"kind": "bounded_cpu_original_data_and_fence_audit", "split_audit": split_audit,
          "format_pairs_audit": format_audit,
          "original_fence_stdout": out.getvalue(), "variation": variation, "settings": settings,
          "read_json_pointers": {"dpo-results-original.json": pointers,
                                 "style-results-original.json": ["/revision", "/seed", "/code_sha256", "/results/arithmetic_data"]},
          "compiled_original_functions": coverage,
          "no_model_loading_training_or_generation": True,
          "checks": "All assertions passed; historical records and file byte fingerprints match deterministic reconstruction."}
(ART / "cpu-audit-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
env = {"python": sys.version, "python_executable": sys.executable, "device": "CPU",
       "platform": platform.platform(), "scope": "stdlib data audit; torch only used by original helper bootstrap"}
(ART / "cpu-audit-environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
