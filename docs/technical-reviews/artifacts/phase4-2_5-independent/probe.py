"""No training: independent window arithmetic and existing checkpoint evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
namespace = {"torch": torch}
source = (OUT / "historical-code/tiny_perceptron/simple.py").read_bytes()
exec(compile(source, "historical-code/tiny_perceptron/simple.py", "exec"), namespace)
ContextMLP = namespace["ContextMLP"]
simple_source = (OUT / "historical-code/scripts/course_experiments/text.py").read_text()
tree = ast.parse(simple_source)
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_simple_examples")
helper_source = ast.get_source_segment(simple_source, helper)
(OUT / "original/simple_examples.py").write_text(helper_source + "\n")
exec(compile(helper_source, "historical-code/scripts/course_experiments/text.py:_simple_examples", "exec"), namespace)
make_examples = namespace["_simple_examples"]
result = {"environment": {"python": sys.version, "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda)},
    "training_performed": False, "original_fence_sha256": sha(OUT / "original/fence-1.py")}
# Exact chapter fence, no bootstrap and no parameter updates.
exec(compile((OUT / "original/fence-1.py").read_bytes(), "course/chapters/02.md:2.5:fence-1", "exec"), {})
examples = [("紅色物體是", "圓"), ("藍色物體是", "方")]
windows = {str(c): [prefix[-c:] for prefix, _ in examples] for c in (1, 3, 4, 5)}
assert windows == {"1": ["是", "是"], "3": ["物體是", "物體是"],
                   "4": ["色物體是", "色物體是"], "5": ["紅色物體是", "藍色物體是"]}
long = ["紅色的小小物體是", "藍色的小小物體是"]
assert [len(s) for s in long] == [8, 8]
assert long[0][-7:] == long[1][-7:] and long[0][-8:] != long[1][-8:]
result["windows"] = windows
result["longer_variant"] = {"lengths": [len(s) for s in long], "7": [s[-7:] for s in long], "8": [s[-8:] for s in long]}
chars = {char: i + 2 for i, char in enumerate(sorted(set("".join(p for p, _ in examples))))}
row_checks = {}
with torch.no_grad():
    for context in (1, 3, 4, 5):
        torch.manual_seed(42)
        model = ContextMLP(len(chars) + 2, context, 16)
        x = torch.tensor([[chars[c] for c in prefix[-context:]] for prefix, _ in examples])
        logits = model(x)
        diff = float((logits[0] - logits[1]).abs().max())
        assert (diff == 0.0) if context < 5 else (diff > 1e-6)
        row_checks[str(context)] = {"input_equal": bool(torch.equal(x[0], x[1])), "max_logit_difference": diff}
    model.hidden.weight.zero_()
    model.hidden.bias.zero_()
    assert torch.equal(model(x)[0], model(x)[1])
    result["visible_but_ignored"] = "Context 5 with hidden weight/bias zero gives identical logits; visibility alone is insufficient."
result["same_input_checks"] = row_checks
result["parameter_formula"] = {"embedding_VD": 17*16, "hidden_HCD": {str(c): 16*c*16 for c in (1, 2, 3, 5)},
    "hidden_bias_H": 16, "output_VH_plus_V": 17*16+17,
    "total_VD_plus_HCD_plus_H_plus_VH_plus_V": {str(c): 17*16+16*c*16+16+17*16+17 for c in (1, 2, 3, 5)}}
assert result["parameter_formula"]["hidden_HCD"]["2"] == 2 * result["parameter_formula"]["hidden_HCD"]["1"]
# Original published receipt and an existing local CPU run are separate evidence versions.
published = json.loads((OUT / "historical/simple_models.json").read_bytes())
local = json.loads((OUT / "historical/existing-local-cpu-result.json").read_bytes())
published_hashes = {entry["path"]: entry["sha256"] for entry in published["artifacts"]}
local_hashes = {entry["path"]: entry["sha256"] for entry in local["artifacts"]}
parts = {}
for split in ("train", "validation", "test"):
    path = OUT / f"historical/data/{split}.jsonl"
    assert sha(path) == published_hashes[f"data/{split}.jsonl"] == local_hashes[f"data/{split}.jsonl"]
    parts[split] = [json.loads(line)["text"] for line in path.read_text().splitlines()]
assert not (set(parts["train"]) & set(parts["validation"]))
assert not (set(parts["train"]) & set(parts["test"]))
assert not (set(parts["test"]) & set(parts["validation"]))
assert len(set(sum(parts.values(), []))) == 12
vocabulary = json.loads((OUT / "historical/vocabulary.json").read_bytes())
assert vocabulary == {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
for path in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py"):
    digest = sha(OUT / f"historical-code/{path}")
    assert digest == published["code_sha256"][path]
    assert sha(OUT / f"existing-local-code/{path}") == local["code_sha256"][path]
    if path.endswith("simple.py"):
        assert digest == sha(OUT / f"existing-local-code/{path}")
result["denominators"] = {split: {"documents": len(docs), "character_targets": sum(map(len, docs)),
    "boundary_targets": len(docs), "effective_next_character_targets": sum(len(s)+1 for s in docs),
    "document_lengths": [len(s) for s in docs]} for split, docs in parts.items()}
result["published_version"] = {key: published[key] for key in ("revision", "device", "seed", "torch_version", "python_version")}
result["existing_local_version"] = {key: local[key] for key in ("revision", "device", "seed", "torch_version", "python_version")}
result["evaluations"] = {}
display_expected = {"mlp1": ["0.33352", "0.43103"], "mlp3": ["0.21302", "0.30577"], "mlp5": ["0.19883", "0.51252"]}
with torch.no_grad():
    for name in ("mlp1", "mlp3", "mlp5"):
        run = published["results"]["runs"][name]
        local_run = local["results"]["runs"][name]
        assert run["steps"] == local_run["steps"] == 200
        checkpoint_path = ROOT / f"outputs/course-experiments/course-v1/simple_models/{name}.pt"
        assert sha(checkpoint_path) == local_hashes[f"{name}.pt"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["width"] == 16 and checkpoint["vocabulary"] == vocabulary
        model = ContextMLP(len(vocabulary)+2, run["context"], 16)
        model.load_state_dict(checkpoint["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == run["parameters"]
        reports = {}
        for split, docs in parts.items():
            x, y = make_examples(docs, vocabulary, run["context"])
            assert y.numel() == result["denominators"][split]["effective_next_character_targets"]
            logits = model(x)
            mean = float(F.cross_entropy(logits, y))
            total = float(F.cross_entropy(logits, y, reduction="sum"))
            explicit = float(-F.log_softmax(logits, -1).gather(1, y[:,None]).sum())
            assert abs(total-explicit) < 2e-5
            assert abs(mean-total/y.numel()) < 2e-7
            published_mean = run["after_nll_same_post_update_time"][split]
            local_mean = local_run["after_nll_same_post_update_time"][split]
            assert abs(mean-local_mean) < 5e-7 and abs(mean-published_mean) < 5e-7
            reports[split] = {"published_mean_nats_per_target": published_mean, "existing_local_mean": local_mean,
                "observed_eval_mean": mean, "eval_sum": total, "explicit_logsoftmax_sum": explicit,
                "denominator_targets": y.numel(), "shape_logits": list(logits.shape), "rounded5": f"{published_mean:.5f}"}
        assert [reports[s]["rounded5"] for s in ("train", "validation")] == display_expected[name]
        result["evaluations"][name] = {"parameters": run["parameters"], "steps": run["steps"], "context": run["context"],
            "checkpoint_original_path": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint_path), "metrics": reports}
        try:
            model(torch.zeros((1, run["context"]+1), dtype=torch.long))
        except ValueError as exc:
            assert str(exc) == "字卡數與模型的固定窗口不一致"
        else:
            raise AssertionError("ContextMLP accepted wrong window length")
training = [published["results"]["runs"][n]["after_nll_same_post_update_time"]["train"] for n in display_expected]
validation = [published["results"]["runs"][n]["after_nll_same_post_update_time"]["validation"] for n in display_expected]
assert training[0] > training[1] > training[2] and validation[0] > validation[1] < validation[2]
result["loss_mean_scope"] = "Mean across next-character targets including one end-boundary target per document; not an unweighted document mean. No GPU, optimizer, parameter update or fresh training."
result["tolerances"] = {"unicode_windows_counts_parameters_hashes_rounding": "exact", "mean_to_sum_per_target": "absolute 2e-7",
    "eval_vs_published_or_existing_local_mean": "absolute 5e-7", "explicit_logsoftmax_vs_cross_entropy_sum": "absolute 2e-5", "table_5dp": "rounding bound 5e-6"}
(OUT / "probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
