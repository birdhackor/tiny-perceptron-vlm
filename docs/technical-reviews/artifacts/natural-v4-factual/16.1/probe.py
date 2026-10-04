"""Independent bounded CPU checks for lesson 16.1; no GPU or training."""
import contextlib
import hashlib
import io
import json
import platform
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from tiny_perceptron.model import TinyLM, ModelConfig

OUT = Path(__file__).parent
torch.manual_seed(42)
source = (OUT / "source-original.md").read_text()
snippet = re.search(r"```python\n(.*?)```", source, re.S)[1]
namespace = {}
baseline_stdout = io.StringIO()
with contextlib.redirect_stdout(baseline_stdout):
    exec(compile(snippet, "course/chapters/16.md#16.1", "exec"), namespace)
model = namespace["model"]
assert torch.get_num_threads() == 1
assert tuple(namespace["result"]["logits"].shape) == (1, 3, 264)
params = [{"name": n, "shape": list(p.shape), "numel": p.numel(),
           "element_bytes": p.element_size(), "dtype": str(p.dtype)}
          for n, p in model.named_parameters()]
assert sum(x["numel"] for x in params) == 6104
assert sum(x["numel"] * x["element_bytes"] for x in params) == 24416
assert all(x["dtype"] == "torch.float32" for x in params)
weights_before = {n: p.detach().clone() for n, p in model.named_parameters()}

rounds = []
with torch.no_grad():
    for round_index in range(3):
        for length in (3, 12):
            ids = torch.arange(1, length + 1)[None]
            for _ in range(3):
                model(ids)
            with profile(activities=[ProfilerActivity.CPU]) as p:
                for _ in range(5):
                    result = model(ids)
            averages = p.key_averages()
            entries = {e.key: {"calls": e.count, "self_cpu_us": e.self_cpu_time_total,
                              "cpu_total_us": e.cpu_time_total} for e in averages}
            assert tuple(result["logits"].shape) == (1, length, 264)
            assert entries["aten::index_select"]["calls"] == 10
            assert entries["aten::gelu"]["calls"] == 5
            assert entries["aten::_softmax"]["calls"] == 5
            assert entries["aten::linear"]["calls"] == 35
            nested = [e for e in p.events() if e.cpu_children]
            assert nested
            residuals = [abs(e.self_cpu_time_total - (e.cpu_time_total -
                         sum(c.cpu_time_total for c in e.cpu_children))) for e in nested]
            assert max(residuals) <= 1e-9
            rounds.append({"round": round_index + 1, "input_ids": ids.tolist(),
                           "shape": list(result["logits"].shape), "warmup": 3,
                           "measured_forwards": 5, "events": entries,
                           "nested_event_count": len(nested),
                           "max_self_definition_residual_us": max(residuals),
                           "summed_cpu_total_us": sum(e.cpu_time_total for e in averages),
                           "summed_self_cpu_us": sum(e.self_cpu_time_total for e in averages)})
            print("ROUND", round_index + 1, "TOKENS", length)
            print(averages.table(sort_by="self_cpu_time_total", row_limit=20))
assert all(torch.equal(p, weights_before[n]) for n, p in model.named_parameters())

# Identify which module owns an operator without confusing its operator name
# with a layer name. Hooks add diagnostic annotations for this separate run.
handles, stack = [], []
def before(label):
    def hook(module, args):
        event = record_function("review:" + label)
        event.__enter__()
        stack.append(event)
    return hook
def after(module, args, output):
    stack.pop().__exit__(None, None, None)
for label, module in [("embedding", model.embedding), ("position", model.position),
                      ("attention", model.blocks[0].attention),
                      ("ffn", model.blocks[0].ffn), ("output", model.output)]:
    handles.extend([module.register_forward_pre_hook(before(label)),
                    module.register_forward_hook(after)])
with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as annotated:
    model(torch.tensor([[1, 2, 3]]))
for handle in handles:
    handle.remove()
origins = {}
for e in annotated.events():
    if e.key not in {"aten::index_select", "aten::gelu", "aten::_softmax",
                     "aten::mm", "aten::matmul", "aten::addmm"}:
        continue
    parent = e.cpu_parent
    while parent is not None and not parent.key.startswith("review:"):
        parent = parent.cpu_parent
    origins.setdefault(e.key, []).append(parent.key if parent else "unannotated")
assert set(origins["aten::index_select"]) == {"review:embedding", "review:position"}
assert set(origins["aten::gelu"]) == {"review:ffn"}
assert set(origins["aten::_softmax"]) == {"review:attention"}

base = torch.arange(6, dtype=torch.float32)
aliases = {"view": base.view(2, 3), "unsqueeze": base.unsqueeze(0),
           "as_strided": base.as_strided((2, 3), (3, 1))}
assert all(t.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
           for t in aliases.values())
zero = base.new_zeros((2, 3))
assert torch.equal(zero, torch.zeros(2, 3))
assert not torch.are_deterministic_algorithms_enabled()
empty_shapes = {"empty": list(torch.empty(2, 3).shape),
                "new_empty": list(base.new_empty((2, 3)).shape),
                "empty_like": list(torch.empty_like(zero).shape)}
# Do not read unspecified contents or assert that empty() returns nonzero data.

audit = []
for name in ("efficiency", "flash_probe", "precision"):
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    doc = json.loads(path.read_text())
    record = {"report": str(path.relative_to(ROOT)),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "runtime": {k: doc[k] for k in ("device", "seed", "torch_version", "python_version", "gpu")},
              "sample_groups": [], "training_groups": []}
    def walk(value, locator):
        if isinstance(value, dict):
            if "samples_seconds" in value:
                samples = value["samples_seconds"]
                assert len(samples) == value["measured_calls"] == 9
                assert value["warmup_calls"] == 3
                assert value["synchronized"] is True
                computed = sorted(samples)[4]
                assert abs(computed - value["median_seconds"]) <= 1e-12
                record["sample_groups"].append({"locator": locator, "samples": samples,
                     "median_recomputed_seconds": computed, "warmup": 3, "measured": 9})
            if "warm_step_median_seconds" in value:
                record["training_groups"].append({"locator": locator,
                     "steps": value.get("steps"), "effective_tokens": value.get("effective_tokens"),
                     "warm_step_median_seconds": value["warm_step_median_seconds"],
                     "limitation": "Raw per-update latencies absent; first-three exclusion checked in source, not numerically rerun."})
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    walk(v, locator + "." + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, (dict, list)):
                    walk(v, locator + "." + str(i))
    walk(doc["results"], "results")
    audit.append(record)

derivation = {"parameter_count": 264*8 + 128*8 + 3*(2*8) + 4*(8*8) + (8*32+32) + (32*8+8) + 8*264,
              "parameter_bytes": 6104*4, "median_even": (2+3)/2,
              "mean_even": (1+2+3+20)/4, "mib_bytes": 2**20}
assert derivation == {"parameter_count": 6104, "parameter_bytes": 24416,
                      "median_even": 2.5, "mean_even": 6.5, "mib_bytes": 1048576}
output = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
           "device": "cpu", "threads": str(torch.get_num_threads()), "dtype": "torch.float32",
           "cuda_available": str(torch.cuda.is_available()), "seed": "42", "model_mode": "eval/no_grad"},
          "exact_lesson_stdout": baseline_stdout.getvalue(), "parameters": params,
          "rounds": rounds, "annotated_operator_origins": origins,
          "allocation_and_views": {"zero_values": zero.tolist(), "empty_shapes": empty_shapes,
               "view_storage_shared": True, "deterministic_algorithms": False,
               "empty_contents": "intentionally uninspected"},
          "arithmetic": derivation, "original_gpu_record_audit": audit,
          "limitations": "CPU profile event timings are instrumented observations in one process; no GPU run, training, speed guarantee, or peak-memory replication."}
(OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
print("EXACT LESSON STDOUT\n" + baseline_stdout.getvalue())
print("ORIGINS", json.dumps(origins, sort_keys=True))
print("DERIVATION", json.dumps(derivation, sort_keys=True))
print("AUDIT_GROUP_COUNTS", [(a["report"], len(a["sample_groups"])) for a in audit])
print("ALL ASSERTIONS PASSED")
