"""Bounded CPU checks and recomputation of existing raw measurements; no GPU execution."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

environment = {"python": sys.version, "torch": str(torch.__version__),
               "torch_git_version": torch.version.git_version,
               "device": "cpu", "cuda_build": str(torch.version.cuda)}
write("verification-environment.json", environment)
original = (OUT / "original-execution/fence-1.py").read_text()
exercise = original.replace("range(0, 3, 2)", "range(0, 3, 1)").replace("start + 2", "start + 1")
assert original.count("start + 2") == 2
(OUT / "exercise-block1.py").write_text(exercise)
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise, str(OUT / "exercise-block1.py"), "exec"), {})
assert exercise_stdout.getvalue().splitlines() == [
    "分母/分子 1.0 10.0", "分母/分子 1.5 25.0", "分母/分子 1.75 42.5",
    "分塊結果 24.2857", "完整結果 24.2857"]
(OUT / "exercise-stdout.txt").write_text(exercise_stdout.getvalue())

def online(scores, values, size, bad_numerator=False):
    m = scores.new_tensor(-math.inf)
    den = scores.new_zeros(())
    num = scores.new_zeros(())
    history = []
    for start in range(0, len(scores), size):
        block = scores[start:start + size]
        new_m = torch.maximum(m, block.max())
        factor = (m - new_m).exp()
        weight = (block - new_m).exp()
        den = den * factor + weight.sum()
        num = num * (1 if bad_numerator else factor) + (weight * values[start:start + size]).sum()
        m = new_m
        history.append({"m": m.item(), "factor": factor.item(), "denominator": den.item(), "numerator": num.item()})
    return (num / den).item(), history

scores = torch.tensor([0., math.log(2), math.log(4)], dtype=torch.float64)
values = torch.tensor([10., 20., 30.], dtype=torch.float64)
variants = []
for order in ([0, 1, 2], [2, 1, 0]):
    for size in (1, 2, 3):
        for offset in (0., 1000.):
            s = scores[order] + offset
            v = values[order]
            got, history = online(s, v, size)
            ref = (s.softmax(0) * v).sum().item()
            assert abs(got - ref) <= 1e-12 and abs(got - 170/7) <= 1e-11
            variants.append({"order": order, "block_size": size, "offset": offset,
                             "online": got, "full_softmax": ref, "absolute_difference": abs(got-ref), "history": history})
bad, _ = online(scores, values, 2, bad_numerator=True)
assert abs(bad - 170/7) > 1
counts = {str(t): t*t for t in (4, 8, 8192)}
assert counts == {"4":16, "8":64, "8192":67108864}
assert abs(math.log(2)-0.6931) < 0.00005 and abs(math.log(4)-1.3863) < 0.00005
numeric = {"grid_counts": counts, "log2": math.log(2), "log4": math.log(4),
           "exp_of_logs": [math.exp(math.log(x)) for x in (1, 2, 4)],
           "exact_weighted_mean": "170/7", "float_weighted_mean": 170/7,
           "variants": variants, "forgot_numerator_scaling_result": bad,
           "first_factor": variants[0]["history"][0]["factor"]}

pointers = {}
copy_hashes = []
for name in ("flash_probe.json", "efficiency.json"):
    source = ROOT / "docs/course-experiments/results" / name
    copied = OUT / "originals/docs/course-experiments/results" / name
    assert sha(source) == sha(copied)
    copy_hashes.append({"original": str(source.relative_to(ROOT)), "copy": str(copied.relative_to(ROOT)), "original_sha256":sha(source), "copy_sha256":sha(copied)})
    d = json.loads(copied.read_text())
    pointers[name] = {"top_level_key_types": {k:type(v).__name__ for k,v in d.items()}, "inspected":{}}
    for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu"):
        pointers[name]["inspected"]["/"+k] = d[k]
    for k in ("scripts/course_experiments/architecture.py",):
        pointers[name]["inspected"]["/code_sha256/"+k.replace("/","~1")] = d["code_sha256"][k]

flash = json.loads((OUT / "originals/docs/course-experiments/results/flash_probe.json").read_text())
assert sha(OUT / "originals/scripts/course_experiments/architecture.py") == flash["code_sha256"]["scripts/course_experiments/architecture.py"]
def inspect_flash(pointer):
    v=flash
    for k in pointer.strip("/").split("/"): v=v[k]
    pointers["flash_probe.json"]["inspected"][pointer] = v
    return v

cfg = inspect_flash("/results/configuration")
assert cfg["shape_B_H_T_D"] == [2,4,512,32] and cfg["warmup_calls"] == 3 and cfg["measured_calls"] == 9
assert cfg["attn_mask"] is None and cfg["is_causal"] and cfg["dropout_p"] == 0
assert not cfg["gqa"] and cfg["requested_backend"] == "SDPBackend.FLASH_ATTENTION only"
for k in ("torch_git_version", "cuda_runtime_version", "gpu", "compute_capability", "flash_attention_built", "driver_query"):
    inspect_flash("/results/runtime/"+k)
rows=[]
expected_times={"fp16": {"forward": [0.513,0.076], "forward_backward":[1.689,0.503]},
                "bf16": {"forward": [0.516,0.071], "forward_backward":[1.502,0.458]}}
for dtype in ("fp16","bf16"):
    correct = inspect_flash(f"/results/routes/{dtype}/correctness")
    assert all(x["finite"] and x["within_declared_tolerance"] for x in [correct["output"],*correct["gradients"].values()])
    assert all(0 <= x["max_absolute_error"] <= x["atol"] for x in [correct["output"],*correct["gradients"].values()])
    for mode in ("forward", "forward_backward"):
        profile = inspect_flash(f"/results/routes/{dtype}/profiles/{mode}")
        assert "aten::_scaled_dot_product_flash_attention" in profile["operator_names"]
        assert profile["cuda_kernel_names"] and any("flash_fwd_kernel" in k for k in profile["cuda_kernel_names"])
        if mode == "forward_backward":
            assert "aten::_scaled_dot_product_flash_attention_backward" in profile["operator_names"]
            assert any("flash_bwd_" in k for k in profile["cuda_kernel_names"])
        for i, backend in enumerate(("manual", "forced_flash")):
            m = inspect_flash(f"/results/routes/{dtype}/measurements/{mode}/{backend}")
            assert m["warmup_calls"]==3 and m["measured_calls"]==len(m["samples_seconds"])==9
            assert m["synchronized"] and m["includes_backward"] == (mode=="forward_backward")
            med = statistics.median(m["samples_seconds"])
            assert med == m["median_seconds"] and round(med*1000,3)==expected_times[dtype][mode][i]
            base, peak, additional = [m[k] for k in ("allocated_before_bytes","peak_allocated_bytes","additional_peak_allocated_bytes")]
            assert base==65*2**20 and peak-base==additional
            expected_additional = {("forward","manual"):24.25,("forward","forced_flash"):0.267,
                                   ("forward_backward","manual"):32.75,("forward_backward","forced_flash"):2.032}[mode,backend]
            assert abs(additional/2**20-expected_additional) <= 0.0005
            rows.append({"dtype":dtype,"mode":mode,"backend":backend,"samples":len(m["samples_seconds"]),
                         "median_ms":med*1000,"baseline_MiB":base/2**20,"peak_MiB":peak/2**20,"additional_MiB":additional/2**20})
eff=json.loads((OUT/"originals/docs/course-experiments/results/efficiency.json").read_text())
assert sha(OUT / "originals/efficiency-revision-architecture.py") == eff["code_sha256"]["scripts/course_experiments/architecture.py"]
e=eff["results"]["manual_vs_sdpa"]
for k in ("shape","dtype","backend","timings"):
    pointers["efficiency.json"]["inspected"]["/results/manual_vs_sdpa/"+k]=e[k]
assert e["dtype"]=="torch.float32" and e["backend"]["selected"]=="memory_efficient"
assert "aten::_scaled_dot_product_efficient_attention" in e["backend"]["operator_names"]
assert not e["backend"]["flash_attention_observed"]
meds={k:statistics.median(v["samples_seconds"]) for k,v in e["timings"].items()}
assert all(meds[k]==v["median_seconds"] for k,v in e["timings"].items())
efficiency_difference_ms=(meds["manual"]-meds["sdpa"])*1000
assert round(efficiency_difference_ms,3)==0.644
write("inspected-pointers.json", pointers)
write("original-copy-hashes.json", copy_hashes)
write("verification-results.json", {"numeric":numeric,"flash_table_recomputed":rows,
                                      "efficiency_difference_ms":efficiency_difference_ms,
                                      "gpu_rerun":False,"training_rerun":False})
print(exercise_stdout.getvalue(),end="")
print("CPU variants",len(variants),"passed; T^2",counts,"bad scaling",bad)
for row in rows: print(json.dumps(row))
print("efficiency difference ms",efficiency_difference_ms)
print("All bounded checks passed; existing measurements inspected, no GPU or training executed.")
