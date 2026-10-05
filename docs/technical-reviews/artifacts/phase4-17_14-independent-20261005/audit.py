"""Fresh 17.14: raw measurements plus bounded CPU mechanisms, no historical scores rerun."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import fake_quantize, quantize_symmetric, QuantizedLinear
from tiny_perceptron.training import load_checkpoint, seed_everything
from scripts.course_experiments.compression import _fit_text, _qat_layers, _packed, _example

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
env = {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "cuda": str(torch.version.cuda), "threads": str(torch.get_num_threads())}
data_path = OUT / "sft-dataset.json"
result_path = ROOT / "docs/course-experiments/results/qat.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parts = json.loads(data_path.read_bytes())
raw = json.loads(result_path.read_bytes())
r = raw["results"]
assert sha(data_path) == r["data"]["sha256"]
tok = ByteTokenizer()
pointers = ["/revision", "/seed", "/device", "/torch_version", "/gpu", "/code_sha256/scripts~1course_experiments~1compression.py", "/code_sha256/tiny_perceptron~1quantization.py", "/results/teacher_provenance", "/results/data", "/results/initialization_sha256", "/results/activation_quantization", "/results/fake_quantization", "/results/same_initialization_and_batch_plan", "/results/fake_deployed_max_logit_difference"]
counts, families, targets = {}, {}, {}
for split, records in parts.items():
    counts[split] = len(records)
    families[split] = {record["family"] for record in records}
    targets[split] = [int((_example(row, 128)[1] != IGNORE).sum()) for row in records]
    assert counts[split] == r["data"]["counts"][split]
    assert len(families[split]) == r["data"]["families"][split]
    assert all(n == len(row["messages"][-1]["content"].encode()) + 1 for n, row in zip(targets[split], records, strict=True))
assert not (families["train"] & families["test"] or families["train"] & families["validation"] or families["test"] & families["validation"])
rng = random.Random(42)
plan = [[rng.randrange(len(parts["train"])) for _ in range(16)] for _ in range(350)]
plan_sha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
effective = sum(targets["train"][i] for batch in plan for i in batch)
assert effective == 39348
for name in ["fp_finetuning", "qat"]:
    v = r["training"][name]
    for field, wanted in {"steps":350,"optimizer_updates":350,"batch_size":16,"learning_rate":0.003,"effective_supervised_tokens":effective,"batch_plan_sha256":plan_sha,"initialization_sha256":r["initialization_sha256"],"weights_changed":True}.items():
        assert v[field] == wanted, (name, field)
        pointers.append(f"/results/training/{name}/{field}")
    assert v["final_sha256"] != v["initialization_sha256"]
    pointers.append(f"/results/training/{name}/final_sha256")
summary = {}
for name, run in r["runs"].items():
    summary[name] = {}
    for split in ["validation", "test"]:
        metric = run[split]
        samples = metric["generated_samples"]
        assert len(samples) == counts[split] == metric["examples"]
        hits, ends, invalid, matched = 0, 0, 0, 0
        for sample, record in zip(samples, parts[split], strict=True):
            ids = sample["generated_ids"]
            assert sample["question"] == record["messages"][-2]["content"]
            assert sample["expected"] == record["messages"][-1]["content"]
            eos = tok.eos_id in ids
            content = ids[:ids.index(tok.eos_id)] if eos else ids
            exact = content == tok.encode(sample["expected"])
            assert exact == sample["exact"] and eos == sample["ended_with_eos"]
            assert sample["generated"] == tok.decode(content)
            illegal = [i for i in ids if i < 8 and i != tok.eos_id]
            invalid += len(illegal)
            hits += exact
            ends += eos
            matched += exact and eos
        assert metric["correct"] == hits and metric["completed_correct"] == matched and metric["eos_count"] == ends
        assert metric["supervised_tokens"] == sum(targets[split])
        assert abs(metric["answer_nll"] - metric["nll_sum"] / sum(targets[split])) < 1e-12
        assert metric["exact_match"] == hits / counts[split] and metric["eos_rate"] == ends / counts[split]
        assert metric["max_new_tokens"] == 24
        summary[name][split] = {"correct":hits,"examples":counts[split],"targets":sum(targets[split]),"answer_nll":metric["answer_nll"],"rounded_nll":f'{metric["answer_nll"]:.4f}',"eos":ends,"invalid_control_ids":invalid}
        pointers.extend(f"/results/runs/{name}/{split}/{key}" for key in ["generated_samples","nll_sum","supervised_tokens","answer_nll","correct","examples","exact_match","completed_correct","eos_count","eos_rate","max_new_tokens"])
        if split == "test":
            assert ends == 10 and invalid == 0
    storage = run["storage"]
    assert storage["tensor_bytes"] == storage["parameter_tensor_bytes"] + sum(storage["buffers"].values())
    assert storage["file_overhead_bytes"] == storage["file_bytes"] - storage["tensor_bytes"]
    pointers.extend(f"/results/runs/{name}/storage/{key}" for key in ["parameter_count","float_parameter_count","parameter_tensor_bytes","buffer_tensor_bytes","tensor_bytes","buffers","file_bytes","file_overhead_bytes"])
pair = [r["runs"][name]["storage"] for name in ["matched_ptq4","qat_packed4"]]
assert pair[0]["tensor_bytes"] == pair[1]["tensor_bytes"] == 168736
assert r["fake_deployed_max_logit_difference"] == 0.0
prompt = "color=red;shape=square;pitch=high;describe"
example = {}
for name in ["matched_ptq4","qat_packed4"]:
    example[name] = next(sample for sample in r["runs"][name]["test"]["generated_samples"] if sample["question"] == prompt)
assert example["matched_ptq4"]["expected"] == example["qat_packed4"]["expected"] == "square"
assert example["matched_ptq4"]["generated"] == "square" and example["qat_packed4"]["generated"] == "circle"

x = torch.tensor([0.1,0.4,0.9], requires_grad=True)
y = fake_quantize(x,bits=4)
y.square().sum().backward()
assert torch.allclose(x.grad, 2*y.detach(), atol=1e-7, rtol=0)
integers, scale = quantize_symmetric(x.detach(), bits=4)
assert integers.tolist() == [1,3,7]
exercise = {"scale":scale.item(),"codes":integers.tolist(),"restored":y.detach().tolist(),"gradient":x.grad.tolist()}

with tempfile.TemporaryDirectory(prefix="factual17_14_micro_") as directory:
    ctx = SimpleNamespace(device="cpu", output=Path(directory), seed=42)
    seed_everything(42)
    base = TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=128,tied=False))
    original_state = {k:v.clone() for k,v in base.state_dict().items()}
    fp, fp_path, fp_fit = _fit_text(ctx,copy.deepcopy(base),parts["train"][:2],"micro_fp",steps=1)
    fake, _, qat_fit = _fit_text(ctx,_qat_layers(copy.deepcopy(base)),parts["train"][:2],"micro_qat",steps=1)
    assert fp_fit["initialization_sha256"] == qat_fit["initialization_sha256"]
    assert fp_fit["batch_plan_sha256"] == qat_fit["batch_plan_sha256"]
    assert fp_fit["optimizer_updates"] == qat_fit["optimizer_updates"] == 1
    assert fp_fit["weights_changed"] and qat_fit["weights_changed"]
    assert fp_fit["first_output_weight_gradient_norm"] > 0 and qat_fit["first_output_weight_gradient_norm"] > 0
    assert all(torch.equal(base.state_dict()[k],v) for k,v in original_state.items())
    float_master = _qat_layers(copy.deepcopy(fake),remove=True)
    deployment,path = _packed(ctx,float_master,"micro_packed",4)
    inp = _example(parts["test"][0],128)[0][None]
    with torch.no_grad():
        a = fake(inp)["logits"]
        b = deployment(inp)["logits"]
        c = float_master(inp)["logits"]
    gap = float((a-b).abs().max())
    assert gap <= 1e-6
    assert any(isinstance(layer,QuantizedLinear) for layer in deployment.modules())
    ordinary_reloaded,_ = load_checkpoint(Path(directory)/"micro_qat.pt","cpu")
    assert not any(type(layer).__name__ == "_QATLinear" for layer in ordinary_reloaded.modules())
    with torch.no_grad():
        assert torch.equal(ordinary_reloaded(inp)["logits"],c)
    cli = [str(ROOT/".venv/bin/python"),str(ROOT/"scripts/infer.py"),str(path),"--chat","--prompt",prompt,"--tokens","2","--json","--device","cpu"]
    completed = subprocess.run(cli,cwd=ROOT,capture_output=True,text=True,timeout=30,check=True)
    cli_result = json.loads(completed.stdout)
    micro = {"config":{"width":8,"layers":1,"heads":1,"max_length":128,"tied":False},"fp_updates":fp_fit["optimizer_updates"],"qat_updates":qat_fit["optimizer_updates"],"same_start":True,"same_plan":True,"fp_weight_changed":fp_fit["weights_changed"],"qat_weight_changed":qat_fit["weights_changed"],"fp_grad_norm":fp_fit["first_output_weight_gradient_norm"],"qat_grad_norm":qat_fit["first_output_weight_gradient_norm"],"logit_shape":list(a.shape),"logit_dtype":str(a.dtype),"fake_packed_max_logit_difference":gap,"fake_float_master_max_logit_difference":float((a-c).abs().max()),"plain_reload_is_float":True,"temporary_deployment_sha256":sha(path),"cli_command":" ".join(cli),"cli_result":cli_result,"temporary_weights_retained":False}
result = {"environment":env,"inputs":{"dataset_path":str(data_path.relative_to(ROOT)),"dataset_sha256":sha(data_path),"raw_qat_path":str(result_path.relative_to(ROOT)),"raw_qat_sha256":sha(result_path)},"raw_provenance":{"revision":raw["revision"],"device":raw["device"],"torch":raw["torch_version"],"python":raw["python_version"],"gpu":raw["gpu"],"seed":raw["seed"],"teacher_checkpoint_sha256":r["teacher_provenance"]["sha256"],"teacher_config":r["teacher_provenance"]["config"],"initialization_sha256":r["initialization_sha256"],"historical_fake_packed_max_logit_difference":r["fake_deployed_max_logit_difference"]},"raw_pointers_inspected":pointers,"counts":counts,"families":{k:len(v) for k,v in families.items()},"targets":{k:sum(v) for k,v in targets.items()},"updates":350,"draws":350*16,"batch_plan_sha256":plan_sha,"effective_training_targets":effective,"runs":summary,"storage_pair":{"matched_ptq4":{k:pair[0][k] for k in ["tensor_bytes","file_bytes","parameter_count"]},"qat_packed4":{k:pair[1][k] for k in ["tensor_bytes","file_bytes","parameter_count"]}},"prompt_example":example,"exercise":exercise,"micro":micro,"scope":"Existing raw scores were audited, not regenerated. One CPU optimizer step per width-8 branch plus packing/reload/2-token CLI checks only; no persistent neural weights."}
(OUT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
