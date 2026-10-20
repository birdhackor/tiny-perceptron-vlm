"""核對參數、前向形狀、求導狀態與固定資料的計分分母，不重跑訓練。"""
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
import tiny_perceptron.attention as attention
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8))
    model.eval()
    tensors = [{"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype), "element_bytes": p.element_size()} for n, p in model.named_parameters()]
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "platform": platform.platform(), "device": "cpu", "threads": str(torch.get_num_threads())}, "config": model.description(), "parameters": tensors}
    assert model.description()["parameters"] == 6104
    assert sum(p.numel()*p.element_size() for p in model.parameters()) == 24416
    result["totals"] = {"parameters": model.description()["parameters"], "bytes": sum(p.numel()*p.element_size() for p in model.parameters()), "block_parameters": sum(p.numel() for p in model.blocks[0].parameters()), "untied_weights": model.embedding.weight.data_ptr() != model.output.weight.data_ptr()}
    assert result["totals"]["block_parameters"] == 840
    shapes = []
    original = attention.manual_attention

    def observed(q, k, v, allowed):
        out, weights = original(q, k, v, allowed)
        shapes.append(list(weights.shape))
        return out, weights

    attention.manual_attention = observed
    probes = []
    for ids in ([[1,2,3]], [[1,2,3,1,2,3]]):
        x = torch.tensor(ids)
        with torch.no_grad():
            output = model(x)
        probes.append({"input": ids, "logits_shape": list(output["logits"].shape), "attention_shape": shapes[-1], "parameter_count": model.description()["parameters"], "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters())})
    attention.manual_attention = original
    result["sequence_probes"] = probes
    assert [p["attention_shape"] for p in probes] == [[1,1,3,3],[1,1,6,6]]
    assert all(p["parameter_count"] == 6104 and p["parameter_bytes"] == 24416 for p in probes)
    x = torch.tensor([[1,2,3]])
    with torch.enable_grad():
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t.numel()) or t), lambda t: t):
            logits = model(x)["logits"]
    with torch.no_grad():
        no_grad_saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (no_grad_saved.append(t.numel()) or t), lambda t: t):
            no_grad_logits = model(x)["logits"]
    result["grad_modes"] = {"module_training_after_eval": model.training, "eval_with_grad_requires_grad": logits.requires_grad, "no_grad_requires_grad": no_grad_logits.requires_grad, "grad_saved_tensor_count": len(saved), "no_grad_saved_tensor_count": len(no_grad_saved)}
    assert logits.requires_grad and not no_grad_logits.requires_grad and saved and not no_grad_saved
    optimizer = torch.optim.AdamW(model.parameters())
    logits.square().mean().backward()
    optimizer.step()
    result["training_storage_probe"] = {"parameter_bytes": result["totals"]["bytes"], "gradient_bytes": sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None), "first_moment_bytes": sum(s["exp_avg"].numel()*s["exp_avg"].element_size() for s in optimizer.state.values()), "second_moment_bytes": sum(s["exp_avg_sq"].numel()*s["exp_avg_sq"].element_size() for s in optimizer.state.values())}
    result["clock"] = vars(time.get_clock_info("perf_counter"))
    timings = {}
    for length in (3,6):
        current = torch.tensor([[1,2,3]*(length//3)])
        with torch.no_grad():
            model(current)
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(100):
                    model(current)
                samples.append((time.perf_counter()-start)/100)
        timings[str(length)] = {"warmup_calls": 1, "repeats": 5, "calls_per_repeat": 100, "average_seconds_per_forward": samples}
    result["timing_samples"] = timings
    large = TinyLM(ModelConfig(width=64, layers=2))
    result["large_model"] = {"parameters": large.description()["parameters"], "bytes": sum(p.numel()*p.element_size() for p in large.parameters()), "block_parameters": sum(p.numel() for p in large.blocks[0].parameters())}
    assert result["large_model"]["parameters"] == 141568
    assert result["large_model"]["bytes"] == 566272
    record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
    record = json.loads(record_path.read_bytes())
    parts = split_records(generate_records("toy-text"), seed=42)
    examples = text_examples(parts["train"], mode="text", max_length=128)
    sampler = random.Random(42)
    effective = 0
    for _ in range(600):
        batch = sampler.choices(examples,k=16)
        _, y, _ = pad_batch(batch)
        effective += int((y != IGNORE).sum())
    config = {"seed": 42, "steps": 600, "batch_size": 16, "draws_with_replacement": 600*16, "train_records": len(parts["train"]), "examples": len(examples), "per_example_targets": [int((y != IGNORE).sum()) for _,y in examples], "recomputed_effective_tokens": effective, "recorded_effective_tokens": record["results"]["training"]["effective_tokens"], "record_sha256": sha(record_path), "records_sha256": records_sha256(parts["train"]), "recorded_records_sha256": record["results"]["training"]["records_sha256"]}
    assert effective == record["results"]["training"]["effective_tokens"] == 342462
    assert config["records_sha256"] == config["recorded_records_sha256"]
    result["denominator_replay"] = config
    result["record_excerpt"] = {key:record[key] for key in ("revision","device","seed","torch_version","python_version","gpu","elapsed_seconds","timing_scope","step_scale","evidence_status")}
    result["record_excerpt"]["training"] = {key:record["results"]["training"][key] for key in ("steps","seconds","effective_tokens","parameters","records")}
    result["record_excerpt"]["scaling_runs"] = list(record["results"]["scaling"])
    inputs = ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py", "scripts/prepare_data.py", "course/training.md"]
    result["input_sha256"] = {p:sha(ROOT/p) for p in inputs}
    result["recorded_code_matches"] = {p:{"current":sha(ROOT/p),"recorded":record["code_sha256"].get(p),"matches":sha(ROOT/p)==record["code_sha256"].get(p)} for p in inputs if p in record["code_sha256"]}
    result["tokenizer"] = {"vocab_size":ByteTokenizer.vocab_size,"special_count":len(SPECIALS),"specials":list(SPECIALS),"byte_values":256}
    result["assertions"] = "全部通過；GPU 的歷史耗時僅核對紀錄與計時原始碼，沒有重新量測。"
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
