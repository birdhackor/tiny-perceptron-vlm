"""核對快取形狀、投影輸入、量測紀錄與固定步數生成的範圍。"""

import ast
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.architecture import _clone_config, _fixed_decode
from scripts.course_experiments.common import evaluate_lm
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, generate


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def shape_probe(length):
    model = TinyLM(ModelConfig(width=8)).eval()
    projections = []
    positions = []
    handles = []
    attention = model.blocks[0].attention
    for name in ("q", "k", "v"):
        def observe(module, args, output, name=name):
            projections.append({"name": name, "input_shape": list(args[0].shape)})
        handles.append(getattr(attention, name).register_forward_hook(observe))
    handles.append(model.position.register_forward_pre_hook(lambda module, args: positions.append(args[0].tolist())))
    prefix = torch.tensor([list(range(1, length + 1))])
    with torch.no_grad():
        prefill = model(prefix)
        decode = model(torch.tensor([[5]]), cache=prefill["cache"])
    for handle in handles:
        handle.remove()
    result = {
        "prefix_length": length,
        "logits": [list(prefill["logits"].shape), list(decode["logits"].shape)],
        "k_shapes": [list(prefill["cache"][0][0].shape), list(decode["cache"][0][0].shape)],
        "v_shapes": [list(prefill["cache"][0][1].shape), list(decode["cache"][0][1].shape)],
        "positions": positions,
        "projections": projections,
        "cache_prefix_equal": all(
            torch.equal(old, new[:, :, :length])
            for old, new in zip(prefill["cache"][0], decode["cache"][0], strict=True)
        ),
        "dict_keys": sorted(prefill),
        "cache_layers": len(prefill["cache"]),
    }
    assert result["logits"] == [[1, length, 264], [1, 1, 264]]
    assert result["k_shapes"] == [[1, 1, length, 8], [1, 1, length + 1, 8]]
    assert result["v_shapes"] == result["k_shapes"]
    assert positions == [list(range(length)), [length]]
    assert result["cache_prefix_equal"]
    assert [x["input_shape"] for x in projections] == [[1, length, 8]] * 3 + [[1, 1, 8]] * 3
    eval_logits = model(prefix)["logits"]
    with torch.no_grad():
        no_grad_logits = model(prefix)["logits"]
    result["grad_mode"] = {
        "eval_training": model.training,
        "eval_logits_require_grad": eval_logits.requires_grad,
        "no_grad_logits_require_grad": no_grad_logits.requires_grad,
        "no_grad_grad_fn": str(no_grad_logits.grad_fn),
    }
    assert not model.training and eval_logits.requires_grad and not no_grad_logits.requires_grad
    return result


class EOSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.config = ModelConfig()
        self.calls = []

    def forward(self, ids, cache=None, **kwargs):
        self.calls.append(ids.shape[1])
        logits = torch.zeros((*ids.shape, self.config.vocab_size)) + self.dummy
        logits[:, :, 2] = 10
        length = ids.shape[1] + (0 if cache is None else cache[0][0].shape[2])
        state = torch.zeros((1, 1, length, 32))
        return {"logits": logits, "cache": [(state, state)]}


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-2/verify_claims.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "shape_probes": [shape_probe(4), shape_probe(6)],
    }
    tok = ByteTokenizer()
    prompt = "Once upon a time, a girl"
    prefix = [tok.bos_id] + tok.encode(prompt)
    output["tokenizer"] = {
        "vocab_size": tok.vocab_size,
        "prompt": prompt,
        "prompt_ids_with_bos": prefix,
        "prompt_positions": len(prefix),
        "character_example": {"text": "中", "ids": tok.encode("中"), "roundtrip": tok.decode(tok.encode("中"))},
    }
    assert len(prefix) == 25 and len(tok.encode("中")) == 3
    record_path = "docs/course-experiments/results/efficiency.json"
    report = json.loads((ROOT / record_path).read_bytes())
    mha = report["results"]["models"]["mha"]
    cache = mha["cache"]
    output["stored_experiment"] = {
        "path": record_path,
        "sha256": sha(record_path),
        "revision": report["revision"],
        "device": report["device"],
        "gpu": report["gpu"],
        "torch_version": report["torch_version"],
        "python_version": report["python_version"],
        "seed": report["seed"],
        "config": mha["model"]["config"],
        "training": {k: mha["training"][k] for k in ("steps", "optimizer_updates", "requested_steps", "effective_tokens", "batch_size", "skipped_updates", "status")},
        "dataset": report["results"]["dataset"],
        "source": report["results"]["source"],
        "generated_ids_full": cache["generated_ids_full"],
        "generated_text": cache["generated_text"],
        "timings": {},
        "current_code_matches_record": {},
        "scope": "只重算保存紀錄與核對量測程式；未重跑 L4 計時或續訓。",
    }
    assert cache["prompt_tokens"] == len(prefix) and cache["generated_tokens"] == 12
    assert len(cache["generated_ids_full"]) == 12 and tok.eos_id in cache["generated_ids_full"][:-1]
    assert tok.decode(cache["generated_ids_full"]) == cache["generated_text"] == "eueeuare"
    for key, expected in (("prefill", 2.421), ("full_recompute_decode", 29.034), ("cached_decode", 27.915)):
        timing = cache[key]
        median = statistics.median(timing["samples_seconds"])
        assert median == timing["median_seconds"]
        assert round(median * 1000, 3) == expected
        assert timing["warmup_calls"] == 3 and timing["measured_calls"] == len(timing["samples_seconds"]) == 9
        output["stored_experiment"]["timings"][key] = {**timing, "recomputed_median_ms": median * 1000, "rounded_ms": round(median * 1000, 3)}
    for path in (
        "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py",
        "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
    ):
        actual = sha(path)
        expected = report["code_sha256"][path]
        check = {"current": actual, "recorded": expected, "equal": actual == expected}
        if actual != expected:
            url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{report['revision']}/{path}"
            command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
            original = subprocess.run(command, capture_output=True, check=True).stdout
            assert hashlib.sha256(original).hexdigest() == expected
            function_checks = {}
            for name in ("run_efficiency", "_cache_probe", "_fixed_decode", "_benchmark", "_sync", "_clone_config", "_copy_matching", "_sft_dataset", "_heldout", "_train", "_steps"):
                bodies = []
                for content in (original.decode(), (ROOT / path).read_text()):
                    node = next(n for n in ast.parse(content).body if isinstance(n, ast.FunctionDef) and n.name == name)
                    bodies.append(ast.get_source_segment(content, node))
                assert bodies[0] == bodies[1], name
                function_checks[name] = {"equal": True, "source_sha256": hashlib.sha256(bodies[0].encode()).hexdigest()}
            check["historical_inspection"] = {"url": url, "command": command, "exit_code": 0, "original_sha256": hashlib.sha256(original).hexdigest(), "original_bytes": len(original), "accessed_on": "2026-10-20", "functions": function_checks}
        output["stored_experiment"]["current_code_matches_record"][path] = check
    sft_path = "docs/course-experiments/results/sft.json"
    sft = json.loads((ROOT / sft_path).read_bytes())
    output["sft_record"] = {"path": sft_path, "sha256": sha(sft_path), "checkpoint": sft["results"]["checkpoint"], "scope": sft["results"]["scope"]}
    assert mha["model"]["config"]["heads"] == mha["model"]["config"]["kv_heads"] == 4
    assert mha["training"]["steps"] == mha["training"]["optimizer_updates"] == 100
    initial = TinyLM(ModelConfig(width=64, layers=2))
    target, shared = _clone_config(initial, SimpleNamespace(device="cpu", seed=42), heads=4, kv_heads=4, experts=0, backend="manual")
    output["mha_clone"] = {"initial_heads": initial.config.heads, "target_heads": target.config.heads, "copied_tables": len(shared), "copied_names_match_record": shared == mha["copied_initial_tables"]}
    assert len(shared) == 29 and shared == mha["copied_initial_tables"]
    eos_model = EOSModel().eval()
    ids = torch.tensor([prefix])
    eos_results = {}
    for cached in (False, True):
        eos_model.calls.clear()
        seq = _fixed_decode(eos_model, ids, 12, cached)
        eos_results[str(cached)] = {"new_ids": seq[0, len(prefix):].tolist(), "forward_input_lengths": list(eos_model.calls)}
        assert seq.shape[1] - len(prefix) == 12
    seq = generate(eos_model, ids, max_new_tokens=12, eos_id=tok.eos_id)
    eos_results["normal_generate_new_ids"] = seq[0, len(prefix):].tolist()
    assert eos_results["normal_generate_new_ids"] == [tok.eos_id]
    rows = [{"messages": [{"role": "user", "content": "color=red;color?"}, {"role": "assistant", "content": "red"}]}]
    evaluation = evaluate_lm(eos_model, rows, mode="sft", tokens=12)
    eos_results["evaluation_generated_ids"] = evaluation["samples"][0]["generated_ids"]
    assert eos_results["evaluation_generated_ids"] == [tok.eos_id]
    output["forced_eos_probe"] = eos_results
    output["result"] = "所有斷言通過；形狀、梯度模式、計時中位數、模型改頭與 EOS 停止範圍符合教材敘述。"
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    Path(__file__).with_name("verification-output.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
