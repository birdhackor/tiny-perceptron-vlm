"""獨立核對玩具算式、模型參數、固定資料切分與已保存的實驗結果。"""

import ast
import hashlib
import io
import inspect
import json
import platform
import random
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _matched_width, _parameter_budget
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    torch.set_num_threads(1)
    raw = (ROOT / "docs/course-experiments/results/moe.json").read_bytes()
    record = json.loads(raw)
    results = record["results"]
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-13/audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "experiment_record_sha256": digest(raw),
        "experiment_environment": {key: record[key] for key in ("revision", "python_version", "torch_version", "device", "gpu", "seed")},
    }
    output["recorded_code_hashes"] = {
        name: {"expected": value, "actual": digest((ROOT / name).read_bytes())}
        for name, value in record["code_sha256"].items()
    }
    used_unchanged = ("tiny_perceptron/attention.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
                      "tiny_perceptron/modern.py", "tiny_perceptron/training.py", "scripts/course_experiments/common.py")
    assert all(output["recorded_code_hashes"][name]["expected"] == output["recorded_code_hashes"][name]["actual"] for name in used_unchanged)
    code_url = f'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{record["revision"]}/scripts/course_experiments/architecture.py'
    code_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", code_url]
    historic_code = subprocess.run(code_command, check=True, capture_output=True).stdout
    assert digest(historic_code) == record["code_sha256"]["scripts/course_experiments/architecture.py"]
    definitions = lambda raw: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    old, current = definitions(historic_code), definitions((ROOT / "scripts/course_experiments/architecture.py").read_bytes())
    used_functions = ("_description", "_copy_matching", "_clone_config", "_text_dataset", "_data_report", "_steps",
                      "_forward", "_nll", "_train", "_heldout", "_parameter_budget", "_matched_width", "_router_gradients", "_routing", "run_moe")
    output["historical_architecture"] = {"url": code_url, "command": code_command,
                                         "sha256": digest(historic_code), "bytes": len(historic_code),
                                         "used_function_ast_equal": {name: old[name] == current[name] for name in used_functions},
                                         "read_locator": list(used_functions), "full_source_saved": False}
    assert all(output["historical_architecture"]["used_function_ast_equal"].values()), output["historical_architecture"]
    output["toy"] = []
    for experts in (4, 8):
        for k in (1, 2):
            one, router = 4 * 2 + 2 * 4, 4 * experts
            total, active = experts * one + router, k * one + router
            output["toy"].append({"experts": experts, "k": k, "per_expert": one, "router": router,
                                  "total": total, "active": active,
                                  "dense_total_hidden": total / 8, "dense_active_hidden": active / 8})
    expected = {
        "dense_active_top1": (141568, 141568, "2.29999", "2.32093"),
        "dense_active_top2": (207680, 207680, "2.28200", "2.30906"),
        "dense_total": (329888, 329888, "2.22603", "2.24943"),
        "top1_aux0": (340608, 142080, "2.30115", "2.32369"),
        "top1_aux0.01": (340608, 142080, "2.29775", "2.31963"),
        "top2_aux0": (340608, 208256, "2.28011", "2.30320"),
        "top2_aux0.01": (340608, 208256, "2.27819", "2.30170"),
    }
    output["variants"] = {}
    for name, variant in results["variants"].items():
        config = ModelConfig(**variant["model"]["config"])
        model = TinyLM(config)
        table = dict(model.named_parameters())
        total = sum(p.numel() for p in table.values())
        experts = sum(p.numel() for n, p in table.items() if ".ffn.experts." in n)
        router = sum(p.numel() for n, p in table.items() if ".ffn.router." in n)
        active = total - experts + (experts * config.top_k // config.experts if config.experts else 0)
        budget = {"total_parameters": total, "active_parameter_proxy_per_token": active,
                  "expert_parameters": experts, "router_parameters": router}
        assert budget == variant["budget"] == _parameter_budget(model)
        train = variant["training"]
        assert (train["steps"], train["optimizer_updates"], train["skipped_updates"], train["effective_tokens"]) == (180, 180, 0, 337761)
        heldout = {}
        for split, count in (("validation", 39256), ("test", 41914)):
            value = variant["heldout"][split]
            assert value["effective_tokens"] == count
            assert abs(value["nll"] - value["nll_sum"] / count) < 1e-12
            assert len(value["samples"]) == 8
            for sample in value["samples"]:
                assert ByteTokenizer().decode(sample["generated_ids"]) == sample["generated"]
            heldout[split] = {key: value[key] for key in ("nll", "nll_sum", "effective_tokens", "records", "examples")}
            heldout[split]["rounded_nll"] = f'{value["nll"]:.5f}'
            heldout[split]["samples_saved"] = len(value["samples"])
        assert (total, active, heldout["validation"]["rounded_nll"], heldout["test"]["rounded_nll"]) == expected[name]
        routing = variant["validation_routing"]
        if routing:
            assert routing["padding_excluded"]
            for layer in routing["layers"]:
                denominator = routing["effective_input_tokens"] * config.top_k
                assert sum(layer["dispatch_counts"]) == layer["dispatch_denominator"] == denominator
                assert all(abs(n / denominator - f) < 1e-12 for n, f in zip(layer["dispatch_counts"], layer["load_fraction"]))
        output["variants"][name] = {"config": variant["model"]["config"], "independent_budget": budget,
                                    "training": {key: train[key] for key in ("steps", "optimizer_updates", "skipped_updates", "effective_tokens", "auxiliary_weight")},
                                    "heldout": heldout, "first_test_sample": variant["heldout"]["test"]["samples"][0],
                                    "embedding_shape": list(model.embedding.weight.shape),
                                    "attention_q_shape": list(model.blocks[0].attention.q.weight.shape),
                                    "routing_denominators_checked": routing is not None}
    config = ModelConfig(width=64, layers=2, heads=4, max_length=128)
    candidates = [(w, sum(p.numel() for p in TinyLM(ModelConfig(width=w, layers=2, heads=4, max_length=128)).parameters())) for w in range(16, 161, 4)]
    output["matching"] = []
    for target in (142080, 208256, 340608):
        width, count = min(candidates, key=lambda item: (abs(item[1] - target), item[0]))
        assert _matched_width(config, target) == (width, count - target)
        output["matching"].append({"target": target, "width": width, "count": count,
                                   "difference": count - target, "relative_shortfall_percent": (target - count) / target * 100})
    output["candidate_parameter_counts"] = candidates
    torch.manual_seed(42)
    model_a = TinyLM(config)
    torch.manual_seed(42)
    model_b = TinyLM(config)
    output["same_shape_seeded_initialization_equal"] = all(torch.equal(v, model_b.state_dict()[k]) for k, v in model_a.state_dict().items())
    assert output["same_shape_seeded_initialization_equal"]
    logits = torch.tensor([[[0.2, -0.3, 1.0], [1.0, 0.0, 0.5], [-1.0, 0.0, 1.0]]], dtype=torch.float64)
    labels = torch.tensor([[2, IGNORE, 1]])
    loss, count = loss_sum(logits, labels)
    manual = -(logits.log_softmax(-1)[0, 0, 2] + logits.log_softmax(-1)[0, 2, 1])
    assert int(count) == 2 and abs(float(loss - manual)) < 1e-12
    output["nll_semantics_probe"] = {"sum": float(loss), "direct_negative_log_probability_sum": float(manual), "valid_targets": int(count)}
    installed_docs = torch.nn.CrossEntropyLoss.__doc__
    assert "LogSoftmax" in installed_docs and "NLLLoss" in installed_docs and "ignore_index" in installed_docs
    output["installed_pytorch_source"] = {}
    for function in (torch.nn.CrossEntropyLoss, torch.nn.Module.parameters, torch.manual_seed):
        source_file = Path(inspect.getsourcefile(function))
        output["installed_pytorch_source"][function.__qualname__] = {
            "version": str(torch.__version__), "file": str(source_file),
            "file_sha256": digest(source_file.read_bytes()),
            "symbol_source_sha256": digest(inspect.getsource(function).encode()),
        }
    from scripts.course_experiments.run import experiment_spec
    spec = experiment_spec("moe")
    assert spec["module"] == "architecture" and spec["function"] == "run_moe"
    help_command = [sys.executable, "-m", "scripts.course_experiments.run", "--help"]
    help_result = subprocess.run(help_command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "--experiment" in help_result.stdout and "--device" in help_result.stdout
    output["reproduction_entrypoint"] = {"spec": spec, "help_command": help_command,
                                          "help_exit_code": help_result.returncode,
                                          "help_stdout": help_result.stdout,
                                          "scope": "僅核對 CLI 介面與配方對應；未啟動 CUDA 訓練。"}
    compression_raw = (ROOT / "scripts/course_experiments/compression.py").read_bytes()
    compression_nodes = {node.name: node for node in ast.parse(compression_raw).body if isinstance(node, ast.FunctionDef)}
    calls = [node for node in ast.walk(compression_nodes["run_distillation"]) if isinstance(node, ast.Call)]
    moe_calls = [node for node in calls if any(isinstance(arg, ast.Constant) and arg.value == "moe" for arg in node.args)]
    assert len(moe_calls) == 1
    assert isinstance(moe_calls[0].func, ast.Name) and moe_calls[0].func.id == "_distill_case"
    output["downstream_teacher"] = {"source": "scripts/course_experiments/compression.py", "sha256": digest(compression_raw),
                                    "caller": "run_distillation", "callee": "_distill_case", "identifier": "moe",
                                    "source_call": ast.unparse(moe_calls[0]),
                                    "scope": "讀取 _teacher、_distill_case 與 run_distillation 的依賴路徑；沒有重跑蒸餾。"}
    asset = record["assets"][0]
    url = f'https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{record["revision"]}/{asset["archive"]}'
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    archive = subprocess.run(command, check=True, capture_output=True).stdout
    assert digest(archive) == asset["archive_sha256"] and len(archive) == asset["archive_bytes"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        member = next(m for m in handle.getmembers() if m.name.endswith("tinystories-train-512.jsonl"))
        rows_raw = handle.extractfile(member).read()
        rows = [json.loads(line) for line in rows_raw.decode().splitlines()]
        prefix_member = next(m for m in handle.getmembers() if m.name.endswith("tinystories-train-prefix-complete.txt"))
        prefix_raw = handle.extractfile(prefix_member).read()
    assert len(rows) == 512
    complete_stories = [text.strip() for text in prefix_raw.decode().split("<|endoftext|>")[:-1]]
    assert len(complete_stories) == 512 and complete_stories == [r["text"] for r in rows]
    provenance = json.loads((ROOT / "assets/training/sources/tinystories.json").read_bytes())
    upstream_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45",
                        "--max-filesize", "1048576", "--range", "0-458751", provenance["source_url"]]
    upstream = subprocess.run(upstream_command, check=True, capture_output=True).stdout
    upstream_stories = [part.decode("utf-8").strip() for part in upstream.split(b"<|endoftext|>")[:512]]
    assert upstream_stories == complete_stories
    for row in rows:
        row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
    data = split_records(rows, seed=42)
    output["dataset"] = {"retrieval_command": command, "archive_url": url, "archive_sha256": digest(archive),
                         "archive_bytes": len(archive), "rows_sha256": digest(rows_raw), "complete_prefix_sha256": digest(prefix_raw),
                         "complete_story_count": len(complete_stories), "splits": {}}
    output["dataset"]["upstream_prefix_receipt"] = {
        "command": upstream_command, "url": provenance["source_url"], "version": provenance["source_revision"],
        "accessed_on": "2026-10-20", "bytes_received": len(upstream), "response_sha256": digest(upstream),
        "first_complete_stories_equal": 512, "full_response_saved": False,
    }
    for split, records in data.items():
        examples = text_examples(records, max_length=128)
        targets = sum(int((y != IGNORE).sum()) for _, y in examples)
        bytes_plus_eos = sum(len(r["text"].encode()) + 1 for r in records)
        assert targets == bytes_plus_eos
        assert records_sha256(records) == results["dataset"][split]["sha256"]
        assert len(records) == results["dataset"][split]["records"]
        output["dataset"]["splits"][split] = {"records": len(records), "records_sha256": records_sha256(records),
                                                 "examples": len(examples), "effective_targets": targets,
                                                 "independent_bytes_plus_eos": bytes_plus_eos}
    sets = [set(r["family"] for r in rows) for rows in data.values()]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    sampler = random.Random(42)
    examples = text_examples(data["train"], max_length=128)
    sampled_targets = sum(int((y != IGNORE).sum()) for _ in range(180) for _, y in sampler.choices(examples, k=16))
    assert sampled_targets == 337761
    output["dataset"]["replayed_training_targets"] = sampled_targets
    output["dataset"]["story_families_disjoint"] = True
    teacher = results["variants"]["top2_aux0.01"]
    assert results["teacher_variant"] == "top2_aux0.01"
    hashes = {item["path"]: item["sha256"] for item in record["artifacts"]}
    assert hashes["model.pt"] == hashes["top2_aux0.01.pt"]
    assert teacher["heldout"]["test"]["samples"][0]["generated"] == "tthe the the the the the the the"
    output["teacher"] = {"variant": results["teacher_variant"], "model_and_variant_checkpoint_sha256": hashes["model.pt"],
                         "fixed_selection_code": 'if name == "top2_aux0.01":',
                         "selection_uses_validation_ranking": False}
    output["scope"] = "重算模型參數及資料預算，核對保存紀錄；沒有重新訓練、下載權重、重測 GPU 或推論一般品質優勢。資料全文只在記憶體處理。"
    output["result"] = "all assertions passed"
    path = DEST / "audit-result.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": output["result"], "variants": len(output["variants"]),
                      "matching": output["matching"], "dataset": output["dataset"]["splits"],
                      "sampled_targets": sampled_targets}, ensure_ascii=False))


if __name__ == "__main__":
    main()
