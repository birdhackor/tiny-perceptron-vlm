"""核對本節算例、梯度、有效分母與保存的短訓紀錄；不執行訓練。"""

import ast
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.architecture import _forward, _routing
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, check=False)
    return p


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-9/verify_facts.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    }
    cases = {
        "較均衡": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        "集中": torch.tensor([[0.8, 0.1, 0.1]] * 3),
        "平手": torch.full((3, 3), 1 / 3),
    }
    result["examples"] = {}
    for name, prob in cases.items():
        chosen = prob.argmax(-1)
        f = torch.bincount(chosen, minlength=3).float() / len(chosen)
        mean = prob.mean(0)
        aux = 3 * (f.detach() * mean).sum()
        result["examples"][name] = {
            "chosen": chosen.tolist(), "fraction": f.tolist(), "mean_probability": mean.tolist(),
            "auxiliary": aux.item(), "total_0.01": (0.5 + 0.01 * aux).item(), "total_1": (0.5 + aux).item(),
            "input_requires_grad": prob.requires_grad, "auxiliary_requires_grad": aux.requires_grad,
        }
    assert list(cases.items())[0][0] == "較均衡"
    assert list(cases.items())[1][0] == "集中"
    assert abs(result["examples"]["較均衡"]["total_1"] - 1.5) < 1e-6
    assert abs(result["examples"]["集中"]["total_1"] - 2.9) < 1e-6
    assert result["examples"]["平手"]["fraction"] == [1.0, 0.0, 0.0]
    assert result["examples"]["平手"]["auxiliary"] == 1.0

    logits = torch.tensor([[math.log(0.8), math.log(0.1), math.log(0.1)]] * 3, dtype=torch.float64, requires_grad=True)
    p = logits.softmax(-1)
    chosen = p.argmax(-1)
    f = torch.bincount(chosen, minlength=3).double() / len(chosen)
    aux = 3 * (f.detach() * p.mean(0)).sum()
    aux.backward()
    analytic = p.detach() * (f - (p.detach() * f).sum(-1, keepdim=True))
    assert torch.allclose(logits.grad, analytic, atol=1e-14, rtol=1e-14)
    result["gradient"] = {
        "formula": "dL/dz[t,j] = N/T * p[t,j] * (f[j] - sum_i(f[i]*p[t,i])); f treated as constant",
        "gradient": logits.grad.tolist(), "analytic_matches": True,
        "chosen_requires_grad": chosen.requires_grad, "fraction_requires_grad": f.requires_grad,
        "congested_logit_gradient_positive": bool((logits.grad[:, 0] > 0).all()),
        "scope": "局部固定選擇下的輔助目標梯度，不保證所有步驟或所有資料都均衡。",
    }
    model = TinyLM(ModelConfig(width=8, heads=2, layers=2, experts=4, top_k=2, max_length=8))
    model.eval()
    examples = [(torch.tensor([1, 73, 74]), torch.tensor([73, 74, 2])), (torch.tensor([1]), torch.tensor([2]))]
    x, y, valid = pad_batch(examples)
    captured = []
    def capture(module, args, output):
        p = module.router(args[0][valid]).float().softmax(-1)
        chosen = output[2][valid.reshape(-1)]
        counts = torch.bincount(chosen.flatten(), minlength=4).float()
        f = counts / (valid.sum() * 2)
        captured.append({"choices": chosen.tolist(), "counts": counts.tolist(), "denominator": int(valid.sum()) * 2,
                         "mean_probability": p.mean(0).detach().tolist(), "auxiliary": float((4 * (f * p.mean(0)).sum()).detach())})
    handles = [block.ffn.register_forward_hook(capture) for block in model.blocks]
    out = _forward(model, x, valid)
    for handle in handles:
        handle.remove()
    aux_sum = sum(layer["auxiliary"] for layer in captured)
    assert abs(float(out["auxiliary"].detach()) - aux_sum) < 1e-6
    total, count = loss_sum(out["logits"], y)
    changed_logits = out["logits"].detach().clone()
    changed_logits[~valid] = 1000
    changed_total, changed_count = loss_sum(changed_logits, y)
    assert torch.allclose(total, changed_total)
    result["masked_two_layer_probe"] = {
        "ids": x.tolist(), "labels": y.tolist(), "valid": valid.tolist(), "effective_inputs": int(valid.sum()),
        "effective_targets": int(count), "layers": captured, "auxiliary_sum": float(out["auxiliary"].detach()),
        "task_mean": float((total / count).detach()), "weighted_total": float((total / count + 0.01 * out["auxiliary"]).detach()),
        "masked_logits_change_preserves_task_loss": True,
        "scope": "CPU 微型前向核對有效分母與相加規則；並非短訓的重現。",
    }
    routing = _routing(model, examples, SimpleNamespace(device="cpu"))
    assert routing["effective_input_tokens"] == 4
    assert all(layer["dispatch_denominator"] == 8 for layer in routing["layers"])
    result["routing_probe"] = routing

    raw = (ROOT / "docs/course-experiments/results/moe.json").read_bytes()
    saved = json.loads(raw)
    result["experiment"] = {"path": "docs/course-experiments/results/moe.json", "sha256": sha(raw),
                            "revision": saved["revision"], "device": saved["device"], "gpu": saved["gpu"],
                            "torch": saved["torch_version"], "python": saved["python_version"], "seed": saved["seed"],
                            "dataset": saved["results"]["dataset"], "variants": {}}
    expected = {
        "top1_aux0": ([0.74231, 0.64755], 2.30115, 2.32369),
        "top1_aux0.01": ([0.99499, 0.97937], 2.29775, 2.31963),
        "top2_aux0": ([0.99800, 0.75365], 2.28011, 2.30320),
        "top2_aux0.01": ([0.99944, 0.99254], 2.27819, 2.30170),
    }
    for name, exp in expected.items():
        v = saved["results"]["variants"][name]
        r = v["validation_routing"]
        recalc = []
        for index, layer in enumerate(r["layers"]):
            counts = layer["dispatch_counts"]
            denom = sum(counts)
            f = [n / denom for n in counts]
            entropy = -sum(p * math.log(p) for p in f if p) / math.log(4)
            assert denom == r["effective_input_tokens"] * v["model"]["config"]["top_k"]
            assert abs(entropy - exp[0][index]) <= 0.000005
            assert abs(entropy - layer["normalized_load_entropy"]) < 1e-14
            recalc.append({"counts": counts, "denominator": denom, "normalized_entropy": entropy,
                           "mean_batch_auxiliary": layer["mean_batch_auxiliary"],
                           "mean_probability": layer["mean_router_probability"],
                           "global_product_auxiliary": 4 * sum(a * b for a, b in zip(f, layer["mean_router_probability"]))})
        heldout = {}
        for split, expected_nll in zip(["validation", "test"], exp[1:]):
            h = v["heldout"][split]
            calc = h["nll_sum"] / h["effective_tokens"]
            assert abs(calc - h["nll"]) < 1e-14
            assert abs(calc - expected_nll) <= 0.000005
            heldout[split] = {k: h[k] for k in ["nll", "nll_sum", "effective_tokens", "examples", "records"]}
        tr = v["training"]
        assert tr["steps"] == tr["optimizer_updates"] == 180
        assert [h["step"] for h in tr["history"]] == [1] + list(range(22, 181, 22))
        result["experiment"]["variants"][name] = {
            "model_config": v["model"]["config"],
            "training": {k: tr[k] for k in ["steps", "optimizer_updates", "requested_steps", "skipped_updates", "effective_tokens", "history", "batch_size", "micro_batch_sizes", "auxiliary_weight", "learning_rate", "schedule"]},
            "heldout": heldout, "routing": recalc,
            "routing_effective_input_tokens": r["effective_input_tokens"], "padding_excluded": r["padding_excluded"],
        }
    assert abs(result["experiment"]["variants"]["top2_aux0.01"]["routing"][1]["mean_batch_auxiliary"] - 0.99063) <= 0.000005
    result["experiment"]["paired_comparisons"] = {}
    for k in [1, 2]:
        zero, aux = [result["experiment"]["variants"][f"top{k}_aux{w}"] for w in ["0", "0.01"]]
        differences = {split: aux["heldout"][split]["nll"] - zero["heldout"][split]["nll"] for split in ["validation", "test"]}
        entropy_gain = [a["normalized_entropy"] - z["normalized_entropy"] for z, a in zip(zero["routing"], aux["routing"])]
        assert all(x < 0 for x in differences.values()) and all(x > 0 for x in entropy_gain)
        result["experiment"]["paired_comparisons"][str(k)] = {"nll_difference": differences, "entropy_gain": entropy_gain}

    result["implementation_provenance"] = {}
    path = "scripts/course_experiments/architecture.py"
    old_ref = saved["revision"] + ":" + path
    unavailable = run(["git", "show", old_ref])
    result["implementation_provenance"]["recorded_revision_read"] = {"command": "git show " + old_ref, "exit_code": unavailable.returncode,
                                                                    "stderr": unavailable.stderr.decode() if unavailable.returncode else ""}
    matched_ref = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
    historical = run(["git", "show", matched_ref + ":" + path])
    assert historical.returncode == 0
    assert sha(historical.stdout) == saved["code_sha256"][path]
    current = (ROOT / path).read_bytes()
    old_tree, current_tree = ast.parse(historical.stdout), ast.parse(current)
    compared = {}
    for name in ["_text_dataset", "_forward", "_train", "_heldout", "_routing", "run_moe"]:
        nodes = [next(n for n in tree.body if getattr(n, "name", None) == name) for tree in [old_tree, current_tree]]
        compared[name] = ast.dump(nodes[0]) == ast.dump(nodes[1])
        assert compared[name]
    result["implementation_provenance"].update({"matching_commit": matched_ref, "historical_sha256": sha(historical.stdout),
                                                 "current_sha256": sha(current), "compared_function_AST": compared})
    result["implementation_provenance"]["supporting_files"] = {}
    for path in ["scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py"]:
        digest = sha((ROOT / path).read_bytes())
        assert digest == saved["code_sha256"][path]
        result["implementation_provenance"]["supporting_files"][path] = digest

    archive_path = ROOT / "assets/training/tinystories-v1.tar.gz"
    archive = archive_path.read_bytes()
    acquisition = {"path": str(archive_path.relative_to(ROOT))}
    if archive.startswith(b"version https://git-lfs.github.com/spec/v1"):
        url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/assets/training/tinystories-v1.tar.gz"
        fetched = run(["curl", "-L", "--fail", "--silent", "--show-error", url])
        acquisition.update({"command": "curl -L --fail --silent --show-error " + url, "exit_code": fetched.returncode})
        if fetched.returncode:
            acquisition["error"] = fetched.stderr.decode()
            result["dataset_reconstruction"] = acquisition
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        archive = fetched.stdout
    assert sha(archive) == saved["assets"][0]["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("tinystories-train-512.jsonl")]
        assert len(members) == 1
        rows_raw = tar.extractfile(members[0]).read()
    rows = [json.loads(line) for line in rows_raw.splitlines() if line.strip()]
    for record in rows:
        record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
    data = split_records(rows, 42)
    observed = {key: {"records": len(value), "sha256": records_sha256(value)} for key, value in data.items()}
    assert observed == saved["results"]["dataset"]
    family_sets = {key: {r["family"] for r in records} for key, records in data.items()}
    intersections = {a + "/" + b: len(family_sets[a] & family_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    assert not any(intersections.values())
    tokenizer = ByteTokenizer()
    tokens = {}
    for name in ["validation", "test"]:
        examples = text_examples(data[name], max_length=128)
        count = sum(len(y) for _, y in examples)
        expected_count = sum(len(tokenizer.encode(r["text"])) + 1 for r in data[name])
        assert count == expected_count
        assert count == saved["results"]["variants"]["top2_aux0.01"]["heldout"][name]["effective_tokens"]
        tokens[name] = {"examples": len(examples), "byte_plus_EOS_targets": count}
    result["dataset_reconstruction"] = {**acquisition, "archive_sha256": sha(archive), "archive_bytes": len(archive), "raw_rows_sha256": sha(rows_raw),
                                       "splits": observed, "family_intersection_counts": intersections, "target_counts": tokens}
    result["specialization_derivation"] = {
        "assumptions": "兩筆相同的目標；expert A 的正確目標機率為 0.99，expert B 為 0.5；只比較選到的 expert 的 NLL。",
        "both_to_A_mean_nll": -math.log(0.99),
        "balanced_A_B_mean_nll": -(math.log(0.99) + math.log(0.5)) / 2,
        "conclusion": "均勻選擇與最小任務 NLL 可以衝突；這是可能性的算例，不是本節實驗的品質測量。",
    }
    result["all_assertions_passed"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
