"""核對凍結小節的梯度、標籤與既有實驗紀錄；不重新訓練。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.modalities import _freeze, _sequence, _vision_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def gradient_case(timing):
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = MultiModalLM(TinyLM(ModelConfig(width=8)))
    model.requires_grad_(False)
    model.image_projector.requires_grad_(True)
    answer = tok.encode("red square") + [tok.eos_id]
    prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
    ids = torch.tensor(prefix + answer)
    labels = torch.tensor([-100] * len(prefix) + answer)
    if timing == "before_forward":
        model.image_projector.requires_grad_(False)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    out = model(ids, labels, image=scene())
    if timing == "after_forward":
        model.image_projector.requires_grad_(False)
    loss = masked_loss(out["logits"], out["labels"])
    selected = out["labels"] != -100
    direct = torch.nn.functional.cross_entropy(out["logits"][selected], out["labels"][selected])
    result = {
        "timing": timing,
        "prefix": prefix,
        "answer_ids": answer,
        "ascii_bytes": len("red square".encode("ascii")),
        "image_shape": list(scene().shape),
        "input_dtype": str(ids.dtype),
        "logits_dtype": str(out["logits"].dtype),
        "logits_shape": list(out["logits"].shape),
        "aligned_labels": out["labels"].tolist(),
        "active_targets": int(selected.sum()),
        "active_labels_are_answer": out["labels"][selected].tolist() == answer,
        "loss": loss.item(),
        "direct_masked_cross_entropy": direct.item(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_requires_grad": loss.requires_grad,
    }
    try:
        loss.backward()
        result["backward"] = "success"
    except RuntimeError as error:
        result["backward"] = {"type": type(error).__name__, "message": str(error)}
    grad = model.image_projector.weight.grad
    result["projector_gradient_norm"] = None if grad is None else grad.norm().item()
    result["embedding_grad_is_none"] = model.language.embedding.weight.grad is None
    result["non_none_grad_parameters"] = [name for name, p in model.named_parameters() if p.grad is not None]
    result["all_parameters_unchanged"] = all(torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    try:
        model.image_projector.weight.grad.norm()
        result["grad_norm_access"] = "success"
    except AttributeError as error:
        result["grad_norm_access"] = {"type": type(error).__name__, "message": str(error)}
    return result


def experiment_audit():
    path = "docs/course-experiments/results/projector.json"
    report = json.loads((ROOT / path).read_bytes())
    sft_path = "docs/course-experiments/results/sft.json"
    sft = json.loads((ROOT / sft_path).read_bytes())
    result = report["results"]
    splits = _vision_records()
    comparison = {}
    for name, rows in splits.items():
        recorded = result["data"]["splits"][name]
        rows_hash = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        comparison[name] = {
            "count": len(rows),
            "offsets": sorted({row["offset"] for row in rows}),
            "combinations": sorted({(row["color"], row["shape"]) for row in rows}),
            "records_equal": rows == recorded["records"],
            "hash_equal": rows_hash == recorded["sha256"],
            "effective_targets": sum(len(ByteTokenizer().encode(row["answer"])) + 1 for row in rows),
        }
    test = result["test"]
    samples = []
    tok = ByteTokenizer()
    for sample in test["samples"]:
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        samples.append({
            "family": sample["family"], "target": sample["target"],
            "generated": sample["generated"], "decoded_matches": tok.decode(raw) == sample["generated"],
            "recomputed_exact": raw == tok.encode(sample["target"]),
            "recorded_exact": sample["exact_match"], "eos": tok.eos_id in generated,
        })
    cfg = result["training"]["config"]
    model = MultiModalLM(TinyLM(ModelConfig(**cfg)))
    _freeze(model, "projector")
    sft_test = sft["results"]["after"]["test"]
    code_paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
                  "scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py",
                  "scripts/course_experiments/text.py"]
    return {
        "record_path": path, "record_sha256": digest(path),
        "sft_record_path": sft_path, "sft_record_sha256": digest(sft_path),
        "recorded_environment": {key: report[key] for key in ("revision", "seed", "device", "torch_version", "python_version", "gpu", "step_scale")},
        "scope": "本次重算既有紀錄與配置；沒有重跑 CUDA 訓練或載入歷史權重。",
        "splits": comparison,
        "recorded_steps": result["training"]["steps"],
        "history_count": len(result["training"]["history"]),
        "history_steps_contiguous": [r["step"] for r in result["training"]["history"]] == list(range(1, 301)),
        "training_effective_targets": result["training"]["effective_targets"],
        "history_effective_targets_sum": sum(r["effective_targets"] for r in result["training"]["history"]),
        "weights_changed_recorded": result["training"]["weights_changed"],
        "language_unchanged_recorded": result["language_weights_unchanged"],
        "configured_trainable_names": [name for name,p in model.named_parameters() if p.requires_grad],
        "configured_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "configured_total_parameters": sum(p.numel() for p in model.parameters()),
        "before_mean_token_nll": result["before"]["mean_token_nll"],
        "after_mean_token_nll": test["mean_token_nll"],
        "before_after_effective_tokens": [result["before"]["effective_tokens"],test["effective_tokens"]],
        "rounded_nll": [round(result["before"]["mean_token_nll"],2),round(test["mean_token_nll"],2)],
        "test_correct_recomputed": sum(s["recomputed_exact"] for s in samples),
        "test_count": len(samples), "test_effective_tokens_recomputed": comparison["test"]["effective_targets"],
        "test_samples": samples,
        "sft_test_matches_recomputed": sum(s["generated_ids"][:s["generated_ids"].index(tok.eos_id)] == tok.encode(s["expected"]) for s in sft_test["samples"]),
        "sft_test_records": sft_test["records"],
        "sft_test_effective_tokens": sft_test["effective_tokens"],
        "sft_checkpoint": sft["results"]["checkpoint"],
        "code_hashes": {path: {"current":digest(path),"recorded":report["code_sha256"].get(path),"equal":digest(path)==report["code_sha256"].get(path)} for path in code_paths},
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps({
        "environment": {"python":platform.python_version(),"torch":str(torch.__version__),"torch_git_version":torch.version.git_version,"device":"cpu"},
        "gradient_cases":[gradient_case(timing) for timing in ("projector_trainable","before_forward","after_forward")],
        "experiment_audit":experiment_audit(),
    },ensure_ascii=False,indent=2))
