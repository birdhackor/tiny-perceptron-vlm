"""核對分數反例、練習、生成規則與既有實驗的適用範圍，保存原始來源小段摘錄。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.prepare_data import generate_records
from tiny_perceptron.model import generate

ARTIFACT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T064458167901Z"
COMMAND = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-15/verify.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (ARTIFACT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ScoreModel(nn.Module):
    """固定分數的單序列生成替身；不代表真實量化模型。"""

    def __init__(self, scores):
        super().__init__()
        self.config = SimpleNamespace(max_length=16)
        self.register_buffer("scores", torch.tensor([-10.0, -10.0, *scores]))
        self.seen = []

    def forward(self, ids, cache=None):
        self.seen.append(ids.tolist())
        return {"logits": self.scores[None, None].expand(ids.shape[0], ids.shape[1], -1)}


def read_original(identifier, url, version, snippets, html=False):
    command = ["curl", "-fsSL", "--max-time", "30", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8", errors="replace")
    if html:
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
    excerpts = []
    for locator, start, end in snippets:
        offset = text.index(start)
        finish = text.index(end, offset) if end else offset + len(start)
        excerpt = text[offset:finish].strip()
        excerpts.append({"locator": locator, "excerpt": excerpt})
    return {
        "id": identifier,
        "url": url,
        "version": version,
        "accessed_at": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.decode(),
        "original_bytes": len(raw),
        "original_sha256": sha(raw),
        "excerpts": excerpts,
        "storage_scope": "原始全文只在記憶體中讀取；只保存核對所需的小段摘錄。",
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    environment = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
        "default_dtype": str(torch.get_default_dtype()),
    }
    copied = {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/17.md#17.15" --execute',
        "original_run_directory": str(RUN.relative_to(ROOT)),
        "files": {},
    }
    for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt"):
        raw = (RUN / name).read_bytes()
        copied["files"][name] = {"sha256": sha(raw), "content": raw.decode()}
    assert json.loads(copied["files"]["execution.json"]["content"])["exit_code"] == 0
    assert json.loads(copied["files"]["extraction.json"]["content"])["source_sha256"] == "84f4f9ba355a6f5cd541bb08b3e01270c12a22a06b8dab29bc126aef42d5a8e5"
    assert copied["files"]["stdout.txt"]["content"] == "分數MAE 0.005\n原版選擇 貓\n量化版選擇 <EOS>\n"
    save("section-execution.json", copied)

    examples = []
    for second, choices in ((1.005, [1, 0]), (2.0, [1, 1])):
        original = torch.tensor([1.0, second])
        quantized = torch.tensor([1.01, second])
        mae = (original - quantized).abs().mean().item()
        observed = [original.argmax().item(), quantized.argmax().item()]
        assert math.isclose(mae, 0.005, abs_tol=1e-8, rel_tol=0)
        assert observed == choices
        examples.append({"original": original.tolist(), "quantized": quantized.tolist(), "dtype": str(original.dtype), "mae": mae, "mae_rounded": round(mae, 4), "argmax_indices": observed, "names": [["<EOS>", "貓"][i] for i in observed]})

    greedy = []
    for scores in ([1.0, 1.005], [1.01, 1.005]):
        model = ScoreModel(scores)
        ids = generate(model, torch.tensor([[1]]), max_new_tokens=2, temperature=0, eos_id=2)
        greedy.append({"scores": scores, "ids": ids.tolist(), "seen_inputs": model.seen, "training_mode_restored": model.training})
    assert greedy[0]["ids"] == [[1, 3, 3]]
    assert greedy[1]["ids"] == [[1, 2]]
    assert greedy[0]["seen_inputs"] == [[[1]], [[1, 3]]]

    seeded = []
    for seed in (0, 1, 42):
        torch.manual_seed(seed)
        left = torch.multinomial(torch.tensor([0.4, 0.6]), 12, replacement=True).tolist()
        torch.manual_seed(seed)
        right = torch.multinomial(torch.tensor([0.4, 0.6]), 12, replacement=True).tolist()
        assert left == right
        seeded.append({"seed": seed, "probabilities": [0.4, 0.6], "left": left, "right": right})

    quant = json.loads((ROOT / "docs/course-experiments/results/quantization.json").read_bytes())
    qat = json.loads((ROOT / "docs/course-experiments/results/qat.json").read_bytes())
    records = generate_records("attributes-sft")
    pairs = {(r["messages"][-2]["content"], r["messages"][-1]["content"]) for r in records}
    inventory = {}
    for name, run in quant["results"]["runs"].items():
        inventory[name] = {}
        for split in ("validation", "test"):
            evaluation = run[split]
            samples = evaluation["generated_samples"]
            assert len(samples) == evaluation["examples"] == quant["results"]["data"]["counts"][split]
            assert all(all(isinstance(s[k], str) for k in ("question", "expected", "generated")) for s in samples)
            assert all((s["question"], s["expected"]) in pairs for s in samples)
            assert all(s["question"].startswith("color=") and ";shape=" in s["question"] and ";pitch=" in s["question"] for s in samples)
            inventory[name][split] = {"examples": len(samples), "generation": evaluation["generation"], "max_new_tokens": evaluation["max_new_tokens"], "samples": [{k: s[k] for k in ("question", "expected", "generated")} for s in samples]}
    for split in ("validation", "test"):
        reference = [(s["question"], s["expected"]) for s in inventory["fp32"][split]["samples"]]
        assert all([(s["question"], s["expected"]) for s in inventory[name][split]["samples"]] == reference for name in inventory)
    controls = {name: {key: fit[key] for key in ("steps", "optimizer_updates", "batch_size", "learning_rate", "initialization_sha256", "batch_plan_sha256", "effective_supervised_tokens")} for name, fit in qat["results"]["training"].items()}
    assert controls["fp_finetuning"] == controls["qat"]
    assert controls["qat"]["optimizer_updates"] == 350
    assert quant["results"]["training"]["optimizer_updates"] == 120
    paths = ("course/chapters/17.md", "course/training.md", "scripts/prepare_data.py", "scripts/course_experiments/compression.py", "tiny_perceptron/model.py", "tiny_perceptron/quantization.py", "docs/course-experiments/results/quantization.json", "docs/course-experiments/results/qat.json")
    hashes = {path: sha((ROOT / path).read_bytes()) for path in paths}
    training = (ROOT / "course/training.md").read_text()
    assert "## T.9 " in training and "--experiment quantization --device cpu" in training

    originals = []
    originals.append(read_original("pytorch_argmax", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py", "PyTorch installed commit 5c4886908584029761b579af026dcfb627c84070; local 2.14.1", [("torch.argmax docstring", "Returns the indices of the maximum value of all elements", "This is the second")]))
    originals.append(read_original("transformers_generation", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/llm_tutorial.md", "Transformers v4.57.1", [("Common Options: eos_token_id", "The token(s) that will cause generation to stop.", " The default value"), ("Decoding strategy", "*greedy search*, which selects the next most likely token", ", unless")]))
    originals.append(read_original("transformers_autoregression", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py", "Transformers v4.57.1; GenerationMixin._sample", [("_sample greedy token choice", "next_tokens = torch.argmax(next_token_scores, dim=-1)", ""), ("_sample context update", "input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)", "")]))
    originals.append(read_original("pytorch_randomness", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/randomness.rst", "PyTorch v2.9.0 documentation; general reproducibility guidance", [("Reproducibility limits", "Completely reproducible results are not guaranteed", "However, there are"), ("PyTorch random number generator", "as long as :meth:`torch.manual_seed()` is set", "It is also possible")]))
    originals.append(read_original("nist_controls", "https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm", "NIST/SEMATECH e-Handbook, Randomized block designs; live page retrieved on recorded UTC date", [("Blocking used for nuisance factors that can be controlled", "The basic concept is to create homogeneous blocks", " Within blocks,")], html=True))
    originals.append(read_original("pytorch_quantization", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/quantization.rst", "PyTorch v2.8.0 documentation; Introduction to Quantization", [("Introduction to Quantization: representation and execution", "This allows for a more compact model representation", "PyTorch supports INT8"), ("Introduction to Quantization: accuracy", "Higher-level\nAPIs are provided", "Quantization API Summary")]))

    report = {"command": COMMAND, "result": "所有明示斷言通過；未訓練模型、未重跑既有 GPU 品質或效能實驗。", "environment": environment, "score_examples": examples, "greedy_eos_probe": greedy, "seed_repeatability_probe": seeded, "repository_sha256": hashes, "quantization_inventory": inventory, "qat_controls": controls, "recorded_quantization_updates": quant["results"]["training"]["optimizer_updates"], "dataset_generator": {"kind": "attributes-sft", "records": len(records), "question_answer_pairs": len(pairs)}, "original_sources": originals}
    save("verification.json", report)
    print(json.dumps({"result": report["result"], "artifact_paths": [str((ARTIFACT / name).relative_to(ROOT)) for name in ("section-execution.json", "verification.json")], "environment": environment}, ensure_ascii=False))


if __name__ == "__main__":
    main()
