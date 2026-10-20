"""離線核對訓練紀錄、加權位置、凍結參數與短程式接續契約。"""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from safetensors import safe_open

from scripts.selftrained import train
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def compact_assets(result):
    """原始資產指紋已留在來源檔；執行收據只保存數量與整表指紋。"""
    for row in result["raw_lineage"].values():
        assets = row.pop("asset_sha256")
        row["asset_fingerprints"] = {
            "entries": len(assets),
            "canonical_sha256": hashlib.sha256(json.dumps(assets, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }


if sys.argv[1:] == ["--compact-existing"]:
    target = ARTIFACT / "behavior-results.json"
    result = read(target)
    compact_assets(result)
    result["storage_note"] = "本次主程序已逐段比較完整資產指紋表；保存時改記各表的項目數及 canonical JSON SHA-256，完整原始表仍由 source_sha256 指向來源檔。"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": "重複資產表已縮成指紋", "bytes": target.stat().st_size}, ensure_ascii=False))
    raise SystemExit(0)


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-4/verify_behavior.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "scope": "每個子程序只做一或兩個更新；合成五種任務族資料只驗證接續契約，不代表正式模型品質或完整訓練重現。",
}

helper = ROOT / "outputs/reviewer-tools/runs/20261020T053603163230Z"
result["section_execution"] = {name: (helper / name).read_text() for name in ["stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json"]}
index = read(ROOT / "docs/selftrained/v2-training-stage-index.json")
result["baseline_rows"] = index["stages"]
result["raw_lineage"] = {}
sources = [
    "scripts/selftrained/train.py", "scripts/selftrained/train_local_stage.py",
    "scripts/selftrained/modal_runner.py", "tiny_perceptron/selftrained/model.py",
    "tiny_perceptron/selftrained/dataset.py", "tiny_perceptron/selftrained/tokenizer.py",
    "docs/selftrained/v2-manifest.json", "docs/selftrained/v2-training-stage-index.json",
    "docs/selftrained/results/v2-moe-weighted-10000-training-stage-index.json",
    "docs/selftrained/results/v2-moe-native-balanced-4000-training-stage-index.json",
    "docs/selftrained/results/v2-dense-weighted-10000-training-stage-index.json",
    "docs/selftrained/results/v2-final-training-source-selection.json",
    "docs/selftrained/results/v2-final-public-results.json",
    "docs/selftrained/TRAINING.md", "pyproject.toml", "scripts/build_course.py",
]
for architecture in ["moe", "dense"]:
    prior = None
    for stage in ["pretrain", "sft", "vision", "ocr", "audio", "joint", "weighted"] + (["native"] if architecture == "moe" else []):
        key = architecture + "-" + stage
        directory = "docs/selftrained/results/training-raw/" + key
        raw = read(ROOT / directory / "raw/train-receipt.json")
        execution = read(ROOT / directory / "raw/execution.json")
        receipt = read(ROOT / directory / "receipt.json")
        row = next((r for r in index["stages"] if r["stage_key"] == key), None)
        if row is None:
            suffix = "moe-native-balanced-4000" if stage == "native" else key + "-10000"
            row = read(ROOT / f"docs/selftrained/results/v2-{suffix}-training-stage-index.json")
        current = {
            "execution_status": execution.get("status"), "run_id": execution.get("run_id"),
            "stage": raw.get("stage"), "steps": raw.get("steps"),
            "completed_requested_steps": raw.get("completed_requested_steps"),
            "origin": raw.get("origin"), "stage_history": raw.get("stage_history"),
            "job": execution.get("job"), "data_sha256": raw.get("data_sha256"),
            "asset_sha256": raw.get("asset_sha256"), "selected": row["selected_private_best_receipt"],
            "selected_step": row["selected_step"], "completed_steps": row["completed_steps"],
            "freeze_perception_backbones": raw.get("freeze_perception_backbones"),
        }
        if prior:
            edge = raw["stage_history"][-1]
            current["parent_sha_matches_previous_selected"] = edge["checkpoint_sha256"] == prior["selected"]["sha256"]
            assert current["parent_sha_matches_previous_selected"], "父檢查點與上一段選定指紋不同"
            assert current["data_sha256"] == prior["data_sha256"], "相鄰訓練段資料不同"
            assert current["asset_sha256"] == prior["asset_sha256"], "相鄰訓練段資產不同"
        assert raw["origin"]["kind"] == "all-neural-weights-random", "訓練起點紀錄不同"
        assert raw["completed_requested_steps"], "來源訓練未完成"
        result["raw_lineage"][key] = current
        prior = current
        sources.extend([directory + "/raw/train-receipt.json", directory + "/raw/execution.json", directory + "/receipt.json"])
result["source_sha256"] = {path: sha(ROOT / path) for path in sources}
result["git_revision"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
old_manifest = subprocess.run(["git", "show", "selftrained-v2:docs/selftrained/v2-manifest.json"], cwd=ROOT, capture_output=True, check=True).stdout
result["manifest_pins"] = {"current": sha(ROOT / "docs/selftrained/v2-manifest.json"), "selftrained-v2": hashlib.sha256(old_manifest).hexdigest()}
result["selected_sources"] = {k: read(ROOT / "docs/selftrained/results/v2-final-training-source-selection.json")[k] for k in ["moe_joint", "dense_joint"]}
public = read(ROOT / "docs/selftrained/results/v2-final-public-results.json")
result["public_sources"] = {k: {"selected_source": v["selected_source"], "safe_weights": v["safe_weights"]} for k, v in public["architectures"].items()}

torch.set_num_threads(1)
model = LimitedAssistant(SelftrainedConfig(width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=24, experts=2, top_k=1))
result["trainable_parameters"] = {}
for stage in train.STAGES:
    train.set_trainable(model, stage, freeze_perception_backbones=stage == "joint")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    result["trainable_parameters"][stage] = names
    expected_prefix = {"vision": "vision_encoder.", "ocr": "ocr_encoder.", "audio": "audio_encoder."}.get(stage)
    if expected_prefix:
        assert all(n.startswith(expected_prefix) for n in names), "感知段更新範圍不同"
    elif stage != "joint":
        assert all(n.startswith("lm.") for n in names), "文字段更新範圍不同"
result["recognition_shapes"] = {"vision_classes": model.vision_encoder.head.out_features, "ocr_characters": len(OCR_CHARACTERS), "audio_intents": model.audio_encoder.head.out_features}

tokenizer = CharacterTokenizer.build(["-12。a3"], required_chars="0123456789-。a")
symbols = ["-", "1", "2", "。", "a", "3", "<eos>", "<ignored>"]
labels = torch.tensor([tokenizer.encode("-12。a3") + [tokenizer.eos_id, -100]])
records = [{"task": "text", "split": "train"}]
result["numeric_positions"] = {"symbols": symbols, "labels": labels.tolist(), "mask": train.numeric_run_loss_mask(labels, tokenizer).tolist(), "weights": train.language_loss_weights(labels, records, tokenizer, 1, 4).tolist()}
assert result["numeric_positions"]["weights"] == [[4, 4, 4, 4, 1, 4, 4, 0]], "數字片段加權核對失敗"
rows = [
    {"task": "tool_call", "split": "train"},
    {"task": "tool_reply", "split": "train", "supervision": {"replay_kind": "actual_executor"}},
    {"task": "tool_reply", "split": "train"},
    {"task": "voice_qa", "split": "train"},
    {"task": "voice_topic_continuation", "split": "train"},
    {"task": "voice_qa", "split": "train", "augmentation": {}},
    {"task": "voice_qa", "split": "validation"},
]
result["row_weights"] = {"records": rows, "tool": [train.genuine_tool_loss_row(r) for r in rows], "native": [train.native_voice_loss_row(r) for r in rows], "native_weights": train.language_loss_weights(labels.repeat(len(rows), 1), rows, tokenizer, 4, 1, 4).tolist()}
result["objective_policy"] = train.language_objective_metadata({"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 1, "native_voice_loss_weight": 4})

with tempfile.TemporaryDirectory(prefix="behavior-", dir=ARTIFACT) as temporary:
    work = Path(temporary)
    records_path = work / "records.jsonl"
    Image.new("RGB", (128, 64), "gray").save(work / "image.png")
    sf.write(work / "audio.wav", np.sin(2 * np.pi * 300 * np.arange(8000) / 16000).astype(np.float32) * 0.1, 16000)
    synthetic = []
    for split in ["train", "validation"]:
        for task in ["text", "tool_call", "vision_clothing", "ocr", "voice_qa"]:
            row = {"id": split + task, "group_id": split + task, "split": split, "task": task, "messages": [{"role": "user", "content": "問題"}, {"role": "assistant", "content": "回答12。"}], "supervision": {}}
            if task == "vision_clothing":
                row.update(image="image.png", image_layout={"axis": "horizontal", "slots": [[0, 0, 64, 64], [64, 0, 128, 64]]})
                row["supervision"]["vision_labels"] = [0, 1]
            elif task == "ocr":
                row.update(image="image.png", roi=[0, 0, 100, 32])
                row["supervision"]["ocr_text"] = "大小"
                row["messages"][-1]["content"] = "大小"
            elif task == "voice_qa":
                row["audio"] = "audio.wav"
                row["supervision"].update(intent="address", intent_id=0)
            synthetic.append(row)
    records_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in synthetic))
    manifest = work / "manifest.json"
    assets = [{"path": name, "bytes": (work / name).stat().st_size, "sha256": sha(work / name)} for name in ["image.png", "audio.wav"]]
    manifest.write_text(json.dumps({"schema_version": 1, "initialization": "random", "records": [{"path": records_path.name, "bytes": records_path.stat().st_size, "sha256": sha(records_path)}], "assets": assets, "model_config": {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 24, "experts": 2, "top_k": 1, "max_length": 512}}))
    runs = {}

    def invoke(label, stage, steps, extra=(), expected=0):
        output = work / label
        command = [sys.executable, str(ROOT / "scripts/selftrained/train_local_stage.py"), "--manifest", str(manifest), "--manifest-sha256", sha(manifest), "--data-root", str(work), "--", "--output-dir", str(output), "--stage", stage, "--architecture", "moe", "--steps", str(steps), "--batch-size", "16", "--context", "512", "--seed", "20261006", "--learning-rate", "0.0002", "--eval-every", "1", "--save-every", "1", "--device", "cpu", "--threads", "1", *extra]
        p = subprocess.run(command, cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, capture_output=True, text=True, timeout=50)
        entry = {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        if (output / "train-receipt.json").is_file():
            entry["training"] = read(output / "train-receipt.json")
            entry["execution"] = read(output / "execution.json")
        runs[label] = entry
        if (p.returncode == 0) != (expected == 0):
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            raise AssertionError("子程序結果不符合預期")
        return output

    whole = invoke("whole", "pretrain", 2)
    first = invoke("first", "pretrain", 1)
    resumed = invoke("resumed", "pretrain", 2, ["--resume", str(first / "latest.pt")])
    a = torch.load(whole / "latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    result["exact_resume"] = {k: same(a[k], b[k]) for k in ["model", "optimizer", "rng", "sampler", "step", "tokens", "target_tokens"]}
    assert all(result["exact_resume"].values()), "直接執行與接續的狀態不同"
    result["checkpoint_keys"] = list(b)
    result["optimizer_state_keys"] = list(next(iter(b["optimizer"]["state"].values())))
    freeze = ["--freeze-perception-backbones", "--sampling-mode", "task-family"]
    joint = invoke("joint", "joint", 1, [*freeze, "--init-checkpoint", str(first / "best.pt")])
    weighted = invoke("weighted", "joint", 1, [*freeze, "--tool-loss-weight", "4", "--numeric-run-loss-weight", "4", "--init-checkpoint", str(joint / "best.pt")])
    native = invoke("native", "joint", 1, [*freeze, "--tool-loss-weight", "4", "--numeric-run-loss-weight", "1", "--native-voice-loss-weight", "4", "--init-checkpoint", str(weighted / "best.pt")])
    invoke("changed-objective-resume", "joint", 2, [*freeze, "--tool-loss-weight", "4", "--numeric-run-loss-weight", "1", "--resume", str(weighted / "latest.pt")], expected=1)
    invoke("invalid-native-parent", "joint", 1, [*freeze, "--tool-loss-weight", "4", "--numeric-run-loss-weight", "1", "--native-voice-loss-weight", "4", "--init-checkpoint", str(joint / "best.pt")], expected=1)
    invoke("incomplete", "pretrain", 2, ["--max-tokens", "1"], expected=1)
    invoke("incomplete-init", "joint", 1, [*freeze, "--init-checkpoint", str(work / "incomplete/best.pt")], expected=1)
    with safe_open(native / "model.safetensors", framework="pt", device="cpu") as f:
        result["inference_export"] = {"tensor_count": len(f.keys()), "only_model_tensors": set(f.keys()) == set(torch.load(native / "best.pt", weights_only=False)["model"]), "metadata": f.metadata()}
    result["synthetic_runs"] = runs
    result["new_stage_reset"] = runs["native"]["training"]["origin"]["new_joint_initialization"]["reset_state"]

result["loss_vs_accuracy_counterexample"] = {
    "before_true_probabilities": [0.51, 0.51], "after_true_probabilities": [0.49, 0.99],
    "before_mean_ce": float(-np.log([0.51, 0.51]).mean()), "after_mean_ce": float(-np.log([0.49, 0.99]).mean()),
    "before_binary_correct": 2, "after_binary_correct": 1,
    "scope": "兩筆等權二類例子；平均驗證交叉熵降低，正確數仍可下降，不能由較低代價推論全部答對。",
}
compact_assets(result)
(ARTIFACT / "behavior-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "核對完成", "exact_resume": result["exact_resume"], "numeric_positions": result["numeric_positions"], "subprocesses": len(result["synthetic_runs"])}, ensure_ascii=False))
