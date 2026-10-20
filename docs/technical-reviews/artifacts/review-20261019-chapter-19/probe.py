"""核對章首成品宣稱、訓練收據與固定公開權重；不訓練或寫入遠端。"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.lesson_products.inference import Product, parse_state
from tiny_perceptron.lesson_products.data import arithmetic_worlds, choose_salt, split_of, render, answer
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, MODALITY_IDS, MODALITY_TOKENS, SelftrainedConfig


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


observed_files = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    observed_files[path] = digest(raw)
    return raw


def load(path):
    return json.loads(read(path))


def fetch(url):
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url],
        capture_output=True,
        check=True,
    )
    return result.stdout


torch.set_num_threads(2)
torch.manual_seed(0)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-19/probe.py",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "limits": "歷史 GPU 訓練與整份測試依已保存的原始執行紀錄核對；本次只執行 CPU 前向與固定公開權重的少量推論。",
}
frozen_path = "docs/course-review-20261019/freeze-01/sources/chapter-19.md"
frozen = read(frozen_path).decode()
freeze = load("docs/course-review-20261019/freeze-04.json")
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-19")
chapter = read("course/chapters/19.md").decode()
result["page"] = {
    "source_sha256": digest(frozen.encode()),
    "matches_freeze": digest(frozen.encode()) == page["source_sha256"],
    "current_intro_equal": chapter.split("## ", 1)[0] == frozen,
    "figures": page["figures_sha256"],
}
links = []
for target in re.findall(r"\]\(([^)]+)\)", frozen):
    name, _, anchor = target.partition("#")
    path = (ROOT / "course/chapters" / name).resolve()
    relative = path.relative_to(ROOT).as_posix()
    content = read(relative).decode()
    links.append({"target": target, "exists": path.is_file(), "anchor_found": not anchor or bool(re.search(r"^## " + re.escape(anchor) + r"(?:\s|$)", content, re.M))})
result["navigation"] = links

lineage = []
selected = []
for stage in ("moe-pretrain", "moe-sft", "moe-vision", "moe-ocr", "moe-audio", "moe-joint", "moe-weighted", "moe-native"):
    base = "docs/selftrained/results/training-raw/" + stage
    receipt = load(base + "/receipt.json")
    training = load(base + "/raw/train-receipt.json")
    expected = next(f["sha256"] for f in receipt["files"] if f["path"] == "train-receipt.json")
    best = next(f["sha256"] for f in receipt["files"] if f["path"] == "best.pt")
    selected.append(best)
    lineage.append({
        "name": stage,
        "origin_kind": training["origin"]["kind"],
        "steps": training["steps"],
        "returncode": receipt["returncode"],
        "receipt_hash_matches": observed_files[base + "/raw/train-receipt.json"] == expected,
        "selected_checkpoint_sha256": best,
        "test_used_for_selection": training["test_used_for_selection"],
        "stage_history_selected_hashes": [entry["checkpoint_sha256"] for entry in training["stage_history"]],
    })
result["moe_lineage"] = {"stages": lineage, "final_history_matches_predecessors": lineage[-1]["stage_history_selected_hashes"] == selected[:-1]}

examples = []
for task in ("text", "vision_clothing", "ocr", "voice_qa"):
    example = load("docs/selftrained/results/public-cpu-raw/" + task + "-1-result.json")
    stdout_path = "docs/selftrained/results/public-cpu-raw/" + task + "-1-stdout.txt"
    stdout_hash = digest(read(stdout_path))
    examples.append({
        "task": task, "returncode": example["returncode"], "answer": example["actual_answer"],
        "eos": [g["eos"] for g in example["actual_generations"]],
        "stdout_hash_matches": stdout_hash == example["stdout_sha256"],
        "model_origin": example["model"]["origin"]["kind"],
        "selected_step": example["model"]["selected_step"],
        "manifest_sha256": example["model"]["manifest_sha256"],
    })
result["saved_public_moe_examples"] = examples
metrics = load("docs/selftrained/results/public-raw/moe/test/metrics.json")
result["moe_full_test"] = {key: metrics[key] for key in ("split", "count", "expected_count", "evaluation_complete", "limited_smoke", "interrupted", "inference_manifest_sha256", "teacher_forcing_used_for_generation", "per_task_final_reply", "voice_scope")}

provenance = {}
for kind in ("vision", "ocr"):
    document = load("docs/selftrained/provenance/" + kind + "-sources.json")
    provenance[kind] = [{"url": entry["url"], "sha256": entry["sha256"], "path": entry["path"]} for entry in document["downloads"]]
voice = load("docs/selftrained/provenance/voice.json")
provenance["voice"] = {key: voice[key] for key in ("dataset", "config", "revision", "parquet", "evaluation_scope", "never_model_inputs")}
result["data_sources"] = provenance

config = SelftrainedConfig(vocab_size=64, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, experts=2, top_k=1, max_length=128)
assistant = LimitedAssistant(config).eval()
payloads = {
    "image": {"kind": "image", "values": torch.zeros(2, 1, 32, 32), "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]])},
    "ocr": {"kind": "ocr", "values": torch.zeros(1, 32, 128)},
    "audio": {"kind": "audio", "values": torch.zeros(41, 40)},
}
forward = {}
with torch.inference_mode():
    for kind, payload in payloads.items():
        tokens = torch.tensor([[1] + [MODALITY_IDS[kind]] * MODALITY_TOKENS[kind] + [11]])
        output = assistant(tokens, modalities=[[payload]])
        forward[kind] = {"input_shape": list(tokens.shape), "logits_shape": list(output["logits"].shape), "perception_kind": output["perception"][0]["kind"], "finite": bool(torch.isfinite(output["logits"]).all())}
    output = assistant(torch.tensor([[1, 11, 12]]))
    forward["text"] = {"logits_shape": list(output["logits"].shape), "finite": bool(torch.isfinite(output["logits"]).all())}
result["random_moe_forward"] = forward

records = "docs/lesson-products/results/"
execution = {}
for product in ("b", "c"):
    original = load(records + "initial-" + product + "-execution.json")
    training = original["training"]
    execution[product] = {
        "run_id": original["run_id"], "status": original["status"], "gpu_name": original["gpu_name"],
        **{key: training[key] for key in ("random_initialization", "resume_source", "steps_this_attempt", "selected_step", "seed", "parameters", "data_manifest_sha256")},
    }
settings = load(records + "settings-freeze-v1.json")
settings_hash = observed_files[records + "settings-freeze-v1.json"]
pretest = load(records + "pretest-freeze-record-v1.json")
start = load(records + "settings-freeze-v1.test-started.json")
started = load(records + "final-report-v1.started.json")
final = load(records + "final-report-v1.json")
source_matches = {}
for name, expected in settings["source_sha256s"].items():
    source_matches[name] = digest(read("tiny_perceptron/lesson_products/" + name)) == expected
result["products_training"] = execution
result["products_frozen_test"] = {
    "settings_freeze_sha256": settings_hash,
    "freeze_binding_matches": all(d["settings_freeze_sha256"] == settings_hash for d in (pretest, start, started, final)),
    "timestamp_order": settings["recorded_at"] < pretest["recorded_at"] < start["recorded_at"] < final["recorded_at"],
    "test_outcomes_observed_at_freeze": settings["test_outcomes_observed"],
    "current_source_matches": source_matches,
    "frozen_product_model_hashes": {p: settings["products"][p]["model.pt"] for p in ("b", "c")},
    "b_by_depth": {depth: {task: values[task] for task in ("boxes", "sum")} for depth, values in final["b"].items()},
    "c": final["c"], "gates": final["gates"], "scope": final["scope"],
}

release = load("tiny_perceptron/lesson_products/release.json")
remote = {}
products = {}
for kind in ("b", "c"):
    info = release["products"][kind]
    base = "https://huggingface.co/" + release["repo"] + "/resolve/" + release["revision"] + "/" + info["prefix"] + "/"
    raw_manifest = fetch(base + "artifact-manifest.json")
    assert digest(raw_manifest) == info["manifest_sha256"], "公開 manifest 指紋不符"
    manifest = json.loads(raw_manifest)
    payload = {}
    for name, expected in manifest["files"].items():
        raw = fetch(base + name)
        assert digest(raw) == expected["sha256"] and len(raw) == expected["bytes"], "公開檔案指紋或大小不符"
        payload[name] = raw
    assert digest(payload["model.pt"]) == settings["products"][kind]["model.pt"], "公開權重與測試凍結不同"
    model_config = Config(**json.loads(payload["config.json"]))
    tokenizer = Tokenizer(json.loads(payload["tokenizer.json"])["characters"])
    model = (LoopedSolver if kind == "b" else CandidateScorer)(model_config).eval()
    model.load_state_dict(torch.load(io.BytesIO(payload["model.pt"]), map_location="cpu", weights_only=True))
    product = Product.__new__(Product)
    product.product = kind
    product.device = torch.device("cpu")
    product.config, product.tokenizer, product.model, product.manifest = model_config, tokenizer, model, manifest
    product.calibration = json.loads(payload["calibration.json"]) if kind == "c" else None
    product.auto_policy = json.loads(payload["auto-policy.json"]) if kind == "b" else None
    products[kind] = product
    remote[kind] = {
        "base_url": base, "revision": release["revision"], "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "manifest_sha256": digest(raw_manifest), "files": manifest["files"], "all_payloads_verified_in_memory": True,
        "model_matches_frozen_test": True,
    }
result["public_fixed_models"] = remote
states = ["兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "三數加法;x=1;y=2;z=3", "三數加法;x=6;y=6;z=6"]
result["trained_solver_examples"] = [{"state": state, "parsed": parse_state(state), "by_depth": {str(d): products["b"].solve(state, d) for d in (1, 3, 5)}} for state in states]
data_manifest = load("docs/lesson-products/data-v2-manifest.json")
assert observed_files["docs/lesson-products/data-v2-manifest.json"] == settings["data_manifest_sha256"], "資料 manifest 與測試凍結不同"
salt = choose_salt()
sum_worlds = [w for w in arithmetic_worlds() if split_of(w["family_id"], salt) == "test"]
sum_predictions = [
    {"state": render(world), "family_id": world["family_id"], "gold": answer(world), "predicted": {str(depth): products["b"].solve(render(world), depth)["answer"] for depth in (1, 3, 5)}}
    for world in sum_worlds
]
summary = {
    str(depth): {"count": len(sum_predictions), "correct": sum(row["predicted"][str(depth)] == row["gold"] for row in sum_predictions)}
    for depth in (1, 3, 5)
}
result["replayed_heldout_arithmetic"] = {
    "salt": salt,
    "families": len({w["family_id"] for w in sum_worlds}),
    "by_depth": summary,
    "matches_recorded_test": all(summary[str(depth)][field] == final["b"][str(depth)]["sum"][field] for depth in (1, 3, 5) for field in ("count", "correct")),
    "predictions": sum_predictions,
}
state = "卡片進度;申請編號=?"
result["trained_judge_examples"] = {"normal": products["c"].judge(state, "info", ["yes", "no"]), "reordered": products["c"].judge(state, "info", ["no", "yes"])}
result["observed_files_sha256"] = observed_files
print(json.dumps(result, ensure_ascii=False, indent=2))
