"""chapter-19 導讀的獨立 CPU／既有原始分數核對；不訓練任何模型。"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.inference import Product, check_manifest, parse_state
from tiny_perceptron.selftrained.inference import verify_export
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")
RAW = Path("/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


torch.set_num_threads(1)
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_chapter_19",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/probe.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "僅核 chapter-19 導讀主張；原始歷史分數重算與短 CPU 介面檢查，無新訓練。",
}

# 精確頁面與連結路線；不把導讀的通過冒稱逐節核准。
freeze = read("docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json")
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-19")
snapshot = (ROOT / page["snapshot"]).read_bytes()
chapter = (ROOT / page["source"]).read_bytes()
actual_intro = chapter[: re.search(rb"^## ", chapter, re.M).start()]
assert actual_intro == snapshot and hashlib.sha256(snapshot).hexdigest() == page["source_sha256"]
assert page["figures_sha256"] == {} and b"```" not in snapshot and b"![" not in snapshot
headings = {}
for ch, targets in {
    "07": ["7.1", "7.2", "7.3"], "09": [f"9.{i}" for i in range(11, 16)],
    "10": [], "11": [], "12": [], "14": [f"14.{i}" for i in range(11, 16)],
    "15": ["15.4", "15.5"], "19": [f"19.{i}" for i in range(1, 25)], "20": [],
}.items():
    lines = (ROOT / f"course/chapters/{ch}.md").read_text().splitlines()
    headings[ch] = [lines[0]] + [line for line in lines if any(re.match(r"^## " + re.escape(t) + r"\s", line) for t in targets)]
    assert len(headings[ch]) == len(targets) + 1
result["source_slice"] = {"snapshot": page["snapshot"], "bytes": len(snapshot), "sha256": page["source_sha256"], "matches_current": True, "figures": {}, "code_fences": 0}
result["navigation_headings"] = headings

# 官方來源已由本人下載／讀取；全文只保留 Git 外，這裡只存收據。
result["official_source_receipts"] = json.loads(Path("/workspace/work/reference/omni-technical-chapter-19-v1/fetch-receipt.json").read_text())
for entry in result["official_source_receipts"]:
    assert sha(entry["path"]) == entry["sha256"]

# 本次實裝官方 PyTorch 初始化實作。這僅說明 constructor，不是歷史訓練證明。
linear = Path(inspect.getsourcefile(torch.nn.Linear))
result["pytorch_original"] = {"git_version": torch.version.git_version, "file": str(linear), "sha256": sha(linear), "inspected": "Linear.__init__ calls reset_parameters; reset_parameters uses kaiming_uniform_ and uniform_."}

# V2 的三個感知入口，完整素材特徵進入同一文字核心；權重隨機不代表能力。
config = SelftrainedConfig(width=32, layers=1, heads=4, kv_heads=2, ffn_hidden=64, experts=2, top_k=1, max_length=128)
torch.manual_seed(19)
assistant = LimitedAssistant(config).eval()
torch.manual_seed(20)
second = LimitedAssistant(config).eval()
random_differences = {name: not torch.equal(assistant.state_dict()[name], second.state_dict()[name]) for name in ["lm.embedding.weight", "vision_encoder.cnn.0.weight", "ocr_encoder.cnn.0.weight", "audio_encoder.temporal.0.weight"]}
assert all(random_differences.values())
tokens = torch.tensor([[1] + [7] * 2 + [8] * 32 + [9] * 16 + [10]], dtype=torch.long)
payloads = [[
    {"kind": "image", "values": torch.zeros(2, 1, 32, 32), "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]])},
    {"kind": "ocr", "values": torch.zeros(1, 32, 128)},
    {"kind": "audio", "values": torch.zeros(40, 40)},
]]
with torch.inference_mode():
    embeddings, perception = assistant._embeddings(tokens, None, payloads)
    output = assistant(tokens, modalities=payloads)
    generated = assistant.generate(tokens, modalities=payloads, max_new_tokens=2)
assert list(embeddings.shape) == [1, 52, 32] and list(output["logits"].shape) == [1, 52, 256]
assert [p["kind"] for p in perception] == ["image", "ocr", "audio"]
assert 1 <= generated.shape[1] <= 2
result["assistant_cpu_contract"] = {"seed": 19, "config": config.__dict__, "inputs": {"ids_shape": list(tokens.shape), "image": [2, 1, 32, 32], "ocr": [1, 32, 128], "audio": [40, 40], "values": "synthetic zeros; public image coordinates [0.25,0.5],[0.75,0.5]"}, "embedding_shape": list(embeddings.shape), "logits_shape": list(output["logits"].shape), "perception_shapes": {p["kind"]: list(p["logits"].shape) for p in perception}, "generated_token_ids": generated.tolist(), "different_seed_changes_learned_weights": random_differences, "limitation": "隨機小配置只證明向量交接與生成通路；不證明語義或 V2 成績。"}

# 歷史權重身分仍與公開／測試凍結一致；不改任何舊檔案。
settings_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
settings = read(str(settings_path.relative_to(ROOT)))
freeze_sha = sha(settings_path)
historical = subprocess.check_output(["git", "show", "821a2c717f05ad7b57e08e88cd965e92ff0c1200:docs/lesson-products/results/settings-freeze-v1.json"], cwd=ROOT)
assert hashlib.sha256(historical).hexdigest() == freeze_sha
products = {}
for key in ["b", "c"]:
    directory = ROOT / "outputs/lesson-products" / key
    manifest = check_manifest(directory, key)
    assert manifest["settings_freeze_sha256"] == freeze_sha
    for name, expected in settings["products"][key].items():
        assert sha(directory / name) == expected
    receipt = json.loads((directory / "training-receipt.json").read_bytes())
    original = read(f"docs/lesson-products/results/initial-{key}-execution.json")
    assert receipt == original["training"]
    assert receipt["random_initialization"] and receipt["weights_changed"]
    assert receipt["environment"]["device"] == "cuda" and receipt["environment"]["gpu"] == "NVIDIA L4"
    products[key] = {"manifest_sha256": sha(directory / "artifact-manifest.json"), "model_sha256": sha(directory / "model.pt"), "training_receipt_sha256": sha(directory / "training-receipt.json"), "last_step": receipt["last_step"], "selected_step": receipt["selected_step"], "seed": receipt["seed"], "initial_weights_sha256": receipt["initial_weights_sha256"], "selected_weights_sha256": receipt["selected_weights_sha256"], "environment": receipt["environment"], "original_execution_sha256": sha(ROOT / f"docs/lesson-products/results/initial-{key}-execution.json")}
b = Product(ROOT / "outputs/lesson-products/b", "b")
c = Product(ROOT / "outputs/lesson-products/c", "c")
state_b = "兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A"
state_sum = "三數加法;x=1;y=2;z=3"
state_c = "卡片進度;申請編號=?"
result["products_cpu_contract"] = {"b_state": state_b, "b_parse": parse_state(state_b), "b_result": b.solve(state_b, loops=5), "sum_state": state_sum, "sum_parse": parse_state(state_sum), "sum_result": b.solve(state_sum, loops=5), "c_state": state_c, "c_result": c.judge(state_c, question="info")}
assert result["products_cpu_contract"]["b_result"]["answer"] == 1
probabilities = result["products_cpu_contract"]["c_result"]["probabilities"]
assert set(probabilities) == {"yes", "no"} and abs(sum(probabilities.values()) - 1) < 1e-6
result["historical_products"] = products

v2_cache = ROOT / "outputs/omni-revision/baseline/moe-joint"
v2_manifest, _ = verify_export(v2_cache)
v2_results = read("docs/selftrained/results/v2-final-public-results.json")
assert sha(v2_cache / "model.safetensors") == v2_results["architectures"]["moe"]["safe_weights"]["sha256"]
v2_stages = {}
for key in ["moe-pretrain", "moe-sft", "moe-vision", "moe-ocr", "moe-audio", "moe-joint", "moe-weighted", "moe-native"]:
    path = ROOT / f"docs/selftrained/results/training-raw/{key}/raw/train-receipt.json"
    receipt = json.loads(path.read_bytes())
    assert receipt["origin"]["kind"] == "all-neural-weights-random" and receipt["test_used_for_selection"] is False
    v2_stages[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "steps": receipt["steps"], "origin_kind": receipt["origin"]["kind"], "source_stages": [s["stage"] for s in receipt["stage_history"]]}
result["historical_v2"] = {"safe_weights_sha256": sha(v2_cache / "model.safetensors"), "inference_manifest_sha256": sha(v2_cache / "inference-manifest.json"), "origin_kind": v2_manifest["origin"]["kind"], "stages": v2_stages, "test_count": v2_results["architectures"]["moe"]["final_test"]["count"], "capability_acceptance_passed": v2_results["architectures"]["moe"]["capability_acceptance_passed"], "limitation": "既有 V2 正式紀錄與權重身分核對；未重新做3734題推論，不外推新版能力。"}

# 本人由逐題原始 score argmax 重算，而非抄用另一 reviewer 的表格。
raw_digest = hashlib.sha256()
counts = defaultdict(lambda: {"count": 0, "correct": 0, "families": set()})
c_counts = Counter()
with gzip.open(RAW, "rb") as handle:
    for line in handle:
        raw_digest.update(line)
        row = json.loads(line)
        if row.get("split") != "test":
            continue
        if row["product"] == "b":
            scores = row["scores_by_depth"]["5"]
            predicted = max(range(len(scores)), key=scores.__getitem__)
            assert predicted == row["answers_by_depth"]["5"]
            item = counts[row["task"]]
            item["count"] += 1
            item["correct"] += predicted == row["gold"]
            item["families"].add(row["family_id"])
        elif row["product"] == "c" and row["question"] == "choice":
            predicted = row["candidates"][max(range(len(row["raw_scores"])), key=row["raw_scores"].__getitem__)]
            assert predicted == row["selected"]
            c_counts["count"] += 1
            c_counts["correct"] += predicted == row["gold"]
            c_counts["reorder_consistent"] += predicted == row["reverse_selected"]
report = read("docs/lesson-products/results/final-report-v1.json")
assert raw_digest.hexdigest() == report["predictions"]["sha256"]
for key, item in counts.items():
    item["families"] = len(item["families"])
    item["accuracy"] = item["correct"] / item["count"]
    assert item["correct"] == report["b"]["5"][key]["correct"] and item["count"] == report["b"]["5"][key]["count"]
gates = {"b_primary": counts["boxes"]["accuracy"] >= 0.90, "b_secondary_target": counts["sum"]["accuracy"] >= 0.85, "c_choice": c_counts["correct"] / c_counts["count"] >= 0.90, "c_reorder": c_counts["reorder_consistent"] / c_counts["count"] >= 0.99}
assert gates == report["gates"]
pretest = read("docs/lesson-products/results/pretest-freeze-record-v1.json")
marker = read("docs/lesson-products/results/settings-freeze-v1.test-started.json")
assert pretest["settings_freeze_sha256"] == marker["settings_freeze_sha256"] == report["settings_freeze_sha256"] == freeze_sha
assert settings["recorded_at"] < pretest["recorded_at"] < marker["recorded_at"] < report["recorded_at"]
result["historical_score_replay"] = {"raw_cache": str(RAW), "gzip_sha256": sha(RAW), "decompressed_sha256": raw_digest.hexdigest(), "b_at_5_loops": dict(counts), "c_choice": dict(c_counts), "gates": gates, "settings_sha256": freeze_sha, "pretest_commit": "821a2c717f05ad7b57e08e88cd965e92ff0c1200", "commit_contains_identical_settings": True, "times": {"freeze": settings["recorded_at"], "pretest": pretest["recorded_at"], "test_started": marker["recorded_at"], "result": report["recorded_at"]}, "limitation": "重算保存的逐題分數與既有單seed分母；無新訓練或整套模型重測，非一般中文／任意素材泛化證明。"}

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(OUT.relative_to(ROOT)), "sha256": sha(OUT), "gates": gates, "b_at_5_loops": dict(counts), "c_choice": dict(c_counts), "b_example": result["products_cpu_contract"]["b_result"]["answer"], "c_example": probabilities}, ensure_ascii=False, indent=2))
