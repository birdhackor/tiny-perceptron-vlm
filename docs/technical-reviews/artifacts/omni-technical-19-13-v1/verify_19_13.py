"""19.13 的有界 CPU／歷史證據查核；不訓練、不改寫原始證據。"""

import argparse
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.selftrained.chat import parser as assistant_parser
from tiny_perceptron.lesson_products.data import answer
from tiny_perceptron.lesson_products.inference import EFFORT, Product, parse_state
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, padded
from tiny_perceptron.lesson_products.training import state_digest


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
STATE = "兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A"
MISSING = "卡片進度;申請編號=?"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def run_cli(arguments):
    command = [sys.executable, "-m", "tiny_perceptron.lesson_products", *arguments]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    return {"argv": command, "exit_code": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "result": json.loads(result.stdout)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    freeze = read(ROOT / FREEZE)
    pages = {}
    # 前置只核相應的文字／介面 slice；不將前置頁之圖視為本頁圖或宣稱已看圖。
    for page in freeze["pages"]:
        if page["page_id"] not in {"19.13", "19.15", "19.19", "19.20", "9.11", "7.23",
                                   "19.21", "19.22", "19.23", "19.24"}:
            continue
        raw = (ROOT / page["snapshot"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
        current = (ROOT / page["source"]).read_text()
        match = re.search(r"^## " + re.escape(page["page_id"]) + r" .*?(?=^## |\Z)",
                          current, re.MULTILINE | re.DOTALL)
        assert match is not None and match.group().encode() == raw
        pages[page["page_id"]] = {
            "snapshot": page["snapshot"], "source_sha256": page["source_sha256"],
            "current_section_bytes_equal": True, "python_fences": raw.count(b"```python"),
        }
    raw = (ROOT / pages["19.13"]["snapshot"]).read_text()
    assert "```" not in raw and not re.search(r"!\[|<img|\.svg", raw)
    assert STATE in raw and MISSING in raw
    assert next(p for p in freeze["pages"] if p["page_id"] == "19.13")["figures_sha256"] == {}

    release = read(ROOT / "tiny_perceptron/lesson_products/release.json")
    history = {}
    products = {}
    essential = ["tiny_perceptron/lesson_products/" + name + ".py"
                 for name in ("data", "model", "training", "train")]
    for product, cls in (("b", LoopedSolver), ("c", CandidateScorer)):
        directory = args.packages / product
        execution_path = ROOT / f"docs/lesson-products/results/initial-{product}-execution.json"
        inspection = read(ROOT / f"docs/lesson-products/results/initial-{product}-inspection.json")
        execution = read(execution_path)
        canonical = json.dumps(execution, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert hashlib.sha256(canonical).hexdigest() == inspection["receipt_canonical_sha256"]
        receipt = execution["training"]
        assert execution["status"] == "completed" and receipt["random_initialization"]
        assert receipt["resume_source"] is None and receipt["weights_changed"]
        assert receipt["selected_step"] > 0
        assert all(sha(ROOT / path) == execution["source_hashes"][path] for path in essential)
        assert sha(directory / "artifact-manifest.json") == release["products"][product]["manifest_sha256"]
        instance = Product(directory, product, "cpu")
        assert sha(directory / "model.pt") == execution["client_artifacts"]["model.pt"]["sha256"]
        assert state_digest(instance.model.state_dict()) == receipt["selected_weights_sha256"]
        assert read(directory / "training-receipt.json") == receipt
        # 只重建初始化，零 backward／optimizer step；不是重跑正式訓練。
        torch.manual_seed(receipt["seed"])
        initial = cls(Config(**read(directory / "config.json")))
        rebuilt = state_digest(initial.state_dict())
        assert rebuilt == receipt["initial_weights_sha256"]
        history[product] = {key: receipt[key] for key in (
            "run_identity", "product", "mode", "random_initialization", "seed", "batch_size",
            "steps_this_attempt", "selected_step", "parameters", "data_manifest_sha256",
            "initial_weights_sha256", "selected_weights_sha256", "weights_changed", "environment")}
        history[product].update({"execution_path": str(execution_path.relative_to(ROOT)),
                                "execution_sha256": sha(execution_path),
                                "canonical_execution_digest_matches_original_inspection": True,
                                "rebuilt_random_initial_digest": rebuilt,
                                "essential_training_sources_equal_historical_bytes": True,
                                "package_files": instance.manifest["files"],
                                "published_state_digest_equals_historical_selected": True})
        products[product] = instance

    b = products["b"]
    tokens = padded([b.tokenizer.encode(STATE)], b.config.max_length)
    repeated = []
    for loops in (1, 3, 5):
        calls = []
        handle = b.model.shared.register_forward_hook(lambda module, inputs, output: calls.append(id(module)))
        with torch.inference_mode():
            logits = b.model(tokens, loops=loops)
        handle.remove()
        assert len(calls) == loops and len(set(calls)) == 1
        assert logits.shape == (1, 19)
        repeated.append({"loops": loops, "same_shared_module_calls": len(calls),
                         "output_shape": list(logits.shape), "answer": int(logits.argmax(-1).item())})
    cli = [run_cli(["solve", "--b", str(args.packages / "b"), "--state", STATE, "--loops", str(k)])
           for k in (1, 3, 5)]
    assert all(item["result"]["answer"] == 1 for item in cli)
    for question, selected in (("choice", "card"), ("info", "no"), ("rubric", "grade0")):
        item = run_cli(["judge", "--c", str(args.packages / "c"), "--state", MISSING,
                        "--question", question])
        assert item["result"]["selected"] == selected
        assert abs(sum(item["result"]["probabilities"].values()) - 1) < 1e-6
        cli.append(item)
    auto = run_cli(["auto", "--b", str(args.packages / "b"), "--c", str(args.packages / "c"),
                    "--state", STATE, "--effort", "auto"])
    assert auto["result"]["solution"]["answer"] == 1
    cli.append(auto)
    world = {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("A", 1)], "query": "A"}
    assert answer(world) == 1
    assert parse_state("三數加法;x=0;y=6;z=6") == {"task": "sum"}
    try:
        parse_state("三數加法;x=7;y=6;z=6")
    except ValueError:
        sum_out_of_range_rejected = True
    else:
        raise AssertionError("0–6 range must be enforced")
    assistant_options = sorted({option for action in assistant_parser()._actions for option in action.option_strings})
    assert "--effort" not in assistant_options and "loops" not in inspect.signature(Product.judge).parameters
    final_path = ROOT / "docs/lesson-products/results/final-report-v1.json"
    final = read(final_path)
    settings = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
    assert final["settings_freeze_sha256"] == sha(settings)
    assert read(settings)["test_outcomes_observed"] is False
    assert read(args.packages / "b/artifact-manifest.json")["settings_freeze_sha256"] == sha(settings)
    assert final["gates"] == {"b_primary": True, "b_secondary_target": False, "c_choice": True, "c_reorder": True}
    original_metrics = ROOT / "docs/selftrained/results/public-raw/moe/test/metrics.json"
    moe = read(original_metrics)
    exported = args.source_cache / "moe-joint-inference-manifest.json"
    export = read(exported)
    assert sha(exported) == moe["inference_manifest_sha256"]
    assert export["files"]["model.safetensors"] == moe["safe_weights_sha256"]
    assert moe["per_task_final_reply"]["tool_call"]["tool_roundtrip"]["rate"] < moe["thresholds"]["tool_roundtrip"]
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(ROOT)) +
                   " --packages " + str(args.packages) + " --source-cache " + str(args.source_cache) +
                   " --output " + str(args.output),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "platform": platform.platform(), "threads": "2"},
        "scope": "19.13 has no executable fences/figures. Its exact inline inputs were run through fixed B/C CLI; no A weight download, new training, GPU or paid calls. Timings are incidental and not benchmark evidence. Historical L4 training/test outputs are inspected, not rerun.",
        "freeze": {"path": FREEZE, "sha256": sha(ROOT / FREEZE)}, "page_bindings": pages,
        "exact_inline_inputs": {"boxes": STATE, "missing_application": MISSING},
        "world_trace": {"initial": {"A": 2, "B": 6}, "after_first": {"A": 5, "B": 6},
                        "after_second": {"A": 1, "B": 6}, "answer": answer(world)},
        "sum_contract": {"accepted": "三數加法;x=0;y=6;z=6", "x_7_rejected": sum_out_of_range_rejected,
                         "possible_gold_sums": "0..18", "not_a_quality_measurement": True},
        "b_shared_module": repeated, "history": history, "cli": cli,
        "interfaces": {"effort": EFFORT, "assistant_options": assistant_options,
                       "assistant_has_effort_option": False, "auto_neural_models": ["b", "c"]},
        "historical_bc_test": {"path": str(final_path.relative_to(ROOT)), "sha256": sha(final_path),
                               "recorded_at": final["recorded_at"], "settings_sha256": sha(settings),
                               "b5_boxes": final["b"]["5"]["boxes"], "b5_sum": final["b"]["5"]["sum"],
                               "c_choice": final["c"]["choice"]["after"], "gates": final["gates"],
                               "raw_predictions_not_downloaded": final["predictions"]},
        "historical_moe_test": {"path": str(original_metrics.relative_to(ROOT)), "sha256": sha(original_metrics),
                                "count": moe["count"], "evaluation_complete": moe["evaluation_complete"],
                                "weight_source": moe["weight_source"], "safe_weights_sha256": moe["safe_weights_sha256"],
                                "inference_manifest_sha256": sha(exported), "origin": export["origin"]["kind"],
                                "tool_roundtrip": moe["per_task_final_reply"]["tool_call"]["tool_roundtrip"],
                                "tool_roundtrip_required": moe["thresholds"]["tool_roundtrip"],
                                "general_image_recognition_not_established": True},
        "result": "PASS: exact overview examples and historical lineage checked within stated scope",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"result": result["result"], "output": str(args.output), "sha256": sha(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
