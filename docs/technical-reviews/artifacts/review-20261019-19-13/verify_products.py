"""核對有限成品的公開檔案、模型入口、算例與既存實測紀錄。"""

import hashlib
import gzip
import inspect
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products.data import CANDIDATES, MEANINGS, QUESTIONS, answer, judgment, render
from tiny_perceptron.lesson_products.inference import EFFORT, Product, auto_solve, parse_state
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
from tiny_perceptron.lesson_products.training import state_digest
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tools import execute_tool_call, serialize_tool_call


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(path):
    return json.loads((ROOT / path).read_bytes())


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    result = subprocess.run(command, check=True, capture_output=True)
    return result.stdout


def cli(arguments):
    command = [sys.executable, "-m", "tiny_perceptron.lesson_products", *arguments]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, env=os.environ)
    return {"command": command, "returncode": result.returncode, "result": json.loads(result.stdout)}


def main():
    torch.set_num_threads(2)
    report = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-sync python docs/technical-reviews/artifacts/review-20261019-19-13/verify_products.py",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "threads": str(torch.get_num_threads()),
        },
    }
    paths = [
        "docs/course-review-20261019/freeze-01/sources/19.13.md",
        "docs/course-review-20261019/freeze-04.json",
        "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/data.py",
        "tiny_perceptron/lesson_products/inference.py",
        "tiny_perceptron/lesson_products/training.py",
        "tiny_perceptron/lesson_products/train.py",
        "tiny_perceptron/lesson_products/__main__.py",
        "tiny_perceptron/lesson_products/delivery.py",
        "tiny_perceptron/lesson_products/evaluation.py",
        "tiny_perceptron/lesson_products/release.json",
        "tiny_perceptron/selftrained/model.py",
        "tiny_perceptron/selftrained/inference.py",
        "tiny_perceptron/selftrained/tools.py",
        "scripts/selftrained/chat.py",
        "docs/lesson-products/results/final-report-v1.json",
        "docs/lesson-products/results/initial-b-execution.json",
        "docs/lesson-products/results/initial-c-execution.json",
        "docs/lesson-products/results/b-artifact-manifest.json",
        "docs/lesson-products/results/c-artifact-manifest.json",
        "docs/lesson-products/results/publication-package-v1.json",
        "docs/selftrained/results/v2-final-public-results.json",
        "docs/selftrained/results/v2-final-training-source-selection.json",
        "docs/selftrained/results/public-raw/moe/test/metrics.json",
    ]
    report["local_sha256"] = {path: sha((ROOT / path).read_bytes()) for path in paths}
    report["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = (ROOT / paths[0]).read_text()
    report["page_inventory"] = {"python_blocks": source.count("```python"), "svg_references": source.count(".svg)")}

    url = "https://docs.typesafe.ai/introduction.md"
    raw = fetch(url)
    text = raw.decode()
    report["typesafe_original"] = {
        "url": url,
        "html_url": "https://docs.typesafe.ai/introduction",
        "retrieval_command": "curl --fail --silent --show-error --location --max-time 45 " + url,
        "accessed_on": report["recorded_at"][:10],
        "response_sha256": sha(raw),
        "response_bytes": len(raw),
        "locator": "Introduction 的開場契約與 TypeSafe primitives 表",
        "excerpt": "Send state and typed questions; get structured answers your code can use directly.",
        "observed_contract": {
            "Choice": "從候選選擇，回傳 choice、probabilities、confidence",
            "Score": "依規準評級，回傳 score、probabilities、confidence",
            "Noul": "判斷敘述是否為真，回傳 0–1",
        },
        "checked": all(word in text for word in ("state", "Choice", "Score", "Noul", "probabilities")),
        "scope": "只核對輸入輸出契約；不據此推斷官方模型架構、訓練或本專案品質。",
    }
    assert report["typesafe_original"]["checked"]

    world = {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("A", 1)], "query": "A"}
    example = "兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A"
    assert render(world) == example and answer(world) == 1
    sums = [answer({"task": "sum", "x": x, "y": y, "z": z}) for x, y, z in itertools.product(range(7), repeat=3)]
    report["world_contract"] = {
        "example": example,
        "trace": [{"A": 2, "B": 6}, {"A": 5, "B": 6}, {"A": 1, "B": 6}],
        "gold_answer": answer(world),
        "sum_input_count": len(sums),
        "sum_answer_range": [min(sums), max(sums)],
        "candidates": CANDIDATES,
        "questions": QUESTIONS,
        "meanings": MEANINGS,
        "missing_card_gold": {q: judgment({"task": "card"}, ("application",), q, CANDIDATES[q]) for q in QUESTIONS},
        "effort": EFFORT,
    }

    release = local("tiny_perceptron/lesson_products/release.json")
    report["release"] = release
    report["products"] = {}
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=Path(__file__).parent) as runtime:
        loaded = {}
        for name, entry in release["products"].items():
            directory = Path(runtime) / name
            directory.mkdir()
            base = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{entry['prefix']}"
            manifest_raw = fetch(base + "/artifact-manifest.json")
            assert sha(manifest_raw) == entry["manifest_sha256"]
            manifest = json.loads(manifest_raw)
            assert manifest == local(f"docs/lesson-products/results/{name}-artifact-manifest.json")
            (directory / "artifact-manifest.json").write_bytes(manifest_raw)
            files = {}
            for filename, expected in manifest["files"].items():
                assert Path(filename).name == filename
                content = fetch(base + "/" + filename)
                assert sha(content) == expected["sha256"] and len(content) == expected["bytes"]
                (directory / filename).write_bytes(content)
                files[filename] = {"url": base + "/" + filename, "sha256": sha(content), "bytes": len(content)}
            receipt = json.loads((directory / "training-receipt.json").read_bytes())
            product = Product(directory, name, "cpu")
            loaded[name] = product
            torch.manual_seed(receipt["seed"])
            initial = (LoopedSolver if name == "b" else CandidateScorer)(product.config)
            recreated = state_digest(initial.state_dict())
            trained = state_digest(product.model.state_dict())
            assert trained != recreated
            assert receipt["random_initialization"] is True and receipt["resume_source"] is None
            assert receipt["weights_changed"] is True and trained == receipt["selected_weights_sha256"]
            original = local(f"docs/lesson-products/results/initial-{name}-execution.json")
            assert original["training"]["initial_weights_sha256"] == receipt["initial_weights_sha256"]
            calls = []
            if name == "b":
                handle = product.model.shared.register_forward_hook(lambda *args: calls.append(1))
                predictions = {str(n): product.solve(example, n) for n in (1, 3, 5)}
                handle.remove()
                assert len(calls) == 9
                assert predictions["3"]["answer"] == predictions["5"]["answer"] == 1
                cli_result = cli(["solve", "--b", str(directory), "--state", example, "--loops", "3"])
                direct = {"depths": predictions, "shared_block_calls": len(calls), "cli": cli_result}
                direct["sum_input"] = {"state": "三數加法;x=0;y=3;z=6", "result": product.solve("三數加法;x=0;y=3;z=6", 3)}
            else:
                state = "卡片進度;申請編號=?"
                predictions = {q: product.judge(state, q) for q in QUESTIONS}
                assert predictions["info"]["selected"] == "no"
                assert predictions["choice"]["selected"] == "card"
                assert predictions["rubric"]["selected"] == "grade0"
                for result in predictions.values():
                    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
                cli_result = cli(["judge", "--c", str(directory), "--state", state, "--question", "info"])
                direct = {"standalone_card": predictions, "cli": cli_result}
            report["products"][name] = {
                "manifest_sha256": sha(manifest_raw),
                "files": files,
                "config": vars(product.config),
                "class": type(product.model).__name__,
                "parameters": sum(p.numel() for p in product.model.parameters()),
                "training_receipt": receipt,
                "recreated_random_initial_sha256": recreated,
                "same_initial_bytes_on_current_cpu": recreated == receipt["initial_weights_sha256"],
                "initialization_scope": "核對訓練程式、原始執行收據、公開收據的隨機起點與選定權重鏈；本機CPU不構成原CUDA環境初值的逐位元重現。",
                "trained_tensor_sha256": trained,
                "inference": direct,
            }
        report["auto"] = auto_solve(loaded["b"], loaded["c"], example)
        assert report["auto"]["status"] == "answered"
        assert report["auto"]["solution"]["answer"] == 1
        report["auto_missing"] = auto_solve(loaded["b"], loaded["c"], "兩盒追蹤;A=?;B=6;事件=交換;查詢=B")
        assert report["auto_missing"]["status"] == "clarification_needed" and report["auto_missing"]["b_loops"] == 0
    report["temporary_weights_removed"] = not Path(runtime).exists()

    final = local("docs/lesson-products/results/final-report-v1.json")
    report["heldout_record"] = {
        "recorded_at": final["recorded_at"],
        "data_manifest_sha256": final["data_manifest_sha256"],
        "settings_freeze_sha256": final["settings_freeze_sha256"],
        "b_five_loops": {k: final["b"]["5"][k] for k in ("boxes", "sum", "repeated_swap")},
        "c": final["c"],
        "gates": final["gates"],
        "predictions": final["predictions"],
        "scope": final["scope"],
    }
    raw_name = "raw/final-report-v1.predictions.jsonl.gz"
    package = local("docs/lesson-products/results/publication-package-v1.json")
    raw_url = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/{raw_name}"
    compressed = fetch(raw_url)
    assert sha(compressed) == package["files"][raw_name]["sha256"]
    raw_scores = gzip.decompress(compressed)
    assert sha(raw_scores) == final["predictions"]["sha256"]
    counts, correct = Counter(), Counter()
    for line in raw_scores.splitlines():
        row = json.loads(line)
        if row["split"] != "test":
            continue
        if row["product"] == "b":
            key = "b/" + row["task"]
            scores = row["scores_by_depth"]["5"]
            selected = max(range(len(scores)), key=scores.__getitem__)
            assert selected == row["answers_by_depth"]["5"]
        elif row["product"] == "c":
            key = "c/" + row["question"]
            candidates, scores = row["candidates"], row["raw_scores"]
            chosen = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
            selected = candidates[chosen]
            assert selected == row["selected"]
        else:
            continue
        counts[key] += 1
        correct[key] += selected == row["gold"]
    recomputed = {key: {"count": counts[key], "correct": correct[key]} for key in counts}
    for task in ("boxes", "sum"):
        assert recomputed["b/" + task] == {k: final["b"]["5"][task][k] for k in ("count", "correct")}
    for question in QUESTIONS:
        assert recomputed["c/" + question] == {k: final["c"][question]["after"][k] for k in ("count", "correct")}
    report["heldout_raw_recomputation"] = {
        "url": raw_url,
        "compressed_sha256": sha(compressed),
        "uncompressed_sha256": sha(raw_scores),
        "uncompressed_bytes": len(raw_scores),
        "counts": recomputed,
        "scope": "在記憶體讀取並核對公開逐題原始分數，再重算五輪與三種判斷的答對數；沒有重訓或重跑整份測試推論。",
    }
    public = local("docs/selftrained/results/v2-final-public-results.json")
    lineage = local("docs/selftrained/results/v2-final-training-source-selection.json")
    metrics = local("docs/selftrained/results/public-raw/moe/test/metrics.json")
    report["moe_preserved"] = {
        "hf_revision": public["hf_revision"],
        "safe_weights": public["architectures"]["moe"]["safe_weights"],
        "selected_source": public["architectures"]["moe"]["selected_source"],
        "origin": {k: lineage["moe_joint"]["origin"][k] for k in ("kind", "seed")},
        "capability_acceptance_passed": public["architectures"]["moe"]["capability_acceptance_passed"],
        "count": metrics["count"],
        "safe_weights_sha256": metrics["safe_weights_sha256"],
        "threshold_ocr_exact": metrics["thresholds"]["ocr_exact"],
        "observed_ocr_exact": metrics["per_task_final_reply"]["ocr"]["exact"],
        "tasks": list(metrics["per_task_final_reply"]),
        "generate_signature": str(inspect.signature(LimitedAssistant.generate)),
    }
    assert not report["moe_preserved"]["capability_acceptance_passed"]
    assert metrics["per_task_final_reply"]["ocr"]["exact"]["rate"] < metrics["thresholds"]["ocr_exact"]
    assistant = LimitedAssistant(SelftrainedConfig(vocab_size=550, width=256, ffn_hidden=512, top_k=2))
    report["moe_preserved"]["model_parts"] = {name: type(module).__name__ for name, module in assistant.named_children()}
    report["moe_preserved"]["architecture"] = assistant.config.architecture
    report["moe_preserved"]["shape_only_forward"] = {}
    assistant.eval()
    payloads = {
        "image": {"kind": "image", "values": torch.zeros(2, 1, 32, 32), "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]])},
        "ocr": {"kind": "ocr", "values": torch.zeros(1, 32, 128)},
        "audio": {"kind": "audio", "values": torch.zeros(64, 40)},
    }
    with torch.inference_mode():
        for kind, token_id, slots in (("image", 7, 2), ("ocr", 8, 32), ("audio", 9, 16)):
            ids = torch.tensor([[1, *([token_id] * slots), 4]])
            value = assistant(ids, modalities=[[payloads[kind]]])
            report["moe_preserved"]["shape_only_forward"][kind] = list(value["logits"].shape)
        report["moe_preserved"]["shape_only_forward"]["text"] = list(assistant(torch.tensor([[1, 11, 4]]))["logits"].shape)
    report["moe_preserved"]["tool_executor"] = execute_tool_call(serialize_tool_call("add", 2, 3))
    help_command = [sys.executable, "scripts/selftrained/chat.py", "--help"]
    help_run = subprocess.run(help_command, cwd=ROOT, check=True, capture_output=True, text=True)
    report["moe_preserved"]["cli_help"] = {"command": help_command, "output": help_run.stdout, "returncode": help_run.returncode}
    report["moe_preserved"]["shape_scope"] = "隨機模型只核對模態形狀與入口；沒有把形狀測試或工具執行器結果視為已訓練MoE的回答品質。"
    for state in (example, "三數加法;x=0;y=3;z=6", "卡片進度;申請編號=?"):
        assert parse_state(state)["task"] in ("boxes", "sum", "card")
    report["rejected_inputs"] = {}
    for state in ("你好", "三數加法;x=7;y=0;z=1"):
        try:
            parse_state(state)
        except ValueError as error:
            report["rejected_inputs"][state] = str(error)
        else:
            raise AssertionError("超出有限語法或數值範圍的輸入沒有被拒絕")
    report["result"] = "公開固定權重指紋、隨機初始化、獨立推論、共用區塊輪數、候選機率、Auto與既存驗收限制均相符。"
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
