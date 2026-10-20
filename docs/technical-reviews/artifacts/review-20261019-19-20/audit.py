"""核對固定公開成績、資料指紋、凍結時序與有限輸入推論。"""

import gzip
import hashlib
import html
import io
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.lesson_products import data
from tiny_perceptron.lesson_products.inference import Product, auto_solve
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer

BASE = ROOT / "docs/lesson-products/results"
HF = "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/"
PRETEST = "821a2c717f05ad7b57e08e88cd965e92ff0c1200"
receipts = []


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def curl(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "90", url]
    result = subprocess.run(command, capture_output=True, check=True)
    receipts.append({"command": command, "exit_code": result.returncode, "url": url,
                     "bytes": len(result.stdout), "sha256": sha(result.stdout),
                     "accessed_at": datetime.now(timezone.utc).isoformat()})
    return result.stdout


def gold(world):
    if world["task"] == "sum":
        return sum(world[k] for k in ("x", "y", "z"))
    boxes = {"A": world["a"], "B": world["b"]}
    for box, value in world["events"]:
        if box == "swap":
            boxes = {"A": boxes["B"], "B": boxes["A"]}
        else:
            boxes[box] = value
    return boxes[world["query"]]


def stats(scores, target, temperature=1.0):
    scaled = [v / temperature for v in scores]
    offset = max(scaled)
    exponentials = [math.exp(v - offset) for v in scaled]
    total = sum(exponentials)
    probabilities = [v / total for v in exponentials]
    return [1, int(max(range(len(scores)), key=lambda i: scores[i]) == target),
            offset + math.log(total) - scaled[target],
            sum((v - int(i == target)) ** 2 for i, v in enumerate(probabilities))]


def plus(target, values):
    for i, value in enumerate(values):
        target[i] += value


def metrics(values):
    n, correct, ce, brier = values
    return {"count": n, "correct": correct, "accuracy": correct / n,
            "cross_entropy": ce / n, "brier": brier / n}


def check_metrics(actual, expected):
    assert actual["count"] == expected["count"]
    assert actual["correct"] == expected["correct"]
    for field in ("accuracy", "cross_entropy", "brier"):
        assert math.isclose(actual[field], expected[field], rel_tol=2e-5, abs_tol=3e-6), (field, actual, expected)


def groups(row):
    result = [row["task"]]
    if row["task"] == "boxes":
        result.append("boxes_length_" + str(len(row["world"]["events"])))
        result.extend(k for k in ("answer_from_initial", "last_write_to_other_box", "write_then_swap", "repeated_swap")
                      if row["strata"][k])
    return result


def main():
    torch.set_num_threads(2)
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-20/audit.py",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu",
                              "platform": platform.platform(), "threads": "2"}}
    report = load("docs/lesson-products/results/final-report-v1.json")
    freeze = load("docs/lesson-products/results/settings-freeze-v1.json")
    package = load("docs/lesson-products/results/publication-package-v1.json")
    manifest = load("docs/lesson-products/data-v2-manifest.json")
    paths = ["docs/course-review-20261019/freeze-03/sources/19.20.md",
             "docs/course-review-20261019/freeze-04.json",
             "docs/lesson-products/data-v2-manifest.json",
             "docs/lesson-products/results/final-report-v1.json",
             "docs/lesson-products/results/settings-freeze-v1.json",
             "docs/lesson-products/results/settings-freeze-v1.test-started.json",
             "docs/lesson-products/results/final-report-v1.started.json",
             "docs/lesson-products/results/pretest-freeze-record-v1.json",
             "docs/lesson-products/results/publication-package-v1.json",
             "docs/lesson-products/results/observed-failure-examples-v1.json",
             "docs/lesson-products/results/public-cpu-acceptance-v1.json",
             "docs/lesson-products/results/initial-b-execution.json",
             "docs/lesson-products/results/initial-c-execution.json",
             "docs/selftrained/results/public-raw/moe/test/metrics.json",
             "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/model.py",
             "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/evaluation.py",
             "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/release.json"]
    result["local_files"] = {p: {"sha256": sha((ROOT / p).read_bytes()), "bytes": (ROOT / p).stat().st_size} for p in paths}
    command = ["git", "show", "-s", "--format=%H%n%cI%n%s", PRETEST]
    commit = subprocess.run(command, cwd=ROOT, capture_output=True, check=True, text=True).stdout
    remote_freeze = curl("https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + PRETEST + "/docs/lesson-products/results/settings-freeze-v1.json")
    assert remote_freeze == (BASE / "settings-freeze-v1.json").read_bytes()
    for name in ("pretest-freeze-record-v1.json", "b-auto-policy.json", "c-calibration.json"):
        old = subprocess.run(["git", "show", PRETEST + ":docs/lesson-products/results/" + name], cwd=ROOT,
                             check=True, capture_output=True).stdout
        assert old == (BASE / name).read_bytes()
    start = load("docs/lesson-products/results/final-report-v1.started.json")
    commit_time = datetime.fromisoformat(commit.splitlines()[1])
    assert commit_time < datetime.fromisoformat(start["recorded_at"]) < datetime.fromisoformat(report["recorded_at"])
    result["chronology"] = {"command": command, "git_output": commit,
                            "test_start": start["recorded_at"], "test_finish": report["recorded_at"],
                            "pretest_files_match": True, "protocol": freeze["protocol"],
                            "scope": "保存的凍結、啟動與公開權重相符；無法排除未留下紀錄的離線試跑。"}
    public_report = curl(HF + "report.json")
    assert public_report == (BASE / "final-report-v1.json").read_bytes()
    public_freeze = curl(HF + "settings-freeze.json")
    assert public_freeze == remote_freeze
    compressed = curl(HF + "raw/final-report-v1.predictions.jsonl.gz")
    assert sha(compressed) == package["files"]["raw/final-report-v1.predictions.jsonl.gz"]["sha256"]
    raw = gzip.decompress(compressed)
    assert sha(raw) == report["predictions"]["sha256"] and len(raw) == report["predictions"]["bytes"]
    print("公開逐題分數與未壓縮指紋相符，開始重建資料及重算。", file=sys.stderr, flush=True)
    b, c, config = data.build_rows()
    audit = data.audit_rows(b, c)
    computed = {"config": config, "audit": audit, "files": {}}
    for product, rows in (("b", b), ("c", c)):
        digest = hashlib.sha256()
        size = 0
        for row in rows:
            value = (data.canonical(row) + "\n").encode()
            digest.update(value)
            size += len(value)
        computed["files"][product + ".jsonl"] = {"sha256": digest.hexdigest(), "bytes": size, "rows": len(rows)}
    assert json.loads(json.dumps(computed)) == manifest
    assert all(gold(row["world"]) == row["gold"] for row in b)
    b_by_key = {(r["split"], r["id"]): r for r in b}
    c_by_key = {(r["split"], r["id"]): r for r in c}
    duplicate = defaultdict(set)
    for row in c:
        if row["state"] == "卡片進度;申請編號=?" and row["split"] in data.SPLITS:
            duplicate[row["question"]].add((row["split"], row["gold"]))
    result["data_rebuild"] = {"manifest_identical": True, "files": computed["files"],
                              "b_gold_checked": len(b), "audit": audit,
                              "duplicate_example": {k: sorted(v) for k, v in duplicate.items()}}
    accumulated = defaultdict(lambda: [0, 0, 0.0, 0.0])
    b_test = {}
    row_counts = Counter()
    seen = set()
    reorder_n = reorder_correct = 0
    failures = {}
    for line in raw.splitlines():
        r = json.loads(line)
        product, split = r["product"], r["split"]
        row_counts[(product, split, r.get("mode", ""))] += 1
        key = (product, split, r.get("mode", ""), r["id"])
        if product != "c":
            assert key not in seen
        seen.add(key)
        if product in ("b", "mechanism"):
            row = b_by_key[(split, r["id"])]
            assert row["gold"] == r["gold"] and row["family_id"] == r["family_id"]
            scores = r.get("scores_by_depth", {str(r.get("depth")): r.get("raw_scores")})
            for depth, values in scores.items():
                if product == "b" and split == "test":
                    assert max(range(len(values)), key=lambda i: values[i]) == r["answers_by_depth"][depth]
                for group in groups(row):
                    plus(accumulated[(product, split, r.get("mode", ""), depth, group)], stats(values, r["gold"]))
            if product == "b" and split == "test":
                b_test[r["id"]] = r
        else:
            row = c_by_key[(split, r["id"])]
            assert (row["gold"], row["family_id"], row["question"], row["candidates"]) == (r["gold"], r["family_id"], r["question"], r["candidates"])
            candidates, scores = r["candidates"], r["raw_scores"]
            selected = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
            target = candidates.index(r["gold"])
            if split == "test":
                assert candidates[selected] == r["selected"]
                if r["question"] == "choice":
                    reorder_n += 1
                    reorder_correct += int(r["selected"] == r["reverse_selected"])
            for state, temperature in (("before", 1.0), ("after", freeze["calibration"]["temperatures"][r["question"]])):
                plus(accumulated[(product, split, "", state, r["question"])], stats(scores, target, temperature))
            failures[r["id"]] = r
    checked_groups = 0
    for (product, split, mode, depth, group), values in accumulated.items():
        actual = metrics(values)
        if product == "b":
            expected = report["b"][depth][group] if split == "test" else report["shifts"][split][group]
        elif product == "mechanism":
            expected = report["mechanisms"][mode]["evaluation"][depth][group]
        else:
            expected = report["c"][group][depth] if split == "test" else report["shifts"]["c_wording"][group][depth]
        check_metrics(actual, expected)
        checked_groups += 1
    assert (reorder_n, reorder_correct) == (11797, 11797)
    policy = load("docs/lesson-products/results/b-auto-policy.json")
    autos = {}
    for effort, fixed in (("low", 1), ("medium", 3), ("high", 5), ("auto", None)):
        count = correct = covered = total_loops = 0
        for rid, r in b_test.items():
            row = b_by_key[("test", rid)]
            count += 1
            if r["c_choice"] not in ("boxes", "sum") or r["c_info"] != "yes":
                continue
            covered += 1
            n = len(row["world"].get("events", []))
            depth = fixed or (policy["sum_depth"] if row["task"] == "sum" else
                              1 if n <= policy["low_max_events"] else 3 if n <= policy["medium_max_events"] else 5)
            total_loops += depth
            correct += int(r["answers_by_depth"][str(depth)] == row["gold"])
        autos[effort] = {"count": count, "correct": correct, "covered": covered, "mean_b_loops": total_loops / count}
        expected = report["auto"][effort]
        assert count == expected["count_complete_eligible"] and correct == expected["correct_end_to_end"] and covered == expected["covered"]
        assert math.isclose(total_loops / count, expected["mean_actual_b_loops_all_complete"])
    examples = load("docs/lesson-products/results/observed-failure-examples-v1.json")
    for example in examples["examples"]:
        actual = failures[example["id"]]
        assert actual["raw_scores"] == example["raw_scores"] and actual["gold"] == example["gold"]
        assert c_by_key[("wording_shift", example["id"])]["state"] == example["state"]
    result["raw_recount"] = {"raw_sha256": sha(raw), "raw_bytes": len(raw),
                             "row_counts": {"/".join(k): v for k, v in row_counts.items()},
                             "metric_groups_checked": checked_groups,
                             "metric_tolerance": "整數精確相同；機率指標rel_tol=2e-5、abs_tol=3e-6，容許原報告float32與此處float64差異。",
                             "b5_boxes": metrics(accumulated[("b", "test", "", "5", "boxes")]),
                             "b5_sum": metrics(accumulated[("b", "test", "", "5", "sum")]),
                             "b5_repeated_swap": metrics(accumulated[("b", "test", "", "5", "repeated_swap")]),
                             "c_choice": metrics(accumulated[("c", "test", "", "after", "choice")]),
                             "c_reorder": {"count": reorder_n, "consistent": reorder_correct},
                             "auto": autos, "failure_examples_match": len(examples["examples"]), "gates": report["gates"]}
    del compressed, raw, b, c, b_by_key, c_by_key, failures
    products = {}
    downloaded = {}
    for product, cls in (("b", LoopedSolver), ("c", CandidateScorer)):
        names = ("artifact-manifest.json", "model.pt", "config.json", "tokenizer.json", "training-receipt.json",
                 "auto-policy.json" if product == "b" else "calibration.json")
        payloads = {name: curl(HF + product + "/" + name) for name in names}
        for name, payload in payloads.items():
            assert sha(payload) == package["files"][product + "/" + name]["sha256"]
            if name in freeze["products"][product]:
                assert sha(payload) == freeze["products"][product][name]
        instance = Product.__new__(Product)
        instance.product, instance.device = product, torch.device("cpu")
        instance.config = Config(**json.loads(payloads["config.json"]))
        instance.tokenizer = Tokenizer(json.loads(payloads["tokenizer.json"])["characters"])
        instance.model = cls(instance.config).eval()
        instance.model.load_state_dict(torch.load(io.BytesIO(payloads["model.pt"]), map_location="cpu", weights_only=True))
        instance.auto_policy = json.loads(payloads["auto-policy.json"]) if product == "b" else None
        instance.calibration = json.loads(payloads["calibration.json"]) if product == "c" else None
        products[product] = instance
        training = json.loads(payloads["training-receipt.json"])
        downloaded[product] = {"weights_sha256": sha(payloads["model.pt"]),
                               "training_environment": training.get("environment"),
                               "training_keys": list(training), "selected_step": training.get("selected_step"),
                               "actual_steps": training.get("actual_steps"), "seed": training.get("seed")}
    state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
    smoke = {"solve": products["b"].solve(state, loops=3),
             "solve_repeat": products["b"].solve(state, loops=3),
             "judge": products["c"].judge("卡片進度;申請編號=?", "info"),
             "auto": auto_solve(products["b"], products["c"], state)}
    assert smoke["solve"]["answer"] == smoke["solve_repeat"]["answer"] == 2
    assert smoke["judge"]["selected"] == "no"
    assert smoke["auto"]["solution"]["answer"] == 2
    smoke["wording_failures"] = [products["c"].judge(e["state"], "choice") for e in examples["examples"]]
    result["public_weights"] = {"products": downloaded, "in_memory_loading": True,
                                "scope": "匿名下載固定檔並核對每檔指紋，直接使用公開Product推論方法；未執行CLI下載器或完整重訓。", "smoke": smoke}
    moe = load("docs/selftrained/results/public-raw/moe/test/metrics.json")
    result["moe_counterexample_to_all_gates"] = {"test_count": moe["count"], "ocr_exact": moe["per_task_final_reply"]["ocr"]["exact"],
                                                "required_ocr_exact": moe["thresholds"]["ocr_exact"]}
    assert moe["per_task_final_reply"]["ocr"]["exact"]["rate"] < moe["thresholds"]["ocr_exact"]
    result["primary_sources"] = []
    specifications = [
        ("https://scikit-learn.org/stable/common_pitfalls.html", "scikit-learn 1.9.1，Data leakage、Controlling randomness", ["Test data should never be used to make choices about the model.", "Controlling randomness"]),
        ("https://neurips.cc/public/guides/PaperChecklist", "NeurIPS Paper Checklist，Claims、Limitations、Experimental Result Reproducibility、Experimental Setting/Details、Experiment Statistical Significance", ["The factors of variability that the error bars are capturing should be clearly stated", "exact command and environment"]),
        ("https://proceedings.mlr.press/v70/guo17a.html", "PMLR 70:1321–1330，2017，作者摘要", ["modern neural networks, unlike those from a decade ago, are poorly calibrated"]),
        ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", "PyTorch 2.14，CrossEntropyLoss 類別索引目標公式與reduction", ["This criterion computes the cross entropy loss between input logits and target."]),
    ]
    for url, version, excerpts in specifications:
        payload = curl(url)
        text = html.unescape(re.sub("<[^>]+>", " ", payload.decode()))
        text = re.sub(r"\s+", " ", text)
        for excerpt in excerpts:
            assert excerpt in text, (url, excerpt)
        result["primary_sources"].append({"url": url, "version": version, "sha256": sha(payload),
                                           "accessed_on": datetime.now(timezone.utc).date().isoformat(),
                                           "short_excerpts": excerpts})
    result["network_receipts"] = receipts
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["result"] = "全部可執行斷言通過；本檔只保存小型收據與重算結果，未保存完整外部頁面、公開權重或逐題原始分數。"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
