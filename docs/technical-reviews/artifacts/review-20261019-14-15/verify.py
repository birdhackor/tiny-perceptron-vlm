"""核對凍結教材的算例、資料分母、實驗收據與曲線座標。"""

import hashlib
import json
import math
import platform
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.lesson_products import data
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded
from tiny_perceptron.lesson_products.training import encoded_data, sample_batch, state_digest, step


ROOT = Path.cwd()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    torch.set_num_threads(1)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-15/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    }
    freeze = read("docs/course-review-20261019/freeze-04.json")
    page = next(p for p in freeze["pages"] if p["page_id"] == "14.15")
    raw = Path("course/chapters/14.md").read_bytes()
    match = re.search(rb"(?ms)^## 14\.15 .*?(?=^## |\Z)", raw)
    result["source"] = {
        "current_sha256": hashlib.sha256(match[0]).hexdigest(),
        "frozen_sha256": sha(page["snapshot"]),
        "freeze_sha256": sha("docs/course-review-20261019/freeze-04.json"),
    }
    assert result["source"]["current_sha256"] == result["source"]["frozen_sha256"] == page["source_sha256"]

    manifest = read("docs/lesson-products/data-v2-manifest.json")
    options = {key: manifest["config"][key] for key in ("per_length", "stress_per_length", "services")}
    b_rows, c_rows, config = data.build_rows(**options)
    rebuilt_files = {}
    for name, rows in (("b.jsonl", b_rows), ("c.jsonl", c_rows)):
        digest, size = hashlib.sha256(), 0
        for row in rows:
            raw_row = (data.canonical(row) + "\n").encode("utf-8")
            digest.update(raw_row)
            size += len(raw_row)
        rebuilt_files[name] = {"sha256": digest.hexdigest(), "bytes": size, "rows": len(rows)}
    assert config == manifest["config"]
    assert rebuilt_files == manifest["files"]
    result["dataset"] = {
        "options": options,
        "config_equal": config == manifest["config"],
        "files": rebuilt_files,
        "method": "依 write_dataset 相同的 build_rows、canonical(row) 與 UTF-8 換行串流計算雜湊；不保存大型原始資料。",
    }
    del c_rows
    train = [r for r in b_rows if r["split"] == "train" and r["task"] == "boxes"]
    test = [r for r in b_rows if r["split"] == "test" and r["task"] == "boxes"]

    def initial(row):
        return row["world"]["a"], row["world"]["b"]

    def events(row):
        return tuple(tuple(event) for event in row["world"]["events"])

    def complete(row):
        return initial(row), events(row)

    families = defaultdict(list)
    for row in test:
        families[row["family_id"]].append(row)
    assert all(len(rows) == 2 and {r["world"]["query"] for r in rows} == {"A", "B"} for rows in families.values())
    train_worlds, test_worlds = {complete(r) for r in train}, {complete(r) for r in test}
    train_events = {events(r) for r in train}
    length_counts = defaultdict(Counter)
    for rows in families.values():
        row = rows[0]
        length_counts[len(events(row))]["families"] += 1
        length_counts[len(events(row))]["event_sequence_seen_in_train"] += events(row) in train_events
    result["family_overlap"] = {
        "train_box_rows": len(train), "test_box_rows": len(test), "test_families": len(families),
        "two_queries_per_family": True,
        "whole_world_overlap": len(train_worlds & test_worlds),
        "training_initial_pairs": len({initial(r) for r in train}),
        "all_initial_pairs": 8 * 8,
        "test_initial_pairs_seen": len({initial(r) for r in test} & {initial(r) for r in train}),
        "event_sequence_seen_families": sum(counter["event_sequence_seen_in_train"] for counter in length_counts.values()),
        "by_event_length": dict(length_counts),
    }
    assert len(test) == 2956 and len(families) == 1478 and not train_worlds & test_worlds
    assert result["family_overlap"]["event_sequence_seen_families"] == 829
    assert sum(length_counts[n]["families"] for n in (1, 2)) == 508
    assert all(length_counts[n]["families"] == length_counts[n]["event_sequence_seen_in_train"] for n in (1, 2))
    assert length_counts[3] == {"families": 473, "event_sequence_seen_in_train": 299}
    assert length_counts[4] == {"families": 497, "event_sequence_seen_in_train": 22}
    example = {"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0)], "query": "B"}
    result["swap_example"] = {query: data.answer({**example, "query": query}) for query in ("A", "B")}
    assert result["swap_example"] == {"A": 6, "B": 2}
    assert all(data.answer(r["world"]) == r["gold"] for r in test)

    report = read("docs/lesson-products/results/final-report-v1.json")
    settings = read("docs/lesson-products/results/settings-freeze-v1.json")
    product_training = read("docs/lesson-products/results/initial-b-execution.json")["training"]
    result["product_training"] = {key: product_training[key] for key in ("mode", "seed", "depth_weights", "last_step", "selected_step")}
    assert result["product_training"] == {"mode": "multi", "seed": 7421, "depth_weights": [1] * 5, "last_step": 1000, "selected_step": 1000}
    assert report["settings_freeze_sha256"] == sha("docs/lesson-products/results/settings-freeze-v1.json")
    assert report["data_manifest_sha256"] == sha("docs/lesson-products/data-v2-manifest.json")
    result["recorded_experiment"] = {"report_sha256": sha("docs/lesson-products/results/final-report-v1.json"), "modes": {}}
    for mode, name in (("fixed3", "mechanism-fixed3-runner-result.json"), ("odd", "mechanism-odd135-runner-result.json")):
        runner = read(Path("docs/lesson-products/results") / name)
        receipt = runner["execution"]["training"]
        assert receipt["data_manifest_sha256"] == report["data_manifest_sha256"]
        assert receipt["seed"] == 8159 and receipt["batch_size"] == 128 and receipt["last_step"] == 1500
        pinned = settings["mechanisms"][mode]
        for key in ("initial_weights_sha256", "last_step", "seed"):
            assert pinned[key] == receipt[key]
        assert pinned["last.pt"] == runner["execution"]["client_artifacts"]["last.pt"]["sha256"]
        assert pinned["training-receipt.json"] == runner["execution"]["client_artifacts"]["training-receipt.json"]["sha256"]
        source_checks = {path: sha(path) == digest for path, digest in runner["execution"]["source_hashes"].items()}
        assert all(source_checks[path] for path in (
            "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/model.py",
            "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/train.py"))
        values = report["mechanisms"][mode]
        assert values["actual_last_step"] == 1500 and values["same_update_count_last_weights"]
        assert values["terminal_supervision"] == ([3] if mode == "fixed3" else [1, 3, 5])
        result["recorded_experiment"]["modes"][mode] = {
            key: receipt[key] for key in ("seed", "batch_size", "last_step", "selected_step", "work", "initial_weights_sha256", "environment")
        }
        result["recorded_experiment"]["modes"][mode].update(
            source_checks=source_checks, settings_last_checkpoint_matches_runner=True,
            evaluation={depth: values["evaluation"][str(depth)]["boxes"] for depth in range(1, 7)})
        for value in values["evaluation"].values():
            boxes = value["boxes"]
            assert boxes["count"] == 2956 and boxes["families"] == 1478
            assert abs(boxes["accuracy"] - boxes["correct"] / boxes["count"]) < 1e-7
    fixed_receipt, odd_receipt = [result["recorded_experiment"]["modes"][key] for key in ("fixed3", "odd")]
    assert fixed_receipt["initial_weights_sha256"] == odd_receipt["initial_weights_sha256"]
    depths = [(1, 3, 5)[int(hashlib.sha256(f"8159:{number}".encode()).hexdigest()[:8], 16) % 3] for number in range(1, 1501)]
    result["work"] = {"depth_histogram": dict(Counter(depths)), "summed_odd_depth": sum(depths),
                      "mean_odd_depth": sum(depths) / 1500, "fixed3": 1500 * 128 * 3,
                      "odd": 128 * sum(depths), "relative_extra_percent": (128 * sum(depths) / (1500 * 128 * 3) - 1) * 100}
    assert result["work"]["fixed3"] == fixed_receipt["work"]["block_sequence_work"] == 576000
    assert result["work"]["odd"] == odd_receipt["work"]["block_sequence_work"] == 580096

    table = []
    for line in match[0].decode().splitlines():
        cells = [part.strip() for part in line.split("|")[1:-1]]
        if len(cells) == 3 and cells[0].isdigit():
            depth = cells[0]
            for mode, text in zip(("fixed3", "odd"), cells[1:], strict=True):
                ce, accuracy = text.split("／")
                value = report["mechanisms"][mode]["evaluation"][depth]["boxes"]
                assert ce == f"{value['cross_entropy']:.4f}"
                assert accuracy == f"{value['accuracy'] * 100:.2f}%"
                table.append({"mode": mode, "depth": int(depth), "ce": ce, "accuracy": accuracy, "verified": True})
    assert len(table) == 12
    result["table"] = table

    tokenizer = Tokenizer()
    config = Config(len(tokenizer))
    torch.manual_seed(8159)
    model = LoopedSolver(config)
    result["local_initialization"] = {"state_sha256": state_digest(model.state_dict()), "recorded_state_sha256": fixed_receipt["initial_weights_sha256"]}
    tokens = padded([tokenizer.encode(data.render(example))], config.max_length)
    states = []

    def retain(module, args, hidden):
        hidden.retain_grad()
        states.append(hidden)

    handle = model.shared.register_forward_hook(retain)
    logits = model(tokens, loops=5)
    loss = F.cross_entropy(logits, torch.tensor([2]))
    loss.backward()
    handle.remove()
    result["terminal_only_gradients"] = {"loss_shape": list(logits.shape), "loss": float(loss.detach()),
                                         "state_gradient_norms": [float(state.grad.norm()) for state in states]}
    assert len(states) == 5 and all(state.grad is not None and state.grad.norm() > 0 for state in states)
    model.eval()
    with torch.inference_mode():
        all_logits = model(tokens, loops=6, all_depths=True)
        separate = torch.stack([model(tokens, loops=depth) for depth in range(1, 7)], dim=1)
    result["inference_depths"] = {"shape": list(all_logits.shape), "same_full_input_every_depth": True,
                                   "max_logit_difference": float((all_logits - separate).abs().max())}
    assert torch.equal(all_logits, separate)

    subset = train[:8]
    encoded = encoded_data(subset, "b", tokenizer, config)
    first_generator = torch.Generator().manual_seed(8160)
    second_generator = torch.Generator().manual_seed(8160)
    first_indices, second_indices = [], []
    for _ in range(4):
        first_indices.extend(sample_batch(encoded, 8, first_generator).tolist())
        second_indices.extend(sample_batch(encoded, 8, second_generator).tolist())
    assert first_indices == second_indices
    calls = []
    original_ce = F.cross_entropy

    def traced_ce(logits, labels, *args, **kwargs):
        calls.append({"logits_shape": list(logits.shape), "labels_shape": list(labels.shape), "reduction": kwargs.get("reduction", "mean")})
        return original_ce(logits, labels, *args, **kwargs)

    F.cross_entropy = traced_ce
    try:
        small = LoopedSolver(Config(len(tokenizer), width=8, heads=2, ff=16))
        optimizer = torch.optim.SGD(small.parameters(), lr=0.001)
        measurements = {mode: step(small, optimizer, encoded, torch.Generator().manual_seed(8160), 2,
                                    mode, [1] * 5, step_number=1, depth_seed=8159)
                        for mode in ("fixed3", "odd", "multi")}
    finally:
        F.cross_entropy = original_ce
    result["supervision_execution"] = {"calls": calls, "step_results": measurements, "minibatch_stream_equal": first_indices == second_indices,
                                          "scope": "微型隨機模型只驗證損失與流程；不代替已訓練權重的品質重測。"}
    assert calls[0]["logits_shape"] == calls[1]["logits_shape"] == [2, 19]
    assert calls[2]["logits_shape"] == [10, 19] and calls[2]["reduction"] == "none"

    tree = ET.parse("course/figures/looped-mechanism-measured.svg")
    ns = {"svg": "http://www.w3.org/2000/svg"}
    series = []
    for group in tree.findall(".//svg:g", ns):
        if not group.get("id", "").startswith("line2d_"):
            continue
        path = group.find("svg:path", ns)
        if path is None or not path.get("clip-path"):
            continue
        style = path.get("style", "")
        color = next((color for color in ("#175a97", "#b34813") if color in style), None)
        if color:
            positions = [(float(x), float(y)) for x, y in re.findall(r"[ML]\s+([-\d.]+)\s+([-\d.]+)", path.get("d", ""))]
            if len(positions) == 6:
                series.append((color, positions))
    assert len(series) == 4
    figure_checks = []
    for index, (color, positions) in enumerate(series):
        mode = "fixed3" if color == "#175a97" else "odd"
        metric = "cross_entropy" if index < 2 else "accuracy"
        values = [report["mechanisms"][mode]["evaluation"][str(d)]["boxes"][metric] for d in range(1, 7)]
        transformed = [math.log10(value) if metric == "cross_entropy" else value * 100 for value in values]
        scale = (positions[-1][1] - positions[0][1]) / (transformed[-1] - transformed[0])
        offset = positions[0][1] - scale * transformed[0]
        y_error = max(abs(y - (scale * value + offset)) for (_, y), value in zip(positions, transformed, strict=True))
        dx = (positions[-1][0] - positions[0][0]) / 5
        x_error = max(abs(x - positions[0][0] - n * dx) for n, (x, _) in enumerate(positions))
        assert scale < 0 and y_error < 1e-5 and x_error < 1e-5
        figure_checks.append({"mode": mode, "metric": metric, "y_scale": scale, "y_offset": offset,
                              "max_y_residual_pt": y_error, "max_x_residual_pt": x_error})
    render_receipt = read("docs/course-review-20261019/figure-renders-04.json")
    render = next(row for row in render_receipt["records"] if row["page_id"] == "14.15")
    assert render["png"]["sha256"] == sha(render["png"]["path"])
    result["figure"] = {"svg_sha256": sha("course/figures/looped-mechanism-measured.svg"), "series": figure_checks,
                         "render_receipt": render, "visual_inspection": "已用 view_image 檢視收據指定 PNG：上下兩圖的曲線、六個深度、圖例配色與正文一致；CE 為對數、正確率為線性刻度。PNG 僅記取得位置與雜湊，不列 artifact。"}
    result["input_sha256"] = {path: sha(path) for path in (
        "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/train.py",
        "tiny_perceptron/lesson_products/evaluation.py", "scripts/lesson_products/plot_results.py",
        "docs/lesson-products/data-v2-manifest.json", "docs/lesson-products/results/final-report-v1.json",
        "docs/lesson-products/results/settings-freeze-v1.json", "docs/lesson-products/results/mechanism-fixed3-runner-result.json",
        "docs/lesson-products/results/mechanism-odd135-runner-result.json", "docs/lesson-products/jobs/train-b-v1.json",
        "docs/lesson-products/results/initial-b-execution.json")}
    result["limitations"] = "未讀 outputs 中的訓練權重或原始預測，未重新訓練或重算該權重的 CE。品質數字核對已保存的原始實驗報告、分母、收據、凍結與程式語義；本次執行只重建資料、算例、微型損失流程與曲線。"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
