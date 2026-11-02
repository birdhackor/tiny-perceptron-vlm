"""19.15 reviewer execution: frozen inline code, arithmetic, and small released B model.

No training or optimizer update is performed. Large data and third-party sources stay outside Git.
"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import runpy
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.lesson_products import data, model as library
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.training import encoded_data, predict, state_digest


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = {
        "started_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-15-v1/audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1", "dtype": "float32"},
    }
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        ns = runpy.run_path(str(ARTIFACTS / "inline.py"))
    result["inline_stdout"] = stdout.getvalue()
    model, logits, tokens = ns["model"], ns["logits"], ns["tokens"]
    assert sum(p.numel() for p in model.parameters()) == 170003
    assert tuple(tokens.shape) == (1, 24) and tuple(logits.shape) == (1, 5, 19)
    assert ns["ids"][:5] == [1, 42, 103, 131, 130]
    assert [round(x, 3) for x in ns["losses"].tolist()] == [2.08, 2.106, 2.124, 2.137, 2.147]
    assert round(float(ns["losses"].mean()), 3) == 2.119
    source_ast = ast.parse(inspect.getsource(library))
    inline_ast = ast.parse((ARTIFACTS / "inline.py").read_text())
    source_classes = {n.name: ast.dump(n, include_attributes=False) for n in source_ast.body if isinstance(n, ast.ClassDef)}
    inline_classes = {n.name: ast.dump(n, include_attributes=False) for n in inline_ast.body if isinstance(n, ast.ClassDef)}
    equality = {name: source_classes[name] == inline_classes[name] for name in ("Block", "LoopedSolver")}
    assert all(equality.values())
    result["inline_checks"] = {"class_ast_matches_current_library": equality, "parameters_by_tensor": {k: v.numel() for k, v in model.named_parameters()}, "ids": ns["ids"], "ce": ns["losses"].tolist(), "mean_ce": float(ns["losses"].mean()), "inline_max_difference": float((ns["inline_logits"] - logits).abs().max()), "inference_only": True}
    captured = {}
    attention = F.scaled_dot_product_attention

    def inspect_attention(q, k, v, **kwargs):
        captured.update(q_shape=list(q.shape), k_shape=list(k.shape), v_shape=list(v.shape), mask_shape=list(kwargs["attn_mask"].shape), mask_dtype=str(kwargs["attn_mask"].dtype), all_sources_valid=bool(kwargs["attn_mask"].all()), dropout_p=kwargs["dropout_p"], is_causal=kwargs.get("is_causal", False))
        return attention(q, k, v, **kwargs)

    F.scaled_dot_product_attention = inspect_attention
    try:
        with torch.no_grad():
            model(tokens, loops=1)
    finally:
        F.scaled_dot_product_attention = attention
    result["attention_axes"] = captured
    with torch.no_grad():
        padded = F.pad(tokens, (0, 3), value=0)
        padded_logits = model(padded, loops=5, all_depths=True)
    result["pad_mask_probe"] = {"three_trailing_pad_max_delta": float((logits - padded_logits).abs().max()), "tolerance": 2e-6}
    assert torch.allclose(logits, padded_logits, atol=2e-6, rtol=0)
    direct_gradients = []
    for depth in range(1, 6):
        model.zero_grad(set_to_none=True)
        ce = F.cross_entropy(model(tokens, loops=depth), torch.tensor([2]))
        ce.backward()
        direct_gradients.append(float(model.shared.qkv.weight.grad.norm()))
    assert all(x > 0 for x in direct_gradients)
    result["direct_supervision_probe"] = {"shared_qkv_gradient_norm_per_direct_depth_loss": direct_gradients, "parameters_unchanged": True, "scope": "Random-model gradient path only; no update and no evidence of learned quality."}
    initial_a, initial_b = 2, 6
    swaps = []
    for _ in range(2):
        initial_a, initial_b = initial_b, initial_a
        swaps.append(initial_b)
    result["transparent_examples"] = {"B_after_one_and_two_swaps": swaps, "sum_1_2_3": 1 + 2 + 3, "A_after_writes_5_1": data.answer({"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("A", 1)], "query": "A"}), "A_after_write_5": data.answer({"task": "boxes", "a": 2, "b": 6, "events": [("A", 5)], "query": "A"})}

    # Execute exactly the original build_rows prefix which constructs B. Avoid generating C.
    tree = ast.parse(inspect.getsource(data.build_rows))
    fn = tree.body[0]
    stop = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "c_worlds" for t in node.targets))
    fn.name = "build_b_rows_for_audit"
    fn.body = fn.body[:stop] + [ast.Return(value=ast.Name(id="b", ctx=ast.Load()))]
    ast.fix_missing_locations(tree)
    context = vars(data).copy()
    exec(compile(tree, "<actual data.build_rows B prefix>", "exec"), context)
    manifest = json.loads((ROOT / "docs/lesson-products/data-v2-manifest.json").read_bytes())
    options = {key: manifest["config"][key] for key in ("per_length", "stress_per_length", "services")}
    rows = context["build_b_rows_for_audit"](**options)
    digest, bytes_total = hashlib.sha256(), 0
    for row in rows:
        raw = (data.canonical(row) + "\n").encode()
        digest.update(raw)
        bytes_total += len(raw)
    reconstructed = {"sha256": digest.hexdigest(), "bytes": bytes_total, "rows": len(rows)}
    assert reconstructed == manifest["files"]["b.jsonl"]
    result["reconstructed_b"] = {**reconstructed, "counts": dict(Counter(row["split"] + "/" + row["task"] for row in rows)), "scope": "Exact B generator prefix; canonical stream hash equals historical full B file, without saving 28 MB into Git."}

    package = ROOT / "outputs/omni-revision/baseline/b"
    product = Product(package, "b", device="cpu")
    historical = json.loads((ROOT / "docs/lesson-products/results/final-report-v1.json").read_bytes())
    settings_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
    settings = json.loads(settings_path.read_bytes())
    assert historical["settings_freeze_sha256"] == sha(settings_path)
    assert historical["data_manifest_sha256"] == sha(ROOT / "docs/lesson-products/data-v2-manifest.json")
    result["released_package"] = {"path": str(package.relative_to(ROOT)), "file_hashes": {name: sha(package / name) for name in product.manifest["files"]}, "state_digest": state_digest(product.model.state_dict()), "settings_hashes_equal": settings["products"]["b"] == {name: sha(package / name) for name in settings["products"]["b"]}, "config": product.config.__dict__}
    assert result["released_package"]["settings_hashes_equal"]
    summary = {}
    for split, depths in (("test", range(1, 7)), ("entity_shift", (5,)), ("long_shift", (5,))):
        selected = [row for row in rows if row["split"] == split]
        encoded = encoded_data(selected, "b", product.tokenizer, product.config, "cpu")
        values = {}
        for depth in depths:
            scores = predict(product.model, encoded, loops=depth, batch_size=128)
            predictions = scores.argmax(-1).tolist()
            groups = {task: [i for i, row in enumerate(selected) if row["task"] == task] for task in ("boxes", "sum")}
            groups["repeated_swap"] = [i for i, row in enumerate(selected) if row["task"] == "boxes" and row["strata"]["repeated_swap"]]
            report = {}
            for name, indices in groups.items():
                if not indices:
                    continue
                correct = sum(predictions[i] == selected[i]["gold"] for i in indices)
                expected = historical["b"][str(depth)][name] if split == "test" else historical["shifts"][split][name]
                assert (len(indices), correct) == (expected["count"], expected["correct"])
                report[name] = {"count": len(indices), "correct": correct, "percent": 100 * correct / len(indices), "matches_historical_counts": True}
            values[str(depth)] = report
        summary[split] = values
    result["independent_replay"] = summary
    result["exercise_model_predictions"] = {text: product.solve(text, loops=5) for text in ("兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A", "兩盒追蹤;A=2;B=6;事件=A設5;查詢=A")}
    result["historical_training_scope"] = "L4 training/resume claims are reviewed against historical receipts and source, not rerun here. This execution performs CPU inference and gradients without optimizer updates."
    result["completed_at"] = datetime.now(UTC).isoformat()
    (ARTIFACTS / "execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"inline": "passed", "reconstructed_b": reconstructed, "inference_replay": "all cited historical count/correct values matched", "artifact": str((ARTIFACTS / "execution.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
