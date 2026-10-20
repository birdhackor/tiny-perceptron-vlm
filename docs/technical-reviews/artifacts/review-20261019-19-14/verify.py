"""重建有限世界資料，在記憶體中獨立查重並核對保存的資料指紋。"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products import data
from tiny_perceptron.lesson_products.evaluation import auto_inputs
from tiny_perceptron.lesson_products.model import Tokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def independent_audit(rows, product):
    observations = defaultdict(lambda: {"splits": set(), "labels": set(), "rows": 0})
    families = defaultdict(set)
    counts = Counter()
    file_hash = hashlib.sha256()
    file_bytes = 0
    lengths = defaultdict(set)
    for row in rows:
        raw = (canonical(row) + "\n").encode()
        file_hash.update(raw)
        file_bytes += len(raw)
        if product == "b" and row["task"] == "boxes":
            lengths[row["split"]].add(len(row["world"]["events"]))
        if row["split"] not in data.SPLITS:
            continue
        key = row["state"] if product == "b" else (
            row["state"], row["question"], tuple(row["candidates"])
        )
        item = observations[key]
        item["splits"].add(row["split"])
        item["labels"].add(row["gold"])
        item["rows"] += 1
        families[row["split"]].add(row["family_id"])
        counts[(row["split"], row["task"] if product == "b" else row["question"])] += 1
    shared = [item for item in observations.values() if len(item["splits"]) > 1]
    return {
        "file": {"rows": len(rows), "bytes": file_bytes, "sha256": file_hash.hexdigest()},
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "family_counts": {split: len(families[split]) for split in data.SPLITS},
        "unique_observations": len(observations),
        "cross_split_shared_observations": len(shared),
        "rows_in_shared_observations": sum(item["rows"] for item in shared),
        "conflicting_labels": sum(len(item["labels"]) > 1 for item in observations.values()),
        "box_lengths_by_split": {key: sorted(value) for key, value in sorted(lengths.items())},
    }


def main():
    b, c, config = data.build_rows()
    manifest = json.loads((ROOT / "docs/lesson-products/data-v2-manifest.json").read_bytes())
    audits = {product: independent_audit(rows, product) for product, rows in (("b", b), ("c", c))}
    comparisons = {"config_matches_manifest": config == manifest["config"]}
    for product, audit in audits.items():
        comparisons[product + "_file_matches_manifest"] = audit["file"] == manifest["files"][product + ".jsonl"]
        for key in ("counts", "family_counts", "unique_observations", "cross_split_shared_observations", "rows_in_shared_observations", "conflicting_labels"):
            comparisons[product + "_" + key + "_matches_manifest"] = audit[key] == manifest["audit"][product][key]
    core_families = defaultdict(set)
    for row in b + c:
        if row["split"] in data.SPLITS:
            core_families[row["family_id"]].add(row["split"])
    family_leaks = sum(len(value) > 1 for value in core_families.values())
    salt = data.choose_salt()
    examples = []
    for a, b_initial in ((2, 6), (2, 3)):
        fid = data.family("boxes", (a, b_initial, (("swap", 0),)))
        bucket = int(sha((salt + ":" + fid).encode())[:12], 16) % 1000
        examples.append({"a": a, "b": b_initial, "family_id": fid, "bucket": bucket, "split": data.split_of(fid, salt)})
    train_sum_labels = sorted({row["gold"] for row in b if row["split"] == "train" and row["task"] == "sum"})
    sum_families = defaultdict(set)
    for row in b:
        if row["task"] == "sum":
            sum_families[tuple(sorted((row["world"]["x"], row["world"]["y"], row["world"]["z"])))].add(row["family_id"])
    card_example = defaultdict(lambda: defaultdict(set))
    for row in c:
        if row["split"] in data.SPLITS and row["state"] == "卡片進度;申請編號=?":
            card_example[row["question"]][row["split"]].add(row["gold"])
    cards = {question: {split: sorted(labels) for split, labels in sorted(splits.items())} for question, splits in sorted(card_example.items())}
    tokenizer = Tokenizer()
    token_sequences = {product: {} for product in ("b", "c")}
    for product, rows in (("b", b), ("c", c)):
        observations = defaultdict(lambda: {"splits": set(), "labels": set()})
        for row in rows:
            if row["split"] not in data.SPLITS:
                continue
            if product == "b":
                key = tuple(tokenizer.encode(row["state"]))
            else:
                key = tuple(tuple(tokenizer.encode(row["state"], row["question"], candidate)) for candidate in sorted(row["candidates"]))
            observations[key]["splits"].add(row["split"])
            observations[key]["labels"].add(row["gold"])
        token_sequences[product] = {
            "unique_token_observations": len(observations),
            "cross_split_shared_token_observations": sum(len(item["splits"]) > 1 for item in observations.values()),
            "conflicting_token_labels": sum(len(item["labels"]) > 1 for item in observations.values()),
        }
    token_record = json.loads((ROOT / "docs/lesson-products/data-v2-token-audit.json").read_bytes())
    comparisons["token_audit_matches"] = all(token_sequences[p] == token_record[p] for p in ("b", "c"))
    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.14.md").read_text()
    fence = source.split("```python\n")[2].split("```")[0]
    excerpt = ast.parse(fence)
    excerpts_match = {}
    for node in excerpt.body:
        if isinstance(node, ast.FunctionDef):
            excerpts_match[node.name] = ast.dump(node) == ast.dump(ast.parse(inspect.getsource(getattr(data, node.name))).body[0])
    report = json.loads((ROOT / "docs/lesson-products/results/final-report-v1.json").read_bytes())
    freeze = json.loads((ROOT / "docs/lesson-products/results/settings-freeze-v1.json").read_bytes())
    record = json.loads((ROOT / "docs/lesson-products/results/pretest-freeze-record-v1.json").read_bytes())
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-14/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "method": "重建全部資料但不儲存語料；用獨立的 exact-input key 查核心四份，再逐列重算 canonical JSONL SHA-256 與 token key。未訓練模型。",
        "source_sha256": sha(source.encode()),
        "audits": audits,
        "comparisons": comparisons,
        "joint_core_families": len(core_families),
        "joint_core_family_leaks": family_leaks,
        "salt": salt,
        "examples": examples,
        "train_sum_labels": train_sum_labels,
        "ordered_sum_family_equal": data.family("sum", (1, 2, 3)) == data.family("sum", (3, 2, 1)),
        "canonical_sum_families_unique": all(len(value) == 1 for value in sum_families.values()),
        "card_example": cards,
        "token_audit": token_sequences,
        "excerpt_functions_match": excerpts_match,
        "auto_input_matches_solver_state": all(row["state"] == routed["state"] for row, routed in zip(b[:8], auto_inputs(b[:8])[::2], strict=True)),
        "split_bucket_sizes": [760, 80, 80, 80],
        "split_bucket_percentages": [76, 8, 8, 8],
        "hash_prefix_bits": 48,
        "hash_modulo_counts": {"buckets_0_through_655": (2 ** 48) // 1000 + 1, "buckets_656_through_999": (2 ** 48) // 1000},
        "stored_evaluation": {"shifts": report["shifts"], "scope": report["scope"], "settings_freeze_sha256_matches": report["settings_freeze_sha256"] == sha((ROOT / "docs/lesson-products/results/settings-freeze-v1.json").read_bytes()), "pretest_record_matches": record["settings_freeze_sha256"] == report["settings_freeze_sha256"], "freeze_predates_test_record": freeze["recorded_at"] < report["recorded_at"], "test_outcomes_observed_at_freeze": freeze["test_outcomes_observed"]},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(comparisons.values()) or family_leaks or not all(excerpts_match.values()):
        raise SystemExit("重建結果與保存紀錄不同，需進一步核對")


if __name__ == "__main__":
    main()
