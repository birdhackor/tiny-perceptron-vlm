"""Independent, read-only 19.14 data/protocol audit; no model evaluation."""

import collections
import hashlib
import io
import json
import platform
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CORE = {"train", "calibration", "policy", "test"}
ARTIFACT = Path(__file__).with_suffix(".json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def main():
    from scripts.reading_time import lesson_slices

    body = lesson_slices((ROOT / "course/chapters/19.md").read_text())["19.14"]
    frozen_path = ROOT / "docs/course-revision-20261016/freeze-10/sources/19.14.md"
    frozen = frozen_path.read_text()
    assert body == frozen
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(frozen.split("```python\n", 1)[1].split("```", 1)[0], {})

    manifest = read("docs/lesson-products/data-v2-manifest.json")
    token_audit = read("docs/lesson-products/data-v2-token-audit.json")
    families = collections.defaultdict(set)
    all_families = collections.defaultdict(set)
    products = {}
    salt = manifest["config"]["split_salt"]
    tokenizer = read("outputs/lesson-products/products/b/tokenizer.json")
    assert tokenizer == read("outputs/lesson-products/products/c/tokenizer.json")
    characters = tokenizer["characters"]
    assert len(characters) == len(set(characters))
    vocabulary = set(characters)
    for product in ("b", "c"):
        path = ROOT / f"outputs/lesson-products/data-v2/{product}.jsonl"
        assert sha(path) == manifest["files"][path.name]["sha256"]
        observations = {}
        all_observations = {}
        counts = collections.Counter()
        groups = collections.defaultdict(set)
        total = 0
        wrong_hash_split = 0
        uncovered_chars = set()
        for line in path.open():
            row = json.loads(line)
            total += 1
            split = row["split"]
            fid = row["family_id"]
            counts[(split, row["task"] if product == "b" else row["question"])] += 1
            groups[split].add(fid)
            all_families[fid].add(split)
            key = row["state"] if product == "b" else (
                row["state"], row["question"], tuple(sorted(row["candidates"]))
            )
            uncovered_chars.update(set(row["state"]) - vocabulary)
            value = all_observations.setdefault(key, [set(), set(), 0])
            value[0].add(split)
            value[1].add(str(row["gold"]))
            value[2] += 1
            if split in CORE:
                families[fid].add(split)
                bucket = int(hashlib.sha256((salt + ":" + fid).encode()).hexdigest()[:12], 16) % 1000
                expected = "train" if bucket < 760 else "calibration" if bucket < 840 else "policy" if bucket < 920 else "test"
                wrong_hash_split += split != expected
                value = observations.setdefault(key, [set(), set(), 0])
                value[0].add(split)
                value[1].add(str(row["gold"]))
                value[2] += 1
        shared = [v for v in observations.values() if len(v[0]) > 1]
        all_shared = [v for v in all_observations.values() if len(v[0]) > 1]
        audit = {
            "rows": total,
            "sha256": sha(path),
            "counts": {"/".join(k): n for k, n in sorted(counts.items())},
            "family_counts": {k: len(v) for k, v in sorted(groups.items())},
            "core_unique_observations": len(observations),
            "core_cross_split_shared_observations": len(shared),
            "core_rows_in_shared_observations": sum(v[2] for v in shared),
            "core_conflicting_labels": sum(len(v[1]) > 1 for v in observations.values()),
            "all_split_cross_split_shared_observations": len(all_shared),
            "all_split_rows_in_shared_observations": sum(v[2] for v in all_shared),
            "core_wrong_hash_split": wrong_hash_split,
            "uncovered_state_characters": sorted(uncovered_chars),
        }
        assert total == manifest["files"][path.name]["rows"]
        assert audit["core_cross_split_shared_observations"] == manifest["audit"][product]["cross_split_shared_observations"]
        assert audit["core_rows_in_shared_observations"] == manifest["audit"][product]["rows_in_shared_observations"]
        assert audit["core_conflicting_labels"] == manifest["audit"][product]["conflicting_labels"]
        assert audit["core_unique_observations"] == token_audit[product]["unique_token_observations"]
        assert wrong_hash_split == 0 and not uncovered_chars
        products[product] = audit
    assert all(len(splits) == 1 for splits in families.values())
    shifted = {
        split: sum(split in splits for splits in all_families.values())
        for split in ("wording_shift", "entity_shift", "long_shift")
    }
    assert all("test" in splits for splits in all_families.values() if "wording_shift" in splits or "entity_shift" in splits)
    assert all(not (splits & CORE) for splits in all_families.values() if "long_shift" in splits)
    freeze_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
    freeze = read(str(freeze_path.relative_to(ROOT)))
    pretest = read("docs/lesson-products/results/pretest-freeze-record-v1.json")
    marker = read("docs/lesson-products/results/settings-freeze-v1.test-started.json")
    final = read("docs/lesson-products/results/final-report-v1.json")
    assert sha(freeze_path) == pretest["settings_freeze_sha256"] == marker["settings_freeze_sha256"] == final["settings_freeze_sha256"]
    assert pretest["recorded_at"] < marker["recorded_at"] < final["recorded_at"]
    assert freeze["data_manifest_sha256"] == sha(ROOT / "docs/lesson-products/data-v2-manifest.json") == final["data_manifest_sha256"]
    assert set(final["shifts"]) == {"wording_shift", "entity_shift", "long_shift", "c_wording"}
    assert all(final["c"][q]["after"]["count"] == products["c"]["counts"]["test/" + q] for q in ("choice", "info", "rubric"))
    checked = [
        "docs/course-revision-20261016/freeze-10/sources/19.14.md",
        "tiny_perceptron/lesson_products/data.py",
        "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/evaluation.py",
        "tiny_perceptron/lesson_products/train.py",
        "docs/lesson-products/data-v2-manifest.json",
        "docs/lesson-products/data-v2-token-audit.json",
        "docs/lesson-products/results/pretest-freeze-record-v1.json",
        "docs/lesson-products/results/settings-freeze-v1.json",
        "docs/lesson-products/results/settings-freeze-v1.test-started.json",
        "docs/lesson-products/results/final-report-v1.json",
        "docs/lesson-products/results/c-calibration.json",
        "docs/lesson-products/results/b-auto-policy.json",
    ]
    result = {
        "scope": "Reviewer-executed read-only raw data/count audit; no training or formal inference. Existing pretest/final primary receipts are inspected, not independently regenerated.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; standard-library data audit"},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "snippet_stdout": stdout.getvalue(),
        "products": products,
        "joint_core_family_overlap": 0,
        "shift_families": shifted,
        "shift_scope": "wording/entity variants reuse only core-test families; long_shift families occur in no core split",
        "token_mapping": {"vocabulary_size": len(characters) + 3, "unique_characters": len(characters), "explanation": "Tokenizer assigns a distinct ID to each character plus reserved IDs; state text has no uncovered characters. Inspected encode preserves fixed question and candidate boundary separators; canonical candidate-order observation equivalence therefore agrees with saved token-audit within core splits."},
        "protocol_receipts": {"pretest_at": pretest["recorded_at"], "test_started_at": marker["recorded_at"], "report_at": final["recorded_at"], "freeze_sha256": sha(freeze_path), "selected_temperatures": freeze["calibration"]["temperatures"], "shift_keys": sorted(final["shifts"]), "receipt_limit": "Chronology/source/data identities support the documented formal run, not proof that no person ever inspected another copy."},
        "source_sha256s": {p: sha(ROOT / p) for p in checked},
    }
    ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
