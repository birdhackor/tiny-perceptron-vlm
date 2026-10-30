"""Offline source/chronology/fairness checks, no formal/final predictions."""
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "original-source"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "original"
sys.path.insert(0, str(SOURCE))
import torch
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.av import AVConfig, AVModel
from tiny_perceptron.omni_products.data import GENERATOR_REVISION, split_a, c_splits, sound_splits, program_id

frozen = SOURCE / "docs/omni-products/v1"
data, metrics, recipe = [json.loads((frozen / name).read_bytes()) for name in ("data-definition.json", "metrics.json", "formal-recipes.json")]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def source_path(n):
    if (SOURCE / n).exists(): return SOURCE / n
    if (BASE / "supplemental-source" / n).exists(): return BASE / "supplemental-source" / n
    if n == metrics["source_contract"]["path"]: return BASE / "external-contract/CONTRACTS.md"
    raise FileNotFoundError(n)
source_checks = {n: sha(source_path(n)) == value for n, value in {
    **data["files_sha256"], **metrics["measurement_sources_sha256"], **recipe["execution_sources_sha256"],
    **{f"tiny_perceptron/omni_products/{n}": v for n, v in recipe["implementation_sha256"].items()},
    metrics["source_contract"]["path"]: metrics["source_contract"]["sha256"],
}.items()}
assert all(source_checks.values()), source_checks
assert sha(frozen / "data-definition.json") == metrics["data_definition_sha256"] == recipe["data_definition_sha256"]
assert sha(frozen / "metrics.json") == recipe["metrics_sha256"]
times = [datetime.fromisoformat(x["created_at"]) for x in (data, metrics, recipe)]
assert times == sorted(times) and data["generator_revision"] == GENERATOR_REVISION
assert data["final_model_predictions_executed"] is False and data["teacher_outputs_used_for_training"] is False
assert metrics["final_opened"] is False and recipe["final_opened"] is False
assert len(recipe["jobs"]) == len(recipe["job_order"]) == recipe["maximum_total_core_attempts"] == 10
assert len(set(recipe["job_order"])) == 10 and all(v["maximum_attempts"] == 1 for v in recipe["jobs"].values())
assert recipe["remaining_research_jobs"] == 2 and recipe["minimum_scope_total_logical_jobs"] == 12
assert data["AV"]["splits"] == split_a()
assert data["C"]["families"] == {k: [list(v) for v in rows] for k, rows in c_splits().items()}
for split in ("train", "dev"):
    assert data["sound"][split] == [[asdict(event) for event in p] for p in sound_splits()[split]]
assert data["AV"]["four_panels"] == {str(i): 50 for i in range(4)}

av = {k: v for k, v in recipe["jobs"].items() if v["kind"] == "av"}
pair_controls = []
for seed in (11, 29, 47):
    a, b = av[f"av-additive-s{seed}"]["training"], av[f"av-time-rope-s{seed}"]["training"]
    assert {k: v for k, v in a.items() if k != "representation"} == {k: v for k, v in b.items() if k != "representation"}
    initial, frequencies, counts = [], [], []
    for representation in ("additive", "time-rope"):
        trainer.deterministic(seed)
        model = AVModel(AVConfig(representation=representation))
        initial.append(trainer.parameter_sha(model))
        frequencies.append(model.blocks[0][1].frequencies.tolist())
        counts.append(sum(x.numel() for x in model.parameters()))
    assert initial[0] == initial[1] and frequencies[0] == frequencies[1] and len(frequencies[0]) == 8
    pair_controls.append({"seed": seed, "same_initial_parameters": True, "parameter_sha256": initial[0],
                          "trainable_parameters": counts, "per_head_frequencies": frequencies[0],
                          "additive_repeats": 4, "updates": a["updates"], "batch_size": a["batch_size"]})

augmented, compact = trainer.choice_training_core(), trainer.choice_training_core("compact")
assert [asdict(x) for x in compact] == [asdict(x) for i, x in enumerate(augmented) if i % 12 < 4]
c_a = recipe["jobs"]["c-augmented"]["training"]
c_b = recipe["jobs"]["c-compact"]["training"]
assert {k: v for k, v in c_a.items() if k != "data_mode"} == {k: v for k, v in c_b.items() if k != "data_mode"}

def capture_choice(config):
    batches, initial = [], []
    real_collate = trainer.collate_questions
    def collate(rows, device):
        batches.append([asdict(x) for x in rows])
        return real_collate(rows, device)
    def capture(model, make_batch, loss, validate, actual_config, output, resume=None):
        initial.append(trainer.parameter_sha(model))
        generator = torch.Generator().manual_seed(config["seed"] + 100)
        for _ in range(4):
            make_batch(generator)
        return {"capture_only": True, "config": actual_config}
    with patch.object(trainer, "collate_questions", side_effect=collate), patch.object(trainer, "run_updates", side_effect=capture):
        captured = trainer.train_choice(config, BASE / "probes/not-a-formal-job", "cpu")
    return captured, batches, initial[0]

a, a_batches, a_initial = capture_choice(c_a)
b, b_batches, b_initial = capture_choice(c_b)
assert a_initial == b_initial
assert [x[16:] for x in a_batches] == [x[16:] for x in b_batches]
value = {"scope": "Read-only frozen definitions and four sampler captures; no final model predictions or actual formal training.",
         "chronology": {name: x["created_at"] for name, x in zip(("data", "metrics", "recipe"), (data, metrics, recipe))},
         "all_source_hashes_match": all(source_checks.values()), "source_checks": source_checks,
         "job_count": 10, "remaining_research_jobs": 2, "maximum_attempts_each": 1,
         "AV_seed_fairness": pair_controls,
         "C_control": {"augmented_questions": len(augmented), "compact_questions": len(compact),
                       "exact_original_variant0_subset": True, "same_architecture_initial_sha256": a_initial,
                       "same_seed_batch_updates_routes": True, "captured_route_halves_equal": True,
                       "augmented_total_questions": a["config"]["train_questions"], "compact_total_questions": b["config"]["train_questions"]},
         "family_counts": {"AV": {k: len(v) for k, v in data["AV"]["splits"].items()},
                           "C": {k: len(v) for k, v in data["C"]["families"].items()},
                           "sound": {k: len(v) for k, v in data["sound"].items()}},
         "final_predictions": 0}
(BASE / (LABEL + "-protocol-result.json")).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(value,ensure_ascii=False,indent=2))
