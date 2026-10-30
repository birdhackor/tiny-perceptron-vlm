"""凍結公開題目定義與共同祖先；不執行任何 final 模型預測。"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.omni_products.choice import Candidate, Question
from tiny_perceptron.omni_products.data import (
    GENERATOR_REVISION,
    av_pair,
    c_splits,
    final_a_panels,
    fingerprint,
    program_id,
    sound_splits,
    split_a,
    visible_c_facts,
)
from tiny_perceptron.omni_products.media_bindings import bind_media, counterfactual_tool_panel, retention_panel
from tiny_perceptron.omni_products.questions import choice_gold, panel_c
from tiny_perceptron.omni_products.routing import auto_panel, b_families, training_routes
from tiny_perceptron.selftrained.tools import TOOLS_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCES = (
    "tiny_perceptron/omni_products/data.py",
    "tiny_perceptron/omni_products/questions.py",
    "tiny_perceptron/omni_products/routing.py",
    "tiny_perceptron/omni_products/media_bindings.py",
    "tiny_perceptron/omni_products/materialize.py",
    "tiny_perceptron/omni_products/choice.py",
    "tiny_perceptron/selftrained/tools.py",
    "scripts/omni_revision/data_freeze.py",
    "docs/selftrained/v2-manifest.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DIAGNOSTIC_CANDIDATES = tuple(
    Candidate(name, f"Device A: {name}") for name in ("stop", "wait", "record", "play", "save", "defer", "escalate")
)


def candidate_diagnostics(question, world_gold):
    """同家族的2–8候選數診斷；不是新增獨立樣本。"""
    rows = []
    for count in range(1, 8):
        candidates = DIAGNOSTIC_CANDIDATES[:count]
        gold = world_gold if world_gold in {item.id for item in candidates} else "none"
        changed = Question(question.state, question.kind, question.text, candidates, gold).validate()
        rows.append({"total_choices_including_none": count + 1, "gold": changed.gold})
    return rows


def frozen_c_panel(split):
    rows = []
    for index, row in enumerate(panel_c(split)):
        facts = visible_c_facts(row["parent"], index, row["panel"])
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "questions"},
                "questions": {kind: asdict(question) for kind, question in row["questions"].items()},
                "world_supported": True,
                "choice_information_sufficient": row["questions"]["choice_info"].gold == "yes",
                "candidate_count_diagnostics": candidate_diagnostics(row["questions"]["choice"], choice_gold(facts[0])),
            }
        )
    return rows


def definition(data_root):
    legacy = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    bindings = bind_media(data_root, legacy)
    automatic = []
    for row in auto_panel():
        item = {key: value for key, value in row.items() if key not in {"request", "payload"}}
        item["request"] = asdict(row["request"])
        if row["gold_route"] == "C":
            item["payload"] = {"questions": {kind: asdict(question) for kind, question in row["payload"].items()}}
        elif row["gold_route"] == "B":
            item["payload"] = {"numbers": list(row["payload"])}
        elif row["gold_route"] == "A" and "operation" in row["payload"]:
            item["payload"] = {
                "old_A_input": {
                    "task": "tool_call",
                    "messages": [
                        {"role": "system", "content": TOOLS_SCHEMA},
                        {"role": "user", "content": row["request"].text},
                    ],
                    "tools": True,
                }
            }
        elif row["gold_route"] == "A":
            name = row["payload"]["requires_frozen_media_binding"]
            bound = bindings["media"][name]
            item.update(gold=bound["gold"], parent=bound["parent"])
            item["payload"] = (
                {"video_audio_definition": bound} if name == "video-audio" else {"old_A_input": bound["input"]}
            )
            item["media_binding"] = bound
        else:
            item["payload"] = {}
        automatic.append(item)
    routes = training_routes()
    train_states = {fingerprint(row.state) for row in routes}
    auto_states = {fingerprint(row["request"].router_state()) for row in auto_panel()}
    if train_states & auto_states:
        raise ValueError("router train／Auto 可見輸入重疊")
    c = c_splits()
    if set(c["auto"]) & set(c["final"]):
        raise ValueError("C core／Auto 的家族不能重疊")
    av = []
    for index, row in enumerate(final_a_panels()):
        pair = av_pair(row["family"], row["variant"], row["panel"])
        av.append(
            {
                **row,
                "predeclared_chain_pair_index": index % 2,
                "labels": [gold for _, gold in pair],
                "audio_start_ms": [item.audio_start_ms for item, _ in pair],
                "frame_ms": pair[0][0].frame_ms.tolist(),
                "waveform_samples": len(pair[0][0].waveform),
                "sample_rate": 16000,
                "duration_ms": pair[0][0].duration_ms,
            }
        )
    sound = sound_splits()
    if any(
        {program_id(x) for x in sound[a]} & {program_id(x) for x in sound[b]}
        for a, b in (("train", "dev"), ("train", "final"), ("dev", "final"))
    ):
        raise ValueError("聲音完整程式家族重疊")
    retention = retention_panel(data_root, legacy)
    parents = defaultdict(list)
    for item in retention:
        parents[item["family"]].append({"task": item["task"], "record": item["record"]})
    intersections = [
        {"auto_family": item["family"], "parent": item["parent"], "retention": parents[item["parent"]]}
        for item in automatic
        if isinstance(item["parent"], str) and item["parent"] in parents
    ]
    return {
        "schema": "omni-products-data-definition-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "author_task": "/root",
        "generator_revision": GENERATOR_REVISION,
        "files_sha256": {name: sha(ROOT / name) for name in DATA_SOURCES},
        "AV": {
            "splits": split_a(),
            "final": av,
            "primary_families": 200,
            "four_panels": {str(panel): sum(row["panel"] == panel for row in av) for panel in range(4)},
            "remove_image_families": 200,
            "remove_audio_families": 200,
            "tensor_generation_runtime": f"Formal scratch training and core evaluation use CPU, torch{torch.__version__}; this public definition was generated in the same runtime. Earlier CUDA2.8 probe/development failures are separate; no cross-version byte-identical DSP claim.",
        },
        "C": {
            "families": c,
            "core_final_families": 60,
            "source_scope": "One project-authored rule-world generator. source names denote authored surface templates; Chinese changes state rendering while questions remain English. held-out means a wholly held-out wording template, not an independent corpus/source or unseen-pretraining claim.",
            "independent_corpus_source_generalization_measured": False,
            "panels": {split: frozen_c_panel(split) for split in ("dev", "calibration", "policy", "final")},
            "diagnostic_candidate_pool": [asdict(item) for item in DIAGNOSTIC_CANDIDATES],
            "diagnostic_rule": "Use first(total_choices-1) pool entries with the primary state/question text. Gold recomputed from visible facts and current candidate set, not copied from removed-option primary. Same-family secondary views, not new n.",
            "three_types_correlated_within_family": True,
        },
        "sound": {
            split: [[asdict(event) for event in program] for program in values] for split, values in sound.items()
        },
        "Auto": {
            "families": automatic,
            "counts": {name: sum(row["gold_route"] == name for row in automatic) for name in ("A", "B", "C", "none")},
            "train_visible_overlap": 0,
            "C_core_parent_overlap": 0,
            "AV_final_parent_overlap": 1,
            "old_A_and_B_overlap": "Retention of fixed historical finite worlds, not new source/pretraining-unseen claims",
            "old_A_retention_intersections": intersections,
            "B_retention_overlap": {
                "families": 30,
                "population": 84,
                "parents": [item["parent"] for item in automatic if item["gold_route"] == "B"],
            },
        },
        "media_bindings": bindings,
        "old_A_retention": retention,
        "counterfactual_tool_diagnostics": counterfactual_tool_panel(data_root, legacy),
        "retention_lineage": {
            "observations": len(retention),
            "unique_parent_families": len(parents),
            "shared_cross_task_parents": {
                key: value for key, value in parents.items() if len({item["task"] for item in value}) > 1
            },
            "statistical_unit": "parent family; per-task observations can be correlated and are never added as independent n",
        },
        "B_retention": {
            "finite_population": 84,
            "values": sorted(b_families()["retention"]),
            "scope": "Evaluation later enumerates all84 sorted 0..6 triples at frozen old depth; Auto30 is a subset of that population",
        },
        "final_model_predictions_executed": False,
        "teacher_outputs_used_for_training": False,
        "freeze_semantics": "Public immutable held-out definitions, not a claim of cryptographic hiding; no final model evaluation occurred while constructing this file.",
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = definition(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
