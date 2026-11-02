"""Read the new mainline modules and preserved core logs without model loading."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CORE = Path("/workspace/work/omni-core-completed-40988311100/verified/core")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inspections = {
        "av.py": {"locator": "AVModel.__init__, lines 139–173", "finding": "Own CNN, embeddings, time attention, random readout and answer head; no pretrained-weight loader."},
        "sound.py": {"locator": "SoundCodec.__init__, lines 29–43; ProgramTalker.__init__, lines 131–144", "finding": "Own initialized encoder, codebook and decoder, plus own embeddings, GRU and head; no pretrained speech codec/talker."},
        "choice.py": {"locator": "ByteEncoder/ChoiceModel.__init__, lines 121–159", "finding": "Own byte embeddings, Transformer layers, branch readers and scoring heads; no pretrained Qwen/Kev component."},
        "training_runtime.py": {"locator": "run_updates lines 132–159; train_av 320–324; train_choice 360–362; train_codec 404–407; train_talker 453–457", "finding": "Each fresh-run trainer sets the seed and constructs its own model. Resume restores the own checkpoint and original initialization digest; Talker freezes this pipeline's own trained codec."},
    }
    local = []
    for name, note in inspections.items():
        path = ROOT / "tiny_perceptron/omni_products" / name
        assert "from_pretrained" not in path.read_text()
        local.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), **note})
    rows = []
    for name in ["av-time-rope-s29", "codec", "talker", "c-augmented"]:
        path = CORE / name / "training/training-result.json"
        row = json.loads(path.read_text())
        assert row["complete"] and row["optimizer_updates_observed"]
        assert row["initial_parameter_sha256"] != row["latest_parameter_sha256"]
        matches = {}
        for file in inspections:
            current = ROOT / "tiny_perceptron/omni_products" / file
            matches[file] = digest(current) == row["implementation_sha256"][file]
        rows.append({"name": name, "original_path": str(path), "bytes": path.stat().st_size, "sha256": digest(path), "complete": row["complete"], "updates": row["updates"], "optimizer_updates_observed": row["optimizer_updates_observed"], "seed": row["config"]["seed"], "initial_parameter_sha256": row["initial_parameter_sha256"], "latest_parameter_sha256": row["latest_parameter_sha256"], "current_source_matches_recorded_version": matches, "source_unchanged_during_training": row["source_unchanged_during_training"]})
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_p5_technical_chapter_20", "local_inspections": local, "preserved_core_training_metadata": rows, "scope": "Supplement for intro paragraph 1: new AV, C-v2, codec and talker also start from own constructors. Only saved JSON is read; no trained model is instantiated, no inference/training/evaluation is run."}
    path = OUT / "mainline-origin-supplement-v1.json"
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path), "source_match_results": [row["current_source_matches_recorded_version"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
