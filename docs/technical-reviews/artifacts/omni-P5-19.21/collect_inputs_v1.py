"""Reviewer-only projection of immutable AV archive rows; never runs a neural model."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = {"created_at": datetime.now(UTC).isoformat(), "primary_archives": [], "files": {}}
    archives = {}
    for label, root, filename, asset_id, expected in (
        ("core", args.core, "core-checkpoints.zip", 690797670,
         "107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee"),
        ("final", args.final, "final-results.zip", 690952957,
         "50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3"),
    ):
        raw = (root / filename).read_bytes()
        assert digest(raw) == expected
        archives[label] = (ZipFile(root / filename), root / "verified")
        value["primary_archives"].append({
            "label": label, "filename": filename, "bytes": len(raw), "sha256": digest(raw),
            "provider_asset_id": asset_id,
            "url": f"https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/assets/{asset_id}",
        })

    def read(label, path):
        archive, verified = archives[label]
        raw = archive.read(path)
        assert raw == (verified / path).read_bytes(), (label, path)
        value["files"][label + ":" + path] = {"bytes": len(raw), "sha256": digest(raw), "zip_equals_local": True}
        return json.loads(raw)

    recipe = read("core", "frozen/formal-recipes.json")
    value["job_order"] = [name for name in recipe["job_order"] if name.startswith("av-")]
    selection = read("core", "frozen/model-selection.json")
    opening = read("final", "frozen/final-opening.json")
    origin = read("final", "measurements/av/part-origin.json")
    completion = read("final", "measurements/av/part-completion.json")
    results = read("final", "measurements/av/results.json")
    assert completion["results_sha256"] == value["files"]["final:measurements/av/results.json"]["sha256"]
    assert origin["model_selection_sha256"] == value["files"]["core:frozen/model-selection.json"]["sha256"]
    assert completion["model_selection_sha256"] == origin["model_selection_sha256"]
    value["chronology"] = {
        "selection_created_at": selection["created_at"], "selection_final_opened": selection["final_opened"],
        "opening": opening, "av_origin_created_at": origin["created_at"],
        "av_completed_at": completion["created_at"], "selected_av": selection["selected"]["av"],
    }
    value["training"] = {}
    for job in value["job_order"]:
        result = read("core", f"core/{job}/training/training-result.json")
        logs = read("core", f"core/{job}/training/training-log.json")
        job_origin = read("core", f"core/{job}/job-origin.json")
        weights = archives["core"][0].read(f"core/{job}/training/weights.safetensors")
        assert digest(weights) == result["files"]["weights.safetensors"]["sha256"]
        assert result["files"]["training-log.json"]["sha256"] == value["files"][f"core:core/{job}/training/training-log.json"]["sha256"]
        value["training"][job] = {
            key: result[key] for key in (
                "complete", "updates", "trainable_parameters", "initial_parameter_sha256", "latest_parameter_sha256",
                "optimizer_updates_observed", "best", "config", "runtime", "final_opened", "implementation_sha256",
            )
        }
        value["training"][job]["logs"] = logs
        value["training"][job]["job_origin"] = {
            key: job_origin[key] for key in ("job", "started_at", "final_opened", "execution_host")
        }
        value["training"][job]["selected_weights"] = result["files"]["weights.safetensors"]
    value["final_row_columns"] = [
        "family", "panel", "answer0", "answer1", "gold0", "gold1", "remove_image_answer", "remove_audio_answer",
        "swap_answer0", "swap_answer1", "swap_gold0", "swap_gold1",
    ]
    value["final_rows"] = {}
    value["reported_comparison_for_crosscheck_only"] = results["paired_family_comparison"]
    for job, model in results["models"].items():
        rows = []
        for row in model["records"]:
            assert row["paired_correct"] == (row["answers"] == row["gold"])
            rows.append([row["family"], row["panel"], *row["answers"], *row["gold"],
                         row["remove_image_answer"], row["remove_audio_answer"],
                         *row["waveform_swap_answers"], *row["waveform_swap_gold"]])
        value["final_rows"][job] = rows
    value["scope"] = "Exact small projection of raw AV logs/results, with independent ZIP equality/hash checks. No NN loading, inference, training, calibration, policy fitting, or final reselection."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "sha256": digest(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
