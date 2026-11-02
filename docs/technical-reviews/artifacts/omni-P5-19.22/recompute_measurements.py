"""Recompute preserved rows/WAVs only; never instantiate or call a neural model.

Without --acquire, replay the small reviewer projection in a future clone.
With --acquire, inspect the two preserved primary archives at the supplied roots.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import struct
import wave
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": digest(raw)}


def audit_events(analysis):
    actual, expected = analysis["measured"]["events"], analysis["expected"]
    valid = analysis["measured"]["legal"] and len(actual) == len(expected)
    for a, e in zip(actual, expected):
        valid &= abs(a["frequency"] - e["frequency"]) / e["frequency"] <= 0.05
        valid &= abs(a["onset_ms"] - e["onset_ms"]) <= 30
        valid &= abs(a["offset_ms"] - e["offset_ms"]) <= 30
        valid &= abs((a["offset_ms"] - a["onset_ms"]) - (e["offset_ms"] - e["onset_ms"])) <= 30
    for a, b, e, f in zip(actual, actual[1:], expected, expected[1:]):
        valid &= abs((b["onset_ms"] - a["offset_ms"]) - (f["onset_ms"] - e["offset_ms"])) <= 30
    valid &= abs(analysis["measured"]["samples"] / 8 - expected[-1]["offset_ms"]) <= 30
    return bool(valid)


def acquire(core_root, final_root):
    from safetensors.torch import load_file
    from tiny_perceptron.omni_products.data import Event, sound_splits
    from tiny_perceptron.omni_products.inference import SOUND_MAP
    from tiny_perceptron.omni_products.signal_metrics import analyzer_controls, program_success

    core, final = core_root / "verified", final_root / "verified"
    result = {"scope": "Original package inspection, row statistics, existing PCM16 signal analysis, checkpoint tensor metadata only. No neural forward, training, calibration, selection change or final reopening.", "primary_inputs": {}, "archive_checks": {}, "training": {}}
    for name, base, filename in [("core", core_root, "core-checkpoints.zip"), ("final", final_root, "final-results.zip")]:
        path = base / filename
        result["primary_inputs"][name + "_archive"] = {"url": "https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/omni-" + ("core-candidate-cpu-core-40988311100-1" if name == "core" else "final-candidate-cpu-final-41004229262-1") + "/" + filename, **ref(path)}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("package-file-manifest.json"))
            assert len(names) == len(set(names))
            relevant = [key for key in manifest if (key.startswith("core/codec/") or key.startswith("core/talker/") or key.startswith("measurements/sound/") or key in ["frozen/codec-selection.json", "frozen/model-selection.json", "frozen/final-opening.json", "frozen/data-definition.json", "frozen/metrics.json"])]
            for key in relevant:
                raw = archive.read(key)
                assert {"bytes": len(raw), "sha256": digest(raw)} == manifest[key], key
                assert raw == (base / "verified" / key).read_bytes(), key
            result["archive_checks"][name] = {"members": len(names), "reviewed_members_verified_against_manifest_and_extracted_bytes": len(relevant), "manifest_sha256": digest(archive.read("package-file-manifest.json"))}
    for name in ["codec", "talker"]:
        base = core / "core" / name / "training"
        value, log = read(base / "training-result.json"), read(base / "training-log.json")
        job = read(core / "core" / name / "job-result.json")
        assert job["training_result_sha256"] == ref(base / "training-result.json")["sha256"]
        assert value["best"] == max(log, key=lambda row: (row["dev_score"], -row["update"]))
        assert log[-1]["update"] == value["updates"]
        checkpoints = {}
        selected_weights = load_file(str(base / "weights.safetensors"))
        for filename in ["best-continuation.pt", "latest-continuation.pt"]:
            assert ref(base / filename) == value["files"][filename]
            checkpoint = torch.load(base / filename, weights_only=True, map_location="cpu")
            steps = sorted({int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()})
            parameter_digest = hashlib.sha256()
            for parameter_name, tensor in checkpoint["model"].items():
                parameter_digest.update(parameter_name.encode())
                parameter_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            selected_equal = all(torch.equal(tensor, selected_weights[key]) for key, tensor in checkpoint["model"].items())
            checkpoints[filename] = {"step": checkpoint["step"], "optimizer_steps": steps, "parameter_count": sum(t.numel() for t in checkpoint["model"].values()), "metadata_source_binding_matches_training_result": checkpoint["metadata"]["implementation_sha256"] == value["implementation_sha256"], "parameter_sha256": parameter_digest.hexdigest(), "selected_weights_tensor_equal": selected_equal}
            assert steps == [checkpoint["step"]]
            if filename == "best-continuation.pt":
                assert selected_equal
        result["training"][name] = {"training_result_input": ref(base / "training-result.json"), "job_result_input": ref(core / "core" / name / "job-result.json"), "updates": value["updates"], "trainable_parameters": value["trainable_parameters"], "best_update": value["best"]["update"], "log_updates": [row["update"] for row in log], "dev_scores": [row["dev_score"] for row in log], "initial_parameter_sha256": value["initial_parameter_sha256"], "latest_parameter_sha256": value["latest_parameter_sha256"], "runtime": value["runtime"], "weights_input": value["files"]["weights.safetensors"], "checkpoints": checkpoints, "finished_at": job["finished_at"], "started_at": job["started_at"]}
    selection = read(core / "frozen/codec-selection.json")
    result["codec_selection"] = {"input": ref(core / "frozen/codec-selection.json"), "created_at": selection["created_at"], "model": selection["model"], "weights_sha256": selection["weights_sha256"], "talker_declared_selection_sha256": read(core / "core/talker/job-result.json")["codec_selection_sha256"]}
    assert selection["weights_sha256"] == result["training"]["codec"]["weights_input"]["sha256"]
    target_codec_digest = read(core / "core/talker/training/training-result.json")["config"]["codec_parameter_sha256"]
    assert target_codec_digest == result["training"]["codec"]["checkpoints"]["best-continuation.pt"]["parameter_sha256"]
    result["codec_selection"]["talker_targets_codec_parameter_sha256"] = target_codec_digest
    assert result["codec_selection"]["input"]["sha256"] == result["codec_selection"]["talker_declared_selection_sha256"]
    assert result["training"]["codec"]["finished_at"] < selection["created_at"] < result["training"]["talker"]["started_at"]
    raw = read(final / "measurements/sound/results.json")
    result["primary_inputs"]["sound_rows"] = ref(final / "measurements/sound/results.json")
    origin = read(final / "measurements/sound/part-origin.json")
    result["sound_origin"] = {"input": ref(final / "measurements/sound/part-origin.json"), "created_at": origin["created_at"], "model_selection_sha256": origin["model_selection_sha256"], "final_opening_sha256": origin["final_opening_sha256"], "reviewed_current_source_matches_original": {}}
    for key in ["tiny_perceptron/omni_products/sound.py", "tiny_perceptron/omni_products/data.py", "tiny_perceptron/omni_products/signal_metrics.py", "tiny_perceptron/omni_products/inference.py", "tiny_perceptron/omni_products/training_runtime.py", "scripts/omni_revision/core_evaluation.py"]:
        result["sound_origin"]["reviewed_current_source_matches_original"][key] = ref(ROOT / key)["sha256"] == origin["evaluation_sources_sha256"][key]
        assert result["sound_origin"]["reviewed_current_source_matches_original"][key]
    definitions = read(final / "frozen/data-definition.json")
    assert [row["program"] for row in raw["codec_and_gold_program"]["rows"]] == definitions["sound"]["final"]
    assert [[event.__dict__ for event in program] for program in sound_splits()["final"]] == definitions["sound"]["final"]
    expected_chain = [(row["family"], row["panel"], row["predeclared_chain_pair_index"]) for row in definitions["AV"]["final"]]
    assert [(row["family"], row["panel"], row["which"]) for row in raw["actual_A_chain"]["rows"]] == expected_chain
    result["fixed_family_definition_match"] = {"unseen_programs": 200, "predeclared_AV_primary_cases": 200}
    result["snr_values_db"] = [row["active_reference_snr_db"] for row in raw["codec_and_gold_program"]["rows"]]
    result["gold_flags"] = [[row["family_index"], row["generation"]["program_success"], row["generation"]["legal"], row["generation"]["eos"], row["generation"]["capped"], len(row["generation"]["codes"]), min(row["generation"]["codes"]), max(row["generation"]["codes"]), audit_events(row["generation"]["analysis"])] for row in raw["codec_and_gold_program"]["rows"]]
    fields = ["family", "panel", "which", "gold", "answer", "semantic_correct", "mapping_correct", "mapping_attempted", "actual_condition_correct", "gold_semantic_condition_correct", "chain_correct", "legal"]
    result["chain_row_columns"] = fields
    result["chain_flags"] = [[row[key] for key in fields] + [audit_events(row["analysis_against_gold_program"]), audit_events(row["analysis_against_actual_condition"])] for row in raw["actual_A_chain"]["rows"]]
    result["chunk_controls"] = raw["codec_and_gold_program"]["chunk_controls"]
    result["timing"] = raw["actual_playable_timing"]
    result["circle_example"] = raw["actual_A_chain"]["rows"][0]["analysis_against_gold_program"]
    result["analyzer_controls"] = analyzer_controls()
    wave_checks = []
    def wav_check(receipt, program):
        path = final / "measurements/sound" / Path(receipt["wav_path"]).parent.name / Path(receipt["wav_path"]).name
        assert digest(path.read_bytes()) == receipt["wav_sha256"]
        with wave.open(str(path), "rb") as stream:
            assert stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getframerate() == 8000
            count = stream.getnframes()
            samples = torch.tensor(struct.unpack("<" + "h" * count, stream.readframes(count)), dtype=torch.float32) / 32767
        assert count == receipt["samples"]
        measured = program_success(samples, program)
        wave_checks.append([path.relative_to(final).as_posix(), receipt["wav_sha256"], count, measured["success"], measured["measured"]["legal"]])
    for row in raw["codec_and_gold_program"]["rows"]:
        wav_check(row["generation"]["waveform"], tuple(Event(**event) for event in row["program"]))
    for row in raw["actual_A_chain"]["rows"]:
        wav_check(row["waveform"], SOUND_MAP[row["gold"]])
        wav_check(row["gold_semantic_condition"]["waveform"], SOUND_MAP[row["gold"]])
    # Retain only compact independently recomputed WAV inspection counts, not 600 duplicate audio files.
    result["pcm16_reanalysis"] = {"files": len(wave_checks), "all_original_wav_digests_and_headers_verified": True, "program_successes": sum(row[3] for row in wave_checks), "legal": sum(row[4] for row in wave_checks), "ordered_inspections_sha256": digest(json.dumps(wave_checks, separators=(",", ":")).encode()), "scope": "PCM16 exports only; original float32 reconstruction SNR cannot be recreated from these generated WAV exports."}
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--acquire", action="store_true")
parser.add_argument("--core-root", type=Path, default=Path("/workspace/work/omni-core-completed-40988311100"))
parser.add_argument("--final-root", type=Path, default=Path("/workspace/work/omni-final-completed-41004229262"))
args = parser.parse_args()
torch.set_num_threads(1)
projection = HERE / "primary-projection.json"
if args.acquire:
    value = acquire(args.core_root, args.final_root)
    projection.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
else:
    value = read(projection)
snrs = value["snr_values_db"]
assert len(snrs) == 200 and all(math.isfinite(x) for x in snrs)
assert len(value["gold_flags"]) == len(value["chain_flags"]) == 200
mean = sum(snrs) / len(snrs)
assert abs(mean - 22.37515606897692) < 1e-12
assert all(row[1] and row[2] and row[3] and not row[4] and 0 < row[5] <= 200 and 0 <= row[6] <= row[7] < 256 and row[8] for row in value["gold_flags"])
assert all(all(row[5:]) and row[3] == row[4] for row in value["chain_flags"])
assert len({row[0] for row in value["chain_flags"]}) == 200
timing = value["timing"]
statistics_ms = {key: {"median_ms": statistics.median(row[key] for row in timing) * 1000, "min_ms": min(row[key] for row in timing) * 1000, "max_ms": max(row[key] for row in timing) * 1000} for key in ["first_playable_seconds", "whole_code_generation_seconds", "whole_completion_seconds"]}
assert [round(statistics_ms[key]["median_ms"], 2) for key in statistics_ms] == [17.16, 16.93, 24.31]
assert round(statistics_ms["first_playable_seconds"]["min_ms"], 2) == 7.36
assert round(statistics_ms["first_playable_seconds"]["max_ms"], 2) == 24.65
assert len(value["chunk_controls"]) == 10 and all(row["chunks_bit_exact"] and row["future_code_prefix_bit_exact"] for row in value["chunk_controls"])
assert value["pcm16_reanalysis"]["files"] == value["pcm16_reanalysis"]["program_successes"] == value["pcm16_reanalysis"]["legal"] == 600
output = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "1"}, "mode": "primary acquisition plus pure recomputation" if args.acquire else "portable projection replay; no outside inputs needed", "projection": ref(projection), "mean_snr_db_from_200_preserved_family_values": mean, "rounded_snr_db": round(mean, 4), "gold_program_correct_and_legal": [200, 200], "chain_all_six_counts": {key: [sum(row[i] for row in value["chain_flags"]), 200] for i, key in enumerate(value["chain_row_columns"]) if i >= 5}, "timing_10_programs": statistics_ms, "chunk_controls": [10, 10], "pcm16_reanalysis": value["pcm16_reanalysis"], "training": {key: {name: record[name] for name in ["updates", "trainable_parameters", "best_update", "checkpoints"]} for key, record in value["training"].items()}, "passed": True, "not_rerun": "No formal neural inference, original float32 SNR power calculation, optimization, model selection or latency benchmarking rerun. Historical producer provenance is retained in the small projection."}
print(json.dumps(output, ensure_ascii=False, indent=2))
