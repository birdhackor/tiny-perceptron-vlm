"""Inspect only named raw configuration/provenance pointers, not author notes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
inspection = []
files = ["docs/course-experiments/results/simple_models.json", "docs/course-experiments/results/text_foundation.json", "docs/course-experiments/plan.json",
         "docs/course-experiments/public-models.json"]
for name in files:
    raw = (ROOT / name).read_bytes()
    value = json.loads(raw)
    entry = {"source": name, "sha256": hashlib.sha256(raw).hexdigest(),
             "topkeys": {k: type(v).__name__ for k, v in value.items()}, "pointers": {}}
    if name.endswith("results/simple_models.json"):
        pointers = ["/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version",
                    "/step_scale", "/results/data", "/results/sections"]
        pointers += ["/results/runs/" + model + "/" + key
                     for model in ["bigram", "mlp1", "mlp3", "mlp5"]
                     for key in ["parameters", "context", "steps", "checkpoint", "checkpoint_loader"]]
        for pointer in pointers:
            selected = value
            for key in pointer.split("/")[1:]:
                selected = selected[key]
            entry["pointers"][pointer] = selected
        assert list(value["results"]["runs"]) == ["bigram", "mlp1", "mlp3", "mlp5"]
        assert {k: v["records"] for k, v in value["results"]["data"].items() if isinstance(v, dict)} == {
            "train": 9, "validation": 1, "test": 2}
        assert {k: v["parameters"] for k, v in value["results"]["runs"].items()} == {
            "bigram": 289, "mlp1": 833, "mlp3": 1345, "mlp5": 1857}
        assert {k: v["context"] for k, v in value["results"]["runs"].items()} == {
            "bigram": 1, "mlp1": 1, "mlp3": 3, "mlp5": 5}
        assert all(v["steps"] == 200 for v in value["results"]["runs"].values())
    elif name.endswith("results/text_foundation.json"):
        pointers = ["/experiment_id", "/revision", "/seed", "/step_scale", "/hf/repo", "/hf/revision",
                    "/hf/prefix", "/hf/result_revision"]
        pointers += ["/results/training/" + key for key in ["steps", "effective_tokens", "checkpoint", "parameters",
                                                          "trainable_parameters", "records", "records_sha256"]]
        for pointer in pointers:
            selected = value
            for key in pointer.split("/")[1:]:
                selected = selected[key]
            entry["pointers"][pointer] = selected
        index = next(i for i, v in enumerate(value["artifacts"]) if v["path"] == "model.pt")
        entry["pointers"][f"/artifacts/{index}"] = value["artifacts"][index]
        assert value["results"]["training"]["steps"] == 600
        assert value["results"]["training"]["checkpoint"] == "model.pt"
        assert value["results"]["training"]["records"] == 9
    elif name.endswith("plan.json"):
        index = next(i for i, v in enumerate(value["sequence"]) if v["id"] == "simple_models")
        selected = value["sequence"][index]
        for key in ["id", "module", "function", "device", "dependencies", "assets", "lessons"]:
            entry["pointers"][f"/sequence/{index}/{key}"] = selected[key]
        assert selected["module"] == "text" and selected["function"] == "run_simple_models"
        assert selected["assets"] == [] and selected["dependencies"] == []
    else:
        for index, selected in enumerate(value["models"]):
            if selected["id"] in ("simple_models", "text_foundation"):
                entry["pointers"][f"/models/{index}"] = selected
    inspection.append(entry)

name = "sources/hf-text-export-manifest.json"
raw = (OUT / name).read_bytes()
value = json.loads(raw)
entry = {"source": name, "sha256": hashlib.sha256(raw).hexdigest(),
         "topkeys": {k: type(v).__name__ for k, v in value.items()}, "pointers": {}}
for index, selected in enumerate(value["files"]):
    for key in ["output", "sha256", "bytes", "source_sha256"]:
        entry["pointers"][f"/files/{index}/{key}"] = selected[key]
for key in ["revision", "experiment_id", "batch_id", "approval_git_revision", "approval_sha256"]:
    entry["pointers"]["/provenance/" + key] = value["provenance"][key]
inspection.append(entry)
manifest = json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())
spec = next(v for v in manifest["models"] if v["id"] == "text_foundation")
file = next(v for v in spec["files"] if v["output"] == "export-manifest.json")
assert hashlib.sha256(raw).hexdigest() == file["sha256"] and len(raw) == file["bytes"]
trained = json.loads((ROOT / "docs/course-experiments/results/text_foundation.json").read_text())
source_model = next(v for v in trained["artifacts"] if v["path"] == "model.pt")
export_model = next(v for v in value["files"] if v["output"] == "model.pt")
public_model = next(v for v in spec["files"] if v["output"] == "model.pt")
assert source_model["sha256"] == export_model["source_sha256"]
assert export_model["sha256"] == public_model["sha256"] and export_model["bytes"] == public_model["bytes"]
(OUT / "pointer-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "all_named_pointer_assertions_passed", "raw_report_runs": ["bigram", "mlp1", "mlp3", "mlp5"],
                  "remote_export_manifest_matches_pinned_manifest": True,
                  "trained_checkpoint_to_public_export_hash_chain": True,
                  "excluded": "No raw report outcome commentary, notes, review, scope correction, or historical technical/reader verdicts were read."}))
