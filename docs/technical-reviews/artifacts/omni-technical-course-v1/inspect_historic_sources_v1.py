"""Check original producer metadata relocation and pins; never read reviewer reports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[4]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel): return json.loads((ROOT / rel).read_text())

def main():
    archive = read("docs/selftrained/results/training-raw/index.json")
    baseline = read(archive["baseline_index"])
    records = []
    for item in archive["files"]:
        p = ROOT / item["copy"]
        actual = sha(p)
        assert actual == item["sha256"] and p.stat().st_size == item["bytes"]
        row = {"stage_key":item["stage_key"],"original":item["original"],"copy":item["copy"],"sha256":actual}
        if item["copy"].endswith("/raw/train-receipt.json") and item["stage_key"].startswith("moe-"):
            data = read(item["copy"])
            row["origin_kind"] = data["origin"]["kind"]
            row["origin_seed"] = data["origin"]["seed"]
            row["stage"] = data["stage"]
            row["completed_steps"] = data["steps"]
            row["stage_history"] = [{k:v for k,v in event.items() if k in ["stage","checkpoint","checkpoint_sha256","source_stage","source_step","source_checkpoint_sha256","kind"]} for event in data.get("stage_history",[])]
            assert row["origin_kind"] == "all-neural-weights-random"
        records.append(row)
    original_first = next(x for x in baseline["stages"] if x["stage_key"] == "moe-pretrain")
    relocated = ROOT / "docs/selftrained/results/training-raw/moe-pretrain/receipt.json"
    assert sha(relocated) == original_first["receipt_sha256"]
    out = {
        "created_at":datetime.now(timezone.utc).isoformat(),
        "environment":{"python":platform.python_version(),"device":"cpu; metadata and git reads only"},
        "archive_index_path":"docs/selftrained/results/training-raw/index.json",
        "archive_index_sha256":sha(ROOT/"docs/selftrained/results/training-raw/index.json"),
        "baseline_index_sha256":sha(ROOT/archive["baseline_index"]),
        "baseline_index_expected_sha256":archive["baseline_index_sha256"],
        "first_recorded_path":original_first["receipt_path"],
        "first_recorded_path_exists":(ROOT/original_first["receipt_path"]).is_file(),
        "relocated_first_path":str(relocated.relative_to(ROOT)),
        "relocated_first_matches_original_stage_index_sha256":True,
        "archive_files_verified":records,
        "scope":"Original archived producer receipt bytes and origin metadata, not a new training run or reviewer judgment. An obsolete output path does not imply absence of the training.",
        "bc_pinned_source_checks":[],
    }
    for product in ["b","c"]:
        receipt = read(f"docs/lesson-products/results/initial-{product}-execution.json")
        for rel in ["tiny_perceptron/lesson_products/model.py","tiny_perceptron/lesson_products/train.py","tiny_perceptron/lesson_products/training.py"]:
            proc = subprocess.run(["git","show",receipt["code_revision"]+":"+rel],cwd=ROOT,capture_output=True)
            digest = hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None
            out["bc_pinned_source_checks"].append({"product":product,"revision":receipt["code_revision"],"path":rel,"expected_sha256":receipt["source_hashes"][rel],"observed_sha256":digest,"matched":digest==receipt["source_hashes"][rel]})
    results = sorted((ROOT/"docs/course-experiments/results").glob("*.json"))
    out["original_experiment_metadata_inventory"] = {"count":len(results),"files":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"keys":list(json.loads(p.read_text()))} for p in results],"scope":"Producer result structure and existence only; not re-running or auditing each historical experiment."}
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
