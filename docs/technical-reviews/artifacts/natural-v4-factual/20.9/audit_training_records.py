"""Inspect fixed original training histories solely for course held-out exclusion."""
from pathlib import Path
import hashlib,json,platform

ROOT=Path(__file__).resolve().parents[5]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path=ROOT/"docs/natural-assistant/v4/manifest.json"
manifest=json.loads(manifest_path.read_text())
index_rows={r["id"]:r for r in manifest["rows"]}
train_ids={r["id"] for r in manifest["rows"] if r["split"]=="train"}
heldout_ids={r["id"] for r in manifest["rows"] if r["task"]=="scene" and r["split"]!="train"}
records=[]
for run in ["37213067566","37217452291"]:
    source_index=ROOT/f"docs/natural-assistant/evidence/v4-runtime/train-{run}/actual-artifact-index.json"
    ix=json.loads(source_index.read_text())
    raw_root=ROOT/ix.get("actual_local_artifact_directory",ix.get("local_original_artifact_directory"))
    p=raw_root/"review/training.json"
    entries=ix["files"]
    expected=entries["review/training.json"] if isinstance(entries,dict) else next(x for x in entries if x["path"]=="review/training.json")
    assert sha(p)==expected["sha256"] and p.stat().st_size==expected["bytes"]
    raw=json.loads(p.read_text())
    assert raw["manifest_sha256"]==sha(manifest_path)
    history=raw["history"]
    assert len(history)==raw["completed_steps"]==2077
    row_ids=[x for step in history for x in step["row_ids"]]
    assert len(row_ids)==4154
    assert set(row_ids)<=train_ids and not set(row_ids)&heldout_ids
    photo_ids={x for x in row_ids if index_rows[x]["task"]=="scene"}
    records.append({"run_id":run,"raw_path":str(p.relative_to(ROOT)),
                    "raw_sha256":sha(p),"original_index_sha256":sha(source_index),
                    "manifest_sha256":raw["manifest_sha256"],"seed":raw["seed"],
                    "completed_steps":len(history),"sampled_row_occurrences":len(row_ids),
                    "unique_sampled_rows":len(set(row_ids)),"unique_sampled_photo_rows":len(photo_ids),
                    "heldout_photo_row_occurrences":sum(x in heldout_ids for x in row_ids),
                    "all_sampled_rows_are_train":True,"recorded_device":raw["device"],
                    "recorded_elapsed_seconds":raw["elapsed_seconds"]})
print(json.dumps({"python":platform.python_version(),"runs":records,
    "scope":"CPU audit of original fixed GPU-run history records; no training or GPU replication. Course train/heldout exclusion does not rule out upstream pretraining exposure."},indent=2))
