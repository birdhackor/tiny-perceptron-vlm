"""Bind the 126 fixed photograph answers to current inputs and score decisions."""
from pathlib import Path
import hashlib,json,platform

ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/"docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path=ROOT/"docs/natural-assistant/v4/manifest.json"
manifest=json.loads(manifest_path.read_text())
photos={r["id"]:r for r in manifest["rows"] if r["task"]=="scene" and r["split"]=="test"}
scores=json.loads((BASE/"scored/scores.json").read_text())
binding=scores["artifact_binding"]
assert binding["manifest_sha256"]==sha(manifest_path)
assert binding["test_result_sha256"]==sha(BASE/"result.json")
assert binding["generation_file"]["sha256"]==sha(BASE/"generations-base.json")
result=json.loads((BASE/"result.json").read_text())
assert result["manifest_sha256"]==sha(manifest_path)
assert result["status"]=="completed" and result["split"]=="test" and result["selected_only"]
generations=json.loads((BASE/"generations-base.json").read_text())
generations={r["id"]:r for r in generations if r["task"]=="scene"}
assert set(generations)==set(photos) and len(generations)==126
decisions={x["case_id"]:x for x in scores["case_decisions"]}
completed=0
for key,row in photos.items():
    g=generations[key]
    assert all(g[k]==row[k] for k in ["split","task","family","user","image"])
    assert g["reference_answer"]==row["answer"] and g["variant"]=="base"
    ids=g["generated_token_ids"]
    complete=(bool(ids) and ids[-1] in g["eos_token_ids"] and len(ids)==g["generated_tokens"]
              and g["ended_with_eos"] is True and g["truncated"] is False
              and g["stop_reason"]=="eos" and g["completion_unknown"] is False)
    assert complete==decisions["scene:"+key]["decoder_complete"]
    completed+=complete
print(json.dumps({"python":platform.python_version(),"photograph_answer_records":126,
    "photographs":len({r["image"] for r in photos.values()}),
    "current_manifest_sha256":sha(manifest_path),"original_result_sha256":sha(BASE/"result.json"),
    "original_generations_sha256":sha(BASE/"generations-base.json"),
    "selected_variant":binding["selected_variant"],"recorded_seed":result["seed"],
    "original_runtime_device":result["device"],"original_runtime_torch":result["versions"]["torch"],
    "photo_records_with_matching_frozen_prompt_image_family_reference":len(generations),
    "photo_records_with_verified_raw_eos":completed,
    "scope":"CPU audit of existing answer records and bindings. Does not regenerate or independently re-grade the 42 held-out photos; prior semantic case decisions are directly recomputed in verify_probe.py."},indent=2))
