import json,hashlib,datetime,collections
from pathlib import Path
from tiny_perceptron.selftrained.dataset import read_records
root=Path("docs/selftrained/results/public-raw")
data=Path("outputs/p7-technical-e-cache/data")
manifest=json.loads(Path("docs/selftrained/v2-manifest.json").read_text())
records=read_records([str(data/x["path"]) for x in manifest["records"]])
test=[r for r in records if r["split"]=="test"]
assert len(test)==3734 and len({r["id"] for r in test})==3734
ids_sha=hashlib.sha256("".join(r["id"]+"\n" for r in test).encode()).hexdigest()
wanted={"text":("semantic",464,429,560),"ocr":("exact",252,281,324),"vision_clothing":("exact",357,356,360),"vision_relation":("exact",1427,1426,1440),"voice_qa":("semantic",42,37,90),"voice_topic_continuation":("semantic",26,24,60),"tool_call":("tool_roundtrip",0,7,276),"tool_reply":("semantic",551,468,552)}
for i,a in enumerate(("moe","dense")):
 base=root/a;m=json.loads((base/"test/metrics.json").read_text());f=json.loads((base/"freeze/frozen.json").read_text());j=json.loads((base/"test/evaluation-receipt.json").read_text());receipt=json.loads((base/"test/receipt.json").read_text())
 assert m["count"]==m["expected_count"]==j["completed_count"]==j["expected_count"]==3734
 assert m["evaluation_complete"] and not m["limited_smoke"] and not m["interrupted"] and not m["teacher_forcing_used_for_generation"]
 assert j["resumed_from"] is None and j["recovered_complete_tail_count"]==0 and j["status"]=="complete"
 assert j["completed_ids_sha256"]==ids_sha
 assert sum(v["count"] for v in m["per_task_final_reply"].values())==3734
 assert receipt["status"]=="completed" and receipt["returncode"]==0
 for name in ["metrics.json","evaluation-receipt.json"]:
  d=next(x for x in receipt["files"] if x["path"]==name)
  p=base/"test"/name;assert p.stat().st_size==d["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==d["sha256"]
 assert hashlib.sha256((base/"freeze/frozen.json").read_bytes()).hexdigest()==j["protocol_sha256"]==receipt["job"]["protocol"]["sha256"]
 assert hashlib.sha256((base/"freeze/metrics.json").read_bytes()).hexdigest()==f["validation_metrics_sha256"]
 assert f["created_at"]<receipt["started_at"]<receipt["finished_at"]
 assert f["test_once"] and f["selection"]=="validation only"
 public=json.loads(Path("docs/technical-reviews/artifacts/p7_technical_e/sources/hf-"+a.replace("moe","moe-joint").replace("dense","dense-joint")+"-manifest.json").read_text())
 assert public["files"]["model.safetensors"]==f["safe_weights_sha256"]==m["safe_weights_sha256"]==j["checkpoint_sha256"]
 assert public["selected_checkpoint_sha256"]==f["selected_checkpoint_sha256"]==m["selected_checkpoint_sha256"]
 assert hashlib.sha256(Path("docs/technical-reviews/artifacts/p7_technical_e/sources/hf-"+a+"-joint-manifest.json").read_bytes()).hexdigest()==f["inference_manifest_sha256"]
 print(a,"count3734 complete one_attempt same_test_idsSHA",ids_sha,"frozen_before_test","validation_and_public_identity_bound")
 for task,(key,mnum,dnum,den) in wanted.items():
  v=m["per_task_final_reply"][task][key];expected=[mnum,dnum][i]
  assert (v["numerator"],v["denominator"])==(expected,den) and abs(v["rate"]-expected/den)<1e-14
  assert sum(r["task"]==task for r in test)==den
  print(task,key,str(expected)+"/"+str(den),"rate",v["rate"])
 vc=m["voice_topic_continuation"];assert vc["end_to_end_correct"]==[26,24][i] and vc["recording_assets"]==vc["source_groups"]==30
 print("voice_end_to_end",vc["end_to_end_correct"],"/60","recordings30")
 print("extra72_records",{k:v["count"] for k,v in m["per_task_final_reply"].items() if k not in wanted})
print("historical_saved_measurements_recalculated_not_regenerated_full_raw_output_not_inspected")
