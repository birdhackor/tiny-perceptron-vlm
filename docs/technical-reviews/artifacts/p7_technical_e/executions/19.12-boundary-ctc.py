import json
from pathlib import Path
wanted={"tool_concept":(6,3,8),"tool_missing":(5,8,8),"tool_unavailable":(29,24,48),"tool_unsupported":(4,4,8)}
for i,a in enumerate(("moe","dense")):
 m=json.loads(Path("docs/selftrained/results/public-raw/"+a+"/test/metrics.json").read_text())
 for task,(mo,de,den) in wanted.items():
  v=m["per_task_final_reply"][task]["semantic"];expected=[mo,de][i]
  assert (v["numerator"],v["denominator"])==(expected,den)
  assert abs(v["rate"]-expected/den)<1e-14;print(a,task,str(expected)+"/"+str(den),v["rate"])
 o=m["perception"]["ocr"];assert o=={"cer":0.0,"exact_numerator":324,"denominator":324,"exact_rate":1.0}
 print(a,"ctc_full_string",324,"/324", "final",m["per_task_final_reply"]["ocr"]["exact"]["numerator"],"/324")
 for task in ["voice_qa","voice_topic_continuation"]:
  v=m["per_task_final_reply"][task];assert v["source_groups"]==v["recording_assets"]==30
 print(a,"voice30recordings_not_speakers","boundary_count72")
print("historical_counts_only_no_full_benchmark_rerun")
