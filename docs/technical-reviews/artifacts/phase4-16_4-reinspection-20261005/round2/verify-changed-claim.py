from pathlib import Path
import hashlib,json,sys,platform
root=Path.cwd();base=root/"docs/technical-reviews/artifacts/phase4-16_4-reinspection-20261005";p=base/"round2"
raw=(root/"docs/course-experiments/results/efficiency.json").read_bytes();assert hashlib.sha256(raw).hexdigest()=="d4bae2fdfab076fe76a65b3555b1a9347a562e1748406c7afc101266e4ee1615"
j=json.loads(raw);fields=set();pitches=set();samples=[]
for split in ("validation","test"):
 for index,s in enumerate(j["results"]["models"]["mha"]["heldout"][split]["samples"]):
  other=j["results"]["models"]["gqa"]["heldout"][split]["samples"][index]
  assert s["messages"]==other["messages"] and s["expected"]==other["expected"]
  prompt=s["messages"][0]["content"];parts=prompt.split(";");attrs=dict(x.split("=",1) for x in parts[:-1]);q=parts[-1];fields.update(attrs);pitches.add(attrs["pitch"])
  correct={"describe":attrs["shape"],"shape?":attrs["shape"],"color?":attrs["color"],"pitch?":attrs["pitch"],"joint?":attrs["shape"]+","+attrs["pitch"]}[q]
  assert correct==s["expected"]
  samples.append({"pointer":f"/results/models/mha/heldout/{split}/samples/{index}","prompt":prompt,"expected":s["expected"],"fields":attrs})
assert fields=={"color","shape","pitch"} and pitches=={"high","low"}
text=(p/"16.4.md").read_text();paragraph=next(x for x in text.splitlines() if x.startswith("這份品質表使用"))
assert "7.13保存的屬性問答基準" in paragraph and "音高標記（high／low）" in paragraph and "位置" not in paragraph
ctx=(base/"inputs/7.13-context.md").read_text();assert "屬性基準" in ctx and "sft/model.pt" in ctx
original_method=(base/"code/original/scripts/course_experiments/architecture.py").read_bytes();assert hashlib.sha256(original_method).hexdigest()=="1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3"
original_method_text=original_method.decode();assert 'tok.encode("Once upon a time, a girl")' in original_method_text
print(json.dumps({"scope":"Only changed task description, own unresolved issue, necessary raw prompt/standard-answer leaves and preserved7.13 baseline context. No training/model eval/fullCPU rerun.","environment":{"python":sys.version,"device":"CPU for JSON/raw field audit","platform":platform.platform()},"raw_result_sha256":hashlib.sha256(raw).hexdigest(),"current_source_sha256":hashlib.sha256((p/"16.4.md").read_bytes()).hexdigest(),"current_paragraph":paragraph,"fields":sorted(fields),"pitch_values":sorted(pitches),"samples":samples,"validation_samples":5,"test_samples":10,"both_model_prompt_sets_identical":True,"7.13_baseline_context_support":"保存屬性基準sft/model.pt，與歷史method讀取sft dataset一致","english_timing_prompt":"Once upon a time, a girl","changed_claim_supported":True,"issue_status":"resolved"},ensure_ascii=False,indent=2))
