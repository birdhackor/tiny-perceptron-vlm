import json,copy,hashlib
from pathlib import Path
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records
import ast
source=Path("outputs/phase7-source-cache/p7_technical_b/behavior-ae7bbbf.py").read_text()
node=ast.parse(source);context={"json":json}
for item in node.body:
 if isinstance(item,ast.FunctionDef) and item.name in {"_conversation","_style_record","_style_metrics"}:exec(compile(ast.Module(body=[item],type_ignores=[]),"original-behavior", "exec"),context)
_style_record=context["_style_record"];_style_metrics=context["_style_metrics"]
from tiny_perceptron.tokenization import ByteTokenizer,generation_report
b=Path("docs/technical-reviews/artifacts/p7_technical_b");raw=Path("docs/course-experiments/results/style.json");p=json.loads(raw.read_text());r=p["results"];parts=split_records(arithmetic_records(),seed=42);out={"revision":p["revision"],"raw_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),"base":{},"styles":{}}
for file in ["scripts/course_experiments/behavior.py","scripts/course_experiments/common.py","tiny_perceptron/data.py","tiny_perceptron/tokenization.py"]:
 actual=Path("outputs/phase7-source-cache/p7_technical_b/behavior-ae7bbbf.py") if file=="scripts/course_experiments/behavior.py" else Path(file)
 assert hashlib.sha256(actual.read_bytes()).hexdigest()==p["code_sha256"][file]
for split,rows in parts.items():
 assert len(rows)==r["arithmetic_data"][split]["records"]
 assert hashlib.sha256(("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows)).encode()).hexdigest()==r["arithmetic_data"][split]["sha256"]
base=r["content_evaluation"]["test"];assert sum(s["exact"] for s in base["samples"])==base["matches"]==0 and len(base["samples"])==7
out["base"]={"matches":base["matches"],"records":base["records"],"training_records":r["content_training"]["records"],"steps":r["content_training"]["steps"]}
for style in ["concise","vivid"]:
 run=r["default_style_runs"][style];assert run["training"]["steps"]==450 and run["training"]["records"]==49
 rows=[_style_record(row,style,False) for row in parts["test"]];rep=copy.deepcopy(run["after"]["test"]);recalc=_style_metrics(rep,rows)
 assert recalc["rubric"]==run["after"]["test"]["rubric"]
 assert recalc["rubric"][style]["content_correct"]==0 and recalc["rubric"][style]["style_correct"]==7
 samples=[]
 for row,s in zip(rows,rep["samples"],strict=True):
  assert row["messages"][0]["content"]==s["messages"][0]["content"]
  assert row["messages"][1]["content"]==s["expected"]
  assert ByteTokenizer().decode(s["generated_ids"])==s["generated"]
  assert s["exact"]==(s["generated"]==s["expected"])
  samples.append({"question":s["messages"][0]["content"],"expected":s["expected"],"generated":s["generated"],"ids":s["generated_ids"],"content":s["content_correct"],"style":s["style_correct"],"eos":s["eos"]})
 probe=next(s for s in samples if s["question"]=="2+2=?");assert probe["generated"]==("3" if style=="concise" else "3，像把兩組積木合在一起再數。")
 out["styles"][style]={"rubric":rep["rubric"],"steps":450,"records":49,"samples":samples}
assert not (b/"sources/style-result.json").exists();(b/"sources/style-result.json").write_bytes(raw.read_bytes())
print(json.dumps(out,ensure_ascii=False,indent=2))
