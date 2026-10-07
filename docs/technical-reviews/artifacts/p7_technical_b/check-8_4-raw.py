import json,ast,copy,hashlib
from pathlib import Path
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records,records_sha256
from tiny_perceptron.tokenization import ByteTokenizer,generation_report
src=Path("docs/technical-reviews/artifacts/p7_technical_b/sources/behavior-ae7bbbf.py").read_text();g={"json":json}
for n in ast.parse(src).body:
 if isinstance(n,ast.FunctionDef) and n.name in {"_conversation","_style_record","_style_metrics"}:exec(compile(ast.Module(body=[n],type_ignores=[]),"originalbehavior","exec"),g)
r=json.load(open("docs/technical-reviews/artifacts/p7_technical_b/sources/style-result.json"))["results"];ap=split_records(arithmetic_records(),seed=42);dates=[]
for day in range(1,25):
 date=f"2026-10-{day:02d}";dates+=[g["_conversation"](f"task=date;date={date};confirm","已確認"+date+"。","date-"+date,style="clarification"),g["_conversation"](f"task=date;id={day};date=?;confirm","請提供日期。","date-"+date,style="clarification")]
dp=split_records(dates,seed=42);parts={s:[g["_style_record"](row,style,True) for row in rows for style in ("concise","vivid","json")]+dp[s] for s,rows in ap.items()};out={"dataset":{},"probes":{}}
for split,rows in parts.items():
 sha=hashlib.sha256("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows).encode()).hexdigest();assert sha==r["data"][split]["sha256"];out["dataset"][split]={"records":len(rows),"arithmetic_perstyle":len(ap[split]),"dates":len(dp[split]),"sha256":sha}
assert records_sha256(parts["train"])==r["training"]["records_sha256"] and r["training"]["steps"]==1000
for split in ["validation","test"]:
 report=copy.deepcopy(r["after"][split]);g["_style_metrics"](report,parts[split]);assert report["rubric"]==r["after"][split]["rubric"];out[split+"_rubric"]=report["rubric"]
for style in ["concise","vivid","json"]:
 report=copy.deepcopy(r["prompt_only_comparison_same_weights"][style]);rows=[g["_style_record"](row,style,True) for row in ap["test"]];g["_style_metrics"](report,rows);m=report["rubric"][style];assert m["records"]==7 and m["style_correct"]==7 and m["content_correct"]==0
 samples=[]
 for row,s in zip(rows,report["samples"],strict=True):
  assert s["messages"][0]["content"]==row["messages"][0]["content"] and s["expected"]==row["messages"][1]["content"]
  assert ByteTokenizer().decode(s["generated_ids"])==s["generated"] and generation_report(ByteTokenizer(),s["generated_ids"])["valid_answer_tokens"] and s["eos"] and s["generated_ids"][-1]==2
  samples.append({"q":s["messages"][0]["content"],"expected":s["expected"],"generated":s["generated"],"content":s["content_correct"],"style":s["style_correct"],"eos":s["eos"]})
 assert abs(report["nll_sum"]/report["effective_tokens"]-report["nll"])<1e-10
 out["probes"][style]={"rubric":m,"eos":sum(s["eos"] for s in report["samples"]),"nll_denominator":report["effective_tokens"],"samples":samples}
out["training"]={"updates":1000,"records":r["training"]["records"],"recordsha":r["training"]["records_sha256"]};out["toy_concise_only"]=[{"提問":f"風格=簡短；{q}","回答":f} for q,f in [("1+1=?","2"),("2+2=?","4")]]
print(json.dumps(out,ensure_ascii=False,indent=2))
