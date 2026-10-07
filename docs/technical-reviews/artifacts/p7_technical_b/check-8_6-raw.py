import json
from pathlib import Path
from tiny_perceptron.tokenization import ByteTokenizer,generation_report
from scripts.course_experiments.common import split_records
rows=[]
for day in range(1,25):
 date=f"2026-10-{day:02d}"
 rows.extend([{"family":"date-"+date,"messages":[{"role":"user","content":f"task=date;date={date};confirm"},{"role":"assistant","content":"已確認"+date+"。"}],"style":"clarification"},{"family":"date-"+date,"messages":[{"role":"user","content":f"task=date;id={day};date=?;confirm"},{"role":"assistant","content":"請提供日期。"}],"style":"clarification"}])
p=split_records(rows,seed=42);out={"splits":{s:{"families":len({r["family"] for r in rs}),"records":len(rs)} for s,rs in p.items()},"samples":[]}
for a in p:
 for b in p:
  if a!=b:assert {r["family"] for r in p[a]}.isdisjoint({r["family"] for r in p[b]})
report=json.load(open("docs/technical-reviews/artifacts/p7_technical_b/sources/style-result.json"))["results"]["after"]["test"];samples=[s for s in report["samples"] if s.get("kind")=="clarification"];assert len(samples)==6
counts={"missing_correct":0,"given_correct":0}
for row,s in zip(p["test"],samples,strict=True):
 assert row["messages"][0]["content"]==s["messages"][0]["content"] and row["messages"][1]["content"]==s["expected"]
 assert ByteTokenizer().decode(s["generated_ids"])==s["generated"] and generation_report(ByteTokenizer(),s["generated_ids"])["valid_answer_tokens"]
 correct=s["generated"]==s["expected"];assert correct==s["exact"]
 missing="date=?" in s["messages"][0]["content"];counts["missing_correct" if missing else "given_correct"]+=int(correct)
 out["samples"].append({"q":s["messages"][0]["content"],"expected":s["expected"],"generated":s["generated"],"ids":s["generated_ids"],"correct":correct,"missing":missing,"family":row["family"]})
assert counts=={"missing_correct":3,"given_correct":0};out["counts"]=counts
out["variation_date_time"]=[{"date":date,"time":time,"ask_missing":[k for k,v in {"date":date,"time":time}.items() if v is None]} for date,time in [("2026-10-03","10:00"),(None,"10:00"),("2026-10-03",None),(None,None)]]
print(json.dumps(out,ensure_ascii=False,indent=2))
