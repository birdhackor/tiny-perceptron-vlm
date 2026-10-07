import ast,json,random,hashlib
from pathlib import Path
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records,text_examples,records_sha256
source=Path("docs/technical-reviews/artifacts/p7_technical_b/sources/behavior-ae7bbbf.py").read_text();context={"json":json}
for n in ast.parse(source).body:
 if isinstance(n,ast.FunctionDef) and n.name in {"_conversation","_style_record"}:exec(compile(ast.Module(body=[n],type_ignores=[]),"original-behavior","exec"),context)
p=json.load(open("docs/technical-reviews/artifacts/p7_technical_b/sources/style-result.json"))["results"];rows=split_records(arithmetic_records(),seed=42)["train"];out={}
for style,wanted in [("concise",16591),("vivid",318991)]:
 records=[context["_style_record"](r,style,False) for r in rows];run=p["default_style_runs"][style]["training"];assert records_sha256(records)==run["records_sha256"]
 ex=text_examples(records,mode="sft",max_length=128);rng=random.Random(42);total=0
 for step in range(450):
  batch=rng.choices(ex,k=16);total+=sum(int((y!=-100).sum()) for x,y in batch)
 assert total==run["effective_tokens"]==wanted
 out[style]={"effective_targets":total,"updates":450,"batch_records":16,"train_records":len(records),"records_sha256":run["records_sha256"],"max_input":max(len(x) for x,y in ex)}
print(json.dumps(out,indent=2))
