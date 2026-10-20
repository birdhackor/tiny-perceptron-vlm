import json,re,sys
from pathlib import Path
root=Path(__file__).parent
P={p["page_id"]:p for p in json.loads(Path("docs/course-review-20261019/freeze-06.json").read_text())["pages"]}
records=[];errors=[]
for line in sys.stdin.read().splitlines():
 if not line.strip():continue
 pid,what,target,needle=(line.split("|")+["","","",""])[:4]
 deps=[]
 if what:
  source=Path(P[target]["snapshot"]).read_text();pars=[p.strip() for p in re.split(r"\n\s*\n",source) if needle in p]
  if not pars:errors.append((pid,target,needle));continue
  para=pars[0];at=para.index(needle);quote=para[max(0,at-40):at+140]
  deps=[{"what":what,"taught_at":("本頁" if target==pid else target)+"：「"+quote+"」","status":"本頁教" if target==pid else "前面教過"}]
 records.append({"page_id":pid,"source_sha256":P[pid]["source_sha256"],"depends_on":deps,"forward_references":[],"issues":[],"inspection":"凍結快照的跨頁依賴段落、入口與程式物件；不冒稱章內逐段首讀。"})
if errors:print(errors);sys.exit(1)
with (root/sys.argv[1]).open("x") as out:
 for r in records:out.write(json.dumps(r,ensure_ascii=False)+"\n")
print("exclusive notes saved:",len(records),records[0]["page_id"],records[-1]["page_id"])
