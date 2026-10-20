import json,re,sys
from pathlib import Path
ROOT=Path(__file__).parent
F=json.loads(Path("docs/course-review-20261019/freeze-06.json").read_text())
P={p["page_id"]:p for p in F["pages"]}
ORDER=json.loads((ROOT/"inventory-01.json").read_text())["reading_order"]
mode=sys.argv[1]
ids=sys.argv[2:]
if mode=="range":
 start,end=ids; ids=ORDER[ORDER.index(start):ORDER.index(end)+1]
for pid in ids:
 p=P[pid];s=Path(p["snapshot"]).read_text()
 print("\n===== "+pid+" "+p["title"]+" =====")
 if mode=="full": print(s);continue
 # Keep definitions and provenance in the source; collapse link destinations only in the inspection view.
 s=re.sub(r"```[^\n]*\n[\s\S]*?```",lambda m:"\n[程式物件 "+" ".join(dict.fromkeys(re.findall(r"(?:from |import |def |class )([A-Za-z_][A-Za-z0-9_.]*)",m.group())))+"]\n",s)
 s=re.sub(r"!?\[([^\]]+)\]\([^)]*\)",r"\1",s)
 s=re.sub(r"<[^>]+>","",s)
 pars=[x.strip().replace("\n"," ") for x in s.split("\n\n") if x.strip() and not x.strip().startswith("|")]
 headings=[x for x in pars if x.startswith("#")]
 print(" / ".join(headings))
 body=[x for x in pars if not x.startswith(("#","[程式物件"))]
 # The inspection view shows the opening and strongest explicit dependency passages.
 if body: print("入口："+body[0])
 dep=[x for x in body[1:] if re.search(r"前[面文章節]|已[經教學看]|回[讀到顧]|沿用|延續|先讀|先[備學]|第\s*\d+\s*章|[0-9ABC]\.[0-9]+|本節.*(?:定義|稱為|叫做)|先.*再",x)]
 budget=int(__import__("os").environ.get("AUDIT_CHARS","650"))
 used=0
 for x in dep:
  if used>=budget: break
  take=x[:max(120,budget-used)]
  print("依賴："+take);used+=len(take)
 symbols=[]
 for m in re.finditer(r"(?:from |import |def |class )([A-Za-z_][A-Za-z0-9_.]*)",Path(p["snapshot"]).read_text()):
  if m.group(1) not in symbols:symbols.append(m.group(1))
 if symbols:print("物件："+" ".join(symbols))
 refs=list(dict.fromkeys(re.findall(r"(?:chapters/)?(?:0?[0-9ABC]+)\.md#([0-9ABC]+\.[0-9]+)",Path(p["snapshot"]).read_text())))
 if refs:print("跨頁連結："+" ".join(refs))
