import ast,hashlib,json,urllib.request
from pathlib import Path
from html.parser import HTMLParser
base=Path(__file__).resolve().parent.parent
provenance=json.loads((base/"download-provenance.json").read_text())
class H(HTMLParser):
 def __init__(self):super().__init__();self.bits=[]
 def handle_data(self,d):self.bits.append(d)
 def handle_starttag(self,t,a):
  if t in ("p","h1","h2","h3","li","pre"):self.bits.append("\n")
results=[]
for record in provenance:
 url=record["url"]
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"independent-factual-recheck-17.12"}),timeout=25) as response:raw=response.read()
 digest=hashlib.sha256(raw).hexdigest()
 assert digest==record["sha256"], (url,"original source version mismatch")
 selected=(base/record["saved_excerpt"]).read_text()
 if record["file"]=="pytorch-torch-docs.py":
  lines=raw.decode().splitlines(True);chunks=[];locators=[]
  for n in ast.parse(raw).body:
   if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args:
    target=ast.unparse(n.value.args[0])
    if target in {"torch.round","torch.clamp","torch.abs","torch.max","torch.sum"}:
     chunks.append(f"\nSOURCE {target}; original lines {n.lineno}-{n.end_lineno}\n"+"".join(lines[n.lineno-1:n.end_lineno]));locators.append([target,n.lineno,n.end_lineno])
  assert "".join(chunks)==selected
 elif record["file"]=="onnx-quantization.html":
  parser=H();parser.feed(raw.decode());text="".join(parser.bits)
  start=text.find("  Dynamic Quantization  ");end=text.find("  Quantization Debugging  ",start)
  assert start>=0 and end>start
  assert selected.split("\n",1)[1]==text[start:end]
  locators=["Dynamic Quantization","Static Quantization"]
 else:
  lines=raw.decode().splitlines(True);locators=record["inspected_original_line_ranges"]
  expected="".join(f"\nOriginal {record['file']} lines {a}-{b}\n"+"".join(lines[a-1:b]) for a,b in locators)
  assert expected==selected
  if record["file"]=="onnx-calibrate.py":
   tree=ast.parse(raw)
   methods={f"{c.name}.{m.name}":[m.lineno,m.end_lineno] for c in tree.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef)}
   assert methods["MinMaxCalibrater.collect_data"]==[415,433]
   assert methods["HistogramCollector.collect_absolute_value"]==[818,874]
   assert methods["HistogramCollector.compute_percentile"]==[954,998]
 results.append({"url":url,"accessed_on":"2026-10-05","full_original_sha256":digest,"saved_excerpt":record["saved_excerpt"],"excerpt_sha256":hashlib.sha256(selected.encode()).hexdigest(),"fresh_download_matches_original":True,"excerpt_matches_fresh_original":True,"locators":locators})
print(json.dumps(results,ensure_ascii=False,indent=2))
