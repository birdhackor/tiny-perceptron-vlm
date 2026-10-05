"""Acquire primary official documentation and the original published paper."""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen
from html.parser import HTMLParser

OUT=Path(__file__).resolve().parent / "sources"
OUT.mkdir(exist_ok=True)
sources=[
 ("python-3.13-stdtypes.html","https://docs.python.org/3.13/library/stdtypes.html"),
 ("python-3.13-hashlib.html","https://docs.python.org/3.13/library/hashlib.html"),
 ("python-3.13-lexical.html","https://docs.python.org/3.13/reference/lexical_analysis.html"),
 ("nist-fips180-4.pdf","https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf"),
 ("lee-2022-acl-long577.pdf","https://aclanthology.org/2022.acl-long.577.pdf"),
]
class Plain(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ("script","style"): self.skip+=1
        elif tag in ("p","section","div","dt","dd","h1","h2","h3","li","pre"): self.parts.append("\n")
    def handle_endtag(self,tag):
        if tag in ("script","style"): self.skip-=1
    def handle_data(self,data):
        if not self.skip: self.parts.append(data)
receipts=[]
for name,url in sources:
    with urlopen(Request(url,headers={"User-Agent":"Independent-course-factual-review/1.0"}),timeout=25) as response:
        raw=response.read(6*1024*1024+1)
        if len(raw)>6*1024*1024: raise ValueError("source exceeds bounded acquisition")
        receipts.append({"url":url,"resolved_url":response.url,"status":response.status,
            "accessed_on":"2026-10-05","name":name,"bytes":len(raw),
            "sha256":hashlib.sha256(raw).hexdigest(),"content_type":response.headers.get("Content-Type")})
    (OUT/name).write_bytes(raw)
    if name.endswith("html"):
        parser=Plain(); parser.feed(raw.decode("utf-8")); (OUT/(name+".txt")).write_text("".join(parser.parts))
    else:
        command=["pdftotext","-layout",str(OUT/name),str(OUT/(name+".txt"))]
        completed=subprocess.run(command,capture_output=True,timeout=20)
        receipts[-1]["text_extraction"]={"command_argv":command,"exit_code":completed.returncode,
            "stdout":completed.stdout.decode(),"stderr":completed.stderr.decode()}
        if completed.returncode: raise RuntimeError("PDF extraction failed")
(OUT/"acquisition-receipt.json").write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(receipts,ensure_ascii=False,indent=2))
