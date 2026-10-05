"""Fetch bounded original papers and official documentation only."""
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

OUT=Path(__file__).resolve().parent/"sources"
OUT.mkdir(exist_ok=True)
sources=[
 ("cawley-talbot-2010.pdf","https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf"),
 ("cawley-talbot-2010.html","https://www.jmlr.org/papers/v11/cawley10a.html"),
 ("sklearn-1.7.2-cross_validation.rst","https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst"),
 ("sklearn-1.7.2-nested_cross_validation_iris.py","https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/examples/model_selection/plot_nested_cross_validation_iris.py"),
 ("tinystories-f54c09f-README.md","https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md"),
 ("chinese-poetry-b8594f8-README.md","https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/README.md"),
 ("cpython-v3.13.5-controlflow.rst","https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/controlflow.rst"),
 ("cpython-v3.13.5-datastructures.rst","https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/datastructures.rst"),
]
receipts=[]
for filename,url in sources:
    req=urllib.request.Request(url,headers={"User-Agent":"Codex-course-factual-review/1.0"})
    with urllib.request.urlopen(req,timeout=25) as r:
        raw=r.read(4*1024*1024+1)
        if len(raw)>4*1024*1024: raise ValueError("bounded source fetch exceeded 4MiB")
        receipt={"filename":filename,"url":url,"final_url":r.url,"http_status":r.status,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"accessed_on":datetime.now(UTC).date().isoformat()}
    (OUT/filename).write_bytes(raw)
    receipts.append(receipt)
    (OUT/"fetch-receipts.json").write_text(json.dumps(receipts,indent=2)+"\n")
    print(json.dumps(receipt))
