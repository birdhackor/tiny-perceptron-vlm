from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json, datetime, time
base = "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/"
paths = ["torch/random.py", "torch/nn/modules/module.py", "torch/_torch_docs.py", "docs/source/notes/numerical_accuracy.rst", "torch/nn/modules/linear.py"]
research = Path("outputs/natural-v4/factual-research/10.8")
receipt = []
for source_path in paths:
    url = base + source_path
    item = {"url": url, "version": "installed torch git commit 5c4886908584029761b579af026dcfb627c84070", "accessed_on": datetime.date.today().isoformat()}
    try:
        start=time.perf_counter()
        with urlopen(Request(url, headers={"User-Agent": "Independent-course-technical-review"}),timeout=25) as response:
            body=response.read()
            item.update(status=response.status, final_url=response.url, content_type=response.headers.get("Content-Type"))
        target = research / source_path.replace("/", "__")
        target.write_bytes(body)
        item.update(research_path=str(target), bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), elapsed_seconds=round(time.perf_counter()-start,3))
    except Exception as exc:
        item.update(error=repr(exc))
    receipt.append(item)
    print(json.dumps(item, ensure_ascii=False))
Path("docs/technical-reviews/artifacts/natural-v4-factual/10.8/authority-retrieval.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
