from pathlib import Path
from datetime import datetime, UTC
import hashlib, json, re, sys
chapter=Path("course/chapters/20.md").read_text()
expected={"20.11":"docs/course-review-20261019/freeze-01/sources/20.11.md","20.13":"docs/course-review-20261019/freeze-05/sources/20.13.md"}
rows=[]
for lid,path in expected.items():
    current=re.search(r"^## "+re.escape(lid)+r" .*?(?=^## |\Z)",chapter,re.S|re.M).group(0).encode()
    snapshot=Path(path).read_bytes()
    assert current==snapshot
    rows.append({"lesson_id":lid,"snapshot":path,"sha256":hashlib.sha256(snapshot).hexdigest(),"current_slice_equal":True})
context=Path(expected["20.13"]).read_text()
labels=["真實照片裡指定區域的逐字抄寫", "真實文字的指定讀序與分行"]
assert all(f"| {x} |" in context for x in labels)
figure=Path("course/figures/natural_reading_order.svg")
frozen=Path("docs/course-publish-20261008/freeze/current/figures/course/figures/natural_reading_order.svg")
assert figure.read_bytes()==frozen.read_bytes()
print(json.dumps({"checked_at":datetime.now(UTC).isoformat(),"environment":{"python":sys.version,"device":"CPU; byte comparison only"},"current_source_bindings":rows,"20.13_separate_capability_row_labels":labels,"20.13_scope_quote":"任意整頁、多欄與即興的長語音對話都沒有測，保持未驗收。","figure":{"path":str(figure),"sha256":hashlib.sha256(figure.read_bytes()).hexdigest(),"frozen_bytes_equal":True},"result":"all assertions passed; cross-reference structure only, no historical quality-score re-audit"},ensure_ascii=False,indent=2))
