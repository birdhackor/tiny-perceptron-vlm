"""Render the existing 11.7 page with system Chromium; do not rebuild the course."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
URL="http://127.0.0.1:8765/11.7.html"
raw=urllib.request.urlopen(URL,timeout=15).read()
(OUT/"rendered-page-input.html").write_bytes(raw)
result={"url":URL,"inspected_on":"2026-10-05","page_input_sha256":hashlib.sha256(raw).hexdigest(),
        "chromium_version":subprocess.check_output(["/usr/bin/chromium","--version"],text=True).strip(),
        "source_section_sha256":hashlib.sha256((OUT/"section.md").read_bytes()).hexdigest(),
        "views":[],"scope":"Current served page rendered; checks are source paragraph/code/table equality and visual inspection of screenshots. No full export or training."}
source=(OUT/"section.md").read_text()
normal=lambda s:re.sub(r"\s+","",s)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox"])
    for label,width,height in [("desktop",1280,800),("mobile",390,844)]:
        page=browser.new_page(viewport={"width":width,"height":height},device_scale_factor=1)
        page.goto(URL,wait_until="networkidle",timeout=30000)
        text=page.locator("body").inner_text()
        paragraphs=[line for line in source.splitlines() if line and not line.startswith(("#","|","<","```"))]
        # Code is checked as a single contiguous preformatted block; skip its individual lines.
        code=(OUT/"fence-1.py").read_text().strip()
        pre=[x.inner_text() for x in page.locator("pre").all()]
        assert any(normal(code) in normal(v) for v in pre),"served page code differs from section"
        required=["recipe 將 1,000 個有效位置分成文字 750、圖文 250，印比例 0.75、0.25；recorded 是人工成績示例，兩類各除固定十題，印前後正確率。這不是 75% 回放的訓練結果，而是報告方式。",
                  "原起點文字都是 5/10。回放版較能保留這批文字題，但沒有完全回到起點；它和其他版本的總有效目標不相同，所以不是嚴格隔離回放比例的因果比較。0.5 抽文字題的機率，也不等於有效位置或時間各半。",
                  "實測要看逐題哪個答案改對、哪個改錯，固定生成方式，不只看淨少兩題。混合 loss 下降也可能來自較容易的組佔比變大。練習只改 recipe 為 500／500，比例成各半；recorded 保持原值，因為改計畫不會自行產生成績。"]
        for line in required:assert normal(line) in normal(text),line
        table=next(t for t in page.locator("table").all() if "更新方案" in t.inner_text() and "原文字題" in t.inner_text())
        cells=table.locator("tr").all_text_contents()
        expected=["更新方案新圖片題原文字題","只訓接頭3/125/10","接頭＋最後文字區塊12/122/10","全部開放9/121/10","全部＋文字回放9/124/10"]
        assert [normal(c) for c in cells] == expected
        page.screenshot(path=str(OUT/f"page-{label}-top.png"))
        table.scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/f"page-{label}-table.png"))
        rect=table.bounding_box()
        metrics=page.evaluate("({innerWidth:innerWidth,scrollWidth:document.documentElement.scrollWidth,bodyScrollWidth:document.body.scrollWidth})")
        result["views"].append({"label":label,"viewport":{"width":width,"height":height},
                                "table_bbox":rect,"table_rows":[normal(c) for c in cells],"page_metrics":metrics,
                                "screenshots":[f"page-{label}-top.png",f"page-{label}-table.png"],
                                "source_paragraph_code_table_checks":"passed"})
        page.close()
    browser.close()
(OUT/"page-render-receipt.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
