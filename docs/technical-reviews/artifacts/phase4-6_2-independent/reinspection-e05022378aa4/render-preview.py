"""Bounded browser renders of the actual current section preview."""
from datetime import datetime,UTC
from importlib.metadata import version
from hashlib import sha256
from pathlib import Path
import json
import sys
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
receipt={"observed_on":datetime.now(UTC).isoformat(),"url":"http://127.0.0.1:8765/6.2.html",
 "environment":{"python":sys.version,"playwright":version("playwright"),"device":"CPU","browser_executable":"/usr/bin/chromium"},"renders":[]}
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox","--disable-dev-shm-usage"],timeout=12000)
  receipt["environment"]["chromium"]=browser.version
  for label,width,height in [("desktop",1280,800),("mobile",390,844)]:
   page=browser.new_page(viewport={"width":width,"height":height},device_scale_factor=1)
   response=page.goto(receipt["url"],wait_until="domcontentloaded",timeout=10000)
   page.wait_for_timeout(300)
   path=OUT/f"preview-{label}.png"
   page.screenshot(path=str(path),full_page=True,timeout=10000)
   (OUT/f"preview-{label}-body.txt").write_text(page.locator("body").inner_text(timeout=5000),encoding="utf-8")
   receipt["renders"].append({"label":label,"viewport":{"width":width,"height":height},"http_status":response.status,"path":str(path),"sha256":sha256(path.read_bytes()).hexdigest()})
   page.close()
  browser.close()
 receipt["exit_code"]=0
except Exception as exc:
 receipt.update(exit_code=1,error_type=type(exc).__name__,error=str(exc))
(OUT/"browser-render-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(receipt,ensure_ascii=False,indent=2))
raise SystemExit(receipt["exit_code"])
