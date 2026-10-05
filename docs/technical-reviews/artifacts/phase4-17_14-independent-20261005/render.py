import asyncio
import hashlib
import json
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SVG = ROOT / "course/figures/rewrite-17-qat-flow.svg"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox"])
        evidence = {"source":str(SVG.relative_to(ROOT)),"source_sha256":hashlib.sha256(SVG.read_bytes()).hexdigest(),"chromium":browser.version,"renders":[]}
        for width,height,name in [(1280,1100,"desktop"),(390,844,"mobile")]:
            page = await browser.new_page(viewport={"width":width,"height":height},device_scale_factor=1)
            await page.set_content('<html><body style="margin:0"><div style="width:100%;max-width:640px">'+SVG.read_text().replace('<svg ','<svg style="display:block;width:100%;height:auto" ',1)+'</div></body></html>')
            await page.locator("svg").screenshot(path=str(OUT/f"qat-flow-{name}.png"))
            evidence["renders"].append({"viewport":[width,height],"path":f"qat-flow-{name}.png","bounding_box":await page.locator("svg").bounding_box()})
            await page.close()
        await browser.close()
        (OUT/"render.json").write_text(json.dumps(evidence,indent=2)+"\n")
        print(json.dumps(evidence,indent=2))

asyncio.run(main())
