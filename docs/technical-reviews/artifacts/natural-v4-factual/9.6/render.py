import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    root = Path(__file__).resolve().parents[5]
    svg = (root/'course/figures/foundations_tensor_axes.svg').read_text()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path='/usr/bin/chromium',
            args=['--no-sandbox','--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width':680,'height':500}, device_scale_factor=1)
        # Managed Chromium blocks file navigation; load the same local SVG bytes
        # directly into the page, with only the document body margins removed.
        await page.set_content('<style>html,body{margin:0;padding:0}</style>'+svg)
        await page.screenshot(path=str(Path(__file__).parent/'foundations_tensor_axes.png'))
        print('Rendered current SVG via page.set_content; Chromium '+browser.version+'; viewport=680x500; scale=1')
        await browser.close()

asyncio.run(main())
