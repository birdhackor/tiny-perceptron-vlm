const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('/opt/codex/runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = __dirname;
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--disable-gpu'], timeout: 15000 });
  const results = [];
  try {
    for (const [name, width, height] of [['desktop', 1280, 900], ['mobile', 390, 844]]) {
      const page = await browser.newPage({ viewport: { width, height } });
      await page.goto('http://127.0.0.1:8765/7.10.html', { waitUntil: 'networkidle', timeout: 15000 });
      await page.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const text = await page.locator('body').innerText();
      if (!text.includes('合法有一格，只表示有分母，不保證完成了完整回答訓練。')) throw new Error('current essential limitation missing');
      if (text.includes('有一格目標只表示平均合法，不表示完整回答有訓練。')) throw new Error('deleted duplicate still present');
      if (!text.includes('首有效索引18') || !text.includes('最低長度19') || !text.includes('完整長度22')) throw new Error('alignment claim missing');
      const image = path.join(base, `${name}-current.png`);
      await page.screenshot({ path: image, fullPage: true, timeout: 15000 });
      fs.writeFileSync(path.join(base, `${name}-body.txt`), text);
      results.push({ viewport: { width, height }, name, screenshot: path.basename(image), screenshot_sha256: hash(image), body_sha256: hash(path.join(base, `${name}-body.txt`)), page_title: await page.title(), details_open: await page.locator('details[open]').count(), deleted_duplicate_absent: true, retained_limitation_present: true, horizontal_overflow: await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth) });
      await page.close();
    }
    const record = { command: `node ${path.relative(process.cwd(), __filename)}`, browser_version: browser.version(), url: 'http://127.0.0.1:8765/7.10.html', device: 'CPU/headless Chromium', results };
    fs.writeFileSync(path.join(base, 'render-execution.json'), JSON.stringify(record, null, 2) + '\n');
    console.log(JSON.stringify(record, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
