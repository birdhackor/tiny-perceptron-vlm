const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('playwright');

(async () => {
  const output = `/workspace/reader-evidence/c_v3_reader_bridges/visual-scroll-v1-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
  const records = [];
  const tasks = [
    ['9.18', 390, 844, 0, 'python-right'],
    ['9.18', 390, 844, 1, 'cpu-output-right'],
    ['19.16', 390, 844, 3, 'model-source-right'],
    ['19.17', 390, 844, 0, 'rule-right'],
    ['19.19', 390, 844, 0, 'release-right'],
    ['19.19', 390, 844, 1, 'commands-right'],
    ['19.19', 1280, 800, 0, 'release-right'],
  ];
  for (const [id, width, height, index, label] of tasks) {
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    const url = `http://127.0.0.1:8765/${id}.html`;
    await page.goto(url, { waitUntil: 'networkidle', timeout: 25000 });
    const pre = page.locator('.md-content__inner pre').nth(index);
    const result = await pre.evaluate(el => {
      const nodes = [el, ...el.querySelectorAll('*')].filter(x => x.scrollWidth > x.clientWidth + 1 && ['auto', 'scroll'].includes(getComputedStyle(x).overflowX));
      const target = nodes.sort((a, b) => (b.scrollWidth - b.clientWidth) - (a.scrollWidth - a.clientWidth))[0];
      if (!target) return { scrollable: false, textStart: el.textContent.slice(0, 100) };
      const before = target.scrollLeft;
      target.scrollLeft = target.scrollWidth;
      window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 95);
      return { scrollable: true, tag: target.tagName, class: target.className, width: target.clientWidth, scrollWidth: target.scrollWidth, before, after: target.scrollLeft, overflowX: getComputedStyle(target).overflowX, textStart: el.textContent.slice(0, 100) };
    });
    await page.waitForTimeout(200);
    const filename = path.join(output, `${id}-${width}x${height}-${label}.png`);
    await page.screenshot({ path: filename });
    records.push({ page_id: id, url, viewport: { width, height }, pre_index: index, label, result, screenshot: { path: filename, sha256: crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex') }, captured_at: new Date().toISOString() });
    await page.close();
  }
  await browser.close();
  const manifest = path.join(output, 'capture-manifest.json');
  fs.writeFileSync(manifest, JSON.stringify({ schema_version: 1, reviewer_task: '/root/c_v3_reader_bridges', acquisition_script: __filename, records }, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ manifest, sha256: crypto.createHash('sha256').update(fs.readFileSync(manifest)).digest('hex'), records }, null, 2));
})();
