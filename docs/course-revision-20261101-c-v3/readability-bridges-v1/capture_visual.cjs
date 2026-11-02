const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('playwright');

(async () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const output = `/workspace/reader-evidence/c_v3_reader_bridges/visual-v1-${stamp}`;
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
  const records = [];
  for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
    for (const id of ['9.18', '19.13', '19.16', '19.17', '19.19']) {
      const url = `http://127.0.0.1:8765/${id}.html`;
      await page.goto(url, { waitUntil: 'networkidle', timeout: 25000 });
      await page.evaluate(() => document.fonts.ready);
      const metrics = await page.evaluate(() => ({
        title: document.title,
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
        documentHeight: document.documentElement.scrollHeight,
        pres: [...document.querySelectorAll('.md-content__inner pre')].map((el, index) => ({
          index, top: el.getBoundingClientRect().top + window.scrollY,
          height: el.getBoundingClientRect().height, width: el.clientWidth, scrollWidth: el.scrollWidth,
          textStart: el.textContent.slice(0, 70), overflowX: getComputedStyle(el).overflowX,
        })),
        tables: [...document.querySelectorAll('.md-content__inner table')].map((el, index) => ({
          index, top: el.getBoundingClientRect().top + window.scrollY,
          width: el.getBoundingClientRect().width, height: el.getBoundingClientRect().height,
          textStart: el.textContent.slice(0, 70),
        })),
        htmlShaInput: document.documentElement.outerHTML,
      }));
      const htmlSha = crypto.createHash('sha256').update(metrics.htmlShaInput).digest('hex');
      delete metrics.htmlShaInput;
      const shots = [];
      async function capture(label, y, fullPage = false) {
        if (!fullPage) {
          await page.evaluate(y => window.scrollTo(0, y), y);
          await page.waitForTimeout(180);
        }
        const filename = path.join(output, `${id}-${viewport.width}x${viewport.height}-${label}.png`);
        await page.screenshot({ path: filename, fullPage });
        const actualY = await page.evaluate(() => window.scrollY);
        shots.push({ label, path: filename, sha256: crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex'), scrollY: actualY, fullPage });
      }
      await capture('top', 0);
      if (metrics.pres.length) {
        await capture('code-start', Math.max(0, metrics.pres[0].top - 95));
      } else if (metrics.tables.length) {
        await capture('table', Math.max(0, metrics.tables[0].top - 95));
      }
      if (id === '19.16') {
        const source = metrics.pres.find(el => el.textStart.includes('節錄自')) || metrics.pres[metrics.pres.length - 1];
        if (source) await capture('model-source', Math.max(0, source.top - 95));
      } else if (id === '19.19' && metrics.pres.length > 1) {
        await capture('commands', Math.max(0, metrics.pres[1].top - 95));
      } else if (id === '9.18' && metrics.pres.length > 1) {
        await capture('output', Math.max(0, metrics.pres[1].top - 95));
      }
      if (metrics.tables.length && id !== '19.13') {
        await capture('last-table', Math.max(0, metrics.tables[metrics.tables.length - 1].top - 95));
      } else {
        await capture('tail', Math.max(0, metrics.documentHeight - viewport.height));
      }
      await capture('full', 0, true);
      records.push({ page_id: id, url, viewport, captured_at: new Date().toISOString(), html_sha256: htmlSha, metrics, screenshots: shots });
    }
    await page.close();
  }
  await browser.close();
  const manifest = path.join(output, 'capture-manifest.json');
  fs.writeFileSync(manifest, JSON.stringify({ schema_version: 1, reviewer_task: '/root/c_v3_reader_bridges', browser: '/usr/bin/chromium', acquisition_script: __filename, records }, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ manifest, sha256: crypto.createHash('sha256').update(fs.readFileSync(manifest)).digest('hex'), pages: records.map(r => ({ id: r.page_id, viewport: r.viewport, height: r.metrics.documentHeight, width: r.metrics.documentWidth, pres: r.metrics.pres.length, tables: r.metrics.tables.length, screenshots: r.screenshots.map(x => ({ label: x.label, path: x.path })) })) }, null, 2));
})();
