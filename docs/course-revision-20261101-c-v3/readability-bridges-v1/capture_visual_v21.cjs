const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('playwright');

(async () => {
  const output = `/workspace/reader-evidence/c_v3_reader_bridges/visual-v21-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext();
  const response = await context.request.get('http://127.0.0.1:8765/build-info.json');
  if (!response.ok()) throw new Error(`build-info status ${response.status()}`);
  const buildRaw = await response.body();
  const buildInfo = JSON.parse(buildRaw.toString('utf8'));
  if (buildInfo.revision !== 'worktree-c-v3-preview-21') throw new Error('Unexpected preview revision');
  const buildPath = path.join(output, 'build-info.json');
  fs.writeFileSync(buildPath, buildRaw, { flag: 'wx' });
  const records = [];
  for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
    const page = await context.newPage({ viewport });
    await page.setViewportSize(viewport);
    for (const id of ['19.17', '19.19']) {
      const url = `http://127.0.0.1:8765/${id}.html`;
      await page.goto(url, { waitUntil: 'networkidle', timeout: 25000 });
      await page.evaluate(() => document.fonts.ready);
      const article = page.locator('.md-content__inner');
      const bodyText = await article.innerText();
      const targets = id === '19.17' ? [
        ['budget-code', article.locator('.language-python pre').filter({ hasText: '可跑預算' }).first()],
        ['budget-prose', article.locator('p').filter({ hasText: '只給 2 輪預算' }).first()],
        ['average-scope', article.locator('p').filter({ hasText: '在本輪完整決策題上' }).first()],
      ] : [
        ['rng-state', article.locator('p').filter({ hasText: 'RNG（Random Number Generator，隨機數生成器）的狀態' }).first()],
      ];
      if (id === '19.17' && bodyText.includes('指定 medium、硬上限 2 → 實跑')) throw new Error('Old budget label still rendered');
      const shots = [];
      async function capture(label, target) {
        await target.evaluate(el => window.scrollTo(0, Math.max(0, el.getBoundingClientRect().top + window.scrollY - 95)));
        await page.waitForTimeout(180);
        const filename = path.join(output, `${id}-${viewport.width}x${viewport.height}-${label}.png`);
        await page.screenshot({ path: filename });
        shots.push({ label, path: filename, sha256: crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex'), scrollY: await page.evaluate(() => window.scrollY), viewport });
      }
      for (const [label, target] of targets) {
        if (await target.count() !== 1) throw new Error(`Cannot find ${id}/${label}`);
        await capture(label, target);
        if (label === 'budget-code' && viewport.width === 390) {
          const scrolls = await article.locator('pre').evaluateAll(elements => elements.filter(el => el.textContent.includes('可跑預算')).map(el => {
            const nodes = [el, ...el.querySelectorAll('*')].filter(x => x.scrollWidth > x.clientWidth + 1 && ['auto', 'scroll'].includes(getComputedStyle(x).overflowX));
            const target = nodes[0];
            if (!target) return { scrollable: false };
            const before = target.scrollLeft;
            target.scrollLeft = 100;
            return { scrollable: true, tag: target.tagName, before, after: target.scrollLeft, width: target.clientWidth, scrollWidth: target.scrollWidth, overflowX: getComputedStyle(target).overflowX };
          }));
          await capture('budget-caption-pan', target);
          shots[shots.length - 1].horizontal_interactions = scrolls;
        }
      }
      records.push({ page_id: id, url, viewport, captured_at: new Date().toISOString(), rendered_html_sha256: crypto.createHash('sha256').update(await page.content()).digest('hex'), documentWidth: await page.evaluate(() => document.documentElement.scrollWidth), documentHeight: await page.evaluate(() => document.documentElement.scrollHeight), targets: await Promise.all(targets.map(async ([label, target]) => ({ label, text: await target.innerText() }))), screenshots: shots });
    }
    await page.close();
  }
  await browser.close();
  const manifest = path.join(output, 'capture-manifest.json');
  fs.writeFileSync(manifest, JSON.stringify({ schema_version: 1, reviewer_task: '/root/c_v3_reader_bridges', acquisition_script: __filename, browser: '/usr/bin/chromium', build_info: { path: buildPath, sha256: crypto.createHash('sha256').update(buildRaw).digest('hex'), revision: buildInfo.revision, publication_ready: buildInfo.publication_ready }, preview_only: true, records }, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ manifest, sha256: crypto.createHash('sha256').update(fs.readFileSync(manifest)).digest('hex'), build_revision: buildInfo.revision, publication_ready: buildInfo.publication_ready, records: records.map(r => ({ page_id: r.page_id, viewport: r.viewport, screenshots: r.screenshots.map(s => ({ label: s.label, path: s.path })) })) }, null, 2));
})();
