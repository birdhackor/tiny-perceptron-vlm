const { chromium } = require('/opt/codex/runtimes/cua/lib/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '../../../..');
const external = '/workspace/work/omni-P5-first-steps-technical-v1-external';
const browserPath = '/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell';
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
(async () => {
  const browser = await chromium.launch({executablePath: browserPath, headless: true, args: ['--no-sandbox']});
  const pagePath = path.join(root, 'outputs/omni-P5-preview-v5/first-steps.html');
  const rows = [];
  for (const width of [1280, 390]) {
    const page = await browser.newPage({viewport: {width, height: 900}, deviceScaleFactor: 1});
    await page.goto('file://' + pagePath, {waitUntil: 'networkidle'});
    for (const figure of ['rewrite-W-3-axes.svg', 'rewrite-W-6-gradient-step.svg']) {
      const img = page.locator('img[src$="' + figure + '"]');
      await img.scrollIntoViewIfNeeded();
      await img.evaluate(node => node.decode());
      const target = path.join(external, figure + '-' + width + '.png');
      await img.screenshot({path: target});
      const bytes = fs.readFileSync(target);
      rows.push({viewport: width, figure, screenshot_external_path: target, screenshot_bytes: bytes.length, screenshot_sha256: sha(bytes), dom: await img.evaluate(node => ({src: node.getAttribute('src'), alt: node.alt, naturalWidth: node.naturalWidth, naturalHeight: node.naturalHeight, displayed: {width: node.getBoundingClientRect().width, height: node.getBoundingClientRect().height}})), original_sha256: sha(fs.readFileSync(path.join(root, 'course/figures', figure))), preview_figure_sha256: sha(fs.readFileSync(path.join(root, 'outputs/omni-P5-preview-v5/figures', figure)))});
    }
    await page.close();
  }
  console.log(JSON.stringify({created_at: new Date().toISOString(), browser: await browser.version(), browser_executable: browserPath, acquisition_command: 'node docs/technical-reviews/artifacts/omni-P5-first-steps/render-probe-v1.cjs', preview_input: 'outputs/omni-P5-preview-v5/first-steps.html', preview_input_sha256: sha(fs.readFileSync(pagePath)), screenshots: rows, scope: 'Actual image elements at two widths; screenshot files remain outside Git. This receipt is portable; re-render from frozen SVGs if the ignored preview is unavailable.'}, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
