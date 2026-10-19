// 用 headless Chrome 的 DevTools 協定截整頁網頁，供協調者抽樣核對桌面與手機版面；不產生審閱判斷。
//
// 用法：node render_pages.mjs <網站根網址> <輸出目錄> <Chrome 設定檔目錄> <寬>x<高>[@mobile] <頁 ID>...
// 每頁開 <網站根網址>/<頁 ID>.html，等 load 事件後再等 1.5 秒讓字型與數學排版完成，截下整頁 PNG。
// stdout 印 JSON：Chrome 版本、每頁的網址、視窗、內容尺寸、整頁是否橫向溢出、PNG 路徑與 SHA-256。
//
// 驗證：截圖時的視窗寬高與是否模擬手機；文件寬度是否超過視窗（document 的 scrollWidth 大於 innerWidth）。
// 不涵蓋：等待時間之後才出現的版面變化、互動後的狀態（折疊、跳轉、捲動中的元素）、其他瀏覽器與真實手機。

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [base, out, profile, size, ...pages] = process.argv.slice(2);
const mobile = size.endsWith("@mobile");
const [width, height] = size.replace("@mobile", "").split("x").map(Number);
const port = 9400 + Math.floor(Math.random() * 500);
const chrome = spawn(
  CHROME,
  ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"],
  { stdio: "ignore" },
);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function devtools(path, init) {
  for (let i = 0; i < 100; i++) {
    try {
      return await (await fetch(`http://127.0.0.1:${port}${path}`, init)).json();
    } catch {
      await sleep(100);
    }
  }
  throw new Error("Chrome DevTools 沒有回應");
}

try {
  const version = await devtools("/json/version");
  const target = await devtools("/json/new?about:blank", { method: "PUT" });
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
  let id = 0;
  const pending = new Map();
  const loads = [];
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    } else if (message.method === "Page.loadEventFired") {
      loads.push(message);
    }
  });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const call = ++id;
      pending.set(call, (m) => (m.error ? reject(new Error(`${method}: ${JSON.stringify(m.error)}`)) : resolve(m.result)));
      ws.send(JSON.stringify({ id: call, method, params }));
    });
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile });
  await send("Emulation.setTouchEmulationEnabled", { enabled: mobile });
  const records = [];
  for (const page of pages) {
    const url = `${base}/${page}.html`;
    loads.length = 0;
    await send("Page.navigate", { url });
    for (let i = 0; i < 300 && loads.length === 0; i++) await sleep(50);
    if (loads.length === 0) throw new Error(`頁面沒有載入完成：${url}`);
    await sleep(1500);
    const metrics = await send("Page.getLayoutMetrics");
    const content = metrics.cssContentSize;
    const overflow = await send("Runtime.evaluate", {
      expression: "document.documentElement.scrollWidth > window.innerWidth",
      returnByValue: true,
    });
    const shot = await send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
      clip: { x: 0, y: 0, width, height: Math.ceil(content.height), scale: 1 },
    });
    const raw = Buffer.from(shot.data, "base64");
    const name = `${page}-${width}x${height}${mobile ? "-mobile" : ""}.png`;
    writeFileSync(join(out, name), raw);
    records.push({
      page,
      url,
      viewport: { width, height, mobile },
      css_content_size: { width: content.width, height: content.height },
      horizontal_overflow: overflow.result.value,
      png: { path: join(out, name), sha256: createHash("sha256").update(raw).digest("hex") },
    });
  }
  ws.close();
  console.log(JSON.stringify({ program: version.Browser, records }, null, 1));
} finally {
  chrome.kill();
}
