import {spawn} from 'node:child_process';
import {writeFile, mkdir} from 'node:fs/promises';
const root = '/workspace/work/omni-book-integrated-v1';
const cache = '/workspace/work/reference/omni-p6-wave04-curriculum-final';
const browser = '/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell';
await mkdir(cache, {recursive: true});
const proc = spawn(browser, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-port=9225', `--user-data-dir=${cache}/cdp-profile`, 'about:blank'], {stdio:'ignore'});
let tabs;
for (let i=0; i<50; i++) {
  try {tabs = await (await fetch('http://127.0.0.1:9225/json/list')).json(); if(tabs.length) break;} catch {}
  await new Promise(r=>setTimeout(r,100));
}
if(!tabs?.length) throw new Error('browser unavailable');
const ws = new WebSocket(tabs[0].webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;
const pending=new Map();
ws.addEventListener('message',event=>{const r=JSON.parse(event.data);if(pending.has(r.id)){pending.get(r.id)(r);pending.delete(r.id);}});
const call=(method,params={})=>new Promise(resolve=>{const rid=++id;pending.set(rid,resolve);ws.send(JSON.stringify({id:rid,method,params}));});
await call('Page.enable');
const receipt=[];
for(const [name,width,height] of [['desktop',1280,1800],['mobile',390,1450]]) {
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:name==='mobile'});
  await call('Page.navigate',{url:`file://${root}/outputs/omni-P6-preview-source18-v2/curriculum.html`});
  for(let i=0;i<50;i++) {
    const r=await call('Runtime.evaluate',{expression:'document.readyState'});
    if(r.result?.result?.value==='complete')break;
    await new Promise(r=>setTimeout(r,100));
  }
  const inspected=await call('Runtime.evaluate',{expression:`(async()=>{await document.fonts.ready;const img=document.querySelector('img[src*="curriculum_learning_flow"]');img.scrollIntoView({block:'start'});await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));window.scrollBy(0,-90);await new Promise(r=>setTimeout(r,500));const b=img.getBoundingClientRect();return {src:img.src,complete:img.complete,naturalWidth:img.naturalWidth,imageBounds:{x:b.x,y:b.y,width:b.width,height:b.height},scrollY:window.scrollY};})()`,awaitPromise:true,returnByValue:true});
  const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  const path=`${cache}/cdp-preview-${name}.png`;
  await writeFile(path,Buffer.from(shot.result.data,'base64'));
  receipt.push({name,width,height,path,inspection:inspected.result?.result?.value});
}
await writeFile(`${cache}/cdp-preview-receipt.json`,JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
ws.close();
proc.kill();
