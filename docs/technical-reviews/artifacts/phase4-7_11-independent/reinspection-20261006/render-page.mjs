import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const out='/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-7_11-independent/reinspection-20261006';
const profile=await mkdtemp(join(tmpdir(),'phase4-7-11-reinspection-chrome-'));
const args=['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--disable-extensions','--disable-sync','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'];
const child=spawn('/usr/bin/chromium',args,{detached:true,stdio:['ignore','ignore','pipe']});let errors='';child.stderr.on('data',b=>errors+=b.toString());
let ws;const pending=new Map();let next=1;
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=next++;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
try {
 let port;for(let i=0;i<70;i++){try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{}await new Promise(r=>setTimeout(r,100));}
 if(!port)throw Error('No Chromium debug port within 7 seconds');
 const targets=await (await fetch('http://127.0.0.1:'+port+'/json')).json();
 const target=targets.find(t=>t.type==='page');ws=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}});
 await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:8765/7.11.html'});
 let data;for(let i=0;i<50;i++){
  const v=await call('Runtime.evaluate',{expression:`(()=>{const a=document.querySelector('article');if(!a)return null;for(const d of a.querySelectorAll('details'))d.open=true;const p=[...a.querySelectorAll('p')].find(p=>p.textContent.includes('成績見'));if(!p)return null;p.scrollIntoView({block:'center'});return {paragraph:p.textContent,links:[...p.querySelectorAll('a')].map(a=>({text:a.textContent,href:a.getAttribute('href')})),target_images:a.querySelectorAll('img').length,url:location.href};})()`,returnByValue:true});
  data=v.result.value;if(data)break;await new Promise(r=>setTimeout(r,100));
 }
 if(!data)throw Error('Current updated paragraph not rendered within 5 seconds');
 await new Promise(r=>setTimeout(r,200));
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(out,'page-current-paragraph.png'),Buffer.from(shot.data,'base64'));
 await writeFile(join(out,'page-render-receipt.json'),JSON.stringify({command:'/usr/bin/node '+join(out,'render-page.mjs'),chromium_argv:args,dom:data,exit_code:0,screenshot:'page-current-paragraph.png'},null,2)+'\n');console.log(JSON.stringify(data));
}finally {if(ws)ws.close();try{process.kill(-child.pid,'SIGKILL');}catch{}await new Promise(r=>setTimeout(r,150));await writeFile(join(out,'page-cdp.stderr.txt'),errors);await rm(profile,{recursive:true,force:true});}
