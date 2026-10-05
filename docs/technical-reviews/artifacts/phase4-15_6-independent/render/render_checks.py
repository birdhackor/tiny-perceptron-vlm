"""Capture genuine Inkscape/Chromium results for the frozen SVG, not course build output."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-15_6-independent'
OUT=BASE/'render'
records=[]
for name,destination in [('rewrite-15-row-restore.svg','row-restore.png'),('rewrite-15-combine.svg','dependency-combine.png')]:
    source=BASE/'inputs/course/figures'/name
    command=['/usr/bin/inkscape',str(source),'--export-type=png','--export-filename='+str(OUT/destination)]
    run=subprocess.run(command,capture_output=True,text=True,timeout=30)
    records.append({'command_argv':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'file':destination})
    assert run.returncode==0 and (OUT/destination).is_file()
html='''<!doctype html><meta charset="utf-8"><title>15.6 frozen figure inspection</title><style>body{font-family:sans-serif;color:#142c45;margin:24px auto;padding:0 16px;max-width:640px}h1{font-size:24px}figure{margin:0}img{width:100%;height:auto}figcaption{line-height:1.6}</style><h1>15.6 Token 怎麼送出再排回來？</h1><figure><img src="../inputs/course/figures/rewrite-15-row-restore.svg"><figcaption>列2的兩份貢獻相加，收回後恢復原token順序。</figcaption></figure>'''
(OUT/'figure-inspection.html').write_text(html)
for viewport in ['1280,800','390,844']:
    filename='desktop.png' if viewport.startswith('1280') else 'mobile.png'
    command=['/usr/bin/chromium','--no-sandbox','--headless','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-proxy-server','--disable-extensions','--virtual-time-budget=2000','--hide-scrollbars','--allow-file-access-from-files','--user-data-dir=/tmp/phase4-15_6-chromium-retry-'+filename,'--window-size='+viewport,'--screenshot='+str(OUT/filename),(OUT/'figure-inspection.html').as_uri()]
    try:
        run=subprocess.run(command,capture_output=True,text=True,timeout=30)
        records.append({'command_argv':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'file':filename,'viewport':viewport})
    except subprocess.TimeoutExpired as error:
        records.append({'command_argv':command,'exit_code':None,'timed_out_seconds':30,'stdout':(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout,bytes) else error.stdout,'stderr':(error.stderr or b'').decode(errors='replace') if isinstance(error.stderr,bytes) else error.stderr,'file':filename,'viewport':viewport})
versions={name:subprocess.run([binary,'--version'],capture_output=True,text=True,timeout=10).stdout.strip() for name,binary in [('inkscape','/usr/bin/inkscape'),('chromium','/usr/bin/chromium')]}
for item in records:
    if (OUT/item['file']).is_file():item['sha256']=hashlib.sha256((OUT/item['file']).read_bytes()).hexdigest()
payload={'environment':versions,'scope':'Standalone frozen original SVG and a simple responsive figure inspection page. This does not assert final course CSS or the complete book layout was tested. Pango/Gtk warnings and Chromium stderr retained verbatim; inspect individual exit/timeout records. Initial Chromium timeout retained separately in first-attempt-timeout-stderr.txt.','runs':records}
(OUT/'render-results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'environment':versions,'renders':[{'file':r['file'],'exit_code':r['exit_code'],'sha256':r.get('sha256'),'timeout':r.get('timed_out_seconds')} for r in records]},ensure_ascii=False))
