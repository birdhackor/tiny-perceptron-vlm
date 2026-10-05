"""Render only this review's frozen section; no course-site acceptance claim."""
import hashlib
import json
import subprocess
from pathlib import Path
import markdown

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-15_4-independent'
section=(OUT/'inputs/section-15.4.md').read_text()
assert '![' not in section, 'Referenced figures would need separate primary inspection'
body=markdown.markdown(section.replace('<details>','<details open markdown="1">'),extensions=['fenced_code','md_in_html'])
page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>15.4 frozen review rendering</title><style>
body{font-family:"Noto Sans CJK TC","Noto Sans CJK SC",sans-serif;font-size:18px;line-height:1.6;margin:24px auto;max-width:920px;padding:0 22px;color:#181818;background:white}h2{font-size:1.55em;line-height:1.35}pre{overflow:auto;background:#f4f6f7;border:1px solid #dde1e3;border-radius:6px;padding:14px;font-size:15px;line-height:1.4}code{font-family:monospace}p code{font-size:.9em;background:#f4f6f7;padding:0 .18em}details{border:1px solid #cbd0d4;padding:12px}summary{font-weight:bold}a{color:#174ea6}@media(max-width:500px){body{font-size:16px;margin:14px auto;padding:0 16px}pre{font-size:12px}}
</style>'''+body+'</html>'
html=OUT/'section-render.html';html.write_text(page)
records=[]
version=subprocess.run(['/usr/bin/chromium','--version'],capture_output=True,check=True,text=True).stdout.strip()
for name,width,height in (('desktop',1280,1800),('mobile',390,4000)):
    image=OUT/('section-'+name+'.png')
    argv=['/usr/bin/chromium','--headless','--no-sandbox','--disable-gpu','--no-proxy-server','--disable-dev-shm-usage','--disable-background-networking','--disable-component-update','--disable-extensions','--no-first-run','--virtual-time-budget=1000','--timeout=8000','--allow-file-access-from-files','--hide-scrollbars','--user-data-dir=/tmp/phase4-15_4-chromium-retry-'+name,'--window-size='+str(width)+','+str(height),'--screenshot='+str(image),html.as_uri()]
    try:
        run=subprocess.run(argv,capture_output=True,timeout=20)
    except subprocess.TimeoutExpired as error:
        (OUT/('render-'+name+'.stdout.txt')).write_bytes(error.stdout or b'')
        (OUT/('render-'+name+'.stderr.txt')).write_bytes(error.stderr or b'')
        records.append({'name':name,'command_argv':argv,'exit_code':124,'timeout_seconds':20,'image_exists':image.is_file()})
        continue
    (OUT/('render-'+name+'.stdout.txt')).write_bytes(run.stdout)
    (OUT/('render-'+name+'.stderr.txt')).write_bytes(run.stderr)
    records.append({'name':name,'command_argv':argv,'exit_code':run.returncode,'width':width,'height':height,'image_exists':image.is_file(),'sha256':hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None})
report={'renderer':version,'markdown_version':markdown.__version__,'view_scope':'Frozen original 15.4 rendered independently, with details expanded; no referenced figures, no check of deployed course CSS.','renders':records}
(OUT/'render-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
