"""Save real renderer command results, including bounded Chromium failures."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-14_1-independent'
records=[]
def run(label,command,timeout=20):
    start=time.monotonic()
    p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:
        out,err=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(p.pid,signal.SIGKILL)
        out,err=p.communicate()
    (BASE/'render'/f'{label}.stdout.txt').write_bytes(out)
    (BASE/'render'/f'{label}.stderr.txt').write_bytes(err)
    records.append({'label':label,'command':command,'cwd':str(ROOT),'timeout_seconds':timeout,
                    'timed_out':timed_out,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start})
    print(label,'exit',p.returncode,'timed_out',timed_out)

run('inkscape-version',['inkscape','--version'])
run('chromium-version',['chromium','--version'])
for source,output in [('rewrite-14-shared-rotation.svg','shared-rotation.png'),
                      ('rewrite-14-rotation-components.svg','rotation-components.png')]:
    run('inkscape-'+output,['inkscape',str(ROOT/'course/figures'/source),'--export-type=png',
                           '--export-filename='+str(BASE/'render'/output)])
for label,size in [('desktop-retry','1280,800'),('mobile','390,844')]:
    screenshot=BASE/'render'/f'{label}.png'
    run('chromium-'+label,['chromium','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                          '--no-first-run','--user-data-dir=/tmp/phase4-14_1-'+label,
                          '--screenshot='+str(screenshot),'--window-size='+size,
                          'file://'+str(BASE/'render/section.html')],timeout=20)
    records[-1]['screenshot_exists']=screenshot.is_file()
(BASE/'render/commands.json').write_text(json.dumps(records,indent=2)+'\n')
