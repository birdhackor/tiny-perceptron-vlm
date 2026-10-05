"""Preserve the already executed original fence; run the independent bounded probe."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent.parent
ORIGINAL=Path('/tmp/phase4-7_7-original')
(ART/'original-run').mkdir(exist_ok=True)
for name in ('section.md','fence-1.py','bootstrap.py','extraction.json','execution.json','environment.json','stdout.txt','stderr.txt'):
    shutil.copyfile(ORIGINAL/name,ART/'original-run'/name)
records=[{'step':'original_fence','command':'.venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.7 --output /tmp/phase4-7_7-original --execute --timeout 45',
    'cwd':str(ROOT),'exit_code':json.loads((ART/'original-run/execution.json').read_text())['exit_code'],
    'preservation':'Only original raw section, fence, bootstrap, metadata, stdout, stderr and environment copied; temporary workspace symlinks not used as formal artifacts.'}]
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
steps=[('probe',[str(ROOT/'.venv/bin/python'),str(ART/'code/bounded_probe.py')],45),
       ('render',['inkscape',str(ROOT/'course/figures/rewrite-07-07-padding-positions.svg'),'--export-type=png','--export-filename='+str(ART/'figure/padding.png'),'--export-width=1280'],25),
       ('inkscape-version',['inkscape','--version'],10),
       ('extract-paper',['pdftotext','-layout',str(ART/'sources/attention-v7.pdf'),str(ART/'sources/attention-v7.txt')],20)]
for name,argv,timeout in steps:
    p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=timeout,check=False)
    (ART/(name+'-stdout.txt')).write_bytes(p.stdout)
    (ART/(name+'-stderr.txt')).write_bytes(p.stderr)
    records.append({'step':name,'command_argv':argv,'cwd':str(ROOT),'timeout_seconds':timeout,'exit_code':p.returncode,
                    'stdout':name+'-stdout.txt','stderr':name+'-stderr.txt'})
    if p.returncode: print(p.stderr.decode(errors='replace')); raise SystemExit(p.returncode)
(ART/'commands.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
