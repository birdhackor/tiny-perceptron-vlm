"""讀取官方原文並只保存本次主張所需的短摘錄。"""
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).with_name('original-source-excerpts.json')
COMMIT = '5c4886908584029761b579af026dcfb627c84070'
BASE = f'https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/'
SPECS = [
    ('cuda', BASE+'docs/source/notes/cuda.md', [
        ('Asynchronous execution', 1700),
        ('# Memory management', 1400),
    ]),
    ('randomness', BASE+'docs/source/notes/randomness.md', [
        ('You can use', 800),
        ('Completely reproducible results', 650),
    ]),
    ('no_grad', BASE+'torch/autograd/grad_mode.py', [('class no_grad(', 950)]),
    ('torch_api', BASE+'torch/_torch_docs.py', [
        ('torch.set_num_threads,', 700),
        ('torch.randn,', 750),
        ('.. function:: where(condition) -> tuple', 550),
        ('torch.topk,', 850),
    ]),
    ('tensor_api', BASE+'torch/_tensor_docs.py', [
        ('"view",', 1000),
        ('"index_add_",', 1200),
    ]),
    ('clip', BASE+'torch/nn/utils/clip_grad.py', [('def clip_grad_norm_(', 1500)]),
    ('adamw', BASE+'torch/optim/adamw.py', [('AdamW.__doc__', 2450)]),
    ('benchmark', BASE+'torch/utils/benchmark/utils/timer.py', [('class Timer:', 1500)]),
    ('gpu_memory_visibility', BASE+'docs/source/torch_cuda_memory.md', [('Any memory allocated', 800)]),
    ('fusion', 'https://raw.githubusercontent.com/pytorch/tutorials/main/recipes_source/recipes/tuning_guide.py', [('Fuse operations', 1100)]),
    ('python_clock', 'https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/library/time.rst', [('.. function:: perf_counter()', 1200)]),
]
records=[]
for name,url,spans in SPECS:
    completed=subprocess.run(['curl','--fail','--silent','--show-error','--location',url],capture_output=True,check=False)
    record={'id':name,'url':url,'command':['curl','--fail','--silent','--show-error','--location',url], 'accessed_at':datetime.now(UTC).isoformat(),'exit_code':completed.returncode}
    if completed.returncode:
        record['error']=completed.stderr.decode()
    else:
        raw=completed.stdout
        text=raw.decode()
        record['original_sha256']=hashlib.sha256(raw).hexdigest()
        record['original_bytes']=len(raw)
        excerpts=[]
        for marker,length in spans:
            start=text.find(marker)
            excerpts.append({'locator':marker,'found':start>=0,'excerpt':text[start:start+length] if start>=0 else ''})
        record['excerpts']=excerpts
    records.append(record)
DEST.write_text(json.dumps({'sources':records},ensure_ascii=False,indent=2)+'\n')
print(DEST.relative_to(ROOT))
for r in records:
    print(r['id'],r['exit_code'], [(e['locator'],e['found']) for e in r.get('excerpts',[])])
