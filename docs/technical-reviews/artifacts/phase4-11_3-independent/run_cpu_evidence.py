"""Capture exact process status and provenance for the two bounded checks."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
records=[]
for name,stdout_name,stderr_name in [('check_cpu.py','cpu.stdout.json','cpu.stderr.txt'),('check_historical.py','historical.stdout.txt','historical.stderr.txt')]:
    command=[str(ROOT/'.venv/bin/python'),str(OUT/name)]
    with (OUT/stdout_name).open('wb') as stdout,(OUT/stderr_name).open('wb') as stderr:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=30,check=False)
    row={'argv':command,'cwd':str(ROOT),'exit_code':process.returncode,'timeout_seconds':30,'code_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'stdout_path':str((OUT/stdout_name).relative_to(ROOT)),'stdout_sha256':hashlib.sha256((OUT/stdout_name).read_bytes()).hexdigest(),'stderr_path':str((OUT/stderr_name).relative_to(ROOT)),'stderr_sha256':hashlib.sha256((OUT/stderr_name).read_bytes()).hexdigest(),'environment_override_names_and_nonsensitive_values':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'python':sys.version}
    records.append(row)
    assert process.returncode==0,row
(OUT/'cpu-command-receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'script':Path(r['argv'][1]).name,'exit_code':r['exit_code']} for r in records]))
