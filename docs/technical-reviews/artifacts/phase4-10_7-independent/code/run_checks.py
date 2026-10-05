"""Save exact commands, outputs and exit codes of this reviewer's actual checks."""
import json, os, subprocess, hashlib, sys
from pathlib import Path
from datetime import datetime, UTC
ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parents[1]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
     'TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
tasks=[
    ('standalone-original',[str(ROOT/'.venv/bin/python'),'-c',"from pathlib import Path; exec(compile(Path('docs/technical-reviews/artifacts/phase4-10_7-independent/original-execution/fence-1.py').read_bytes(),'original-10.7-fence','exec'))"]),
    ('bounded-checks',[str(ROOT/'.venv/bin/python'),str(ART/'code/bounded_checks.py')]),
    ('browser-render',[str(ROOT/'.venv/bin/python'),str(ART/'code/render_section.py')]),
]
records=[]
for name,command in tasks:
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,timeout=45)
    (ART/f'{name}.stdout.txt').write_bytes(result.stdout)
    (ART/f'{name}.stderr.txt').write_bytes(result.stderr)
    record=dict(name=name,argv=command,cwd=str(ROOT),exit_code=result.returncode,timeout_seconds=45,
                stdout_path=f'{name}.stdout.txt',stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
                stderr_path=f'{name}.stderr.txt',stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
    records.append(record)
    print(name,result.returncode)
receipt=dict(executed_at=datetime.now(UTC).isoformat(),runner_python=sys.version,commands=records,
             declared_environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS']})
(ART/'execution-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(any(r['exit_code'] for r in records))
