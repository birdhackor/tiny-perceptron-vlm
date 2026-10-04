"""Fresh isolated CPU rebuild, with durable stdout/results and per-file comparison."""
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/technical-reviews/artifacts'
BASE=Path('/tmp/natural-20.3-rebuild')
assert not BASE.exists(), 'Fresh directory required, do not reuse cached rebuilt data'
manifest=json.loads((ROOT/'docs/natural-assistant/manifest.json').read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(kind):
    cmd=[str(ROOT/'.venv/bin/python'),f'scripts/prepare_natural_{kind}.py','--output',str(BASE/kind)]
    if kind=='ocr': cmd+=['--font-cache','outputs/natural-extension/font-cache','--seed','42']
    result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    record={'kind':kind,'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'generator_sha256':digest(ROOT/f'scripts/prepare_natural_{kind}.py')}
    if result.returncode==0:
        specs={x['path']:x for x in manifest['files'] if x['path'].startswith(kind+'/')}
        names={kind+'/'+p.relative_to(BASE/kind).as_posix():p for p in (BASE/kind).rglob('*') if p.is_file()}
        assert set(names)==set(specs),(kind,'path mismatch')
        checks=[]
        for name,p in sorted(names.items()):
            actual=digest(p); size=p.stat().st_size;spec=specs[name]
            assert size==spec['bytes'] and actual==spec['sha256'],(name,'content mismatch')
            checks.append({'path':name,'bytes':size,'sha256':actual})
        record.update(fresh_output_file_count=len(checks),all_files_byte_identical_to_frozen_snapshot=True,files=checks)
    path=OUT/f'natural-20.3-rebuild-{kind}-execution.json'
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'},ensure_ascii=False),flush=True)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(run,('vision','ocr','speech')))
assert all(x['returncode']==0 for x in results), 'Preparation failed; inspect persisted per-kind records'
summary={'fresh_cpu_rebuild':True,'total_file_count':sum(x['fresh_output_file_count'] for x in results),'all_345_files_match_fixed_manifest':True,'result_paths':[f'docs/technical-reviews/artifacts/natural-20.3-rebuild-{x["kind"]}-execution.json' for x in results],'environment':{'python':sys.version,'pillow':importlib.metadata.version('pillow'),'soundfile':importlib.metadata.version('soundfile'),'device':'CPU; no model inference or GPU use','platform':platform.platform()},'scope':'New network/source rebuild at fixed revisions/generations; selected source files/WAVs only. Does not verify complete upstream FLEURS audio archives, every label, unknown pretraining, or model performance.'}
(OUT/'natural-20.3-rebuild-execution.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False),flush=True)
