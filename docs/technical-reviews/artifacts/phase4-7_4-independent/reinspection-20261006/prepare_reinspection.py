"""Same-owner current-input reinspection; no model execution or external source refetch."""
from pathlib import Path
import json
import re
import hashlib
import difflib
import subprocess
import sys
import importlib.metadata
from datetime import UTC, datetime

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-7_4-independent'
OUT=BASE/'reinspection-20261006'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

reportpath=ROOT/'docs/technical-reviews/7.4.json'
priorraw=reportpath.read_bytes()
prior=json.loads(priorraw)
priorhash=hashlib.sha256(priorraw).hexdigest()
history=ROOT/('docs/technical-reviews/history/phase4-7_4-own-before-reinspection-'+priorhash+'.json')
assert history.is_file() and history.read_bytes()==priorraw
raw=(ROOT/'course/chapters/07.md').read_bytes()
headers=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
def section(sid):
    i=next(i for i,x in enumerate(headers) if x[0].startswith(b'## '+sid.encode()+b' '))
    return raw[headers[i].start():headers[i+1].start() if i+1<len(headers) else len(raw)]
current=section('7.4')
old=(BASE/'original/section.md').read_bytes()
assert hashlib.sha256(current).hexdigest()=='d10598ceb807b31b5480617aa75c6c224c7e1376617e522a9555e712ec3812ca'
assert old.replace('接着'.encode(),'接著'.encode())==current
(OUT/'current-section.md').write_bytes(current)
(OUT/'current-context-7.3.md').write_bytes(section('7.3'))
(OUT/'current-vs-own-frozen.diff').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),current.decode().splitlines(True),fromfile='own-frozen-7.4',tofile='current-7.4')))
(OUT/'current-factual-reviewer-instructions.md').write_bytes((ROOT/'docs/review-tools/factual-reviewer-instructions.md').read_bytes())
(OUT/'current-pyproject.toml').write_bytes((ROOT/'pyproject.toml').read_bytes())
(OUT/'pyproject-vs-own-frozen.diff').write_text(''.join(difflib.unified_diff((BASE/'inputs/pyproject.toml').read_text().splitlines(True),(ROOT/'pyproject.toml').read_text().splitlines(True),fromfile='own-frozen-pyproject',tofile='current-pyproject')))
fence=re.findall(rb'(?m)^```python\n(.*?)^```',current,re.S)
assert len(fence)==1 and fence[0]==(BASE/'original/fence-1.py').read_bytes()
code=[]
for name in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','scripts/build_course.py','scripts/check_technical_reviews.py']:
    row={'path':name,'current_sha256':sha(ROOT/name),'own_frozen_sha256':sha(BASE/'inputs'/name)}
    row['unchanged']=row['current_sha256']==row['own_frozen_sha256']
    assert row['unchanged']
    code.append(row)
artifacts=[]
for a in prior['artifacts']:
    actual=sha(ROOT/a['path'])
    assert actual==a['sha256'],a['id']
    artifacts.append({'artifact_id':a['id'],'path':a['path'],'prior_sha256':a['sha256'],'current_sha256':actual,'unchanged':True})
fig=ROOT/'course/figures/rewrite-07-04-answer-alignment.svg'
assert sha(fig)==prior['figure_sha256'][str(fig.relative_to(ROOT))]
(OUT/'current-figure.svg').write_bytes(fig.read_bytes())
command=['inkscape',str(fig),'--export-type=png','--export-width=1280','--export-filename='+str(OUT/'current-figure.png')]
render=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=30,check=False)
(OUT/'figure-render.stdout.txt').write_bytes(render.stdout)
(OUT/'figure-render.stderr.txt').write_bytes(render.stderr)
assert render.returncode==0
record={'kind':'same_original_technical_owner_input_comparison','reviewer_task':prior['reviewer_task'],
 'created_at_utc':datetime.now(UTC).isoformat(), 'prior_report':{'path':str(history.relative_to(ROOT)),'sha256':priorhash},
 'current_section':{'path':str((OUT/'current-section.md').relative_to(ROOT)),'sha256':hashlib.sha256(current).hexdigest()},
 'own_prior_section':{'path':str((BASE/'original/section.md').relative_to(ROOT)),'sha256':hashlib.sha256(old).hexdigest()},
 'current_context_7_3':{'path':str((OUT/'current-context-7.3.md').relative_to(ROOT)),'sha256':sha(OUT/'current-context-7.3.md')},
 'exact_change':'Only 接着→接著, in the exercise paragraph; replacing those exact UTF-8 bytes in the original reproduces current section byte-for-byte.',
 'changed_substantive_claim_ids':[], 'fence':{'number':1,'unchanged':True,'sha256':hashlib.sha256(fence[0]).hexdigest()},
 'current_code_comparison':code,'prior_evidence_rehash':artifacts,
 'figure':{'path':str(fig.relative_to(ROOT)),'sha256':sha(fig),'unchanged':True,
           'render_command_argv':command,'actual_exit_code':render.returncode,'render_path':str((OUT/'current-figure.png').relative_to(ROOT))},
 'environment':{'python':sys.version,'python_executable':sys.executable,'torch_distribution':importlib.metadata.version('torch'),
                'playwright_distribution':importlib.metadata.version('playwright')},
 'execution_policy':'No current model/fence rerun: original fence, relevant code, original source snapshots and execution output hashes unchanged; only spelling changed. Actual new work: firsthand current section/context/source-diff reading, hash comparison, figure/page render and view, owner receipt, single-section checker.'}
write(OUT/'input-comparison.json',record)
print(json.dumps({'source_sha256':record['current_section']['sha256'],'substantive_changes':[],
 'prior_report':record['prior_report'],'original_artifacts_rehashed':len(artifacts),'render_exit_code':render.returncode},ensure_ascii=False,indent=2))
