"""Informed callback for freeze15; retain original evidence, no download/test rerun."""
import difflib
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.inference import check_manifest

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT/'docs/technical-reviews/artifacts/revision-20261016/backfill'
WORK = Path('/workspace/work/curriculum-20261016/technical-19.19')
PYTHON = str(ROOT/'.venv/bin/python')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
original_path = ROOT/'docs/course-revision-20261016/reviews/technical-19.19-backfill-original-01.json'
original = json.loads(original_path.read_text())
freeze_path = ROOT/'docs/course-revision-20261016/freeze-15.json'
freeze = json.loads(freeze_path.read_text())
entry = next(p for p in freeze['pages'] if p['page_id']=='19.19')
body = lesson_slices((ROOT/'course/chapters/19.md').read_text())['19.19']
old_body = (ROOT/original['source_freeze']['snapshot']).read_text()
snapshot = ROOT/entry['snapshot']
result = {
    'reviewer_task': '/root/backfill_tech_19_19', 'current_context': 'informed',
    'scope': 'Full new lesson and two hunks read, dependency/evidence hashes checked, three bounded CPU inference commands and two help commands. No repeated download, HTTP, training, formal-test inference or remote writes.',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'torch_git': torch.version.git_version, 'platform': platform.platform(), 'device': 'cpu'},
    'original_report': {'path': str(original_path.relative_to(ROOT)), 'sha256': sha(original_path)},
    'new_source': {'freeze': str(freeze_path.relative_to(ROOT)), 'freeze_sha256': sha(freeze_path), 'snapshot': entry['snapshot'], 'snapshot_sha256': sha(snapshot), 'current_slice_sha256': hashlib.sha256(body.encode()).hexdigest(), 'full_page_reread': True, 'same_as_snapshot': body==snapshot.read_text(), 'figures': entry['figures_sha256']},
    'diff': ''.join(difflib.unified_diff(old_body.splitlines(True), body.splitlines(True), fromfile='freeze14/19.19', tofile='freeze15/19.19')),
    'source_dependencies': [], 'original_artifacts': [], 'runs': [],
}
assert result['new_source']['same_as_snapshot']
assert result['new_source']['current_slice_sha256']==entry['source_sha256']
for source in original['sources']:
    if source['kind']=='repository_code':
        current = sha(ROOT/source['path'])
        record = {'source_id': source['id'], 'path': source['path'], 'original_sha256': source['sha256'], 'current_sha256': current, 'same_bytes': current==source['sha256']}
        result['source_dependencies'].append(record)
        assert record['same_bytes']
for artifact in original['artifacts']:
    current = sha(ROOT/artifact['path'])
    record = {'artifact_id': artifact['id'], 'path': artifact['path'], 'original_sha256': artifact['sha256'], 'current_sha256': current, 'same_bytes': current==artifact['sha256']}
    result['original_artifacts'].append(record)
    assert record['same_bytes']
seal = json.loads((BASE/'19.19-original-seal.json').read_text())
result['original_seal_matches'] = all(sha(ROOT/path)==item['sha256'] for path,item in seal['files'].items())
assert result['original_seal_matches']
result['models'] = {}
release = json.loads((ROOT/'tiny_perceptron/lesson_products/release.json').read_text())
for product in ('b','c'):
    folder=WORK/'fresh-products'/product
    manifest=check_manifest(folder,product)
    digest=sha(folder/'artifact-manifest.json')
    assert digest==release['products'][product]['manifest_sha256']
    result['models'][product] = {'directory':str(folder),'manifest_sha256':digest,'matches_original_release':True,'verified_files':manifest['files']}
raw = ROOT/'outputs/lesson-products/final-report-v1.predictions.jsonl'
prior = json.loads((BASE/'19.19-existing-evidence.json').read_text())
result['raw_evidence_reuse'] = {'path':str(raw.relative_to(ROOT)),'original_sha256':prior['existing_error_checks']['raw_sha256'],'current_sha256':sha(raw),'rescored_again':False}
assert result['raw_evidence_reuse']['original_sha256']==result['raw_evidence_reuse']['current_sha256']

wrapper = """import runpy,socket,sys,urllib.request
attempts=[]
def denied(*args,**kwargs):
    attempts.append('network')
    raise RuntimeError('19.19 callback offline guard: network disabled')
urllib.request.urlopen=denied
socket.create_connection=denied
socket.socket.connect=denied
sys.argv=['tiny_perceptron.lesson_products']+sys.argv[1:]
try:
    runpy.run_module('tiny_perceptron.lesson_products',run_name='__main__')
finally:
    print('OFFLINE_NETWORK_ATTEMPTS='+str(len(attempts)),file=sys.stderr)
"""
bargs=['--b',str(WORK/'fresh-products/b')]
cargs=['--c',str(WORK/'fresh-products/c')]
state='兩盒追蹤;A=2;B=6;事件=交換;查詢=B'
for args in [
    ['solve','--help'], ['auto','--help'],
    ['solve',*bargs,'--state',state,'--loops','3'],
    ['auto',*bargs,*cargs,'--state',state,'--effort','medium'],
    ['auto',*bargs,*cargs,'--state',state,'--effort','medium','--max-loops','2'],
]:
    command=[PYTHON,'-c',wrapper,*args]
    p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=25)
    result['runs'].append({'command':command,'cwd':str(ROOT),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'offline_guard':True})
    assert p.returncode==0
    assert 'OFFLINE_NETWORK_ATTEMPTS=0' in p.stderr
solve=json.loads(result['runs'][2]['stdout'])
medium=json.loads(result['runs'][3]['stdout'])
cap=json.loads(result['runs'][4]['stdout'])
assert solve['answer']==2 and solve['loops']==3
assert medium['requested_loops']==3 and medium['b_loops']==3 and medium['status']=='answered'
assert cap['requested_loops']==3 and cap['b_loops']==2 and cap['status']=='budget_limited'
result['conclusion']={'issue_id':'19.19-I1','resolved':True,'reason':'Full new page now assigns loops to independent B and effort to Auto; second hunk explicitly scopes medium/max-loops examples to Auto. Parser/help and actual CPU commands agree. Original solve --effort failure remains valid evidence of the API boundary, not a remaining contradiction.'}
out=BASE/'19.19-callback-execution.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'new_source_sha256':entry['source_sha256'],'unchanged_repo_dependencies':len(result['source_dependencies']),'unchanged_original_artifacts':len(result['original_artifacts']),'runs':len(result['runs']),'issue_resolved':result['conclusion']['resolved']},ensure_ascii=False))
