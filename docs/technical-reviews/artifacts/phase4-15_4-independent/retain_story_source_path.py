"""Narrow evidence retention check by the original15.4 reviewer; no model runs."""
import copy
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-15_4-independent'
TASK='/root/phase4_factual_coordinator/factual_15_4'
INITIAL='712c98e7407c8f69cf02598ec770e8a20c7cf97d04348122c2e041c645d4fc84'
DATA_SHA='7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0'
CACHE='data/training/text-initial/tinystories-train-512.jsonl'
PERMANENT='docs/technical-reviews/artifacts/phase4-15_4-independent/inputs/tinystories-original-512.jsonl'
COMMAND='.venv/bin/python docs/technical-reviews/artifacts/phase4-15_4-independent/retain_story_source_path.py'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

report_path=ROOT/'docs/technical-reviews/15.4.json'
raw_report=report_path.read_bytes()
assert digest(raw_report)==INITIAL
report=json.loads(raw_report)
assert report['reviewer_task']==TASK and report['verdict']=='pass'
before=copy.deepcopy(report)
cache=(ROOT/CACHE).read_bytes();permanent=(ROOT/PERMANENT).read_bytes()
assert cache==permanent and digest(cache)==digest(permanent)==DATA_SHA
assert not (ROOT/CACHE).is_symlink() and not (ROOT/PERMANENT).is_symlink()

# Preserve exactly the initial report bytes; do not normalize or transplant it.
timestamp=datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
history_dir=OUT/'history';history_dir.mkdir(exist_ok=True)
history=history_dir/('15.4-initial-pass-'+INITIAL[:16]+'-'+timestamp+'.json')
with history.open('xb') as stream:stream.write(raw_report)
assert digest(history.read_bytes())==INITIAL

source=next(s for s in report['sources'] if s['id']=='original-stories')
assert source['path']==CACHE and source['sha256']==DATA_SHA
source['path']=PERMANENT
source['original_input_provenance']={
    'cache_path':CACHE,'sha256':DATA_SHA,'bytes':len(cache),
    'scope':'当次原输入cache定位；不作为fresh Git checkout所需证据路径。正式source path是逐byte相同的永久副本。',
}

receipt_id='source-path-retention-receipt'
receipt_path=OUT/('source-path-retention-receipt-'+timestamp+'.json')
receipt={
    'reviewer_task':TASK,'checked_at_utc':datetime.now(UTC).isoformat(),
    'kind':'narrow_source_path_retention_check',
    'command':COMMAND,'python':sys.version,'device':'CPU file-byte/hash inspection only',
    'old_report':{'canonical_path':'docs/technical-reviews/15.4.json','initial_pass_sha256':INITIAL,'opaque_history_path':history.relative_to(ROOT).as_posix(),'history_sha256':digest(history.read_bytes())},
    'checked_inputs':[
        {'role':'original-cache-provenance-only','path':CACHE,'sha256':digest(cache),'bytes':len(cache),'symlink':False},
        {'role':'formal-permanent-source','path':PERMANENT,'sha256':digest(permanent),'bytes':len(permanent),'symlink':False},
    ],
    'bytes_equal':True,
    'original_source_support_scope':'原review使用的512篇原故事输入逐byte保留。原CPU验证已仅使用永久copy重建seed42验证集51篇的SHA与39256有效输入位置；这次只确认原输入cache与正式copy相同，不重新训练、重评或重算科学测量。',
    'source_change':{'source_id':'original-stories','old_path_provenance_only':CACHE,'new_formal_path':PERMANENT,'sha256_unchanged':DATA_SHA},
    'unchanged':{
        'source_sha256':before['source_sha256'],
        'claims_json_sha256':digest(json.dumps(before['claims'],ensure_ascii=False,sort_keys=True).encode()),
        'verdict':before['verdict'],'figure_sha256':before['figure_sha256'],
        'checks_unchanged':True,'original_fence_sha256':'b447939f95a05b48aef7c64bdcc9fb46168e02a7172f3d2424d285fd0d0f96c8',
    },
    'work_scope':'只改本人source路径并追加本人receipt/history/code留存证据；不改教材、图、verdict、claims，不force-add ignored训练数据，不重新生成数据或运行CPU模型检查。',
}
write_json(receipt_path,receipt)

for identifier,path,kind,description in (
    ('source-path-retention-code',Path(__file__),'code','Original reviewer narrow source-path retention code; compares actual bytes/SHA and preserves initialPASS history before own source-only mutation.'),
    (receipt_id,receipt_path,'execution','Original reviewer personally checked original cache and already-preserved raw source copy byte equality/SHA; records exact support scope and unchanged scientific claims.'),
    ('initial-pass-opaque-history',history,'source_snapshot','Exact initialPASS report bytes preserved opaquely as actual history; not substituted for current substantive evidence.'),
):
    assert not any(a['id']==identifier for a in report['artifacts'])
    artifact={'id':identifier,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path.read_bytes()),'kind':kind,'description':description}
    if kind=='execution':artifact.update(command=COMMAND,result='Exit0; two593489-byte inputs are byte-identical with SHA7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0; initialPASS opaque history preserved; source path updated only.',environment={'python':sys.version,'python_executable':sys.executable,'device':'CPU file-byte/hash inspection only','cwd':str(ROOT)})
    report['artifacts'].append(artifact)

report['path_retention_review']={'reviewer_task':TASK,'receipt_id':receipt_id,'receipt_path':receipt_path.relative_to(ROOT).as_posix(),'initial_pass_opaque_history':history.relative_to(ROOT).as_posix(),'scope':'Formal source path retention only; all scientific claims/verdict unchanged.'}
for field in ('source_sha256','verdict','claims','checks','figure_sha256','issues','reviewer_task'):
    assert report[field]==before[field],field
write_json(report_path,report)
after=json.loads(report_path.read_text())
assert after['reviewer_task']==TASK
for field in ('source_sha256','verdict','claims','checks','figure_sha256','issues'):
    assert after[field]==before[field],field
print(json.dumps({'reviewer_task':TASK,'new_report_sha256':digest(report_path.read_bytes()),'initial_pass_sha256':INITIAL,'history_path':history.relative_to(ROOT).as_posix(),'receipt_id':receipt_id,'receipt_path':receipt_path.relative_to(ROOT).as_posix(),'unchanged_source_sha256':after['source_sha256'],'verdict_unchanged':after['verdict'],'bytes_equal':True,'source_sha256':DATA_SHA},ensure_ascii=False,indent=2))
