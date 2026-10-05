"""Recheck narrow current navigation dependencies and immutable raw citation."""
from pathlib import Path
from datetime import UTC, datetime
import hashlib
import json
import re
import sys

OUT=Path(__file__).resolve().parent
OWN=OUT.parent
ROOT=OUT.parents[4]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def section(raw,key):
    heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    i=next(i for i,h in enumerate(heads) if h[0].startswith(('## '+key+' ').encode()))
    return raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)]

history=next(OUT.glob('prior-pass-R.4-*.opaque.json'))
prior=json.loads(history.read_bytes())
source=next(s for s in prior['sources'] if s['id']=='nav-training')
assert prior['verdict']=='pass'
assert source['path']=='course/training.md'
frozen=OWN/'frozeninput-course-training.md'
assert sha(frozen)==source['sha256']=='22f3115b0d09a83d82342e82c101df8b8516aa5de7796b137a5243c213396324'
assert frozen.read_bytes()==(OWN/'recheck-20261005/frozeninput-course-training.md').read_bytes()
current_training=(ROOT/'course/training.md').read_bytes()
assert current_training==(OUT/'frozeninput-current-course-training.md').read_bytes()
assert sha(ROOT/'course/training.md')!=source['sha256']
matches=[]
for key in ('T.3','T.11'):
    current=section(current_training,key)
    old=section(frozen.read_bytes(),key)
    assert current==old==(OUT/('current-'+key+'.md')).read_bytes()
    assert current==(OWN/('navigation-source-'+key+'.md')).read_bytes()
    assert current==(OWN/'recheck-20261005'/('current-'+key+'.md')).read_bytes()
    matches.append({'section':key,'current_equals_personally_read_frozen_section':True,'section_sha256':hashlib.sha256(current).hexdigest()})
r4=section((ROOT/'course/README.md').read_bytes(),'R.4')
assert r4==(OUT/'current-R.4.md').read_bytes()==(OWN/'recheck-20261005/current-R.4.md').read_bytes()
assert hashlib.sha256(r4).hexdigest()==prior['source_sha256']
assert '各局部實驗的操作見[操作頁](training.md)' in r4.decode()
assert 'scripts/fetch_course_models.py --model text_foundation' in section(current_training,'T.3').decode()
assert '--experiment simple_models --device cpu' in section(current_training,'T.3').decode()
assert '其能力表和交付指引與局部實驗分開' in section(current_training,'T.11').decode()
unchanged=[]
for source_item in prior['sources']:
    if source_item['kind']=='repository_code' and source_item['id']!='nav-training':
        assert sha(ROOT/source_item['path'])==source_item['sha256']
        unchanged.append(source_item['id'])
environment={'python':sys.version,'python_executable':sys.executable,'device':'cpu (UTF-8 bytes/navigation only)','scope':'No network, weights, datasets, training, inference, scientific-claim expansion or unrelated CPU rerun.'}
observation={'reviewer_task':prior['reviewer_task'],'inspected_on_utc':datetime.now(UTC).isoformat(),'read_scope':['complete current course/README.md#R.4','current course/training.md#T.3','current course/training.md#T.11','previously personally read raw frozen training snapshots: verified whole bytes/sourceSHA, relevant T.3/T.11 slices reread'],'prior_pass_history':{'path':str(history.relative_to(ROOT)),'sha256':sha(history)},'source_sha256':prior['source_sha256'],'historical_frozen_navigation_input':{'path':str(frozen.relative_to(ROOT)),'sha256':sha(frozen),'saved_in_review_on':'2026-10-05','meaning':'Actual original raw input previously personally saved. Its historical SHA must refer to this immutable snapshot, never mutable course/training.md.'},'current_whole_navigation_input_frozen_now':{'path':str((OUT/'frozeninput-current-course-training.md').relative_to(ROOT)),'sha256':sha(OUT/'frozeninput-current-course-training.md'),'meaning':'Actual current whole bytes saved for this inspection; only named navigation scope personally read.'},'necessary_navigation_scope_matches':matches,'other_original_repository_evidence_hashes_still_exact':unchanged,'finding':'Current operation-page whole file changed, but both actual necessary navigation ranges are byte-identical and were personally reread. R.4 and its original method claims are unchanged. The fixed source citation will bind the original historical rawSHA to its existing permanent frozeninput; current-scope equality is separately supported by this actual check. No change to technical judgments or removal of meaningful claims.'}
(OUT/'portability-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
(OUT/'portability-observations.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observation,ensure_ascii=False,indent=2))
print('All narrow current-scope and immutable-navigation-citation assertions passed.')
