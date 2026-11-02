"""Immutable-retention identity checks and arithmetic on saved C-v3 observations only."""

import hashlib
import json
import math
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path('/tmp/tpb').resolve()
LEGACY=Path('/workspace/work/omni-book-integrated-v1')
OUT=Path(__file__).resolve().parent
OLD='docs/technical-reviews/artifacts/omni-p6-wave04-curriculum-final/'
PARENT='docs/course-revision-20261101-c-v3/'
ARCHIVE=Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def digest(data): return hashlib.sha256(data).hexdigest()


def locate(path):
    path=Path(path)
    if path.is_absolute(): return path
    if (ROOT/path).is_file(): return ROOT/path
    return LEGACY/path


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(locate(path).read_bytes())


checks=[]


def check_ref(ref,category):
    path=ref.get('outside_git_path',ref.get('cache_path',ref.get('path')))
    p=locate(path); actual=file_sha(p) if p.is_file() else None
    row={'category':category,'recorded_path':path,'physical_path':str(p),'sha256':ref['sha256'],'actual_sha256':actual,'match':actual==ref['sha256']}
    checks.append(row)
    return row


mapping=read(PARENT+'technical-coordination-v21/prior-evidence-reference-map-resume-v1.json')
mapped=next(x for x in mapping['pages'] if x['page_id']=='curriculum')
proof=mapped['prior_technical_proof']
for name in ['original_report','report','original_source_freeze']: assert check_ref(proof[name],name)['match']
prior=read(proof['report']['path']); freeze=read(proof['original_source_freeze']['path'])
page=next(x for x in freeze['pages'] if x['page_id']=='curriculum')
old_body=locate(page['snapshot']).read_bytes(); new_body=(ROOT/'docs/curriculum.md').read_bytes()
assert digest(old_body)==page['source_sha256']==prior['source_sha256']
oldlines=old_body.decode().splitlines(keepends=True); newlines=new_body.decode().splitlines(keepends=True)
blocks=[]
for tag,a,b,c,d in SequenceMatcher(None,oldlines,newlines,autojunk=False).get_opcodes():
    blocks.append({'type':tag,'prior_lines':[a+1,b],'current_lines':[c+1,d],'prior_block_sha256':digest(''.join(oldlines[a:b]).encode()),'current_block_sha256':digest(''.join(newlines[c:d]).encode()),**({'prior_text':''.join(oldlines[a:b]),'current_text':''.join(newlines[c:d])} if tag!='equal' else {})})

# Reconstruct the original complete chapter from verified original intro+sections.
chapter_pages=[p for p in freeze['pages'] if p['source']=='course/chapters/19.md']
intro=[p for p in chapter_pages if p['selector']=='before-first-section']
sections=[p for p in chapter_pages if p['selector'].startswith('section:')]
assert len(intro)==1 and len(sections)==24
pieces=[]; piece_rows=[]
for part in intro+sections:
    data=locate(part['snapshot']).read_bytes(); assert digest(data)==part['source_sha256']
    pieces.append(data);piece_rows.append({'page_id':part['page_id'],'snapshot':part['snapshot'],'sha256':digest(data)})
old19=b''.join(pieces)
guide=next(p for p in freeze['pages'] if p['source']=='course/README.md' and p['selector']=='whole-file')
oldguide=locate(guide['snapshot']).read_bytes(); assert digest(oldguide)==guide['source_sha256']
source_rows=[]
for s in prior['sources']:
    if s['kind']=='repository_code':
        row=check_ref(s,'prior_repository_code')
        if not row['match']:
            assert s['id']=='guide' and digest(oldguide)==s['sha256']
            row['bounded_resolution']='Original guide recovered from SHA-verified source18 whole-file snapshot. Current routing checked independently by this actor; do not retain old guide hash as current.'
        source_rows.append(row)
for a in prior['artifacts']: assert check_ref(a,'prior_artifact')['match']
oldraw=read(OLD+'raw-audit-output.json')
for ref in oldraw['inputs']: check_ref(ref,'prior_raw_input')
download=read(OLD+'primary-download-receipt.json')
inspection=read(OLD+'primary-inspection-receipt.json')
provider=read(OLD+'public-provider-receipt.json')
citation=read(OLD+'source-citation-corrections.json')


def recurse_refs(value,category):
    if isinstance(value,dict):
        if 'sha256' in value and any(k in value for k in ['path','cache_path','outside_git_path']): check_ref(value,category)
        for v in value.values(): recurse_refs(v,category)
    elif isinstance(value,list):
        for v in value: recurse_refs(v,category)


for value,category in [(download,'prior_official_original'),(inspection,'prior_paper_original'),(provider,'prior_provider_original'),(citation,'prior_citation_original')]: recurse_refs(value,category)
# The retained data audit already checked each JSONL member; retain that audit,
# checking the fixed archive identity, without recounting training data again.
archive_path=locate('assets/training/selftrained-v2.tar.gz')
checks.append({'category':'v2_training_package_identity','physical_path':str(archive_path),'sha256':oldraw['data']['archive_sha256'],'actual_sha256':file_sha(archive_path) if archive_path.is_file() else None,'match':archive_path.is_file() and file_sha(archive_path)==oldraw['data']['archive_sha256']})

# Exact source21 CPU output retention for just the eighteen original new lessons.
new_ids=['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
kernel_path=PARENT+'evidence/kernel-refresh-source21-v1.json';kernel=read(kernel_path)
f21=read(PARENT+'source-freeze-v21.json')
kernel_rows=[]
for sid in new_ids:
    page21=next(p for p in f21['pages'] if p['page_id']==sid)
    authored=read(page21['notebook'])
    output_path=LEGACY/'outputs/omni-c-v3-kernel-bundle-v21'/Path(page21['notebook']).relative_to('notebooks')
    executed=json.loads(output_path.read_bytes())
    a=[c['source'] for c in authored['cells'] if c['cell_type']=='code']
    b=[c['source'] for c in executed['cells'] if c['cell_type']=='code']
    code=[c for c in executed['cells'] if c['cell_type']=='code']
    row={'lesson_id':sid,'authored_notebook_sha256':file_sha(locate(page21['notebook'])),'frozen_notebook_sha256':page21['notebook_sha256'],'executed_path':str(output_path),'executed_sha256':file_sha(output_path),'code_source_equal':a==b,'executed_code_cells':sum(c.get('execution_count') is not None for c in code),'code_cells':len(code),'error_outputs':sum(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))}
    assert row['authored_notebook_sha256']==row['frozen_notebook_sha256'] and row['code_source_equal'] and row['code_cells']==row['executed_code_cells'] and row['error_outputs']==0
    kernel_rows.append(row)

# Necessary old university/learner originals, obtained in the previous round.
research_ref={'path':'docs/technical-reviews/artifacts/omni-technical-curriculum-v1/original-source-receipts.json','sha256':'d6f2b9e5ed1746ad428c1633b267d3440ed5e3a737295dbff63da07bc4bd990e'}
assert check_ref(research_ref,'research_receipt')['match']
research=read(research_ref['path']); research_rows=[]
for i in list(range(39,55))+[63]:
    ref=research['receipts'][i]; assert check_ref(ref,'necessary_course_learner_original')['match']
    research_rows.append({'json_pointer':f'/receipts/{i}',**ref})
research_docs=[]
for path in ['docs/public-course-review-2025-2026.md','docs/learner-feedback-review.md']:
    oldpage=next((p for p in freeze['pages'] if p['source']==path),None)
    actual=file_sha(ROOT/path)
    if oldpage:
        data=locate(oldpage['snapshot']).read_bytes(); assert digest(data)==oldpage['source_sha256']
        research_docs.append({'source':path,'prior_snapshot':oldpage['snapshot'],'prior_sha256':digest(data),'current_sha256':actual,'source_identical':actual==digest(data)})
        assert actual==digest(data)
    else:
        research_docs.append({'source':path,'current_sha256':actual,'prior_whole_doc_identity':'Not supplied by canonical source18 freeze. Do not claim this linked whole document was frozen there. The relevant curriculum history paragraphs themselves are byte-identical, and necessary fixed original receipts are directly checked here instead of treating this narrative as authority.'})

# Parse saved logits and manual observations. Never run a model here.
assert ARCHIVE.stat().st_size==6168350 and file_sha(ARCHIVE)=='552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
members={}; payload={}
with zipfile.ZipFile(ARCHIVE) as z:
    def member(name):
        raw=z.read('tiny-perceptron-c-v3/'+name);members[name]={'bytes':len(raw),'sha256':digest(raw)};payload[name]=raw;return json.loads(raw)
    acquisition=read('docs/decision-product/v3/metadata/acquisition.json')
    for ref in acquisition['files']:
        raw=z.read(ref['archive_member']);assert digest(raw)==ref['sha256']==file_sha(ROOT/ref['path']);assert len(raw)==ref['bytes']
    opening=member('evidence/final-opening.json'); observations=member('evidence/final-observations.json'); families=member('evidence/final-families.json'); final=member('evidence/final-results.json'); manual=member('evidence/product-checks.json'); regression=member('evidence/legacy-regression.json'); registry=member('evidence/registry-before-final.json'); selection=member('evidence/selection.json');policy=member('bundle/policy.json');completion=member('evidence/completion.json');lineage=member('evidence/lineage.json')
assert len(observations)==4500 and len(families)==1500
assert opening['weights_sha256']==completion['artifacts']['best.safetensors']=='1090421d703662b5844c9522df628e644d2b052104cbe6ecceb7c689fc68793f'
assert members['evidence/selection.json']['sha256']==opening['selection_sha256']
assert final['final_opening_sha256']==members['evidence/final-opening.json']['sha256']
assert final['observations_sha256']==members['evidence/final-observations.json']['sha256']
assert final['families_sha256']==members['evidence/final-families.json']['sha256']
assert opening['product_checks_sha256']==members['evidence/product-checks.json']['sha256']
assert opening['legacy_regression_sha256']==members['evidence/legacy-regression.json']['sha256']
for name,h in opening['implementation'].items(): assert file_sha(ROOT/'tiny_perceptron/decision_product'/name)==h
for r in [manual['provenance'],regression['provenance']]:
    assert r['weights_sha256']==opening['weights_sha256'] and r['selection_sha256']==opening['selection_sha256']
    for name,h in r['inference_sources'].items(): assert file_sha(ROOT/'tiny_perceptron/decision_product'/name)==h
assert completion['created_at']<selection['created_at']<opening['created_at']<final['created_at']
keys={f['key'] for f in families}; assert len(keys)==1500
exposed={k for entry in registry['entries'] for k in entry['family_keys']}
assert not keys & exposed and len(exposed)==opening['previous_exposed_family_count']
assert {len(f['facts']) for f in families}=={4,5,6}
counts=Counter(); correct=Counter(); per_family=defaultdict(list);accepted=errors=answerable_accepted=0
for row in observations:
    assert row['family'] in keys and len(row['ids'])==len(row['logits'])
    assert 0<=row['label']<len(row['ids']) and row['ids'][row['label']]==row['gold_id']
    logits=[float(v) if v is not None else -math.inf for v in row['logits']]
    best=max(range(len(logits)),key=logits.__getitem__); assert math.isfinite(logits[best])
    counts[row['kind']]+=1;correct[row['kind']]+=best==row['label'];per_family[row['family']].append(row['kind'])
    if row['kind']=='choice':
        t=policy['temperatures']['choice'];peak=logits[best]/t
        confidence=1/sum(math.exp(v/t-peak) for v in logits)
        adopt=row['ids'][best]!='none' and row['eligible'][best] and confidence>=policy['threshold']
        accepted+=adopt;errors+=adopt and best!=row['label'];answerable_accepted+=adopt and row['answerable']
assert counts==correct==Counter(choice=1500,info=1500,rubric=1500)
assert all(sorted(v)==['choice','info','rubric'] for v in per_family.values())
assert accepted==answerable_accepted==1025 and errors==0 and accepted/1500>=0.6
z95=1.6448536269514722;upper=z95*z95/(accepted+z95*z95)
assert upper<0.05
manual_counts=Counter();manual_ok=Counter();states=defaultdict(set)
for row in manual['observations']:
    task,p,r=row['task'],row['permission'],row['ready'];kind=row['kind']
    states[row['devices']].add((task,p,r))
    if kind=='removed': expected='none'
    elif kind=='choice': expected='none' if p is None or r is None else 'stop' if not p else 'wait' if not r else task
    elif kind=='info': expected='yes' if (p is not None and r is not None if task=='record' else p is not None if task=='play' else r is not None) else 'no'
    else: expected='none' if p is None or r is None else str(int(p)+int(r) if task=='record' else 2 if (p if task=='play' else r) else int(r if task=='play' else p))
    assert row['expected']==expected
    predicted=row['prediction']['selected'] or 'none';manual_counts[kind]+=1;manual_ok[kind]+=predicted==expected
assert manual_counts==manual_ok==Counter(choice=972,info=972,rubric=972,removed=432)
assert {n:len(v) for n,v in states.items()}=={3:27,4:27,5:27,6:27}
legacy_counts=Counter();legacy_ok=Counter();legacy_families=set()
for row in regression['observations']:
    legacy_families.add(row['original_family']);legacy_counts[row['kind']]+=1;legacy_ok[row['kind']]+=(row['prediction']['selected'] or 'none')==row['gold']
assert legacy_counts==legacy_ok==Counter(choice=60,info=60,rubric=60) and len(legacy_families)==60
shared_ref={'path':'docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json','sha256':'375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959'}
assert check_ref(shared_ref,'shared_original_execution')['match']
shared=read(shared_ref['path']);sharedcounts=shared['retained_measurement_recalculation'];assert sharedcounts['unique_final_families']==1500 and sharedcounts['choice_accepted']==accepted
unresolved=[row for row in checks if not row['match'] and not row.get('bounded_resolution')]
result={
    'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/c_v3_technical_coord/canonical_curriculum','environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU; SHA/JSON arithmetic only; torch not imported'},
    'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-curriculum/retention-audit.py',
    'prior_review_identity':{'original_report':proof['original_report'],'report':proof['report'],'original_created_at':'2026-10-30T17:34:36.589716+00:00','current_created_at':prior['created_at'],'reviewer_task':prior['reviewer_task'],'raw_execution_created_at':oldraw['created_at'],'source_inspection_created_at':inspection['created_at'],'retention_not_new_execution':True},
    'source_comparison':{'prior_sha256':digest(old_body),'current_sha256':digest(new_body),'blocks':blocks,'semantic_delta_note':'331/357 become332/358, checked independently; OCR/Attention/TTS name expansions keep meaning; first18 scope clarified as historical first batch; old C is historical, newC3 statement separately verified; 19.25 adds current service and read order. All other paragraphs retain identical bytes.'},
    'original_chapter19_reconstruction':{'whole_file_sha256':digest(old19),'bytes':len(old19),'pieces':piece_rows,'method':'Verified original source18 intro and all24 section snapshot hashes, concatenated in freeze order entirely in memory; no guessed Git old whole-file blob.'},
    'dependency_checks':checks,'unresolved_dependencies':unresolved,
    'research_retention':{'receipt':research_ref,'owner':research['reviewer_task'],'rows':research_rows,'docs':research_docs,'scope':'2026-10-01 planning-history record and fixed upstream originals; no new live-site, video-viewing or contemporary course-year status assertion.'},
    'current_cpu_kernel_retention':{'receipt':{'path':kernel_path,'sha256':file_sha(ROOT/kernel_path)},'receipt_created_at':kernel['created_at'],'declared_scope':kernel['scope'],'original_new18':kernel_rows,'claim_scope':'Eighteen original新增節 only, not fresh execution of332 by this actor.'},
    'c3':{'archive':{'path':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':file_sha(ARCHIVE)},'members':members,'opening_created_at':opening['created_at'],'selection_created_at':selection['created_at'],'completion_created_at':completion['created_at'],'final_created_at':final['created_at'],'family_count':1500,'observation_count':4500,'per_kind_counts':dict(counts),'per_kind_correct':dict(correct),'one_per_kind_per_family':True,'device_counts':[4,5,6],'previous_exposed_count':len(exposed),'overlap_with_exposed':0,'accepted_choice':accepted,'accepted_errors':errors,'coverage_all':accepted/1500,'one_sided_wilson95_upper':upper,'policy':policy,'manual_counts':dict(manual_counts),'manual_correct':dict(manual_ok),'manual_local_states_per_device_count':{n:len(v) for n,v in states.items()},'legacy_families':60,'legacy_counts':dict(legacy_counts),'legacy_correct':dict(legacy_ok),'retained_invariance_failures':manual['invariance_failures'],'shared_raw_execution':{**shared_ref,'owner':shared['reviewer_task'],'started_at':shared['started_at'],'completed_at':shared['completed_at']},'scope':'Raw saved labels/logits and manual observations rechecked; no NN inference, training, final opening, full17-check rerun or public transport. 4500 observations are three correlated kinds on1500families;3-device manual/regression is exposed engineering evidence. Invariance0 is retained original measurement, not independently rerun.'},
    'result':'PASS identities and bounded raw arithmetic' if not unresolved else 'UNRESOLVED dependency identities listed',
}
with (OUT/'retention-audit.json').open('x') as target: target.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'result':result['result'],'dependency_checks':len(checks),'unresolved_dependencies':unresolved,'kernel18':len(kernel_rows),'c3':{k:result['c3'][k] for k in ['family_count','observation_count','per_kind_correct','accepted_choice','accepted_errors','coverage_all','manual_counts','legacy_counts']}},ensure_ascii=False,indent=2))
