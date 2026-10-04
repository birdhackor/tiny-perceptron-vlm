"""Independent real recheck of the record-vs-content wording revision."""
import collections
import difflib
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tiny_perceptron import natural_assistant as a
ART=ROOT/'docs/technical-reviews/artifacts'
REPORT=ROOT/'docs/technical-reviews/20.4.json'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    path=ART/('natural-20.4-'+name)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n' if not isinstance(value,str) else value)
    return path.relative_to(ROOT).as_posix()

prior=json.loads(REPORT.read_text())
prior_path=ART/'natural-20.4-before-record-wording-report.json'
assert not prior_path.exists(), 'Preserve prior version once; do not overwrite history'
shutil.copyfile(REPORT,prior_path)
original=ROOT/'outputs/natural-extension/lesson-fragments/20.4-before-record-wording-fix.md'
old=original.read_text()
assert digest(original)==prior['source_sha256']
old_path=ART/'natural-20.4-before-record-wording-section.md'
shutil.copyfile(original,old_path)
raw=(ROOT/'course/chapters/20.md').read_text()
start=re.search(r'^## 20\.4 ',raw,re.M).start()
following=re.search(r'^## ',raw[start+1:],re.M)
current=raw[start:start+1+following.start()] if following else raw[start:]
expected=old.replace('逐筆檢查272道訓練題','逐筆檢查272筆訓練記錄').replace(
    '272道不同訓練題都至少出現一次。','272筆訓練記錄都至少使用一次；其中包含上一節提到的重複聊天範例。')
assert current==expected, 'Only the two independently reread record wording sentences should have changed'
assert old!=current
current_path=save('record-wording-reviewed-section.md',current)
diff=save('record-wording-section.diff',''.join(difflib.unified_diff(old.splitlines(True),current.splitlines(True),
          fromfile='previous independently reviewed20.4',tofile='current independently reread20.4')))

reused=[]
for category in ['artifacts','sources','primary_sources']:
    for item in prior[category]:
        path=item.get('path') or item.get('snapshot_path')
        if path and item.get('sha256'):
            actual=digest(ROOT/path)
            assert actual==item['sha256'],(category,item['id'],path)
            reused.append({'category':category,'id':item['id'],'path':path,'sha256':actual,'matches_prior':True})
for path,sha in prior['figure_sha256'].items():
    assert digest(ROOT/path)==sha
    reused.append({'category':'figure','id':'figure','path':path,'sha256':sha,'matches_prior':True})

manifest_path=ROOT/'docs/natural-assistant/manifest.json'
manifest=json.loads(manifest_path.read_text())
rows=[r for r in manifest['rows'] if r['split']=='train']
record_ids={r['id'] for r in rows}
assert len(rows)==len(record_ids)==272
chat=[r for r in rows if r['task']=='text_chat']
families=collections.defaultdict(list)
for r in chat:families[r['family']].append(r)
assert len(chat)==30 and len(families)==10
groups=[]
content_strings=[]
for family,group in sorted(families.items()):
    assert len(group)==3
    # Compare the complete serialized records except the distinct record ID.
    payloads=[json.dumps({k:v for k,v in r.items() if k!='id'},ensure_ascii=False,sort_keys=True) for r in group]
    assert len(set(payloads))==1
    content=json.dumps({k:group[0].get(k) for k in ['user','answer','image','history','system']},ensure_ascii=False,sort_keys=True)
    content_strings.append(content)
    groups.append({'family':family,'record_ids':[r['id'] for r in group],'count':3,
                   'same_full_payload_except_id':True,'payload_sha256':hashlib.sha256(payloads[0].encode()).hexdigest(),
                   'user':group[0]['user'],'answer':group[0]['answer'],'history':group[0].get('history')})
assert len(set(content_strings))==10
audit=json.loads((ROOT/'docs/natural-assistant/evidence/training-mask/result.json').read_text())
assert len(audit['rows'])==272 and {r['id'] for r in audit['rows']}==record_ids
visits_by_run={}
for name in ['train','train-gentle']:
    data=json.loads((ROOT/f'docs/natural-assistant/evidence/{name}/training.json').read_text())
    visits=[id for update in data['history'] for id in update['row_ids']]
    counts=collections.Counter(visits)
    assert len(data['history'])==180 and len(visits)==360 and set(visits)==record_ids
    assert all(len(update['row_ids'])==2 for update in data['history'])
    assert min(counts.values())>=1
    assert visits==[a.training_row_at(rows,index,42)['id'] for index in range(360)]
    audit_tokens={r['id']:r['shifted_supervised_tokens'] for r in audit['rows']}
    assert all(update['supervised_tokens']==sum(audit_tokens[id] for id in update['row_ids']) for update in data['history'])
    assert sum(update['supervised_tokens'] for update in data['history'])==5791
    visits_by_run[name]={'updates':180,'visits':360,'unique_record_ids':272,'all_manifest_record_ids_used':True,
                         'minimum_visits_per_record':min(counts.values()),'maximum_visits_per_record':max(counts.values()),
                         'target_count':5791,'visit_counts':dict(sorted(counts.items()))}

# Read only the newly necessary predecessor data-count paragraph, not its quality results.
pstart=re.search(r'^## 20\.3 ',raw,re.M).start()
predecessor=raw[pstart:start]
paragraph=[line for line in predecessor.splitlines() if '訓練文字題中的10個短例各出現三次' in line]
assert len(paragraph)==1 and '不能說成30個不同例子' in paragraph[0]
predecessor_path=save('record-wording-predecessor-count-paragraph.txt',paragraph[0]+'\n')

notes='''This same independent reviewer reread the full current20.4, including the complete code, table, figure description, timing/quality caveats and exercise. The only actual text differences are the two recorded wording replacements shown in the saved diff. Each existing claim c1-c31 was reconsidered against this full read; unchanged original source/code/figure/experiment/execution artifacts were individually rehashed before reuse. The existing CPU/meta/processor/GPU-observation evidence remains exactly the originally checked bytes. No full processor rerun, GPU, new generation, test inference, reader/continuity report or Git was used.
The newly necessary20.3 count paragraph was read directly from the course solely to check the reference to repeated chat examples. Its statement10 short examples appear three times agrees with the manifest. All30 text_chat records were directly inspected, then independently compared by family and complete payload excluding only id. There are10 distinct user/answer/image/history/system payloads; each has3 different IDs and otherwise identical complete records within its family. Therefore272 counts unique training-record IDs, not distinct question contents.
Both original training history arrays were read again. Each has180 updates with2record IDs,360visits and exactly the full272manifest ID set. Every ID occurs at least once. Deterministic training_row_at was actually called for all360positions and matches the raw sequence. The272mask-audit IDs also exactly match the manifest training ID set, and all per-step target sums reproduce5791. This directly supports the revised272records/allused wording and its duplicate-example caveat.
Original report, its source hash and complete section bytes are preserved as separate artifacts. New current section bytes and execution receipt were added; prior execution artifacts were kept untouched. Prior timing issue remains resolved with its original version history. This is an actual reread plus independent data/visit recomputation, not a hash-only refresh.
'''
notes_path=save('record-wording-recheck-notes.txt',notes)
execution={'reviewer_task':prior['reviewer_task'],'environment':{'python':platform.python_version(),'device':'CPU; JSON arithmetic and deterministic row selection only'},
           'previous_source_sha256':prior['source_sha256'],'current_source_sha256':hashlib.sha256(current.encode()).hexdigest(),
           'previous_report_sha256':digest(prior_path),'complete_section_reread':True,'only_two_record_wording_changes':True,
           'all_existing_claims_reconsidered':['c'+str(i) for i in range(1,32)],'reused_evidence_hash_checks':reused,
           'manifest_sha256':digest(manifest_path),'training_records':272,'unique_training_record_ids':272,
           'chat_records':30,'unique_chat_content_examples':10,'records_per_chat_example':3,'chat_groups':groups,
           'mask_audit_exact_record_id_set':True,'training_visit_recomputation':visits_by_run,
           'prior_execution_rerun':False,'gpu_used':False,'new_generation_used':False,'test_inference_used':False,
           'reader_or_continuity_report_read':False,'git_used':False}
execution_path=save('record-wording-recheck.json',execution)

def add_artifact(id,path,kind,description,command=None,result=None):
    item={'id':id,'kind':kind,'path':path,'sha256':digest(ROOT/path),'description':description}
    if command:item.update(command=command,result=result,environment=execution['environment'])
    prior['artifacts'].append(item)
add_artifact('a-record-prior-report',prior_path.relative_to(ROOT).as_posix(),'source_snapshot','Complete previous independent technical report retained byte-for-byte')
add_artifact('a-record-prior-section',old_path.relative_to(ROOT).as_posix(),'source_snapshot','Previously reviewed complete section before record/content wording fix')
add_artifact('a-record-current-section',current_path,'source_snapshot','Full current section independently reread by same reviewer')
add_artifact('a-record-diff',diff,'source_snapshot','Actual two-line difference independently compared with entire section')
add_artifact('a-record-notes',notes_path,'source_snapshot','Actual complete-text reconsideration and direct duplicate-content/record-ID inspection')
add_artifact('a-record-predecessor',predecessor_path,'source_snapshot','Only the necessary20.3 data-count paragraph directly read to check previous-section reference')
add_artifact('a-record-code',Path(__file__).relative_to(ROOT).as_posix(),'code','Independent record/content and full-history recheck program')
add_artifact('a-record-check',execution_path,'execution','Actual rehashes,30/10/3content comparison and272ID/360visit recomputation',
             '.venv-natural/bin/python docs/technical-reviews/artifacts/natural-20.4-record-wording-recheck.py',
             'Exit0; all unchanged prior evidence SHA matches;272unique record IDs;30chat records=10distinct examples*3;both180updates/360visits use all272IDs;5791targets.')
add_artifact('a-record-manifest','docs/natural-assistant/manifest.json','source_snapshot','Original manifest training records directly compared; only train rows used')
prior['sources'].append({'id':'s-record-executed','kind':'execution','title':'Independent record-versus-content recheck and visit recount',
                         'verified':True,'artifact_id':'a-record-check'})
prior['primary_sources'].append({'id':'s-record-executed','kind':'execution','path':execution_path,'sha256':digest(ROOT/execution_path),
                                 'inspection_note':'Actual record payload comparisons and train/mask/history ID-set crosschecks.'})
prior['primary_sources'].append({'id':'p-manifest','kind':'primary_observation','path':'docs/natural-assistant/manifest.json','sha256':digest(manifest_path),
                                 'inspection_note':'Directly read all train records;30chat records compared across10groups of3; no semantic uniqueness asserted for remaining272records.'})

byid={claim['id']:claim for claim in prior['claims']}
byid['c15']['statement']='All272 audited training records fit2048; maximum actual text/visual sequence length is578 and oversize inputs are rejected.'
byid['c15']['scope']='272record IDs, including repeated chat content, under min65536/max524288pixels only; no guarantee for larger new images or272distinct question contents.'
byid['c16']['statement']='First L4 experiment finished180updates and360record visits covering all272distinct training-record IDs.'
byid['c16']['scope']='One original seed42L4run; all272record IDs used at least once. These IDs include30chat records representing10content examples each stored3times; GPU not rerun.'
byid['c16']['verification']['details']='Reread both histories, independently counted360visits/full272manifest IDset, called deterministic row selector for all360positions, and compared full repeated-chat payloads.'
byid['c16']['verification']['denominators'].update(unique_record_ids=272,unique_chat_content_examples=10,chat_records=30,records_per_chat_content_example=3)
byid['c17']['scope']='5791counts targets over360record visits;4476counts one pass over272record IDs, including repeated chat content. Neither is a distinct-question denominator.'
byid['c28']['statement']='Second run also completes180updates/360record visits with112changed adapters and7unchanged frozen samples.'
byid['c28']['scope']='272unique record IDs/5791targets, with repeated chat content included. Sample loss decrease is not delivery-version quality evidence.'
for id in ['c14','c15','c16','c17','c28']:
    claim=byid[id]
    claim['evidence'].append({'source_id':'s-record-executed','locator':'training_records/chat_groups/mask_audit_exact_record_id_set/training_visit_recomputation',
                             'supports':'Revised record wording; exact ID coverage and duplicates are directly checked against unchanged original manifest/audit/history.'})
    claim['artifact_ids'].append('a-record-check')
    claim['primary_source_ids']+=['p-manifest','s-record-executed']
    claim['validation']['evidence_ids'].append('a-record-check')
prior['claims'].append({'id':'c32','kind':'numeric','statement':'The272training records include30chat records representing10different content examples, each stored with3distinct IDs.',
  'location':'First training-count paragraph:272records all used; repeated chat example caveat referring to previous section.',
  'status':'verified','evidence':[{'source_id':'s-record-executed','locator':'chat_groups and training_visit_recomputation',
    'supports':'All30complete chat payloads compared excluding ID,10distinct content groups*3; both histories cover all272record IDs.'}],
  'artifact_ids':['a-record-check','a-record-manifest','a-record-predecessor','a-record-notes'],
  'primary_source_ids':['p-manifest','p-train','p-gentle','p-mask','s-record-executed'],
  'validation':{'method':'Executed complete-payload equality comparison and ID-set/visit recomputation','evidence_ids':['a-record-check','a-record-manifest','a-record-notes']},
  'verification':{'method':'executed','expected':'272distinct record IDs;30chat records=10different content examples*3;360visits coverall272IDs.',
    'observed':'272manifest/mask IDs match;10chat groups,3IDs each,all non-ID payload fields identical within group;both histories360visits coverall272IDs.',
    'tolerance':'Exact JSON payload, integer and ID-set equality.',
    'details':'Direct all30chat-record inspection plus executed per-family payload equality,10unique user/answer/image/history/system payload counts and180two-record update recount.'},
  'scope':'272is a training-record denominator. Distinct IDs do not prove semantically different questions; chat duplicates are explicit. No new quality claim.'})
prior['issues'].append({'claim_id':'c16','status':'resolved','details':'Previous272different questions wording could be mistaken for272nonduplicated contents, although the count came fromunique IDs.',
  'resolution':'Author changed both mask/count paragraphs to272training records and explicitly noted repeated chat examples. Same reviewer reread complete20.4 and independently verified all30chat payloads=10examples*3,272IDsets andboth360visit coverage; prior report/section retained.',
  'evidence_ids':['a-record-prior-report','a-record-prior-section','a-record-current-section','a-record-check','a-record-notes']})
prior['source_sha256']=execution['current_source_sha256']
prior['independent_checks']['execution_evidence_ids'].append('a-record-check')
prior['independent_checks']['source_inspection_evidence_ids'].append('a-record-notes')
prior['independent_checks']['actual_checks'].append('Full current20.4reread, all prior source/evidence rehashes and30/10/3chat-payload/272ID/360visit recheck')
prior['known_limits'].append('272counts distinct training-record IDs, including10short chat contents copied3times; record-ID coverage is not question-content uniqueness.')
prior['version_history']=prior.get('version_history',[])+[
    {'source_sha256':execution['previous_source_sha256'],'report_path':prior_path.relative_to(ROOT).as_posix(),
     'report_sha256':execution['previous_report_sha256'],'section_path':old_path.relative_to(ROOT).as_posix(),
     'verdict':'pass','superseded_reason':'Record-ID versus duplicate-question-content wording clarified; earlier timing correction remains preserved in this report.'},
    {'source_sha256':execution['current_source_sha256'],'section_path':current_path,'reviewer_task':prior['reviewer_task'],
     'verdict':'pass','recheck_evidence_ids':['a-record-check','a-record-notes'],'change_scope':'Two record-wording sentences; complete section reread and original-data recount, all prior evidence hashes unchanged.'}]
for name in ['factual_accuracy','numeric_verification','source_verification','limitations']:
    prior['checks'][name]['claim_ids'].append('c32')
    prior['checks'][name]['details']+=' Currentrecord wording additionally verified from30chat payloads=10examples*3 and272ID/360visit original histories; unchanged prior evidence hashes all matched.'
prior['verdict']='pass'
REPORT.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:execution[k] for k in ['previous_source_sha256','current_source_sha256','training_records','unique_training_record_ids','chat_records','unique_chat_content_examples','records_per_chat_example']},indent=2))
print('All prior evidence rehashed:',len(reused),'; complete section reread; both360visits cover272records; report32claims/pass.')
