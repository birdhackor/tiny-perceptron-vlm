"""Same-owner narrow current navigation/dependency callback, no model reruns."""
from pathlib import Path
from datetime import UTC, datetime
import difflib
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
    return raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)],raw[:heads[i].start()].count(b'\n')+1

history=next(OUT.glob('prior-report-R.4-*.opaque.json'))
prior=json.loads(history.read_bytes())
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_r_4_clean'
assert prior['verdict']=='pass'
now=datetime.now(UTC).isoformat()
read_map={'course/README.md':['R.4'],'course/chapters/19.md':['19.6','19.11','19.12'],'course/chapters/20.md':['20.1','20.5'],'course/training.md':['T.3','T.11']}
sources=[];ranges=[];differences=[]
for path,ids in read_map.items():
    raw=(ROOT/path).read_bytes()
    snapshot=OUT/('frozeninput-current-'+Path(path).parent.name+'-'+Path(path).name)
    assert raw==snapshot.read_bytes()
    sources.append({'original_path':path,'permanent_frozen_input_path':str(snapshot.relative_to(ROOT)),'raw_sha256':sha(snapshot),'frozen_at_utc':now,'actually_read_sections':ids,'meaning':'Exact current whole raw source bytes saved at this callback; only named sections personally read. This is not whole-chapter scientific approval.'})
    for key in ids:
        body,line=section(raw,key)
        assert body==(OUT/('current-'+key+'.md')).read_bytes()
        if key=='R.4':
            assert hashlib.sha256(body).hexdigest()==prior['source_sha256']
            assert body==(OWN/'recheck-20261005/current-R.4.md').read_bytes()
        else:
            original=OWN/('navigation-source-'+key+'.md')
            old=original.read_bytes()
            if old!=body:
                differences += list(difflib.unified_diff(old.decode().splitlines(keepends=True),body.decode().splitlines(keepends=True),fromfile=str(original.relative_to(ROOT)),tofile=str((OUT/('current-'+key+'.md')).relative_to(ROOT)),n=1))
            ranges.append({'section':key,'origin':path,'first_line':line,'current_raw_section_sha256':hashlib.sha256(body).hexdigest(),'previously_personally_read_section_sha256':sha(original),'current_equals_previous_section':body==old,'permanent_current_section_path':str((OUT/('current-'+key+'.md')).relative_to(ROOT))})
assert {row['section'] for row in ranges if not row['current_equals_previous_section']}=={'19.11','T.3'}
(OUT/'actual-section-differences.txt').write_text(''.join(differences))

current=lambda key:(OUT/('current-'+key+'.md')).read_text()
assert '主線計畫' in current('19.6') and '接進問題前面的輸入序列' in current('19.6')
assert '真人語音' in current('19.6')
assert '待填的驗收矩陣' in current('19.12')
assert '不替主文待填的新驗收矩陣提供分數' in current('19.12')
assert '新成品的操作與權重待實作完成後另回填' in current('19.11')
assert 'python scripts/fetch_capstone.py --stage joint' in current('19.11')
assert 'python scripts/capstone.py train --stage pretrain' in current('19.11')
assert '四份階段／分支' in current('19.11')
assert re.findall(r'```bash\n(.*?)```',current('19.11'),re.S)==re.findall(r'```bash\n(.*?)```',(OWN/'navigation-source-19.11.md').read_text(),re.S)
assert '錄音先由 Whisper 聽寫' in current('20.1') and '圖文核心 Qwen' in current('20.1')
assert '不能排除上游預訓練曾見過' in current('20.5')
assert 'scripts/fetch_course_models.py --model text_foundation' in current('T.3')
assert '--experiment simple_models --device cpu' in current('T.3')
assert '兩類模型、四個設定' in current('T.3')
assert re.findall(r'```bash\n(.*?)```',current('T.3'),re.S)==re.findall(r'```bash\n(.*?)```',(OWN/'navigation-source-T.3.md').read_text(),re.S)
assert '其能力表和交付指引與局部實驗分開' in current('T.11')

routes=[]
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',current('R.4')):
    path,_,anchor=target.partition('#');destination=(ROOT/'course'/path).resolve()
    assert destination.is_file()
    if anchor:assert re.search(r'^## '+re.escape(anchor)+r' ',destination.read_text(),re.M)
    routes.append({'label':label,'target':str(destination.relative_to(ROOT)),'anchor':anchor,'exists':True})

retained=[]
for item in prior['artifacts']:
    assert sha(ROOT/item['path'])==item['sha256']
    retained.append({'artifact_id':item['id'],'path':item['path'],'sha256':item['sha256'],'support':'Exact previously personally inspected/actually executed evidence retained; no new execution or scientific approval inferred.'})
unchanged=[];changed_whole=[]
for source in prior['sources']:
    if source['kind']!='repository_code':continue
    actual=sha(ROOT/source['path'])
    if source['id'] in ('nav-19','nav-20'):
        changed_whole.append({'source_id':source['id'],'prior_path':source['path'],'prior_report_whole_sha256':source['sha256'],'current_whole_source_frozen_in':next(s for s in sources if s['original_path']==source['path'])})
    else:
        assert actual==source['sha256']
        unchanged.append({'source_id':source['id'],'path':source['path'],'sha256':actual})
historical=next(s for s in prior['sources'] if s['id']=='nav-training')
assert historical['path']==str((OWN/'frozeninput-course-training.md').relative_to(ROOT))
assert sha(ROOT/historical['path'])==historical['sha256']=='22f3115b0d09a83d82342e82c101df8b8516aa5de7796b137a5243c213396324'

policy=[]
for path in ('docs/review-tools/factual-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md'):
    snapshot=OUT/Path(path).name
    assert snapshot.read_bytes()==(ROOT/path).read_bytes()
    policy.append({'path':path,'personally_reread':True,'raw_sha256':sha(snapshot)})
support={'planned-mainline':'Personally reread current19.6/19.12: feature projection/prefix plan and finite task acceptance remain explicitly pending; no new ability score asserted. Both sections are byte-identical to my originally inspected sections. Original implementation/paper/CPU evidence retained by exact fingerprints.',
 'upstream-extension':'Personally reread current20.1/20.5: Qwen image/chat and Whisper transcript route still adopts upstream weights, separate from self-trained mainline. Both needed sections are byte-identical. Original load_core/load_asr and official source contracts are fingerprint-exact and retained without invoking models.',
 'navigation':'Personally reread current19.11,T.3,T.11 and completeR.4.19.11 now calls four models stages/branches rather than four stations; T.3 describes two model types/four settings. R.4 only claims these are old integration/local-operation destinations. Their actual bash recipe bytes are unchanged; earlier original-parser validation remains applicable without rerun. New-product pending boundary remains explicit. Current T.3 differs from historical22f311 whole input and is supported by a new dated current-source snapshot, never presented as byte-identical to historical training input.'}
environment={'python':sys.version,'python_executable':sys.executable,'device':'cpu (text/UTF-8 fingerprints/navigation only)','scope':'No network, model/data download, GPU, training, model inference, new ability score, unrelated CPU rerun or full other-section scientific acceptance.'}
observation={'reviewer_task':prior['reviewer_task'],'callback_at_utc':now,'prior_complete_report':{'path':str(history.relative_to(ROOT)),'sha256':sha(history),'contains':'Own complete prior report, issues and revision history, preserved opaquely before change.'},'source_sha256':prior['source_sha256'],'personally_read_current_scopes':read_map,'current_frozen_source_inputs':sources,'necessary_section_comparisons':ranges,'current_R4_link_targets':routes,'affected_claim_support':support,'historical_nav_training_input_retained':historical,'mutable_whole_navigation_sources_to_replace':changed_whole,'unchanged_original_repository_sources':unchanged,'retained_original_evidence_fingerprints':retained,'latest_method_instructions_read':policy,'decision':'pass','decision_scope':'Narrow current navigation/dependency callback only. No substantive uncertainty found in R.4 claims; old whole-file mutable citations will be replaced with personally checked dated current frozen raw inputs, and historicaltraining input remains explicitly historical. No initial or intervening evidence overwritten.'}
(OUT/'callback-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
(OUT/'callback-observations.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'reviewer_task':prior['reviewer_task'],'source_sha256':prior['source_sha256'],'callback_at_utc':now,'read_scope':read_map,'section_comparisons':ranges,'affected_claim_support':support,'all_prior_evidence_fingerprints_exact':True,'historical_training_source_sha256':historical['sha256'],'decision':'pass'},ensure_ascii=False,indent=2))
print('All actual current navigation/dependency and retained-evidence fingerprint assertions passed.')
