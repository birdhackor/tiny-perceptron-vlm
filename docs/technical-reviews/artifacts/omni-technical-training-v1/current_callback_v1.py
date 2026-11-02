"""Whole-training callback: immutable evidence reuse, real T reports, bounded CPU replay."""

import copy
import hashlib
import inspect
import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import scripts.evaluate as evaluator
from scripts.export_course import DOCUMENTS, reading_markdown
from scripts.prepare_data import generate_records
from tiny_perceptron.model import generate
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = Path('/workspace/work/omni-technical-training-current-v1')
OLD = Path('/workspace/work/omni-technical-training-v1')
WORK.mkdir(parents=True, exist_ok=True)
RESULT = HERE/'callback-current-v1.json'
assert not RESULT.exists()
torch.set_num_threads(2)
started = datetime.now(timezone.utc).isoformat()


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(): return datetime.now(timezone.utc).isoformat()


def loadref(ref):
    p = ROOT/ref['path']
    assert digest(p)==ref['sha256'], ref
    return json.loads(p.read_text())


def reference(path):
    p=Path(path)
    return {'path':str(p.relative_to(ROOT)),'sha256':digest(p)}


def run(label,args,allowed=(0,),cwd=WORK):
    p=subprocess.run([sys.executable,*map(str,args)],cwd=cwd,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),capture_output=True,text=True,timeout=40)
    row={'label':label,'command':[sys.executable,*map(str,args)],'cwd':str(cwd),
         'completed_at':timestamp(),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    assert p.returncode in allowed,row
    runs.append(row)
    return row


def section(body,lesson):
    m=re.search(r'^## '+re.escape(lesson)+r' .+$',body,re.M)
    assert m,lesson
    end=re.search(r'^## ',body[m.end():],re.M)
    return body[m.start():m.end()+end.start() if end else len(body)]


original_path=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_training/technical-original-v1.json'
assert digest(original_path)=='376398ac0de4919142fde2a6ba0294606ba7617bcf824d5f94f715dacca0e72a'
original=json.loads(original_path.read_text())
side_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v3.json'
assert digest(side_path)=='d8c8015797dff2e030b211499822873f236d54f5c6025f0ac9e08fce4218adba'
side=json.loads(side_path.read_text())
freeze=loadref(side['canonical_source_freeze'])
aux=loadref(side['section_freeze'])
inventory=loadref(side['issue_inventory'])
collection=loadref(side['actual_collection_snapshot'])
manifest_path=ROOT/'docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v4.json'
manifest=json.loads(manifest_path.read_text())
assignments=loadref(manifest['assignments'])
assert len(side['sections'])==26 and len(inventory['reports'])==35 and len(inventory['issues'])==5
page=next(p for p in freeze['pages'] if p['page_id']=='training')
snapshot=ROOT/page['snapshot']
body=snapshot.read_text()
assert digest(snapshot)==page['source_sha256']==digest(ROOT/'course/training.md')
assert page['figures_sha256']=={}

all_identity=[]
owners=[]
t_reports=[]
for entry in side['sections']:
    current=loadref(entry['report'])
    first=loadref(entry['original_report'])
    oldfreeze=loadref(entry['original_section_freeze'])
    assert current['lesson_id']==first['lesson_id']==entry['lesson_id']
    assert current['reviewer_task']==first['reviewer_task']
    assert current['reviewer_context']==first['reviewer_context']=='fresh'
    owner=current['reviewer_task'];owners.append(owner)
    assert owner not in manifest['author_tasks'] and owner!='/root/omni_reader_entries'
    task=next(t for t in assignments['tasks'] if t['task']==owner and t.get('lesson_id')==entry['lesson_id'])
    assert task['context']=='fresh' and task['role']=='technical'
    assert datetime.fromisoformat(first['created_at'])>=datetime.fromisoformat(task['dispatch_window']['not_before'])
    actual_receipts=[]
    for key,ckey,report in [('original_report','original_collection',first),('report','current_collection',current)]:
        receipts=[r for r in collection['collection_index'] if all(r[k]==entry[key][k] for k in ('path','sha256'))
                  and r['received_at']==entry[ckey]['received_at']]
        assert len(receipts)==1,(entry['lesson_id'],key,len(receipts))
        receipt=receipts[0]
        assert receipt['reviewer_task']==owner and receipt['lesson_id']==entry['lesson_id']
        assert receipt['wave']==entry[ckey]['wave'] and receipt['collection_role']=='technical'
        assert datetime.fromisoformat(receipt['received_at'])>=datetime.fromisoformat(report['created_at'])
        actual_receipts.append(receipt)
    latest_page=next(p for p in aux['pages'] if p['page_id']==entry['lesson_id'])
    assert current['source_sha256']==latest_page['source_sha256']
    assert current['figure_sha256']==latest_page['figures_sha256']
    first_page=next(p for p in oldfreeze['pages'] if p['page_id']==entry['lesson_id'])
    assert first['source_sha256']==first_page['source_sha256']
    assert digest(ROOT/first_page['snapshot'])==first['source_sha256']
    all_identity.append({'lesson_id':entry['lesson_id'],'owner':owner,'fresh':True,
                         'original_created_at':first['created_at'],'current_created_at':current['created_at'],
                         'original_received_at':actual_receipts[0]['received_at'],
                         'current_received_at':actual_receipts[1]['received_at'],
                         'assignment':task,'source_and_receipts_match':True})
    if entry['lesson_id'].startswith('T.'):
        assert section(body,entry['lesson_id'])==(ROOT/latest_page['snapshot']).read_text()
        assert current['verdict']=='pass' and all(c['status']=='verified' for c in current['claims'])
        assert all(i['status']=='resolved' for i in current['issues'])
        artifacts=[]
        for a in current['artifacts']:
            assert digest(ROOT/a['path'])==a['sha256'],a['path']
            artifacts.append({'id':a['id'],'path':a['path'],'sha256':a['sha256'],'kind':a['kind']})
        t_reports.append({'lesson_id':entry['lesson_id'],'owner':owner,'original':entry['original_report'],
          'current':entry['report'],'original_verdict':first['verdict'],'current_verdict':current['verdict'],
          'original_claims':len(first['claims']),'current_claims':len(current['claims']),
          'original_issues':first['issues'],'current_issues':current['issues'],
          'original_issue_resolutions':current.get('original_issue_resolutions',[]),
          'current_artifacts':artifacts,'exact_current_slice':True})
assert len(set(owners))==26 and len(t_reports)==11

schema_index_path=ROOT/'docs/course-revision-20261029-omni/reviews/technical-stage/technical-t4-schema-correction-index-v1.json'
schema_index=json.loads(schema_index_path.read_text())
for key in ['raw_initial_original','raw_initial_failed_selfcheck','formal_original_and_current',
            'formal_selfcheck','owner_comparison_receipt']:
    loadref(schema_index[key])
t4raw=loadref(schema_index['raw_initial_original'])
t4formal=loadref(schema_index['formal_original_and_current'])
assert t4raw['claims']==t4formal['claims'] and t4raw['sources']==t4formal['sources']
t4diagnostic={'index':reference(schema_index_path),'raw':schema_index['raw_initial_original'],
 'failed_selfcheck':schema_index['raw_initial_failed_selfcheck'],'formal_original':schema_index['formal_original_and_current'],
 'claims_and_sources_unchanged':True,'raw_issues':t4raw['issues'],
 'note':'Metadata-only corrected formal original, not a new fresh judgment or fabricated issue closure.'}

source_identities=[]
for s in original['sources']:
    if s['kind']=='repository_code':
        assert digest(ROOT/s['path'])==s['sha256'],s['path']
        source_identities.append({'id':s['id'],'path':s['path'],'sha256':s['sha256'],'unchanged':True})
for a in original['artifacts']:assert digest(ROOT/a['path'])==a['sha256'],a['path']
primary=json.loads((HERE/'primary-source-receipts.json').read_text())
primary_identities=[]
for s in primary['sources']:
    p=Path(s['full_text_local_path'])
    assert digest(p)==s['full_text_sha256'],str(p)
    lines=p.read_text().splitlines()
    for excerpt in s['excerpts']:
        start,end=map(int,excerpt['lines'].split('-'))
        assert '\n'.join(lines[start-1:end])==excerpt['text']
    primary_identities.append({'id':s['id'],'original_sha256':s['full_text_sha256'],
                               'locators_and_short_excerpts_still_match':True})
prior_extra=json.loads((HERE/'evidence-probe-results.json').read_text())
for name,h in prior_extra['historical_results'].items():
    assert digest(ROOT/h['path'])==h['sha256']
    raw=json.loads((ROOT/h['path']).read_text())
    assert round(raw['elapsed_seconds'],1)==h['rounded_time']

exporter=ROOT/'scripts/export_course.py'
assert digest(exporter)=='662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf'
targets={(ROOT/path).resolve():name+'.md' for name,path in DOCUMENTS.items()}
rendered=reading_markdown(body,ROOT/'course/training.md',targets,{},'callback-review')
fences=lambda text:re.findall(r'^```([^\n]*)\n(.*?)^```\s*$',text,re.M|re.S)
before=fences(body);after=fences(rendered)
assert before==after
inline=re.findall(r'```python\n(.*?)```',body,re.S)
assert len(inline)==2
assert inline[0]==(HERE/'inline_save_start.py').read_text()
assert inline[1]==(HERE/'inline_size_check.py').read_text()
export_probe={'source':reference(exporter),'before_fenced_blocks':len(before),
 'after_fenced_blocks':len(after),'all_fenced_block_bytes_equal':True,
 'inline_python_identical_to_executed_original':True,
 'note':'Executed current reading_markdown only, no site build/publication. Prior792f exporter is not used as current source.'}

runs=[]
data_root=WORK/'data/generated'
run('current-prepare',[ROOT/'scripts/prepare_data.py','--kind','attributes-sft','--seed','42','--output',data_root])
train=[json.loads(line) for line in (data_root/'attributes-sft/train.jsonl').read_text().splitlines()]
validation=[json.loads(line) for line in (data_root/'attributes-sft/validation.jsonl').read_text().splitlines()]
prompt='color=blue;shape=square;pitch=low;color?'
matches=[(n,r) for n,r in enumerate(train) if r['messages'][0]['content']==prompt]
assert len(matches)==1 and matches[0][0]==37 and matches[0][1]['family']=='blue:square:low'
assert not any(r['messages'][0]['content']==prompt for r in validation)
generated=generate_records('attributes-sft')
families=sorted({r['family'] for r in generated});random.Random(42).shuffle(families)
retrain=[r for r in generated if r['family'] in set(families[:9])]
assert [dict(r,split='train') for r in retrain]==train

checkpoint_files={name:OLD/'checkpoints'/f'{name}.pt' for name in ('attributes','attributes-int4','attributes-int8')}
for p in checkpoint_files.values():assert p.is_file(),str(p)
fixture={name:{'path':str(p),'sha256':digest(p)} for name,p in checkpoint_files.items()}
for name in ('attributes','attributes-int4'):
    run('current-train-family-'+name,[ROOT/'scripts/infer.py',checkpoint_files[name],'--chat','--prompt',prompt,
                                    '--tokens','24','--device','cpu','--json'])
run('current-inline-size',[HERE/'inline_size_check.py'],cwd=OLD)
run('current-evaluate-help',[ROOT/'scripts/evaluate.py','--help'])
run('current-evaluate-reject-temperature',[ROOT/'scripts/evaluate.py',checkpoint_files['attributes'],
                                        '--temperature','0.7'],allowed=(2,))

model,_=load_checkpoint(checkpoint_files['attributes'],'cpu')
default_calls=[]
real_generate=evaluator.generate
real_multinomial=torch.multinomial


def no_sampling(*args,**kwargs):raise AssertionError('Unexpected sampling in greedy evaluate')


def observed_generate(model,ids,*args,**kwargs):
    bound=inspect.signature(generate).bind(model,ids,*args,**kwargs);bound.apply_defaults()
    maxima=[];old_forward=model.forward
    def forward(*a,**k):
        result=old_forward(*a,**k)
        maxima.append(result['logits'][:,-1].argmax(-1).tolist())
        return result
    model.forward=forward
    try:output=real_generate(model,ids,*args,**kwargs)
    finally:model.forward=old_forward
    new=output[:,ids.shape[1]:].tolist()[0]
    assert new==[m[0] for m in maxima]
    default_calls.append({'temperature':bound.arguments['temperature'],'new_ids':new,
                          'argmax_ids':[m[0] for m in maxima],'equal':True})
    return output


evaluator.generate=observed_generate;torch.multinomial=no_sampling
rng=torch.get_rng_state().clone()
try:greedy_eval=evaluator.evaluate(model,validation[:1],'sft',24)
finally:evaluator.generate=real_generate;torch.multinomial=real_multinomial
assert torch.equal(rng,torch.get_rng_state()) and all(c['temperature']==0 for c in default_calls)
greedy_probe={'calls':default_calls,'rng_unchanged':True,'sampling_guard_not_called':True,
              'records':1,'scope':'Two-update untrained fixture only; verifies policy, not answer quality.'}

policy_path=HERE/'selection-policy-current-v1.json'
assert not policy_path.exists()
policy={'created_at':timestamp(),'candidate_bits':[4,8],
 'rule':'Maximize validation completed_exact_match, then exact_match, then minimize logical model-state bytes.',
 'scope':'Predeclared deterministic audit choice on an already-known two-update fixture; not a new model-quality or novel test evaluation.'}
policy_path.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
evaluations={}
for name,path in checkpoint_files.items():
    output=WORK/f'{name}-validation.json'
    run('current-validation-'+name,[ROOT/'scripts/evaluate.py',path,'--data',data_root/'attributes-sft/validation.jsonl',
        '--mode','sft','--tokens','24','--limit','all','--device','cpu','--output',output])
    evaluations[name]=json.loads(output.read_text())
candidates=[]
for bits in (4,8):
    name=f'attributes-int{bits}'
    payload=torch.load(checkpoint_files[name],map_location='cpu',weights_only=True)
    size=sum(t.numel()*t.element_size() for t in payload['model'].values())
    candidates.append({'bits':bits,'completed_exact_match':evaluations[name]['completed_exact_match'],
                       'exact_match':evaluations[name]['exact_match'],'logical_model_bytes':size})
selected=min(candidates,key=lambda c:(-c['completed_exact_match'],-c['exact_match'],c['logical_model_bytes']))
selection_path=HERE/'selection-result-current-v1.json'
assert not selection_path.exists()
selection={'created_at':timestamp(),'policy':reference(policy_path),'validation_candidates':candidates,
 'selected_bits':selected['bits'],'validation_outputs':[reference(WORK/f'{name}-validation.json')
    if (WORK/f'{name}-validation.json').is_relative_to(ROOT) else
    {'external_path':str(WORK/f'{name}-validation.json'),'sha256':digest(WORK/f'{name}-validation.json')}
    for name in checkpoint_files],
 'not_reselected_after_test':True,'scope':policy['scope']}
selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
selection_sha=digest(selection_path)
test_receipts=[]
for name,path in checkpoint_files.items():
    output=WORK/f'{name}-test.json'
    row=run('current-test-'+name,[ROOT/'scripts/evaluate.py',path,'--data',data_root/'attributes-sft/test.jsonl',
        '--split-label','test','--mode','sft','--tokens','24','--limit','all','--device','cpu','--output',output])
    test_receipts.append(row['completed_at'])
assert digest(selection_path)==selection_sha
assert all(datetime.fromisoformat(t)>datetime.fromisoformat(selection['created_at']) for t in test_receipts)
test=[json.loads(line) for line in (data_root/'attributes-sft/test.jsonl').read_text().splitlines()]
old_prompt='color=blue;shape=circle;pitch=low;color?'
old_test=[n for n,r in enumerate(test) if r['messages'][0]['content']==old_prompt]
assert old_test==[7] and not any(r['messages'][0]['content']==prompt for r in test)
for name,p in checkpoint_files.items():assert digest(p)==fixture[name]['sha256']
split_probe={'new_prompt':prompt,'train_row':37,'family':'blue:square:low','target':'blue',
 'new_prompt_absent_validation_test':True,'old_prompt':old_prompt,'old_test_rows':old_test,
 'generator_function_matches_current_cli':True,'selection':reference(selection_path),
 'validation_before_selection_before_test':True,'counts':{'train':len(train),'validation':len(validation),'test':len(test)},
 'note':'T.9 owner first found this necessary issue. The whole-file original missed it and remains unchanged with zero original issues. Earlier test exposure is not asserted to prove actual test-driven tuning or invalidated scores.'}

helper_output=HERE/'numbered-intake-current-v1.json'
helper_command=['uv','run','--extra','cpu','python','scripts/omni_revision/review_setup.py','check-numbered',
 '--manifest',str(manifest_path.relative_to(ROOT)),'--intake','--output',str(helper_output.relative_to(ROOT))]
p=subprocess.run(helper_command,cwd=ROOT,text=True,capture_output=True,timeout=40)
assert p.returncode==0,(p.stdout,p.stderr)
helper={'command':helper_command,'cwd':str(ROOT),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,
 'output':reference(helper_output),'scope':'Intake schema/source/receipt validation only; not a completed stage seal or all-gate.'}

own_claim_audit=[]
for c in original['claims']:
    own_claim_audit.append({'id':c['id'],'original_kind':c['kind'],'original_status':c['status'],
      'evidence_source_ids':[e['source_id'] for e in c['evidence']],
      'current_assessment':'verified_after_full_current_body_read_and_scope_reassessment',
      'basis':'Original code/artifact/primary byte identities verified; original observations remain within their explicit scope. '
              +('Current probe/split/order additionally replayed; prior whole-page failure to detect T.9 issue acknowledged.'
                if c['id'].startswith('t9-') else
                'New greedy paragraph tested with actual evaluate→generate and sampling guard.'
                if c['id'].startswith('t4-') else
                'Historical claims remain old single-run reports, not new GPU/model measurements.'
                if c['id'].startswith('time-') else 'No contrary new passage or section finding.')})

result={'started_at':started,'completed_at':timestamp(),
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
 'original':reference(original_path),'sidecar':reference(side_path),'canonical_freeze':side['canonical_source_freeze'],
 'section_freeze':side['section_freeze'],'current_source':{'path':page['source'],'snapshot':page['snapshot'],'sha256':page['source_sha256']},
 'all26_identity_and_receipts':all_identity,'t_section_reviews':t_reports,'t4_schema_history':t4diagnostic,
 'issue_index':side['issue_inventory'],'original_issue_count':len(original['issues']),
 'own_original_artifacts_verified':len(original['artifacts']),'original_repository_source_identities':source_identities,
 'original_primary_source_identities':primary_identities,'historical_time_records_rechecked':23,
 'original104_claim_reassessment':own_claim_audit,'exporter':export_probe,'fixture_checkpoints':fixture,
 'greedy_probe':greedy_probe,'t9_probe_and_order':split_probe,'runs':runs,'numbered_intake':helper,
 'scope':'Full canonical training callback plus T.1–T.11 reports; all26 owner metadata and receipts. '
         'No other-page claim judgments, reader answers, formal training, GPU, paid work, publication or completed-stage-seal claim.'}
RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'callback_result':reference(RESULT),'sections':len(t_reports),'all_distinct_owners':len(set(owners)),
                  'new_probe_train_row':37,'old_probe_test_rows':old_test,'greedy_verified':True,
                  'selection_before_test':True,'helper_exit_code':p.returncode,'runs':len(runs)}))
