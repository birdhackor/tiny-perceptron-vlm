"""Post-original historical c09/c52/c53 raw metadata/data/journal inspection; no NN calls."""
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OLDROOT=Path('/workspace/selftrained-v2')
OUT=Path(__file__).with_name('historical-lineage-data-results-v1.json')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(p.read_text())

def main():
    idxpath=ROOT/'docs/selftrained/results/training-raw/index.json';idx=read(idxpath)
    copies=[]
    for row in idx['files']:
        p=ROOT/row['copy'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        original=Path(row['original']);original=original if original.is_absolute() else OLDROOT/original
        native_match=original.exists() and sha(original)==row['sha256']
        assert native_match
        copies.append({'stage':row['stage_key'],'copy':rec(p),'native_original_path':str(original),'native_original_sha_matches':True})
    grouped={}
    for row in idx['files']:
        st=row['stage_key'];grouped.setdefault(st,{})
        key='train' if row['copy'].endswith('/raw/train-receipt.json') else 'execution' if row['copy'].endswith('/raw/execution.json') else 'receipt'
        grouped[st][key]=read(ROOT/row['copy'])
        grouped[st][key+'_path']=ROOT/row['copy']
    runs={d['receipt']['run_id']:d for d in grouped.values()}
    best={next(f['sha256'] for f in d['receipt']['files'] if f['path']=='best.pt') for d in grouped.values()}
    stages=[]
    for st,d in grouped.items():
        r,e,t=d['receipt'],d['execution'],d['train'];fs={f['path']:f for f in r['files']}
        assert fs['execution.json']['sha256']==sha(d['execution_path']) and fs['train-receipt.json']['sha256']==sha(d['train_path'])
        assert r['run_id']==e['run_id'] and r['job']==e['job'] and r['returncode']==0 and e['returncode']==0
        assert t['completed_requested_steps'] and t['steps']==r['job']['steps'] and t['origin']['kind']=='all-neural-weights-random' and t['test_used_for_selection'] is False
        parent=r['job'].get('init_checkpoint');binding=None
        if parent:
            pd=runs[parent['run_id']];pbest=next(f['sha256'] for f in pd['receipt']['files'] if f['path']=='best.pt')
            assert parent['sha256']==pbest
            binding={'source_run':parent['run_id'],'source_stage':parent['stage'],'source_checkpoint_sha256':parent['sha256'],'matches_parent_actual_best':True}
        for old in t['stage_history']:
            assert old['checkpoint_sha256'] in best
        stages.append({'stage_key':st,'run_id':r['run_id'],'architecture':t['architecture'],'stage':t['stage'],'completed_steps':t['steps'],
                       'origin_kind':t['origin']['kind'],'test_used_for_selection':False,'parent_binding':binding,
                       'history_parents':[{k:x[k] for k in ['stage','step','checkpoint_sha256']} for x in t['stage_history']],
                       'selected_best_sha256':fs['best.pt']['sha256'],'safe_weight_sha256':fs['model.safetensors']['sha256'],
                       'train_records':t['train_records'],'validation_records':t['validation_records']})
    manifest=read(ROOT/'docs/selftrained/v2-manifest.json');data=OLDROOT/'outputs/selftrained-v2/data';data_rows=[];totals=Counter();test_ids=set()
    for row in manifest['records']:
        p=data/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        records=[json.loads(line) for line in p.read_text().splitlines()];splits=Counter(r['split'] for r in records);totals.update(splits)
        for r in records:
            if r['split']=='test':test_ids.add(r['id'])
        data_rows.append({'original':rec(p),'manifest_sha_match':True,'record_count':len(records),'split_counts':dict(splits)})
    assert dict(totals)=={'test':3734,'train':28876,'validation':2435}
    final=[]
    for arch,st in [('moe','moe-native'),('dense','dense-weighted')]:
        d=grouped[st];indexrow=next(x for x in idx['files'] if x['stage_key']==st and x['copy'].endswith('/raw/train-receipt.json'))
        native=Path(indexrow['original']);native=native if native.is_absolute() else OLDROOT/native
        mp=native.parent/'metrics.jsonl';declared=next(x for x in d['receipt']['files'] if x['path']=='metrics.jsonl')
        assert sha(mp)==declared['sha256']
        training=[json.loads(line) for line in mp.read_text().splitlines()];valid=[r for r in training if r.get('event')=='validation'];bestrow=min(valid,key=lambda r:r['validation_loss'])
        assert bestrow['step']==1000 and max(r.get('step',0) for r in training)==d['train']['steps']
        pub=ROOT/'docs/selftrained/results/public-raw'/arch;frozen=read(pub/'freeze/frozen.json');journal=read(pub/'test/evaluation-receipt.json');outer=read(pub/'test/receipt.json')
        ownbest=next(x['sha256'] for x in d['receipt']['files'] if x['path']=='best.pt');safesha=next(x['sha256'] for x in d['receipt']['files'] if x['path']=='model.safetensors')
        assert frozen['selected_checkpoint_sha256']==ownbest and frozen['safe_weights_sha256']==safesha==journal['checkpoint_sha256']
        rawbase=OLDROOT/f'outputs/selftrained-v2/operations/{arch}-public-final-test-attempt-1/monitor/artifact'
        nativejournals=list(rawbase.rglob('raw/evaluation-receipt.json'));assert len(nativejournals)==1
        jp=nativejournals[0];op=jp.with_name('outputs.jsonl');assert jp.read_bytes()==(pub/'test/evaluation-receipt.json').read_bytes()
        assert sha(op)==journal['outputs_sha256'] and op.stat().st_size==journal['output_byte_count']
        outputs=[json.loads(line) for line in op.read_text().splitlines()];ids=[r['record']['id'] for r in outputs]
        idsha=hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest()
        assert len(outputs)==len(set(ids))==3734 and set(ids)==test_ids and idsha==journal['completed_ids_sha256']
        assert journal['status']=='complete' and journal['resumed_from'] is None and journal['recovered_complete_tail_count']==0
        final.append({'architecture':arch,'source_stage':st,'actual_training_metrics':rec(mp),'validation_checkpoints':[{k:r[k] for k in ['step','validation_loss','selected']} for r in valid],
                      'completed_steps':d['train']['steps'],'independently_selected_minimum_validation_step':bestrow['step'],
                      'selected_best_sha256':ownbest,'public_safe_sha256':safesha,'freeze_created_at':frozen['created_at'],'test_started_at':outer['started_at'],
                      'original_final_journal':rec(jp),'original_final_outputs':rec(op),'output_rows':len(outputs),'unique_ids':len(set(ids)),'IDs_match_manifest_bound_test':True,
                      'completed_ids_sha256':idsha,'original_journal_status':journal['status'],'original_journal_resumed_from':journal['resumed_from'],
                      'same_protocol_once_scope':'Exactly the complete3734-row recorded run per fixed protocol, with no resume/recovered tail. Static evaluator marker/prefix checks inspected; no universal claim about unrecorded external runs.'})
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
            'environment':{'python':platform.python_version(),'device':'CPU; JSON/bytes only; no neural model or tensor weights loaded'},
            'scope':'Post-original prompted historicalc09/c52/c53 evidence: native original receipt/metadata chain, fixed raw JSONL counts, original trainer metric minima, original final outputs/journals. No NN calls.',
            'archive_index':rec(idxpath),'raw_metadata_file_bindings':copies,'training_stage_count':len(stages),'training_stages':stages,
            'direct_manifest_bound_data_counts':data_rows,'split_totals':dict(totals),'final_selected_and_journal_bindings':final,
            'evaluator_source':rec(ROOT/'scripts/selftrained/evaluate.py'),'evaluator_inspection':'EvaluationJournal validates same schema/conditions/checkpoint/protocol, exact committed byte/ID prefix and source records; completed test resume rejected. main lines782–807 creates exclusive protocol marker, only latest incomplete attempt may resume. Read source only; never called evaluator main.',
            'limits':['No checkpoint tensors or neural models loaded','No training/selection/final inference rerun','Original metadata/journals establish the supplied recorded protocol chain, not absence of every possible unregistered external action','Data-row counts do not assert independent samples, unseen text or people']}
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bytes':OUT.stat().st_size,'sha256':sha(OUT),'stages':len(stages),'native_original_files':len(copies),'split_totals':dict(totals),'final':[{k:r[k] for k in ['architecture','completed_steps','independently_selected_minimum_validation_step','output_rows']} for r in final]},ensure_ascii=False))

if __name__=='__main__':main()
