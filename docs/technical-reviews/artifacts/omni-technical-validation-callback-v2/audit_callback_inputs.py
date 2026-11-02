"""Source14 callback：獨立讀原始收據與身份，不把舊結論當新證明。"""
import datetime,difflib,hashlib,json,platform,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def node(d,p):
    for k in p.split('/')[1:]:d=d[int(k)] if isinstance(d,list) else d[k.replace('~1','/').replace('~0','~')]
    return d

if __name__=='__main__':
    original_path=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_validation/technical-original-v1.json'
    original=read(original_path)
    protocol=ROOT/'docs/course-revision-20261029-omni/REVIEW-PROTOCOL.md'
    receipt_path=ROOT/'docs/technical-reviews/artifacts/omni-author-visuals-index-fix01/protocol-previous-source-receipt-v1.json'
    previous=read(receipt_path);old_protocol=ROOT/previous['snapshot']['path']
    own_s_policy=next(s for s in original['sources'] if s['id']=='s_policy')
    source_diff=list(difflib.unified_diff(old_protocol.read_text().splitlines(),protocol.read_text().splitlines(),fromfile='registered previous protocol',tofile='current protocol',lineterm=''))
    source_rows=[]
    for s in original['sources']:
        if s['kind']=='repository_code':source_rows.append(dict(source_id=s['id'],path=s['path'],original_sha256=s['sha256'],current_sha256=sha(ROOT/s['path']),unchanged=s['sha256']==sha(ROOT/s['path'])))
    artifact_rows=[dict(id=a['id'],path=a['path'],unchanged=sha(ROOT/a['path'])==a['sha256']) for a in original['artifacts']]
    freezes={}
    for label,name in [('original','source-freeze-technical-fix13.json'),('current','source-freeze-technical-fix14.json')]:
        p=ROOT/'docs/course-revision-20261029-omni/reviews'/name;f=read(p);page=next(x for x in f['pages'] if x['page_id']=='validation');freezes[label]=dict(reference=ref(p),page=page,snapshot_sha256=sha(ROOT/page['snapshot']),live_sha256=sha(ROOT/page['source']))
    manifest=read(ROOT/'docs/selftrained/v2-manifest.json');expected={x['path']:x['sha256'] for x in manifest['records']}
    stages={};inputs=[]
    for directory in sorted((ROOT/'docs/selftrained/results/training-raw').iterdir()):
        if not directory.is_dir():continue
        rp=directory/'receipt.json';tp=directory/'raw/train-receipt.json';ep=directory/'raw/execution.json'
        r,t,e=read(rp),read(tp),read(ep);files={x['path']:x for x in r['files']}
        for p in [rp,tp,ep]:inputs.append({**ref(p),'bytes':p.stat().st_size})
        stages[directory.name]=dict(run_id=r['run_id'],revision=r['revision'],architecture=t['architecture'],stage=t['stage'],steps=t['steps'],origin=t['origin']['kind'],seed=t['origin']['seed'],train_records=t['train_records'],validation_records=t['validation_records'],test_used_for_selection=t['test_used_for_selection'],job=e['job'],best_sha256=files['best.pt']['sha256'],safe_sha256=files['model.safetensors']['sha256'],stage_history=t.get('stage_history',[]),checks=dict(run_matches=r['run_id']==e['run_id'],completed=r['status']==e['status']=='completed' and e['returncode']==0 and t['completed_requested_steps'] is True and t['interrupted'] is False,steps_match=t['steps']==e['job']['steps'],random_origin=t['origin']['kind']=='all-neural-weights-random',data_sha_match=t['data_sha256']==expected,test_not_selection=t['test_used_for_selection'] is False,raw_train_receipt_matches=sha(tp)==files['train-receipt.json']['sha256'] and tp.stat().st_size==files['train-receipt.json']['bytes'],raw_execution_matches=sha(ep)==files['execution.json']['sha256'] and ep.stat().st_size==files['execution.json']['bytes']))
    by_run={x['run_id']:x for x in stages.values()}
    for x in stages.values():
        parent=x['job'].get('init_checkpoint')
        if parent:
            previous_stage=by_run[parent['run_id']];x['parent_link']=dict(run_id=parent['run_id'],sha256=parent['sha256'],parent_best_sha256=previous_stage['best_sha256'],match=parent['sha256']==previous_stage['best_sha256'],same_architecture=x['architecture']==previous_stage['architecture'])
        else:x['parent_link']=dict(root=x['stage']=='pretrain')
    leaves={}
    for arch,name in [('moe','moe-native'),('dense','dense-weighted')]:
        p=ROOT/f'docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json';index=read(p);cp=index['checkpoint'];leaf=stages[name]
        inputs.append({**ref(p),'bytes':p.stat().st_size})
        public=read(ROOT/'docs/selftrained/results/v2-final-public-results.json')['architectures'][arch]
        leaves[arch]=dict(run=index['run'],checkpoint=cp,count=index['completed_count'],checks=dict(leaf_run=cp['source_training_run']==leaf['run_id'],best=cp['sha256']==leaf['best_sha256']==public['selected_source']['sha256'],safe=cp['safe_weights_sha256']==leaf['safe_sha256']==public['safe_weights']['sha256'],step_count=cp['completed_requested_steps']==leaf['steps'],selected_step=cp['selected_step']==1000,task_counts=index['completed_count']==index['expected_count']==sum(x['count'] for x in index['per_task_final_reply'].values())==3734))
    saved_recalculations={}
    for label in ['b-mechanisms','c']:
        op=ROOT/f'docs/lesson-products/reviews/engineering-final-{label}-original.json';ep=ROOT/f'docs/lesson-products/reviews/engineering-final-{label}-execution.json';o,e=read(op),read(ep)
        raw_sha=o['verification']['raw_sha256'] if label=='b-mechanisms' else o['raw_reference']['sha256']
        saved_recalculations[label]=dict(original=ref(op),execution=ref(ep),original_owner=o['reviewer_task'],execution_owner=e['reviewer_task'],command=e['command'],raw_sha256=raw_sha,scope=e['scope'],boundary=e.get('boundary'),script_sha256=e.get('script_sha256',e.get('probe_script_sha256')),execution_bound_by_original=(o['execution']['sha256']==sha(ep) if label=='b-mechanisms' else any(x.get('sha256')==sha(ep) for x in o['execution_artifacts'])))
    requests_path=ROOT/'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json';requests=read(requests_path);identities={}
    for gi in [12,13]:
        group=requests['groups'][gi];rows=[]
        for row in group['items']:
            orig=read(ROOT/row['original_report']['path']);disposition=node(read(ROOT/row['disposition_ref']['path']),row['disposition_ref']['json_pointer'])
            rows.append(dict(identity=[row['original_report']['path'],row['original_report']['sha256'],row['json_pointer']],issue_id=row['issue_id'],original_issue=node(orig,row['json_pointer']),necessity=row['necessity'],accepted_action=disposition['action'],accepted_reason=disposition['reason'],accepted_change_burden=disposition['change_burden'],original_ref_hash_matches=sha(ROOT/row['original_report']['path'])==row['original_report']['sha256'],disposition_ref_hash_matches=sha(ROOT/row['disposition_ref']['path'])==row['disposition_ref']['sha256'],same_issue_identity=disposition['report']==row['original_report'] and disposition['json_pointer']==row['json_pointer'],same_necessity=disposition['final_necessity']==row['necessity'],original_verification_present=row['historical_fix_verification'] in disposition['verification']))
        identities[str(gi)]=dict(owner=group['old_owner'],count=len(rows),distinct_identities=len({tuple(x['identity']) for x in rows}),items=rows,unavailability=group['unavailability'])
    kernel_path=ROOT/'docs/technical-reviews/artifacts/review-20261019-validation/kernel-results-current-01.json';old_kernel=read(kernel_path)
    value=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),environment=dict(python=sys.version,platform=platform.platform(),device='CPU metadata only'),protocol=dict(current=ref(protocol),old=ref(old_protocol),old_authentication_receipt=ref(receipt_path),reconstruction_after_edit=True,original_s_policy_matches_reconstruction=own_s_policy['sha256']==sha(old_protocol)==previous['snapshot']['sha256'],diff=source_diff,dependency_assessment='Only final publishing paragraphs changed: baseline v1 five restores -> v2 six restores plus explicit original-figure metadata/byte preservation. Existing target claims c076/c097 stage order and full-unit recheck rules unchanged. Target has no figures. New tool scope does not certify a release or figure viewing; no new ability evidence follows.'),original_source_inventory=source_rows,original_artifacts=artifact_rows,source_freezes=freezes,training_stages=stages,training_primary_input_receipts=inputs,final_leaf_links=leaves,saved_recalculation_records=saved_recalculations,historical_identity_metadata=identities,historical_kernel_primary=dict(reference=ref(kernel_path),real_notebook_summary=old_kernel.get('real_notebook_summary'),isolated_probe=old_kernel.get('isolated_probe'),checks=old_kernel.get('checks'),scope=old_kernel.get('scope')),limits=['Metadata and current-code reinspection only; no new training, model forward, GPU or paid action.','The old archived full-book summary path is absent in this workspace. No current full331 notebook completion is inferred from old312 coverage. The current page is a command/protocol description, independently checked with bounded current CPU kernels.','Groups10/11 continuity findings were excluded; only groups12/13 identities were inspected.','Source reconstruction authenticates the old registered s_policy bytes after the edit; it is explicitly not a pre-edit backup or a current-source review.'])
    assert len(stages)==15
    assert all(all(x['checks'].values()) for x in stages.values())
    assert all(x['parent_link'].get('match',x['parent_link'].get('root')) and x['parent_link'].get('same_architecture',True) for x in stages.values())
    assert all(all(x['checks'].values()) for x in leaves.values())
    assert len(identities['13']['items'])==identities['13']['distinct_identities']==24
    out=HERE/'callback-input-audit-v2.json'
    with out.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(out.relative_to(ROOT)),sha256=sha(out),training_stages=len(stages),primary_records=len(inputs),identity_counts={k:v['count'] for k,v in identities.items()},protocol_authenticated=value['protocol']['original_s_policy_matches_reconstruction']),ensure_ascii=False))
