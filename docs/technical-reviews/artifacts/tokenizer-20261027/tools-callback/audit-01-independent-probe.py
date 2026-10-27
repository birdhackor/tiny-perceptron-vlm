"""Independent reviewer probes; fixture supplies synthetic records, no reviewer conclusions."""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.util
import json
import platform
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
SCRATCH = Path('/workspace/work/tokenizer-dynamic-20261027/tools-callback')
OUT = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/tools-callback'
spec = importlib.util.spec_from_file_location('callback_probe_fixture', ROOT/'tests/test_historical_rechecks.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

def swap(value, old, new):
    if isinstance(value, dict):
        if (value.get('path'), value.get('sha256')) == (old['path'], old['sha256']):
            value.update(new)
        for child in value.values():
            swap(child, old, new)
    elif isinstance(value, list):
        for child in value:
            swap(child, old, new)

def change_report(root, built, path, mutate):
    m, _, _, seals = built
    side = t.review.document(root, m['historical_fix_rechecks'])
    old = {'path':path, 'sha256':t.sha((root/path).read_bytes())}
    value = t.review.document(root, old)
    mutate(value)
    new = t.save(root,path,value)
    for target in (m, seals, side):
        swap(target,old,new)
    for phase in seals['new'].values():
        phase['report_sha256s'] = [new['sha256'] if v == old['sha256'] else v for v in phase['report_sha256s']]
    m['historical_fix_rechecks'] = t.save(root,'current/historical-rechecks.json',side)

def mutate(root,built,name):
    m, _, _, seals = built
    side = t.review.document(root,m['historical_fix_rechecks'])
    if name == 'tool_helper_drift':
        p=root/'docs/review-tools/historical_fix_rechecks.py'
        p.write_bytes(p.read_bytes()+b'\n# independent mutation\n')
    elif name == 'empty_sidecar':
        side['rechecks']=[]
    elif name == 'altered_old_verification':
        side['rechecks'][0]['extra']='ignored extension'
        change_report(root,built,'current/appendix.json',lambda d:d['historical_issue_rechecks'][0]['historical_fix_verification'].update(json_pointer='/claims/0'))
        return
    elif name == 'wrong_original_identity':
        change_report(root,built,'current/appendix.json',lambda d:d['historical_issue_rechecks'][0].update(original_report=m['tool_audit']['report']))
        return
    elif name == 'full_receipt_missing':
        seals['new']['technical']['collection_index']=[r for r in seals['new']['technical']['collection_index'] if r['path']!='current/tool-audit.json']
    elif name == 'full_created_after_appendix':
        change_report(root,built,'current/tool-audit.json',lambda d:d.update(created_at=t.at(27)))
        return
    elif name == 'unavailability_without_events':
        change_report(root,built,'current/unavailable.json',lambda d:d['records'][1].update(events=[]))
        return
    elif name == 'invalid_name_without_listing':
        change_report(root,built,'current/unavailable.json',lambda d:d['records'][1].update(events=d['records'][1]['events'][:1]))
        return
    elif name == 'full_unresolved_issue':
        change_report(root,built,'current/full.json',lambda d:d['issues'].append({'id':'NEW','status':'unresolved','details':'independent unresolved marker'}))
        return
    elif name == 'current_evidence_hash_tamper':
        p=root/'docs/technical-reviews/artifacts/calculation.txt'
        p.write_bytes(b'changed independently\n')
    elif name == 'prior_pointer_chain_success':
        prior=side['prior_active_pointer']
        middle=t.save(root,'current/middle-pointer.json',{'previous_pointer':prior})
        t.save(root,'docs/course-reviews-active.json',{'previous_pointer':middle})
    else:
        raise AssertionError(name)
    m['historical_fix_rechecks']=t.save(root,'current/historical-rechecks.json',side)

CASES = {
    'empty_sidecar':'empty rechecks',
    'tool_helper_drift':'historical tool audit version stale',
    'altered_old_verification':'historical verification is not original',
    'wrong_original_identity':'historical issue identity differs',
    'full_receipt_missing':'missing full-review receipt',
    'full_created_after_appendix':'appendix predates full review',
    'unavailability_without_events':'missing exact original-task followup receipt',
    'invalid_name_without_listing':'missing empty original-task-prefix listing',
    'full_unresolved_issue':'technical validation failed',
    'current_evidence_hash_tamper':'artifact changed',
    'prior_pointer_chain_success':None,
}
results=[]
for name, expected in CASES.items():
    root=SCRATCH/'probe-01-fixtures'/name
    root.mkdir(parents=True, exist_ok=False)
    built=t.fixture(root)
    mutate(root,built,name)
    m,live,_,seals=built
    for wave in m['review_waves']:
        for role,phase in seals[wave['id']].items():
            wave['seals'][role]=t.save(root,f"{wave['id']}/seal-{role}.json",phase)
    try:
        returned=t.check(root,built)
        observed='accepted '+repr(returned)
        gate_ok=expected is None
    except Exception as e:
        observed=f'{type(e).__name__}: {e}'
        gate_ok=expected is not None and expected in str(e)
    inv=t.inventory.inventory(root,m,live,'sealed','technical')
    inventory_ok=bool(inv['failing']) == (expected is not None)
    results.append({'case':name,'expected_rejection_fragment':expected,'gate':observed,'inventory_failing':inv['failing'],'inventory_errors':inv['errors'],'inventory_disposition_errors':inv['dispositions']['failing'],'passed':gate_ok and inventory_ok})

record={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'command':f'{sys.executable} {__file__}','environment':{'python':sys.version,'platform':platform.platform(),'cwd':str(ROOT)},'scope':'11 independently selected mutations of synthetic fixture; gate check_dispositions and issue_inventory both executed; no report conclusions reused','results':results}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'audit-01-independent-probe.py').write_bytes(Path(__file__).read_bytes())
dest=OUT/'audit-01-independent-probes.json'
dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':sum(r['passed'] for r in results),'total':len(results),'results':[{'case':r['case'],'passed':r['passed'],'gate':r['gate']} for r in results],'artifact':str(dest.relative_to(ROOT))},ensure_ascii=False,indent=2))
raise SystemExit(0 if all(r['passed'] for r in results) else 1)
