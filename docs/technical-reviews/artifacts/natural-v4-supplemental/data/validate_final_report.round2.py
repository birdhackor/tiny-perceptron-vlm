"""Own final integrity check; does not read or edit project review checkers."""
from pathlib import Path
import hashlib,json

P=Path('docs/technical-reviews/artifacts/natural-v4-supplemental/data')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
r=json.loads((P/'report.json').read_bytes())
assert r['reviewer_task']=='/root/v4_review_coordinator/factual_guide_data'
assert r['reviewer_context']=='fresh' and r['review_round']==2
assert r['complete_documents']==['docs/natural-assistant/v4/DATA.md']
assert r['assigned_document_scope']['complete_read_to_EOF'] is True
assert r['assigned_document_scope']['line_start']==1 and r['assigned_document_scope']['line_end']==169
assert r['current_document_sha256']=={'docs/natural-assistant/v4/DATA.md':sha('docs/natural-assistant/v4/DATA.md')}
assert (P/'DATA.current.fullfile.md').read_bytes()==Path('docs/natural-assistant/v4/DATA.md').read_bytes()
assert len(Path('docs/natural-assistant/v4/DATA.md').read_text().splitlines())==169
for path,digest in r['figure_sha256'].items():assert sha(path)==digest
for path,digest in r['prerequisite_document_sha256'].items():assert sha(path)==digest
sources={s['id']:s for s in r['sources']}
artifacts={a['id']:a for a in r['artifacts']}
assert len(sources)==len(r['sources']) and len(artifacts)==len(r['artifacts'])
for a in artifacts.values():assert sha(a['path'])==a['sha256']
for s in sources.values():
    if 'path' in s:assert sha(s['path'])==s['sha256']
    if 'ignored_original_path' in s:
        assert s['url'].startswith('https://') and s['version'] and s['inspection_note']
        assert sha(s['ignored_original_path'])==s['retrieval_sha256']
    if s['kind']=='execution':assert artifacts[s['artifact_id']]['kind']=='execution'
for c in r['claims']:
    assert c['status']=='verified' and c['scope'] and c['location']
    assert all(e['source_id'] in sources and e['locator'] and e['supports'] for e in c['evidence'])
    assert set(c['artifact_ids'])<=artifacts.keys()
    if c['kind'] in ['software','numeric','empirical']:
        v=c['verification']
        assert all(v.get(field) for field in ['method','expected','observed','details'])
    if c['kind'] in ['software','empirical']:
        assert v['method']=='executed'
        assert any(artifacts[a]['kind']=='execution' and artifacts[a]['command'] and artifacts[a]['result'] and artifacts[a]['environment'] for a in c['artifact_ids'])
    if c['kind']=='numeric':assert v['tolerance']
    if c['kind']=='empirical':assert isinstance(v['denominators'],dict) and v['denominators']
required={'factual_accuracy','numeric_verification','figure_consistency','source_verification','limitations'}
assert required==set(r['checks']) and all(r['checks'][name]['status']=='pass' for name in required)
assert r['verdict']=='pass' and r['issues']==[] and len(r['claims'])==16
assert not (P/'cpu-audit.stderr.txt').read_bytes() and not (P/'browser.stderr.txt').read_bytes()
first=json.loads((P/'report.first-issued.raw.json').read_bytes())
assert r['revision_history'][0]['report_sha256']==sha(P/'report.first-issued.raw.json')
changed={'cpu_audit.py','cpu-audit.stdout.txt','cpu-audit-result.json'}
first_receipts=[]
for a in first['artifacts']:
    oldpath=Path(a['path'])
    path=P/'first-issued-evidence'/oldpath.name if oldpath.name in changed else oldpath
    assert sha(path)==a['sha256'],(a['id'],str(path))
    first_receipts.append({'artifact_id':a['id'],'preserved_path':str(path),'original_sha256':a['sha256']})
result={'status':'pass','review_round':2,'reviewer_task':r['reviewer_task'],'claims':len(r['claims']),
        'sources':len(r['sources']),'artifacts':len(r['artifacts']),'required_checks':sorted(required),
        'report_sha256':sha(P/'report.json'),'first_issued_report_sha256':sha(P/'report.first-issued.raw.json'),
        'current_document_sha256':r['current_document_sha256'],'figure_sha256':r['figure_sha256'],
        'first_issued_artifacts_preserved':first_receipts,
        'scope':'Own report identity/schema/reference/current-byte integrity only; no extra model/data execution and no project checker change.'}
(P/'final-integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='first_issued_artifacts_preserved'},ensure_ascii=False,indent=2))
