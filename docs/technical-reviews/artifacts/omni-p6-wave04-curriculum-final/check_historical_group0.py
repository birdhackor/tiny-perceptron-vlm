"""Inspect only historical request groups[0], after own full-review intake."""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
def sha(raw): return hashlib.sha256(raw).hexdigest()
def checked(ref):
    raw = (ROOT / ref['path']).read_bytes()
    assert sha(raw) == ref['sha256'], ref['path']
    return json.loads(raw)
def node(value, pointer):
    for p in pointer.strip('/').split('/') if pointer else []:
        p = p.replace('~1', '/').replace('~0', '~')
        value = value[int(p)] if isinstance(value, list) else value[p]
    return value

request_ref = {'path':'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json','sha256':'714f16f7d5c20924e20f91a8b48bca7a87bd77e2602d1a1ae501d06d93f819e5'}
group = checked(request_ref)['groups'][0]
assert group['page_id'] == 'curriculum' and len(group['items']) == 1
item = group['items'][0]
issue_report = checked(item['original_report'])
issue = node(issue_report, item['json_pointer'])
disposition = node(checked(item['disposition_ref']), item['disposition_ref']['json_pointer'])
verification_report = checked(item['historical_fix_verification']['report'])
verification = node(verification_report, item['historical_fix_verification']['json_pointer'])
unavailable_document = checked(group['unavailability']['report'])
unavailable = node(unavailable_document, group['unavailability']['json_pointer'])
assert issue['necessary'] is False and item['necessity'] == disposition['final_necessity'] == 'optional'
assert disposition['action'] == 'fixed' and disposition['issue_id'] == item['issue_id']
assert disposition['report'] == item['original_report'] and disposition['json_pointer'] == item['json_pointer']
assert item['historical_fix_verification'] in disposition['verification']
assert issue_report['reviewer_task'] == verification_report['reviewer_task'] == group['old_owner']
assert unavailable['reviewer_task'] == group['old_owner'] and unavailable['unavailable'] is True
assert any(e['tool']=='collaboration.followup_task' and e['request']['target']==group['old_owner'] and group['old_owner'] in e['response'] and 'not found' in e['response'] for e in unavailable['events'])
assert any(e['tool']=='collaboration.list_agents' and e['request']['path_prefix']==group['old_owner'] and e['response']['agents']==[] for e in unavailable['events'])
prior_pointer = checked(group['prior_active_pointer'])
prior_manifest = checked({'path':prior_pointer['manifest'],'sha256':prior_pointer['manifest_sha256']})
assert prior_manifest['dispositions']['path'] == item['disposition_ref']['path']
assert prior_manifest['dispositions']['sha256'] == item['disposition_ref']['sha256']
prior_freeze = checked(prior_manifest['source_freeze'])
old_page = next(p for p in prior_freeze['pages'] if p['page_id']=='curriculum')
assert verification['verdict']=='pass' and verification['source_sha256']==old_page['source_sha256']
assert verification['figures_sha256']==old_page['figures_sha256']
assert sha((ROOT / old_page['snapshot']).read_bytes())==old_page['source_sha256']
old_figure=prior_freeze['figure_snapshots']['course/figures/curriculum_learning_flow.svg']
assert sha((ROOT / old_figure['path']).read_bytes())==old_page['figures_sha256']['course/figures/curriculum_learning_flow.svg']

current_ref={'path':'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-curriculum-final/current-v1.json','sha256':'866baf877523110ad5fe93ee88b8c5905f48b6e8de9d02ae52e75c0a013bb224'}
current=checked(current_ref)
freeze_ref={'path':'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json','sha256':'d74eb6e0fc958fbd410c7588dbe4d96a7ced72ff512c7ea8f8863c87f68c9108'}
freeze=checked(freeze_ref)
page=next(p for p in freeze['pages'] if p['page_id']=='curriculum')
body=(ROOT / page['snapshot']).read_bytes()
assert sha(body)==current['source_sha256']==page['source_sha256']
assert sha((ROOT / page['source']).read_bytes())==page['source_sha256']
assert current['figure_sha256']==page['figures_sha256']
line_number,quote=next((n,line) for n,line in enumerate(body.decode().splitlines(),1) if 'BPE（' in line)
assert 'BPE（位元組對編碼，[6.2](../course/chapters/06.md#6.2)）可回讀' in quote
target=(ROOT / 'docs' / '../course/chapters/06.md').resolve()
target_text=target.read_text()
section=re.search(r'^## 6\.2 .+?(?=^## |\Z)',target_text,re.M|re.S).group(0)
target_freeze=next(p for p in freeze['pages'] if p['page_id']=='6.2')
assert sha(section.encode())==target_freeze['source_sha256']
assert '位元組對編碼' in section and 'Byte Pair Encoding，BPE' in section
assert '合併最常見的一對' in section
assert sha((ROOT / 'course/figures/curriculum_learning_flow.svg').read_bytes())==page['figures_sha256']['course/figures/curriculum_learning_flow.svg']
out={'reviewer_task':'/root/omni_p6_technical_curriculum_final','created_at':datetime.now(timezone.utc).isoformat(),'command':'python docs/technical-reviews/artifacts/omni-p6-wave04-curriculum-final/check_historical_group0.py','environment':{'python':sys.version.split()[0],'device':'CPU; JSON/Markdown/hash only; no model imported'},'request':request_ref,'group_index':0,'original_item':item,'original_issue':issue,'immutable_disposition_summary':{'action':disposition['action'],'final_necessity':disposition['final_necessity'],'reason':disposition['reason']},'original_fix_verified_on_prior_frozen_source':{'source_sha256':verification['source_sha256'],'figure_sha256':verification['figures_sha256'],'verdict':verification['verdict']},'old_owner_and_role':{'task':group['old_owner'],'role':group['old_role']},'unavailability_exact_record_checked':True,'prior_pointer':group['prior_active_pointer'],'full_review':current_ref,'current_source_sha256':page['source_sha256'],'current_figure_sha256':page['figures_sha256'],'current_source_quote':quote,'current_source_line':line_number,'target':{'path':'course/chapters/06.md#6.2','frozen_section_sha256':target_freeze['source_sha256'],'section_title':section.splitlines()[0],'relevant_quote':section.splitlines()[4]},'judgment':'still_fixed','necessity':'optional','reason':'Current curriculum names 位元組對編碼 and directly links6.2. That exact target opens with a concrete common-adjacent-pair example and expands Byte Pair Encoding/BPE. At this optional overview point the route supplies the explanation when needed; no algorithm retelling is required in curriculum. Original optional classification and fixed disposition are retained.','result':'All original identities, hashes, fixed/optional status, prior fix source/figure, actual unavailable events and current BPE label/link/target checks passed. Original/current reports and old records remain unchanged.'}
(OUT / 'historical-group-0-evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'group':0,'issue_id':item['issue_id'],'status':'still_fixed','necessity':'optional','current_source_sha256':page['source_sha256'],'target_section_sha256':target_freeze['source_sha256']},ensure_ascii=False))
