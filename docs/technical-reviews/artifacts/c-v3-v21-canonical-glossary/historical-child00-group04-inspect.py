"""Prompted, same-owner inspection of four authenticated historical identities."""

import copy
import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
ROUND = 'docs/course-revision-20261101-c-v3/'
ART = 'docs/technical-reviews/artifacts/c-v3-v21-canonical-glossary/'
BASE = ROUND + 'technical-v21/canonical_glossary/'
PACKET = ROUND + 'technical-coordination-v21/history-prompts/child-00-group-04-historical-identities.json'
APPENDIX = BASE + 'historical-child-00-group-04-appendix-v1.json'
OUTPUT = ART + 'historical-child00-group04-inspection-v1.json'
COMMAND = ('UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb '
           'UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python '
           + ART + 'historical-child00-group04-inspect.py')

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def ref(path):
    return {'path': path, 'sha256': sha(path)}

def load(path):
    return json.loads((ROOT / path).read_text())

def read_reference(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return load(reference['path'])

def pointer(value, path):
    for key in path.strip('/').split('/') if path else []:
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value

def save(path, value):
    with (ROOT / path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

assert sha(PACKET) == 'c4ee84c4dd41ebe643719a9e0a1fde7068ffc68a96047fac855691420dc19cc1'
packet=load(PACKET)
assert packet['page_id']=='glossary' and packet['source_group_index']==4
assert len(packet['historical_identities_only'])==4
currentpath=BASE+'current-v1.json'
assert sha(currentpath)=='88bbcebc23e1570cc2b50a135bb006315ba25415b3f49bac9a0c8aeeb48838c2'
originalpath=BASE+'original-v1.json'
assert sha(originalpath)=='60ff0ebdea32be3a94018117ccba586ac67dbdc17ff6e07b658f55c5827f27b9'
current=load(currentpath)
receiptpath=ROUND+'technical-coordination-v21/receipts/canonical-glossary-current-v1.json'
assert sha(receiptpath)=='05eded6a42cfc70e4fab417e47e75d1e59d6f473cccc996ebc4fe249b9f6e456'
receipt=load(receiptpath)
assert receipt['report']==ref(currentpath)
assert datetime.fromisoformat(receipt['received_at'])<datetime.now(UTC)
read_reference(packet['source_chain'])
read_reference(packet['prior_active_pointer'])
original_unavailability=pointer(read_reference(packet['unavailability']['report']),packet['unavailability']['json_pointer'])
assert original_unavailability['reviewer_task']=='/root/rv1019_technical_glossary'

fixref=packet['historical_identities_only'][0]['historical_fix_verification']['report']
fix=read_reference(fixref)
assert fix['reviewer_task']=='/root/rv1019_technical_glossary'
fix_artifacts={x['id']:x for x in fix['artifacts']}
selected_ids=['sources_current','audit_current','replay_current','diff_current','code_current','a6']
selected_artifacts=[]
for aid in selected_ids:
    artifact=copy.deepcopy(fix_artifacts[aid])
    assert sha(artifact['path'])==artifact['sha256']
    selected_artifacts.append(artifact)
sources_raw=load(fix_artifacts['sources_current']['path'])
audit=load(fix_artifacts['audit_current']['path'])
replay=load(fix_artifacts['replay_current']['path'])
assert audit['replayed_result']['sha256']==sha(fix_artifacts['replay_current']['path'])
assert audit['emoji']['codepoints']==['U+1F44D','U+1F3FD']
assert audit['emoji']['utf8_byte_count']==8
assert replay['ppo']['ratio']>1.7 and 1.19<replay['ppo']['clipped']<1.21
assert replay['ppo']['ratio'] != replay['ppo']['clipped']
sourcepath='course/glossary.md'
body=(ROOT/sourcepath).read_text();lines=body.splitlines()
assert sha(sourcepath)==current['source_sha256']=='c71853d2368dc981d52c279eaedd7029e8f69d65fbf39a652606c83510f6f056'
assert len(audit['diff']['changed_rows'])==2
character_change=next(x for x in audit['diff']['changed_rows'] if '字元、碼點' in x['after'])
ppo_change=next(x for x in audit['diff']['changed_rows'] if '近端策略最佳化' in x['after'])
assert lines[15]==character_change['after']
assert lines[179]==ppo_change['after']
assert '少數字元由幾個碼點組成' in lines[15]
assert 'U+D800–U+DFFF 除外' in lines[15]
assert '但不保證每次的改變都很小' in lines[179]
assert character_change['before'] not in body and ppo_change['before'] not in body
posttraining_path='tiny_perceptron/posttraining.py'
assert sha(posttraining_path)=='3d0e2ae3b29f95abebde9b10d0cd106639a131fab1b2c7a509e732ba8daf295b'
post=(ROOT/posttraining_path).read_text()
assert 'ratio = (new_log_prob - old_log_prob.detach()).exp()' in post
assert 'clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * fixed_advantage' in post
assert 'surrogate = torch.minimum(unclipped, clipped)' in post

appendix=copy.deepcopy(current)
appendix['created_at']=datetime.now(UTC).isoformat()
appendix['review_scope']='Bounded prompted historical appendix for child-00/group-04 only. Complete current152-claim/source/artifact review is retained by exact identity, not redone. Four authenticated historical issue identities (two substantive fixes) are inspected against original issues, original fix raw evidence and current L16/L180; no prior still_fixed conclusion is copied.'
appendix['appendix_kind']='bounded_historical_fix_recheck'
appendix['historical_packet']=ref(PACKET)
appendix['source_group_index']=packet['source_group_index']
appendix['prior_active_pointer']=copy.deepcopy(packet['prior_active_pointer'])
appendix['historical_unavailability']=copy.deepcopy(packet['unavailability'])
appendix['original_unavailability_record']=original_unavailability
appendix['historical_source_chain']=copy.deepcopy(packet['source_chain'])
appendix['current_report_basis']=ref(currentpath)
appendix['current_actual_intake_receipt']=ref(receiptpath)
appendix['historical_callback_scope']='Four identities only; no fresh full-file science review, no NN/whole-glossary rerun, training, final, public transport or global T/C closure.'

for artifact in selected_artifacts:
    item=copy.deepcopy(artifact);item['id']='histfix_'+item['id']
    item['description']='Authenticated original fix raw evidence retained from /root/rv1019_technical_glossary, 2026-10-20; personally inspected in this bounded callback. '+item['description']
    appendix['artifacts'].append(item)
source_lookup={x['id']:x for x in fix['sources']}
for sid in ['unicode_current','utf8_current','gcb_current','ppo_current','r_posttraining','audit_execution_current','execution_current']:
    item=copy.deepcopy(source_lookup[sid]);item['id']='histfix_'+sid
    if item['kind']=='execution':
        item['artifact_id']='histfix_'+item['artifact_id']
    if item.get('retrieval'):
        item['retrieval']['artifact_id']='histfix_'+item['retrieval']['artifact_id']
    if 'inspection_note' in item:
        item['inspection_note']='Original fix primary read by /root/rv1019_technical_glossary, 2026-10-20, retained at the exact SHA/version; current actor directly read the original raw quotes/output, fixed rows and necessary current dependency. '+item['inspection_note']
    appendix['sources'].append(item)
appendix['sources'].append({'id':'s_historical_prompted_inspection','kind':'execution',
 'title':'Current actor prompted group04 original-issue/fix/current inspection receipt','verified':True,'artifact_id':'a_historical_prompted_inspection'})

character_evidence=[
 {'source_id':'histfix_unicode_current','locator':'UAX #29 r47 §3 Grapheme Cluster Boundaries / GB9; sources-current-01.json unicode_current',
  'supports':'User-perceived character need not be one code point; the retained original source explicitly distinguishes them.'},
 {'source_id':'histfix_gcb_current','locator':'Unicode17 GraphemeBreakProperty 1F3FB..1F3FF ; Extend; sources-current-01.json modifier_property',
  'supports':'U+1F3FD is Extend; together with GB9, the two-codepoint thumb/skin-tone example is a grapheme cluster.'},
 {'source_id':'histfix_utf8_current','locator':'RFC3629 §3–4; sources-current-01.json utf8_current',
  'supports':'1–4 bytes applies per scalar value, excluding surrogate U+D800–U+DFFF.'},
 {'source_id':'histfix_audit_execution_current','locator':'audit-output-current-01.json: diff.changed_rows, emoji, utf8_exhaustive',
  'supports':'Original raw fix contains the exact corrected row now at L16, two thumb codepoints/eight bytes and separately attributed original scalar-range check; no range rerun here.'},
 {'source_id':'s_historical_prompted_inspection','locator':'historical-child00-group04-inspection-v1.json: current_locations.character and row_comparisons',
  'supports':'Current actor read present L16 and verified exact equality to the authenticated original corrected row; the former one-to-one sentence is absent.'}]
ppo_evidence=[
 {'source_id':'histfix_ppo_current','locator':'Spinning Up PPO Background / Key Equations / You Should Know; sources-current-01.json ppo_current and focused a6 PPO short_quote',
  'supports':'PPO-Clip has no hard constraint; clipping reduces incentives and still may move the policy too far, with KL early stopping as a separate control.'},
 {'source_id':'histfix_r_posttraining','locator':'tiny_perceptron/posttraining.py: ppo_clipped_objective L31–52',
  'supports':'Current unchanged source computes raw ratio, clamps only the surrogate factor, then min; no policy-parameter/probability projection occurs.'},
 {'source_id':'histfix_execution_current','locator':'verification-output-current-01.json: ppo',
  'supports':'Original raw fixture leaves ratio≈1.8 while surrogate factor≈1.2, showing the actual ratio is not projected to the clipping range; not a new training test.'},
 {'source_id':'s_historical_prompted_inspection','locator':'historical-child00-group04-inspection-v1.json: current_locations.ppo, posttraining_dependency and row_comparisons',
  'supports':'Current actor read L180 and ppo_clipped_objective, verified exact fixed wording and full source SHA; no regression to every-update-small guarantee.'}]

rechecks=[]
inspections=[]
for index,row in enumerate(packet['historical_identities_only']):
    issue=pointer(read_reference(row['original_report']),row['json_pointer'])
    disposition=pointer(read_reference(row['disposition_ref']),row['disposition_ref']['json_pointer'])
    assert disposition['issue_id']==row['issue_id']
    assert disposition['final_necessity']==row['original_necessity']=='necessary'
    assert disposition['report']==row['original_report'] and disposition['json_pointer']==row['json_pointer']
    assert row['historical_fix_verification']['report']==fixref
    # Authenticate prior appendix bytes only; do not import its still_fixed row.
    assert sha(row['prior_appendix_row']['report']['path'])==row['prior_appendix_row']['report']['sha256']
    character='visual_character_codepoint' in row['issue_id']
    was_resolved=issue['status']=='resolved'
    identity_origin=('此identity指向原current的resolved紀錄；本人將其與原fix原始證據及當前句子比對，不以resolved狀態當本次證據。'
                     if was_resolved else '此identity指向原original的contradicted問題；本人保留原問題必要性，核對原修正文句與本次原碼／原文後獨立判斷。')
    reason=(identity_origin+
        ('現行L16仍逐一交代「少數字元由幾個碼點組成」、👍🏽與surrogate排除，與原audit.diff的after整列完全相同。原UAX #29/GB9、UCD Extend及RFC收據和原emoji原始輸出吻合；1–4byte只適用scalar而非整個視覺圖案。沒有以少數字串推跨語系字元比例，也未重跑全部scalar。'
         if character else
         '現行L180仍明示「但不保證每次的改變都很小」，與原audit.diff的after整列完全相同。本人再讀unchanged ppo_clipped_objective：raw ratio先算，clamp只用於surrogate，未投影策略機率；原raw fixture ratio約1.8仍在而surrogate約1.2。這和原Spinning Up no-constraint/too-far限制一致，並非因舊pass而保留，未做新訓練。'))
    result={key:copy.deepcopy(row[key]) for key in ['issue_id','disposition_ref','original_report','json_pointer','historical_fix_verification','original_necessity','prior_appendix_row']}
    result.update({'status':'still_fixed','verdict':'pass','owner':current['reviewer_task'],
                   'necessity':disposition['final_necessity'],'reason':reason,
                   'current_source_sha256':current['source_sha256'],'current_figure_sha256':copy.deepcopy(current['figure_sha256']),
                   'current_evidence':copy.deepcopy(character_evidence if character else ppo_evidence),
                   'artifact_ids':['a_historical_prompted_inspection','histfix_sources_current','histfix_diff_current','histfix_audit_current']+
                        ([] if character else ['histfix_replay_current','histfix_a6'])})
    rechecks.append(result)
    inspections.append({'issue_id':row['issue_id'],'original_issue':issue,
                        'accepted_final_necessity':disposition['final_necessity'],
                        'current_line':16 if character else 180,'current_quote':lines[15 if character else 179],
                        'fixed_row_exact_equal':True,'reason':reason,
                        'original_identity_fields':{key:copy.deepcopy(row[key]) for key in ['disposition_ref','original_report','json_pointer','historical_fix_verification']}})

inspection={'schema_version':1,'kind':'prompted_bounded_historical_fix_inspection',
 'created_at':datetime.now(UTC).isoformat(),'reviewer_task':current['reviewer_task'],
 'packet':ref(PACKET),'source_group_index':4,'prior_active_pointer':packet['prior_active_pointer'],
 'original_unavailability':packet['unavailability'],'current_report':ref(currentpath),
 'actual_current_intake_receipt':ref(receiptpath),'actual_current_received_at':receipt['received_at'],
 'source':{'path':sourcepath,'sha256':sha(sourcepath)},'figure_sha256':{},
 'current_locations':{'character':{'line':16,'quote':lines[15]},'ppo':{'line':180,'quote':lines[179]}},
 'row_comparisons':{'character_equal_original_fixed_after':lines[15]==character_change['after'],
                    'ppo_equal_original_fixed_after':lines[179]==ppo_change['after'],
                    'former_bad_rows_absent':character_change['before'] not in body and ppo_change['before'] not in body},
 'original_fix_report':fixref,'original_fix_actor':fix['reviewer_task'],'original_fix_date':'2026-10-20',
 'inspected_raw_artifacts':[{'id':x['id'],'path':x['path'],'recorded_sha256':x['sha256'],'actual_sha256':sha(x['path'])} for x in selected_artifacts],
 'original_primary_receipts':sources_raw,
 'original_emoji_fixture':audit['emoji'],'original_ppo_fixture':replay['ppo'],
 'posttraining_dependency':{'path':posttraining_path,'sha256':sha(posttraining_path),
                           'locator':'ppo_clipped_objective L31–52','finding':'ratio remains raw; only surrogate factor is clamped; no policy projection'},
 'issues':inspections,'prior_appendix_conclusions_used':False,
 'scope':'Only four authenticated identities/two existing fixes. No new mature-topic investigation, scalar-range replay, NN, training, final or global seal.',
 'command':COMMAND,'environment':{'python':platform.python_version(),'device':'CPU','operation':'local read/hash/row comparison and native schema validation only'},
 'result':'All four authenticated necessary historical identities remain fixed in present L16/L180, supported by original raw fix evidence and current exact dependency checks.'}
save(OUTPUT,inspection)
appendix['historical_issue_rechecks']=rechecks
appendix['artifacts'] += [
 {'id':'a_historical_prompted_code','kind':'code','path':ART+'historical-child00-group04-inspect.py','sha256':sha(ART+'historical-child00-group04-inspect.py'),
  'description':'Current actor packet-specific identity/source/raw-fix inspection; does not read prior appendix conclusions or rerun old scientific probes.'},
 {'id':'a_historical_prompted_inspection','kind':'execution','path':OUTPUT,'sha256':sha(OUTPUT),
  'description':'Actual prompted group04 inspection preserving all four identity fields, necessity and original unavailability, with current fixed clauses and dependency hashes.',
  'command':COMMAND,'result':inspection['result'],'environment':inspection['environment']}]
appendix['checks']=copy.deepcopy(current['checks'])
for name,check in appendix['checks'].items():
    if name=='figure_consistency':
        check['details']='No figures in the full glossary or the two issue-specific rows; current figure identity remains{}.'
    else:
        check['details']='Complete own current review retained unchanged by current_report_basis; this appendix adds only the four authenticated historical identities/two fixes. '+check['details']
appendix['checks']['factual_accuracy']['details']+=' Personally read original issues, original raw correction evidence and present L16/L180; current fixed rows exactly equal the original after rows.'
appendix['checks']['source_verification']['details']+=' Historical original/current/disposition/unavailability pointers authenticated; original raw fix artifacts and current posttraining source fingerprints match. Original/current historical records are distinct identities, not four separate new defects.'
appendix['checks']['limitations']['details']+=' No Unicode one-visible-character/one-codepoint guarantee or PPO every-update-small guarantee; no copied still_fixed, new unavailability or invented pre-stop receipt time.'
appendix['verdict']='pass'

spec=importlib.util.spec_from_file_location('glossary_historical_validator',ROOT/'scripts/check_technical_reviews.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
errors=validator._validate(ROOT,ROOT/sourcepath,'glossary',body,appendix,{current['reviewer_task']:['glossary']},set())
assert not errors,errors
sources={x['id'] for x in appendix['sources']};artifacts={x['id'] for x in appendix['artifacts']}
for row,request in zip(rechecks,packet['historical_identities_only'],strict=True):
    assert all(row[k]==request[k] for k in ['issue_id','disposition_ref','original_report','json_pointer','historical_fix_verification','original_necessity'])
    assert row['necessity']=='necessary' and row['status']=='still_fixed' and row['reason']
    assert all(x['source_id'] in sources and x['locator'] and x['supports'] for x in row['current_evidence'])
    assert row['artifact_ids'] and all(x in artifacts for x in row['artifact_ids'])
save(APPENDIX,appendix)
assert sha(currentpath)=='88bbcebc23e1570cc2b50a135bb006315ba25415b3f49bac9a0c8aeeb48838c2'
assert sha(originalpath)=='60ff0ebdea32be3a94018117ccba586ac67dbdc17ff6e07b658f55c5827f27b9'
print(json.dumps({'path':APPENDIX,'sha256':sha(APPENDIX),'created_at':appendix['created_at'],
 'verdict':appendix['verdict'],'native_errors':errors,'historical_issue_rechecks':len(rechecks),
 'packet':ref(PACKET),'inspection':ref(OUTPUT),'scope':'child-00/group-04 bounded only'},ensure_ascii=False))
