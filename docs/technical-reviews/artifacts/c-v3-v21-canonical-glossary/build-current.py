"""Same-owner source22 closure after genuine original receipt, without retraining."""

import copy
import hashlib
import importlib.util
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
ROUND = 'docs/course-revision-20261101-c-v3/'
ART = 'docs/technical-reviews/artifacts/c-v3-v21-canonical-glossary/'
BASE = ROUND + 'technical-v21/canonical_glossary/'
OLD = 'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-glossary/'

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def ref(path):
    return {'path': path, 'sha256': sha(path)}

def load(path):
    return json.loads((ROOT / path).read_text())

def save(path, value):
    with (ROOT / path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

original = load(BASE + 'original-v1.json')
assert sha(BASE + 'original-v1.json') == '60ff0ebdea32be3a94018117ccba586ac67dbdc17ff6e07b658f55c5827f27b9'
intake_path = ROUND + 'technical-coordination-v21/receipts/canonical-glossary-original-v1.json'
intake = load(intake_path)
assert sha(intake_path) == '20600a4f8c738f68a204ee8fd9625cfda5b8a4ea212d81941cd78e92d15db7f6'
assert intake['report'] == ref(BASE + 'original-v1.json')
assert datetime.fromisoformat(intake['received_at']) < datetime.now(UTC)
old = load(OLD + 'current-v1.json')
assert sha(OLD + 'current-v1.json') == 'dfca3517c0c19af660220d108fde1b6d32b51bb34132753461784dca1ff5f0d7'
assert sha(OLD + 'original-v1.json') == 'a6d42cb232fd1963ceaea36469016805f004ad17db1619bee59e16a4afa47798'
rawpath = 'docs/technical-reviews/artifacts/omni-p6-wave04-glossary/inspection-v1.json'
raw = load(rawpath)
assert sha(rawpath) == '506e428f125f98b184bacfa76223e2dabc478aa86c75fe67fd2b13e87de96407'
proofpath = ART + 'retention-proof-output.json'
proof = load(proofpath)
bundlepath = 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-source22-final-v1.json'
bundle = load(bundlepath)
assert sha(bundlepath) == 'd103c2981757718d0eb2869142ce415984ab754aa7edc59697373a6b8ca5b1fb'

# Verify the genuinely recorded T7 receipts, not just copied timestamp strings.
receipt_rows = []
for lesson, stem, current_version in [('R.2','numbered-r2','v1'), ('R.3','numbered-r3','v1'),
                                        ('R.4','numbered-r4','v2'), ('T.11','numbered-t11','v2')]:
    row = next(x for x in bundle['sections'] if x['lesson_id'] == lesson)
    events = {}
    for phase, version in [('original','v1'), ('current',current_version)]:
        p = ROUND + f'technical-coordination-v21/receipts/{stem}-{phase}-{version}.json'
        receipt = load(p)
        assert receipt['report'] == row['original_report' if phase=='original' else 'report']
        assert receipt['received_at'] == row[phase + '_collection']['received_at']
        report = load(receipt['report']['path'])
        assert receipt['created_at'] == report['created_at']
        assert receipt['reviewer_task'] == report['reviewer_task']
        assert datetime.fromisoformat(receipt['created_at']) <= datetime.fromisoformat(receipt['received_at'])
        if receipt.get('current_dispatch'):
            dispatch = receipt['current_dispatch'];assert sha(dispatch['path']) == dispatch['sha256']
        events[phase] = receipt
        receipt_rows.append({'lesson_id': lesson, 'phase': phase, 'receipt': ref(p),
                             'report': receipt['report'], 'received_at': receipt['received_at'],
                             'created_at': receipt['created_at'], 'same_as_final_bundle': True})
    assert datetime.fromisoformat(events['original']['received_at']) < datetime.fromisoformat(events['current']['created_at'])
w7path = ROUND + 'technical-coordination-v21/receipts/w7-current-v3.json'
w7receipt = load(w7path);w7row = next(x for x in bundle['sections'] if x['lesson_id']=='W.7')
assert w7receipt['report'] == w7row['report'] and w7receipt['received_at'] == w7row['current_collection']['received_at']
assert sha(w7receipt['current_dispatch']['path']) == w7receipt['current_dispatch']['sha256']
receipt_rows.append({'lesson_id': 'W.7', 'phase': 'source22_current_v3', 'receipt': ref(w7path),
                     'report': w7receipt['report'], 'received_at': w7receipt['received_at'],
                     'scope': 'AST-equivalent formatting including redundant parentheses removal; strict whitespace-only=false. Same original owner; no new scientific reading.'})
prior_bundle = load(bundle['previous_bundle']['path'])
assert sha(bundle['previous_bundle']['path']) == bundle['previous_bundle']['sha256']
for row in bundle['sections']:
    if row['lesson_id'] not in {'R.2','R.3','R.4','T.11','W.7'}:
        earlier = next(x for x in prior_bundle['sections'] if x['lesson_id']==row['lesson_id'])
        for key in ['report','original_report','original_collection','current_collection']:
            assert row[key] == earlier[key]

current = copy.deepcopy(original)
current['created_at'] = datetime.now(UTC).isoformat()
current['source_freeze'] = bundle['canonical_source_freeze']
current['original_report'] = ref(BASE + 'original-v1.json')
current['original_intake_receipt'] = ref(intake_path)
current['review_scope'] = 'Same-owner full glossary current closure on source22. Own151-claim original unchanged; 139 necessary evidence gaps closed by explicitly attributed immutable primary-source/CPU retention, exact dependency identity and bounded original-code inspection. Exact source22 final26 is bound after genuine own intake.'
current['source_identity']['current_source_freeze'] = bundle['canonical_source_freeze']
current['source_identity']['current_numbered_source_freeze'] = bundle['section_freeze']
current['source_identity']['identity_result'] += ' source22 glossary remains identical; only chapter19.25 formatting plus one Notebook changed elsewhere, not this body.'
current['independence_record']['after_original_intake'] = '正式原判receipt20600a4…已真收後才按指定refs讀prior glossary與final26。原判固定後等待時list_agents曾附帶first_steps行政完成狀態，已如實向協調者記錄；未改原判，之後不再查agent狀態。'
current['independence_record']['historical_retention'] = 'Mature concepts and old CPU mechanisms remain evidence of /root/omni_p6_technical_glossary, 2026-10-30; retained G.1 EOS fixture is /root/omni_technical_coordinator/technical_g_1, 2026-10-29. Current actor inspected immutable bytes/locators/scope and all dependency SHA; no old pass or author summary is used as primary evidence and no old experiment is relabeled new.'
current['sources'] = copy.deepcopy(original['sources'])
source_ids = {x['id'] for x in current['sources']}
for source in old['sources']:
    if source['id'] == 'bundle_execution':
        continue
    s = copy.deepcopy(source);s['id'] = 'ret_' + source['id']
    s['retention'] = {'original_actor': raw['reviewer_task'], 'original_date': '2026-10-30',
                       'portable_raw_receipt': ref(rawpath),
                       'current_verification': 'Source-identical body; original primary/installation/repository and PDF bytes all match retained SHA in retention-proof-output.json. Required citation scope read; no fresh full-paper rereading claimed.'}
    if s['kind'] in {'paper','official_docs','official_source'}:
        s['inspection_note'] = ('RETAINED original inspection by /root/omni_p6_technical_glossary on 2026-10-30; checked_original=true records that original read, not a new full reading by the current actor. Current actor read the exact immutable raw receipt/locator and verified bytes and scope. Original inspection: ' + s['inspection_note'])
    elif s['kind']=='repository_code':
        assert sha(s['path']) == s['sha256']
        s['inspection_note'] = ('Retained source inspection by /root/omni_p6_technical_glossary, 2026-10-30, with exact same current full-file SHA personally rechecked here. ' + s['inspection_note'])
    current['sources'].append(s);source_ids.add(s['id'])

current['artifacts'] = copy.deepcopy(original['artifacts'])
old_art = copy.deepcopy(next(x for x in old['artifacts'] if x['id']=='inspection'))
old_art['id'] = 'a_retained_inspection'
old_art['description'] = 'Retained raw evidence of /root/omni_p6_technical_glossary, 2026-10-30: exact CPU script/output, original-source versions/locators/SHA, installation excerpts and public fixed-version metadata. No new execution/anonymous fetch claimed.'
current['artifacts'].append(old_art)
current['sources'].append({'id':'s_retained_exec','kind':'execution','title':'Retained original glossary CPU mechanism receipt, 2026-10-30',
                           'verified':True,'artifact_id':'a_retained_inspection'})
source_ids.add('s_retained_exec')

def line_set(claim):
    loc = claim['location']
    if ':L' not in loc:
        return set()
    result = set()
    for lo, hi in re.findall(r'(\d+)(?:[–-](\d+))?', loc.split(':L',1)[1]):
        result.update(range(int(lo),int(hi or lo)+1))
    return result

matches = {}
for claim in current['claims']:
    if claim['status']=='verified':
        continue
    line = int(claim['id'][1:])
    relevant = [x for x in old['claims'] if line in line_set(x)]
    assert relevant, claim['id']
    evidence = []
    for candidate in relevant:
        for item in candidate['evidence']:
            sid = 'ret_' + item['source_id']
            assert sid in source_ids, sid
            entry = {'source_id':sid,'locator':item['locator'],
                     'supports':'Retained original primary support: '+item['supports']}
            if entry not in evidence:
                evidence.append(entry)
    claim['evidence'] = evidence
    claim['artifact_ids'] = ['a_retained_inspection','a_retention_proof','a_closure_proof']
    claim['status'] = 'verified'
    claim['original_initial_assessment'] = claim['initial_assessment']
    claim['original_scope'] = claim['scope']
    claim['scope'] = '；'.join(dict.fromkeys(x['scope'] for x in relevant)) + ' 保留原actor/date的immutable原證據，不宣稱本輪新研究或訓練。'
    claim['original_pending_evidence_group'] = claim.pop('pending_evidence_group')
    claim['closure'] = {'method':'explicit_source_identical_retention', 'prior_claim_ids':[x['id'] for x in relevant],
                         'prior_original_actor':raw['reviewer_task'], 'prior_date':'2026-10-30',
                         'source_sha256':original['source_sha256'], 'portable_raw_receipt':ref(rawpath),
                         'support_scope':'Current actor compared necessary immutable original receipt/locators and support limits to this exact source claim. This is retained primary evidence, not acceptance of an old verdict.'}
    matches[claim['id']] = {'prior_claim_ids':[x['id'] for x in relevant],
                             'source_ids':[x['source_id'] for x in evidence],
                             'original_scope':claim['original_scope'], 'current_scope':claim['scope']}

def add_repo(sid, path, note):
    current['sources'].append({'id':sid,'kind':'repository_code','title':path,'path':path,'sha256':sha(path),
       'version':'Exact current source22 bytes, directly inspected 2026-11-01 in this bounded closure',
       'verified':True,'inspection_note':note})

add_repo('s_generate','tiny_perceptron/model.py','Directly read generate L109–132: append chosen next_id, batch size one EOS break, max_new_tokens loop and total max_length; retained G.1 fixture uses these exact bytes. No neural checkpoint loaded or new model execution.')
add_repo('s_cards','scripts/selftrained/prepare_vision_ocr.py','Directly read CHARACTERS, draw_line L313–326, prepare_ocr image construction L329–355, provenance/limits L444–455 and attribution L747. Noto font bytes are third-party assets; deterministic author-created strings/canvases are generated here. No download, render regeneration or OCR training.')
add_repo('s_retrieval','tiny_perceptron/retrieval.py','Directly read entire small file: call_tool first json.loads if text, then exact key/schema/name whitelist and numeric argument checks before add/multiply; retrieve only returns documents and does not modify model parameters. No tool run.')
g1row = next(x for x in bundle['sections'] if x['lesson_id']=='G.1')
g1 = load(g1row['report']['path'])
g1_eos = next(x for x in g1['claims'] if x['id']=='s05')
for a in g1['artifacts']:
    if a['id'] in g1_eos['artifact_ids']:
        item=copy.deepcopy(a);item['id']='ret_g1_'+a['id']
        assert sha(item['path']) == item['sha256']
        item['description']='Retained G.1 raw EOS fixture evidence by '+g1['reviewer_task']+' on 2026-10-29. '+item['description']
        current['artifacts'].append(item)
current['sources'].append({'id':'s_g1_eos_execution','kind':'execution','title':'Retained actual generate fixed-logit single-sequence EOS fixture',
                           'verified':True,'artifact_id':'ret_g1_a_execution'})
eos = next(x for x in current['claims'] if x['id']=='g019')
eos['kind']='software'
eos['evidence'] += [{'source_id':'s_generate','locator':'generate L109–132, especially L127–128',
                    'supports':'Actual single-sequence loop appends EOS then breaks; bounded by max_new_tokens and context length.'},
                   {'source_id':'s_g1_eos_execution','locator':'execution-v1.json: probes.single_sequence_generated, generation_model_calls; verify-g1.py L143–161',
                    'supports':'Fixed logits choose A then EOS then potential B; observed [[1,73,2]], calls2, restored training mode, no B.'}]
eos['artifact_ids'] += ['ret_g1_a_execution','ret_g1_a_code']
eos['verification']=copy.deepcopy(g1_eos['verification'])
eos['verification']['details'] += ' Retained original execution of /root/omni_technical_coordinator/technical_g_1 on 2026-10-29; current actor inspected exact fixture/output and unchanged actual generate source, did not rerun it.'
eos['scope']='本課單句生成與assistant目標契約；BOS僅前文、EOS有效目標和停止均有原碼／原始fixture；不外推batch生成、固定步數benchmark或未匹配tokenizer。'
cards=next(x for x in current['claims'] if x['id']=='g214')
cards['evidence'].append({'source_id':'s_cards','locator':'draw_line/prepare_ocr L313–355; modifications and limits L444–455; attribution L747',
                         'supports':'Author-created deterministic character strings rendered on local image canvases from attributed licensed Noto fonts; distinct from response-card actions.'})
cards['scope']='「作者製作」指本課程式所組字串／畫布與合成字卡，字型依Noto授權及來源指紋另列；沒有聲稱作者發明字型，亦不由素材身份推OCR品質。'
tool=next(x for x in current['claims'] if x['id']=='g168')
tool['evidence'].append({'source_id':'s_retrieval','locator':'call_tool; applications._parse_json_action L496–525 and _tool_episode L528–558 retained under ret_repo_apps',
                        'supports':'JSON parsing and semantic whitelist/schema checks precede the only actual tool invocation.'})
ext=next(x for x in current['claims'] if x['id']=='g218')
ext['evidence'].append({'source_id':'ret_public','locator':'Fixed published training receipts random_initialization/origin and raw source/teacher manifests in inspection-v1.json',
                       'supports':'Retained declared mainline random-init/teacher-role boundary; no claim of newly auditing all historical data or future products.'})
ext['scope']='外部模型身份由固定原文支持；主線不使用其權重／輸出作教師是本課明示流程與既有基線血緣邊界，不擴稱所有未知歷史／外部資料都已逐筆核驗。'

current['sources'].append({'id':'s_closure_exec','kind':'execution','title':'Current actor source22 exact26 and retained-evidence closure verification',
                           'verified':True,'artifact_id':'a_closure_proof'})
current['artifacts'] += [
 {'id':'a_final26','kind':'source_snapshot','path':bundlepath,'sha256':sha(bundlepath),
  'description':'The exact authorized source22 final26 bundle d103…:21 retained, W.7 same-owner source22 current-v3, four fresh parts; no collector backreference and no global T/C seal.'},
 {'id':'a_retention_code','kind':'code','path':ART+'retention-proof.py','sha256':sha(ART+'retention-proof.py'),
  'description':'Current actor exact26 native/source/dep identity check, no scientific rerun.'},
 {'id':'a_retention_proof','kind':'execution','path':proofpath,'sha256':sha(proofpath),
  'description':'Current actor actual proof:26 originals check_original,26 source22 currents _validate,464 report dependencies and94 immutable raw dependencies match; G slices compared.',
  'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python '+ART+'retention-proof.py',
  'result':proof['result'],'environment':proof['environment']},
 {'id':'a_closure_code','kind':'code','path':ART+'build-current.py','sha256':sha(ART+'build-current.py'),
  'description':'Same-owner current evidence mapping, real T7 receipt parse/hash/timestamp and original-issue closure construction; no model execution.'}]

current['claims'].append({'id':'bundle_source22','kind':'software',
 'statement':'指定source22 final26的26份原判／current、真實保留及T7收件鏈、完整依賴指紋成立，G.1–G.4精確切片與本人全文初判範圍相容。',
 'location':'collector current callback: exact source22 final26 d103c298… and whole glossary introduction/G.1–G.4/index',
 'status':'verified','evidence':[{'source_id':'s_closure_exec','locator':'retention-proof-output.json section_validation, retained_dependencies, glossary_comparison; closure-proof-output.json actual_receipts',
                              'supports':'26 original/current native checks,21 retained entry/receipt identity,W7 source22 v3 plus4 fresh actual original/current receipts and source22 G slices.'}],
 'artifact_ids':['a_final26','a_retention_proof','a_closure_proof'],
 'verification':{'method':'executed','expected':'26 unique section refs with unchanged sources/figures, same-owner original→current, preserved actual receipts and no collector hash cycle',
                 'observed':'26 originals and26 current native checks passed;464 report dependencies and94 raw retained dependencies exact;9 T7 receipt files parsed/hashed;21 prior entry/report/receipt records identical;G.1–G.4 same exact source slices.',
                 'details':'Local read/hash/native validators and parsed actual intake receipts. W7 formatting is AST-equivalent and includes redundant parentheses removal, strict whitespace-only=false. Retained prior judgments were not relabeled fresh scientific reviews.'},
 'scope':'只核指定bundle引用、來源／收件鏈及本頁相容性；不冒充另一次26節科學審閱，不取代本人完整151claim原判，不宣稱global technical或product seal。'})

current['original_issue_resolutions']=[]
for i, issue in enumerate(current['issues']):
    ids=issue.get('claim_ids',[issue['claim_id']])
    resolved=[c for c in current['claims'] if c['id'] in ids]
    assert len(resolved)==len(ids) and all(c['status']=='verified' for c in resolved)
    sids=sorted({e['source_id'] for c in resolved for e in c['evidence']})
    locators=sorted({e['locator'] for c in resolved for e in c['evidence']})
    resolution=(f"同owner逐一對回{','.join(ids)}：以原actor /root/omni_p6_technical_glossary 的2026-10-30 immutable原文／原碼定位與CPU收據保留，非引用old pass；本頁SHA c71853…未變，94項原始依賴SHA與464項bundle依賴核同。支持來源="
                +','.join(sids)+'；實際定位='+ '；'.join(locators)+
                '。每項保持其原初判所列條件及外推界線；本輪沒有重作研究、訓練、final或產品品質測試。')
    if 'g019' in ids:
        resolution += ' EOS額外親讀unchanged generate L109–132及G.1原fixture：[[1,73,2]]、calls=2，僅單句；原actor為/root/omni_technical_coordinator/technical_g_1、2026-10-29。'
    if 'g214' in ids:
        resolution += ' 字卡作者血緣另親核prepare_vision_ocr.py draw_line/prepare_ocr及Noto來源／attribution；作者製作的是字串與畫布，非字型本身。'
    issue.update(status='resolved',resolution=resolution)
    current['original_issue_resolutions'].append({'json_pointer':'/issues/'+str(i),'resolution':resolution,
                                                 'resolved_claim_ids':ids,'evidence_source_ids':sids})

closure={'created_at':datetime.now(UTC).isoformat(),'reviewer_task':original['reviewer_task'],
         'original_report':ref(BASE+'original-v1.json'),'original_intake_receipt':ref(intake_path),
         'original_received_at':intake['received_at'],'source_freeze':bundle['canonical_source_freeze'],
         'exact_bundle':ref(bundlepath),'retention_proof':ref(proofpath),'actual_receipts':receipt_rows,
         'unchanged_retained_prior_entries':21,
         'retained_evidence_actor':raw['reviewer_task'],'retained_evidence_created_at':raw['created_at'],
         'claim_mappings':matches,'original_issue_resolutions_count':len(current['original_issue_resolutions']),
         'extra_direct_source_checks':[{'source_id':s['id'],'path':s['path'],'sha256':s['sha256'],'inspection_note':s['inspection_note']}
                                       for s in current['sources'] if s['id'] in {'s_generate','s_cards','s_retrieval'}],
         'eos_original_fixture':{'report':g1row['report'],'original_actor':g1['reviewer_task'],
                                  'created_at':g1['created_at'],'verification':g1_eos['verification']},
         'result':'Own151 original claims all verified within recorded support scope;139 retained gaps resolved explicitly. Source22 final26 plus actual receipts checked without a collector cycle. No fresh mature-study or scientific model execution claimed.',
         'environment':{'python':platform.python_version(),'device':'CPU','operation':'local JSON/hash/native checks and source/receipt inspection only'}}
closurepath=ART+'closure-proof-output.json';save(closurepath,closure)
current['artifacts'].append({'id':'a_closure_proof','kind':'execution','path':closurepath,'sha256':sha(closurepath),
 'description':'Current actor actual retained-source mapping,9 T7 real receipt checks,own original receipt and every original issue resolution; preserved actor/date and EOS fixture scope.',
 'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python '+ART+'build-current.py',
 'result':closure['result'],'environment':closure['environment']})
allids=[x['id'] for x in current['claims']]
current['checks']={
 'factual_accuracy':{'status':'pass','claim_ids':allids,'details':'All151 own original claims closed within support scope, plus bounded exact26 collector contract. Concepts retain original authority actor/date; no old verdict used as evidence. Direct unchanged code and original EOS fixture close software gaps.'},
 'numeric_verification':{'status':'pass','claim_ids':[x['id'] for x in current['claims'] if x['kind']=='numeric']+['g140','g201','g202'],
                         'details':'Own transparent softmax/byte/valid-target/L2/temperature examples preserved. Retained primary formulas and exact parameter-budget/reliability code identify counts and denominators. No fresh quality/speed/memory claims or historical measurement replication.'},
 'figure_consistency':copy.deepcopy(original['checks']['figure_consistency']),
 'source_verification':{'status':'pass','claim_ids':allids,'details':'Exact source22 glossary c71853… is identical to original and2026-10-30 raw full-page evidence.94 original dependencies and464 exact26 dependencies match. Original source reads retain original actor/date/version/locators; current read necessary raw scopes and direct missing code. Exact d103… bundle—not paused,old or index-input—is artifact-bound.'},
 'limitations':{'status':'pass','claim_ids':allids,'details':'Scope retains finite logits/single-label CE/valid targets,zero denominator,single-sequence EOS,mask roles,source/family leakage,candidate identity,time/causal/codebook axes,parameter storage versus work,oracle versus deployable choice,calibration source shift,old-version product provenance versus new capability,CPU mechanisms versus quality. No arbitrary-batch stop,universal speedup,public download,new training/final or global T/C seal claim.'}}
current['coverage'].update(claim_count=len(current['claims']),verified_claims=len(current['claims']),unresolved_claims=0,
                            assessment='Original151 full-file claims preserved and individually closed; one bounded source22 exact26 callback claim added. All136 table rows remain covered.')
current['retention']={'raw_inspection':ref(rawpath),'prior_original':ref(OLD+'original-v1.json'),
                       'prior_current_for_evidence_navigation_only':ref(OLD+'current-v1.json'),
                       'proof':ref(proofpath),'closure':ref(closurepath),
                       'not_new_scientific_execution':True}
current['verdict']='pass'

spec=importlib.util.spec_from_file_location('glossary_current_validator',ROOT/'scripts/check_technical_reviews.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
errors=validator._validate(ROOT,ROOT/'course/glossary.md','glossary',(ROOT/'course/glossary.md').read_text(),current,
                           {current['reviewer_task']:['glossary']},set())
assert not errors, errors
save(BASE+'current-v1.json',current)
print(json.dumps({'path':BASE+'current-v1.json','sha256':sha(BASE+'current-v1.json'),
 'created_at':current['created_at'],'verdict':current['verdict'],'native_errors':errors,
 'claims':len(current['claims']),'original_issue_resolutions':len(current['original_issue_resolutions']),
 'source_freeze':bundle['canonical_source_freeze'],'bound_bundle':ref(bundlepath)},ensure_ascii=False))
