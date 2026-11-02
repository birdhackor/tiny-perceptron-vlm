"""原兩條裝置通則問題的有界prompted inspection與獨立appendix。"""

import ast
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.omni_revision import review_setup
from tiny_perceptron.training import choose_device

ROOT=Path('/tmp/tpb').resolve()
HERE=Path(__file__).resolve().parent
ROUND='docs/course-revision-20261101-c-v3/'
TASK='/root/c_v3_technical_coord/canonical_training'
PACKET=ROUND+'technical-coordination-v21/history-prompts/child-00-group-05-historical-identities.json'
PACKET_SHA='25a691b616a3897c7e62ad2e67174bd572b99c83c02aeec58175fbb72380fda9'
OUT=ROUND+'technical-v21/canonical_training/historical-child-00-group-05-appendix-v1.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {'path':path,'sha256':sha((ROOT/path).read_bytes())}


def load(path):
    return json.loads((ROOT/path).read_bytes())


def verified(reference):
    assert sha((ROOT/reference['path']).read_bytes())==reference['sha256']
    return load(reference['path'])


def pointer(value,location):
    for key in location.strip('/').split('/') if location else []:
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


assert ref(PACKET)['sha256']==PACKET_SHA
packet=load(PACKET);assert packet['source_group_index']==5 and packet['page_id']=='training'
current_path=ROUND+'technical-v21/canonical_training/current-v1.json'
original_path=ROUND+'technical-v21/canonical_training/original-v1.json'
assert ref(current_path)['sha256']=='f9048f741959652b22fe9d45b1f8283c2d97c184e53a9e0edd5e0f7fe101a041'
assert ref(original_path)['sha256']=='2f936e78a12c68fa409118a0ad7e62c901ecf6461f1aa882a62bf52a5124e043'
receipt_path=ROUND+'technical-coordination-v21/receipts/canonical-training-current-v1.json'
assert ref(receipt_path)['sha256']=='2224fbd626f8c466ca65e5f6507fbd6e176c3eecf9f20d26a30c8ae292662fc1'
receipt=load(receipt_path);assert receipt['report']==ref(current_path)
started=datetime.now(UTC).isoformat();assert started>receipt['received_at']
body=(ROOT/'course/training.md').read_text();source_sha=sha(body.encode())
assert source_sha=='4e304bb370cf257e185122b8eddd9a1422608f27cb784d438faafcd8b5e6a990'
line9=body.splitlines()[8]
for phrase in ('`scripts/train.py`、`evaluate.py`、`infer.py`','`scripts.course_experiments.run` 沒指定時用 CPU',
               '`quantize.py` 固定用 CPU','這三個工具都沒有 `--device` 參數','沒有在 MPS 上驗證整套訓練'):
    assert phrase in line9,phrase
paths=['scripts/train.py','scripts/evaluate.py','scripts/infer.py','tiny_perceptron/training.py',
       'scripts/quantize.py','scripts/prepare_data.py','scripts/fetch_training_assets.py','scripts/course_experiments/run.py']
parsers={};trees={}
for path in paths:
    tree=ast.parse((ROOT/path).read_text());trees[path]=tree;arguments=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args:
            if isinstance(node.args[0],ast.Constant) and node.args[0].value=='--device':
                arguments.append({keyword.arg:ast.literal_eval(keyword.value) for keyword in node.keywords if keyword.arg in ('default','choices')})
    parsers[path]=arguments
for path in paths[:3]:assert parsers[path]==[{'default':'auto'}]
for path in paths[4:7]:assert parsers[path]==[]
assert parsers[paths[7]]==[{'choices':('cpu','cuda'),'default':'cpu'}]
loader=next(node for node in trees['tiny_perceptron/training.py'].body if isinstance(node,ast.FunctionDef) and node.name=='load_checkpoint')
assert [node.arg for node in loader.args.args][:2]==['path','device'] and ast.literal_eval(loader.args.defaults[0])=='cpu'
calls=[node for node in ast.walk(trees['scripts/quantize.py']) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='load_checkpoint']
assert len(calls)==1 and len(calls[0].args)==1 and not calls[0].keywords
cpu=str(choose_device('cpu'));auto=str(choose_device('auto'));assert cpu=='cpu'
if not torch.cuda.is_available() and not torch.backends.mps.is_available():assert auto=='cpu'
commands=[]
for args in (['scripts/quantize.py',str(HERE/'not-present-never-opened.pt'),'--device','cpu'],
             ['scripts/prepare_data.py','--device','cpu'],['scripts/fetch_training_assets.py','--list','--device','cpu']):
    process=subprocess.run([sys.executable,*args],cwd=ROOT,text=True,capture_output=True,check=False)
    assert process.returncode==2 and 'unrecognized arguments' in process.stderr and '--device' in process.stderr
    commands.append({'argv':[sys.executable,*args],'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr})
assert not (HERE/'not-present-never-opened.pt').exists()
old_fix_path=packet['historical_identities_only'][0]['historical_fix_verification']['report']['path']
fix=verified(packet['historical_identities_only'][0]['historical_fix_verification']['report'])
fix_claim=next(row for row in fix['claims'] if row['id']=='device_generalization')
fix_execution=next(row for row in fix['artifacts'] if row['id']=='current_execution')
old_current=verified(fix_execution)
original_audit=next(row for row in fix['artifacts'] if row['id']=='audit');audit=verified(original_audit)
assert all(row['returncode']==2 for row in audit['runs'][:3])
assert [row['auto'] for row in old_current['observations']['device_priority']]==['cuda','mps','cpu']
assert all(row['explicit_cpu']=='cpu' for row in old_current['observations']['device_priority'])
identities=[]
for row in packet['historical_identities_only']:
    disposition=pointer(verified(row['disposition_ref']),row['disposition_ref']['json_pointer'])
    original=verified(row['original_report']);issue=pointer(original,row['json_pointer'])
    assert disposition['issue_id']==row['issue_id'] and disposition['final_necessity']=='necessary'
    assert issue['claim_id']=='device_generalization' and issue['necessary'] is True
    assert row['historical_fix_verification']==packet['historical_identities_only'][0]['historical_fix_verification']
    # Verify the historical reference bytes only; do not read/copy prior still_fixed judgments.
    assert ref(row['prior_appendix_row']['report']['path'])['sha256']==row['prior_appendix_row']['report']['sha256']
    identities.append({'issue_id':row['issue_id'],'original_issue':issue,'original_owner':original['reviewer_task'],
                        'original_necessity':row['original_necessity'],'final_necessity':disposition['final_necessity'],
                        'four_identity_fields':{key:row[key] for key in ('disposition_ref','original_report','json_pointer','historical_fix_verification')}})
verified(packet['source_chain']);verified(packet['prior_active_pointer'])
unavailable=pointer(verified(packet['unavailability']['report']),packet['unavailability']['json_pointer'])
inspection={'started_at':started,'completed_at':datetime.now(UTC).isoformat(),'actor':TASK,
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'CPU parser/static only','root':str(ROOT)},
            'scope':'Only original device_generalization fix now; zero model instantiation/NN/training/final/research/download. Three genuine parser failures are expected contract evidence, not failed training.',
            'packet':ref(PACKET),'full_current_received':{'reference':ref(receipt_path),'received_at':receipt['received_at']},
            'source_sha256':source_sha,'current_line9':line9,'repository_sources':{path:ref(path) for path in paths},
            'device_argument_AST':parsers,'quantize_cpu_loader_contract':'quantize passes only checkpoint; load_checkpoint device default is cpu',
            'current_actual_choose_device':{'explicit_cpu':cpu,'auto':auto,'cuda_available':torch.cuda.is_available(),'mps_available':torch.backends.mps.is_available()},
            'current_parser_failures':commands,'original_identities':identities,
            'historical_fix_evidence':{'report':ref(old_fix_path),'claim':fix_claim,'execution_reference':ref(fix_execution['path']),
                                      'original_audit_reference':ref(original_audit['path']),
                                      'original_three_rejected_commands':audit['runs'][:3],
                                      'original_device_priority_mock_cases':old_current['observations']['device_priority'],
                                      'limits':'Original CUDA/MPS priority cases were mocked. They are never described as real CUDA/MPS training. Only the original fix semantics and unchanged current code are retained.'},
            'unavailability_preserved':packet['unavailability'],'authenticated_old_unavailability_record':unavailable,
            'prior_active_pointer_preserved':packet['prior_active_pointer'],'new_unavailability_events_created':False,
            'judgment':'Both original identity rows remain necessary; qualified current text and exact current parser/CPU logic demonstrate still_fixed independently of old pass/still_fixed.'}
inspection_path=str((HERE/'history-child-00-group-05-inspection-v1.json').relative_to(ROOT))
with (ROOT/inspection_path).open('x',encoding='utf-8') as stream:json.dump(inspection,stream,ensure_ascii=False,indent=2);stream.write('\n')

sources=[]
for i,path in enumerate(['course/training.md',*paths]):
    sources.append({'id':'s'+str(i),'kind':'repository_code','title':'Issue-specific current source: '+path,**ref(path),
                    'version':'source22 current bytes personally inspected after actual full-current intake', 'verified':True,
                    'inspection_note':'只核歷史裝置通則的第9行、--device parser/default/choices、choose_device優先順序或quantize CPU loader；本次AST/三個parser拒絕與actual cpu選擇在a_inspection，不重新模型訓練。'})
sources.append({'id':'s_inspection','kind':'execution','title':'本人兩原identity的當前修正核對','verified':True,'artifact_id':'a_inspection'})
artifacts=[{'id':'a_inspection','kind':'execution',**ref(inspection_path),
            'description':'本人實讀兩original issues、accepted dispositions、原fix actual raw，再核source22第9行與8個入口依賴、AST與三個真parser拒絕；不讀/複製舊still_fixed判定。',
            'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-training/history_child_00_group_05.py',
            'result':'三個不接受--device的工具均真正exit2/unrecognized arguments，資料與權重未開啟；train/evaluate/infer AST default auto，run AST default cpu；quantize loader CPU；本機actual explicit cpu/auto=cpu。兩歷史necessary identity目前仍修妥。',
            'environment':inspection['environment']},
           {'id':'a_inspection_code','kind':'code',**ref(str(Path(__file__).relative_to(ROOT))),'description':'本人有界prompted inspection與appendix保存程式，不呼叫訓練/模型推論。'},
           {'id':'a_fix_execution',**{key:value for key,value in fix_execution.items() if key!='id'},
            'description':'原審閱者已保存的裝置優先順序/CLI原fix執行，不在本輪重跑原模型；mock CUDA/MPS與實CPU範圍保持。'},
           {'id':'a_original_audit',**{key:value for key,value in original_audit.items() if key!='id'},
            'description':'原必要問題的三個真不接受--device反例raw stdout/stderr；原bytes/hash保持，只作歷史原fix身份。'},
           {'id':'a_final26','kind':'source_snapshot',**ref('docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-source22-final-v1.json'),
            'description':'已在完整current親核的真source22 final26 d103唯一bundle；此appendix只重核本packet兩歷史修正，不代整頁current或globalTseal。'}]
assert artifacts[-1]['sha256']=='d103c2981757718d0eb2869142ce415984ab754aa7edc59697373a6b8ca5b1fb'
def evidence(identifier,locator,supports):return {'source_id':identifier,'locator':locator,'supports':supports}
current_evidence=[evidence('s0','course/training.md L9「Python怎麼寫」','現行通則只限定train/evaluate/infer；MPS未整套驗證、run預設CPU、quantize固定CPU及三工具無--device均明列。'),
                  evidence('s4','choose_device L13–20、load_checkpoint device=cpu','自動選擇按CUDA/MPS/CPU，explicit cpu固定CPU；不從route邏輯推出實GPU/MPS完成訓練。'),
                  evidence('s_inspection','device_argument_AST/current_parser_failures/current_actual_choose_device/historical_fix_evidence','本人已核原fix實際raw與當前八入口；三不支持device工具實際parse拒絕而未觸發資料/權重；current仍符合原修正的限定條件。')]
claims=[{'id':'h_device_auto_scope','kind':'software','statement':'training現行裝置自動選擇敘述只限採用choose_device(auto)的train/evaluate/infer入口，未宣稱MPS整套訓練已驗證。',
         'location':'L9「Python怎麼寫」；scripts/train/evaluate/infer --device與tiny_perceptron/training.choose_device',
         'status':'verified','evidence':current_evidence,'artifact_ids':['a_inspection','a_fix_execution'],
         'verification':{'method':'executed','expected':'三本機入口default auto；choose_device CUDA→MPS→CPU，explicit cpu固定，正文限定適用入口及MPS限制。',
                         'observed':'本人AST核三入口default auto與現行source；actual explicit cpu/auto均cpu。原fix raw的三個mock優先組合依序cuda/mps/cpu且explicit cpu均cpu。',
                         'details':'本次只做AST、真parser/CPU device選擇與已保存原fix raw核對；沒有NN或硬體品質實驗，mock不可冒作實GPU/MPS驗證。'},
         'scope':'本packet兩歷史修正現在仍有效；不重新驗收跨平台訓練或推論品質。'},
        {'id':'h_device_exceptions','kind':'software','statement':'training現行第9行正確明列run預設CPU、quantize固定CPU，以及quantize/prepare/fetch三工具不接受--device。',
         'location':'L9末三句；run.main parser、quantize.main→load_checkpoint、prepare/fetch parser',
         'status':'verified','evidence':[current_evidence[0],current_evidence[2],evidence('s8','scripts/course_experiments/run.py main --device','default cpu且choices僅cpu/cuda，未給時不走auto')],
         'artifact_ids':['a_inspection','a_original_audit','a_fix_execution'],
         'verification':{'method':'executed','expected':'三工具--device cpu均parse拒絕；run未給參數default cpu；quantize CPU loader不接受--device覆寫。',
                         'observed':'三個真命令均returncode2及unrecognized arguments --device；AST三parser無device，run default cpu，quantize單參數load_checkpoint且loader default cpu。',
                         'details':'參數解析在權重/資料開啟前中止。保留原audit三反例與原fix raw；本次沒有呼叫固定execute或訓練器。'},
         'scope':'只驗原必要修正的工具參數與正文範圍，不新增模型、資料或能力結論。'}]
rechecks=[]
for i,row in enumerate(packet['historical_identities_only']):
    value=copy.deepcopy(row)
    value.update({'owner':TASK,'necessity':'necessary','status':'still_fixed','verdict':'pass',
                  'reason':('原original-01' if i==0 else '原original-02（其source-locator補強未改此必要問題）')+'的/ issues/0都反對把auto/--device cpu通則套到所有CLI。本人讀兩份原issue、accepted final_necessity=necessary及原fix真正CLI/mocked priority收據後，再核現行第9行與八入口：正文明確限定train/evaluate/infer，run default cpu，quantize CPU loader，三工具無device且真parse拒絕。因此原修正目前仍成立。判斷依當前raw/AST/三實命令，不複製舊pass或still_fixed；兩原identity分別保留。',
                  'current_source_sha256':source_sha,'current_figure_sha256':{},'current_evidence':copy.deepcopy(current_evidence),
                  'artifact_ids':['a_inspection','a_fix_execution','a_original_audit','a_final26'],'claim_ids':['h_device_auto_scope','h_device_exceptions']})
    rechecks.append(value)
appendix={'schema_version':1,'review_stage':'technical','lesson_id':'training','source':'course/training.md','reviewer_task':TASK,'reviewer_context':'fresh',
          'created_at':datetime.now(UTC).isoformat(),'source_sha256':source_sha,'figure_sha256':{},'verdict':'pass',
          'scope':'Bounded historical appendix for child-00/group05 only. Same actor full original/current genuinely received first. Two exact old device_generalization identities rechecked now, not replacement full review, new research or globalT completion.',
          'full_current':ref(current_path),'full_original':ref(original_path),'full_current_intake_receipt':ref(receipt_path),
          'historical_packet':ref(PACKET),'source_chain':packet['source_chain'],'source_group_index':5,
          'prior_active_pointer':packet['prior_active_pointer'],'unavailability':packet['unavailability'],
          'claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
          'checks':{'factual_accuracy':{'status':'pass','details':'兩歷史原必要問題的同一裝置通則現已限定；本人重核第9行與八入口，不採舊still_fixed結論。','claim_ids':[x['id'] for x in claims]},
                    'numeric_verification':{'status':'not_applicable','details':'此兩問題是CLI參數／裝置route的software契約，沒有數值或品質/速度實測主張；returncode2只作parse邊界。','claim_ids':[]},
                    'figure_consistency':{'status':'not_applicable','details':'此全頁及兩issue當前位置無圖，current figure_sha256={}。','claim_ids':[]},
                    'source_verification':{'status':'pass','details':'兩original/四identity與disposition205/206、原fix raw與目前source SHA均核真；prior pointer/unavailability refs原樣，未讀/抄prior appendix still_fixed。','claim_ids':[x['id'] for x in claims]},
                    'limitations':{'status':'pass','details':'只判原fix現在仍在。實CPU與mock CUDA/MPS分開，不重做舊模型/訓練/final/研究；三parse失敗符合已說明限制，沒有補造不可用事件。','claim_ids':[x['id'] for x in claims]}},
          'historical_issue_rechecks':rechecks}
review_setup.ROOT=ROOT
validator=review_setup.module('canonical_training_historical_validator','scripts/check_technical_reviews.py')
errors=validator._validate(ROOT,ROOT/'course/training.md','training',body,appendix,{TASK:['training']},set());assert not errors,errors
for original_row,current_row in zip(packet['historical_identities_only'],rechecks,strict=True):
    for key in ('issue_id','disposition_ref','original_report','json_pointer','historical_fix_verification'):assert current_row[key]==original_row[key]
assert ref(current_path)['sha256']=='f9048f741959652b22fe9d45b1f8283c2d97c184e53a9e0edd5e0f7fe101a041'
assert ref(original_path)['sha256']=='2f936e78a12c68fa409118a0ad7e62c901ecf6461f1aa882a62bf52a5124e043'
with (ROOT/OUT).open('x',encoding='utf-8') as stream:json.dump(appendix,stream,ensure_ascii=False,indent=2);stream.write('\n')
native={'checked_at':datetime.now(UTC).isoformat(),'actor':TASK,'packet':ref(PACKET),'report':ref(OUT),'created_at':appendix['created_at'],
        'verdict':'pass','native_current_errors':errors,'historical_identity_exact_match':True,'historical_rows':2,
        'preserved_original_necessities':['necessary','necessary'],'prior_pointer_and_unavailability_refs_unchanged':True,
        'original_and_current_unchanged':True,'scope':'Only child00/group05 original fix now; standard native schema/source/hash plus explicit exact historical identity/current evidence checks.'}
native_path=HERE/'check-history-child-00-group-05-v1.json'
with native_path.open('x',encoding='utf-8') as stream:json.dump(native,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(native,ensure_ascii=False))
