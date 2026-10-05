"""Update this same reviewer's own report after actual revised executions."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[6]
RUN=Path(__file__).resolve().parent.parent
ART=RUN.parent
REPORT=ROOT/'docs/technical-reviews/7.7.json'
INITIAL_SHA='d2962934adc5fdadc16d47966e29d6475b2f82ef03c7db4d52b213111c65c07a'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def read(p): return json.loads(p.read_text())
assert sha(REPORT)==INITIAL_SHA
initial=ROOT/f'docs/technical-reviews/history/phase4-7_7-own-initial-revise-{INITIAL_SHA}.json'
assert sha(initial)==INITIAL_SHA
report=read(REPORT)
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_7_7' and report['verdict']=='revise'
probe=read(RUN/'probe-stdout.json'); extraction=read(RUN/'original-run/extraction.json'); reuse=read(RUN/'reuse-verification.json')
assert read(RUN/'original-run/execution.json')['exit_code']==0
assert reuse['revised_source_sha256']==extraction['source_sha256']
assert all(x['same_bytes'] for x in reuse['checked_files'])
env=probe['environment']
old_claim=copy.deepcopy(next(c for c in report['claims'] if c['id']=='c-tolerance'))
old_issues=copy.deepcopy(report['issues'])
report['source_sha256']=extraction['source_sha256']; report['figure_sha256']=extraction['figure_sha256']; report['verdict']='pass'
report['scope']['actual_read'].extend(['Recheck 1: 完整新版 course/chapters/07.md#7.7 親讀與重新擷取原始UTF-8 bytes',
    'Recheck 1: 自身allclose官方tagged原始契約766–795及installed公共docstring69–98重新親讀',
    'Recheck 1: 實際attention_mask/manual_attention與TinyLM.forward重新親讀；其餘未變來源/圖依自有初審證據及指紋合法重用'])
report['scope']['recheck_reuse']=reuse
def artifact(identifier,path,kind,description,command=None,result=None):
    v={'id':identifier,'path':rel(path),'sha256':sha(path),'kind':kind,'description':description}
    if kind=='execution': v.update(command=command,result=result,environment=env)
    report['artifacts'].append(v)
artifact('e-revised-original',RUN/'original-run/execution.json','execution','NEW independently executed revised raw original 7.7 fence; source/fence/stdout/environment preserved in recheck-1/original-run.',
    '.venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.7 --output /tmp/phase4-7_7-recheck-1 --execute --timeout 45',
    'Exit0; revised original fence includes rtol=0.0; stdout 有效位置最大差 0.0; CPU torch2.14.1+cpu, CUDA None, no guard events.')
artifact('e-recheck-probe',RUN/'probe-stdout.json','execution','NEW same-model exact exercise-tail executions plus pure absolute-tolerance boundary checks; preserves original mixed-tolerance counterexample.',
    '.venv/bin/python docs/technical-reviews/artifacts/phase4-7_7-independent/recheck-1/code/recheck_probe.py',
    'Exit0. Exact revised fence0.0; remove valid0.22275620698928833; remove positions1.0177843570709229, both revised assert failures. Pure tolerance accepts1e-6 and rejects1.000001e-6/2e-6. No parameter updates.')
artifact('e-recheck-commands',RUN/'commands.json','execution','Actual bounded rerun commands and return codes, not inferred from source hash.',
    '.venv/bin/python docs/technical-reviews/artifacts/phase4-7_7-independent/recheck-1/code/run_recheck.py',
    'Exit0; revised section bytes verified exactly equal to old except assert adding rtol=0.0; all reused repository implementations and figure hashes match.')
already={a['path'] for a in report['artifacts']}
for p in sorted(RUN.rglob('*')):
    if not p.is_file() or rel(p) in already or p.name.startswith('checker-'): continue
    name=p.relative_to(RUN).as_posix()
    artifact('a-recheck-'+name.replace('/','-').replace('.','-'),p,'code' if p.suffix=='.py' else 'source_snapshot',
        'Own new recheck raw code / stdout / environment / source / provenance bytes: '+name)
report['sources'].extend([
    {'id':'s-revised-original','kind':'execution','title':'Own NEW revised original fence CPU execution','verified':True,'artifact_id':'e-revised-original'},
    {'id':'s-recheck-probe','kind':'execution','title':'Own NEW same-model exercises and pure-atol boundary execution','verified':True,'artifact_id':'e-recheck-probe'}])
def ev(source,locator,supports): return {'source_id':source,'locator':locator,'supports':supports}
for c in report['claims']:
    if c['id'] in ('c-invariance','c-original-api','c-number'):
        c['evidence'].append(ev('s-revised-original','recheck-1/original-run/fence-1.py, execution/stdout/environment','新版rtol=0.0原碼新實跑仍最大差0.0，核心前向與輸出不變。'))
        c['artifact_ids'].append('e-revised-original')
    if c['id']=='c-original-api':
        c['scope']+=' 新版assert明設rtol=0.0；官方混合公式因此退化為逐元素abs<=1e-6。'
        c['verification']['observed']+=' 新版原fence重新實跑Exit0，stdout0.0。'
        c['verification']['details']+=' 新版exercise probe仍同模型，無backward/step，參數hash相同。'
    if c['id']=='c-number':
        c['verification']['observed']+=' 新版原fence新實跑最大差0.0。'
        c['verification']['details']+=' 不同長度batch數字來自未變原自有probe，依reuse-verification合法重用；未重跑該不受assert修改影響的機制probe。'
    if c['id']=='c-tolerance':
        c['original_review_claim']=old_claim
        c['statement']='「檢查容許百萬分之一的絕對差」現對應 assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)，純絕對差約定與API一致。'
        c['status']='verified'
        c['scope']='協調者只在原assert加入rtol=0.0；rtol×|other|項為0，所有有限元素需差<=1e-6。原錯誤claim/反例/initial revise/issue保留，不宣稱舊原碼曾符合此約定。'
        c['evidence']=[ev('s-allclose','官方_torch_docs.py766–795; installed公共docstring69–98重新親讀','|input-other|<=atol+rtol|other|；rtol0給純atol界限。'),
            ev('s-revised-original','新版raw fence assert與stdout/execution','真正執行新版原碼，純abs1e-6門檻，原三格最大差0.0。'),
            ev('s-recheck-probe','pure_atol_boundary; preserved_counterexample_rechecked','float64零baseline差0/.5e-6/1e-6通過，1.000001e-6/2e-6失敗；舊1/1.000002反例現在正確拒絕。')]
        c['artifact_ids']=['e-probe','e-revised-original','e-recheck-probe']
        c['verification']={'method':'executed','expected':'新版純atol接受abs<=1e-6，拒絕超過界限；真實三格應仍相等。',
            'observed':'新原fenceExit0，差0.0；邊界1e-6 True，1.000001e-6 False；原約2e-6反例由舊True變新版False。',
            'details':'重新親讀全節及API來源契約，NEW原fence、同模型變體、float64純絕對界限probe均實際執行，非僅換source SHA；原矛盾紀錄保留。',
            'tolerance':'rtol=0.0, atol=1e-6；float64零baseline精確比較閾值，避免1+1e-6的表示舍入造成邊界歧義。'}
    if c['id']=='c-exercise':
        c['evidence'].append(ev('s-recheck-probe','exercise_variants; code/exercise-remove-valid.py and exercise-remove-positions.py','新版純atolassert兩變體在同一原初始化模型逐字tail實跑，按預期raise AssertionError。'))
        c['artifact_ids'].append('e-recheck-probe')
        c['verification']['observed']='NEW新版同模型：只刪valid差0.22275620698928833；只刪positions差1.0177843570709229；兩者純atolassert按預期失敗。'
        c['verification']['details']+=' 改動的是新原fence呼叫tail，沒有重新初始化模型；舊rightPAD邊界proof依未變實作hash合法重用。'
    if c['id']=='c-figure':
        c['scope']+=' 複查確認SVG SHA完全相同；合法重用自身已view的實際render，沒有把未重新渲染寫成新渲染。'
for issue in report['issues']:
    issue['original_status']='unresolved'
    issue['status']='resolved'
    issue['resolution']='協調者將原assert加rtol=0.0，正文與圖未變；原審閱者親讀新版全節和原API契約，再擷取並親跑NEW原fence0.0、兩練習與pureatol界限。新版接受abs<=1e-6、拒絕超過值，且原約2e-6反例新版False。原問題/反例/initial revise retained。'
    issue['recheck_artifact_ids']=['e-revised-original','e-recheck-probe','e-recheck-commands']
for name,value in report['checks'].items(): value['status']='pass'
report['checks']['factual_accuracy']['details']='核心PAD/attention/positions/loss維持原核查範圍；原allclose矛盾已由最小rtol0修改及原reviewer NEW執行解決，原claim/issue retained。'
report['checks']['numeric_verification']['details']='NEW原fence0.0；NEW兩練習差0.2227562/1.0177844；pure-atol邊界1e-6通過、更大值拒絕。其餘未變key分母/batch/空行proof經精確hash核對合法重用。'
report['checks']['figure_consistency']['details']+=' Recheck1：SVG SHA仍1965d232bc7a4851c236de6dbf171500c4a999df63621fdcd4484dc0be03190a，合法重用自身初審render/view；沒有聲稱新browser/新render。'
report['revision_history']=[{'reviewer_task':report['reviewer_task'],'stage':'own genuine recheck1','previous_verdict':'revise','new_verdict':'pass',
    'previous_report_sha256':INITIAL_SHA,'previous_report_path':rel(initial),'old_source_sha256':reuse['old_source_sha256'],'new_source_sha256':extraction['source_sha256'],
    'original_issue_records':old_issues,'original_claim':old_claim,'actual_actions':'Full revised section read, exact original UTF-8 re-extracted, revised original fence executed in fresh CPU process, same-model exact exercise tails and pure-atol boundaries executed, own official API contract reread, unchanged implementation/figure hashes verified.',
    'new_artifact_ids':['e-revised-original','e-recheck-probe','e-recheck-commands'],'reuse_scope':reuse['legal_reuse_scope']}]
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],'report_sha256':sha(REPORT),'claims':len(report['claims']),'all_original_claims_retained':True},ensure_ascii=False,indent=2))
