"""Original-owner informed dependency-only callback; no inference, network, training or score recomputation."""
import copy
from datetime import datetime, UTC
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re

ROOT=Path(__file__).resolve().parents[6]
OUT=Path(__file__).parent
BASE=OUT.relative_to(ROOT).as_posix()+'/'
PRIOR='docs/course-revision-20261016/reviews/technical-readme-backfill-callback-01.json'
TARGET='docs/course-revision-20261016/reviews/technical-readme-backfill-callback-02.json'
TASK='/root/tech_transport_readme_20261016'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_bytes())
prior=load(ROOT/PRIOR)
assert sha(ROOT/PRIOR)=='4e3d9080205588b53e8c8ca326677ffd5c5ea3d5e75514f54700e117c74cc351'
f15=load(ROOT/'docs/course-revision-20261016/freeze-15.json')
f16=load(ROOT/'docs/course-revision-20261016/freeze-16.json')
p15=next(x for x in f15['pages'] if x['page_id']=='19.13')
p16=next(x for x in f16['pages'] if x['page_id']=='19.13')
before=(ROOT/p15['snapshot']).read_text()
chapter_path=ROOT/'course/chapters/19.md'
chapter=chapter_path.read_text()
current=re.search(r'^## 19\.13 .*?(?=^## |\Z)',chapter,re.S|re.M)[0]
assert hashlib.sha256(current.encode()).hexdigest()==p16['source_sha256']
assert (ROOT/p16['snapshot']).read_text()==current
old_whole_sha=next(s['sha256'] for s in prior['sources'] if s['id']=='s_ch19')
reconstructed=chapter.replace(current,before,1)
assert hashlib.sha256(reconstructed.encode()).hexdigest()==old_whole_sha
readme16=next(x for x in f16['pages'] if x['page_id']=='readme')
assert sha(ROOT/'README.md')==prior['source_sha256']==readme16['source_sha256']
changed=[];source_identities=[]
for s in prior['sources']:
 if s['kind']=='repository_code':
  actual=sha(ROOT/s['path']);row={'id':s['id'],'path':s['path'],'previous_sha256':s['sha256'],'current_sha256':actual,'unchanged':actual==s['sha256']};source_identities.append(row)
  if actual!=s['sha256']:changed.append(s['id'])
assert changed==['s_ch19'],changed
artifact_identities=[]
for a in prior['artifacts']:
 actual=sha(ROOT/a['path']);assert actual==a['sha256'];artifact_identities.append({'id':a['id'],'path':a['path'],'sha256':actual,'unchanged':True})
for path,digest in prior['figure_sha256'].items():assert sha(ROOT/path)==digest
result={
 'at':datetime.now(UTC).isoformat(),'reviewer_task':TASK,'current_context':'informed',
 'command':'uv run --frozen --extra cpu python '+BASE+'dependency-applicability-02.py',
 'scope':'本人真正閱讀當前19.13全文及freeze15→16差異後的窄範圍依賴效力確認；不重新盲審README或其他小節。',
 'prior_report':{'path':PRIOR,'sha256':sha(ROOT/PRIOR)},
 'source':{'path':'README.md','sha256':sha(ROOT/'README.md'),'unchanged_from_callback01':True},
 'dependency':{'path':'course/chapters/19.md','previous_whole_sha256':old_whole_sha,'current_whole_sha256':sha(chapter_path),'current_section':'19.13','current_section_sha256':p16['source_sha256'],'personally_read_current_full_text':current,'source_diff_15_to16':''.join(difflib.unified_diff(before.splitlines(True),current.splitlines(True),fromfile=p15['snapshot'],tofile='current19.13')),'whole_file_change_bounded':{'method':'以freeze15的19.13原bytes替回當前整份chapter19，重建的整檔SHA與callback01實際pin的整檔SHA精確相同。','reconstructed_previous_sha256':hashlib.sha256(reconstructed.encode()).hexdigest(),'matches_callback01':True}},
 'source_identities':source_identities,'artifact_identities':artifact_identities,
 'judgments':[
  {'change':'C最小輸出由「建議澄清與其依據」改成「yes/no機率、選中類別與集中度」。','source_basis':'親讀當前19.13表列；本人01已親讀Product.judge實作並實跑CLI。其JSON欄位是probabilities、selected、concentration；Auto才另處理clarification_needed。','effect':'和01親查的實作／輸出更精確對齊，不新增C能力或校準宣稱；README c36、c39、c40、c42效力不變。未把集中度當成答對機率。','affected_claim_ids':['c36','c39','c40','c42'],'necessary_readme_issue':False},
  {'change':'「共同介面會讓人選成品」改為「三個成品各有自己的推論入口，使用者依任務選擇」。','source_basis':'親讀當前19.13全文；01已驗A既有入口、B獨立solve、C獨立judge，以及Auto同時載入B/C。','effect':'與README的A/B/C映射及產品操作轉介一致，不要求新增共同UI；c26、c36、c41、c42、c43均維持成立。','affected_claim_ids':['c26','c36','c41','c42','c43'],'necessary_readme_issue':False},
 ],
 'all44_claims_applicable':True,'issues':[],
 'not_rerun':'沒有HF/API請求、CPU模型推論、HTTP服務、raw score重算、訓練、formaltest重啟、付費或遠端寫入。只用標準庫讀來源、計SHA、重建依賴bytes並執行本地validator。其他來源與primary/artifact同bytes後沿用本人01與更早的原始查證。',
 'environment':{'python':platform.python_version(),'device':'CPU; standard-library source/hash checks only','model_execution':'none'},
}
result_path=OUT/'dependency-applicability-results-02.json'
assert not result_path.exists(),'Preserve every existing02 artifact'
result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
r=copy.deepcopy(prior)
r.update(recorded_at=datetime.now(UTC).isoformat(),review_round='original-owner-informed-narrow-dependency-callback-02',current_context='informed',source_freeze_reference='docs/course-revision-20261016/freeze-16.json')
r['previous_callback']={'path':PRIOR,'sha256':sha(ROOT/PRIOR),'preserved_bytes':True,'scope_at_that_time':'完整README知情複查，44項主張；本輪只確認19.13必要依賴變動的效力。'}
for s in r['sources']:
 if s['id']=='s_ch19':
  s['callback01_sha256']=s['sha256'];s['sha256']=sha(chapter_path)
  s['version']='2026-10-16當前worktree bytes，freeze16窄依賴複查'
  s['callback01_inspection_note']=s['inspection_note']
  s['inspection_note']='本人真正讀19.13當前全文及freeze15→16兩處差異，再把舊section替回重建整檔，SHA精確吻合01所pin的整檔。C輸出改成probabilities/selected/concentration與原已驗JSON一致；各自推論入口與A/B/C及Auto操作轉介一致。只重驗必要依賴，不讀其他owner報告，不重跑HF/CPU/分數。'
for a in r['artifacts']:
 a['description']='01或更早輪次本人已取得的原始證據；本次只核bytes一致、未重新執行。原用途：'+a['description']
r['artifacts'].extend([
 {'id':'a_dependency02','kind':'execution','path':result_path.relative_to(ROOT).as_posix(),'sha256':sha(result_path),'description':'本次19.13窄依賴實讀、差異、完整chapter19重建與全部其餘source/artifact身分核對。','command':result['command'],'result':'只改19.13的兩處介紹；以舊slice重建整檔SHA精確匹配01。README/圖/其餘source與primary artifact bytes不變；44claims/轉介維持效力，沒有新必要問題。','environment':result['environment']},
 {'id':'code_dependency02','kind':'code','path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(Path(__file__)),'description':'本次實際來源／SHA及validator檢查程式；不執行模型或網路。'}
])
affected={'c26','c36','c39','c40','c41','c42','c43'}
for c in r['claims']:
 if 'callback_revalidation' in c:c['callback01_revalidation']=c.pop('callback_revalidation')
 c['callback_revalidation']={'round':'02','current_context':'informed','dependency_relevance':'直接涉及19.13角色／輸出／轉介者已對本次兩處變動重新判效力。' if c['id'] in affected else '本項主張正文、必要實作與primary證據未變，且本次只變19.13兩處介紹，不改其支持範圍。','status':'verified','execution_provenance':'沿用本人01及更早原始查證；本輪沒有重跑HF、模型或分數。'}
 c['artifact_ids'].append('a_dependency02')
r['callback01_dependency_audit']=r.pop('callback_dependency_audit')
r['callback_dependency_audit']={'round':'02','artifact_id':'a_dependency02','scope':result['scope'],'judgments':result['judgments'],'changed_repository_source_ids':changed,'all_other_identities_unchanged':True,'chapter19_previous_sha256':old_whole_sha,'chapter19_current_sha256':sha(chapter_path),'readme_sha256':sha(ROOT/'README.md'),'source_freeze_reference':'docs/course-revision-20261016/freeze-16.json','no_new_runs':result['not_rerun']}
r['callback01_scope_and_independence']=r.pop('scope_and_independence')
r['scope_and_independence']={'scope':'原owner的窄依賴適用性callback02，不是新的整頁盲審。01整頁44項結論及原始證據保持不變；本次真正讀19.13全文和15→16差異後確認效力。','role':'reviewer_context=fresh仍只描述最初task；current_context=informed。','prior_report_preserved':r['previous_callback'],'source_materials':'只讀本人的來源、freeze15/16原文與本人01報告；未讀別人的technical report、作者fix plan或審閱結論。','new_work':result['not_rerun'],'not_human_testing':True}
r['checks']['factual_accuracy']['details']='01已完成整頁44claims實質查證；本輪只親讀19.13當前全文及兩處差異。C輸出與各自入口的介紹都和本人01實驗／程式證據一致，44claims與README轉介維持效力，無新必要修訂。'
r['checks']['numeric_verification']['details']='本輪沒有重算分數。原數字、raw primary資料、結果、程式與artifact均逐檔SHA相同；19.13本次兩個介紹句不改數字或評測條件，因此沿用01及更早本人的驗證。'
r['checks']['figure_consistency']['details']='README與其唯一SVG的SHA均與01相同；本次19.13無圖、也無數值或方向改動。沿用本人原先親看的render證據，不冒稱重新渲染或看頁面。'
r['checks']['source_verification']['details']='本人實讀19.13全文與freeze15→16差異後，將舊slice替回當前chapter19重建舊整檔SHA，精確吻合01；因此真正核清whole19改動僅在已讀兩處，沒有只換hash。其他sources/primary artifacts全部核bytes一致，沿用本人原始查證；沒有重新HF/API請求。'
r['checks']['limitations']['details']='current_context=informed；本輪只是窄依賴適用性確認，不是新盲審、HF重下載、CPU模型執行或raw分數重算。01的Actions/client failure與remote completed分層結論及有限能力界線都原樣保留；未訓練、重啟formal test、付費或遠端寫入。'
r['issues']=[]
target=ROOT/TARGET;assert not target.exists();target.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('readme_cb02_validator',ROOT/'scripts/check_technical_reviews.py');validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
errors=validator._validate(ROOT,ROOT/'README.md','readme',(ROOT/'README.md').read_text(),r,{TASK:['readme']},set())
validation={'at':datetime.now(UTC).isoformat(),'command':result['command'],'report':{'path':TARGET,'sha256':sha(target)},'current_context':'informed','errors':errors,'source_sha256':sha(ROOT/'README.md'),'scope':'Direct _validate on the current whole README and complete44-claim callback02; old01 preserved; narrow dependency applicability only.'}
validation_path=OUT/'direct-validation-02.json';validation_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
assert not errors,errors
seal={'schema_version':1,'sealed_at':datetime.now(UTC).isoformat(),'reviewer_task':TASK,'role':'technical','lesson_id':'readme','current_context':'informed','round_type':'same-owner narrow dependency applicability callback02; no new blind review','report':{'path':TARGET,'sha256':sha(target)},'source':{'path':'README.md','sha256':sha(ROOT/'README.md')},'chapter19':{'path':'course/chapters/19.md','sha256':sha(chapter_path),'actually_read_changed_section':'19.13','reconstruction_matches_previous_whole_file':True},'verdict':r['verdict'],'claim_count':len(r['claims']),'previous_callback':r['previous_callback'],'validation':{'path':validation_path.relative_to(ROOT).as_posix(),'sha256':sha(validation_path),'errors':errors},'new_artifacts':{a['path']:a['sha256'] for a in r['artifacts'] if a['id'] in ('a_dependency02','code_dependency02')},'no_new_inference_network_or_scores':True,'independence':'No other reviewer conclusions, technical reports or author fix plans read.'}
seal_path=OUT/'owner-callback-seal-02.json';seal_path.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':TARGET,'report_sha256':sha(target),'seal_path':seal_path.relative_to(ROOT).as_posix(),'seal_sha256':sha(seal_path),'validation_sha256':sha(validation_path),'verdict':r['verdict'],'claims':len(r['claims']),'errors':errors,'prior01_preserved_sha256':sha(ROOT/PRIOR)},ensure_ascii=False))
