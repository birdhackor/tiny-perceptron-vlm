"""Save a separate complete informed-owner callback without changing original evidence."""
import copy
import hashlib
import json
from pathlib import Path

BASE=Path('docs/technical-reviews/artifacts/revision-20261016/backfill')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
original_path=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-original-01.json')
original=read(original_path)
report=copy.deepcopy(original)
run=read(BASE/'19.13-callback-execution-01.json')
assert sha(original_path)==run['original_report_sha256']
report['source_sha256']=run['freeze']['source_sha256']
report['verdict']='pass'
report['reviewer_context']='fresh'
report['current_context']='informed'
report['callback_context']={
 'initial_reviewer_context':'fresh',
 'current_reviewer_context':'informed',
 'original_report':str(original_path),
 'original_report_sha256':sha(original_path),
 'freeze_path':run['freeze']['path'],
 'snapshot':run['freeze']['snapshot'],
 'informed_by':'Coordinator supplied the two fixes and clarified that authorized product delivery uses independent CLI/API entries, without a required shared A/B/C GUI. Reviewer did not read the author plan, dispositions, or other reviewers conclusions.',
 'reread_scope':'Entire new 19.13 reread, exact two-replacement diff verified, affected Product.judge return, independent CLI selection branches, HTTP dispatch, A chat CLI and A InferenceAssistant loading/reply inspected.',
 'newly_executed':'A CLI --help; independent B solve --b; independent C judge --c. No new A generation, HTTP run, GPU work or formal-test predictions.',
 'retained_checks':'c1–c9 and c12 retain original substantive verification after all 16 original repository sources, old raw predictions and frozen product files were SHA-compared unchanged. Original random-initial tensor reconstruction, selected/best identity checks, full core count/correct recount and download receipts were not rerun.',
 'original_evidence_unchanged':True,
}
for source in report['sources']:
 if source['id']=='s_frozen':
  source.update(path=run['freeze']['snapshot'],sha256=run['freeze']['snapshot_sha256'],title='freeze-16 指向的完整 19.13 原始小節快照',version='freeze-16; independently reread whole 19.13 on 2026-10-16',inspection_note='完整重讀新19.13；current lesson_slices與freeze16 snapshot逐字一致。與freeze15原快照只有C表格輸出及獨立推論入口兩處修改。未把其他章節當已重新全面查證。')

def repo_source(id,path,title,note):
 return {'id':id,'kind':'repository_code','title':title,'path':path,'sha256':sha(path),
         'version':'Independently inspected exact working-tree bytes on 2026-10-16 for freeze16 owner callback',
         'verified':True,'inspection_note':note}
report['sources'].extend([
 repo_source('s_a_cli','scripts/selftrained/chat.py','A 獨立公開推論 CLI','完整讀 parser/main。--model-dir/messages/asset-dir 與有限 task 輸入；直接建 InferenceAssistant 並呼叫 reply。無 B/C 依賴，無 effort 選項。親跑 --help exit0。'),
 repo_source('s_a_inference','tiny_perceptron/selftrained/inference.py','A 自身 safe export 載入與推論 API','實讀 module TASKS、verify_export/fetch_public_export 以及 InferenceAssistant.__init__/reply（lines1–137、215–329）：完整自訓權重與tokenizer/模型配置的獨立驗證/載入，reply直接執行模型生成；不是對B/C包裝。其餘history helper未在本次全面重審。'),
 repo_source('s_a_cli_raw','docs/selftrained/results/public-cpu-raw/ocr-1-result.json','A 既有獨立 CLI 原始操作收據','讀完整原始JSON；chat=true、returncode=0、argv為scripts/selftrained/chat.py、actual_answer=大小、safe model SHA與固定公開revision。只用來支持A既有獨立入口的實際操作，不當本callback新生成。'),
 {'id':'s_callback_execution','kind':'execution','title':'freeze16 19.13 原owner知情複查與獨立CLI實跑','verified':True,'artifact_id':'a_callback_execution'},
])
def ev(id,locator,supports):return {'source_id':id,'locator':locator,'supports':supports}
claims={c['id']:c for c in report['claims']}
c10=claims['c10']
c10.update(status='verified',statement='C 的資訊充分性輸出包含 yes/no 機率、選中類別與集中度。',
 evidence=[ev('s_inference','Product.judge完整return字典 lines140–149','三項欄位分別為probabilities、selected與concentration；concentration_scope明示非答對機率。'),ev('s_cli','judge子命令及獨立--c參數','用C自己的權重提供此輸出。'),ev('s_callback_execution','newly_executed_commands[2]與c_table_checks','親跑獨立C CLI，輸出欄位與表格一致。')],
 artifact_ids=['a_callback_execution'],
 scope='固定語法卡片材料與info問題；集中度只描述候選間分布，不宣稱已校準的正確率。表格已刪除澄清建議/缺欄依據輸出；本節亦不要求新增工程以生成它們。',
 verification={'method':'executed','expected':'獨立C judge對卡片缺申請編號回yes/no機率、selected類別及concentration。','observed':'CLI exit0；selected=no；P(yes)=9.260809206282686e-16、P(no)=1.0，集中度=1.0，另有concentration_scope。','details':'新親跑 python -m tiny_perceptron.lesson_products judge --c outputs/lesson-products/products/c --state 卡片進度;申請編號=? --question info；檢查候選key、總和容忍1e-6、選中類別與集中度範圍。'}
)
c11=claims['c11']
c11.update(status='verified',statement='三個成品各有自己的推論入口，使用者可依任務選擇。',
 evidence=[ev('s_a_cli','parser/main','A走scripts/selftrained/chat.py，直接呼叫自身InferenceAssistant.reply。'),ev('s_a_inference','InferenceAssistant.__init__/reply','A直接驗證並載自身safe export，使用自身模型生成。'),ev('s_a_cli_raw','argv/returncode/actual_answer/model/public_source','既有原始A CLI操作成功收據。'),ev('s_cli','solve/judge分支 lines55–60及--b/--c選項','B solve只載B；C judge只載C。'),ev('s_delivery','serve POST /solve /judge','B/C各有對應HTTP endpoint。'),ev('s_callback_execution','newly_executed_commands與independent_entry_checks','親跑A help及分別B solve/C judge核對實際入口。')],
 artifact_ids=['a_callback_execution'],
 scope='此句現在只描述已交付的獨立CLI/API入口；不再預告共同A/B/C選擇器。A的實際生成沿既有primary raw receipt，本callback只親跑其help；B/C各親跑一例，未重驗所有平台或外部下載。',
 verification={'method':'executed','expected':'能定位A、B、C各自的推論入口，B/C命令不需要另一成品權重；A不因新增B取得effort。','observed':'A scripts/selftrained/chat.py --help exit0，具messages/model-dir/task選項且無effort；獨立B solve exit0，answer=1/loops=3；獨立C judge exit0。原始A OCR CLI收據returncode=0、actual_answer=大小。','details':'新親跑三個各自CLI：A只help，B只傳--b，C只傳--c。讀A實際main與InferenceAssistant/API、B/C CLI/HTTP來源；A完整生成未重跑。'}
)
for issue in report['issues']:
 issue['original_status']=issue['status']
 issue['status']='resolved'
 if issue['id']=='19.13-I1':
  issue['resolution']='freeze16 C表格改為「yes／no機率、選中類別與集中度」，完整刪除未交付澄清建議/依據的宣稱。重新讀Product.judge完整return及獨立C CLI親跑得到probabilities/selected/concentration，已與文字一致；無新增模型/介面。'
 else:
  issue['resolution']='freeze16 改為「三個成品各有自己的推論入口，使用者依任務選擇」。已定位並讀A的chat CLI/InferenceAssistant與既有原始操作收據，親跑A help、B獨立solve、C獨立judge；現有獨立CLI/API與修文一致。不再以未交付共同A/B/C選擇器作為能力預告，也不要求新增共同GUI。'
report['artifacts'].extend([
 {'id':'a_callback_execution','kind':'execution','path':str(BASE/'19.13-callback-execution-01.json'),
  'sha256':sha(BASE/'19.13-callback-execution-01.json'),
  'description':'freeze16全文/兩處diff、原證據同byte查核、固定產品SHA、獨立A help及B/C CLI實際輸出，明列未重跑範圍。',
  'command':run['command'],'result':'完成且斷言通過。新C欄位與修文一致；三個獨立入口具直接來源/操作證據；原16項來源、權重與既有逐題分數同byte。','environment':run['environment']},
 {'id':'a_callback_probe','kind':'code','path':str(BASE/'19.13-callback-probe-01.py'),
  'sha256':sha(BASE/'19.13-callback-probe-01.py'),'description':'只讀owner callback程式，保存diff/來源身分與3個有限CLI操作；沒有train、調參或正式評估入口。'},
])
report['checks']={
 'factual_accuracy':{'status':'pass','details':'完整重讀freeze16，兩處原問題均由實際輸出與入口定位解決。A既有有限模型、B/C獨立隨機訓練及Auto組合沿原實質查證，同byte來源/權重/分數已比對。','claim_ids':['c1','c2','c3','c5','c6','c7','c8','c10','c11','c12']},
 'numeric_verification':{'status':'pass','details':'未改任何數值主張、模型、資料或測試。原覆寫例與正式核心count/correct重算證據保留，原始逐題檔SHA未變；不稱本callback重算。新增獨立B CLI再回1/3輪；C選中no並回既有機率/集中度。','claim_ids':['c1','c4','c7','c9','c10']},
 'figure_consistency':{'status':'not_applicable','details':'完整freeze16 19.13仍無任何圖像/SVG引用，figure map為空。本callback沒有網站版面驗收。','claim_ids':[]},
 'source_verification':{'status':'pass','details':'原官方HTML與原始程式/runner/metrics的實際閱讀證據保留，16項既有repository來源SHA均未變。另完整重讀相關B/C code及A chat CLI、讀A獨立inference載入/生成API與原始OCR CLI receipt；重新執行相稱入口操作，兩項已可核實。','claim_ids':['c5','c10','c11']},
 'limitations':{'status':'pass','details':'C修文只保留已交付typed probability/選中類別/集中度，未冒稱澄清解釋；產品選用只承諾獨立入口。保留有限語法/類別、A歷史失敗、B算術失敗、effort非品質保證及Auto只組合B/C的原邊界。','claim_ids':['c1','c2','c3','c4','c6','c7','c8','c9','c10','c11','c12']},
}
report['review_scope_note']='原owner對freeze16 19.13進行知情callback：完整重讀本節並比對diff；只重新查核受影響C輸出及獨立入口，加相稱CPU CLI操作。其餘原報告實質查證在byte相同前提下沿用，明記未重跑。初始任務context=fresh，當前context=informed；不將callback冒充第二個fresh reviewer。原revise報告與全部原artifact均未修改；沒有讀其他/舊review結論或作者plan，沒有訓練/正式test重啟/外部寫入。'
out=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-callback-01.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(out)
