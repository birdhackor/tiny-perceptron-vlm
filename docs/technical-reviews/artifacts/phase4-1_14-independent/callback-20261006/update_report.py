"""Same original owner applies the completed narrow callback to their report."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
REL = OUT.relative_to(ROOT).as_posix()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

archive = json.loads((OUT / 'archive.receipt.json').read_bytes())
facts = json.loads((OUT / 'callback-facts.json').read_bytes())
prior = ROOT / archive['prior_opaque_file']
assert sha(prior) == archive['prior_sha256']
report = json.loads(prior.read_bytes())
assert report['reviewer_task'] == facts['reviewer_task'] == '/root/phase4_factual_coordinator/factual_1_14'
assert facts['verdict'] == 'pass' and not facts['unresolved_substantive_issues']
assert (OUT / 'narrow-checks.stderr.txt').read_bytes() == b''
stdout = json.loads((OUT / 'narrow-checks.stdout.txt').read_bytes())
assert stdout['verdict'] == 'pass' and stdout['section_sha256'] == report['source_sha256']
assert all(v['unchanged'] for v in facts['prior_artifacts_actual_sha256_check'])

def add(identifier, filename, kind, description, **extra):
    report['artifacts'].append({'id':identifier,'path':REL+'/'+filename,'sha256':sha(OUT/filename),'kind':kind,'description':description,**extra})

add('callback_20261006_archive_code','archive_prior.py','code','原owner先opaque保留自己舊raw報告的實際程式；沒有解析或列印其內容。')
add('callback_20261006_archive_receipt','archive.receipt.json','source_snapshot','此次opaque prior FILE/SHA、实际命令與環境。')
add('callback_20261006_archive_stdout','archive.stdout.txt','source_snapshot','實際opaque保存命令stdout；只列metadata。')
add('callback_20261006_archive_stderr','archive.stderr.txt','source_snapshot','實際opaque保存命令stderr為空。')
add('callback_20261006_prior',Path(archive['prior_opaque_file']).name,'source_snapshot','原owner自己的前版報告完整raw不覆寫；保留前次scope與問題歷史。')
add('callback_20261006_code','narrow_checks.py','code','本次窄範圍讀取/AST/指紋核驗程式；無torch import、模型推論、訓練、下載或網路。')
command = '.venv/bin/python docs/technical-reviews/artifacts/phase4-1_14-independent/callback-20261006/narrow_checks.py > docs/technical-reviews/artifacts/phase4-1_14-independent/callback-20261006/narrow-checks.stdout.txt 2> docs/technical-reviews/artifacts/phase4-1_14-independent/callback-20261006/narrow-checks.stderr.txt'
add('callback_20261006_facts','callback-facts.json','execution','canonical callback artifact：本人重讀当前1.14/T.3、必要CLI/helper契約、真支持範圍；43舊artifacts/8本地sources逐檔hash核真並明示沿用原CPU證據。',command=command,result='exit 0；原1.14 bytes未變；T.3修訂fixed configurations與AST tuple一致；43 artifacts及8 local sources hash unchanged；無未核實實質claim，pass；本次沒有模型執行。',environment=facts['environment'])
add('callback_20261006_stdout','narrow-checks.stdout.txt','source_snapshot','本次成功窄核對命令實際stdout；與canonical callback-facts對上。')
add('callback_20261006_stderr','narrow-checks.stderr.txt','source_snapshot','本次成功窄核對命令stderr為空。')
add('callback_20261006_section','section.current.raw.md','source_snapshot','此次實讀1.14全節raw UTF-8；和原CPU輸入bytes完全相同。')
add('callback_20261006_T3','T.3.current.raw.md','source_snapshot','此次實讀最新T.3全節raw UTF-8；full read不等於T.3整節技術判定。')
add('callback_20261006_T3_delta','T.3.source-delta.txt','derivation','由本人過去讀取的原稿raw與本次T.3原稿raw比較；沒有讀作者repair或他人報告。')
add('callback_20261006_exploratory_code','exploratory-ast-attempt.py','code','保留真實首次AST探索過寬選到第二loop而失敗的原始heredoc；不是教材程式失敗。')
add('callback_20261006_exploratory_receipt','exploratory-ast-attempt.receipt.json','source_snapshot','保留真正exit1/ValueError與定位；後續narrow_checks選定實際config tuple並成功，未捏造成首命令全過。')
for i,(path,item) in enumerate(facts['new_source_snapshots'].items()):
    file = Path(item['snapshot']).relative_to(OUT.relative_to(ROOT)).as_posix()
    add('callback_20261006_source_'+str(i),file,'code' if path.endswith('.py') else 'source_snapshot','本次讀取的精確原始來源snapshot；'+path)
add('callback_20261006_update_code','update_report.py','code','原owner根據成功查核與本人判断更新報告的程式；保留prior/current實讀範圍與舊證據再用限制。')

report['sources'].extend([
    {'id':'callback_20261006_current_T3','kind':'repository_code','path':REL+'/T.3.current.raw.md','sha256':sha(OUT/'T.3.current.raw.md'),'title':'本次親讀的目前T.3配方與原結果定位入口','version':'2026-10-06 current original raw SHA '+facts['current_T3']['sha256'],'verified':True,'inspection_note':facts['current_T3']['actual_read_scope']+' 真正依賴範圍：'+facts['current_T3']['necessary_scope']+' 版本及支持範圍：'+facts['current_T3']['dependency_reason']},
    {'id':'callback_20261006_execution','kind':'execution','title':'原owner最新T.3引用/契約/舊證據hash窄回呼','verified':True,'artifact_id':'callback_20261006_facts'},
])

for claim in report['claims']:
    claim['artifact_ids'].append('callback_20261006_facts')
    if claim['id']=='C5':
        claim['evidence'].append({'source_id':'callback_20261006_current_T3','locator':'T.3開頭及simple_models固定比較/原始結果連結末段；當前原稿L'+str(facts['current_T3']['first_line'])+'–'+str(facts['current_T3']['last_line']),'supports':'只核本節L553引用仍指向中文simple字元模型的配方與原始結果入口；現在說兩類模型/四個設定，與run_simple_models tuple相符。T.3文字沒有替代原始實驗/原CPU證據，也不當作T.3整節PASS。'})
        claim['evidence'].append({'source_id':'callback_20261006_execution','locator':'callback-facts.json current_T3、actual_contract_inspection、contracts_unchanged、prior_artifacts_actual_sha256_check','supports':'原owner此次實讀必要新版引用/CLI契約、靜態AST確認bigram與1/3/5窗口MLP設定；相關原碼/原結果與本人原CPU artifacts全部hash核真。只沿用2026-10-05本人執行結果，不聲稱新推論或重訓。'})
        claim['verification']['details'] += ' 2026-10-06 narrow callback：本節原文/字元helper/原CPU證據指紋未變，故沿用原執行；新版T.3只作配方與原结果定位/契約銜接查核，不全節重新驗收。'

report['initial_reviewed_on']=report['reviewed_on']
report['reviewed_on']='2026-10-06'
report['verdict']='pass'
report['read_scope']['linked_recipe']='2026-10-06 親讀目前T.3全文 L'+str(facts['current_T3']['first_line'])+'–'+str(facts['current_T3']['last_line'])+'，raw SHA '+facts['current_T3']['sha256']+'；真正必要範圍為中文字元simple配方/原結果定位及固定比較身份。Transformer/byte補充只讀以防跨編碼混淆，不作其能力/CLI實測或T.3整節PASS。先前linked_recipe宣告原樣留於opaque prior與callback-facts.current_T3.prior_declared_read_scope。'
report['read_scope']['latest_callback_original']='2026-10-06 原owner重讀1.14全節：'+facts['current_section']['actual_inspection']
report['read_scope']['latest_callback_protocol']='本次完整親讀最新docs/review-tools/factual-reviewer-instructions.md及精確.agents/skills/clear-tutorial/references/review-protocol.md；原樣snapshot/hash見callback sources。技術callback並非新第一輪reader身份。'
report['read_scope']['latest_callback_contracts']=facts['actual_contract_inspection']
report['same_owner_callback']={'date':'2026-10-06','reviewer_task':report['reviewer_task'],'prior_opaque_file':archive['prior_opaque_file'],'prior_opaque_sha256':archive['prior_sha256'],'canonical_artifact_id':'callback_20261006_facts','facts':REL+'/callback-facts.json','source_sha256_unchanged':True,'T3_source_sha256':facts['current_T3']['sha256'],'true_scope_reason':facts['current_T3']['dependency_reason'],'reused_evidence_scope':facts['reused_evidence_scope'],'current_visual_check':'未重新驗證preview桌面/手機；無可呼叫browser工具，本次只作未變section之必要引用/契約版本查核，沒有以rootparity當本人證據。','notes':'初稿跨節EOS0過寬指向的真實複查歷史仍保留於scope_reinspection/其note及opaque prior，沒有換身份或重新寫成未曾有問題。'}
report['checks']['source_verification']['details'] += ' 2026-10-06 same owner callback已親讀最新T.3必要引用與契約，當前T.3 raw SHA明示；官方原来源与本人原证据逐檔hash核真沿用，未重抓來源或讀他人report/repair。'
report['checks']['limitations']['details'] += ' 2026-10-06此次只重新讀必要原稿/契約及hash/AST核對，不新增模型推論；T.3只是1.14重做與原證據定位入口，原有full-read宣告保留于prior，沒有以当前全節閱讀聲稱T.3整節PASS。未重新驗證preview桌面/手機呈現。'
report['checks']['figure_consistency']['details'] += ' 此次no-figure問題亦親判：文字/4字列表/數字/短碼已展示row/current流，兩個等號末字在prompt可見，不需想像未提供圖片或空間素材；沒有新圖引用。'

target=ROOT/'docs/technical-reviews/1.14.json'
assert sha(target)==archive['prior_sha256'],'Another writer changed the report during callback'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
result={'verdict':report['verdict'],'reviewer_task':report['reviewer_task'],'new_report_sha256':sha(target),'prior_opaque_file':archive['prior_opaque_file'],'prior_opaque_sha256':archive['prior_sha256'],'canonical_artifact_id':'callback_20261006_facts','canonical_artifact_path':REL+'/callback-facts.json','canonical_artifact_sha256':sha(OUT/'callback-facts.json'),'callback_command':command,'environment':facts['environment'],'source_sha256':report['source_sha256'],'T3_sha256':facts['current_T3']['sha256']}
(OUT/'update-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
