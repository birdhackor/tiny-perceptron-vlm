"""Format this owner's supplied freeze06 judgments; retain own immutable old proofs."""
import copy,datetime,hashlib,json
from pathlib import Path
B=Path('docs/technical-reviews/artifacts/p7_technical_c')
M=Path('docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json')
T=Path('docs/course-revision-20261007-phase7/reviews/freeze-06/traces/technical/c/6443625578e24aa2a621a00f4148cf52.jsonl')
OLD=B/'group-c-freeze04-recheck-report.json'; OUT=B/'group-c-freeze06-recheck-report.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p); assert not p.exists(),f'Preserve existing {p}';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert sha(OLD)=='f617b44006b0265ba05c0e7deae96fde28486da3319ac3a3714b0bbb7734d0c3'
assert sha(M)=='0bbf4113fbd8288786bca540330fa9e2109e4fe848304a43b0321efa7bd23709'
old=json.loads(OLD.read_text()); report=copy.deepcopy(old); m=json.loads(M.read_text()); meta={p['page_id']:p for p in m['pages']}; changed=['11.1','11.13','12.11']
rows=[json.loads(x) for x in T.read_text().splitlines()];assert rows[-1]['event']=='complete'; assert sha(T)=='1458f0dcec18c0faa9eac5234c943c7b1e3bb42082dc418a6bd0532be1b8d4c6'
reused=[]
for p in old['pages']:
 current=meta[p['page_id']]
 assert sha(current['snapshot'])==current['source_sha256']
 for figure,h in current['figures_sha256'].items():assert sha(figure)==h
 if p['page_id'] not in changed:
  assert p['source_sha256']==current['source_sha256'] and p['figures_sha256']==current['figures_sha256']
  reused.append({k:p[k] for k in ['page_id','source_sha256','figures_sha256','trace_file']})
assert len(reused)==45 and len(old['pages'])==48
seal={'reviewer_task':'/root/p7_technical_c','manifest':{'path':str(M),'sha256':sha(M)},'trace':{'path':str(T),'sha256':sha(T)},'events':len(rows),'final_event':rows[-1],'checkpoint_units':{p:sorted(x['unit_index'] for x in rows if x.get('event')=='checkpoint' and x.get('page_id')==p) for p in ['9.8','9.9','9.10']+changed},'note':'Actual completed sequential callback; no textbook edits or new model training.'}
dump(B/'freeze06-completion-seal.json',seal)
def execution_update(p):
 page=p['page_id']; info=json.loads((B/f'freeze06-{page}-execution.json').read_text());assert info['exit_code']==0
 run=next(a for a in p['artifacts'] if a['id']==f'{page}-run')
 run.update(path=info['output']['path'],sha256=info['output']['sha256'],description='本人完整逐段讀完本版頁後執行原Python fences及明示的少量控制；真stdout/stderr，無訓練或ASR執行。',command=info['command'],result='實際退出碼0；版本和CPU裝置見首行，原輸出和控制輸出各有明示標記。',environment={'python':'3.13.5','torch':'2.14.1+cpu','device':'CPU'})
 p['artifacts'].append({'id':f'{page}-code-freeze06','kind':'code',**info['code'],'description':'本版實際執行程式；原fence與自己控制分開標示，未改教材。'})
 p['artifacts'].append({'id':f'{page}-execution-receipt-freeze06','kind':'execution','path':str(B/f'freeze06-{page}-execution.json'),'sha256':sha(B/f'freeze06-{page}-execution.json'),'description':'真命令、已完成單元、程式/輸出SHA和退出碼的保存紀錄。','command':info['command'],'result':'實際退出碼0，完整原結果另存stdout artifact。','environment':{'python':'3.13.5','torch':'2.14.1+cpu','device':'CPU'}})
 for s in p['sources']:
  if s['id']==f'{page}-exec':s['title']='本人freeze06逐段實讀後短CPU原fence及控制實際輸出'
 receipt=B/f'freeze06-{page}-page-view.json';v=json.loads(receipt.read_text())
 p['page_visual_check']={'status':'verified','required':True,'source_sha256':v['source_sha256'],'details':v['details'],'observation':v['observation'],'receipt':{'path':str(receipt),'sha256':sha(receipt)},'artifacts':v['artifacts']}
 p.update(source_sha256=meta[page]['source_sha256'],figures_sha256=meta[page]['figures_sha256'],trace_file=str(T),question_refs=[])
 p['recheck_of']={'path':old['pages'][next(i for i,x in enumerate(old['pages']) if x['page_id']==page)]['trace_file'],'sha256':'cd40820911ac411874568f2908ccc0d82635f3f7119d56b3306a58645ddb3a87','note':'本次指定完整頁版本回查，未設虛構issue ID；新的逐段理解和核對以本次trace為準。'}
 return p
# 11.1: new instruction ordering and old tensor behaviour personally verified.
p=execution_update(next(x for x in report['pages'] if x['page_id']=='11.1'))
p['summary']='本版把清零明確放在difference計算之前。原順序輸出shape(1,8)/True；清零後舊difference仍1.48053074，重新計算0/False，手設一列.5/3.5，短CPU全部一致。官方Linear和原投影研究核到維度相容仍需語義訓練；新增指令桌機/手機完整實看。'
p['variation']={'change':'將兩行no_grad清零放在計算difference之前，再重新計算。','prediction':'新差0/False、shape仍1,8；已存在difference不會隨W修改自動重算。','reason':'y=xAᵀ+b，兩輸入差中bias抵消；實際舊tensor和新forward對照吻合。'}
p['reviewer_corrections']=[{'checkpoint':'freeze06 11.1 unit1→unit2','correction':'unit1先前本次已讀forward措辭過早：該次返回只至127；其後另行返回並本人實讀130–134，unit2已明確更正。保留原記錄與時序，不是教材issue。'}]
linear=next(s for s in p['sources'] if s['id']=='linear');linear['inspection_note']='本次返回後本人實讀linear.py50–127的公式、shape、weight(out,in)和初始化；其後另行返回並實讀130–134的forward F.linear。前一單元過早措辭已於unit2更正，原記錄不改。'
llava=next(s for s in p['sources'] if s['id']=='llava');llava['inspection_note']='本次實讀原§4.1頁4提取行243–269：trainable W映射至等寬word embedding；另實讀280–315 Table2及Stage1，答案/停止訓練與凍結上游、只訓練W對齊。只支持投影及訓練分工，不把原預訓練模型能力移給本課。'
p['sources']=[s for s in p['sources'] if s['id']!='autograd']
p['sources'].append({'id':'autograd-notes','kind':'official_source','title':'PyTorch v2.9原Autograd mechanics與No-grad mode','verified':True,'url':'https://github.com/pytorch/pytorch/blob/v2.9.0/docs/source/notes/autograd.rst','version':'v2.9.0 SHA256 79b6775c349d0ba2815873e3eff510c3693da43c3bb1400331ab96f3b00261bd','authority_reason':'PyTorch原官方版本文件，說明forward建圖與no-grad原位更新。','checked_original':True,'inspection_note':'本次實讀13–43的執行時圖及每次forward，以及250–281 no-grad不記錄反向圖、參數原位更新後下一次forward再使用。舊tensor數值不反應修改另由實際CPU控制核到。','accessed_on':'2026-10-07'})
p['artifacts'] += [{'id':'linear-original','kind':'source_snapshot','path':str(B/'originals/torch-linear-v2.9.py'),'sha256':sha(B/'originals/torch-linear-v2.9.py'),'description':'原官方Linear定義與forward。'},{'id':'autograd-notes-original','kind':'source_snapshot','path':str(B/'originals/torch-autograd-notes-v2.9.rst'),'sha256':sha(B/'originals/torch-autograd-notes-v2.9.rst'),'description':'原官方forward/No-grad mode文件，非外部全文論文。'}]
c=p['claims'][0];c['evidence'][0]['locator']='linear.py50–89公式/shape與130–134 forward';c['evidence'][1]['locator']='原§4.1 Eq1頁4；§4.2 Stage1頁5（提取243–269、293–302）'
c=p['claims'][2];c['statement']='原fence先得shape(1,8)/True；本版指令先清零再重新計算得0/False；清零不使舊difference tensor自動重算。';c['verification'].update(expected='原順序True；清零後舊difference保持原值，新計算0/False。',observed='shape(1,8)，舊差1.4805307388305664/True，新差0.0/False。',details='原兩fences依序實跑；自己的控制明確再forward計算difference。沒有把依序清零後的舊difference當成False。')
c['evidence'].append({'source_id':'autograd-notes','locator':'13–43、250–281','supports':'forward每次建立運算結果與no-grad原位更新；是否重算由新forward控制，數值另以CPU驗證。'})
p['checks']['source_verification']['details']='本人本次真讀Linear公式及分開返回的forward、原投影論文§4.1/Stage1及官方No-grad文件；只核機制，不外推能力。'
p['checks']['limitations']['details']='W清零是手動控制非訓練；輸入敏感與8維相容不足以證明語義；修改W之前已算tensor保留舊值需重新計算。'
# 11.13: explicit insertion/deletion explanations and general edit-distance checks.
p=execution_update(next(x for x in report['pages'] if x['page_id']=='11.13'))
p['summary']='本版新增10100刪0、010補開頭1，各一次編輯/參考4，通用DP與原官方helper一致；不是zip錯位計數。原fence.75/False/.25/.5905及練習1010/101均核對。直接重算原30題仍40/57、EOS30、exact2；新增段落兩viewport實看。'
p['claims'][0]['location']='11.13 units0/2';p['claims'][0]['evidence'][0]['locator']='cer.py _cer_update23–50與_compute53–55；helper.py _edit_distance330–351';p['claims'][0]['evidence'][1]['locator']='官方seq2seq.py725–748評估原說明及EOS停止分支';p['claims'][0]['evidence'][1]['supports']='原eval取自身預測直到EOS停止，不檢查是否等於目標答案，支持停止與內容正確分開。'
p['claims'][1]['verification'].update(expected='.75/False/.25/.5905；1000、10100、010、101編輯距1，1010距0。',observed='原輸出一致；上述四例距1/CER.25，正確串距0/CER0。',details='自己逐行DP與本人本次實讀的原官方helper相互assert相等；.9^5=.59049四位舍入。')
p['claims'][2]['verification'].update(expected='原等長輸出；10100、010、101均由assert拒絕，通用DP可各算一次編輯。',observed='原輸出吻合；三個不同長例都捕获预期AssertionError，整次退出0。',details='原fence没有通用CER算法；本版文字插刪例由独立DP及原helper核對。')
p['claims'][3]['verification'].update(expected='原30題40編輯/57參考，EOS30、完整2。',observed='逐題原target/generated重算40/57=.7017543859649122，EOS30/30、exact2/30。',details='本次實讀原raw30samples後逐題DP與官方helper一致；原raw SHA9b08fb8ad9007160cb5ad4448e924ce1df81c9edab92487035dc71ea71a41fb7。同歷史任務計數，未重訓。',denominators={'test_strings':30,'reference_characters':57})
for s in p['sources']:
 if s['id']=='cer':s['inspection_note']='本次實讀cer.py全部80行，_cer_update逐樣本最小編輯數求和、_compute除參考總字數；另本次AST完整返回並讀helper._edit_distance330–351。用ASCII字串核新增位移例，沒有移用語音辨識成绩。'
 if s['id']=='seq2seq':s['inspection_note']='本次實讀385–412 target/self-prediction分支及725–751評估说明和746–748 EOS停止；支持停止标记与完整答案正确两件事，不用原教程能力或模型成绩。'
p['sources'].append({'id':'edit-helper','kind':'official_source','title':'TorchMetrics v1.8.2原最小編輯距DP','verified':True,'url':'https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/text/helper.py','version':'v1.8.2 SHA256 d80506f7929b1b1c807c3833749b7c57fbaf7edb4f1a014095fb0de5f1458ed2','authority_reason':'CER實際呼叫的原官方最小編輯距實作。','checked_original':True,'inspection_note':'本次AST抽取並實讀330–351完整函數，初始化邊界、相等不增加、三項min+1；之後执行本函数与自己独立一行DP逐例/30題一致。','accessed_on':'2026-10-07'})
p['sources'].append({'id':'ocr-raw-original','kind':'repository_code','title':'原OCR歷史逐題測量JSON','verified':True,'path':str(B/'originals/ocr-raw.json'),'sha256':sha(B/'originals/ocr-raw.json'),'version':'original recorded measurement SHA unchanged','inspection_note':'本次直接讀test30條target/generated/EOS及原計數；執行只重算已记录字串，不採人工摘要或重新训练。'})
p['artifacts'] += [{'id':'edit-helper-original','kind':'source_snapshot','path':str(B/'originals/torchmetrics-edit-helper-v1.8.2.py'),'sha256':sha(B/'originals/torchmetrics-edit-helper-v1.8.2.py'),'description':'官方DP原程式，實際控制抽取執行此原函數。'},{'id':'ocr-raw-original','kind':'source_snapshot','path':str(B/'originals/ocr-raw.json'),'sha256':sha(B/'originals/ocr-raw.json'),'description':'原逐題測量JSON；原test30條字串為本次核算輸入。'}]
p['claims'][1]['evidence'].append({'source_id':'edit-helper','locator':'_edit_distance330–351','supports':'新增不同長字串以通用最少插/刪/替換核算，不用zip。'})
p['claims'][3]['evidence'].append({'source_id':'ocr-raw-original','locator':'results.test.samples30條target/generated/eos','supports':'歷史測量的原逐條字串與停止標記，本次直接重算而非人工摘要。'})
p['checks']['factual_accuracy']['details']='新增插/刪例一次編輯与参考4分母相符；逐位、exact、EOS不同，独立乘法仅条件假设。'
p['checks']['numeric_verification']['details']='本版CPU原数值、插删DP和原30題重算40/57完全一致；历史率仅此固定OCR任务。'
p['checks']['source_verification']['details']='本人本次實讀官方CER完整文件、helper完整函数、seq2seq EOS原分支與原raw30條；执行交叉比對不採他人摘要。'
# 12.11: precise deterministic rule and rich input exception, with direct ASR source.
p=execution_update(next(x for x in report['pages'] if x['page_id']=='12.11'))
p['summary']='本版新增只根據同字串的確定性回答規則，条件充分：g(你好)同输出不能同時匹配low/high。原手寫dict True/False与两列输出实跑，自己的小控制给同答案而仅一列匹配；不运行ASR。原Whisper接口及rich输入例外本次核，新增段落桌機/手機均完整实看。'
p['claims'][0]['statement']='ASR能输出词文本；只收到同一转写字串且确定回答仅取决于字串，不能同时区分两不同pitch标签。时间/说者/音高扩展输入改变条件，直接声学入口仍须学会判别。'
p['claims'][0]['location']='12.11 units0/2新增確定性規則段'
p['claims'][0]['derivation_note']='本次新句核实：s1=s2=你好，g仅由s决定且确定，故g(s1)=g(s2)；y1=low≠y2=high，不能两者都正确。索引、外部状态、rich信息不在此仅字串条件内。'
p['sources'].append({'id':'same-input-derivation','kind':'derivation','title':'本人相同输入确定函数推导','verified':True,'details':'设s1=s2，确定函数g仅取s，则g(s1)=g(s2)。标签low与high不同，因此不能同时等于两标签。附加pitch或时间等信息使输入不再相同。短CPU规则仅例示此推导，不声称实验ASR模型能力。'})
p['claims'][0]['evidence'].append({'source_id':'same-input-derivation','locator':'本次同输入函数推导','supports':'新增确定性且仅字串条件下，两个异标签不能同时正确；rich输入是例外条件改变。'})
p['claims'][1]['statement']='本版手写dict比较输出转写相同True、音高相同False和你好low/high；同字串确定规则控制输出同low，无法同时匹配异标签。'
p['claims'][1]['verification'].update(expected='原True/False及你好low/high；同输入确定回答相同，仅一列匹配，rich inputs differ True。',observed='原四行吻合；控制答案[low,low]equalTrue，matches[True,False]，rich inputs differTrue，退出0。',details='手设条件与确定规则实例，不是真音讯/ASR/音高测量。')
for s in p['sources']:
 if s['id']=='asr-original':s['inspection_note']='本次返回后本人实读README8/15 ASR多任务原定义及115–125 result[text]转写接口。仅支持可输出文字，不称所有转写都只含字词，也不移用Whisper成绩。确定同输入限制由自己的明示推导与小CPU例支持。'
p['checks']['factual_accuracy']['details']='新句明确确定性规则只由相同字串决定，函数同输入同输出；官方ASR接口支持可取文字，不误称所有输出都无附加信息。'
p['checks']['limitations']['details']='手写无听写、pitch非波形估计；rich转写改变输入条件，直接音频也要学判别；不承诺新增入口必懂语气或所有任务绕过ASR。'
for page in changed:
 p=next(x for x in report['pages'] if x['page_id']==page);dump(B/'rechecks'/f'freeze06-{page}.json',p)
report.update(manifest_sha256=sha(M),verdict='pass',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary='本人freeze06完整48頁technical c版本報告：11.1、11.13、12.11及指定9.8–9.10前文全頁逐段回讀；新增機制/插刪/確定性規則查原官方來源、比例相稱短CPU與兩viewport位置。其餘45頁實際source/figure SHA exact相同，沿用本人原實讀、來源與證據。保留初始revise、全部原issue/trace/限制與不可用early callback，無教材修改或模型重訓。')
report['trace_files'].append({'path':str(T),'sha256':sha(T)})
report['freeze06_recheck_provenance']={'prior_full_group_report':{'path':str(OLD),'sha256':sha(OLD)},'changed_primary_pages':changed,'same_byte_reused_pages':reused,'callback_recheck_of':{'path':'docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/c/2bcb00175a2949dd82090328f77c573f.jsonl','sha256':'cd40820911ac411874568f2908ccc0d82635f3f7119d56b3306a58645ddb3a87'},'completion_seal':{'path':str(B/'freeze06-completion-seal.json'),'sha256':sha(B/'freeze06-completion-seal.json')},'note':'Original historical recheck_provenance retained unaltered below original field. New callback no invented issue ID; other45 page dictionaries unchanged from own previous report; original10.8 resolved issue and traces retained.'}
report['freeze06_reviewer_corrections']=[{'page_id':'9.9','units':'2→3','correction':'unit2 quote about安全判斷 actually unit3, recalled from own previously read same-byte source; unit3 actual returned reading and explicit correction preserved in new trace, no old record altered.'},{'page_id':'11.1','units':'1→2','correction':'source forward timing statement premature, returned 130–134 independently read before unit2, own explicit correction preserved; no textbook issue invented.'}]
assert report['unusable_prior_sessions']==old['unusable_prior_sessions'] and report['limitations']==old['limitations']
for before,after in zip(old['pages'],report['pages']):
 if before['page_id'] not in changed:assert before==after
assert next(p for p in report['pages'] if p['page_id']=='10.8')==next(p for p in old['pages'] if p['page_id']=='10.8')
dump(OUT,report)
print(json.dumps({'report':str(OUT),'sha256':sha(OUT),'verdict':report['verdict'],'pages':len(report['pages']),'new_primary_pages':changed,'same_byte_reused':len(reused),'trace':{'path':str(T),'sha256':sha(T)}},ensure_ascii=False))
