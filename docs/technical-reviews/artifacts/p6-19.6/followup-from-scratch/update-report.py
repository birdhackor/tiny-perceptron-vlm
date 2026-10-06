from pathlib import Path
import hashlib,json
out=Path(__file__).resolve().parent;root=out.parents[4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((out/'initial-pass-report.opaque.json').read_text())
result=json.loads((out/'result.json').read_text())
report['source_sha256']=result['new_section_sha256']
report['verdict']='pass'
new=[]
def art(identifier,name,kind,description,**more):
    p=out/name
    new.append(dict(id=identifier,kind=kind,path=str(p.relative_to(root)),sha256=sha(p),description=description,**more))
art('a_followup_exec','result.json','execution','本人精確復核新從零訓練句、真正依賴、所有未變初審證據SHA、RMSNorm全1初始化及原階段訓練來源；無神經重跑。',command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.6/followup-from-scratch/recheck.py',result='exit 0；只有該句替換；125既有artifacts與75原件不變；81個inert parameter names均在本專案stage選參契約內，RMSNorm顯式從ones起始；新從零訓練主張核實。',environment=result['environment'])
for identifier,name,kind,description in [
 ('a_followup_initial','initial-pass-report.opaque.json','source_snapshot','本人初次pass報告的原始opaque保留；未回改先前版本與判定。'),
 ('a_followup_current','19.6.current.md','source_snapshot','本次本人完整重讀之目前小節原始UTF-8 bytes。'),
 ('a_followup_diff','section.diff.txt','source_snapshot','本人以兩版原始bytes精確確認單句變更之diff。'),
 ('a_followup_note','inspection.md','source_snapshot','本人新句語意、RMSNorm初值、模組來源/訓練鏈與證據復用範圍的實際復核記錄。'),
 ('a_followup_code','recheck.py','code','本次有界原碼AST、原provenance指標及SHA復查程式，不匯入或執行神經網路。'),
 ('a_followup_stdout','recheck.stdout.txt','source_snapshot','本次真執行stdout：新舊版本、SHA復用和局部構造發現。'),
 ('a_followup_stderr','recheck.stderr.txt','source_snapshot','本次真執行stderr（成功時空檔）。')]:art(identifier,name,kind,description)
for i,(original,source) in enumerate(result['dependency_sources'].items()):
    p=root/source['copy']
    new.append(dict(id='a_followup_source_'+str(i),kind='source_snapshot',path=source['copy'],sha256=sha(p),description='本人此次依AST重新定位並讀取必要構造/訓練範圍；保存完整原始檔和SHA：'+original))
report['artifacts'].extend(new)
report['sources'].append(dict(id='s_followup_exec',kind='execution',title='Owner from-scratch wording/provenance recheck',verified=True,artifact_id='a_followup_exec'))
for identifier,original,note in [
 ('s_followup_core','tiny_perceptron/model.py','AST後親讀imports3–11、Block.__init__32–41、TinyLM.__init__54–66：全部locally constructed nn零件，未載外部模型。'),
 ('s_followup_modern','tiny_perceptron/modern.py','AST後親讀RMSNorm8–16、DenseFFN.__init__36–43、MoEFFN.__init__59–65；RMSNorm.weight是nn.Parameter(torch.ones(width))，並非每個scalar隨機。'),
 ('s_followup_attention','tiny_perceptron/attention.py','AST後親讀CausalAttention.__init__32–46：四個新的nn.Linear，不是預訓練attention權重。')]:
    p=root/result['dependency_sources'][original]['copy']
    report['sources'].append(dict(id=identifier,kind='repository_code',title=original+' (owner current wording follow-up)',path=str(p.relative_to(root)),sha256=sha(p),version=result['utc']+' current source frozen at this SHA',verified=True,inspection_note=note))
c=next(x for x in report['claims'] if x['id']=='c13')
c['statement']='文字核心、圖片/讀字/語音入口和接頭等神經零件，均在本專案建立並由本專案從零開始的分階段訓練鏈訓練；共同訓練固定感知骨幹/分類頭而更新文字核心與接頭。'
c['location']='最後訓練段，現句「所有神經零件都由本專案從零訓練」及前句分階段與joint安排'
c['scope']='成立於本專案神經模組與own-checkpoint訓練來源，不聲稱每一個scalar初值都隨機，也不聲稱每個參數都有非零梯度或品質達標。RMSNorm縮放係數明確从ones起始；固定log-mel/字型/資料處理不是預訓練神經模型。沒有重跑訓練或神經推論。'
c['evidence']=[
 dict(source_id='s_model',locator='MaskedBlock.__init__109–116；SelftrainedLanguageModel.__init__134–151；三感知encoder constructors188–261；LimitedAssistant296–302',supports='共享文字核心與三個感知入口由本專案local class/nn零件建立，沒有外部預訓練neural constructor。'),
 dict(source_id='s_followup_core',locator='Block.__init__32–41；TinyLM.__init__54–66',supports='再追到核心的實際本地構造：embedding、attention、FFN和norm都是新建。'),
 dict(source_id='s_followup_modern',locator='RMSNorm.__init__9–12；DenseFFN36–43；MoEFFN59–65',supports='norm的可訓練scale初值是ones，說明從零訓練的語意容許確定初值；FFN/router/expert亦為本地新建。'),
 dict(source_id='s_followup_attention',locator='CausalAttention.__init__32–46',supports='核心attention權重是新建的四個linear。'),
 dict(source_id='s_train',locator='PERCEPTION_BRIDGE_PREFIXES48–61；set_trainable284–296；objective519–554；main723–789；update loop904–932',supports='感知stage有head/backbone loss，core/bridge由shared language loss續訓；初始化/載入僅沿own project checkpoint而joint按明確prefix固定感知骨幹及head。'),
 dict(source_id='s_data',locator='perception_loss244–271',supports='各感知head具有真image CE/audio CE/OCR CTC訓練目標；supervision供loss，不輸入prompt。'),
 dict(source_id='s_followup_exec',locator='result.json /actual_original_commands、/stage_provenance、/deterministic_initialization_example、/trainability_contract_coverage、/reused_artifact_hashes',supports='實際核對root pretrain argv無init/resume、後續沿本專案checkpoint鏈，階段原紀錄已完成；再次驗未變SHA並用inert names執行原選參規則。')]
c['artifact_ids']=list(dict.fromkeys(c['artifact_ids']+['a_followup_exec','a_followup_note','a_followup_diff']))
c['verification']={
 'method':'executed',
 'expected':'本專案建立各神經零件，沿自己從零開始的stage/checkpoint來源訓練；新句不要求所有scalar隨機初值。',
 'observed':'本人重讀的local構造和原stage argv/provenance相符；RMSNorm.weight显式torch.ones；125旧artifacts與75原件hash吻合；原選參規則覆蓋81個inert parameter names，沒有新神經執行。',
 'details':'本次實際執行SHA/AST/原JSON pointers及inert選參審查，重讀objective/perception_loss/update branches，復用未變的初審神經結果。選參覆蓋不是每一scalar曾有非零梯度的證明；raw origin marker沿用原schema，不按字面解釋為every-scalar random。'}
report['issues'].append(dict(claim_id='c13',status='resolved',details='原句「所有神經權重都由隨機初始化開始」可能被理解為每一個scalar均隨機；本人在此次復查原碼親見RMSNorm.scale從全1起始。初審報告已原樣保留。',resolution='現稿改為「所有神經零件都由本專案從零訓練」，本人追查local構造與own stage來源，改写c13支持範圍及驗證，確認新句與確定normalization初值並不矛盾。'))
report['read_scope']='初審範圍與證據保留。此次本人完整重讀目前19.6；精確單句替換及真正依賴的構造/初始化/感知loss/joint選參/update分支已重新核對。未讀reader、其他technical報告或作者log。原三圖及已執行神經結果經SHA核實不變後復用；未重render或神經重跑。'
report['review_history']=[dict(event='initial independent pass retained opaque',section_sha256=result['old_section_sha256'],report_path=str((out/'initial-pass-report.opaque.json').relative_to(root)),report_sha256=sha(out/'initial-pass-report.opaque.json')),dict(event='owner genuine changed-sentence/provenance follow-up',checked_at=result['utc'],section_sha256=result['new_section_sha256'],artifact_id='a_followup_exec',changed_claim_ids=['c13'],result='pass')]
report['checks']['factual_accuracy']['details']+=' 此次新句已由原owner重讀local構造及own stage來源，c13已改写而非只換hash。'
report['checks']['source_verification']['details']+=' 新依賴core/modern/attention構造親讀並凍結；已核125初審artifacts和75原件SHA未變。'
report['checks']['limitations']['details']+=' c13限定為從零建立與本專案訓練，不要求every-scalar random或所有scalar非零梯度；RMSNorm ones已明確核對。'
report['checks']['figure_consistency']['details']+=' 本次SVG SHA未變，沿用原owner先前實際640/360 render/view；沒有宣稱重render。'
(root/'docs/technical-reviews/19.6.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Updated own c13 and provenance/scope evidence; current section pass; original report preserved opaque.')
