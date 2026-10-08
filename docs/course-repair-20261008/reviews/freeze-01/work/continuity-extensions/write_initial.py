import sys,json,hashlib,copy
from pathlib import Path
from datetime import datetime,timezone
B=Path('docs/course-repair-20261008/reviews/freeze-01')
sys.path.insert(0,str(B/'work/continuity-extensions/pydeps'))
from opencc import OpenCC
cc=OpenCC('s2t')
M=json.loads((B/'manifest.json').read_text()); N=json.loads((B/'reports/continuity-extensions-main-notes.json').read_text()); S=json.loads((B/'work/continuity-extensions/main-notes-seal.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B/'reports/continuity-extensions-main-notes.json')==S['sha256']
extra={
'A.2':{'read':'原位置選讀1完整文字、無新增圖；未點正式報告。','quote':'模型由隨機權重建立，沒有接續前面的SFT或文字預訓練權重。','clarification':'新店名分組、地址/來源成对只改当次输入，12題與雙版皆对對數不同；个别成功不保全部跟新公告。不是A.2手写source欄直接当回答。','necessity_now':'基本RAG分工/非权重更新/读錯可能已在主文；这些局部实测与重做配方可选，不替main补洞。'},
'B.3':{'read':'原位置選讀1完整文字、無新增圖；未點tools.json。','quote':'訊息的角色、system裡的規則文字與這個前綴都要一起保留。','clarification':'补完整system英文及CALC:add9,9五段真軌跡；仍assistant請求/外層18/user工具結果/assistant done各做不同工作。COPY不呼叫；旧messages快照必须与勘误一起看。','necessity_now':'回填user而非tool、模板限制、工具成功非整题成功已在主文；固定協議逐字重做需选读原文是合理完整配方。'},
'B.4':{'read':'原位置選讀1完整文字、無新增圖；未點实測報告。','quote':'56/56停止率仍不能替代20道正常任務的19/20成功率。','clarification':'18正常算術×2生成+2COPY+18一步=56；19/20的成功还要求算術真工具/COPY不工具，一步軌跡是重新生成不是套上旧最终答案。36/56文字快照失真但ID保存正确的勘误边界明确。','necessity_now':'動作預算、EOS/done与trace计量主文已教；56分母是强化证据解释，可选不影響四run机制。'},
'B.6':{'read':'原位置選讀1完整文字、無新增圖；未點Toolformer/旧模型报告。','quote':'它沒有接續先前的工具模型，也沒有把新選卡器與舊JSON生成模型串接成已驗證的助手。','clarification':'明确tool_choice独立重做入口；Toolformer仅研究相关方法，資料篩選訓練不同，本課未重做。','necessity_now':'陌生Toolformer名字不用于选卡目标/监督推理，属可选研究联系；主文已把不同權重与有限策略分开。'},
'C.7':{'read':'原位置選讀1完整文字、無新增圖；未點reasoning.json或实现。','quote':'正式組把梯度交給AdamW更新產生分數的參數…主例則只用人工固定的動作，直接調兩個分數。','clarification':'3数各one-hot6格→18输入→17固定動作，16数值+一列举；REINFORCE抽动作按相對回馈求方向，AdamW更新網絡，非前两語言模型RL。24新题×16抽樣=384动作，代理满分/嚴格0并非384新題。','necessity_now':'主文策略梯度方向及固定行动界限已教；完整有限策略身份/分母是可选核旧结果。AdamW未读实现也不用于本页两分数例的理由。'},
' training':{},
'readme':{'read':'helper确认本頁無折疊區','clarification':'無新增釐清；保留main原判及未验页面/执行边界。','necessity_now':'不適用'},
'environment':{'read':'helper确认本頁無折疊區','clarification':'無新增釐清；未驗安裝与外部链接。','necessity_now':'不適用'},
'curriculum':{'read':'helper确认本頁無折疊區','clarification':'無新增釐清；捕捉图浮动工具条遮挡仅静态觀察，未验证持续影响。','necessity_now':'不適用'},
'readme-en':{'read':'helper确认本頁無折疊區','clarification':'無新增釐清；英文入口不隐含另一份教材。','necessity_now':'不適用'},
' tool-trace-errata':{},
}
extra.pop(' training');extra.pop(' tool-trace-errata')
extra['tool-trace-errata']={'read':'helper確認本頁無折疊區','clarification':'B.3/B.4與T.11選讀均指向本勘误；已封存main的时序/字段解释没有变成實作或数字验证。','necessity_now':'看旧逐步messages时是条件式必要勘误；当前来源已公开warning，不能误用污染文字推前一步见到未来。'}
extra['training']={'read':'原位置九個折疊區全部实读；未点外部資料/实验报告/原模型或实现。另看三份固定配方各1280/390开启截图。','quote':'T.10的MoE教師使用第二組輸出的moe/model.pt及moe/dataset.json。','clarification':'三份固定教师能定位sft、style、moe与配对切分。sft block含多组选做，sft明确直接对话/两个预訓支线不同；style完整另含safety/LoRA但T.10只需style；MoE block明确第二组为本路線。其他补充依次分Transformer下载vs中文bigram/MLP、固定vqa partial只接头+最后区块vs本机CLI partial首末、DPOvsPPO预写卡、量化先120更新vs直接quantize、多模态joint直读sft+encoders不承接vqa，以及19章v2vs20章成熟上游。','necessity_now':'CE-OPT-1保留原可选分级。T.10固定三教师的命令是选择该完整路線者必要，main已经明說去展开；extra后可识别三组，不声称无需GPU、可全main执行或已验锚点。其它局部主文路线已各自成立，不要求搬全部长命令回主文。','visual_addition':{'captures':['training-1280-sft-recipe-open.png','training-390-sft-recipe-open.png','training-1280-fixed-style-recipe-open.png','training-390-fixed-style-recipe-open.png','training-1280-fixed-moe-recipe-open.png','training-390-fixed-moe-recipe-open.png'],'observed':'桌面三块文字与相关命令可辨；手机三块正文/产物分界可读，但命令右边的experiment/装置选項在截图中被横向截去。','not_verified':'未亲自點main锚点、details或横捲/複製，不能判手机版完整选命令路径通过；静态截图不证明链接自动展开。'}}
report={
'reviewer':N['reviewer'],'role':N['role'],'group':'extensions','at':datetime.now(timezone.utc).isoformat(),
'criteria_read_and_applied':{k:M['criteria_sha256'][k] for k in ['SKILL.md','references/review-protocol.md','references/calibration.md']},
'criteria_limits':'main notes的criteria保存manifest全criteria指纹作身份资料；实际只读/应用上列三份通用criteria，未读project-context、writing-prompt或其他角色材料。',
'gate':N['gate'],'manifest_sha256':sha(B/'manifest.json'),'scoped_route':N['scoped_route'],
'main_notes_seal':S,'main_notes_immutable_path':'reports/continuity-extensions-main-notes.json',
'independence':dict(N['independence'],extra_sources_read=True,peer_reports_read=False,author_answers_read=False),
'pages':[],'necessary_findings':[],'optional_findings':[],'unknowns_and_limits':[],
'cumulative_burden':'主文明确分开外部资料与永久权重、人工请求与真生成、工具成功与整题done、一次EOS与动作预算、外层trace与模板角色、选卡与JSON不同权重、固定action两分数与实际策略/语言模型、独立小实验与共同成品。几次回读都有明示对象和理由，没有借同行答案补先备。T.10选择完整蒸餾路線需回三处details，已明示不能使用小练习；保留可选标示改善而不误认必要语义缺口。curriculum截图遮挡及手机完整命令辨認未知，不从原SVG正确推页面通过。',
'overall':'就實讀来源语义及依赖，無已確定的主要概念阻礙或必要语义缺漏；保留1项可选路線标示改善。当前是有限范围AI来源初判，不是整书、人类首读或完整实机/页面验收。',
'visual_evidence':N['visual_read']}
for i,p in enumerate(N['pages']):
 # Preserve the full original judgment as a distinct read stage, never backfill optional explanations.
 report['pages'].append({'page_id':p['page_id'],'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],'main_evidence_reference':{'path':'reports/continuity-extensions-main-notes.json','sha256':S['sha256'],'json_pointer':f'/pages/{i}'},'original_promise':p['promise'],'actual_dependencies':p['dependencies'],'source_pass_basis':{'identity_operation_example':{k:v for k,v in p['four_questions'].items() if k not in ['why_mechanism_property','why_need_use']},'second_question_mechanism_property':p['four_questions']['why_mechanism_property'],'second_question_need_use':p['four_questions']['why_need_use']},'main_unknowns_preserved':p['unknowns'],'main_findings_preserved':p['findings'],'burden':p['cumulative_burden'],'extras_added_after_seal':extra[p['page_id']]})
 report['optional_findings'] += p['findings']
 if p['unknowns']: report['unknowns_and_limits'].append({'page_id':p['page_id'],'items':p['unknowns']})
report['visual_evidence']['extras_capture_addition']=extra['training']['visual_addition']
visualpaths=[]
for stem in N['visual_read']['assigned_original_svgs_and_isolated_pngs']:
 for w in [640,360]: visualpaths.append(B/'renders'/f'{stem}-{w}.png')
for f in N['visual_read']['page_capture_images_seen']+extra['training']['visual_addition']['captures']: visualpaths.append(B/'page-captures'/f)
report['visual_evidence']['inspected_artifact_sha256']={str(p.relative_to(B)):sha(p) for p in visualpaths}
report['limits']=['所有原始main报告bytes保持不变；本报告只追加选读理解。','只看冻结源码、原SVG、孤立PNG与指定捕捉图；未亲自操控网页或實機训练，不把delivery當理解證據。','依賴前文的图并未视验，依賴判断只用其实读主文文字；不宣称全书依赖覆盖。','未查看任何同行reader/technical/continuity报告、root诊断/作者答案或旧问题；未读取原始模型结果与实现替代正文。']
path=B/'reports/continuity-extensions-initial.json'
assert not path.exists(),'Refuse to overwrite an existing initial report'
path.write_text(cc.convert(json.dumps(report,ensure_ascii=False,indent=2))+'\n')
seal={'reviewer':N['reviewer'],'at':datetime.now(timezone.utc).isoformat(),'path':'reports/continuity-extensions-initial.json','sha256':sha(path)}
sp=B/'reports/continuity-extensions-initial.seal.json'; assert not sp.exists()
sp.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
assert sha(B/'reports/continuity-extensions-main-notes.json')==S['sha256']
print(json.dumps({'report':str(path),'report_sha256':seal['sha256'],'seal':str(sp),'seal_sha256':sha(sp),'main_notes_sha256':S['sha256'],'pages':len(report['pages']),'necessary_findings':0,'optional_findings':len(report['optional_findings'])},ensure_ascii=False,indent=2))
