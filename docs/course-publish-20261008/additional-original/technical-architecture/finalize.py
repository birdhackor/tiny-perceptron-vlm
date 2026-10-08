import json,hashlib,datetime
from pathlib import Path
B=Path('/workspace/work/tutorial-audit-20261008');W=Path('/workspace/work/technical-architecture');D=Path('/workspace/tiny-perceptron-vlm/docs/course-experiments/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
m=json.loads((B/'manifest.json').read_text());rows=json.loads((W/'page-work.json').read_text());idx={r['page_id']:r for r in rows}
assert len(rows)==71 and set(idx)==set(m['groups']['architecture']['pages'])
for p in m['inventory']['pages']:
 if p['page_id'] in idx:
  r=idx[p['page_id']];s=(B/'freeze/sources'/f"{p['page_id']}.md").read_text();assert r['source_sha256']==sha(B/'freeze/sources'/f"{p['page_id']}.md")==p['source_sha256']
  assert all(q in s for q in r['quoted_basis'])
# Close source checks that had been held open while separately reading the direct raw results.
replacements={
'14.1':('modern原報告核對待完成，折疊實測另查。','modern.json:相同seed42/data409/51/52、64wide2layer4heads、240updates0skip/452102targets；RoPE133376参数、testNLL1.778118vsbaseline2.291627；Onceupon生成was a loked…仍错，正文限定相符。'),
'14.4':('freeze DenseFFN gelu/relu2分支與正文相符；整模型數字後續原JSON核。','freeze DenseFFN gelu/relu2分支与正文相符；modern.json同141568参数、240updates0skip：baseline testNLL2.291627、ReLU²2.264379，示例重复the，未由平均NLL称可用回答。'),
'15.4':('freeze MoEFFN先topk再where分派，仅选中expert；统计分母token×k将查原JSON。','freeze MoEFFN先topk再where分派，仅选中expert；moe.json routing top1分母39256、top2 78512=2×39256，与token×k定义相符。'),
'15.8':('counts8536+25630+4875+215=39256；熵/ln4≈.64755，后续原JSON核。','moe.json top1aux0 layer2counts8536+25630+4875+215=39256；normalizedentropy.6475547；meanprob与dispatch另列，正文不混。'),
'15.9':('原实现基础aux对全表；正式训练PAD排除处需查train路径，来源说明两个层相加。','原实现基础aux对全表；frozenarchitecture.py117—138 hook取routing_weights[valid]与expert_indices[valid]重算EΣf·p，两个层相加；moe.json top2aux.01平均aux.9906323及meanprob/dispatch不同相符。'),
'15.10':('proxy142080/208256需对原JSON与实作参数表。','moe.json:total340608=experts264704+router512+other75392，top1proxy142080=264704/4+512+75392，top2proxy208256；实作_parameter_proxy估计同口径，不称FLOPs。'),
'15.11':('source引用L4全更新时间，不将CPU短例与GPU训练混计；原JSON待核。','source引用L4全更新时间，不将CPU短例与GPU训练混计；moe.json每版180updates337761targets，dense64 12.675974ms/top1aux.01 27.630116/top2aux.01 27.241951；CPU冻结实作实际参数280/336吻合。'),
'15.13':('七组原JSON核对待完成；不以active proxy叫精确FLOPs或最佳教师选择。','七组moe.json原始参数/proxy/valtestNLL/timing/allocated字段实读：dense64/80/104 total141568/207680/329888，top1/2 aux0/.01均340608；testdense2.320927/2.309058/2.249426，对aux.01top1 2.319628/top2 2.301704；相同180updates0skip337761targets与409/51/52数据，不以proxy称精确FLOPs或最佳教师。')}
for pid,(old,new) in replacements.items():
 a=idx[pid]['checks'];assert old in a; a[a.index(old)]=new
idx['14.2']['checks'].append('modern.json RMS141248比baseline141568少320，与5个LN移除每个64beta一致；test2.286315vs2.291627、RMS13.951992ms vsbaseline13.062209，不许诺少操作必快。')
idx['14.5']['checks'].append('modern.json SwiGLU174848vsbaseline141568，相同hidden256而多33280；test2.280086/val2.254443，非匹配容量对照，正文保留区别。')
idx['14.6']['checks'].append('modern.json tied124672vs141568少264×64=16896；test2.384216、initialNLL43.87647与随机共享可能大logit的限定吻合，未称必然改善。')
idx['15.7']['checks'].append('moe.json task-only router gradient before有效1004目标两层norm.00667636/.00657814、训练后.02446164/.0139968均非零；unweightedaux另记不混coef.01与task-only。')
idx['14.10']['quoted_basis']=['為什麼還有下面那一步？注意力會比較很多位置的分數，分數彼此差多少，會影響權重集中在少數位置還是分散到更多位置。','分頻縮放決定每組指針怎麼轉，尺度調整決定比對分數換成權重時有多集中，兩者處理不同部分。']
idx['14.10']['source_judgment']['need_purpose']='第一项近/远需求有连結；第二项只说明尺度控制集中度，未说明窗口延长后何以需要另调此尺度。与正文自问「为什么还要这一步」相称的最短用途关系仍缺。'
idx['14.10']['checks'][1]='已实际读YaRN2309.00071v3 §3.3（pdftotext289—336）及AppendixA.3（743—786）；论文原始用途是extendedwindow里temperature带来perplexity改善的经验观察、按extensionfactor拟合；非要求完整频段算法/最佳参数。'
for pid in ['17.14','18.2','18.8','18.12','18.14']:
 idx[pid]['checks'].append('额外CPU冻结实现小检查已执行，结果在cpu-compression-contracts.json；仅随机结构/forward/局部backward，不做optimizer/训练：teacher16960/67840、student6104/24416、MoE18048、packedstudent15680，KL.19274476/ignoredgrad0/teacherNone，STEgrad[1,1,1]按对应页核。')
# Each directly referenced empirical artifact has explicit actual lookup scope, version and limits.
specs=[
('modern.json',['14.1','14.2','14.4','14.5','14.6'],'results variants config/parameters, initialization and final losses, optimizer updates/skips/effective targets, train/validation/test counts/hashes, generated examples, timing; histories not fully read'),
('moe.json',[f'15.{i}'for i in range(4,14)],'seven variant config/parameter proxy/losses, task-only and unweighted auxiliary router-gradient probes, per-layer routing counts/normalized entropy/mean probability and dispatch share/mean aux, common data counts and target budgets, timings and allocated baseline/peak; histories not fully read'),
('efficiency.json',[f'16.{i}'for i in range(2,12)],'MHA/GQA config/budget/evaluation/cache; cache exact IDs and errors; packing output/gradient/isolation/updates; accumulation microcounts and total targets; manual/SDPA actual profiled backend; checkpoint errors; update baselines/peaks/timings; Inductor compiler status/operator count/first-call/eager/compiled/amortization; histories not fully read'),
('precision.json',['16.7'],'FP32/BF16/FP16 config, common update and target budgets/skips, main weight/logit dtypes and finiteness/GradScaler, validation/test denominators/NLL/EM/EOS/sample0, allocated baseline/peak/additional and timings; histories not fully read'),
('flash_probe.json',['16.8','16.9'],'forced backend flags, shapes/dtypes/timing and memory scopes, manual-route arithmetic, output and Q/K/V gradient max errors/tolerances/finiteness, forward/backward profiler operators and CUDA kernels, forward/backward allocated baseline/peak/additional and 9-repetition median timing; no full pretrained model involved'),
('quantization.json',['17.1','17.7','17.9','17.10','17.15'],'teacher/source/data provenance; shared init and training120updates13610targets; same source/reload flags; every FP32/packed4/packed8 storage part and file bytes; validation/test counts/NLL/EM/EOS; fixed prompt/decode timing and co-resident allocator baseline; generated samples. Files/weights/data not independently downloaded'),
('qat.json',['17.11','17.14','17.15'],'weight-only per-row quantizer, STE, activation_quantization=false, matched init/batchplan350updates39348targets; six run storage/eval validation36/test69targets EM/EOS/NLL; deployed vs simulated logit equality; red:square:high per-branch output. Histories and physical artifacts not fully read'),
('distillation.json',[f'18.{i}'for i in range(1,13)]+['18.14'],'three tasks teacher checkpoint/config/provenance/frozen flag, original-family data splits/intersections and cache costs, actual hard audit all five wrong dates, student branch init/batchplan/update/effective-target/objective/alpha/T/T² flags, validation/test/storage, teacher and student style rubrics and 2+2 JSON samples, attribute blue→ble packed sample. Every training history and all story generated samples not fully read'),
('multimodal_distillation.json',['18.13'],'VQA/joint teacher provenance/data split/frozen and visual16→4/language64→32; CE/CE_KL init/plans350updates/effective answers/objective/T²/alignment examples; test/blank image/blank audio counts/NLL/EOS, all joint CE_KL normal and blank-image generated-ID lists and circle/pitch observations. Histories/physical modal inputs not fully read')]
evidence=[]
for fn,pages,scope in specs:
 p=D/fn;x=json.loads(p.read_text());e={'id':fn,'path':str(p),'sha256':sha(p),'version':{k:x.get(k) for k in ['schema_version','revision','device','seed','torch_version','python_version']},'actual_read_scope':scope,'pages':pages,'verification':'原始JSON欄位及相称冻结程式核；不是GPU重跑/物理checkpoint校验/多seed复现','unverified':['GPU硬件条件与私有artifact SHA未独立再量','长期/跨seed/跨设备外推不验证']};evidence.append(e)
 for pid in pages:
  if pid in idx:idx[pid].setdefault('empirical_evidence_refs',[]).append(fn)
# Seen images are recorded only after actual view_image calls, not inferred from SVG text.
visual_notes={
'rewrite-14-shared-rotation':'Q/K2/5→12/15，共同450°，差135°/dot−.7071不变。',
'rewrite-14-rotation-components':'45°cos/sin+.7071，135°cos−.7071/sin+.7071，符号/轴一致。',
'rewrite-14-normalization':'[1,2,3]LN均值2/std√(2/3)±1.2247；RMS√(14/3) .4629/.9258/1.3887且γ1β0。',
'rewrite-14-7-position-range':'两行卡仍连续0—15，原0—7蓝/new8—15橘，dist7/15明确。',
'rewrite-14-8-position-interpolation':'16卡不合并，0→0、1→.5、14→7、15→7.5。',
'rewrite-14-9-nearby-angles':'近邻快60→30、慢15→7.5；原与除2两列可对比。',
'rewrite-14-10-yarn-components':'快保持/中逐步过渡/慢较多插值；独立positive scale在softmax前，机制正确但未补用途关系。',
'rewrite-15-combine':'[1,0]/[0,2]权.7/.2，直加.7/.4与除.9后.7778/.4444对照。',
'rewrite-15-row-restore':'来源2,0,2与贡献1,2,3累加，row2=4/4、row0=2/2、row1zero。',
'rewrite-16-cache-append':'旧0/1/2追加3，新Q读全4；无删除旧KV。',
'rewrite-16-query-kv-sharing':'4Q heads各KV与一共享KV，3位置每4特征FP32 cache384/96。',
'rewrite-16-packed-mask':'queryrow/keycol，A0/A1/B2/B3只各自causal下三角，跨文档全零。',
'rewrite-16-causal-mask':'4×4tril，True可见，row是query/column是key。',
'rewrite-16-online-softmax':'旧1.5/25×.5→.75/12.5，再+1/30→1.75/42.5，24.2857。',
'flash_allocated_memory':'灰65MiB基线；forward新增24.25/.267总89.25/65.27；backward新增32.75/2.032总97.75/67.03。',
'rewrite-16-12-sliding-path':'input3→layer1pos5→layer2pos7，每层5/6/7直接讀法，跨层路径标明。',
'rewrite-16-13-visible-pairs':'完整/滑W3/blocklocal3三张6×6，位置3读0—3/1—3/3。',
'p7-16.14-draft-verify':'接受今天，雨天首错由晴天替换，后缀吗丢弃；greedy不是vote。',
'rewrite-17-grid-mapping':'.7/.5=1.4→码1→.5与1.2→码2→1，原与还原差.2。',
'rewrite-17-int4-byte':'−8/−1→码0/7，首低次高01110000=112，先low解码−8，顺序一致。',
'rewrite-17-qat-flow':'floatmaster→fakequant/STE→step→deploypack；单向量例无更新提示。',
'p7-18-three-targets':'同一2+2=4，原与teacherhard同，soft.7/.2/.1，三平行路非先后。',
'p7-18-temperature':'[4,1,0]T1/2/4，4/3/100蓝/橘/绿稳定；93.62/4.66/1.71→73.61/16.43/9.96→54.34/25.67/19.99。',
'rewrite-18-vocab-alignment':'teacherID0猫.8/1狗.2，studentID0狗.2/1猫.8；重排[1,0]MAE.6→0。',
'rewrite-18-modal-answer-rows':'A0教师15/student3、A1 16/4、A2 17/5，prediction在answerinput前一位。'}
visual=[]
for p in m['inventory']['pages']:
 if p['page_id'] not in idx:continue
 for f,h in p['figures_sha256'].items():
  name=Path(f).stem;assert name in visual_notes
  render=B/'renders'/f'{name}-640.png';assert render.exists()
  v={'page_id':p['page_id'],'source_asset':f,'source_sha256':h,'render':str(render),'render_sha256':sha(render),'actual_view':'640px PNG via view_image; result recorded after viewing','judgment':'數值/索引/箭头/图文机制吻合，未发现技术图错。','basis':visual_notes[name],'unverified':['360px未亲看；真实网页布局/交互未验']};visual.append(v);idx[p['page_id']].setdefault('visual_refs',[]).append(f)
assert len(visual)==25
prereqs=[]
for pid in ['4.1','4.3','4.4','4.6']:
 p=B/'freeze/sources'/f'{pid}.md';prereqs.append({'page_id':pid,'path':str(p),'source_sha256':sha(p),'actual_read_scope':'完整冻结正文及details','purpose':'回查Q/K/V、causal mask、分数到softmax与head的技术语义；未把跳读其余前文当教材缺失。'})
p=B/'freeze/sources/training.md';t=p.read_text();sec=t[t.index('## T.8'):t.index('## T.9')];prereqs.append({'page_id':'training:T.8','path':str(p),'source_sha256':sha(p),'section_sha256':hashlib.sha256(sec.encode()).hexdigest(),'actual_read_scope':'T.8完整，source lines356—413；未读其余training当作先修已教证据','purpose':'正文实测条件/一次换一个条件的原始入口。'})
impl=[]
for fn,scope in [('tiny_perceptron/modern.py','全文'),('tiny_perceptron/attention.py','全文'),('tiny_perceptron/model.py','全文'),('tiny_perceptron/quantization.py','全文'),('tiny_perceptron/alignment.py','仅45—65 distillation_kl/loss'),('scripts/course_experiments/architecture.py','117—138、154—226、333—427、496—559；其他只定位定义，不称全文'),('scripts/course_experiments/compression.py','350—449、574—665、887—990；其他只定位定义，不称全文')]:
 p=B/'freeze/implementation'/fn;impl.append({'path':str(p),'sha256':sha(p),'actual_read_scope':scope})
paper={'path':str(W/'yarn-2309.00071v3.pdf'),'sha256':sha(W/'yarn-2309.00071v3.pdf'),'url':'https://arxiv.org/pdf/2309.00071v3','text_path':str(W/'yarn-2309.00071v3.txt'),'text_sha256':sha(W/'yarn-2309.00071v3.txt'),'actual_read_scope':'§3.3 pdftotext layout lines289—336；AppendixA.3 lines743—786，不称整篇或§3.2全文','quote':'introducing a temperature t on the logits before the attention softmax has a uniform impact on perplexity ... over the extended context window','purpose':'只核第二步用途是否源于长窗口经验现象；不把作者LLaMA结果当本课实测。'}
issue={'id':'technical-architecture-14.10-yarn-scale-purpose','page_id':'14.10','severity':'burden','classification':'必要设计选择的需要/用途关系缺口，非机制/数字错误','quoted_basis':idx['14.10']['quoted_basis'],'already_taught':'原训练圈数分频、快保近差/慢压远范围/中间过渡已清楚；softmax前正尺度放大既有分数差、较尖不等正确、需训练验收已教。','missing_minimum_relation':'为什么延长窗口并改变位置规则后，还考虑调整注意力集中程度；现在「为什么还要这一步」的答案只解释尺度能做什么，未接到本段长窗口问题。','current_impact':'读者能操作分频和scale，却仍需自行补发明scale第二步的理由；本节本身把两部分都列为所介绍YaRN的必要组成。','minimum_repair':'在现有为什么段后补一小段：扩展窗口会改变可比较位置与权重分布，YaRN作者在指定长窗口实验里观察到调温度有助perplexity，因此把scale作为与位置插值搭配的可调项；用短/长任务验证，而不是保证更集中必正确。可只给经验依据，不需完整频段算法、最佳c或entropy定理。','rewrite_scale':'paragraph','does_not_require':'章节重写/顺序大改/增加模型训练/撤掉现有手算与限定','discovery_time':idx['14.10']['recorded_at'],'discovery_source':'完整14.10来源首个技术判断时发现；未看其他reviewer/reader/旧问题清单；其后只读正文直接引用YaRN原文核用途','original_evidence_ref':paper,'unverified':['该缺口对真人初学者的理解影响未测；等待独立交叉裁决','未验证任意模型/扩展倍数都须相同scale或通用entropy因果']}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report={'schema_version':1,'kind':'source-isolated technical initial audit','reviewer':'/root/technical_architecture','role':'technical','group':'architecture','sealed_at':now,'baseline_commit':m['baseline_commit'],'criteria_sha256':{k:v for k,v in m['criteria_sha256'].items() if k in ['SKILL.md','references/review-protocol.md','references/calibration.md','references/project-context.md']},'actual_prerequisites':prereqs,'pages':[idx[p]for p in m['groups']['architecture']['pages']],'issues':[issue],'coverage':{'assigned':71,'fully_read':71,'page_ids':m['groups']['architecture']['pages'],'individual_source_judgments':71,'individual_quoted_basis':71,'source_hashes_matched_manifest':71,'read_boundary':'每页正文与所有details/代码/练习完整实际阅读；技术轮不是无提示逐段首读。没有用标题/搜索/抽样代替全页。','chapter_counts':{'14':11,'15':14,'16':15,'17':16,'18':15}},'visual_scope':{'assigned_unique_figures':25,'actually_viewed_640':25,'actually_viewed_360':0,'figures':visual,'unverified':['实际网站desktop/mobile/折叠交互和可读字号未验；未开展真人学生看图测试']},'execution_scope':{'frozen_implementation_reads':impl,'cpu_contracts':[{'path':str(W/f),'sha256':sha(W/f),'scope':json.loads((W/f).read_text())['scope']}for f in ['cpu-contracts.json','cpu-compression-contracts.json']],'arithmetic':'逐页固定数值、轴、索引、分母、梯度/缩放等小算例按比例检查；不是对每个完整snip执行训练。','raw_evidence':evidence,'external_original':paper,'not_run':['长训练/GPU基准/统计多seed/新模型训练','私有checkpoint或dataset下载后重算指纹与家族交集','compile完整LM、稀疏/滑窗kernel或神经推测解码','实际网页布局/真人学生测试'],'failed_attempts':[{'action':'用默认python读CPU contracts','result':'默认环境无torch；改用已安装repo .venv完成，未判教材缺陷。'},{'action':'某些原始JSON集合字段打印','result':'两次聚合字段输出过大截断；数值/训练/对齐相关项另以精确字段重读。未把被截断训练history标成全文已读。'}]},'isolation':{'before_seal':'未读reports/synthesis/notes/traces/人类checks判断/旧reader technical continuity validation记录/其他reviewer结论。可见manifest的结构ID仅用于来源路线，未读其reviewer内容。','technical_round_only':True,'source_first':True,'cross_review_used':False,'human_student_test':False,'tutorial_modified':False},'repair_assessment':{'necessary':{'blocker':0,'burden':1,'scale':'一处paragraph；本组技术证据不支持章节或顺序大改'},'optional':{'count':0,'note':'未为此轮凭空加新目标或把选读细节设成必须教。'},'unverified':'上述实测physical-artifact/GPU复制、跨seed/任务泛化及真实网页/学生效果仍未验；未当缺证自动判错。'},'work_record':{'path':str(W/'page-work.json'),'note':'独立逐页实际记录、必要前文、原始field-check与看图记录；非盲首读轨迹'}}
dump(W/'page-work.json',report['pages']);dump(W/'evidence-work.json',{'raw_evidence':evidence,'external_original':paper,'visual':visual})
report['work_record']['sha256']=sha(W/'page-work.json')
p=B/'reports/technical-architecture-initial.json';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(), 'initial already exists';dump(p,report)
seal={'schema_version':1,'reviewer':'/root/technical_architecture','role':'technical','group':'architecture','report':str(p),'report_sha256':sha(p),'sealed_at':now,'coverage':71,'source_only_initial':True,'criteria_sha256':report['criteria_sha256'],'work_record_sha256':sha(W/'page-work.json'),'evidence_work_sha256':sha(W/'evidence-work.json'),'independence':'封存前未读其他reviewer或历史结论；封存后此初判不可改，交叉结果另写。'}
sp=B/'reports/technical-architecture-initial.seal.json';assert not sp.exists();dump(sp,seal)
print(json.dumps({'report':str(p),'seal':str(sp),'sha256':seal['report_sha256'],'coverage':71,'issues':{'blocker':0,'burden':1,'optional':0},'figures640':len(visual),'report_bytes':p.stat().st_size},ensure_ascii=False))
