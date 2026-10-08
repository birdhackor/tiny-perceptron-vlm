import json,pathlib,hashlib,re,datetime
B=pathlib.Path('/workspace/work/tutorial-audit-20261008');D=B/'work/continuity-modalities'
M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}
notes_path=B/'reports/continuity-modalities-main-notes.json'
main=json.loads(notes_path.read_text()); main_seal=json.loads((D/'main-notes-seal.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(notes_path)==main_seal['sha256']
extras_judgments={
'chapter-10':'没有折叠区；主线导览已在正文。',
'10.1':'回顾W.3的tensor/轴，正文已定义并给出本图RGB/索引/批次轴；未新增必需解释。',
'10.2':'回顾10.1的批次/RGB，顺序主线已读，未用选读补排序约定。',
'10.3':'回顾10.2形状与W.5矩阵；正文和实际补读2.3已有线性共享/矩阵关系。',
'10.4':'回顾10.3与文字4.1的位置；本页同内容不同格需求、广播与重排区分已正文完成。',
'10.5':'折叠另说明六类颜色+形状宽16真的训练，不等同本页宽8两形状梯度例；版本/任务区别属扩充查证，不用于主文通路判断。',
'10.6':'只回顾10.3和W.5特征轴；正文独立说明8→12合介面與資訊限。',
'10.7':'回顾10.6和7.4；实际所读7.4与本页同位labels/展开/shift规则均已可用。',
'10.8':'回顾10.7、4.6与后面的16.8；本页264词候选解释/同lm路径一致已给，后链接不作为当前必要先备。',
'10.9':'回顾W.5点积与10.5特征，并提供contrastive.json原始结果入口；方向相似比较步骤已经正文解释，无需外部报告才懂。',
'10.10':'复习相似表/CE/更新并提供contrastive结果链接；本页手设scores与假负例区别不依赖折叠。',
'10.11':'复习10.10和转置并链接contrastive；正文不同候选竞争和有限6/6边界已足够，链接用于核技术结果。',
'chapter-11':'没有折叠区；介面→正确答案的阶段需求正文已说。',
'11.1':'回顾10.6/10.5，无隐藏语义对齐必需解释；zero_动作时点正文已明确。',
'11.2':'回顾W.6/10.6/5.17；冻结、列优化器、历史grad和eval区别均正文或已实讀前文。',
'11.3':'回顾11.1/11.2/10.7与梯度，projector.json只原始成果入口；teacher forcing与完整0/6失败主文已给。',
'11.4':'回顾11.3描述与7.1对话，vqa结果链接；同图多问+同问换图、像素非名字、家族split不依选读。',
'11.5':'回顾11.2与4.5区块，vqa链接；当前partial/all含义與CLI区别主文已说，不需要由外部实现猜约定。',
'11.6':'回顾描述/问答/有效位置并给vqa来源；预算口径、阶段接头继承与直接版本文字保留未知已在正文。',
'11.7':'回顾旧能力遗忘/回放与介面一致，vqa来源；本页已重新说明任务对照与回放，未假定读选读懂回放。',
'11.8':'回顾答案依图/问题与敏感≠正确，vision_ablation来源；新图truth/原答案控制的区别正文完成。',
'11.9':'回顾坐标并给vision_ablation入口，必要裁切素材和45→8在正文图；查证链接不是缺素材。',
'11.10':'回顾块数/多位置注意力，vision_ablation来源；N×N格數與入口矩阵变化正文解释。',
'11.11':'回顾分布影响与vision_ablation来源；micro/macro及相同大小组不揭弱点正文足够。',
'11.12':'回顾像素/分类监督与ocr来源；熟悉字元新串、来源家族和RGB转换的必要限定均正文。',
'11.13':'回顾字形/循环并给ocr来源；等长zip非通用编辑距離和EOS非读对均主文已教。',
'11.14':'只回顾11.4及10.2；同格平均失左右与序列保差的工作已由本页图/文字完成。',
'11.15':'回顾裁切与OCR，另限定第20章成熟底座非本自训验收。当前机制无须熟悉第20章，属路线边界选读。',
'11.16':'只回顾11.13，核心Unicode与参考分母/次序约定已正文。',
'11.17':'补官方Fashion-MNIST 28×28、MIT授权及手绘非原图；正文已限定灰阶三类、手绘设计、无成效主张。未打开主线来源笔记或官方链接，不借它作判案。',
'11.18':'补Noto/OFL字型渲染与无打包字型，及11.15/16联结；主文可见字卡/选区/读字操作和目的已充分，字型许可不是当前推理必需。',
'chapter-12':'没有折叠区；单音与真人需求分界已正文导览。',
'12.1':'只回顾W.3一维tensor；波形/样本/周期/振幅意义正文可用。',
'12.2':'回顾12.1并给real_modal原始记录；rate时间/重取样基本关系不依赖文件格式详情。',
'12.3':'只回顾12.2换算，窗边界和拆分家族正文已说明。',
'12.4':'只回顾12.3；Hann选择的接点跳变、峰宽代价与时间不变正文已说明。',
'12.5':'回顾12.2/3/4；四点匹配两相、complex强弱与频谱坐标均未错放补充。',
'12.6':'回顾12.5和W.5矩阵；相邻重叠、mel分布密度与不可反演正文已足于本操作范围。',
'12.7':'回顾W.4和mel功率；log/floor/尺度与弱差取舍已正文，不需选读才解释负数。',
'12.8':'回顾12.7/10.3/W.3并给encoders结果；输入wave、内部logmel、频带时间转轴、保序与非单音中文能力正文已给。',
'12.9':'回顾12.8/10.7/11.3并给audio成果；梯度示范与11/14真正训练、共变限制主文分开。',
'12.10':'回顾音接口和替图类比并给audio记录；新答案/静音旧一致率与家族split已正文。',
'12.11':'回顾频率振幅/时间特征并给audio结果；ASR vs direct各能提供线索的边界主文成立。',
'12.12':'回顾图/声展开及替换，并给joint记录；交叉组合、两部分控制与先前预训频率限定已正文。',
'12.13':'回顾沿时间保特征与转写取舍；只倒特征而非倒波形的必要限定已正文，不需技术外部答案。',
'12.14':'两折叠分别说明手写转写/回覆无录音证据、成熟延伸非从零验收，及历史/分工回顾；正文已称手写与假设、指标None，扩展边界未用于补基本机制。',
'12.15':'補MInDS-14資料卡许可/502/no speakerID、原研究ASR预训及direct研究边界；正文不声称跨说话者或现有性能，设计图手写已可辨。本文没有取用/播放真实录音的限制归实测未验，不能把示例当收集证据。',
'12.16':'回顾ASR/direct与7.1，只把user一轮内容换特征，补无特定公开录音；正文已保持音特征和旁注转写分离。未打开来源笔记。',
'chapter-13':'没有折叠区；两可选路线正文导览已给。',
'13.1':'补64加法题49/8/7按交换加数家族split，预定两设置与前后最后题量测，不按结果改设；这些是实测设计补充，不用于本手写偏好条件。',
'13.2':'补自然UltraFeedback截120byte后label可能不适用，仅资料/通路支线；无主文自然聊天品质主张，不是必须移入同题两续写。',
'13.3':'补EOS完整sum、各250步共9448曝光非独立答案数；主文已明确sum/EOS/前文不计，细预算属重做证据。',
'13.4':'补弱content起点、复制ref/数值指纹与相对提升仍负的具体验证题；主文已有reference非truth与improvement≠胜出。',
'13.5':'补4+2候选log6>7却自由答8/EOS细数；主文已保该差异，未拿选读救概念。',
'13.6':'补真实两beta同起点/数据/种子/步数的验证/最后排序与自由生成0/7，弱底座且一seed非DPO无效；主文只局部饱和算例，未承诺效能结论。',
'13.7':'补正确短答对正确附加句的格式支线、相对gap改善可由reject降更多而chosen也降；属结果解释扩充，本节主线已有长度混淆和固定ref不保证消除。',
'13.8':'没有折叠区；温和准确与确认正确/喜好反例均正文。',
'13.9':'补未测风格/安全/多轮/反附和及自然支线仅排序，可靠标签截断限制；正文metrics已None，不改未知为通过。公开dpo报告未打开。',
'13.10':'补评员训练后固定、加数家族split与预写正确≠生成；正文卡片任务与人工标签已限定。',
'13.11':'没有折叠区；adv固定/critic另训与单步/多token边界均正文。',
'13.12':'补PPO与InstructGPT原论文公式/算法出处；旧策略按收集批更换和ref固定时间都已正文，未需论文才能懂ratio。',
'13.13':'补PPO原论文公式7/图1；min/clamp正负边界及梯度正文已教，无隐藏规则。',
'13.14':'补exact_kl接logits先softmax故手指定prob先log，另给约束系数/奖励与偏移取舍和DPO论文；主文只公式数值与roles，选读是具体复做接口和理论扩充。没有正文代码调用exact_kl而约定藏折叠的问题。',
'13.15':'第一折叠解释Categorical抽样/log_prob和gather取已选卡，完整一次收集/更新code及随机reward未教好；第二折叠细列家族split、随机网示范阶段与LLM SFT区别、label smoothing用途、rollout/重用/有效tokens0/CPU耗时。正文四步已定义数据动作与更新对象，完整代码为可选复做，不用于后文必需前提。',
'13.16':'补一句话带算式的有限判准/90对reward与policy18题不同、训练侧句题亦0；主文已讲未学句式非观察hacking，补充不是必要限定漏教。',
'13.17':'补DPO原论文与两个smallnet真实更新数/成本/14?家族细项；给same步≠same预算、没有同预算SFT的原因。主文已列成本和范围，理论论文不是路由比较必需。'
}
extras=[]
for pid in M['groups']['modalities']['pages']:
 raw=pathlib.Path(P[pid]['snapshot']).read_text()
 blocks=re.findall(r'<details\b[^>]*>.*?</details>',raw,re.S)
 quotes=[]
 for i,block in enumerate(blocks):
  body=re.sub(r'^.*?</summary>','',block,count=1,flags=re.S)
  q=next((line.strip() for line in body.splitlines() if line.strip() and line.strip()!='</details>'),'')
  if q: quotes.append({'fold':i+1,'quote':q,'line':raw[:raw.index(q)].count('\n')+1})
 extras.append({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'fold_count':len(blocks),'quoted_basis':quotes,'source_judgment':extras_judgments[pid],'main_route_dependence':'没有发现必要解释错放折叠；主文初判保持原封存内容。','unverified':'只读折叠正文/代码，未打开其外部论文、数据卡、原报告或作者来源笔记；未运行。'})
# Correct a private final-draft typo; sealed main notes are not touched.
for x in extras:
 if x['page_id']=='13.17': x['source_judgment']=x['source_judgment'].replace('/14?家族细项','/家族细项')
(D/'actual-extras-notes.json').write_text(json.dumps(extras,ensure_ascii=False,indent=2)+'\n')
log=[json.loads(s) for s in (D/'source-delivery.jsonl').read_text().splitlines()]
for phase in ['main','extras']:
 assert set(M['groups']['modalities']['pages'])<={x['page_id'] for x in log if x['phase']==phase}
prereq_quotes={
'7.1':'一筆對話是消息清單，每條用role記user或assistant，用content記原文字。',
'7.3':'忽略user的直接代價不會刪問題，也不會禁止後面注意力讀它。',
'7.4':'`render_chat` 已完成這個配對，TinyLM和loss按同一位置比較，不能再移一次。',
'7.5':'第一項回答「這個位置的候選分數有沒有直接答案代價」，第二項回答「這份輸入表示有沒有影響後面的答案」。',
'5.17':'固定權重仍是可微配方，輸入若需要梯度，輸出仍能對它求導。',
'3.6':'位置 i 只准讀位置 j≤i，即自己與更早的位置。',
'1.8':'`F.cross_entropy` 接收原始分數和答案 ID，內部已完成穩定的比例化與對數計算。',
'1.10':'`loss.backward()` 沿關係反向求導，把 24 存到 `w.grad`。',
'1.11':'它與參數形狀相同，每格敏感度都有自己的參數可對應。',
'1.12':'舊的 `before` 儲存的是原來算出的 4，不會因為 w 改了就自動變 2.56。',
'2.1':'看到「貓看貓」，就按 `[1,2,1]` 查三次。',
'2.3':'Linear 只沿最後一個特徵軸計算。',
'5.3':'本節的 v 則在更新後仍保留，讓過去方向繼續影響下一輪。',
'5.5':'若 `.grad=None`，該參數本次會被跳過；若有一份零梯度，優化器知道它參與了這一輪，仍可應用衰減。',
'4.3':'不是沿兩個位置一起算。',
'4.4':'共享配方不等於位置互相讀取：改變位置 1，只會透過此 FFN 改位置 1，跨位置交流由注意力做。',
'7.18':'這條人類回饋流程叫**人類回饋強化學習**（Reinforcement Learning from Human Feedback，RLHF），**近端策略最佳化**（Proximal Policy Optimization，PPO）是其中一種更新方法；兩名詞層次不同。'
}
for p in main['actual_prerequisites']:
 raw=pathlib.Path(P[p['page_id']]['snapshot']).read_text();q=prereq_quotes[p['page_id']];assert q in raw,(p['page_id'],q)
 p['quoted_basis']={'quote':q,'line':raw[:raw.index(q)].count('\n')+1}
# Strong passes cite the strongest supported scope; none claims technical or student validation.
pass_specs=[
(['10.1','10.2','10.3','10.6'],'像素、位置、特征宽与张数分开；切块保存资讯、压缩可丢、变宽非新增知识、接口相容非语义已学，转换用途与限均已教。'),
(['7.4','10.7','12.9'],'同位labels→模态展开→唯一一次下一目标配对→按实际长度位置/mask，接口动作和时间责任明确。'),
(['10.5','10.8','11.1','11.2','11.3'],'梯度通路、优化器名单、step真变、入口一致、差异敏感与完整生成不同证据，冻结核心仍可传梯度的先备已实读。'),
(['10.9','10.10','10.11'],'方向归一/行列候选/标签监督/双向不同竞争连续；选卡检索与逐字生成不混成能力证据。'),
(['11.4','11.8','11.9','11.11'],'同图多问/同问换图、替图新真值与遮图旧一致、分组/单总分、材料可见性控制各完成其用途，失败限定贴当前问题。'),
(['11.12','11.13','11.16','11.18'],'可见笔画→文字标签→整串/CER/EOS→次序→指定范围，变化只有一个主要新需求；固定裁切与自动定位分开，给反例和成对控制。'),
(['11.14','11.15','11.17','12.13'],'空间/时间平均只保总量可失顺序；原生任务问左右/先后才需要位置，完整序列可能保线索不等于自动学会关系。'),
(['12.1','12.2','12.3','12.4','12.5','12.6','12.7','12.8'],'时间与振幅先给意义，窗边界/加权/STFT二相与谱轴/重叠带/logfloor/转轴保持时间分别解释，固定工具与训练入口职责分开。'),
(['12.11','12.14','12.15','12.16'],'ASR字串与直接意图所留线索按任务区别，共用历史需系统保角色/内容/序且模型受跨轮示范；手写设计非真人录音成绩。'),
(['13.1','13.2','13.3','13.4','13.5','13.6'],'偏好依条件，各答案自前文与EOS求sum；固定参考非真值，relative改善非排序胜出，排序非自由答对，beta饱和局部性质非结局。'),
(['13.10','13.11','13.12','13.13','13.14','13.15'],'reward估已选卡、critic估选前均分，固定adv/old收集/当前ratio/ref起点分别时间；min裁切停有利鼓励，固定参考KL度量整体偏移，四步不依折叠完成机制理解。'),
(['13.9','13.13','13.12','13.14'],'固定参考KL的用途以已读短推论成立：共享权重可能影响旧任务，clipping非硬锁概率而仍要量实际偏移，old只本轮而ref固定起点；加偏移代价约束变动，不推出保证旧技能。'),
(['13.16','13.17'],'奖励模型评分/策略实际选卡/任务完整要求不同，PPO/DPO分叉并比较额外工作，同步非同预算，无hacking观察与有限失败不泛化。')]
source_quotes={p['page_id']:p['quoted_basis'] for p in main['pages']}
source_quotes.update({p['page_id']:[p['quoted_basis']] for p in main['actual_prerequisites']})
strong=[{'pages':ids,'judgment':why,'quoted_basis':[{'page_id':pid,**source_quotes[pid][-1]} for pid in ids]} for ids,why in pass_specs]
figs={k:v for pid in M['groups']['modalities']['pages'] for k,v in P[pid]['figures_sha256'].items()}
seen360=['rewrite-10-pixels','rewrite-10-patch-order','rewrite-10-image-targets','p7-12-frequency-match','rewrite-13-clip-cases','rewrite-13-model-roles']
seen_capture=['11.17-390-context-0.png','11.17-390-context-1.png','11.17-1280-context-0.png','12.5-390-context-0.png','12.5-390-context-1.png','12.5-1280-context-0.png','13.14-390-context-0.png','13.14-1280-context-0.png']
for p in main['pages']:
 p['supplement_review']=next(x for x in extras if x['page_id']==p['page_id'])
 p['main_note_origin']='封存main notes中的逐页判断；final只追加补充路线与范围，不改main notes原档。'
 p['issues']=[]
 p['strong_pass_refs']=[i+1 for i,st in enumerate(strong) if p['page_id'] in st['pages']]
unknown=[
 {'id':'U1','scope':'原始实验结果、训练/验证/最后题分组与完整配置','status':'unverified','why':'正文与折叠的任务、分母、限定已读；本角色没有打开原始结果JSON或执行纪录，不能独立核真数值。','reader_impact':'不阻断当下机制/控制用途，但实測真实性与控制設定尚未由本轮认证。'},
 {'id':'U2','scope':'程序执行和模型训练','status':'unverified','why':'仅手核正文中形状、目标位置、简单乘加/平均/符号；未运行程序、长训练或benchmark。','reader_impact':'不能将预期True/形状/数值说成现场执行结果。'},
 {'id':'U3','scope':'实际网页、交互与剩余360图','status':'unverified','why':'28张640全部看，6张360及3页桌机/手机局部新站captures已看；其余页面、折叠操作、链接跳转与runtime outputs/reading annotations未验。','reader_impact':'只认已见图中对象/箭头和3页局部可读；不宣称全站手机/桌面阅读路径验收。'},
 {'id':'U4','scope':'外部来源和理论主张','status':'unverified','why':'Fashion-MNIST/MInDS-14/Noto来源及许可、13.6 beta理论/DPO-PPO论文未打开；只读教材提供的界线。','reader_impact':'不能把设计素材来源/理论性质当已外部查证；不因此自动判正文缺陷。'},
 {'id':'U5','scope':'实际补读先备的图与其他未读前文','status':'unverified','why':'17个前文通过helper主文实读；其图和折叠未读，也未宣称验证全书介绍时序。','reader_impact':'本轮依可见表/文字/代码关系核支持，不认定所有入口背景和前文版面完整验收。'}]
report={'reviewer':'/root/continuity_modalities','role':'continuity','group':'modalities','review_type':'第三轮全文来源銜接审，不是第一轮无提示逐段增量读者审，亦非真人学生测试','saved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'criteria_sha256':main['criteria_sha256'],'criteria_files':{str(p.relative_to(B)):sha(p) for p in [B/'freeze/criteria/SKILL.md',B/'freeze/criteria/references/review-protocol.md',B/'freeze/criteria/references/calibration.md',B/'independent-review-instructions.md']},'independence':{'source_only_before_initial_seal':True,'old_or_peer_reports_read':False,'parent_messages':'仅收到派工、程序性提醒与自己进度回报；无具体已知问题/同行结论提示。','source_note_links':'教材fold出现作者来源笔记链接，仅读链接文本，未打开笔记或历史记录。','sealed_main_notes':str(notes_path),'sealed_main_notes_sha256':main_seal['sha256'],'notes_unchanged_after_extras':True},'actual_prerequisites':main['actual_prerequisites'],'pages':main['pages'],'issues':[],'unknowns':unknown,'strong_passes':strong,'coverage':{'assigned_pages':M['groups']['modalities']['pages'],'actually_read_main_pages':66,'actually_read_extras_pages':66,'actually_read_folded_blocks':62,'main_notes_saved_before_extras':True,'actual_prerequisite_pages':17,'receipt_is_not_automatic_pass':True,'quoted_basis_count':sum(len(p['quoted_basis']) for p in main['pages']),'page_specific_notes':True},'visual_scope':{'640_actually_viewed':[{ 'source':k,'source_sha256':v,'render':str(B/'renders'/f'{pathlib.Path(k).stem}-640.png'),'render_sha256':sha(B/'renders'/f'{pathlib.Path(k).stem}-640.png')} for k,v in figs.items()],'360_actually_viewed':[{'render':str(B/'renders'/f'{f}-360.png'),'render_sha256':sha(B/'renders'/f'{f}-360.png')} for f in seen360],'page_captures_actually_viewed':[{'path':str(B/'page-captures'/f),'sha256':sha(B/'page-captures'/f)} for f in seen_capture],'sequence_rule':'所有画面先由view_image取得，再在下一工具回合保存判断；未将预写描述当看图证据。','capture_limits':'既有新站预览省略runtime outputs/reading annotations，仅局部图文；无新浏览器交互测试。'},'execution_scope':{'tutorial_code_run':False,'training_run':False,'checked_by_hand':'主文形状/格数、RGB/分类目标、目标展开/移位、简单乘加/平均、梯度符号、预算和长度单位；未宣称所有数值独立核算。','source_hashes_checked':True,'all_quotes_verified_exact':True,'sealed_notes_hash_verified':True},'counts':{'blocker':0,'burden':0,'necessary_total':0,'optional':0,'unverified_issues':0,'unknown_scope_groups':5},'conclusion':'没有在实际已读主文、必要前文和fold分界中发现需修的銜接缺口；这一判断限于来源教学关系。原始成绩、程序行为、外部资料和全站版面仍有明确未验范围，未知不计通過，也不计缺陷。'}
# Verify every cited source quote and page hash, then save independent initial seal.
for page in report['pages']:
 raw=pathlib.Path(P[page['page_id']]['snapshot']).read_text()
 assert sha(pathlib.Path(P[page['page_id']]['snapshot']))==page['source_sha256']
 for q in page['quoted_basis']:assert q['quote'] in raw
for st in strong:
 for q in st['quoted_basis']:assert q['quote'] in pathlib.Path(P[q['page_id']]['snapshot']).read_text()
assert sha(notes_path)==main_seal['sha256']
out=B/'reports/continuity-modalities-initial.json';assert not out.exists()
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
seal=B/'reports/continuity-modalities-initial-seal.json';assert not seal.exists()
seal.write_text(json.dumps({'reviewer':report['reviewer'],'role':'continuity','group':'modalities','sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_path':str(out),'report_sha256':sha(out),'main_notes_path':str(notes_path),'main_notes_sha256':sha(notes_path),'source_delivery_sha256':sha(D/'source-delivery.jsonl'),'independence':'Before peer/old-review comparison; no later modification of this initial report.','counts':report['counts']},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'initial':str(out),'sha256':sha(out),'seal':str(seal),'main_notes_unchanged':sha(notes_path)==main_seal['sha256'],'counts':report['counts']},ensure_ascii=False))
