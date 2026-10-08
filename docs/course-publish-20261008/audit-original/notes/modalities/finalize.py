import json, hashlib, re, datetime, collections, copy
from pathlib import Path
r=Path('/workspace/work/tutorial-audit-20261008')
main_path=r/'reports/reader-modalities-main.json';main_bytes=main_path.read_bytes();main_hash=hashlib.sha256(main_bytes).hexdigest()
assert main_hash=='7e936ed456bf3f85631a0afcf1e776938a227d71e5c9542816150faf6cf458b9'
main=json.loads(main_bytes)
learned={
'10.5':'六类为红绿蓝×方块/圆，原入口width16实际更新；本节width8仅两形状梯度。',
'11.15':'第20章原底座未启用微调修正，能力不代自训任务；只读此界线未开第20章。',
'11.17':'Fashion-MNIST来源及MIT署名说明，作者图非数据集取图/模型输出；未打开来源或作者笔记。',
'11.18':'字卡/期望答案作者写，SVG Noto CJK TC有系统备援，OFL字体与生成文件授权情況分开；未读外链。',
'12.14':'转写/回覆手写流程示意，无录音或真实ASR输出，成熟底座延伸不作自训验收。',
'12.15':'中文502条、无公开说话者ID不能称跨说话者；资料原研究现成ASR/预训文字不作本课随机模型证据；作者未播放/转写录音，图问句答案设计例。',
'12.16':'此轮只换声音特征接7.1角色，未取特定录音；所指作者来源笔记未打开。',
'13.1':'真实加法0..7共64题按交换加数家族49/8/7，设定预定且先记起点/再测，不按检查分改参；与格式支线分开。',
'13.2':'UltraFeedback100笔作者家族80/10/10非官方测试，最多120 UTF8bytes保字符；截短可能删理由，原chosen不能保证仍适用，仅通路不证聊天。',
'13.3':'EOS纳sum、排prompt/padding，两支各250更新曝光9448targets非不同答案数，未临时改mean。',
'13.4':'固定content.pt原生成0/7非真值老师；验证3+5正确8-vs9差-10.89346→-8.44789相对改善2.44557仍错候选更高。',
'13.5':'4+2候选6log-24.88083胜7-28.11597，自由生8后EOS，两指标同真。',
'13.6':'同width64两层141568params/batch8/lr.001/250step，beta.1与1实际生成仍0/7；局部beta表不预言重训，单seed四留出家族失败不证DPO无效。',
'13.7':'附加句格式支线短含EOS2长19，原已7/7短排前，200步相对全改善但仍生0/7；正确短log可下降而长下降更多，差改善非真值信心。',
'13.9':'实际未测整套风格安全多轮反附和；加法原0/7保持非已会算术被保住。自然width32一层80SFT+80DPO相对/排序不同且未评自由生成，截短标签限制保留。',
'13.10':'reward先训家族偏好后固定，未见排序及训练loss另记；预卡有正确不等模型会写。',
'13.12':'PPO及InstructGPT原论文定位仅作为出处，未开外链；old按收集批、reference起始SFT。',
'13.13':'PPO公式7/图1出处定位，未读论文本身。',
'13.14':'exact_kl接logits先softmax，所以指定p先log；KL另系数加入policy代价，critic仍估rewardmodel分，有限配方非所有LM逐token。',
'13.15':'完整例Categorical logits→sample/action log，gather卡轴+squeeze得到每题reward；no_grad收集固定、更新重新算可微。报告seed42卡0ratio1、policy变、reward/ref无grad，未执行。真实55家族44/5/6各3条件132/15/18，148policy60示范步、241reward300步、97critic，label smoothing.2把onehot成.85/.05/.05/.05用于保探索不承诺输出。120×64=7680实采、每批重用3=360update/23040读纪录，effective_tokens0因非token任务。',
'13.16':'一句话只含算式结果；1+2 reward句13.5565>短8.7679而PPO短p.7928>句.2014，训练句也0/44，非仅泛化；未观察hacking。',
'13.17':'同148params起点、同44训练家族，PPO/DPO各360步但PPO另300reward/360critic/7680采，DPO23040偏好对；.66/.34秒与reward.27秒本机小网路非大LM性能；无同预算同资料SFT对照。'
}
extras=[]
for ch in [10,11,12,13]:
 source=r/f'freeze/original/course/chapters/{ch}.md';text=source.read_text()
 for section in re.split(r'(?=^## )',text,flags=re.M):
  heading=section.splitlines()[0]
  pid=f'chapter-{ch}' if not heading.startswith('## ') else re.match(r'## (\d+\.\d+)',heading).group(1)
  blocks=re.findall(r'<details[\s\S]*?</details>',section)
  extras.append({'page_id':pid,'folded_sections_read':len(blocks),'summaries':[re.search(r'<summary>([\s\S]*?)</summary>',b).group(1) for b in blocks], 'content_sha256':[hashlib.sha256(b.encode()).hexdigest() for b in blocks], 'new_understanding':learned.get(pid,'仅回顾/查证链接，无新的机制或判定依据。' if blocks else '本页无折叠区。'), 'source':str(source.relative_to(r)), 'read_scope':'所有details全文；正文此前已逐单元读完；未跟进链接', 'judgment_effect':'下方issues逐项另记，其他初判不变。'})
assert len(extras)==66 and sum(e['folded_sections_read'] for e in extras)==62
reassess={
'M-10.11-empirical-reference':'折叠直接定位contrastive实报，10.5折叠说明相关六类红绿蓝×形状。首读主机制已由非对称矩阵说明两方向竞争且明示有限候选边界；全题/划分与实验复现是进一步查证，不应要求扩写主要理由。保留为可选实验摘要改进，不称首读阻碍。',
'M-11.3-empirical-reference':'折叠提供projector实报定位；正文已明确另一个真正更新实验及red/重复e失败、标准前文loss与自由生成区别，足以支持本页边界。六题全表及底座详细条件是额外查证，不应把它们当梯度/生成理解前提。',
'M-11.5-experiment-scope':'折叠提供vqa实报定位；主线11.4已教同图换问题、原图家族划分，11.5给参数/有效目标/验证与最后题分数及不作普遍结论。较宽模型与逐题材料详情有助复现，但不阻碍成本效果比较原则，降为可选短摘要。',
'M-11.6-experiment-scope':'折叠提供vqa实报及描述/问答/有效位置回顾。主线已给两方案阶段、总预算差18、9/12vs10/12与未测原文字边界；完整材料与范围配置是复现细节，不能以此要求整段补课，降可选共用实验卡。'
}
final=copy.deepcopy(main)
final['mode']='independent main-first with sealed initial record and folded extras follow-up'
final['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
final['main_archive']={'path':str(main_path.relative_to(r)),'sha256':main_hash,'bytes_unchanged':True,'initial_distribution':main['summary']['page_distribution'],'initial_issue_count':len(main['issues'])}
for issue in final['issues']:
 issue['initial_severity']=issue['severity']
 if issue['id'] in reassess:
  issue['severity']='optional';issue['followup_assessment']=reassess[issue['id']];issue['scale']='可选局部摘要／定位，无需重写'
 else:
  issue['followup_assessment']='折叠只回顾/查证链接，无新增动作或独立用途解释；保留初判burden。'
for page in final['pages']:
 page['initial_assessment']=copy.deepcopy(page['assessment'])
 page['extras_followup']=next(e for e in extras if e['page_id']==page['page_id'])
 if page['page_id'] in ['10.11','11.3','11.5','11.6']:
  page['assessment']={'status':'optional','evidence':reassess[next(i['id'] for i in main['issues'] if i['page_id']==page['page_id'])], 'scale':'可选短摘要，无需正文重写'}
final['coverage']['extras_pages_checked']=66
final['coverage']['extras_folded_blocks_read']=62
final['coverage']['extras_route']='main封存之后，直接从冻结10–13章逐页提取并读完全部details；按授权无需next重走正文。reader extras标记可选阶段，后续实际范围由此ledger记录。'
final['coverage']['extras_link_targets_not_read']=['历史实验实报','实验README','作者来源笔记','第20章等其他未需前文','外部论文/资料卡/授权链接']
final['coverage']['no_execution']='未执行教材代码或重训；数字输出是正文/折叠报告内容及纸算，不声称复现实验或发布验收。'
final['visual_scope']['post_seal_recheck']=json.loads((r/'notes/modalities-visual-recheck.json').read_text())
final['visual_scope']['current_main_figure_count']=sum(v['scope']=='current main' for v in final['visual_scope']['figures'])
final['visual_scope']['prior_context_figure_count']=sum(v['scope']=='earlier context' for v in final['visual_scope']['figures'])
final['summary']={'page_distribution':dict(collections.Counter(p['assessment']['status'] for p in final['pages'])),'issue_distribution':dict(collections.Counter(i['severity'] for i in final['issues'])),'active_required_repairs':2,'optional_improvements':4,'blockers':0,'rewrite_scale':'问题不多：66页中60页直接通过，4页可选查证摘要，2页局部必要介绍负担。12.2补最小超界成分处理动作/理由，12.6补mel压频带及非均匀分配的独立声音表示用途即可；没有整小节重写或章节重排依据。','main_initial_judgments_preserved':True,'followup_limit':'四项降级兼含对标准的重新收敛，不冒称可选区补全了所有实测细节；详细实报仍未读。视觉初判时序限制公开，封存后八图重看均无教材判定变化。'}
(r/'notes/modalities-extras.json').write_text(json.dumps({'reviewer':'/root/read_modalities','main_archive_sha256':main_hash,'pages':extras,'reassessments':reassess},ensure_ascii=False,indent=2)+'\n')
out=r/'reports/reader-modalities.json';out.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
assert hashlib.sha256(main_path.read_bytes()).hexdigest()==main_hash
print(json.dumps({'final':str(out),'pages':len(final['pages']),'page_distribution':final['summary']['page_distribution'],'issue_distribution':final['summary']['issue_distribution'],'visual_current':final['visual_scope']['current_main_figure_count'],'visual_prior':final['visual_scope']['prior_context_figure_count'],'main_unchanged':True},ensure_ascii=False))
