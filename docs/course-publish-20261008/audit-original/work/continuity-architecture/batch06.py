from write_notes import *
notes['actual_prerequisites'] += [{'page_id':p,'source_sha256':P[p]['source_sha256'],'scope':'main','reason':r} for p,r in [('7.3','提示在输入、有效回答与EOS目标、-100非输入'),('7.11','SFT同下一token目标、实际随机例仅求梯度'),('7.4','首答前一格预测与已shift不能再移'),('10.7','模态展开先扩展同位labels后shift一次')]]
add('chapter-18','教師是否可靠、訊號能否對齊、學生有沒有改善，是三個需要分別回答的問題。','通過导览：部署更小选架构与教师学习目标分开，可靠/对齐/改善都预告。','软目标温度KL以后逐页教，入口不需全部公式。','同任务品质与成本为目标，不能用更像/更小代任务完成。',['17.15','15.13'],'18.1先判新信号是什么。')
add('18.1','問題沒換，收到的學習目標卻可能不同。','强通過：原标签/同教师硬答/教师分布三路保同学生与原题，额外讯息不是真值。','两分布同第一4不同次选，toy候选非byte词表；黑盒文字student重新切，白盒整列逐回答位需对身份。','小学生原料已在仍可加教师指引，次选差别是新增学习目标，不虚构数值距离/必提升。',['4.6','7.11'],'18.4算目标改变梯度，18.6逐栏对齐。')
add('18.2','蒸餾改變學習目標，不會直接刪掉模型的參數。','强通過：student名与signal不缩参数，width/layers实际架构成本途径齐。','sameV264 teacher16x2/student8x1，字表线性不重层与矩阵平方分别，random只结构count。','部署小成本需先较小架构，教师希望弥补容量损失不保成功；从头或预训学生与向谁学为两独立选择。',['18.1','15.10','2.1'],'18.10same学生基准才能归教师訊号。')
add('18.3','使用同樣的答案誤差與樣本權重時，沒有新增監督訊號','强通過：黑盒SFT数据审核与原目标相同条件、教师文本新增整串而非分布连起。','人工4/5验证排5，JSONL概念不写文件不训练，assistant masks/EOS student tokenization；same labels/context/EOS才相同监督。','只能获取文字仍能SFT、教师错须先核；版本生成保存与成本为可追原料需要；同语义译文未必同token目标。',['7.3','7.11','18.1'],'18.4软比例与硬目标梯度不同。')
add('18.4','原答案與相同教師硬答案提出一樣的要求，分布則改變了方向和強度。','强通過：软目标额外候选关系实际改变更新要求，且不保教师偏好真或泛化。','CE硬/软不同目标不可loss排行，q-p手算硬[-.5,.4,.1]/soft[-.2,.2,0]，零logit梯度不等概率不变。','相同原题学生保住需要，非第一名候选关系可提供额外指导；条件若对新输入有用才可能帮，未另加数值距离承诺。',['18.1','18.3','first-steps W.6'],'18.5尖分布次选信号太小用温度。')
add('18.5','其他候選只剩少許份量，這些細小比例在軟目標誤差中的加權份量很小。','强通過：温度为何在蒸馏教与softmax比例变化直接连起。','same候选logits/T，指数比exp(3/T)保排名增次选，T2/T4条总100%；两方同T，helper已T²防重复。','次选在软CE权重小为独立need，平分可让它参与但过大抹差/错偏好也升明说；训练温度非生成温度。',['14.3','18.4'],'18.8交回同T分布KL与T²约定。')
add('18.6','此外雙方要在同一前文、同一待預測位置比較。','强通過：候选身份、词表顺序/粒度、前文/预测位/特殊token各条件齐。','猫狗反列数值误差.6→按身份order[1,0]为0；粒度分裂不可simple swap。','需同答案比例学错栏会教狗，shape一致非充分；EOS角色基本身份给，未要求任意词表映射完整算法。',['18.1','18.4'],'18.8 KL每列同语义，18.13不同模态前文按回答位对齐。')
add('18.7','優化器只接收學生參數，決定更新哪些數字。','强通過：固定teacher版本与eval/freeze/no_grad/update对象及snapshot身份齐。','两Linear人工随机只固定更新检查，clone before避免引用同变；probe输入可微教师仍图。','目标不随student更新移动为需要，no_grad省无用反向资料但teacher forward仍成本；不以none grad判教师能力。',['5.17','first-steps W.6','18.2'],'18.8 helper目标detach另核。')
add('18.8','這個KL函數只用labels決定哪些格有效，不按標準答案索引挑教師機率。','强通過：forward KL teacher权重方向、teacher forcing有效回答分母与helper契约齐；MiniLLM来源指向选读不需依赖。','KLp||q加teacher p，可负单项总非负；labels仅mask两有效梯度-.15/.15；teacher断目标、kl_div参数顺与公式不同、CE-H同student gradient代数。','同第一名未学整分布为独立need，teacher p当权重明确；同前文/有效位防不同问题，T²是已标尺度约定，不要求此处证明最佳超参。',['18.4','18.5','18.6','18.7','7.3'],'18.9与真CE混合，未看MiniLLM外部方法不对其额外验证。',['MiniLLM及Hinton原文未读取，不验历史/完整变体','helper未执行','站页runtime outputs未验'])
add('18.9','alpha越大，對這格真值的直接要求越弱，說明教師出錯時保留CE路徑的意義','强通過：混合目标的need、alpha端点、helperT²及labels双角色齐。','两位置真0/1教师皆偏0，CE .6931/K .1927不同目标值非质量；alpha0端点契约。','希望保真值又利用教师候选关系独立need，错教师第二格冲突实际呈现，不把alpha当可信自动量测。',['18.4','18.8'],'18.10独立任务比较不混目标loss。')
add('18.10','按學生更新預算比較能回答「多給教師訊號有沒有幫忙」；按同樣總時間比較則回答另一個問題','强通過：同学生/数据/起点/token与总成本两公平不同、部署非蒸馏额外缩齐。','人工10+40=50与3/4 vs4/4非实测，offline文本成本另列；teacher能力需验证非random例。','想归因额外指导及成本值得分别，原真值非teacher预测；同架构部署耗时需量，不用teacher-vs-small节省代学生比较。',['18.2','18.3','18.9'],'18.11继承错误/风格须任务规范。')
add('18.11','教師一致率是100%，但前兩題沒有完成；第三題則合理。','强通過：模仿一致与独立内容/风格/拒答判准不同，有反例而非泛好处。','三未知答案copy，真[2,4,None]人工acceptable两措辞仅此小判准，1 vs1/3；JSON格式过内容错例边界明说。','符合原任务且缺个人info应揭示不足独立need；过度拒绝不保可答题，风格长短依用户规约；不能泛套字符串判分。',['18.3','18.10'],'正式teacher和SFT共同条件比较，对失败有明确不夸宣称。',['09.2/09.6未另读取，但本页已自足说明有限任务规则；不验全部安全情景','style原始教师/学生结果未读取','未生成或执行'])
add('18.12','這個共同輸出介面讓分布蒸餾可行','强通過：MoE内部专家不搬入Dense，输出身份相同允许跨隐藏维度。','teacher16宽3expert top2/student8Dense，两输出1,2,264，labels只有效不完整chat；随机仅梯度接口。','同下一token分布为目标，内部不同仍可比较是独立结构安排理由；小学生可能不能拟全部与不保更快明说。',['15.1','15.4','15.10','18.6','18.8'],'18.13模态前文长不同也需同答界面。')
add('18.13','這是在配同一個待預測答案，不是配相同絕對索引。','强通過：展开不同长度同答案左移预测列精确，真实输入要求与视听观察范围齐。','teacher前16/student4，A0输入16/4预测15/3，A1/2往后各1；raw人工19/7抽3行，KL只答案。','整logits同绝对列意义不同为问题，按同答身份和前一格对齐解决；内视觉映射非直接文字KL，blank sameID只限批生成不概言内部无作用。',['7.4','10.7','18.6','18.8'],'18.14最终压缩需真数据行为再核。',['未读取12串图/音原始结果及blank control','人工代码未执行，完整多模态蒸馏未做','站页runtime outputs未验'])
add('18.14','只比較教師與最終量化學生，會看不出哪一步帶來品質變化。','强通過：四版比较隔离架构/学习信号/PTQ，payload params+buffers不漏及成本归因齐。','random三结构67840/24416/15680仅数字成本，real四版same distilled→packed blue变ble，scale/bias未全4bit。','部署省存+保任务目标，架构与格式可组合但蒸馏不再减同学生参数；weight-only无activation校准并明说不能改bits算完成。',['18.2','18.10','17.9','17.10','17.15'],'本章收回质量成本，后章方向不作范围外全文审。',['屬性四版原始实报未读取，数字/生成未核','未运行压缩或训练','站页runtime outputs未验'])
notes['visual_observations'].append({'figures':['p7-18-three-targets','p7-18-temperature','rewrite-18-vocab-alignment','rewrite-18-modal-answer-rows'],'size':640,'viewed_before_judgment':True,'judgment':'three-targets原题分三路非先后训练，同student架构和起始weights明确；温度三条等长固定色4/3/100、排名4仍最大；vocab图teacherID0猫配studentID1猫而非同ID，order一致；modal-answer三框A0/A1/A2各teacher/student预测列15/3,16/4,17/5，预测vs输入前一格底注。'})
for p in notes['pages']:
 if p['page_id']=='16.5': p['checks']['actual_dependency']=['16.3','3.7','本页自足重教mask交集，无依赖后页16.8']
 if p['page_id']=='17.4': p['checks']['mechanism']=p['checks']['mechanism'].replace('16码15间隔-.1?实际-1..2/.2','16码15间隔，-1..2/.2')
 p['checks']['mechanism_status']='已足夠'
 p['checks']['need_status']='需補說明' if p['page_id']=='14.10' else '已足夠（导览/纯性质按原目标判）'
notes['main_completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert set(p['page_id'] for p in notes['pages'])==set(json.loads((B/'manifest.json').read_text())['groups']['architecture']['pages'])
notes['main_route_scope']={'all_71_group_pages':'逐页完整主文已实际读，details隐藏','extra_prerequisites':'仅main实际读取，详actual_prerequisites；没有选读作为预设先备','previous_human_conclusions':'未读取reports/synthesis/notes/traces/checks人类结论或旧/同行review'}
save()
(B/'reports').mkdir(exist_ok=True)
(B/'reports/continuity-architecture-main-notes.json').write_bytes(N.read_bytes())
print('main notes sealed-ready',len(notes['pages']))
