from pathlib import Path
import json,hashlib,re
from datetime import datetime,timezone
B=Path('/workspace/work/tutorial-audit-20261008');M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}
mp=B/'reports/continuity-foundations-main-notes.json';N=json.loads(mp.read_text());seal=json.loads((B/'work/continuity-foundations/main-notes-seal.json').read_text())
assert hashlib.sha256(mp.read_bytes()).hexdigest()==seal['sha256']
S={}
def E(p,q,result):
 raw=Path(P[p]['snapshot']).read_text();assert q in raw or q.replace('**','') in raw.replace('**',''),(p,q)
 S[p]={'route':'extras after immutable main notes','quoted_basis':q,'added_understanding':result,'main_required_explanation_misplaced':False,'judgment':'可留選讀；未以此回寫主文初判。'}
E('course','阅读不需要先下載所有資料或權重。'.replace('阅读','閱讀'),'授权/原始实验入口不替代正文教法；当前阅读与安装门槛不因折叠提高。')
E('first-steps','本節只用 Python 內建寫法。','W2怎样到Notebook执行已在W1主文有可操作步骤；折叠重做指引非新的必要机制。')
E('1.1','本例只處理字表已有的字。','明确缺鸟的KeyError及6章扩字；当前编码源句全部在表内，主线没有未教字需要此补充。')
E('1.2','不是所有來源洩漏的自動證明。','代码断言仅train/val完全重复，原主文已要求逐句核与按来源分，补充防误解不挽救缺少定义。')
E('1.3','兩項檢查各有用途，不能互相代替。','shift与因果先给后教；当前一次当前字题不依赖整段许可，3.6/4.7主文各自教足。')
E('1.4','電腦儲存小數有微小誤差','abs和1e-6容忍阈值/小数误差是代码重做细节，主文已说总和接近1，不依赖容忍值选择作推理。')
E('1.5','讓一欄分母配到整列的運算叫廣播。','广播名称与N+V总式为延伸；主文已经每行同分母、具体9与练习2.7，机制不在折叠。')
E('1.6','它們都不表示已訓練。','Parameter/grad_fn为原始输出注记，不是更新；主文手动偏好/未更新已明确。')
E('1.7','只把同一組分數換成比例。','allclose/assert显示与数值误差复习，不额外答案/训练动作。')
E('1.8','比較平均代價需固定資料及文字拆分方式','nat/bit与未来token切分边界无当下比较需要；当前同候选固定例在主文完整。')
E('1.9','h 太大可能不能代表附近','差分h太大小限制属深入数值核对；主文已有局部范围，小h=.001目的与结果可理解。')
E('1.10','各路用鏈式法則求出影響後還要相加','当前单路机制足够，多路4.2主文自己给1+2和小变化，折叠不是以后必需独读先备。')
E('1.11','中間結果的 `.grad` 不一定自動儲存','叶节点与中间grad储存为API扩展，当前只从参数w读grad，不需该区别才能懂操作。')
E('1.12','本節只做一次求導，沒有清梯度的問題','no_grad去掉会报错补技术限定；主文已解释不记更新运算并实际包住，后1.13主文教清累计。实测来源T3仍未读取，不宣称数字核实。')
E('1.13','本例每次重新計算 w²','计算图释放、刻意累计小批属于进阶；主文每次新forward/清None并已有当前需求，不需要retained graph理论。')
E('1.14','不保證跨版本完全一樣','seed跨环境限定与range/_名称补充；当前主文同环境重做/固定12次已有，不需要外部原始日志解释生成。')
E('1.15','T 必須大於 0','零/并列argmax例外为扩展，主文正数温度与唯一最高例足以当前操作。')
E('2.1','D 不必等於候選數 V','VD参数数和后层不在代码提示；主文五行三特征及手算额外配方已清楚，未靠选读成立。')
E('2.2','本節沒有求導或更新。','离no_grad后保持追踪与ViewBackward注记为重做帮助；主文仅查/拼及同格变形，非新信息/学习。')
E('2.3','沒有學習或能力評測','参数可由loss更新沿1.11/12，主文手填已界定；无额外更优结果。')
E('2.4','本節沒有 `backward()`、更新或泛化評測','仅追踪图输出不证明学习；主文已明确手填算得出，折叠可留。')
E('2.5','橫軸是視窗，不是訓練時間。','原window_training曲线额外连结同主文表，种子/宽16/T3重做可选。主文已承认参数变、仅一验证文与非所有任务最佳；曲线未实看、T3外部未读，保持未验证。')
E('3.1','一般 Linear 權重可負、也不必加總為 1','这里概率式比例限制主文已给，Linear前2.3已用负权重，不需要折叠才区分。')
E('3.2','沒有取回內容，也先不限制讀哪些位置','逐步QK匹配子任务边界，主文下一节/尺度预告与后3.4–6解释足够，当前数字手填非训练序列不需先mask。')
E('3.3','不同查詢之間不一起做 softmax','Tq/Tk/Dk泛化轴式为补充；主文已经每查询一行、练习沿候选栏，各题不抢比例1.7已教。')
E('3.4','完整縮放規則在下一節。','Tq/Tk/Dv一般式与未缩放边界，主文两位置/三格output例即核心关系，3.5主文解尺度。')
E('3.5','D 要用每頭 Q/K 的寬度','每头D具体推论：3.5定义D是匹配向量格，3.7每头四格Q/K，所以使用4而非总8可正常推得，不需迁入主文才能懂；温度别在生成/匹配也已分别有作用。')
E('3.6','`clone()` 避免改動原 V','副本概念1.13主文clone已有；手填读取范围非学参数，当前clone changed/out两次比较有明确输入/输出。')
E('3.7','本節程式使用另一份隨機輸入','封装含许可/缩放与手算另数据；主文明确改用四位置随机特征，只核尺寸，所以重做段没有补当前核心缺口。')
E('4.1','本例輸入位置 5 會超出表格範圍','两表同宽/超界/AddBackward注记；主文位置5列有效0..4且两边3格相加已有，不需选读补对应。')
E('4.2','兩格屬於同一位置的特徵','I+df/dx一般矩阵式无必要，主文一位置两格与1+2已明确；图同一位置标题亦对应。')
E('4.3','`unbiased=False` 用三格平方偏離的平均','API方差分母、完整j式及共享γ/β是补细节；主文明确每位置三格平方偏离平均/先整理再可调与不同轴的具体作用，选择跨轴不是折叠才知道。')
E('4.4','本節關心非線性放在擴張與縮回之間，以及位置數不變。','明确GELU完整公式不需，当前定性非线性角色足够；仍未解释为何特定封装改用GELU，保留原optional（非必要理由缺口）。')
E('4.5','本節只用 pre-norm。','post-norm替代顺序为延伸；保直接路/分支先整理的当下设计可由4.2/3/5正文短推得，不要求所有变体比较。')
E('4.6','沒有共享權重','本例输入/输出分表、字表超界与假设ID例限制；当前主文已有候选0..19及无真实字义，未拿权重共享作推理。')
E('4.7','短機製程式與這個訓練實報是不同層次的證據。','T4原始训练/因果0数值可作为额外证据，主文三题求导机制与3.6遮罩已足。原始JSON未读，保留数字未验证；选读实测不强迫变必读。')
E('4.8','合計 840。','参数明细可加总为256+512+40+32=840，入口160+1024+16+160=1360；此为正文计数的补推导，主文每层独立/入口共用及公式已能说明非三倍；另模型141568与训练T4未核实，非深度优越性证明。')
for p in N['pages']:
 p['supplement_judgment']=S.get(p['page_id'],{'route':'extras','judgment':'本页无折叠区，已实际由helper取得确认。','main_required_explanation_misplaced':False})
 p['quoted_basis']=p['quoted_basis']+[{'page':p['page_id'],'quote':S[p['page_id']]['quoted_basis'],'route':'extras'}] if p['page_id'] in S else p['quoted_basis']
N['issues']=N.pop('main_issues')
N['main_notes_seal']=seal
N['supplements_reviewed_at']=datetime.now(timezone.utc).isoformat()
N['coverage']={'required_pages':43,'actually_read_main_pages':43,'actually_read_extras_delivery_pages':43,'page_ids':M['groups']['foundations']['pages'],'all_page_evidence_present':True,'method':'实际helper正文输出后逐页依赖/机制/用途判断；全组notes写定并helper字节seal后才折叠补读。首批输出first-steps中段截断后单页完整重取。','not_claimed':'并非无提示逐段首读，不是人类学生实验，不靠source delivery本身当pass。'}
V=json.loads((B/'work/continuity-foundations/visual-notes.json').read_text())
N['visual_scope']={'actually_acquired_and_next_turn_judged':V,'figures_640_count':16,'figures_360_count':8,'actual_page_capture_count':10,'page_capture_scope':'first-steps W1 desktop1280×800/mobile390×844;1.3 context同两宽;3.4 context同两宽;2.1/3.7/4.2/4.7手机390图元素crop。','unverified':'其他完整页面的图文位置/折叠开关/目录切换/捲動/运行输出未验；2.5折叠链接式window_training.svg曲线未取得图片，不宣称验证它。'}
N['execution_scope']={'tutorial_code_executed':False,'long_training_run':False,'source_mutation':False,'own_record_generation_only':True,'implementation_read':False,'external_evidence_read':False,'arithmetic_scope':'按正文短算关系核对：梯度平方/链乘；ReLU绝对值；QKV对应；1,3,5视窗；CD→HCD；LN分母；折叠参数840/1360直接求和。没有重跑模型或已实测训练。'}
N['summary']={'blocker':0,'necessary_burden':1,'optional':1,'paragraph':2,'page':0,'chapter':0,'order':0,'unverified_substantive_pages':['course','1.12','1.14','2.5','4.6','4.7','4.8'],'scope_conclusion':'本组未发现需要整页/整章/调整章节顺序的正文机制銜接缺口；已见必要项是W1手机浮动控件遮字，另GELU选择说明为可选一句。实测/全页界面未验证不算内容通过。'}
N['independence']={'sources_seen':'本组冻结sources、criteria、manifest元数据、指定renders/page-captures、自己工作笔记','forbidden_materials_seen':[],'no_peer_or_historical_conclusions_before_seal':True,'reviewer_authorship':'没有参与教材作者工作；AI辅助全页銜接角色。'}
N['created_at']=datetime.now(timezone.utc).isoformat()
out=B/'reports/continuity-foundations-initial.json';out.write_text(json.dumps(N,ensure_ascii=False,indent=2)+'\n')
sha=hashlib.sha256(out.read_bytes()).hexdigest()
s={'reviewer':'/root/continuity_foundations','role':'continuity','group':'foundations','sealed_at':datetime.now(timezone.utc).isoformat(),'initial_report_path':str(out),'initial_report_sha256':sha,'main_notes_sha256':seal['sha256'],'coverage_pages':43,'independence':'封存前未读取历史/同行/reports/synthesis/notes/traces或human checks结论；只读自己的必要记录以生成report。'}
sp=B/'reports/continuity-foundations-initial-seal.json';sp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':sha,'seal':str(sp),'summary':N['summary']},ensure_ascii=False,indent=2))
