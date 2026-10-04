"""Assemble only this reviewer's own 13.17 report from actual evidence."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent
REL=ART.relative_to(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((ART/'audit-result.json').read_text())
environment={k:str(v) for k,v in audit['environment'].items()}
prerequisites=json.loads((ART/'prerequisite-receipt.json').read_text())
retrievals=json.loads((ART/'authority-retrieval-receipts.json').read_text())
figures=json.loads((ART/'figure-render-receipt.json').read_text())

artifacts=[]
def artifact(id,kind,name,description,command=None,result=None):
    a={'id':id,'kind':kind,'path':str(REL/name),'sha256':sha(ART/name),'description':description}
    if kind=='execution':a.update(command=command,result=result,environment=environment)
    artifacts.append(a)
native_cmd='.venv/bin/python -m scripts.course_experiments.posttraining --output outputs/natural-v4/factual-research/13.17/cpu-run > docs/technical-reviews/artifacts/natural-v4-factual/13.17/cpu-run.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/13.17/cpu-run.stderr.txt'
audit_cmd='.venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/13.17/audit_cpu.py > docs/technical-reviews/artifacts/natural-v4-factual/13.17/audit.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/13.17/audit.stderr.txt'
artifact('a-section','source_snapshot','original-section.md','本人完整读取的当前13.17未正规化UTF8，包括所有空行；首份源文证据。')
artifact('a-prereq','source_snapshot','prerequisite-receipt.json','本人完整读取必要前置各节的原始字节SHA/byte counts；对应全文保存在同目录prerequisite-<ID>.md。')
artifact('a-retrieval','source_snapshot','authority-retrieval-receipts.json','本人实际下载原论文与官方源代码的URL/version/date/retrieval SHA收据；完整第三方文件仅在ignored research。')
artifact('a-authority','source_snapshot','authority-and-scope.md','本人原始权威来源阅读定位、条件与scope分析，含CPU小实验和成品分支界限。')
artifact('a-code','code','audit_cpu.py','本人独立解析计算、exact lesson/exercise execution、原始概率argmax/规则/家族/预算/重载与dispatch核对代码。')
artifact('a-native','execution','cpu-run.stdout.txt','实际原生小CPU训练stdout，固定配置，完成所有更新并得到6/18、12/18、12/18。',native_cmd,'Exit 0; native tiny CPU run completed; no GPU or token language-model training.')
artifact('a-native-stderr','execution','cpu-run.stderr.txt','对应真实原生CPU命令stderr。',native_cmd,'Exit 0; stderr empty.')
artifact('a-audit','execution','audit-result.json','实际独立核对结果，包括完整配置、版本、54项测试概率摘要、原始record SHA、数据分母、各checkpoint SHA与浮点差界限。',audit_cmd,'Exit 0; all numerical/split/draw/own-state/reload/original argmax assertions passed; maximum historical-vs-rerun probability error 9.5367431640625e-07; discrete outcomes exact.')
artifact('a-audit-stdout','execution','audit.stdout.txt','exact current lesson/exercise及CPU解析复算的真实stdout。',audit_cmd,'Exit 0; printed losses0.6931/0.3567/1.204 and verified structured results.')
artifact('a-audit-stderr','execution','audit.stderr.txt','最终独立核对命令真实stderr。',audit_cmd,'Exit 0; stderr empty.')
artifact('a-first-code','code','first-attempt-audit_cpu.py','保留本人第一尝试代码：跨运行权重byte相等断言过强，后来改为真实命题的同次运行一致性及数值误差比较。')
artifact('a-first-stderr','source_snapshot','first-attempt-audit.stderr.txt','保留真实第一次AssertionError；这是审阅者额外断言失败，教材没有承诺跨运行逐字节重现。')
artifact('a-figure-notes','source_snapshot','figure-inspection.md','本人render+view_image后的图文方向、位置、数值与限定范围核对。')
artifact('a-render-receipt','source_snapshot','figure-render-receipt.json','安装的Inkscape实际渲染命令、exit/stdout/stderr和当前SVG SHA。')
for i,f in enumerate(figures):artifact('a-figure-'+str(i+1),'figure_render',Path(f['render']).name,'本人实际渲染并view_image '+f['source']+'；图核对说明见a-figure-notes。')
closure_cmd='.venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/13.17/closure_cpu.py > docs/technical-reviews/artifacts/natural-v4-factual/13.17/closure.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/13.17/closure.stderr.txt'
artifact('a-closure-code','code','closure_cpu.py','本人同一owner重读必要前置13.15/13.16/7.17后执行的复核代码，保留初版。')
artifact('a-closure','execution','closure-result.json','实际current13.15完整代码块执行、旧新代码块相等、必要前置原current SHA/受影响主张及真实术语scope复核。',closure_cmd,'Exit0; action0, ratio1, policy changesTrue, reward/reference gradientsFalse, shapes[1,4]; inspected experiment code/config/results unchanged; no new unresolved13.17 facts.')
artifact('a-closure-stdout','execution','closure.stdout.txt','真实同一owner复核执行stdout。',closure_cmd,'Exit0; current13.15 output and closure provenance printed.')
artifact('a-closure-stderr','execution','closure.stderr.txt','真实同一owner复核执行stderr。',closure_cmd,'Exit0; stderr empty.')
artifact('a-current-prereq','source_snapshot','closure-prerequisite-receipt.json','保存原版不覆盖，新增本人真重读后的current必要前置section SHA和新增7.17依赖。')
for ID in ['13.15','13.16','7.17']:
    artifact('a-current-'+ID,'source_snapshot','closure-current-prerequisite-'+ID+'.md','本人完整重读的current必要前置'+ID+' raw UTF8。')

sources=[]
def external(id,kind,title,url,version,reason,note,index):
    sources.append({'id':id,'kind':kind,'title':title,'url':url,'version':version,
      'verified':True,'checked_original':True,'accessed_on':'2026-10-04','authority_reason':reason,
      'inspection_note':note,'retrieval_sha256':retrievals[index]['retrieval_sha256']})
external('s-dpo','paper','Direct Preference Optimization: Your Language Model Is Secretly a Reward Model',
 'https://arxiv.org/pdf/2305.18290v3','arXiv:2305.18290v3 (29 July 2024)',
 '原方法作者公开的指定v3原论文；直接给出DPO变换与其条件。',
 '本人实际读§3 Eq.1–3、§4 Eq.4–7及梯度/DPO outline、Appendix A.1 pp.15–16、§7 Limitations p.10。固定参考的KL奖赏目标经Bradley–Terry模型重参数化得到Eq.7；没有另一独立reward network。β>0与可用log ratios/参考支持是条件。理想非参数最优式不保证任意有限PPO实现得到同样权重。全文只下载到ignored research，本人摘要与检索SHA见a-authority/a-retrieval。',0)
external('s-instruct','paper','Training language models to follow instructions with human feedback',
 'https://arxiv.org/pdf/2203.02155v1','arXiv:2203.02155v1 (4 March 2022)',
 'Ouyang等原始InstructGPT论文，其真实human-labeler方法定义此处典型RLHF流程。',
 '本人实际读§3.1三步骤：真人示范SFT；真人比较教RM；PPO对RM reward更新策略。支持“真人比较时可构成典型RLHF”的条件描述，不能支持本项目程式规则标签已是真人实验。',1)
external('s-ppo','paper','Proximal Policy Optimization Algorithms',
 'https://arxiv.org/pdf/1707.06347v2','arXiv:1707.06347v2 (28 August 2017)',
 'Schulman等原PPO算法与Actor-Critic更新顺序。',
 '本人实际读§3 Eq.6–7、§5 Eq.9与Algorithm1：新/old比率、min裁切目标、value误差项、先取样后K epochs再更新old。只有有限contextual bandit机制用于本节，未把它外推成token LLM PPO。',2)
external('s-torch','official_source','PyTorch functional.logsigmoid original source at runtime-reported commit',
 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py',
 'commit 5c4886908584029761b579af026dcfb627c84070; actual runtime2.14.1+cpu',
 'PyTorch官方仓库精确commit源码，对应运行时报告git_version；不使用可变stable页面冒充指定release。',
 '本人下载并读functional.py lines2049–2058，logsigmoid绑定torch._C._nn.log_sigmoid并定义log(1/(1+exp(-x)))；本地import API docstring一致。实际float32值由本人CPU执行验证，而非只读网页。',3)
def repo(id,path,title,note):
    p=ROOT/path;sources.append({'id':id,'kind':'repository_code','title':title,'path':path,'sha256':sha(p),
        'version':'current full-file SHA256 '+sha(p),'verified':True,'inspection_note':note})
repo('s-alignment','tiny_perceptron/alignment.py','本项目DPO与序列log概率实现',
 '本人实际读全文件；dpo_loss lines38–42拒绝非正β，扣固定参考差的detach，再-logsigmoid取batch mean。sequence_log_probability是有效位置sum；CPU代码真实调用并验证参考无梯度。')
repo('s-models','tiny_perceptron/posttraining.py','有限策略、RM、critic与PPO helper',
 '本人实际读全文件；FiniteResponsePolicy 4→16→4，RM(4+4)→24→1，value4→16→1；context而非中文token；exact_kl和固定优势/old log detach与clipped objective。本人实际实例化/训练/重载并复算参数。')
repo('s-driver','scripts/course_experiments/posttraining.py','正式有限CPU posttraining原始训练驱动',
 '本人实际读全文件并运行原生module到ignored research。CONFIG、build_records/split_records、SFT/RM/PPO/DPO loops、_evaluate、timing boundaries、same-start deepcopy与reference before/after SHA均已核对。原正式记录的三源代码SHA与当前全文件SHA一致。')
repo('s-common','scripts/course_experiments/common.py','CPU experiment Context/seed/JSON helper',
 '本人实际读全文件；此次只调用Context、seed与write_json，不运行其他LM/数据/GPU helper。')
repo('s-seed','tiny_perceptron/training.py','本项目seed_everything helper',
 '本人实际读lines23–27的seed_everything：random.seed与torch.manual_seed，只有CUDA可用才设CUDA seeds；当前CUDA unavailable。全文件SHA用于依赖一致性，不声称核对本文件每个无关功能。')
repo('s-record','docs/course-experiments/results/posttraining.json','既有原始正式CPU逐题/配置记录',
 '本人直接读取原record config/splits/code SHA/state identities/update counters/timing/evaluation rows。独立从actual probabilities取argmax，按mode规则重计三split指标，重建family/pair分母。只支持seed42固定配方；原byte指纹和本人新run不相同，不冒称bitwise historical replication。')
repo('s-capstone-driver','scripts/course_experiments/capstone.py','成品分支父checkpoint与DPO配置',
 '本人读lines1–315，特查EXPERIMENTS、train_stage父阶段检查、DPO目标、_run_context与run_preference，并读部署函数前段312–348。实际运行_run_context(dispatch only)，拦截train_stage记录传入capstone_joint/model.pt；未加载/训练成品或运行GPU。')
repo('s-capstone','tiny_perceptron/capstone.py','成品模型stage与sequence偏好接口',
 '本人读lines1–28与330–380：STAGES pretrain/sft/joint/dpo；preference_pairs、sequence preference_loss与frozen_reference。支持代码分支隔离与固定参考机制，未认证本节无关全部成品能力。')
repo('s-context13','course/chapters/13.md','必要同章前置的当前整文件版本绑定',
 '本人完整读取13.3–13.6、13.10、13.12、13.14–13.16与本节13.17的原始section bytes；原版独立SHA/全文保存在a-prereq。随后本人完整重读修改后的13.15/13.16并实际执行current13.15代码块；新的原current SHA分别保存在a-current-prereq/a-closure而没有覆盖初版。该文件只给上下文/必要依赖，不代替DPO原论文权威。')
repo('s-context07','course/chapters/07.md','必要后训练定义7.17–7.18当前版本绑定',
 '本人完整读7.18；closure后完整读新增必要7.17，其pretrained→SFT说明与已实际读取的InstructGPT v1 §3.1相合，图posttrain_stages已实际render/view。此教材原文给上下文版本，不作概念独立权威。')
sources.append({'id':'s-derivation','kind':'derivation','title':'两卡DPO及相对增益解析推导','verified':True,
 'details':'令p_chosen=p、p_rejected=1-p、reference各0.5、β=1。m=ln(p/(1-p))，sigmoid(m)=p，L=-ln p。p=.5→m0,L.6931471805599453；p=.7→m.8472978603872034,L.35667494393873245；p=.3→m-.8472978603872036,L1.2039728043259361。float32实算分别四位小数.6931/.3567/1.2040，abs tolerance2e-7。梯度dL/dchosen=-βsigmoid(-βm)，dL/drejected相反；m0,β.1→-.05/+.05。相对偏好不是绝对单调：参考[.2,.1,.7]→策略[.15,.05,.8]，前两项都下降但m=ln(3/2)>0，loss降为-ln(.6)=.5108256。预算360×64=23040，120×64=7680，120×3=360，300×64=19200。全部CPU复算保存a-audit。'})
sources.append({'id':'s-execution','kind':'execution','title':'本人真实CPU解析/记录/重载/dispatch核对','verified':True,'artifact_id':'a-audit'})

claims=[]
def ev(source,locator,supports):return {'source_id':source,'locator':locator,'supports':supports}
def claim(id,kind,statement,location,evidence,artifact_ids,scope,verification=None,denominators=None):
    c={'id':id,'kind':kind,'statement':statement,'location':location,'status':'verified','evidence':evidence,
       'artifact_ids':artifact_ids,'scope':scope}
    if verification:c['verification']=verification
    if denominators:c['denominators']=denominators
    claims.append(c)
def calc(expected,observed,details,tolerance='absolute error <=2e-7 for float32; rounded4 exact'):
    return {'method':'executed','expected':expected,'observed':observed,'tolerance':tolerance,'details':details}
claim('c1','concept','典型PPO偏好路线先教奖赏模型，再以策略作答获得奖赏并更新。','开头第一段PPO路线与倒数第二段RLHF条件',
 [ev('s-instruct','§3.1 Steps2–3','human comparison→RM→PPO；是典型路线而非所有PPO必须有RM'),ev('s-dpo','§3 Eq.2–3','先拟合比较奖励再优化带参考约束策略')],['a-authority'],'仅偏好学习/PPO-based RLHF路径；PPO本身也可用其他来源奖励。')
claim('c2','concept','DPO可直接从同题chosen/rejected偏好对更新策略，省去额外独立奖赏模型训练，仍需偏好标准。','第一段DPO路线、两卡算例末句',
 [ev('s-dpo','§4 Eq.5–7 and DPO outline','重参数化后直接策略loss与offline comparison labels')],['a-authority'],'原始DPO路线仍使用固定参考与可靠比较数据；“无评分网路”不等于没有监督标准。')
claim('c3','concept','DPO的margin是策略较佳-较差log机率差减固定参考同一差，loss为-log sigmoid(beta×margin)，beta正。','第二段公式',
 [ev('s-dpo','§4 Eq.7','两项log(policy/reference)相减恰给所述margin与β缩放'),ev('s-alignment','dpo_loss lines38–42','同一表达式与正β条件')],['a-authority','a-audit'],'需正β与有限可用log分数；正文两卡是归一化候选分布，正式程序batch mean。')
claim('c4','concept','sigmoid将βmargin变为0至1的模型偏好机率，负log形成比较代价。','第二段sigmoid说明',
 [ev('s-dpo','§3 Eq.1–2; §4 Eq.6–7','Bradley–Terry建模偏好概率，不是任务正确率'),ev('s-torch','functional.py lines2049–2058','logistic的精确定义')],['a-authority'],'有限实数的sigmoid严格在(0,1)；偏好概率是该模型下的比较量，并非校准真实正确率。')
claim('c5','concept','减参考衡量相对起点的偏好变化，不强迫每篇回答绝对机率只升不降。','第二段“不是强迫每篇回答只增加不减少”',
 [ev('s-dpo','§4 Eq.7 and gradient after Eq.7','目标取相对log比差、通过共享参数更新'),ev('s-derivation','三卡归一化counterexample','chosen/rejected绝对值都降而相对比值提高，loss仍下降')],['a-audit'],'局部偏好方向和绝对序列机率单调不同；反例只说明目标没有该强制条件。')
for id,p,where in [('c6',0.5,'透明例子：相同策略与参考'),('c7',0.7,'透明例子：策略0.7与0.3'),('c8',0.3,'末段练习：策略0.3与0.7')]:
    row=next(x for x in audit['toy_cases'] if x['chosen']==p)
    claim(id,'numeric',f'两卡chosen机率{p}、reference各0.5、beta1时loss约{row["analytic_loss"]:.4f}。',where,
      [ev('s-derivation','m=ln(p/(1-p)), sigmoid(m)=p, L=-ln p','逐项独立代入与解析值'),ev('s-torch','logsigmoid definition','实际使用API函数定义')],
      ['a-code','a-audit','a-audit-stdout'], '只支持两卡β1算例；正式β.1不能混用此loss数。',
      calc(str(row['analytic_loss']),str(row['observed_float32_loss']),'实际执行当前正文代码块与练习替换，再以math解析值逐项比较；合法归一化输入。'))
claim('c9','software','当前dpo_loss扣固定参考差且不给参考梯度，拒绝非正β；此例没有奖赏网络。','Python import/call代码块及beta条件',
 [ev('s-alignment','whole file; dpo_loss','真实实现的detach/logsigmoid/mean/validation'),ev('s-execution','toy_cases and prerequisite_gradient','实际代码与backward/非正β probe')],['a-code','a-audit','a-audit-stdout'],
 'Py3.13.5 torch2.14.1+cpu；实算local float32。接口读取源代码commit与实际版本分开记录，不推论所有平台。')
claim('c10','concept','原论文带参考约束目标的推导不保证任何PPO实作得到同一模型，实际模型与数据限制须另验。','链接DPO论文第4节段落',
 [ev('s-dpo','§3 Eq.3; §4 Eq.4–7; AppendixA.1; §7 Limitations','理想非参数最优分布/偏好模型条件；未给任意有限实现的权重等同保证')],['a-authority'],'原理论等价的条件不等于所有有限模型、数据、优化配方、seed得到同权重或普遍能力。')
claim('c11','numeric','正式小选卡策略有148个参数。','正式CPU比较段“148参数”',
 [ev('s-models','FiniteResponsePolicy default dimensions','4→16→4含bias'),ev('s-execution','rerun_state_audit.sft/ppo/dpo','实例化和真实重载numel')],['a-audit'],
 '4×16+16 +16×4+4=148；仅有限卡网络。',calc('148','148','通过实算sum(p.numel())独立核对；附解析64+16+64+4。','integer exact'))
claim('c12','empirical','PPO与DPO从保存的同一示范选卡起点逐项相同权重开始，固定reference全程不变。','图前段与正式CPU比较开头',
 [ev('s-driver','reference deepcopy; ppo/dpo deepcopy; initial state hashes; final assert','独立深拷贝同一起点'),ev('s-record','results.ppo/dpo.initial_state_sha256; reference_state_sha256_before/after','原记录一致fingerprints'),ev('s-execution','same_start_and_fixed_reference','新CPU运行同次身份/固定参照，重载指纹与独立修改fork probe')],['a-native','a-audit'],
 '同次run的两分支逐项一致。本人新run与历史run完整tensor bytes不相同，未声称跨run bitwise重现。',denominators={'seed':42,'runs':{'original':1,'own_native_cpu':1},'reference_checks':'before/after and both branch initial identities within each run'})
claim('c13','software','使用同一无序加数家族划分，交换加数/改条件不拆家族，两支偏好都只取44训练家族。','正式CPU比较段家族定义与44训练组',
 [ev('s-driver','build_records/split_records; train-only pairs','a<=b代表无序家族，三mode同行家族；train索引'),ev('s-execution','split_audit/family_swap_check','独立构建55family和seed42 shuffle；44/5/6不交叠，与原record逐项比较')],['a-audit'],
 '实际只存a<=b一种代表，没有声称把a+b与b+a都实际加入；train132、validation15、test18，原分组1+2预留。')
claim('c14','numeric','DPO360次更新、每批64对，抽用23040偏好对。','正式CPU比较段DPO预算',
 [ev('s-driver','CONFIG and DPO loop','360批，每批随机有放回64训练对'),ev('s-execution','budgets.dpo_pair_draws/configuration','360×64独立乘算且原/新run计数相同')],['a-native','a-audit'],
 '23040是重复抽用量；训练pool660不同对，不是23040独立标签。',calc('360×64=23040','23040','使用固定DPO config，核原record和native run schedule；非长token训练。','integer exact'))
claim('c15','numeric','PPO360策略更新另有360critic更新、300奖赏模型更新与7680真实选卡；相同策略步数不等于相同预算。','正式CPU比较段PPO额外预算',
 [ev('s-driver','RM/PPO loops and counter construction','RM300，PPO120rollouts×3epochs×64 draws'),ev('s-ppo','Algorithm1','一批样本可重复用于K epochs'),ev('s-execution','budgets','独立计算并核原记录与新run')],['a-audit'],
 '120×64=7680新抽样；复用120×64×3=23040；RM300×64=19200对，critic不是新增回答。',calc('policy360, critic360, reward300, new samples7680','policy360, critic360, reward300, new samples7680','分别保留动作、更新与偏好对单位；CPU配置seed42。','integer exact'))
claim('c16','empirical','最后18题PPO与DPO都12题成功：数答6/6、说明0/6、求补信息6/6。','正式CPU比较段结果与13.16分项引用',
 [ev('s-record','results.evaluations.test.rows and test.policies','每题actual概率原记录'),ev('s-execution','independent_results.test and test_predictions','由概率独立argmax并按number0/explain1/missing3规则重算；新run重载相同离散分项')],
 ['a-native','a-audit'],'每mode六个留出无序家族，greedy候选选择成功，非生成中文字/加法能力；单seed固定配方。',
 denominators={'seed':42,'test_contexts':18,'test_families':6,'contexts_per_mode':6,'rules':'number action0; explain action1; missing action3','successes':{'sft':6,'ppo':12,'dpo':12},'by_mode':['6/6','0/6','6/6'],'selection':'greedy argmax over four prepared cards','own_runs':1})
claim('c17','empirical','原CPU记录PPO段约0.66秒、DPO约0.34秒、评员另约0.27秒，只是本机小网路stage时间。','CPU时间段',
 [ev('s-record','results.ppo.seconds/dpo.seconds/reward.seconds','原计时0.657001449/0.343355792/0.270443069秒'),ev('s-driver','each stage_started and elapsed boundaries','per-stage timer覆盖对应loop，PPO包含critic及取样；RM另测'),ev('s-execution','original_record_timing/timing_scope','独立读原值round2；本人native总体3.1438859829977446秒独立保存')],['a-audit','a-native'],
 '原单次CPU、两个线程，无暖机/重复benchmark；本人未复现同墙钟数字。不能外推大型LLM、GPU性能或推论吞吐。',
 denominators={'original_runs':1,'warmup_runs':0,'repeat_benchmark_runs':0,'cpu_threads':2,'device':'cpu','seed':42,'stage_budgets':'PPO120×64×3 plus critic; DPO360×64; RM300×64','unit':'wall-clock stage seconds; not GPU benchmark'})
claim('c18','empirical','偏好阶段加入SFT示范未教的explain/missing条件，因此该表不能证明PPO/DPO胜过同数据同预算SFT。','CPU时间段后半限制',
 [ev('s-driver','sft_indices mode number; _pairs all train modes; comparison_limits','SFT44number-only，偏好660对来自全部三mode'),ev('s-execution','configuration/budgets/split_audit','数据/更新/抽用数量不相等；原表只新增missing成功而explain失败')],['a-audit'],
 '固定seed配方观察到有限候选求补选择改善；没有同资料预算SFT对照，无方法优越性结论。',denominators={'seed':42,'sft_number_demonstrations':44,'sft_updates':60,'sft_draws':1920,'preference_unique_pairs':660,'preference_modes':3,'equal_data_budget_sft_control_runs':0})
claim('c19','software','此实验偏好标签由作者写定规则经Python应用，不是招募真人标注的RLHF实证。','倒数第二段回馈来源',
 [ev('s-driver','rule_best_action/build_records label_source','固定排行与mode规则直接生成标签'),ev('s-instruct','§3.1 human-labeler steps','典型human feedback来源的必要区别'),ev('s-execution','split_audit each row expected action and preference pairs','逐条独立核规则生成标签，native run使用同records')],['a-audit','a-authority'],
 'human-designed rules与human-rater comparisons是不同来源；仍有作者定义的标准，不声称程序无价值选择。')
claim('c20','software','有限选卡PPO与成品token语言模型分支分开；现行成品DPO从joint父checkpoint续训。','倒数第二段链接19.8与权重来源说明',
 [ev('s-capstone-driver','EXPERIMENTS; train_stage; _run_context; run_preference','成品dpo父阶段joint，不引入有限posttraining权重'),ev('s-capstone','STAGES; preference_pairs/preference_loss/frozen_reference','sequence DPO路径和独立model'),ev('s-execution','capstone_dispatch_probe','实际运行dispatch并拦截training callable，input指向capstone_joint/model.pt')],['a-audit','a-authority'],
 '只验证当前成品代码/依赖路径与教材限定，dispatch probe没有加载成品或GPU训练；没有替19.8所有结果背书。')
claim('c21','concept','偏好loss下降仍需实际选择/生成任务结果、原能力、格式与诚实检查，才能支持助理发布用途。','最后练习段检查缺证据',
 [ev('s-dpo','§7 Limitations & Future Work p.10','偏好训练的泛化/overoptimization仍要评估'),ev('s-record','test modes outcomes despite preference training','两个分支说明仍0/6；偏好目标不蕴含所有要求达标')],['a-authority','a-audit'],
 '这些是最低必要检查而非充分发布保证；有限卡实验不证明通用诚实/安全，文中未给该外推。')
claim('c22','concept','主图从同一SFT分叉PPO/DPO，各保留参考并汇入共同独立规则检查，没有要求PPO后再DPO。','ppo_dpo_routes.svg与图前后两段',
 [ev('s-dpo','§3 RLHF vs §4 DPO','可选择的两种偏好训练路径'),ev('s-driver','same reference forks and _evaluate policies','项目真正两个独立分支、相同evaluation rows')],
 ['a-figure-1','a-figure-notes','a-render-receipt'],'图是流程/角色图，无性能量尺；已本人实际render/view数值、方向和位置，额外三前置图亦亲自看过。')

verification_rows={
 'c9':('positive beta accepted; nonpositive rejected; fixed reference gradients absent',
        'beta0/-1 raised ValueError; reference gradients None; policy chosen/rejected gradients -0.05/+0.05',
        'Actual current lesson code plus independent backward and ValueError probes in audit_cpu.py.'),
 'c12':('Each run has identical PPO/DPO initial/reference states and unchanged reference',
        'Original record initials/before/after all f6ebe6766d91b324a2af12d628af5e7df9cfd144a1961e5ce551f46f2467aacc; own initials/before/after all 766d27d9335137e11faa375cd536580cfffb41f5e50e0f40cd930d5346fc250a',
        'Parse original record identities; actual native rerun; reload own sft state and compare to own branch initial and before/after reference fingerprints; independently mutate copied fork while reference remains unchanged. Cross-run hashes intentionally differ.'),
 'c13':('55 unordered families; disjoint44/5/6 split; three contexts per family; both preferences train-only',
        'Independent reconstruction matches original and native split44/5/6 and132/15/18 contexts;660/75/90 preference pairs; pair1:2 holds both swapped inputs in normalization probe',
        'Independently enumerate a<=b, shuffle Random42 after reserved pair1:2, compare family sets and actual row features/rankings and serialized-record fingerprints. Original loops use only train pairs.'),
 'c16':('Original and own greedy success SFT6/18,PPO12/18,DPO12/18; PPO/DPO by mode6/6,0/6,6/6',
        'Independent original argmax/mode-rule counts exactly match; native reloaded predictions reproduce discrete outcomes, historical-vs-rerun probabilities maximum error9.5367431640625e-07',
        'Read all original rows, independently argmax probability arrays and apply number0/explain1/missing3 rule. Load actual own checkpoints and forward the same features; compare probabilities and mode counts. No raw-text generation claim.'),
 'c17':('Original record durations round2 to PPO0.66,DPO0.34,RM0.27 seconds',
        'Original values0.6570014490000062,0.34335579199995436,0.2704430690000095; round2 matches0.66/0.34/0.27. Own native elapsed3.1438859829977446 recorded separately',
        'Executed parser/rounding of real original CPU record; inspect exact stage timer boundaries in original driver. No assertion that a new run has the same wall time, and no repeat performance benchmark.'),
 'c18':('SFT number-only44; preference pool all3 modes660 pairs; no equal-data/equal-budget SFT control',
        'Actual config/loop and independent row counters confirm number-only44 demonstrations×60 steps×32 draws versus train132 contexts/all3 modes660 pairs and different updates/sampling budgets',
        'Native run uses inspected config; reconstruct and compare all mode/pair counts and success metrics. This establishes the missing fair-control condition; it does not execute or invent such a control.'),
 'c19':('Labels generated by explicit mode/ranking rules applied by Python',
        'All original rows match independently reconstructed ranking and expected action for all165 contexts; native run uses identical recreated records SHA; no human-rater collection is reported',
        'Execute build_records and independently check every generated/original mode, preference ranking and expected action. Read label_source and original driver; compare with InstructGPT human-labeler method scope.'),
 'c20':('Real capstone DPO dependency dispatcher chooses joint predecessor; finite PPO is separate',
        'Executed _run_context(ctx,dpo) calls intercepted train_stage with input_checkpoint ending dependencies/capstone_joint/model.pt,steps100,seed42,devicecpu',
        'Execute actual dispatcher with train_stage mocked only to capture arguments; inspect real STAGES/EXPERIMENTS/sequence-DPO branch. The probe never loaded/trained a token model or GPU.')}
for c in claims:
    if c['id'] in verification_rows:
        expected,observed,details=verification_rows[c['id']]
        c['verification']=calc(expected,observed,details,
            'integer/boolean/discrete selection exact; within-run identity exact; cross-run probability comparison <=5e-6; timing only rounding of original fields')
    if 'denominators' in c:
        c['verification']['denominators']=c.pop('denominators')
    if c['id'] in ['c1','c11','c12','c13','c15','c16','c18','c22']:
        c['artifact_ids'].append('a-closure')
    if c['id'] in ['c12','c18','c22']:
        c['scope']+=' 同一owner完整重读current13.15/13.16及新增7.17：有限随机起点借用SFT示范阶段教法；图与记录的SFT/checkpoint命名不表示预训练token语言模型。'
numeric=[c['id'] for c in claims if c['kind'] in ['numeric','empirical']]
allids=[c['id'] for c in claims]
report={'schema_version':1,'review_stage':'technical','lesson_id':'13.17',
 'source':'course/chapters/13.md#13.17','reviewer_task':'/root/v4_review_coordinator/factual_v4_13_17',
 'reviewer_context':'fresh','source_sha256':sha(ART/'original-section.md'),
 'figure_sha256':{f['source']:f['sha256'] for f in figures},'verdict':'pass','claims':claims,'sources':sources,
 'artifacts':artifacts,'issues':[],
 'checks':{
   'factual_accuracy':{'status':'pass','details':'22项实质主张逐条本人核对；原DPO/典型RLHF/PPO方法与项目有限配方区分，未发现需要修改的事实矛盾。','claim_ids':allids},
   'numeric_verification':{'status':'pass','details':'当前代码块与0.3/0.7练习真实CPU执行，解析log-sigmoid和梯度复算。重计148参数、44/5/6家族、预算23,040/7,680、各mode6题和12/18；原计时只读/round原记录，不当新benchmark。','claim_ids':numeric},
   'figure_consistency':{'status':'pass','details':'本人Inkscape渲染并view_image主图及四必要前置图，核箭头/左右分支/合流、梯度数与固定角色；closure新增7.17预训练分叉图亦本人看过。完整SVG当前SHA与本人PNG/观察说明已保存。','claim_ids':['c3','c5','c12','c22']},
   'source_verification':{'status':'pass','details':'本人从原始HTTPS指定arXiv版本和PyTorch精确commit实际下载阅读公式/API，并记录date/URL/retrievalSHA/条件。项目原始记录实际codeSHA与现行代码一致，本人执行独立原始概率复算与bounded CPU native run。候选摘要或旧verdict未作证据。','claim_ids':allids},
   'limitations':{'status':'pass','details':'无方法普遍优越/等预算SFT/GPU或大型LLM速度保证。数据是规则标签、预写四卡、显式features、seed42单次小family split。新旧权重bytes不同、概率最大误差9.5367431640625e-07，原/新离散分项一致；只声称同run共同起点，保留首次过强断言失败。成品只执行依赖dispatcher，未重跑正式大训练。','claim_ids':allids}},
 'scope_provenance':{'full_current_section_read':True,'raw_utf8_bytes':(ART/'original-section.md').stat().st_size,
   'all_blank_lines_preserved':True,'first_numbered_section':False,'introduction_not_applicable_reason':'13.17是第13章末节；已完整读取本节和必要明确前置，非first section。',
   'prerequisites':prerequisites,'current_prerequisites':json.loads((ART/'closure-prerequisite-receipt.json').read_text()),
   'same_owner_closure':{'artifact_id':'a-closure','actual_full_reread':['13.15','13.16','7.17'],'initial_report_preserved':str(REL/'first-report.json'),'completed_initial_report_preserved':str(REL/'completed-initial-report.json'),'new_issues':[]},
   'prior_report_saved_unread':{'path':str(REL/'prior-report-unread.json'),'sha256':sha(ART/'prior-report-unread.json')},
   'old_review_or_author_reasoning_read':False,'authority_notes':str(REL/'authority-and-scope.md'),
   'native_cpu_training':True,'gpu_training_replication':False,'first_failed_probe_preserved':True}}
# Bind to actual current section independently of the snapshot.
b=(ROOT/'course/chapters/13.md').read_bytes();matches=list(re.finditer(rb'^## ',b,re.M))
current=next(b[m.start():matches[i+1].start() if i+1<len(matches) else len(b)]
  for i,m in enumerate(matches) if b[m.start():].startswith(b'## 13.17 '))
assert sha(ART/'original-section.md')==hashlib.sha256(current).hexdigest()
for f in figures:assert sha(ROOT/f['source'])==f['sha256']
path=ROOT/'docs/technical-reviews/13.17.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
first=ART/'first-report.json'
if not first.exists():first.write_bytes(path.read_bytes())
print('Written',path.relative_to(ROOT),'verdict',report['verdict'],'claims',len(claims),'sourceSHA',report['source_sha256'])
