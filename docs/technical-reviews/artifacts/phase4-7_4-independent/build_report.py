"""Write this reviewer's independent claim map and preserve verification provenance."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import subprocess

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = Path('docs/technical-reviews/artifacts/phase4-7_4-independent')
OUT = ROOT / BASE
REVIEWER = '/root/phase4_factual_coordinator/factual_7_4'

def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def read(name): return json.loads((OUT/name).read_text())
def write(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

meta=read('original/extraction.json')
bounded=read('bounded.stdout.json')
env=bounded['environment']
original_execution=read('original/execution.json')
version='Working tree read 2026-10-05; baseline git HEAD 022dc9b2ffde92c14ca133406849c1c378bb8e8f; exact file SHA-256 controls version.'
cpu_env={'python':env['python'],'torch':env['torch'],'torch_git_version':env['torch_git_version'],
         'device':'cpu','cuda_build':env['cuda_build'],'threads':'1', 'network':'Offline environment flags; no network calls in bounded probe'}
bounded_command="CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_4-independent/bounded_cpu.py > docs/technical-reviews/artifacts/phase4-7_4-independent/bounded.stdout.json 2> docs/technical-reviews/artifacts/phase4-7_4-independent/bounded.stderr.txt"
write('bounded-execution-receipt.json',{'command':bounded_command,'cwd':str(ROOT),'exit_code':0,
    'actual_completed':True,'environment':cpu_env,'stdout':'bounded.stdout.json','stderr':'bounded.stderr.txt',
    'scope':'Short CPU construction, three chat variants, wrong-mask counterexample, synthetic-logit backward and random-model forward. No optimizer, trained checkpoints, dataset preparation or downloads.',
    'code_sha256':sha(BASE/'bounded_cpu.py'),'stdout_sha256':sha(BASE/'bounded.stdout.json')})

inspection = '''獨立審閱範圍：目前 course/chapters/07.md 的 7.4 全節、前置 7.3 全節、7.4 所引用 SVG 原始內容及以 Inkscape 1.4 渲染的 PNG；目前 data.py、model.py、attention.py 全檔，modern.py 依赖契约；指定兩份 review 方法、checker、section_facts 与 build_course.BOOTSTRAP。未讀舊技術或讀者報告正文、結論或他人判定，沒有派出子代理。
非章首小節，chapter_intro=not_applicable；不以章節導言摘要充當本節主張。7.4 只有一個 Python fence，沒有原實測數字或需要執行的長 recipe；補充段只指向既有訓練配方，不宣稱短碼有訓練成果。未下載模型資料、未準備課程資料、未讀權重、未做模型評測、未重訓、未使用 GPU。
親讀原論文：Attention Is All You Need 原始 PDF 首頁標示 arXiv:1706.03762v7 / 2 Aug 2023、作者與標題；p2 §3 說 auto-regressive next-symbol generation；p3 §3.1 Decoder 說 preventing subsequent-position attention and one-position embedding offset，使 predicted output i 只依賴此前 outputs。此為一般 autoregressive/causal 原理，並非 ByteTokenizer 角色格式的來源。檔案由 locator 指向的原 PDF 複製，本人親自核對版本、正文與 SHA，不沿用審查判定。
親讀三個 HTTPS 原始官方 API 頁面實際 article：PyTorch 2.9 nonzero default 2-D index tensor and lexicographic order；Tensor.item single-element tensor to Python number；functional.cross_entropy input/target shapes、ignore_index=-100 excludes direct gradient、sum reduction。均有 HTTP200 source-fetch-receipt。執行環境是 Python3.13.5 / PyTorch2.14.1+cpu，官方頁版本2.9明確區分，本輪原碼與短測試實際確認所用 API 行為，未宣稱頁面為最新版本。
圖像親看：圖不是連續量座標圖，而是六列離散序列位置表，欄位位置／輸入X／下一項Y；0-3忽略目標，4 assistant·4→A·73，5 A·73→EOS·2。原SVG定義了arrow marker但未使用，畫面沒有箭頭；同列三欄的對應與文字assistant→A、A→EOS一致。灰列文字對應淡藍灰背景，屬色彩稱法，沒有改變有效目標範圍。圖下「輸入仍可讀」只應理解為在 causal 前文範圍內可讀，而非每列任意看未來；短CPU已證索引4可見0..4且不可見5。
自行數值／分母：Q=0x51=81、A=0x41=65、B=0x42=66；加8得89、73、74。完整七項原索引0..6；X長6，原A索引5映到Y4、EOS索引6映到Y5；有效token分母2=A一byte+assistant EOS。均為離散整數，精確比較。額外契約檢查採均勻264類logits：sum=2 ln264=11.151898206292632 nats，mean=ln264=5.575949103146316 nats/token，絕對容差1e-12；梯度僅logit位置4、5，無再位移。這是計算檢查，沒有教材模型成績的含義。
必要小變化：QQ/A的first5、X4/Y73；QQ/B的first5、X4/Y74。另用userQ/assistantA/userR保存相同形狀与兩有效目標，錯误未位移targets[:-1]讓first5讀A猜A，正確targets[1:]first4。隨機未訓練TinyLM：未來A→B不改0..4logits，完整prefix与截至assistant前綴logits一致；忽略直接loss的Q→R可影響assistant輸出。僅證程式資料依賴，不能證學會回答或學習成效。
結論：本節實質主張均有範圍明確的原始來源或執行支援；無未解決技術問題，不修改正文或圖，不commit。
'''
(OUT/'independent-inspection.txt').write_text(inspection)

artifacts=[]
def artifact(id,name,kind,description,command=None,result=None,environment=None):
    a={'id':id,'path':str(BASE/name),'sha256':sha(BASE/name),'kind':kind,'description':description}
    if command is not None: a.update(command=command,result=result,environment=environment)
    artifacts.append(a)

artifact('original_fence','original/fence-1.py','code','逐byte原fence，與section_facts extraction中的SHA核對；沒有修補原碼。')
artifact('section','original/section.md','source_snapshot','7.4 原UTF-8 bytes；含標題直到下一##，未正規化換行。')
artifact('extract','original/extraction.json','source_snapshot','原節/fence/圖/helper/build bootstrap的SHA與行號。')
artifact('bootstrap','original/bootstrap.py','code','真正 notebook 暖身原碼；helper先用它配置CPU環境與import root。')
artifact('original_stdout','original/stdout.txt','execution','原fence實際stdout，兩項assert通過。',
    original_execution['command'],'exit 0; X=[1,3,89,2,4,73], Y=[-100,-100,-100,-100,73,2], first=4 X=4 Y=73; stderr empty',cpu_env)
artifact('original_execution','original/execution.json','execution','原fence worker實際執行receipt，timeout45秒，完成約2.18秒。',
    original_execution['command'],'exit_code=0; one original Python fence executed; no guard events',cpu_env)
artifact('original_env','original/environment.json','source_snapshot','原fence真實軟體/裝置與離線guard記錄。')
artifact('original_stderr','original/stderr.txt','source_snapshot','原碼stderr空檔亦保存SHA。')
artifact('bounded_code','bounded_cpu.py','code','自己寫的有界CPU變化、同數不同位mask、代價分母、causal資料依賴檢查。')
artifact('bounded_result','bounded.stdout.json','execution','逐項變化與loss/causal實際結果及環境/hash。',bounded_command,
    'exit 0; original and QQ/A and QQ/B arrays exact; same-count wrong mask first5; sum=2ln264; mean=ln264; gradients4,5; future difference0; user difference>0',cpu_env)
artifact('bounded_receipt','bounded-execution-receipt.json','execution','有界CPU命令、cwd、實際完成exit0、環境、原碼與stdout SHA。',bounded_command,'exit 0; all assertions completed',cpu_env)
artifact('bounded_stderr','bounded.stderr.txt','source_snapshot','有界CPU stderr空檔。')
artifact('render','figure.png','figure_render','Inkscape 1.4 实际render，1280×1500 PNG 已用view_image親看全部六列与底部说明。')
artifact('svg_snapshot','original/figures/course/figures/rewrite-07-04-answer-alignment.svg','source_snapshot','引用SVG原bytes及版本。')
artifact('prepare_receipt','prepare-receipt.json','source_snapshot','原fence提取/執行、pdftotext、SVG渲染实际command与exit0，原PDF provenance。')
artifact('paper_pdf','sources/vaswani-1706.03762v7.pdf','source_snapshot','原作者論文v7 PDF，親核首頁版本，非技術報告。')
artifact('paper_text','sources/vaswani-1706.03762v7.txt','source_snapshot','自己pdftotext -layout獲得的原論文正文，親讀p2 §3與p3 §3.1。')
artifact('fetch_receipt','source-fetch-receipt.json','source_snapshot','三個官方API版本URL、HTTP200、存取日与下载SHA。')
for name in ['nonzero','item','cross_entropy']:
    artifact('docs_'+name,'sources/pytorch-2.9-'+name+'.html','source_snapshot','独立取得並親讀的PyTorch2.9官方'+name+' API页。')
    artifact('docs_'+name+'_text','sources/pytorch-2.9-'+name+'.txt','source_snapshot','官方API实际article文本，對应HTML而非搜索摘要。')
artifact('inspection','independent-inspection.txt','derivation','親讀來源定位、图像检查、整数/单位/分母/容差、限制和独立范围。')
for name in ['prepare.py','acquire_sources.py','build_report.py']:
    artifact(name.replace('.','_'),name,'code','本次真实证据准备／来源取得／报告生成原码。')
for relative in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py',
                 'scripts/check_technical_reviews.py','docs/review-tools/section_facts.py','scripts/build_course.py',
                 'docs/review-tools/factual-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md','pyproject.toml']:
    artifact('input_'+relative.replace('/','_').replace('.','_'),'inputs/'+relative,'code' if relative.endswith('.py') else 'source_snapshot','本次親讀的当前方法、依賴或程式原bytes快照；非舊報告。')

sources=[
 {'id':'transformer','kind':'paper','title':'Attention Is All You Need','url':'https://arxiv.org/pdf/1706.03762v7',
  'version':'arXiv:1706.03762v7, 2 Aug 2023; original NIPS 2017 paper', 'accessed_on':'2026-10-05',
  'authority_reason':'原作者論文於arXiv的固定版本PDF，作者與首頁arXiv版本親核，SHA bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697。',
  'verified':True,'checked_original':True,'inspection_note':'親讀首頁，p2 §3 Model Architecture autoregressive next-symbol input，p3 §3.1 Decoder one-position offset and preventing subsequent attention。僅支持一般causal next-token原理，不支持repo特有token IDs或角色格式。Snapshot copied from locator PDF; no previous verdict read.'},
]
for name,title,note in [
 ('nonzero','torch.nonzero','default as_tuple=False: indices are 2-D rows, sorted lexicographically; 1-D input first result row contains a single position'),
 ('item','torch.Tensor.item','single-element tensor returns a standard Python number; non-differentiable'),
 ('cross_entropy','torch.nn.functional.cross_entropy','input logits and class-index targets shape contract; ignore_index -100 excludes direct input gradient; sum reduction')]:
    sources.append({'id':'torch_'+name,'kind':'official_docs','title':title,
        'url':'https://docs.pytorch.org/docs/2.9/generated/'+({'nonzero':'torch.nonzero','item':'torch.Tensor.item','cross_entropy':'torch.nn.functional.cross_entropy'}[name])+'.html',
        'version':'Versioned PyTorch 2.9 documentation; runtime independently checked under PyTorch 2.14.1+cpu',
        'accessed_on':'2026-10-05','authority_reason':'PyTorch官方版本化API頁，HTTP200原頁已保存HTML與article文字。',
        'verified':True,'checked_original':True,'inspection_note':'Actual original article independently read: '+note+'. Does not establish model training success.'})
for id,path,note in [
 ('data','tiny_perceptron/data.py','親讀全檔；lines10-21特殊ID/UTF8byte offset；54-68 render_chat完整原ids、角色目标ignore、assistant content+EOS及唯一的ids[:-1]/targets[1:]。'),
 ('model','tiny_perceptron/model.py','親讀全檔；TinyLM forward68-86只輸出同长逐位置logits；loss_sum92-100直接reshape logits/labels且數非IGNORE，masked_loss103-105 total/count，無額外位移。'),
 ('attention','tiny_perceptron/attention.py','親讀全檔；attention_mask10-18使key_position<=query_position，forward48-76实际调用该mask。'),
]:
    sources.append({'id':id,'kind':'repository_code','title':path,'path':path,'sha256':sha(path),
                    'version':version,'verified':True,'inspection_note':note})
sources += [
 {'id':'original_run','kind':'execution','title':'Unchanged 7.4 Python fence CPU run','verified':True,'artifact_id':'original_stdout'},
 {'id':'bounded_run','kind':'execution','title':'Independent bounded CPU alignment and causality checks','verified':True,'artifact_id':'bounded_result'},
 {'id':'arithmetic','kind':'derivation','title':'Independent integer positions and byte-ID arithmetic','verified':True,
  'details':'Q ASCII81 +8=89; A65+8=73; B66+8=74. Original indices0..6; x=ids[:-1], y=targets[1:] means original target at k maps output position k-1. A index5→y4; EOS6→y5. Assistant-content1byte plus EOS1 gives2 supervised tokens. Uniform264 classes: each loss ln264 nats; sum2ln264; mean divided by2, abs tolerance1e-12.'}
]
def ev(source,locator,supports): return {'source_id':source,'locator':locator,'supports':supports}
def verify(expected,observed,details,tolerance=None):
    result={'method':'executed','expected':expected,'observed':observed,'details':details}
    if tolerance: result['tolerance']=tolerance
    return result

claims=[
 {'id':'autoregressive_first_answer','kind':'concept','statement':'第一個A在assistant輸入格預測，因該格正在猜下一項；讀入A的格負責下一項EOS。',
  'location':'7.4 開場與第一段位移說明','scope':'causal next-token位置語義以及本課特定單user/assistant例；首輸出位置4只可讀X0..4，不可讀未來A位5。',
  'status':'verified','evidence':[ev('transformer','p2 §3 and p3 §3.1 Decoder','autoregressive generation、one-position offset与mask阻止预测读将来答案。'),
      ev('data','data.py54-68','本课以角色/内容/EOS建立序列並只配一次下一項目標。'),
      ev('attention','attention.py10-18, forward48-76','query4允许key0..4，禁止key5；监督ignore不是causal mask。'),
      ev('bounded_run','causal_visibility','未来A→B不影响0..4，prefix等价；小型随机模型只核查依赖。')],
  'artifact_ids':['paper_pdf','paper_text','bounded_result','inspection']},
 {'id':'ids_positions_and_figure','kind':'numeric','statement':'Q/A各一ASCIIbyte，ID89/73；原七项序列映到X=[1,3,89,2,4,73]与Y=[-100,-100,-100,-100,73,2]，A5→Y4、EOS6→Y5；图六列表对应同样的位置、输入ID与目标ID。',
  'location':'7.4 第二段、图、原fence后输出解释','scope':'本课ByteTokenizer特殊ID偏移8与Q/A例的整数对齐；图为离散位置表没有单位轴，不能把位置4误当回答ID4。',
  'status':'verified','evidence':[ev('data','data.py10-21,54-68','特殊token ID和byte+8规则、原序列/supervision构建。'),
      ev('arithmetic','Q81+8, A65+8; original target k maps k-1','独立整数算例与索引定位，精确无浮点舍入。'),
      ev('original_run','原stdout兩行','输出与正文和SVG逐列一致。')],
  'artifact_ids':['original_stdout','bounded_result','render','svg_snapshot','inspection'],
  'verification':verify('上述X/Y、first4、X4/Y73、EOS at y5，图位置0..5每列一对','原fence与独立构造全部精确相等；PNG亲看同列对应一致','亲读原SVG并实际render+view；未使用的arrow marker不形成实际箭头，位置是表行索引。','整数/ID/位置精确相等；不适用浮点容差。')},
 {'id':'fence_api_contract','kind':'software','statement':'原fence的(y!=-100).nonzero()[0].item()得到首个有效位置4，兩项assert核对assistant輸入與第一個回答byte。',
  'location':'7.4 唯一Python fence与nonzero解释','scope':'一维Y且有有效目标时default nonzero的索引结果与item；原码只有tokenization、定位、输出與assert，没有梯度或参数更新。',
  'status':'verified','evidence':[ev('torch_nonzero','torch.nonzero Notes; When as_tuple is False','返回2-D nonzero indices，按字典序；一维条件首行包含最小有效索引。'),
      ev('torch_item','Tensor.item() contract','一个元素张量转Python数值。'),ev('original_run','fence-1.py + stdout','原fence实际CPU执行，first4，输入4，目标73，两assert通过。')],
  'artifact_ids':['original_fence','original_stdout','original_execution','original_env','docs_nonzero','docs_item'],
  'verification':verify('first4 X4 Y73；原码成功且不更新模型','exit0，输出完全一致，原码没有创建模型/optimizer/backward','PyTorch2.9官方API文字与2.14.1+cpu实际原fence行为分别记录；不可写成训练或评测。')},
 {'id':'mask_alignment_no_double_shift','kind':'software','statement':'有效目标数不能保证位置正确；mask需与答案一起配到前一预测格。render_chat已配对，TinyLM与loss按同一位置比较，不能再移一次。',
  'location':'7.4「有效目標數對，不保證位置對」整段','scope':'本课render_chat/TinyLM/loss_sum契约；不是所有外部语言模型API的通用labels约定。',
  'status':'verified','evidence':[ev('data','data.py54-68','targets在原答案索引建立，返回targets[1:]同时X=ids[:-1]，确实唯一配对。'),
      ev('model','model.py68-86,92-105','同长逐位置logits与labels直接reshape交叉熵，没有第二次shift。'),
      ev('torch_cross_entropy','input/target parameters and Shape','cross_entropy接受匹配预测列与类索引，本身不实现语言序列位移。'),
      ev('bounded_run','unshifted_mask_counterexample and loss_alignment_and_denominator','同形状同有效数2的错mask first5读A猜A，正mask first4；synthetic gradients仅4、5。')],
  'artifact_ids':['bounded_code','bounded_result','bounded_receipt','docs_cross_entropy'],
  'verification':verify('相同有效数/形状可有不同first；正确loss监督位置4、5且无再shift','userQ/assistantA/userR的正确first4与错误first5均2有效目标；Q/A synthetic grad只4、5','反例的尾部user R用于保留同样两个assistant目标，明确是额外有界变体；不冒充正文原例。')},
 {'id':'eos_mask_and_valid_denominator','kind':'software','statement':'下一格X=73预測EOS2；mask标记哪些目标要计代价，忽略目标的输入仍可作为后续causal前文。',
  'location':'7.4 fence后EOS说明、mask段、图底部「灰列輸入仍可讀」','scope':'正文Q/A只有A一byte与assistant EOS两有效目标；额外uniform loss核分母2，单位nats/token，不是课程模型测量。Ignored user是后续assistant的可见前文，不代表任意读未来。',
  'status':'verified','evidence':[ev('torch_cross_entropy','ignore_index and reduction parameters','-100排除直接input-gradient contribution；sum后由repo除非忽略目标数。'),
      ev('data','data.py63-68','assistant内容+EOS监督，user和role的直接目标忽略。'),
      ev('model','model.py92-105','count=(labels!=-100).sum，sum loss/count，分母为有效token而不是输入长度或消息数。'),
      ev('attention','attention.py10-18','可见范围独立於loss mask，以因果位置决定。'),
      ev('bounded_run','loss_alignment_and_denominator and causal_visibility','两目标sum=2ln264/mean=ln264；前4直接logit梯度为0，改Q→R仍影响assistant logits。')],
  'artifact_ids':['bounded_result','bounded_receipt','render','inspection','docs_cross_entropy'],
  'verification':verify('y5=EOS2；有效分母2；uniform loss mean ln264；忽略user仍可影响后续预测','count2,sum11.151898206292632,mean5.575949103146316；ignored Q→R assistant logits差0.04846435785293579','原例EOS及位置精确；uniform math绝对容差1e-12，nats和nats/token；random model因果可见性仅数据依赖，非准确率或学习证明。')},
 {'id':'qq_b_exercise','kind':'numeric','statement':'将Q改QQ后first从4变5但X[first]仍4、Y[first]仍73；再把A改B，first仍5而答案ID74。',
  'location':'7.4 练习段','scope':'题目长度增加一个ASCIIbyte与保持长度的答案字节替换；位置与token ID的作用分别核查。',
  'status':'verified','evidence':[ev('data','data.py20-21,54-68','Q增加byte使角色/答案索引增加1；B byte66+8=74，长度不变。'),
      ev('bounded_run','QQ/A and QQ/B records','按正文顺序实际执行两次变体，X/Y和first都与手工构造一致。'),
      ev('arithmetic','B66+8=74; source length +1','数字无需近似。')],
  'artifact_ids':['bounded_code','bounded_result','inspection'],
  'verification':verify('QQ/A: first5 X4 Y73; QQ/B: first5 X4 Y74','与预期精确相同，均有效目标2','无模型推理、学习或结果质量断言；仅token位置与编码变化。','整数/ID/索引精确相等。')}
]
all_ids=[c['id'] for c in claims]
report={'schema_version':1,'review_stage':'technical','lesson_id':'7.4','source':'course/chapters/07.md#7.4',
 'source_sha256':meta['source_sha256'],'figure_sha256':meta['figure_sha256'],'verdict':'pass',
 'reviewer_task':REVIEWER,'reviewer_context':'fresh','author_tasks':[],
 'read_scope':'7.4完整当前正文和SVG、前置7.3，实际原fence及有界CPU变体；所列当前源码与原始权威资料，未读旧审阅正文或结论。',
 'chapter_intro':{'status':'not_applicable','reason':'7.4非本章第一小节，章首导言指纹/摘要要求不适用。'},
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[],
 'checks':{
   'factual_accuracy':{'status':'pass','details':'六组实质claim覆盖autoregressive首答案、位置/ID、原API、mask和唯一shift、EOS/loss与causal输入、QQ/B练习；没有未确定技术主张。','claim_ids':all_ids},
   'numeric_verification':{'status':'pass','details':'独立构造原七项序列与2有效token；81+8=89、65+8=73、66+8=74，索引5→4及6→5，QQ/B位置5精确；额外uniform264类loss检査有效分母2，ln264单位nats/token且绝对容差1e-12。','claim_ids':['ids_positions_and_figure','eos_mask_and_valid_denominator','qq_b_exercise']},
   'figure_consistency':{'status':'pass','details':'实际Inkscape1.4渲染1280×1500 PNG并view_image亲看六行全图；0..5位置、X和Y列全部与原stdout一致，绿色有效行4/5，前文ignored input在因果范围内仍可读；没有实际箭头或连续数量轴。','claim_ids':['ids_positions_and_figure','autoregressive_first_answer','eos_mask_and_valid_denominator']},
   'source_verification':{'status':'pass','details':'亲核原作者论文arXiv1706.03762v7首页与p2§3/p3§3.1；三个版本化PyTorch2.9官方原网页HTTP200、实际article亲读；current repo helper契约与SHA已保存。文档2.9和实跑2.14.1+cpu明确区分。','claim_ids':all_ids},
   'limitations':{'status':'pass','details':'原fence是token定位示范，不训练；synthetic logit backward仅核loss位置、分母和ignored直接梯度；随机TinyLM前向仅核causal依赖，不证明学习、SFT质量或用户问题准确率。没有原实测数据需验证；未做数据下载/课程资料准备/权重评测/重训/GPU。','claim_ids':all_ids}
 }}
(ROOT/'docs/technical-reviews/7.4.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],
    'claims':len(claims),'artifacts':len(artifacts),'reviewer_task':REVIEWER},ensure_ascii=False))
