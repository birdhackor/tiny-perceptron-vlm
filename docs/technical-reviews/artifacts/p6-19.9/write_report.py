"""Write this reviewer's independently checked section 19.9 report."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
PREFIX=OUT.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
env=json.loads((OUT/'cpu-results.json').read_text())['environment']
artifacts=[]
def artifact(id,name,kind,description,**extra):
    p=f'{PREFIX}/{name}'
    artifacts.append(dict(id=id,kind=kind,path=p,sha256=sha(p),description=description,**extra))

artifact('a_cpu','cpu-results.json','execution','CPU 原 fence、split 小變化、錯位置對照、模態 cache、routing、batching 及交付配置結構核對。',
    command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.9/verify_cpu.py',
    result='Exit 0; all_assertions_passed=true；split3 最大差4.76837158203125e-7，split2差0，模態編碼各一次。',environment=env)
artifact('a_raw','raw-audit-results.json','execution','既有原始訓練/公開推理記錄的具名 pointers、配置及 stdout/stderr hash 程式核對；沒有重跑模型評測。',
    command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.9/audit_raw.py',
    result='Exit 0; final/training/public configurations and exported weight hashes match; original text/image/audio CPU argv and outputs agree.',environment={'python':env['python'],'device':'cpu','mode':'archived receipt audit only'})
artifact('a_sources','source-audit-results.json','execution','官方 PyTorch 提交與本機檔案逐 byte 相同；八個實際 MoE 訓練 revision 的 batching AST 均同現稿。',
    command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.9/audit_sources.py',
    result='Exit 0; 3 official PyTorch files byte-identical and 8 archived MoE stage batch methods match.',environment=env)
for name,id,desc in [('verify_cpu.py','a_cpu_code','實際 CPU 檢查原碼，含有界原 fence 子程序及機制對照。'),
    ('audit_raw.py','a_raw_code','實際原始 pointers/hash 核對程式。'),('audit_sources.py','a_sources_code','實際版本/AST 核對程式。'),
    ('fence-split3.py','a_fence3','本節原始 Python fence，不改內容。'),('fence-split2.py','a_fence2','只把 split=3 改為2的練習原碼。')]:
    artifact(id,name,'code',desc)
artifact('a_section','section.md','source_snapshot','首次讀取的本節完整原始 UTF-8 bytes，含標題與末尾空行。')
artifact('a_frozen_chapter','frozen-input/course/chapters/19.md','source_snapshot','當次 frozen input 整章原始 bytes；保存全章不代表閱讀全章，本次实际正文閱讀只有19.9。')
artifact('a_ast','ast-locations.json','source_snapshot','必要實作的 AST 類別/函式定位及原檔 SHA-256。')
artifact('a_revision','training-revision-comparison.json','source_snapshot','最終 native training revision 的 encode/batch/objective/main AST 比較。')
artifact('a_torch_match','installed-torch-source-match.json','source_snapshot','三份下載的官方 exact-commit 原碼與本機安裝檔 SHA-256/byte equality。')

official_files=['roformer.pdf','roformer.txt','rmsnorm.pdf','rmsnorm.txt','gqa.pdf','gqa.txt','tied.pdf','tied.txt','moe.pdf','moe.txt',
    'functional-installed-commit.py','module-installed-commit.py','grad_mode-installed-commit.py','sdp_utils-installed-commit.cpp',
    'cache-v4.57.1.md','document_mask.py','fetch-record.json','torch-fetch-record.json']
for filename in official_files:
    artifact('a_external_'+filename.replace('.','_').replace('-','_'),'external/'+filename,'source_snapshot',
        '實際取得並查核的原始外部來源或其 pdftotext 轉換/取得紀錄：'+filename)
for i,p in enumerate(sorted((OUT/'code').rglob('*.py'))):
    artifact('a_code_'+str(i),p.relative_to(OUT).as_posix(),'code','完整未修改實作副本：'+p.relative_to(OUT/'code').as_posix())
for i,p in enumerate(sorted((OUT/'raw').rglob('*'))):
    if p.is_file():artifact('a_rawfile_'+str(i),p.relative_to(OUT).as_posix(),'source_snapshot','完整原始證據副本，核对原件/副本 hash 相同；真正读取范围见 a_raw 的 checked_pointers：'+p.name)
for i,p in enumerate(sorted((OUT/'pipeline-revisions').rglob('*.py'))):
    artifact('a_pipeline_'+str(i),p.relative_to(OUT).as_posix(),'code','原始實際訓練提交的完整 dataset.py；只檢查 RecordEncoder.batch。')

sources=[]
def external(id,kind,title,url,version,note,authority):
    sources.append(dict(id=id,kind=kind,title=title,url=url,version=version,verified=True,checked_original=True,
        accessed_on='2026-10-06',authority_reason=authority,inspection_note=note))
external('s_rope','paper','RoFormer: Enhanced Transformer with Rotary Position Embedding','https://arxiv.org/pdf/2104.09864v5','arXiv:2104.09864v5',
    '亲读§3.1/§3.2.1–3.2.2，式11–16；Q/K 各以位置相关正交旋转矩阵变换，点积依相对位置。保存PDF及文本。','RoPE 提出者的原始论文。')
external('s_rms','paper','Root Mean Square Layer Normalization','https://papers.neurips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf','NeurIPS 2019 proceedings PDF',
    '亲读§4式4：a_i/RMS(a)×g_i，RMS=sqrt(mean(a_i^2))；去除均值中心化。未引用性能数据。','NeurIPS 官方论文原件。')
external('s_gqa','paper','GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints','https://arxiv.org/pdf/2305.13245v3','arXiv:2305.13245v3',
    '亲读§2.2、Figure2；每组Query共享一Key与一Value头，缩减KV cache。论文效果不转作本成品效果。','GQA 提出者原始论文。')
external('s_tied','paper','Using the Output Embedding to Improve Language Models','https://arxiv.org/pdf/1608.05859v3','arXiv:1608.05859v3 (21 Feb 2017)',
    '亲读p1输出h3=Vh2、p2 §2 的 In tied NNLMs, we set U=V=S；核对输入/输出权重共享。','权重绑定方法的原始研究论文。')
external('s_moe','paper','Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer','https://arxiv.org/pdf/1701.06538v1','arXiv:1701.06538v1 (23 Jan 2017)',
    '亲读§2式1、§2.1式2–5；gate稀疏选择expert并加权组合。原论文有noisy gating，本文代码为确定性top-2，未称复现论文完整训练方案。','稀疏gated MoE 提出者原始论文。')
commit='5c4886908584029761b579af026dcfb627c84070'
base='https://raw.githubusercontent.com/pytorch/pytorch/'+commit+'/'
external('s_sdpa','official_source','PyTorch scaled_dot_product_attention official source',base+'torch/nn/functional.py','PyTorch 2.14.1+cpu git '+commit,
    '亲读L6367–6472：SDPA公式、True可读的bool mask、dropout_p=0规则、三种实现及输入限制。下载文件与安装functional.py逐byte相同（a_sources）。','PyTorch 官方仓库 exact installed-commit 原码。')
external('s_sdpa_select','official_source','PyTorch CUDA SDPA eligibility checks',base+'aten/src/ATen/native/transformers/cuda/sdp_utils.cpp','PyTorch git '+commit,
    '亲读L447–454、L894–915、L1047–1095；Flash eligibility检查装置、硬体、shape、head dimension、dtype、mask等条件。只查原码，未运行CUDA。','PyTorch 官方SDPA后端选择原码。')
external('s_eval','official_source','PyTorch Module.eval / Module.train',base+'torch/nn/modules/module.py','PyTorch 2.14.1+cpu git '+commit,
    '亲读L2894–2934；eval等于train(False)，只影响有training-specific行为的模块，如Dropout/BatchNorm。安装原码与官方逐byte相同。','PyTorch 官方 Module API 实现。')
external('s_no_grad','official_source','PyTorch no_grad',base+'torch/autograd/grad_mode.py','PyTorch 2.14.1+cpu git '+commit,
    '亲读L21–43、L75–84；关闭反向梯度记录，factory function与forward AD例外。本节只是普通tensor计算。','PyTorch 官方autograd实现。')
external('s_cache','official_docs','Transformers Caching explanation','https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md','Transformers v4.57.1',
    '亲读 Attention matrices 与 Cache class：因果前缀K/V不随未来token改变，逐层concat新KV，mask覆盖past+current，cache_position继续递增。只引用这些机制，未引用文档近似速度表。','Hugging Face Transformers 官方缓存文档原件。')
external('s_packing','official_source','PyTorch attention-gym document mask','https://raw.githubusercontent.com/pytorch-labs/attention-gym/main/attn_gym/masks/document_mask.py',
    'main fetched 2026-10-06; SHA-256 '+sha(f'{PREFIX}/external/document_mask.py'),
    '亲读L34–59 generate_doc_mask_mod：多序列堆成一长序列，document IDs阻挡跨文件attention；L65–81 packed causal mask。','PyTorch-labs官方attention示范源码。')

def repository(id,path,title,inspection,version='frozen current input b3a6684a1af47f1086c1b9e56ed090ff6699220a'):
    sources.append(dict(id=id,kind='repository_code',title=title,path=path,sha256=sha(path),version=version,verified=True,inspection_note=inspection))
repository('s_model','tiny_perceptron/selftrained/model.py','Selftrained multimodal core',
    '先AST定位，只读L25–185、188–282、295–419；未读description L421–457的额外字符串值。核对top-2有效行路由、默认cache、位置offset与首次模态编码。')
repository('s_attention','tiny_perceptron/attention.py','Actual GQA/RoPE/SDPA/cache attention',
    '先AST定位后读L10–74；因果mask，GQA投影尺寸，未repeat缓存，在读取时repeat_interleave，RoPE及KV append/offset。')
repository('s_modern','tiny_perceptron/modern.py','RMSNorm, RoPE and FFN primitives',
    '先AST定位后读L8–84；RMSNorm最后feature轴、eps=1e-5，RoPE偶数特征配对，DenseFFN及MoE expert列表。')
repository('s_lm','tiny_perceptron/model.py','Shared base language model',
    '先AST定位后读L15–86；Block RMSNorm分支，embedding/output Parameter身份共享、rotary移除绝对position embedding。')
repository('s_inference','tiny_perceptron/selftrained/inference.py','Public inference actual model.generate call',
    '先AST定位后读L215–328；严格加载安全导出，reply内L277–294调用model.generate，未覆盖use_cache默认值。')
repository('s_chat','scripts/selftrained/chat.py','Public CLI dispatch',
    '先AST定位后读L37–110；实际CLI构造InferenceAssistant并调用reply，CPU线程/装置参数。')
repository('s_dataset','tiny_perceptron/selftrained/dataset.py','RecordEncoder encode/batch',
    '先AST定位后读L108–116、174–241；每笔单独encode、每笔一batch row、较短row填pad、mask，未传segments。')
repository('s_train','scripts/selftrained/train.py','Actual training objective and sampler',
    '先AST定位后读sampler.batch L159–174、objective L519–558、main训练loop L905–936；objective调用encoder.batch。原始最后训练commit关键AST同现行；八阶段的RecordEncoder.batch AST也同现行。')
for id,art,title in [('s_cpu','a_cpu','Own bounded CPU run'),('s_raw','a_raw','Own raw receipt audit'),('s_sources','a_sources','Own exact-version and archived training audit')]:
    sources.append(dict(id=id,kind='execution',title=title,verified=True,artifact_id=art))

claims=[]
def ev(source,locator,supports):return {'source_id':source,'locator':locator,'supports':supports}
def claim(id,kind,statement,location,evidence,artifact_ids,scope,verification=None):
    item=dict(id=id,kind=kind,statement=statement,location=location,status='verified',evidence=evidence,artifact_ids=artifact_ids,scope=scope)
    if verification:item['verification']=verification
    claims.append(item)
def ver(expected,observed,details,tolerance=None):
    result=dict(method='executed',expected=expected,observed=observed,details=details)
    if tolerance:result['tolerance']=tolerance
    return result

claim('c1','concept','因果生成逐层保存前缀Key/Value；续算必须维持相同前文、位置与可读范围。','L419',
    [ev('s_cache','Attention matrices; Cache class steps1–3','逐层KV缓存复用、因果不变前缀、mask与cache_position条件'),
     ev('s_attention','L48–74','已旋转K和V按时间concat；查询位置用缓存长度offset')],['a_cpu'],
    '在相同固定权重、因果前缀和相同位置/mask下；本实现cache续算限定无padding、无packing。CPU错误位置对照只检验此局部条件。')
claim('c2','concept','RoPE以位置相关旋转变换Query和Key的特征，使注意力点积包含位置信息。','L425',
    [ev('s_rope','§3.1/§3.2.1–3.2.2, Eq11–16','Query/Key旋转及相对位置信息'),ev('s_modern','rope L19–32','特征配对角度和旋转')],['a_external_roformer_pdf','a_cpu'],
    '偶数head_dim；没有从此机制推出长上下文外推或本模型问答改善。')
claim('c3','concept','RMSNorm按特征均方根调整尺度，再用可学习缩放系数。','L426',
    [ev('s_rms','§4 Eq4','RMS与逐维gain公式'),ev('s_modern','RMSNorm L8–16','最后一轴mean平方+eps、rsqrt和可学Parameter')],['a_external_rmsnorm_pdf','a_cpu'],
    '本实现eps=1e-5，并以float32计算尺度；无均值中心化，不声称本成品速度/品质改善。')
claim('c4','concept','GQA让Query头按组共享Key/Value头，减少这部分投影权重与缓存。','L427',
    [ev('s_gqa','§2.2 and Figure2','每组Query共享一K/V头，降低KV存储'),ev('s_attention','L32–46, L66–70','窄K/V投影与未扩张KV缓存；读取时才repeat')],['a_external_gqa_pdf','a_cpu'],
    '本配置4Q/2KV：K/V权重65536对MHA131072；每token每层KV元素256对512，只说KV部分，SDPA调用前扩张临时KV不等于所有工作量减半。')
claim('c5','concept','tied embeddings使输入查字与输出候选分数共享同一权重表。','L428',
    [ev('s_tied','p1 h3=Vh2; p2 §2 U=V=S','输入/输出embedding共用参数'),ev('s_lm','TinyLM.__init__ L60–66','output.weight直接设为embedding.weight')],['a_external_tied_pdf','a_cpu'],
    '本模型的词表/隐藏宽度一致，输出为权重转置作用于hidden；不泛指所有embedding或所有模型。')
claim('c6','concept','SDPA是注意力接口；FlashAttention是可能后端，装置/dtype/形状影响选择，backend=sdpa本身不能证明已用FlashAttention。','L429、L432',
    [ev('s_sdpa','L6367–6472','接口公式、后端自动选择及各实现输入限制'),
     ev('s_sdpa_select','L447–454, L894–915, L1047–1095','CUDA/硬体、dtype、shape/head/mask eligibility检查')],
    ['a_external_functional_installed_commit_py','a_external_sdp_utils_installed_commit_cpp','a_sources'],
    '官方源码对应实装2.14.1+cpu提交。未做GPU profiling或声称FlashAttention被选中；CPU运行只能支持CPU路径。')
claim('c7','concept','top-k MoE用gate为每个有效token稀疏选expert前馈网络并组合输出。','L430',
    [ev('s_moe','§2 Eq1, §2.1 Eq2–5','gate稀疏选择expert和加权输出'),ev('s_model','PaddingSafeMoE.forward L59–105','只路由有效行，topk=2归一化权重，PAD chosen=-1')],
    ['a_external_moe_pdf','a_cpu'],'本文是确定性、dropless的Python示范，不声称使用原论文noisy gate、容量限制或专用稀疏核心。')
claim('c8','software','交付的MoE核心实际为4Q/2KV、RMSNorm+RoPE、tied、SDPA、每层4expert/top2。','L421–432的实际设置表',
    [ev('s_raw','raw-audit-results.json selected config pointers','最终记录、native训练receipt、实际公开CPU输出配置相同，weight hash相同'),
     ev('s_model','SelftrainedLanguageModel.__init__ L134–151; MaskedBlock L109–116','传递norm=rms/rotary/tied/heads/backend并建立每层expert'),
     ev('s_cpu','/delivered_structure','依原始config构造核心并检查各层实际结构')],['a_raw','a_cpu'],
    '对象为最终公开MoE核心；以原始配置、导出hash和构造结构验证，未重评模型能力或重训。',
    ver('四层各4expert/top2；4Q/2KV、rotary、RMSNorm、SDPA、共享同一Parameter','全部吻合；q=[256,256]，k/v=[128,256]，tied_identity=true',
        '核对三个相互对应原记录的config以及model.safetensors hash；再在CPU构造同配置模型检查结构。'))
claim('c9','software','这份MoE实现是Python逐expert选行、前馈计算和加权累加，不能据此称已接专用稀疏运算核心。','L432',
    [ev('s_model','L81–89','topk概率及for index,expert逐expert dispatch/index_add'),ev('s_modern','DenseFFN L35–53','expert为普通Linear/GELU/Linear')],['a_cpu'],
    '只确定这段实现路径；不推断实际性能，不否认普通PyTorch算子内部可能有优化。',
    ver('有效token各选择两expert，PAD不派发','3有效tokens合计6派发，两个PAD行chosen均[-1,-1]',
        '实际随机核心执行带PAD mask的forward，检查routing counts/chosen；源码分支无专用稀疏调用。'))
claim('c10','software','快取示范为随机初始化宽16/1层模型，用5个人工ID，比较末位置32项候选分数。','L434–450',
    [ev('s_model','SelftrainedConfig L25–53; LimitedAssistant L296–302','构造随机核心而不加载权重'),ev('s_cpu','/split_invariants','末位置输出[1,32]，缓存为两张[1,1,5,8]')],['a_cpu','a_fence3'],
    '机制示范，没有训练或问答目标，因此不提供已学会问答或交付模型品质的证据。',
    ver('5个输入ID，末位置32logits；prefix3/suffix2与full5比较','输出[1,32]；cache累计长度5，输入和配置吻合',
        '实际执行原fence；独立短检查记录logit/cache维度；无checkpoint load和optimizer。'))
claim('c11','numeric','原CPU示范最大分数差约4.8e-7，atol=rtol=1e-4的allclose为True。','L453',
    [ev('s_cpu','/fences/0 and /split_invariants/2','原fence与额外结构记录的同一数值')],['a_cpu','a_fence3'],
    'seed42、CPU float32、当前源码与PyTorch2.14.1+cpu、2threads；此局部浮点近似不是所有平台逐bit等式。',
    ver('约4.8e-7；allclose=True','4.76837158203125e-7；True','逐元素取32项候选logit差的abs max；两路用同一个随机模型。',
        '4.8e-7为两位有效数的四舍五入；allclose条件abs(a-b)<=1e-4+1e-4*abs(b)。'))
claim('c12','software','eval切到评估模式，no_grad省去反向梯度记录；本程式未更新权重或量速度。','L453',
    [ev('s_eval','Module.train/eval L2894–2934','eval=train(False)，只影响特定模块的训练行为'),ev('s_no_grad','class no_grad L21–43, L75–84','关闭反向梯度记录'),
     ev('s_cpu','/no_learning; /split_invariants','state_dict不变、grad全None、输出requires_grad=False')],['a_cpu','a_fence3'],
    'eval不是关闭所有随机API的总开关；本核心没有在forward使用采样、noisy gate或dropout，SDPA明确dropout_p=0。未进行速度/内存或问答品质测量。',
    ver('评估模式、输出不记录反向图、参数不变','training=false；requires_grad=false；所有parameter.grad=None；state_dict逐tensor相同',
        '复制forward前state_dict并比较，检查grad及模式；原fence没有optimizer、backward或计时/内存测量API。'))
claim('c13','software','公开生成默认使用KV cache；每次生成呼叫的完整前文在首次计算编码图像/声音，随后token续算不重编码。','L455前两句',
    [ev('s_inference','reply inner generate L277–294','公开调用未覆盖use_cache，传完整encoded input与模态payload'),
     ev('s_model','generate L384–419; forward L357–371','use_cache=True默认，首次全prefix，后续一token+cache不重送模态'),
     ev('s_raw','public_original_runs and original argv/stdout pointers','公开文本/图像/声音原始CPU运行使用相同配置/导出hash且退出0')],['a_cpu','a_raw'],
    '只在同一次生成中复用。新CLI轮次或tool hop会再构造新完整前文并prefill，不称跨调用永久缓存。随机模态hook检验执行机制，未重评公开模型留出题。',
    ver('默认cache时图像/声音各编码1次，首次prefix随后只输入新token','image=1/audio=1，query输入长度[21,1,1]；关闭cache则各3次/[21,22,23]；两路生成相同IDs',
        '用人工图像和13帧音频、随机微核心，hook真正encoder与Q投影，强制有界3token；另外只核对公开原argv/stdout/stderr/result的指定字段和hash。'))
claim('c14','concept','sequence packing把多笔短序列装在同一长序列，并用文件边界遮罩隔开注意力。','L455的packing定义',
    [ev('s_packing','generate_doc_mask_mod L34–59; generate_packed_causal_doc_mask_mod L65–81','sequence stacked格式与same_doc AND inner causal mask')],['a_external_document_mask_py'],
    '这里指保持各记录独立的packing；还必须正确保留对应位置/标签，本节没有把局部packing示范当成正式使用证据。')
claim('c15','software','正式MoE训练每笔记录一列，短列补PAD，未采用packing。','L455后半',
    [ev('s_dataset','RecordEncoder.batch L222–235','每条record独立encode，row长度max，PAD填充和valid mask，无segments'),
     ev('s_train','objective L531–534; loop L907–928','sampler记录列表直接交encoder.batch再model'),
     ev('s_sources','/training_pipeline_batches, eight archived revisions','实际八个MoE stage revision的batch AST同现行无packing实现')],['a_cpu','a_sources','a_revision'],
    '结论针对实际MoE正式pipeline；依据历史执行revision原码，非由前章示范或作者说明推断，未重训。',
    ver('两条7/11token记录对应两row；短row4PAD，训练pipeline采用该batch方法','shape=[2,11]，mask counts7/11，4个PAD labels=-100，无segments；8阶段batch AST一致',
        'CPU用两笔人工文本记录运行真正RecordEncoder.batch；亲自git show原execution各revision，对batch AST核对。'))
claim('c16','numeric','练习只改split=2，两路仍读取相同5个位置并得到接近的末位置候选分数。','L457',
    [ev('s_cpu','/fences/1 and /split_invariants/1','仅修改split的原fence运行结果及cache/logit形状'),ev('s_model','L155–172','offset=cache时间长度，续算位置自动从prefix长度继续')],['a_cpu','a_fence2'],
    '同seed/配置的局部机制检查，不作速度或问答品质比较；split1/4也通过本次容差，错误重设位置的对照不通过。',
    ver('split2读取prefix2+suffix3，总5位置；allclose=True','最大差0.0；allclose=True；cache时间长度5','只将原fence split=3换为split=2，没有变更ID、权重或容差。',
        '正文预计接近：atol=rtol=1e-4；本次0.0是观察值，不承诺其他平台完全相等。'))

report=dict(schema_version=1,review_stage='technical',lesson_id='19.9',source='course/chapters/19.md#19.9',
    reviewer_task='/root/p6_fact_19_9',reviewer_context='fresh',source_sha256=sha(f'{PREFIX}/section.md'),figure_sha256={},verdict='pass',
    claims=claims,sources=sources,artifacts=artifacts,issues=[],
    frozen_input={'path':f'{PREFIX}/frozen-input/course/chapters/19.md','sha256':sha(f'{PREFIX}/frozen-input/course/chapters/19.md'),
        'meaning':'最初保存的完整Markdown frozen input，而非宣称当前全章或全章阅读。'},
    inspection_record={'read_range':'course/chapters/19.md L417–458 (whole section 19.9 only); no figures referenced.',
        'method':'先讀兩份規約；先AST定位必要函式，后读实际分支；JSON先topkeys后具名pointers。未讀旧reader/technical判定、controller review或作者额外结果结论。',
        'additional_raw_pointer_note':'最初观察final/capacity字段结构时也见到active_count_convention、initialization、generation_policy、dense_comparison方法/范围规约；无实测胜败或旧审阅结论，未据其作判定。正式证据仅具名原config/provenance字段。',
        'limits':'Only bounded CPU mechanism checks and archived original receipt checks. No GPU, full training, or held-out quality reevaluation.'},
    checks={
        'factual_accuracy':{'status':'pass','details':'原论文/官方原码和实际源码逐项相符；配置/缓存/有效路由/packing范围均已独立核对。','claim_ids':[c['id'] for c in claims]},
        'numeric_verification':{'status':'pass','details':'原fence差4.76837158203125e-7与约4.8e-7一致；练习split2差0，各自allclose=True，候选轴和缓存长度正确。','claim_ids':['c11','c16']},
        'figure_consistency':{'status':'not_applicable','details':'19.9无图片/SVG引用。','claim_ids':[]},
        'source_verification':{'status':'pass','details':'亲读五份原论文、官方cache/packing/SDPA/eval/no_grad；PyTorch官方exact commit三文件与实装逐byte相同；原记录hash和选用指针已保存。','claim_ids':[c['id'] for c in claims]},
        'limitations':{'status':'pass','details':'区分成熟机制、实际配置与随机微型一致性示范；无GPU Flash选择或性能收益证据；cache限单次unpacked生成，正式batch未packing，未以此推出回答品质。','claim_ids':[c['id'] for c in claims]}})
target=ROOT/'docs/technical-reviews/19.9.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'initial-report.json').write_bytes(target.read_bytes())
print(json.dumps({'report':str(target.relative_to(ROOT)),'verdict':report['verdict'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'section_sha256':report['source_sha256']},ensure_ascii=False))
