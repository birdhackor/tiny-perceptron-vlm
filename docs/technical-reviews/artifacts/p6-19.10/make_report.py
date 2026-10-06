"""Write this reviewer's new report without opening any prior report."""
from pathlib import Path
import hashlib
import json

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(REPO).as_posix()
artifacts=[]
def artifact(id,kind,p,desc,**kw):
    artifacts.append({'id':id,'kind':kind,'path':rel(p),'sha256':sha(p),'description':desc,**kw})

execution=json.loads((BASE/'verification.json').read_bytes())
environment=execution['environment']
command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.10/verify.py > docs/technical-reviews/artifacts/p6-19.10/verification.stdout.txt'
artifact('a_exec','execution',BASE/'verification.json','本審閱者實際执行原始公開載入器、原節fence及有界CPU路由變體；只讀既有彙總，完整記錄原JSON pointers與SHA。',command=command,result='所有assert通過；兩版完整FP32模型載入；unique参数5447107/2288067；FP32算例20.78/8.73 MiB；top-k變體保留全部expert；既有3734筆彙總分母與完成收據一致。',environment=environment)
artifact('a_stdout','source_snapshot',BASE/'verification.stdout.txt','上述同一次CPU核驗的标准输出，未生成任務答案或評測成績。')
artifact('a_program','code',BASE/'verify.py','實際執行的獨立有界CPU驗證程式。')
artifact('a_section','source_snapshot',BASE/'inputs/19.10.md','首次读取的19.10原始UTF-8 bytes。')
artifact('a_chapter','source_snapshot',BASE/'inputs/19.md.frozen','當次frozen完整章檔；僅作當次輸入來源，完整章SHA不是目前19.10的source_sha256。')
artifact('a_public_fetch','source_snapshot',BASE/'public-fetch.json','從教材19.11固定HF revision取得兩版final原檔的URL、原檔SHA/bytes与精确header副本SHA。')
artifact('a_external_fetch','source_snapshot',BASE/'external-fetch.json','實際取得原論文及官方文件的URL、日期、SHA與原始bytes数量。')
artifact('a_code_pin_program','code',BASE/'verify_frozen_code.py','实际运行的原freeze /code_sha256指针核验程序。')
pin=json.loads((BASE/'frozen-code-check.json').read_bytes())
artifact('a_code_pin','execution',BASE/'frozen-code-check.json','原MoE/Dense frozen code SHA逐个对照实际六份实现的核验。',command=pin['command'],result=pin['result'],environment=pin['environment'])
for name in ['moe-joint','dense-joint']:
    tag=name.split('-')[0]
    for filename,suffix,desc in [('model.safetensors.header.json','header','精確原safetensors header bytes，含每個tensor dtype/shape/offset；不保留大權重payload。'),('model-config.json','config','原公開結構配置。'),('inference-manifest.json','manifest','原公開manifest完整bytes；僅檢查files、stage、selected_step、origin及objective/provenance pointers。')]:
        artifact('a_'+tag+'_'+suffix,'source_snapshot',BASE/'public'/name/filename,desc)

raw_paths={
 'a_release':'docs/selftrained/v2-jobs/release-final-four-exports.json',
 'a_training_manifest':'docs/selftrained/v2-manifest.json',
 'a_moe_train':'docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json',
 'a_dense_train':'docs/selftrained/results/training-raw/dense-weighted/raw/train-receipt.json',
 'a_moe_frozen':'docs/selftrained/results/public-raw/moe/freeze/frozen.json',
 'a_dense_frozen':'docs/selftrained/results/public-raw/dense/freeze/frozen.json',
 'a_moe_metrics':'docs/selftrained/results/public-raw/moe/test/metrics.json',
 'a_dense_metrics':'docs/selftrained/results/public-raw/dense/test/metrics.json',
 'a_moe_eval_receipt':'docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json',
 'a_dense_eval_receipt':'docs/selftrained/results/public-raw/dense/test/evaluation-receipt.json',
}
for id,p in raw_paths.items():
    snapshot=BASE/'raw'/p
    assert sha(snapshot)==sha(REPO/p)
    artifact(id,'source_snapshot',snapshot,'原始證據完整byte副本；已親自核对原件与副本SHA；实際讀取范围見verification.json或inspection.json，不讀notes/review/*scope_correction。',original_path=p,original_sha256=sha(REPO/p))

sources=[]
def source(id,kind,title,**kw):sources.append({'id':id,'kind':kind,'title':title,'verified':True,**kw})
code_specs=[
 ('s_model','tiny_perceptron/selftrained/model.py','完整有限多模態模型、PaddingSafeMoE与cache','SelftrainedConfig 25–53; PaddingSafeMoE.forward 59–105; MaskedBlock 108–130; language model 133–185; vision/OCR/audio 188–282; LimitedAssistant 295–419。先AST定位；沒有讀description額外字串。'),
 ('s_modern','tiny_perceptron/modern.py','DenseFFN及MoEFFN构造','DenseFFN 35–53; MoEFFN.__init__ 59–65：nn.ModuleList保留全部expert。'),
 ('s_lm','tiny_perceptron/model.py','文字模型共享embedding/output权重','TinyLM.__init__/forward 53–86，63–66設定tied同一Parameter。'),
 ('s_attention','tiny_perceptron/attention.py','GQA KV cache构造','CausalAttention 31–74；61–62构建k,v并返回未扩张cache，之后repeat_interleave為工作計算。'),
 ('s_loader','tiny_perceptron/selftrained/inference.py','公开FP32完整载入器','常量19–27；verify_export 74–98；fetch_public_export 101–136；InferenceAssistant.__init__ 218–264：全部鍵、形狀、dtype、有限性、tied一致性驗證，load_state_dict与model.to(device)。'),
 ('s_train','scripts/selftrained/train.py','FP32安全输出与Dense/MoE自己的训练目标','先AST定位；export_inference 235–281(全部state_dict tensor逐個clone，保留tied兩個名稱); set_trainable 284–296; weighted_language_loss 434–451; validate_weighted_source 454–471; objective 519–607; main 689–780(不允许跨architecture初始化)。不读comparison或结果解释字符串。'),
]
for id,p,title,note in code_specs:
    snapshot=BASE/'code'/p
    assert sha(snapshot)==sha(REPO/p)
    aid='a_code_'+id.removeprefix('s_')
    artifact(aid,'code',snapshot,'实际读取原实现的完整byte副本；具体读取范围在source.inspection_note。',original_path=p,original_sha256=sha(REPO/p))
    source(id,'repository_code',title,path=p,sha256=sha(REPO/p),version='2026-10-06工作树；与两版原frozen.json /code_sha256 对应hash亲自核对',inspection_note=note,snapshot_artifact_id=aid)

fetched={Path(x['path']).name:x for x in json.loads((BASE/'external-fetch.json').read_bytes())}
external_specs=[
 ('s_moe','paper','Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer','moe-paper.pdf','arXiv:1701.06538v1, 2017-01-23','MoE原作者论文，提出稀疏按输入选择expert。','亲读摘要、§1.2、§2 equation(1)與§2.1 equation(3)–(5)：G_i=0时无需计算E_i(x)，逐位置可有不同门控；论文并不声称本repo可卸载未用权重。'),
 ('s_quant','paper','Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference','quantization-paper.pdf','arXiv:1712.05877v1, 2017-12-15','量化方案原作者论文。','亲读abstract、§1、Figure1.1、§2.1 equation(1) r=S(q-Z)、§2.2；8-bit权重/activation及32-bit bias和scale开销，不推出全程序RAM/速度按位宽同比缩减，不证明本V2量化验收。'),
 ('s_distill','paper','Distilling the Knowledge in a Neural Network','distillation-paper.pdf','arXiv:1503.02531v1, 2015-03-09','知识蒸馏原作者论文。','亲读§1与§2 equation(1)：较小模型训练匹配teacher高温soft targets，可混合正确标签；较小参数量本身并不构成蒸馏关系。'),
 ('s_seq','paper','Sequence-Level Knowledge Distillation','sequence-distillation-paper.pdf','arXiv:1606.07947v1, 2016-06-25','Kim/Rush对文字序列蒸馏的原论文。','亲读Introduction、Figure1、§3.1與§3.2：学生可匹配teacher word distributions或以teacher beam-search生成文字序列作为监督。'),
 ('s_dtype','official_docs','PyTorch Tensor Attributes / torch.dtype','torch-dtypes.html','PyTorch 2.14官方文档，accessed 2026-10-06；实际runtime 2.14.1+cpu','PyTorch官方dtype接口契约。','亲读torch.dtype Floating point dtypes表，float32/float=32-bit，float16/half与bfloat16=16-bit；不由此推导部署精度保持。'),
 ('s_size','official_docs','PyTorch Tensor.element_size','torch-element-size.html','PyTorch 2.14官方文档，accessed 2026-10-06；实际runtime 2.14.1+cpu','PyTorch官方Tensor存储字节API。','亲读API：返回单元素byte size，默认float例4 bytes；独立实际CPU核验float32=4,float16=bfloat16=2。'),
 ('s_safe','official_source','Safetensors format specification','safetensors-readme.md','huggingface/safetensors tag v0.8.0 README；runtime safetensors 0.8.0','格式维护者官方原代码库说明。','亲读README Format lines74–101：8-byte header length、UTF8 JSON(dtype,shape,data_offsets)、byte-buffer；END-BEGIN为tensor bytes，metadata/header也占文件。'),
 ('s_shared','official_docs','Safetensors Torch shared tensors','safetensors-sharing.html','Safetensors官方main文档快照，accessed 2026-10-06；runtime 0.8.0','格式维护者官方文档解释共享tensor支持与save_model/load_model。','亲读What are shared tensors、Why are shared tensors not saved、How does it work；dict格式不能自动维持任意storage sharing，save_model可选代表名称。本repo用save_file且逐tensor clone，因此必须查实际文件，不能套用自动去重假设。'),
]
for id,kind,title,filename,version,authority,note in external_specs:
    p=BASE/'external'/filename;f=fetched[filename];aid='a_external_'+id.removeprefix('s_')
    assert sha(p)==f['sha256']
    artifact(aid,'source_snapshot',p,'实际亲读的原始权威来源完整bytes；定位与支持边界见source.inspection_note。',url=f['url'])
    source(id,kind,title,url=f['url'],version=version,checked_original=True,accessed_on='2026-10-06',authority_reason=authority,inspection_note=note,sha256=sha(p),snapshot_artifact_id=aid)

pub=json.loads((BASE/'public-fetch.json').read_bytes())
for rows in pub['exports']:
    row=next(x for x in rows if x['path']=='model.safetensors');arch=row['name'].split('-')[0]
    source('s_public_'+arch,'official_source',arch+'最终公开safetensors原权重',url=row['url'],version='HF immutable revision '+row['revision'],checked_original=True,accessed_on='2026-10-06',authority_reason='本V2交付模型的发布者原始不可变payload；仅证明此payload的结构、dtype、内容与bytes。',inspection_note='匿名下载完整原文件，独立算SHA与bytes对照原release recipe及final freeze/metrics；亲读全header并经原InferenceAssistant加载完整模型；只永久保存精确header与原metadata，权重可按同一URL复取；未作held-out题目生成。',sha256=row['sha256'],snapshot_artifact_id='a_'+arch+'_header')
source('s_exec','execution','本审阅者独立有界CPU数值/源码行为核验',artifact_id='a_exec',sha256=sha(BASE/'verification.json'))

sm={s['id']:s for s in sources};am={a['id']:a for a in artifacts}
claims=[]
def ev(id,locator,supports):return {'source_id':id,'locator':locator,'supports':supports}
def claim(id,kind,statement,location,evidence,arts,scope,verification=None):
    c={'id':id,'kind':kind,'statement':statement,'location':location,'status':'verified','evidence':evidence,'artifact_ids':arts,'scope':scope,
       'raw_source_sha256':{r['source_id']:sm[r['source_id']]['sha256'] for r in evidence if 'sha256' in sm[r['source_id']]},
       'raw_artifact_sha256':{a:am[a]['sha256'] for a in arts}}
    if verification:c['verification']=verification
    claims.append(c)
def check(expected,observed,details,**kw):return {'method':'executed','expected':expected,'observed':observed,'details':details,**kw}

claim('c1','concept','MoE每个token只选择部分expert，后续token可选择不同expert；减少expert计算不等于卸载其他权重。','L461第一段',
 [ev('s_moe','§1.2; §2 equation(1); §2.1 equations(3)–(5)','稀疏G仅使未选Ei无需计算，不约定RAM卸载。'),ev('s_model','PaddingSafeMoE.forward 59–105','本实作为每个valid token取top_k并只调用选中expert。')],['a_external_moe','a_code_model','a_exec'],
 '本V2 top_k=2/experts=4；所有非expert工作仍存在。未宣称总参数减半、整体FLOPs/延迟减半，或较Dense更快。')
claim('c2','software','本公开载入器将全部四个expert及完整多模态权重载入CPU RAM；token路由不会移除未选expert权重。','L461“本實作仍把所有expert的權重載入記憶體”',
 [ev('s_loader','InferenceAssistant.__init__ 218–264','完整全部键载入model；CPU为默认/公开命令环境。'),ev('s_modern','MoEFFN.__init__ 59–65','所有expert都在nn.ModuleList。'),ev('s_exec','/models; /controlled_routing_variants','原载入器成功；每版4 synthetic tokens，所有参数地址未变；top-k1/2及正负输入选不同expert仍保留四expert。')],['a_exec','a_program','a_code_loader','a_code_modern'],
 'CPU常驻此原实现；CUDA会将完整模型搬至设备。没有offload或按token按需载入；并非所有MoE实现都必须采用此存储策略。',check('路由变而完整权重仍保留','top-k1: expert0→3；top-k2: experts[0,1]→[3,2]；两者resident_parameters=320不变，final加载参数地址也不变。','只执行原公开constructor、4个合成文本token和width4/hidden8的单token变体；hooks记录真实expert调用。'))
claim('c3','numeric','MoE为5447107约5.447M，Dense为2288067约2.288M；包括文字、影像、OCR、语音全部参数，共享权重只计一次；M为10^6。','L463参数总量段',
 [ev('s_public_moe','完整header及加载后的model.parameters','原final结构与dtype。'),ev('s_public_dense','完整header及加载后的model.parameters','原Dense结构与dtype。'),ev('s_exec','/models/*/unique_parameters,/components','独立unique计数并对照原训练收据total_parameters。')],['a_exec','a_moe_header','a_dense_header','a_moe_train','a_dense_train'],
 '参数总量含已学习但final阶段冻结的感知backbones；不等于单tokenactive参数数或训练阶段trainable_parameters。',check('5447107及2288067；M除以1000000','lm=5140224/1981184；vision91907、OCR126829、audio88147；合计5447107/2288067；M=5.447107/2.288067。','以原loader模型去重parameters求numel；tied embedding/output同一Parameter；差值3159040=4*(3*262912+1024)。',tolerance='整数精确相等；M保留3位小数'))
claim('c4','numeric','此公开版FP32每元素4 bytes；5447107×4=21788428 bytes≈20.78 MiB，2288067×4=9152268 bytes≈8.73 MiB；MiB为1024² bytes。','L463–472及Python fence',
 [ev('s_dtype','torch.dtype Floating point dtypes表','float32为32-bit。'),ev('s_size','Tensor.element_size API与例','单个元素bytes。'),ev('s_exec','/chapter_fence_stdout; /models; /scalar_element_sizes','原节fence实际执行；公开header全部F32，原loader模型参数FP32。')],['a_exec','a_program','a_external_dtype','a_external_size','a_moe_header','a_dense_header'],
 '未启用额外autocast或dtype override的公开CPU路径；只算不重复参数数值，不能代替文件大小、峰值或最低RAM。',check('MoE20.78/Dense8.73 MiB','21788428/1048576=20.779064178466797；9152268/1048576=8.728282928466797；round(...,2)=20.78/8.73。','执行原始fence，另外按所有原model.parameters numel*element_size独立计算并检查全部原header dtype。',tolerance='bytes精确；MiB按原round至小数2位'))
claim('c5','software','实际公开文件含张量保存安排及格式信息，不能用unique参数×4直接当文件bytes；本文件将tied embedding/output分别保存。','L472“公開檔保存張量的安排與格式資訊也占空間”',
 [ev('s_safe','README Format lines74–101','header/metadata与payload分开，offset差为tensor bytes。'),ev('s_shared','What are shared tensors; Why not saved; How does it work','必须核对使用的API和共享保存策略。'),ev('s_train','export_inference 249–254','每个state_dict条目clone后save_file，保留两名称。'),ev('s_exec','/models serialized_parameter_occurrences,serialized_data_bytes,safetensors_file_bytes,format_bytes','真实final文件bytes与header独立核对。')],['a_exec','a_moe_header','a_dense_header','a_public_fetch','a_code_train','a_external_safe','a_external_shared'],
 '共享embedding有550*256=140800个元素，分别clone多563200 bytes；这是此公开safe_file export的行为，非safetensors永远无法去重。',check('文件bytes不同于21788428/9152268','MoE22366908 bytes=22351628数据+15280格式；Dense9725076=9715468数据+9608格式；两者序列化各多140800个F32元素。','原文件完整SHA与release recipe/frozen交叉比对；从精确原header shape与offset求数值数据bytes；原load_file两个embedding storage不同，live模型重新共享。'))
claim('c6','software','权重数值bytes不覆盖载入暂存副本、模态特征、KV cache和计算工作空间，不能据此给整程序RAM峰值或电脑最低需求。','L472末三句',
 [ev('s_loader','InferenceAssistant.__init__ 232–252','state已载入，另外分配完整model并copy，载入期间同时存在。'),ev('s_model','_embeddings 308–355; forward/generate 357–419','模态特征与输出/sequence为额外张量。'),ev('s_attention','CausalAttention.forward 48–74','k,v cache与GQA repeat等工作张量。'),ev('s_exec','/models state_and_live_model_have_distinct_storages; KV_cache_bytes_at_4_tokens','独立原loader/loaded tensors存储不同，4 token cache每版16384 bytes，额外于参数。')],['a_exec','a_code_loader','a_code_model','a_code_attention'],
 '本核验只证明额外存储存在，不量测系统RSS峰值、模型常驻/临时total、实际硬件最低需求；不能把16384作为完整runtime overhead。',check('载入副本及cache另占存储','live模型embedding与另读原safe state storage不同；两版4token各返回16384 bytes KV cache。','只用固定4 synthetic token，不做生成性能、RAM峰值或CUDA测试；需按实际输入长度/模态/硬件另外量测。'))
claim('c7','concept','量化用较少位元近似表示原网络权重；需转换/保存/重载并检验任务与资源，位宽算术不证明部署效果。','L474量化句；L478末句',
 [ev('s_quant','§1; Figure1.1; §2.1 equation(1); §2.2','低位值及scale/zero-point对应原实数；权重压缩、算术实现、性能与准确率各有条件。')],['a_external_quant'],
 '支持量化一般机制，不声称本V2已经量化，亦不保证全模型均为同一低位dtype、文件/运行RAM按同比减半或答案精度保持。')
claim('c8','concept','蒸馏可先选较小学生，向教师的soft分布或教师生成的文字序列学习；参数较少本身不构成蒸馏。','L474蒸餾句；L476“不能把它改稱MoE的蒸餾學生”',
 [ev('s_distill','§1; §2 equation(1)及soft targets训练段','知识转移要求teacher提供训练信号。'),ev('s_seq','Figure1; §3.1; §3.2','word distributions或teacher生成完整序列训练学生。')],['a_external_distill','a_external_seq'],
 '较小学生是本节部署目标，不说所有蒸馏在定义上必须严格更小；不将随机初始化或模型小自动视为已完成teacher/student训练。')
claim('c9','software','本轮V2选定并公开验收的两版final仍为原FP32完整模型；本节没有把第17/18章机制或2-byte练习称为已验收的最终全量量化/蒸馏版本。','L474粗体本轮交付范围句',
 [ev('s_public_moe','HF fixed final文件完整header与原loader','公开MoE全部F32。'),ev('s_public_dense','HF fixed final文件完整header与原loader','公开Dense全部F32。'),ev('s_exec','/models; /existing_aggregate_checks; inspected_json_pointers','final公开payload SHA与freeze/metrics对应，原完成收据对应3734笔。'),ev('s_train','export_inference 235–281; objective 519–607','原导出直接保存浮点模型，原训练不是转换压缩或teacher distillation路径。')],['a_exec','a_release','a_moe_frozen','a_dense_frozen','a_moe_manifest','a_dense_manifest','a_moe_metrics','a_dense_metrics','a_moe_eval_receipt','a_dense_eval_receipt','a_public_fetch'],
 '核对的是此轮固定HF revision的final交付集及其原验收绑定，未遍搜作者工作笔记；不对其他章节toy compression、未公开候选或未来计划断言全局不存在。两版final完整safe payload都为浮点，原验收没有绑定新quantized/distilled final payload。',check('原浮点final与原既有验收绑定一致','公开MoE/Dense全部F32；weight/manifest SHA与原freeze、metrics、receipt一致；各原test完成3734。','只读/汇总已产生count、per_task count与provenance，绝未重新生成留出题或计算新成绩。'))
claim('c10','software','Dense虽参数较少，原最终训练路径是自己的Dense监督训练与阶段checkpoint，不是从MoE教师蒸馏的学生。','L476“不能把它改稱MoE的蒸餾學生”',
 [ev('s_train','weighted_language_loss 434–451; objective 519–607; main 721–780','以现有labels做CE/感知head/aux目标，没有teacher分布forward；禁止跨Dense/MoE载入初始化。'),ev('s_exec','/raw_lineage_measurements; inspected pointers /origin/kind,/stage_history/*','Dense原receipt为独立随机神经来源及自己的阶段历史。')],['a_exec','a_dense_train','a_moe_train','a_dense_manifest','a_code_train'],
 '不把“random initialization”单独当非蒸馏证明：结论同时依赖实际目标、同架构checkpoint约束和原独立历史；只覆盖此final路径，不排除未来为MoE另做Dense student。',check('原Dense自己的训练路径而非MoE teacher','Dense来自all-neural-weights-random，stage pretrain/sft/vision/ocr/audio/joint；实际监督CE目标和本架构checkpoint约束，无MoE教师训练路径。','亲读真正objective/weighted CE和main初始化分支；核对实际原Dense final receipt，不以模型大小、作者备注或旧报告推断。'))
claim('c11','empirical','两final路线有各自既有最终结果；它们训练目标与历程不同，这个比较不能单独归因于架构。','L476“最終能力也有自己的結果”至段末',
 [ev('s_exec','/raw_lineage_measurements; /existing_aggregate_checks','独立重算既有分母并核对实际训练计数和权重配置。'),ev('s_train','weighted_language_loss 434–451; objective 519–607','配置差异实际进入目标归一化与加权CE。')],['a_exec','a_moe_train','a_dense_train','a_moe_metrics','a_dense_metrics','a_moe_eval_receipt','a_dense_eval_receipt','a_code_train'],
 '只证实两版已有分母完整、权重绑定的原结果及比较条件差异，不在19.10重审19.12每项答案分数，不外推质量因果、显著性或架构优劣。',check('两版结果分别对应原final且训练条件并非只改architecture','各最终3734笔；MoE tool/numeric/native=4/1/4，final4000步7977108tokens；Dense=4/4/1，final10000步19945149tokens；SFT完成步也8000/6000不同。','仅从实际原train receipts/config及已产生aggregate count求值；未训练或重评。',denominators={'test_records_per_architecture':'3734','test_tasks':'12','moe_final_completed_updates':'4000','moe_final_tokens':'7977108','dense_final_completed_updates':'10000','dense_final_tokens':'19945149','seed':'20261006 for both; one recorded seed; distinct lineages','weight_payloads':'MoE26102d2e…; Dense5f8b5b04…; full hashes in a_exec/a_public_fetch'}))
claim('c12','numeric','理想每数值2 bytes，两个模型的参数数值存储都恰为FP32算例的一半；这一步仅算术。','L478练习',
 [ev('s_dtype','torch.dtype float16/bfloat16表','存在16-bit元素dtype；不保证实际部署。'),ev('s_exec','/models logical_FP16_bytes; /scalar_element_sizes','2 bytes理想值由unique参数重新算并核对dtype单元素大小。')],['a_exec','a_program','a_external_dtype'],
 '保持unique count假定只改变每元素bytes；没有转换或宣称完整低精度/量化版验收，实际低位文件还会有scale、header及共享保存等开销。',check('两个count*2==count*4/2','MoE10894214 bytes=10.389532089233398 MiB≈10.39；Dense4576134 bytes=4.364141464233398 MiB≈4.36，精确减半。','独立逐参数数值count*2；同时CPU element_size(float16/bfloat16)=2；不把练习写成已实际转换/评测。',tolerance='整数bytes与比例精确；显示MiB round到2位'))

inspection={
 'reviewer_task':'/root/p6_fact_19_10','fresh_context':True,'started_scope':'course/chapters/19.md#19.10 only',
 'read_course_scope':['19.10 raw bytes lines459–478','necessary fixed public pin/package roles in 19.11 lines480–493','targeted rg matches within current chapter for exact pins/paths; no old reports'],
 'initial_frozen_chapter':{'path':rel(BASE/'inputs/19.md.frozen'),'sha256':sha(BASE/'inputs/19.md.frozen'),'meaning':'full chapter at first freeze, not a claim of current entire chapter review'},
 'source_ast_before_read':True,'code_read_ranges':{x[1]:x[3] for x in code_specs},
 'extra_constructor_contract_read':'tiny_perceptron/selftrained/dataset.py RecordEncoder.__init__ 111–116 only: stores tokenizer/path/context/device/cache, performs no data download or test inference.',
 'json_method':'Enumerate topkeys/types before named pointer inspection. Raw copies retain all bytes including untouched annotations. Actual automated pointers in verification.json; manual pointers below.',
 'manual_json_pointers':{'docs/selftrained/v2-manifest.json':['/model_config','/initialization'],'docs/selftrained/v2-jobs/release-final-four-exports.json':['/release/repo_id','/release/prefix','/release/private','/release/manifest_sha256','/release/exports (original selected file metadata/source bindings)'],'public/*/inference-manifest.json':['/files','/stage','/selected_step','/origin/kind','/tool_loss_weight','/numeric_run_loss_weight','/native_voice_loss_weight (MoE only)','/origin/new_joint_initialization/source_integrity/source_run_id (MoE)','/origin/new_joint_initialization/source_completed_steps (MoE)'],'raw final train receipts':['/config','/total_parameters','/trainable_parameters','/steps','/tokens','/seed','/origin/kind','/origin (method/provenance only; no extra result interpretations)','/language_objective_policy (MoE original target definition)','/stage_history/*/stage,step,tokens'],'raw frozen.json':['/weight_source','/safe_weights_sha256','/inference_manifest_sha256','/architecture','/code_sha256 (verify read implementation hashes)']},
 'annotation_exclusions':['notes','review','*_scope_correction','comparison','old reader/technical conclusions and author result interpretations'],
 'retrieval_corrections':[{'event':'First quantization PDF candidate 1712.05877v2 returned 404; corrected to actual original v1, successfully read.'},{'event':'Recipe export.revision is training source Git commit; first treating it as HF revision returned 404. Used current textbook 19.11 immutable HF release pin 979cdfacc588ad0536f1c64fff96f264571cf054; final original hashes match recipe and freeze.'},{'event':'Torch stable tensors URL returned an HTML redirect; followed the stated version2.14 target, read actual official versioned pages.'}],
 'scope':'No old/other review results read. No training, model uploads, paid compute, held-out re-evaluation, or new compression acceptance result. Original aggregate arithmetic only.',
}
(BASE/'inspection.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')
artifact('a_inspection','source_snapshot',BASE/'inspection.json','本审阅者真实读取范围、独立性、JSON pointer边界、外部定位及检索修正记录。')

report={'schema_version':1,'review_stage':'technical','lesson_id':'19.10','source':'course/chapters/19.md#19.10',
 'reviewer_task':'/root/p6_fact_19_10','reviewer_context':'fresh','source_sha256':sha(BASE/'inputs/19.10.md'),'figure_sha256':{},
 'verdict':'pass','claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
 'checks':{
  'factual_accuracy':{'status':'pass','details':'12项实质主张分别核对；原公开FP32完整模型、独立Dense路线及理论机制与原稿一致。','claim_ids':[x['id'] for x in claims]},
  'numeric_verification':{'status':'pass','details':'实际执行原fence并按原loaded模型独立计数；unique bytes、MiB、共享重复tensor、实际file bytes与2-byte理想算术均准确；原最终3734分母亲自求和核对。','claim_ids':['c3','c4','c5','c11','c12']},
  'figure_consistency':{'status':'not_applicable','details':'19.10没有直接图解/图片引用；不把别节图归入本节。','claim_ids':[]},
  'source_verification':{'status':'pass','details':'亲读四篇原论文与PyTorch/Safetensors官方原版；匿名取固定HF final原权重并独立SHA核对；原实现与freeze code hashes一致；每项claim附原raw source/artifact SHA。','claim_ids':[x['id'] for x in claims]},
  'limitations':{'status':'pass','details':'区分resident/selected专家、unique逻辑数值/file实际bytes/runtimeRAM；无速度或最低RAM断言。量化蒸馏只作机制与交付边界核对；现有聚合不当作重评；不同目标/历程不能作架构因果。','claim_ids':['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12']},
 },'inspection_artifact_id':'a_inspection','reviewed_on':'2026-10-06',
}
out=REPO/'docs/technical-reviews/19.10.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':rel(out),'sha256':sha(out),'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'verdict':report['verdict']}))
