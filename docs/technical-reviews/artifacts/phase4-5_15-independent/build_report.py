"""Serialize this reviewer's own inspected evidence; never read other reports."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
REL=BASE.relative_to(ROOT).as_posix()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
meta=json.loads((BASE/'original/extraction.json').read_text())
probe=json.loads((BASE/'cpu-probe-results.json').read_text())
env={k:str(v) for k,v in probe['environment'].items()}
artifacts=[]
def artifact(identifier,path,kind,description,command=None,result=None):
 item={'id':identifier,'path':REL+'/'+path,'sha256':sha((BASE/path).read_bytes()),'kind':kind,'description':description}
 if kind=='execution':item.update(command=command,result=result,environment=env)
 artifacts.append(item)
 return identifier

artifact('original_section','original/section.md','source_snapshot','原小節 UTF-8 bytes；不正規化換行。')
artifact('original_fence','original/fence-1.py','code','本節唯一原始 Python fence。')
artifact('extraction','original/extraction.json','source_snapshot','小節原始指紋、開始行與 fence 指紋；零圖引用。')
artifact('original_run','original/stdout.txt','execution','唯一原始 fence 實際 CPU stdout。',
 '.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.15 --output /tmp/phase4-5_15-independent-original --execute --timeout 30',
 'exit=0；三個 std=1.035154938697815/1.0004743337631226/1.0190640687942505；同種子重建 True。')
artifact('original_receipt','original/execution.json','source_snapshot','helper 子程序 argv、工作目錄、timeout、exit與stdout/stderr/environment雜湊。')
artifact('original_environment','original/environment.json','source_snapshot','原 fence CPU版本及 guard 無事件、實際導入模組雜湊。')
artifact('original_stderr','original/stderr.txt','source_snapshot','原 fence 空 stderr。')
artifact('probe_code','cpu_probe.py','code','獨立有界 CPU probe；算例、精確練習變化、RNG state 與資料 generator。')
artifact('exercise_code','exercise-without-second-reset.py','code','只刪除建立 b 之前最後一次 reset 的精確原碼變化。')
artifact('probe_run','cpu-probe-results.json','execution','CPU probe 實際結構化結果、版本、輸入hash與分母。',
 "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 30s .venv/bin/python docs/technical-reviews/artifacts/phase4-5_15-independent/cpu_probe.py",
 'exit=0；精確變化 False；重設全部參數相同；std與手算差2.09e-8；width8/16共用流抽樣不同、獨立資料流相同；配對差+0.01/-0.01/+0.04。')
artifact('probe_stdout','cpu-probe.stdout.txt','source_snapshot','真實 probe stdout 含完整精確變化列印。')
artifact('probe_stderr','cpu-probe.stderr.txt','source_snapshot','CPU probe 空 stderr。')
artifact('provenance','input-provenance.json','source_snapshot','所有原始輸入身份、必要上下文、無weight/empiricalJSON/figure說明。')
artifact('commands','commands.json','source_snapshot','實際命令、退出狀態、bounded限制與環境。')
artifact('inspection','inspection-notes.md','derivation','親讀範圍、權威定位、算例分母、版本差及證據邊界。')
artifact('installed_environment','installed-environment.json','source_snapshot','PyTorch 2.14.1+cpu實際依賴與所讀installed契約來源。')
artifact('acquisition','acquisition.json','source_snapshot','固定官方HTTPS snapshots含真實PMLR404紀錄。')
artifact('extra_acquisition','extra-acquisition.json','source_snapshot','作者arXiv v1 PDF、生成器文件的實際HTTPS來源與hash。')
artifact('item_acquisition','item-acquisition.json','source_snapshot','Tensor.item官方文件來源與hash。')
for name in ['prepare_evidence.py','fetch_variance_paper.py','finalize_inputs.py','build_report.py']:
 artifact('code_'+name.replace('.','_'),name,'code','本輪自身來源取得、輸入保存或報告產生程式。')
for name in ['Embedding.py','manual_seed.py','normal_.py','_manual_seed_impl.py','_no_grad_normal_.py','Linear_reset_parameters.py','LayerNorm_reset_parameters.py','Tensor-item.txt','torch-std.txt','torch-equal.txt']:
 artifact('installed_'+name.replace('.','_').replace('-','_'),'installed/'+name,'code','亲讀安裝版原始契約快照；不宣稱與2.8版本相同。')
artifact('model_snapshot','inputs/tiny_perceptron/model.py','code','本節實際TinyLM與ModelConfig契約，以及可隨機生成分支。')
artifact('attention_snapshot','inputs/tiny_perceptron/attention.py','code','TinyLM建構所用預設線性層契約。')
artifact('modern_snapshot','inputs/tiny_perceptron/modern.py','code','本節DenseFFN建構所用預設線性層契約。')

sources=[]
def official(identifier,title,path,url,version,kind,inspection):
 aid=artifact('snapshot_'+identifier,'sources/'+path,'source_snapshot','本輪親自取得並閱讀的原始權威來源。')
 item={'id':identifier,'kind':kind,'title':title,'verified':True,'url':url,'version':version,
       'accessed_on':'2026-10-05','authority_reason':'作者提供的固定版本原始論文。' if kind=='paper' else 'PyTorch 專案自己的原始碼或官方API文件。',
       'checked_original':True,'inspection_note':inspection,'artifact_ids':[aid]}
 sources.append(item)
 return aid
official('variance_paper','Accounting for Variance in Machine Learning Benchmarks','variance-2103.03098v1.pdf',
 'https://arxiv.org/pdf/2103.03098v1','arXiv:2103.03098v1, 2021-03-01','paper',
 '親讀PDF實體第1–5頁相關內容、第7–10頁§4–§6相關內容。支持初始化以外變異、有限測試分母、配對與少量重複及多次比較的定性限制，不借用其成績/閾值。')
artifact('paper_text','sources/variance-2103.03098v1.txt','source_snapshot','由本輪原PDF用pdftotext -layout取得的親讀文字；實體頁碼由form-feed核對。')
official('torch_random','PyTorch torch.random 原碼','pytorch-v2.8.0-random.py',
 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/random.py','v2.8.0','official_source','親讀10–60行get/set_rng_state與manual_seed；對照安裝版manual_seed/_manual_seed_impl，CPU最後呼叫default_generator.manual_seed。')
official('torch_reproducibility','PyTorch Reproducibility 原始文件','pytorch-v2.8.0-randomness.rst',
 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/randomness.rst','v2.8.0','official_docs','親讀全文，重点為跨版本/裝置限制、PyTorch random number generator 重設與連續抽樣、Python/NumPy自有來源、DataLoader的generator用法。')
official('torch_embedding','PyTorch nn.Embedding 原碼與契約','pytorch-v2.8.0-sparse.py',
 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/sparse.py','v2.8.0','official_source','親讀15–44、135–190行，lookup rows、trainable weight、shape及N(0,1)初始化；安裝版Embedding相應契約也已亲讀。')
official('torch_init','PyTorch nn.init.normal_ 原碼','pytorch-v2.8.0-init.py',
 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/init.py','v2.8.0','official_source','亲讀75–82、240–264行：normal_預設mean0/std1/generatorNone，no_grad下取樣填值；對照安裝版normal_與_no_grad_normal_。')
official('torch_std','PyTorch torch.std API','pytorch-2.8-std.html',
 'https://docs.pytorch.org/docs/2.8/generated/torch.std.html','2.8','official_docs','親讀主API完整契約與公式，dim=None降全部軸、correction=1；並讀安裝版__doc__及CPU手算分母2111核對。')
official('torch_equal','PyTorch torch.equal API','pytorch-2.8-equal.html',
 'https://docs.pytorch.org/docs/2.8/generated/torch.equal.html','2.8','official_docs','親讀完整主API契約，same size/elements才True，NaN與dtype說明；本節兩張同dtype有限初值表適用逐格一致。')
official('torch_item','PyTorch Tensor.item API','pytorch-2.8-item.html',
 'https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.item.html','2.8','official_docs','親讀完整主API契約：單元素Tensor取得標準Python數字、不為可微分操作。只列印std。')
official('torch_generator','PyTorch torch.Generator API','pytorch-2.8-generator.html',
 'https://docs.pytorch.org/docs/2.8/generated/torch.Generator.html','2.8','official_docs','親讀前段及get_state/manual_seed：生成器自己管理偽亂數狀態。用於有界資料流分離反例。')
official('torch_randperm','PyTorch torch.randperm API','pytorch-2.8-randperm.html',
 'https://docs.pytorch.org/docs/2.8/generated/torch.randperm.html','2.8','official_docs','親讀完整主API：0到n-1的排列，generator可指定自己的取樣來源。只核對20個ID順序。')
sources.append({'id':'tiny_model','kind':'repository_code','title':'本輪TinyLM/ModelConfig實作快照','verified':True,
 'path':REL+'/inputs/tiny_perceptron/model.py','sha256':sha((BASE/'inputs/tiny_perceptron/model.py').read_bytes()),
 'version':'working tree 2026-10-05; base commit 8905e5cf043eff2fed430439ec33030228c4cb47; exact bytes SHA-256 recorded',
 'inspection_note':'亲讀ModelConfig14–28、TinyLM53–86、generate108–131；embedding=nn.Embedding(vocab,width)，未自定generator；生成temperature>0時用multinomial。讀attention31–46及DenseFFN35–43預設初始化作依賴核對。'})
sources.extend([
 {'id':'original_execution','kind':'execution','title':'本節原碼的本輪CPU實跑','verified':True,'artifact_id':'original_run'},
 {'id':'bounded_probe','kind':'execution','title':'本輪精確變化、算例與RNG CPU檢查','verified':True,'artifact_id':'probe_run'},
 {'id':'paired_derivation','kind':'derivation','title':'假設A/B三對值的獨立Decimal算例','verified':True,
  'details':'A=(.70,.72,.69)，B=(.71,.71,.73)，逐對B-A=(+.01,-.01,+.04)，maxB-maxA=.01；meanA=.703333…、meanB=.716666…、meanDelta=.013333…；各三值與三配對；全是假設數字，無品質實驗。'},
])
def e(source,locator,supports):return {'source_id':source,'locator':locator,'supports':supports}
claims=[
 {'id':'paired_scores','kind':'numeric','statement':'假設A/B三次比較的差值為+0.01/-0.01/+0.04，只報最高值會隱去第二對下降。',
  'location':'5.15 原節第3與24行','scope':'每方法3個假設值、3個配對；只是算例，不是模型或任務成績、成功率、顯著性或B普遍較佳證據。','status':'verified',
  'evidence':[e('paired_derivation','逐對Decimal減法、max與mean','確認正負差、兩組範圍與最高值比較資訊不足。'),e('bounded_probe','cpu-probe-results.json.hypothetical_arithmetic','實際執行Decimal代入核對全部數值。')],
  'artifact_ids':['probe_run','probe_code','inspection'],
  'verification':{'method':'executed','expected':'差+0.01/-0.01/+0.04，第二對負；最高值差0.01。','observed':'全部精確相符；A[0.69,0.72]、B[0.71,0.73]，平均差0.013333…；2正1負。','details':'Decimal精確核對輸入小數；平均為循環小數，未把自行算出的均值變成新模型成績。','tolerance':'輸入、差值、範圍與最大值精確相等；均值按Decimal28位顯示。','denominators':{'scores_per_method':3,'paired_repeats':3}}},
 {'id':'comparison_limits','kind':'concept','statement':'保存全部結果與範圍才看得見波動；少數重複只給局部線索，題數、家族與嘗試數也影響改進結論。',
  'location':'5.15 原節第3、24、26行','scope':'定性評估與證據範圍；沒有聲稱三次足以顯著、穩定或泛化。家族是題目/資料群涵蓋與相關性，不是新增獨立題數或本節新統計指標。','status':'verified',
  'evidence':[e('variance_paper','PDF實體pp1–2 §1/§2.1，pp4–5 §2.2/§3.1，pp7–8 §4.1，pp9–10 §5/§6','波動分布、有限樣本與相關錯誤、重複數、跨資料集與多次比較皆影響可靠結論；只量初始化未涵蓋全部變異。'),e('paired_derivation','第二對0.71-0.72=-0.01與三對分布','本節假設算例確實展示最高值遮住差異。')],
  'artifact_ids':['snapshot_variance_paper','paper_text','inspection','probe_run']},
 {'id':'embedding_fence','kind':'software','statement':'原fence只建立width=8的TinyLM、列印整張輸入embedding初值標準差並逐格比較；每ID查一排可調特徵，三個std接近1但不同。',
  'location':'5.15 原節第5、8–19、22行','scope':'固定264×8=2112個N(0,1)初值；std()用全部元素與correction=1，item僅列印；沒有backward、optimizer、品質評測。三個實測std只描述初始化。','status':'verified',
  'evidence':[e('tiny_model','ModelConfig14–28；TinyLM53–71','width=8與vocab264的可學embedding和ID lookup。'),e('torch_embedding','sparse.py15–44、135–190','查表、權重形狀、trainable parameter、預設N(0,1)與reset。'),e('torch_init','init.py75–82、240–264','normal_的mean0/std1、default generator與no_grad初始化。'),e('torch_std','torch.std signature、公式與dim/correction參數','全表樣本標準差，分母2112-1。'),e('torch_item','Tensor.item Returns','一元素std轉Python數字列印。'),e('original_execution','原stdout四行','原碼真的列印三個近1的不同值與True。'),e('bounded_probe','embedding結果物件','ID0/1/263 lookup正確、requires_grad True、grad None、手算std。')],
  'artifact_ids':['original_run','original_fence','original_receipt','original_environment','probe_run','probe_code','model_snapshot','installed_Embedding_py','installed_normal__py'],
  'verification':{'method':'executed','expected':'三個std近1且不相同；重設同seed比較True；全表2112元素、列印但未訓練。','observed':'1.035154938697815/1.0004743337631226/1.0190640687942505；True；shape[264,8]、ID rows match、grad None。','details':'原碼一次CPU實跑；独立N-1分母double手算1.0351549596293472，差2.09e-8<1e-6。有限初始化不是quality score。','tolerance':'手算std對float32結果絕對差<1e-6；shape/lookup/True精確。'}},
 {'id':'seed_semantics','kind':'concept','statement':'初始化、batch抽樣及生成可能使用亂數；種子設定重現的起點而不代表品質排名。',
  'location':'5.15 原節第5與22行','scope':'manual_seed控制此TinyLM在同一CPU版本的PyTorch預設生成器亂數初始化；不涵蓋未另設seed的Python/NumPy/私有generator，不承諾跨版本/平台或整套GPU訓練逐位重現。生成只有隨機取樣分支使用亂數。','status':'verified',
  'evidence':[e('torch_random','random.py32–60','manual_seed是生成器seed/state設定，不存在品質排序語義。'),e('torch_reproducibility','首段；PyTorch random number generator；Python/NumPy；DataLoader','重現須限定環境與其他亂數來源，batch抽樣可獨立設generator。'),e('tiny_model','generate108–124','temperature>0才multinomial取樣，其他分支argmax。'),e('variance_paper','PDFpp1–3 §1/§2.1/§2.2','權重初始化與例子訪問順序是不同變異來源，seed不是分數。')],
  'artifact_ids':['snapshot_torch_random','snapshot_torch_reproducibility','installed__manual_seed_impl_py','model_snapshot','inspection']},
 {'id':'reset_exercise','kind':'software','statement':'兩次建立相同架構前各重設seed1得到逐格相同表；只刪建立b前的重設，建構a已前進RNG狀態，b繼續抽樣而比較False。',
  'location':'5.15 原節第15–19、22、28行','scope':'這個實際CPU/版本/config；沒有保證所有随机抽樣永遠不同。變量仍保存整數種子也不會自動重設generator state。','status':'verified',
  'evidence':[e('torch_reproducibility','PyTorch random number generator段落：consecutive calls與reset between calls','序列會前進，重設相同起點可使同序列重現。'),e('torch_equal','torch.equal Returns與Notes','有限同shape表True代表元素相同，不是近似比較。'),e('original_execution','原fence最後列印','原碼True。'),e('bounded_probe','EXACT EXERCISE MUTATION OUTPUT與rng物件','精確只刪第二seed1行的實跑False；兩次建構state前進；重新seed後全部参数相同。')],
  'artifact_ids':['original_run','probe_run','probe_stdout','exercise_code','probe_code','installed__manual_seed_impl_py'],
  'verification':{'method':'executed','expected':'原碼True、精確指定變化False；建構前後state不同，重設可重建相同參數。','observed':'全部相符；原loop其他內容未改動，variant SHA另記。','details':'保存原始與rfind最後seed1行刪除後bytes；真正exec精確變化，不以另寫相似程序替代練習。'}},
 {'id':'separate_rng','kind':'concept','statement':'兩方法可共用種子作配對，但同種子不保證batch相同；不同架構可消耗不同亂數，必要時分開模型與資料亂數。',
  'location':'5.15 原節第26行','scope':'共用同一生成器串流時有可能錯開；可比較同組seed，但配對本身不保證相同所有條件或充分的統計證據。本次反例只查20-ID順序，不宣稱真訓練loader已分離。','status':'verified',
  'evidence':[e('torch_reproducibility','PyTorch random number generator與DataLoader段落','連續抽樣消耗生成器狀態，DataLoader可另用generator管理資料流。'),e('torch_generator','Generator定義及get_state/manual_seed','不同generator獨立管理自身抽樣狀態。'),e('torch_randperm','torch.randperm signature及generator參數','排列可指定資料生成器。'),e('bounded_probe','rng.shared_default_rng_width8/16_permutation與independent_data_generator_width8/16_permutation','同seed初始化不同寬度後，共用流資料順序不同，独立data seed17時相同。'),e('variance_paper','PDFpp7–8 §4.1的paired empirical risks；pp1–3資料順序變异','支持保留可比較的配對結果與其他條件，不支持只靠同整數seed保證公平。')],
  'artifact_ids':['probe_run','probe_code','snapshot_torch_reproducibility','snapshot_torch_generator','snapshot_torch_randperm','inspection']},
]
report={'schema_version':1,'review_stage':'technical','lesson_id':'5.15','source':'course/chapters/05.md#5.15',
 'source_sha256':meta['source_sha256'],'source_file_sha256':meta['source_file_sha256'],
 'reviewer_task':'/root/phase4_factual_coordinator/factual_5_15','reviewer_context':'fresh','author_tasks':[],
 'verdict':'pass','figure_sha256':{},'reviewed_on':'2026-10-05',
 'applicability':{'substantive_claims':True,'reason':'種子、初始化、配對數值與少量比較的結論範圍均需獨立核對。'},
 'read_scope':{'lesson':'目前5.15全節原bytes/fence','necessary_context':['course/chapters/03.md#3.5','course/chapters/05.md#5.14'],
   'chapter_introduction':'非章首，未讀章首','old_reviews':'未讀舊technical/reader報告正文或判定','external_originals':'逐來源inspection_note與inspection-notes.md記錄','empirical_json':'無本節既有實測JSON引用；A/B明确假設算例，無新模型成績','figures':'沒有引用圖，未render或宣稱頁面視覺驗證'},
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[],
 'checks':{
  'factual_accuracy':{'status':'pass','details':'六組實質主張逐項核對；seed為RNG起點，embedding預設N(0,1)，樣本比較不構成普遍品質進步。','claim_ids':[c['id'] for c in claims]},
  'numeric_verification':{'status':'pass','details':'假設三對差精確核對；每方法三數/三對，2正1負；最高值差0.01。全表std分母2111，float32/double差2.09e-8。未生成模型成績。','claim_ids':['paired_scores','embedding_fence']},
  'figure_consistency':{'status':'not_applicable','details':'5.15無圖片或SVG引用；數字分布與RNG行為由文字和已執行程式足以說明，不需猜空間素材。未宣稱render/桌面手機頁面檢查。','claim_ids':[]},
  'source_verification':{'status':'pass','details':'親讀作者arXiv v1 PDF與固定PyTorch2.8權威原件，并親讀安裝版2.14.1CPU契約；精確URL/版本/日期/hash/支持範圍各來源與artifact保存。PMLR404已真記錄並補作者原件。','claim_ids':[c['id'] for c in claims]},
  'limitations':{'status':'pass','details':'有界CPU原碼與精確練習變化；無訓練/GPU/data/model下載。A/B是三對假設值，無既有JSON需核對、無統計顯著性或泛化保證。近1是初值而非quality；same seed只限同環境、default stream。資料流反例不代表真loader驗收。','claim_ids':['paired_scores','comparison_limits','embedding_fence','seed_semantics','reset_exercise','separate_rng']}
 }}
out=ROOT/'docs/technical-reviews/5.15.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'verdict':report['verdict'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'source_sha256':meta['source_sha256']},ensure_ascii=False,indent=2))
