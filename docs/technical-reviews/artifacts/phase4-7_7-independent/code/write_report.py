"""Write only this reviewer's own report, retaining the original unresolved claim."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def load(name): return json.loads((ART/name).read_text())
env=load('probe-stdout.txt')['environment']
env={k:str(v) for k,v in env.items()}
extraction=load('original-run/extraction.json')
probe=load('probe-stdout.txt')
artifacts=[]
ids={}
def artifact(identifier,name,kind,description,command=None,result=None):
    p=ART/name
    v={'id':identifier,'path':rel(p),'sha256':sha(p),'kind':kind,'description':description}
    if kind=='execution': v.update(command=command,result=result,environment=env)
    artifacts.append(v); ids[name]=identifier

artifact('e-original','original-run/execution.json','execution','Original raw Python fence executed by the read helper in a fresh CPU process; stdout/environment/fence preserved beside it.',
    '.venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.7 --output /tmp/phase4-7_7-original --execute --timeout 45',
    'Exit 0, fence 1 executed, stdout 有效位置最大差 0.0; no audit-guard events, torch 2.14.1+cpu, CUDA unavailable.')
artifact('e-probe','probe-stdout.txt','execution','Independent small CPU perturbations, query/key matrix, softmax denominator, loss contract and allclose counterexample, raw JSON.',
    '.venv/bin/python docs/technical-reviews/artifacts/phase4-7_7-independent/code/bounded_probe.py',
    'Exit 0; all bounded assertions completed. Remove valid: 0.22275620698928833; remove positions: 1.0177843570709229; mixed allclose accepts a 2e-6 absolute difference.')
artifact('a-figure','figure/padding.png','figure_render','Actual Inkscape rendering at 1280px, personally viewed with view_image; exact original SVG bytes also saved.')
artifact('e-commands','commands.json','execution','Original execution command and independently run probe/render/PDF extraction commands with bounded timeouts and actual return codes.',
    '.venv/bin/python docs/technical-reviews/artifacts/phase4-7_7-independent/code/run_and_preserve.py',
    'Exit 0 for probe, render, inkscape --version and pdftotext; Inkscape 1.4. No training or model download.')
artifact('a-provenance','sources/fetch-provenance.json','source_snapshot','Own HTTPS fetch provenance for official source versions and permitted original-PDF locator; not a copied review judgment.')
artifact('a-installed','sources/installed-api-docstrings.txt','source_snapshot','Personally inspected public API docstrings in installed torch 2.14.1+cpu, distinct from official pinned v2.11.0 snapshot.')
for p in sorted(ART.rglob('*')):
    if not p.is_file() or p.name in {'manifest.json','checker-receipt.json','checker-stdout.txt','checker-stderr.txt'}: continue
    name=p.relative_to(ART).as_posix()
    if name in ids: continue
    identifier='a-'+name.replace('/','-').replace('.','-')
    kind='code' if p.suffix=='.py' else 'source_snapshot'
    if name.startswith('figure/') and p.suffix=='.png': kind='figure_render'
    artifact(identifier,name,kind,'Permanent original bytes / actual command output retained for this independent 7.7 review: '+name)

sources=[]
def original(identifier,kind,title,url,version,why,note):
    sources.append({'id':identifier,'kind':kind,'title':title,'url':url,'version':version,'authority_reason':why,
        'verified':True,'checked_original':True,'accessed_on':'2026-10-05','inspection_note':note})
original('s-paper','paper','Vaswani et al., Attention Is All You Need','https://arxiv.org/pdf/1706.03762v7','arXiv:1706.03762v7, 2023-08-02; NeurIPS 2017',
    'Original authors’ paper hosted on arXiv; own PDF page-one identifier and official abstract history both checked.',
    'Personally read original PDF extraction pages 3–6: §3.1 decoder causality; §3.2.1 Eq.(1) QKᵀ/sqrt(dk), softmax then V; §3.2.3 illegal links -infinity; §3.3 position-wise FFN; §3.5 adding positional encoding. PDF page one says arXiv:1706.03762v7 2 Aug 2023; official abs HTML title, authors, and history confirm. No paper experiment scores used. Local original-paper-locators served only to locate the immutable PDF; I extracted and read it independently.')
original('s-functional','official_source','PyTorch functional.py official public API contracts','https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/functional.py','PyTorch tag v2.11.0 (official snapshot); runtime 2.14.1+cpu recorded separately',
    'Source under official pytorch/pytorch repository and explicit released tag, fetched personally by HTTPS.',
    'Personally read softmax lines 2119–2163: exp(x_i)/sum_j exp(x_j), axis dim; cross_entropy 3414–3455: logits/target, ignore_index=-100, ignored target gradients and nonignored reduction; SDPA 5948–5989,6065–6084: [N,H,L,S] mask broadcast, True participates, sqrt(E) scale, softmax last/key axis. The section executes its own manual implementation, not SDPA. Runtime is newer, 2.14.1+cpu; exact relevant behavior checked by installed public docs and bounded execution rather than claiming version identity.')
original('s-sparse','official_source','PyTorch nn.Embedding official source and docstring','https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/modules/sparse.py','PyTorch v2.11.0; installed runtime 2.14.1+cpu independently inspected',
    'Official pytorch/pytorch tagged implementation.',
    'Read Embedding lines 14–42,134–181: integer lookup maps (* ) indices to (*,H), padding_idx=None by default, normal-initialized weights; explicit padding_idx zero/fixed embedding does not replace attention masking. Repository uses two Embedding tables without padding_idx, so PAD ID0 is an ordinary nonzero vector unless excluded as key; positions are separately looked up.')
original('s-allclose','official_source','PyTorch allclose official contract','https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/_torch_docs.py','PyTorch v2.11.0; installed 2.14.1+cpu exact public docstring corroborates',
    'Official PyTorch repository docstring registered on public API.',
    'Read lines 766–795: default rtol=1e-5 and formula |input-other|<=atol+rtol|other|. Personally read same formula/default in installed API docs lines 69–98 and ran float64 counterexample. This original API inspection contradicts the manuscript’s absolute-only description; it is not derived from a prior report.')
original('s-tokenizer','official_source','Transformers tokenizer padding and attention_mask contract','https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/tokenization_utils_base.py','Transformers v4.57.1 tagged source; read-only reference, no installation/model load',
    'Official huggingface/transformers repository, explicit version tag and personally fetched raw source.',
    'Read return documentation 1317–1320 and _pad 3770–3845. Masks returned when requested or model_input_names includes attention_mask; constructed from original length as ones then zeros on the chosen padding side. Supports an established batching-tool pattern, not a universal guarantee that every tokenizer always returns a field literally named valid.')
for identifier,path,note in [('s-model','tiny_perceptron/model.py','Read ModelConfig 14–28 defaults: one layer/head, learned absolute positions, manual backend; TinyLM 53–86 token and position embeddings, default positions from physical arange, kwargs forwarding, dictionary logits; Block 43–50 residual and pointwise FFN; loss_sum 92–105 explicit ignore target and denominator. Public forward does not invoke optimizer/backward.'),
    ('s-attention','tiny_perceptron/attention.py','Read complete file 1–74: attention_mask 10–18 uses physical key<=query and valid[:,None,None,:], masking keys only; manual_attention 21–28 weights key-axis softmax, special empty-row zeros; CausalAttention 48–74 custom model position IDs do not change the physical causal axes; default backend manual, no dropout, bias-free output projection.'),
    ('s-data','tiny_perceptron/data.py','Read ByteTokenizer 14–25 pad_id=0 vs CharTokenizer 31–43 unknown ID0; IGNORE=-100 at line10; pad_batch 71–86 returns ids/labels/valid separately, right padding and validity by lengths, labels initialized IGNORE. Parenthetical lesson-5.2 contract also directly read.'),
    ('s-modern','tiny_perceptron/modern.py','Read DenseFFN 35–53: position-wise Linear/GELU/Linear in this default configuration; no cross-token operations outside attention. Only necessary supporting contract; advanced branches not treated as validated by the original fence.'),
    ('s-bootstrap','scripts/build_course.py','Read BOOTSTRAP lines 26–60: local notebook finds repository and inserts import path, sets one torch thread and seed; Colab installation branch is not executed here. Original helper worker actually used local bootstrap; guard_events empty.')]:
    p=ROOT/path
    sources.append({'id':identifier,'kind':'repository_code','title':path+' personally inspected actual implementation','path':path,'sha256':sha(p),
        'version':'Git HEAD 607dc4287d12a6043de59457785aa91bf13af331 plus exact checked file SHA; originals preserved in inputs/', 'verified':True,'inspection_note':note})
sources.extend([{'id':'s-execution','kind':'execution','title':'Unmodified section fence own CPU execution','verified':True,'artifact_id':'e-original'},
    {'id':'s-probe','kind':'execution','title':'Independent bounded CPU mechanism checks','verified':True,'artifact_id':'e-probe'},
    {'id':'s-derivation','kind':'derivation','title':'Padding-invariance and key-denominator derivation','verified':True,
     'details':'For contiguous padding with retained real tokens in order, physical causal predecessors among real tokens match the unpadded sequence. The same token vector plus the same learned position vector gives identical real hidden states initially; valid excludes padded keys at every attention layer. Key-axis softmax normalizes over the same permitted real keys, while LayerNorm and DenseFFN act at each token. Inductively retained real logits agree up to finite precision. With equal allowed logits the real rows have denominators 1,2,3 and weights 1, 1/2, 1/3. Separate float64 unequal-score check gives scores 0,1 after sqrt(2) division, denominator 1+e=3.718281828459045, weights .26894142137/.73105857863, forbidden key weight0. No empirical training conclusion follows.'}])

def evidence(source,locator,supports): return {'source_id':source,'locator':locator,'supports':supports}
claims=[]
def claim(identifier,kind,statement,location,scope,evidences,artifact_ids=None,verification=None,status='verified'):
    d={'id':identifier,'kind':kind,'statement':statement,'location':location,'scope':scope,'status':status,'evidence':evidences,'artifact_ids':artifact_ids or []}
    if verification: d['verification']=verification
    claims.append(d)
def numeric(expected,observed,details,tolerance='Absolute 1e-6 for float32 retained real logits; exact Boolean assertions'):
    return {'method':'executed','expected':expected,'observed':observed,'details':details,'tolerance':tolerance}

claim('c-invariance','concept','左PAD後，禁止讀PAD key並保留真實token位置0、1、2，可使本模型真實三格的logits與未PAD版相同。','course/chapters/07.md:219,240',
    '本節固定同一個未訓練 TinyLM，預設 learned absolute positions、manual attention、無dropout/cache；只比較非PAD三格。不是所有架構必須採取同一位置處理的普遍定理。',
    [evidence('s-paper','§3.2.1 Eq.(1), §3.3, §3.5 pp.4–6','attention按key取加權和，position encoding加到token表示；position-wise FFN本身不跨格混合。'),evidence('s-model','TinyLM.forward 68–86; Block 43–50','同token同position輸入及逐格residual/FFN。'),evidence('s-derivation','padding-invariance induction in details','物理位置偏移後，非PAD前驅集合与原序列相同，逐層等價。'),evidence('s-probe','equivalence.correct_max_abs; unequal_length_batch','原三格及有界不同長度變化數值核實。')],['e-original','e-probe'])
claim('c-mask-axes','concept','因果只禁止未來；valid另禁止填充key，需與causal同時成立，物理欄位與model position IDs不同。','course/chapters/07.md:221,240,244',
    'allowed [B,H,query,key]，valid沿最後key軸broadcast，沒有query-valid條件。causal用物理arange，positions只影響 learned table / optional RoPE。',
    [evidence('s-paper','§3.2.3 pp.5','causal非法連結於softmax前設-infinity。'),evidence('s-functional','SDPA 5948–5989,6065–6074','query/key軸與boolean許可語義。'),evidence('s-attention','10–18,64–70','實際AND遮罩與物理因果軸。'),evidence('s-probe','mask_axis_and_empty_queries.allowed/right_padding_allowed','leftPAD matrix精確核對、rightPAD無效query仍可讀真key。')],['e-probe'])
claim('c-original-api','software','原短碼固定seed建構TinyLM，建立ids/valid/positions，取得字典logits，以[:,2:]取最後三格，abs/max/item列印最大差；本例只前向比較。','course/chapters/07.md:226–237,240,246',
    '逐API組：manual_seed(42)固定初始化；ModelConfig(vocab_size10,width8)其餘defaults；tensor整數long、!=0得bool；model kwargs有效；[B,T,V] logits；位置slice取[1,3,10]；差的abs().max()是全30元素無單位logit差，item普通數字。無backward/step/既有權重load。',
    [evidence('s-model','14–28,53–86','建構配置、kwargs和dict/logitscontract。'),evidence('s-sparse','Embedding doc Shape 38–42 and defaults 134–143','ids/positions為lookup indices。'),evidence('s-bootstrap','BOOTSTRAP 26–60','本地import及CPU bootstrap，無Colab branch。'),evidence('s-execution','original-run/fence-1.py, stdout/environment/execution','逐字原fence親跑exit0。'),evidence('s-probe','equivalence, model_parameters_before/after_sha256','shape、dtype、切片、參數hash不變及所有grad None。')],['e-original','e-probe','a-installed'],
    {'method':'executed','expected':'原fence能完成；base/actual均[1,3,10]，全30分數最大差接近0。','observed':'Exit0，最大差0.0；padded全輸出[1,5,10]；參數SHA前後同為77401875fc078338a82eb11c51222c478b0bfd4d84e4fd6187ae3c10c69941ec。','details':'API public docstrings由實際installed2.14.1+cpu親讀；官方source pinned2.11.0是不同版本。原程式沒有更新參數，probe也沒有。'})
claim('c-number','numeric','有效位置最大差應為0或極小浮點誤差。','course/chapters/07.md:236,240',
    '輸出1個batch×3個real tokens×10候選logits的逐元素最大絕對差。沒有token accuracy、資料集或訓練測量。',
    [evidence('s-execution','stdout.txt','原碼最大差0.0。'),evidence('s-probe','equivalence.correct_max_abs; unequal_length_batch','有界batch版本最大差2.384185791015625e-7。')],['e-original','e-probe'],numeric('float32差<=1e-6','原例0.0；兩筆不同長度、左右PAD範例1.7881393432617188e-7與2.384185791015625e-7','相同固定模型，真實格數3與2，物理長度5，結果不拿來支持訓練能力。'))
claim('c-tolerance','numeric','「檢查容許百萬分之一的絕對差」，對應原 assert torch.allclose(base, actual, atol=1e-6)。','course/chapters/07.md:237,240',
    '這一項原主張有矛盾，保留原句與原碼待協調者修正。實際API是混合atol+rtol，而非只容許max absolute error<=1e-6。',
    [evidence('s-allclose','_torch_docs.py 766–795; installed-api-docstrings.txt 69–98','rtol預設1e-5，正式條件|input-other|<=atol+rtol|other|。'),evidence('s-probe','allclose_tolerance_counterexample','1.0/1.000002差約2e-6仍被原call接受；rtol0才拒絕。')],['e-probe'],numeric('若原句成立，2e-6差應被拒絕','原 atol1e-6呼叫True；加rtol0則False；实际容忍1.100002e-5','請在原assert加rtol=0.0以符合原正文，再由本reviewer實際複查。','原宣稱絕對1e-6；實際atol1e-6+rtol1e-5|other|'),status='contradicted')
claim('c-pad-convention','concept','ID0在本例專供PAD，ids!=0才可推出valid；其他字表不能一概將0當空格，batch工具可直接供給有效遮罩。','course/chapters/07.md:242',
    '本例ID0為自訂input padding約定；ByteTokenizer pad_id0但CharTokenizer unknown_id0。提供valid是repo pad_batch及Transformers return_attention_mask的具體支持，不聲稱所有工具同名欄位/總會返回。',
    [evidence('s-sparse','14–42,134–143','Embedding預設不給0特殊忽略語義。'),evidence('s-tokenizer','_pad 3770–3845, return docs1317–1320','由長度和padding側生成0/1attention_mask，按請求或model defaults返回。'),evidence('s-data','10,18,31–43,71–86','PAD/IGNORE/UNK與repo三份batch輸出的實際約定。'),evidence('s-probe','batch_helper_contract','親核pad_batch ids/labels/valid及CharTokenizer未知字0。')],['e-probe'])
claim('c-empty','concept','最前PAD query沒有可讀key；全-infinity softmax會NaN，本工具把這種空讀取的attention輸出設0。','course/chapters/07.md:244',
    '只涉及本例manual_attention的權重與attention讀取結果。不是整個TinyLM PAD logits為0；key-valid也不會禁止無效query去讀較早真key（右PAD情況）。',
    [evidence('s-functional','softmax 2127–2139','softmax分母sum(exp)；全-inf沒有正分母，須另外定義空行。'),evidence('s-attention','manual_attention21–28','先把空row分數改0，softmax後再將空row權重設0，故weights@v=0。'),evidence('s-probe','mask_axis_and_empty_queries; equivalence.pad_logits_max_abs','原生全-inf全NaN；guard後空前兩row output/weights0；PAD logits max .9720391035。')],['e-probe'])
claim('c-softmax-denominator','numeric','combined mask排除PAD與future key，softmax在key軸只歸一化可讀key；空行是特殊零讀取。','course/chapters/07.md:221,244（必要實作核對）',
    '為核對本節機制的獨立算例，沒有將新增數字冒充教材原測量。shape[B,H,Q,K]、score divisor sqrt(head dim)、probability無單位、加權輸出與V同單位。',
    [evidence('s-paper','§3.2.1 Eq.(1)','QKᵀ/sqrt(dk),softmax,V。'),evidence('s-functional','softmax formula2127–2139','分子exp分數，分母同key slice求和。'),evidence('s-derivation','equal-key counts and unequal-score calculation','1/2/3key分母與1+e算例。'),evidence('s-probe','mask_axis_and_empty_queries; scaled_softmax_calculation','親跑float32零score行與float64非均勻score行。')],['e-probe'],numeric('allowed key數0,0,1,2,3；有效weights sum1，空rows sum0；float64 weights [1/(1+e),e/(1+e),0]','精確counts0,0,1,2,3；real row sums1；denominator3.718281828459045；output17.31058578630005','query1 vs3 keys,d_k2，score0/1/非法，最後key=999被排除。','key counts/booleans精確；float32weights1e-7；float64非均勻weights絕對1e-12'))
claim('c-loss-separate','concept','PAD對應labels=-100使它們不計入直接答案代價；valid遮罩可讀key是另一件事，本例没有計算loss。','course/chapters/07.md:244; 5.2 cross-reference',
    'ignore_index僅class-index target的loss及直接logit-gradient規則；不自動移除embedding/attention輸入。必要probe新增loss僅作對比，不冒充原fence有loss。',
    [evidence('s-functional','cross_entropy3414–3455','ignore_index targets不貢獻代價/直接輸入logit梯度。'),evidence('s-model','loss_sum92–105','有效target計數及sum/count分開。'),evidence('s-data','pad_batch71–86','inputPAD、labelIGNORE、valid三個欄位分離。'),evidence('s-probe','loss_attention_separation','忽略兩個PAD label分母仍3；不傳valid即使labelIGNORE也會影響真實logits/loss。')],['e-original','e-probe'])
claim('c-exercise','software','只刪valid或只刪positions，維持同seed、同模型，本例各自破壞左PAD等價，練習assert失敗。','course/chapters/07.md:244,246',
    '限本例學得絕對位置且左PAD在past、隨機模型seed42。不是說右PADreal輸出也必然改變，亦不適用宣稱所有相對位置架構都有同種差。',
    [evidence('s-model','76–79','不提供positions則用物理0..4，real位置由0..2變2..4。'),evidence('s-attention','10–18,21–28','不提供valid則leftPAD過去key會進softmax。'),evidence('s-probe','equivalence.without_valid/without_positions','各一次只改呼叫參數，assertallclose分別false。')],['e-probe'],
    {'method':'executed','expected':'兩種原練習均比原close門檻大且assert失敗。','observed':'刪valid差0.22275620698928833；刪positions差1.0177843570709229；兩者allcloseFalse。','details':'同模型未重初始化，比較float32全30logits元素；rightPAD去掉兩kwargs差0.0，僅作邊界核對。'})
claim('c-figure','numeric','圖與原例一致：未PAD ids1,2,3模型位置0,1,2；左補兩PAD物理欄0..4，valid假假真真真，real位置仍0,1,2。','course/figures/rewrite-07-07-padding-positions.svg; course/chapters/07.md:223,240',
    'PAD模型位置畫—表示不作真實對應；原程式仍傳0並實際lookup，未宣稱PAD沒有內部向量。無數據流箭頭或測量分數可另外核對。',
    [evidence('s-execution','original-run/fence-1.py lines ids/valid/positions','原數列與valid/positions與圖對應。'),evidence('s-probe','equivalence.valid/positions,changed_pad_position_real_max_abs','PAD位置0改8/9不改retained real logits，但內部PADlookup仍存在。')],['a-figure','e-original','e-probe'],numeric('ids/valid/realpositions精確一致，畫面标签/色框可讀','親閱1280px Inkscape PNG，物理/valid/model三行與實際数组一致，無截字；PAD兩格—語義與正文「不用於真實對應」一致','實際render並view_image；不是只搜尋SVG字串。','離散數列精確對應；render尺寸1280×1540，沒有對圖像數值作浮點比較。'))

report={'schema_version':1,'review_stage':'technical','lesson_id':'7.7','source':'course/chapters/07.md#7.7','source_sha256':extraction['source_sha256'],
    'figure_sha256':extraction['figure_sha256'],'reviewer_task':'/root/phase4_factual_coordinator/factual_7_7','reviewer_context':'fresh','verdict':'revise',
    'scope':{'actual_read':['course/chapters/07.md#7.7 original raw bytes','course/chapters/07.md#7.5–7.6 necessary preceding context','course/chapters/05.md#5.2 linked loss contract','course/training.md heading/command inventory and setup preamble only; recipes not executed','tiny_perceptron/model.py complete','tiny_perceptron/attention.py complete','tiny_perceptron/data.py1–86','tiny_perceptron/modern.py1–53','scripts/build_course.py BOOTSTRAP','four required review-method/checker files','own official snapshots and exact locations listed per source','original referenced SVG plus actual rendered PNG viewed'],
        'chapter_intro':{'status':'not_applicable','reason':'7.7不是章首第一節；未做章首導言審閱。'},'independence':'Fresh single-section technical reviewer; no prior technical/reader report body or judgment read. Prior 7.7 JSON was copied only as opaque bytes to history without decoding; original-paper locator used only for PDF path/hash, official version/authority personally checked. No subagent spawned.',
        'historical_measurements':'not_applicable: 本節只有未訓練模型前向數值，無歷史train/accuracy/step指標可核；沒有重訓、既有checkpoint評測、課程資料準備、模型下載、GPU、付費操作。',
        'recipe_boundary':'Only original short fence and bounded CPU probes executed. Linked long training/download recipes are not prerequisites: TinyLM random initialization and explicit tiny tensor inputs use no external weights/data. Original bootstrap used local import, not the Colab clone/install branch. No environment changes required.',
        'remaining_limits':'This is technical evidence and figure inspection, not a new first-reader session or browser desktop/mobile page validation. Installed torch2.14.1+cpu differs from official v2.11.0 snapshots; APIs relevant here were checked against installed docs and executed. No cache/rotary/dropout claims validated by original example.'},
    'sources':sources,'artifacts':artifacts,'claims':claims,
    'issues':[{'id':'i-allclose-absolute','status':'unresolved','original_claim':'檢查容許百萬分之一的絕對差','original_code':'assert torch.allclose(base, actual, atol=1e-6)','location':'course/chapters/07.md:237,240',
        'evidence':'官方allclose公式包含default rtol1e-5；親跑差約2e-6原call True,rtol0 False。','impact':'初學者會把atol參數誤當assert對最大絕對差的唯一上限；本例差0仍成功，不會揭露這個契約誤解。',
        'recommendation':'在原assert加rtol=0.0以符合正文的絕對1e-6規則；或明確寫混合公式。由協調者修正後，我親跑新fence再更新自己報告，保留原问题與證據。','resolution':'尚未修正或複查；不得因schema要求pass而刪去矛盾claim。'}],
    'checks':{'factual_accuracy':{'status':'revise','details':'核心PAD/attention/positions/loss機制核實；allclose精度文字與實際API有矛盾，原claim保留為contradicted。','claim_ids':[c['id'] for c in claims]},
        'numeric_verification':{'status':'revise','details':'原fence0.0，兩練習差0.2227562/1.0177844；key分母与空行及有界batch已實算。absolute-only容忍差主張未通過。','claim_ids':['c-number','c-tolerance','c-softmax-denominator','c-figure']},
        'figure_consistency':{'status':'pass','details':'原SVG已實際Inkscape render及view；物理列、real位置、valid與原例逐項一致。','claim_ids':['c-figure']},
        'source_verification':{'status':'pass','details':'原論文PDFidentifier與arXivofficialabs/version/author親核；官方PyTorch/Transformers固定tag原source親讀，不把來源庫或摘要當查證。runtime版本差明記。','claim_ids':['c-invariance','c-mask-axes','c-pad-convention','c-empty','c-loss-separate','c-tolerance']},
        'limitations':{'status':'pass','details':'original only前向、未更新，限定左PAD与本模型。rightPAD/unequal length探針揭示邊界，不推訓練能力；空attention0不等於PADlogits0。無原歷史測量/章intro適用。','claim_ids':['c-original-api','c-invariance','c-empty','c-exercise']}}}
(ROOT/'docs/technical-reviews/7.7.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
manifest=[{'path':rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='manifest.json']
(ART/'manifest.json').write_text(json.dumps({'reviewer_task':report['reviewer_task'],'source_sha256':report['source_sha256'],'files':manifest},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],'claims':len(claims),'artifacts':len(artifacts),'report_sha256':sha(ROOT/'docs/technical-reviews/7.7.json')},ensure_ascii=False))
