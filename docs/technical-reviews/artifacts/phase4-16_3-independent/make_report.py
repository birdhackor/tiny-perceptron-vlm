import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = Path(__file__).resolve().parent.relative_to(ROOT)
def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
meta = json.loads((ROOT / P / 'input/extraction.json').read_bytes())
env = {'python': '3.13.5', 'torch': '2.14.1+cpu', 'device': 'cpu', 'cuda_build': 'None', 'default_dtype': 'torch.float32'}
artifacts = []
def artifact(identifier, filename, kind, description, **kwargs):
    path = P / filename
    artifacts.append(dict(id=identifier, path=path.as_posix(), sha256=digest(path), kind=kind, description=description, **kwargs))

for identifier, filename, kind, description in [
    ('section', 'input/section.md', 'source_snapshot', 'Exact original UTF-8 lesson16.3 bytes.'),
    ('chapter-frozen', 'input/chapter16-frozen-input.md', 'source_snapshot', 'Full chapter frozen at first extraction; actual saved bytes, not a later whole-chapter version or introduction-review claim.'),
    ('original-fence', 'input/fence-1.py', 'code', 'Unchanged original Python fence.'),
    ('bootstrap', 'input/bootstrap.py', 'code', 'Actual helper bootstrap used for unchanged fence.'),
    ('extraction', 'input/extraction.json', 'source_snapshot', 'Original helper extraction metadata with source/fence/figure hashes.'),
    ('original-execution', 'execution/original-execution.json', 'source_snapshot', 'Actual helper command, cwd, timeout, exit0 and artifact hashes.'),
    ('original-environment', 'execution/original-environment.json', 'source_snapshot', 'Actual CPU helper environment and imported-code hashes.'),
    ('original-stderr', 'execution/original-stderr.txt', 'source_snapshot', 'Actual original-fence stderr, empty.'),
    ('cpu-code', 'cpu_checks.py', 'code', 'Bounded forward variants, near-tie calculation and selected existing raw-measurement arithmetic/Git provenance.'),
    ('cpu-metadata', 'execution/cpu-checks.execution.json', 'source_snapshot', 'Actual bounded CPU command, timeout, exit0, versions, code/stdout/stderr hashes.'),
    ('cpu-stderr', 'execution/cpu-checks.stderr.txt', 'source_snapshot', 'Actual bounded CPU stderr, empty.'),
    ('raw-efficiency', 'primary/efficiency-original.json', 'source_snapshot', 'Full untouched original JSON; only selected logged pointers inspected.'),
    ('raw-architecture', 'code/architecture-at-raw-revision.py', 'code', 'Immutable original architecture code independently matched to raw JSON hash and Git48a4f3e.'),
    ('model-code', 'code/model.py', 'code', 'TinyLM/ModelConfig/Block bytes matched to original raw Git/code hash.'),
    ('attention-code', 'code/attention.py', 'code', 'Mask/attention/cache bytes matched to original raw Git/code hash.'),
    ('ffn-code', 'code/modern.py', 'code', 'Frozen source; only DenseFFN35–53 read.'),
    ('tokenizer-code', 'code/data.py', 'code', 'ByteTokenizer/SPECIALS bytes matched to original raw Git/code hash.'),
    ('figure-raw', 'input/rewrite-16-cache-append.svg', 'source_snapshot', 'Original referenced SVG bytes.'),
    ('figure-render', 'execution/cache-append-inkscape.png', 'figure_render', 'Actually rendered640×694 PNG personally viewed; saved0/1/2, appended3, Query reads four.'),
    ('figure-execution', 'execution/figure-execution.json', 'source_snapshot', 'One bounded Chromium timeout, successful Inkscape fallback, versions and visual scope; no desktop/mobile page verification.'),
    ('figure-stderr', 'execution/inkscape-stderr.txt', 'source_snapshot', 'Actual Inkscape stderr.'),
    ('inspection', 'inspection.json', 'source_snapshot', 'Actual reading scope, raw pointers and honest independent limitations.'),
    ('primary-inspections', 'primary-inspections.json', 'source_snapshot', 'Personally read original sources: HTTPS, version, access date, authority, locators, support and original SHA.'),
    ('report-generator', 'make_report.py', 'code', 'This reviewer’s report generation code; no old canonical report content read.'),
]:
    artifact(identifier, filename, kind, description)
artifact('original-run', 'execution/original-stdout.txt', 'execution', 'Unchanged original fence stdout.', command='.venv/bin/python docs/review-tools/section_facts.py course/chapters/16.md#16.3 --output /tmp/phase4-16_3-independent-run --execute --timeout 40', result='exit0; shapes(1,264)/(1,264); max absolute error2.384185791015625e-7; assert passed.', environment=env)
artifact('cpu-run', 'execution/cpu-checks.stdout.txt', 'execution', 'Actual bounded CPU forward/measurement checks stdout.', command='.venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-independent/cpu_checks.py', result='exit0; all bounded assertions passed; zero optimizer updates; weights unchanged; raw12-ID arrays equal and max1.9073486328125e-6 per model.', environment=env)

sources = []
for row in json.loads((ROOT/P/'primary-inspections.json').read_bytes())['sources']:
    sid = row['id']
    artifact('primary-'+sid, row['snapshot'], 'source_snapshot', row['locator']+'; original-source snapshot.')
    sources.append(dict(id=sid, kind='paper' if sid=='gqa-paper' else ('official_docs' if sid in {'hf-cache', 'torch-numeric'} else 'official_source'), title={'hf-cache':'Transformers Caching','torch-allclose':'PyTorch allclose API original documentation','torch-eval':'PyTorch Module.eval/train','torch-no-grad':'PyTorch no_grad','torch-numeric':'PyTorch Numerical accuracy','gqa-paper':'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints'}[sid], url=row['url'], version=row['version'], accessed_on=row['accessed_on'], authority_reason=row['authority_reason'], verified=True, checked_original=True, inspection_note=row['locator']+'. '+row['supports'], snapshot_artifact_id='primary-'+sid, original_sha256=row['original_sha256']))
for sid, filename, title, locator, version in [
    ('tiny-model', 'code/model.py', 'TinyLM/ModelConfig/Block', 'ModelConfig15–28, Block31–50, TinyLM53–89; logits/cache and offset contract.', 'Original Git48a4f3e; same bytes as current inspected source'),
    ('tiny-attention', 'code/attention.py', 'CausalAttention and masks', 'AST-located then read10–74; causal mask, projection, concat dim2, unexpanded cache and explicit padding rejection.', 'Original Git48a4f3e; same bytes as current inspected source'),
    ('tiny-ffn', 'code/modern.py', 'DenseFFN', 'AST-located DenseFFN35–53; two Linear layers and last-axis activation.', 'Current source frozen at review'),
    ('byte-tokenizer', 'code/data.py', 'ByteTokenizer/SPECIALS', 'SPECIALS11 and ByteTokenizer14–28; BOS1, EOS2, PAD0; decode excludes ID<8.', 'Original Git48a4f3e; same bytes as current inspected source'),
    ('experiment-method', 'code/architecture-at-raw-revision.py', 'Original efficiency cache method', 'AST-located _fixed_decode514–521, _cache_probe525–555, _copy_matching51–63, _clone_config66–69, _runtime97–103, _amp111–112, _train signature/computation197–206, run_efficiency916–935. Fixed steps/no EOS break; shared-reference score probe; cache call without autocast.', 'Git48a4f3e912b483d70aee57c42c2aac226534a9a6; independently verified original JSON hash'),
    ('raw-measurements', 'primary/efficiency-original.json', 'Original efficiency measurements', 'Only logged measurement/config/provenance pointers read; counts/arrays/maxima/EOS/decode recalculated; no author result explanation read.', 'Original revision48a4f3e912b483d70aee57c42c2aac226534a9a6; full raw SHA d4bae2fdfab076fe76a65b3555b1a9347a562e1748406c7afc101266e4ee1615'),
]:
    path=P/filename
    sources.append(dict(id=sid, kind='repository_code', title=title, path=path.as_posix(), sha256=digest(path), version=version, verified=True, inspection_note=locator))
for sid, aid, title in [('executed-original','original-run','Unchanged original lesson fence'), ('executed-bounded','cpu-run','Independent bounded CPU and raw-measurement checks')]:
    sources.append(dict(id=sid,kind='execution',title=title,verified=True,artifact_id=aid))

def evidence(source, locator, supports):
    return dict(source_id=source, locator=locator, supports=supports)
def verification(expected, observed, details, **kwargs):
    return dict(method='executed', expected=expected, observed=observed, details=details, **kwargs)
def claim(identifier, kind, statement, location, scope, references, aids, check=None):
    row=dict(id=identifier, kind=kind, statement=statement, location=location, scope=scope, status='verified', evidence=references, artifact_ids=aids)
    if check is not None: row['verification']=check
    return row
E=evidence
claims=[
claim('causal-cache','concept','固定前文、權重、位置規則與推理條件下，因果注意力可重用各層舊K/V；新token仍查表、計Q/K/V、對全部可見K/V注意力及逐位置FFN。','16.3 lines103–109及圖','固定長度deterministic eval、手寫全因果TinyLM；不宣稱免除新位置工作或任意位置政策皆可沿用。',[
 E('hf-cache','Attention matrices38–73; Cache storage93–109','因果旧狀態不受future token改變；每層append新K/V，新Query讀past+current。'),
 E('tiny-model','Block43–50; TinyLM68–86','新位置仍經查表、attention、FFN及output。'),
 E('tiny-attention','CausalAttention48–74; attention_mask10–18','Q/K/V投影、cache concat第2維及全部可見K/V加權。'),
 E('tiny-ffn','DenseFFN35–53','兩層Linear逐位置最後特徵軸轉換。')],['model-code','attention-code','ffn-code','figure-render']),
claim('original-fence-api','software','原fence比較讀完ID4的full最後位置與cached新位置0，兩路shape(1,264)，assert通過；eval/no_grad/allclose依契約運作，未更新參數。','16.3 Python fence112–128及line130','coverage: TinyLM(ModelConfig(width=8)),eval(),torch.no_grad(),ids[:,:3]/[:,3:],logits[:,-1]/[:,0],prefix cache,abs/max/item,allclose(atol1e-6/default rtol1e-5)；固定seed0CPU推理。',[
 E('tiny-model','ModelConfig15–28; TinyLM68–86','vocab264、logits位置軸及cache return契約。'),
 E('torch-eval','Module.eval2916–2932','eval=train(False)，回傳self。'),
 E('torch-no-grad','no_grad22–86','暫停本次reverse-mode gradient記錄。'),
 E('torch-allclose','allclose834–866','abs(input-other)<=atol+rtol*abs(other)，default rtol1e-5。'),
 E('executed-original','original stdout/environment/execution','unchanged fence exit0及實測shape/error。'),
 E('executed-bounded','variants in stdout','無梯度記錄，權重前後torch.equal，optimizer_updates0。')],['original-fence','bootstrap','original-run','original-execution','original-environment','cpu-run'],verification('shape(1,264)/(1,264)，allclose assert pass且無update','shape(1,264)/(1,264)，max absolute error2.384185791015625e-7，exit0；weights unchanged','實際allclose含default rtol1e-5；這次最大絕對差也小於1e-6。兩路都讀完ID4預測next token；不比較只讀ID3的prefill分數。')),
claim('position-and-prefix','software','cache三格使新token位置從3開始；只換最後ID5可沿用prefix，改第一ID須重建；舊cache依賴原前文、權重與位置計算。','16.3 lines132、136','固定evalCPU變體；權重及位置政策依賴由embedding/projection契約核對，未測所有cache政策。',[
 E('tiny-model','TinyLM73–80','offset=cache[0][0].shape[2]，position embedding及舊狀態依原計算。'),
 E('tiny-attention','CausalAttention54–65','K/V由原x與projection得到，queries與append offset對齐。'),
 E('hf-cache','Cache class83–91','reuse cache必須對齐位置和mask。'),
 E('executed-bounded','variants stdout','last-ID及rebuilt-prefix接近；stale prefix和錯offset不同。')],['cpu-code','cpu-run','model-code','attention-code'],verification('last ID5沿用prefix仍接近；更改prefix重建仍接近；錯offset/stale cache可能不同','last-ID/rebuild error均2.384185791015625e-7；wrong-position1.2473084926605225；stale-prefix0.10371372103691101','有界forward-only；舊prefix K追加future ID後差0。僅所測變體，不是任意提示保證。')),
claim('padding-limit','software','PAD為專用佔位ID，不是真實前文；此TinyLM cache不處理同batch不同有效長度。','16.3 line134','ByteTokenizer PAD0和CausalAttention單長度cache契約；未擴充變長batch cache。',[
 E('byte-tokenizer','SPECIALS11; ByteTokenizer14–18','PAD是專用special ID0。'),
 E('tiny-attention','attention_mask14–15; CausalAttention59–61','valid排除PAD；cache帶valid/segments會ValueError，未保存per-row有效長度。'),
 E('executed-bounded','variants.padding_cache_rejection','[1,2,3]/[4,5,PAD] cache decode實際拒絕valid mask。')],['cpu-run','attention-code','tokenizer-code'],verification('padding valid mask配cache明確拒絕','ValueError:最小 cache 路線只接受無 padding、無 packing 的單長度 batch','B2，prefix三格，第二列有效長度2；沒有宣稱該情況正確。')),
claim('query-kv-heads','concept','四頭MHA各Query頭有自身K/V；另一設定四個Query頭共用一組K/V，是一個KV頭（GQA-1/MQA）。','16.3 line140前半','只核sharing定義及本實報config，不套用論文速度/品質。',[
 E('gqa-paper','§2.2 and Figure2, PDF p2','MHA H組Q/K/V；MQA共享單K/V；GQA-1=MQA。'),
 E('tiny-attention','CausalAttention32–46,64–69','K/V projection大小kv_heads*head_dim；cache未擴張，attention才repeat。'),
 E('raw-measurements','/results/models/{mha,gqa}/model/config','兩者heads4、kv_heads4/1、backend manual。')],['primary-gqa-paper','raw-efficiency','attention-code']),
claim('raw-cache-measurements','empirical','指定L4原件每模型25提示位置、12生成步；full/cached12個ID逐項相同，最高logit絕對差均1.90735e-6；EOS後仍續跑，decode字串隱去結構ID。','16.3 lines140–142；必要前文16.2測量條件','核既有原件，未重訓/跑GPU；每模型自己的兩路。誤差probe共用full-reference history，獨立greedy arrays亦相符；原件無完整raw logits，不補造。',[
 E('raw-measurements','/results/models/{mha,gqa}/cache/{prompt_tokens,generated_tokens,generated_ids_full,generated_ids_cached,generated_text,identical_greedy_ids,per_step_logit_max_error}; /results/runtime; /device,/gpu,/torch_version,/python_version','raw counts/arrays/maxima及設備版本；精確pointers見inspection。'),
 E('experiment-method','_fixed_decode514–521; _cache_probe525–555; run_efficiency923–935; _train dtype=None/_amp111–112','BOS+byte prompt，固定12步無EOS break；共同reference score probe，每模型自己的cache；FP32無autocast。'),
 E('byte-tokenizer','ByteTokenizer18–25','BOS1/EOS2及decode過濾ID<8。'),
 E('executed-bounded','mha/gqa records and raw_code_provenance stdout','独立核prompt25、12array长度、逐項IDs、max/rounding、EOS/decode及rawGit/SHA。')],['raw-efficiency','raw-architecture','cpu-code','cpu-run','tokenizer-code'],verification('每模型prompt25，12生成ID和12誤差，ID相同，max1.90735e-6<1e-5；字串eueeuare/eredquare','每模型max精確1.9073486328125e-6，round1.90735e-6；IDs12/12相符；EOS indices MHA[1,4,6,11]、一KV[1,5,11]；decode與原文相同','只核既有GPU數字；heads/kv_heads不同，未跨模型比logits。原torch2.14.1+cu126,L4；核算CPU2.14.1+cpu。',denominators={'models':2,'prompt_positions_including_BOS_per_model':25,'generated_ids_per_path_per_model':12,'per_step_error_maxima_per_model':12,'matching_ID_positions_per_model':'12/12','absolute_tolerance':1e-5})),
claim('floating-scope','concept','FP32兩路可在容差內但不逐bit相同；本次匹配不保證任意提示、精度或長度。','16.3 lines130、140','官方浮點限制和本節有限測試；不以CPU1e-6要求L4結果。',[
 E('torch-numeric','Numerical accuracy; Batched computations or slice computations, original1–38','有限精度/順序/平台/切片，數學相同不保證bit相同。'),
 E('torch-allclose','allclose834–866','指定逐元素容差不是bit比較。')],['primary-torch-numeric','primary-torch-allclose','original-run','cpu-run']),
claim('near-tie','numeric','最高分候選極接近，小浮點差可能改argmax；所以同時核logit誤差和原始生成ID。','16.3 line142末句','兩候選FP32算例展示可能性，未宣稱原GPU樣本曾改選擇。',[
 E('executed-bounded','near_tie stdout','[1,1+1e-7]/[1+1e-7,1]，FP32 max差1.1920928955078125e-7，allclose但argmax1/0。'),
 E('torch-allclose','allclose834–866','API約束分數差，不約束排序。')],['cpu-code','cpu-run'],verification('極小差可allclose且argmax不同','max差1.1920928955078125e-7；allclose true，argmax1/0','實際FP32CPU算例；只支持可能性。',tolerance='allclose(atol1e-6,default rtol1e-5)；argmax精確不同，无四捨五入容差。'))
]
report=dict(schema_version=1, review_stage='technical', lesson_id='16.3', source='course/chapters/16.md#16.3', source_sha256=meta['source_sha256'], reviewer_task='/root/phase4_factual_coordinator/factual_16_3', reviewer_context='fresh', reviewed_on='2026-10-05', verdict='pass', frozen_input={'path':(P/'input/chapter16-frozen-input.md').as_posix(),'sha256':digest(P/'input/chapter16-frozen-input.md'),'meaning':'Original full-chapter bytes frozen at initial extraction with actual snapshot; not current whole-chapter version or chapter-introduction-review claim.'}, actual_reading_scope='Entire16.3 and needed16.2; referencedSVG raw/render personally viewed; AST-selected code; selected original measurement/provenance pointers; original primary authorities. Detailed honest scope in inspection artifact. Not introduction reviewer.', figure_sha256=meta['figure_sha256'], artifacts=artifacts, sources=sources, claims=claims, issues=[], checks={
'factual_accuracy':{'status':'pass','details':'Causal/per-layer mechanism、new work、model/API/offset/head sharing/padding契約親核，原fence與變體一致。','claim_ids':['causal-cache','original-fence-api','position-and-prefix','padding-limit','query-kv-heads']},
'numeric_verification':{'status':'pass','details':'shape/error及原件12step counts/IDs/maxima/EOS/decode独立核算；1.9073486328125e-6 round1.90735e-6；near-tie實際執行。','claim_ids':['original-fence-api','raw-cache-measurements','near-tie']},
'figure_consistency':{'status':'pass','details':'Inkscape render後view_image親看：0/1/2保存、3append、length4、new Query讀四格。Chromium唯一25sec嘗試逾時；desktop/mobile整頁未驗證，無整頁視覺pass宣稱。','claim_ids':['causal-cache']},
'source_verification':{'status':'pass','details':'Transformersv4.57.1、PyTorchv2.14.1官方原件與GQAv3原PDF親讀/逐claim定位；GPU原碼和Git48a4f3e/rawJSON hash匹配；未用旧report或來源摘要答案。','claim_ids':[c['id'] for c in claims]},
'limitations':{'status':'pass','details':'无重訓/GPU/下載weights/data或完整模型成績重評；原測量只25位置12stepFP32，每模型自己的兩路。CPU bounded forward0updates。variablelengthpadding不支持；allclose保留defaultrelative tolerance。SVG有視覺核查，整頁未查。','claim_ids':['position-and-prefix','padding-limit','raw-cache-measurements','floating-scope','near-tie']}})
path=ROOT/'docs/technical-reviews/16.3.json'
# The old canonical file is overwritten without reading it.
path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
own=json.loads(path.read_bytes())
assert own['reviewer_task']=='/root/phase4_factual_coordinator/factual_16_3'
assert own['source_sha256']==meta['source_sha256']
print(json.dumps({'report':str(path.relative_to(ROOT)),'reviewer_task':own['reviewer_task'],'source_sha256':own['source_sha256'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verdict':own['verdict'],'claims':len(claims),'artifacts':len(artifacts)},ensure_ascii=False))
