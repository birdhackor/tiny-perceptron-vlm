"""Assemble this owner's completed proof, retaining all original provisional records."""
import copy, datetime, hashlib, inspect, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from record import ROOT, BASE, MAN, TRACE, sha, write
sys.path.insert(0, str(ROOT))
from scripts import check_technical_reviews as schema
import torch

manifest=json.loads((ROOT/MAN).read_text())
group=next(g for g in manifest['groups'] if g['group']=='d')
rows=[json.loads(s) for s in (ROOT/TRACE).read_text().splitlines()]
assert rows[-1]['event']=='complete', rows[-1]
assert len(group['primary_page_ids'])==58
notes={pid:[r for r in rows if r.get('event')=='checkpoint' and r.get('page_id')==pid] for pid in group['primary_page_ids']}
assert all(notes.values())

paper_files={
 '2305.18290':'dpo','1707.06347':'ppo','1910.07467':'rmsnorm',
 '1606.08415':'gelu','2109.08668':'primer','1612.00796':'ewc',
 '2309.00071':'yarn','2310.13548':'sycophancy','2104.09864':'roformer',
 '2306.15595':'position-interpolation','2404.06654':'ruler',
 '1706.03762':'attention','2209.13085':'reward-hacking',
 '2002.05202':'glu','2010.04245':'qknorm','2310.03716':'length-rlhf',
 '1003.0146':'contextual-bandit','1608.05859':'weight-tying',
 '2203.02155':'instructgpt-v1'}
needles={'dpo':'implicit reward','ppo':'pessimistic','gelu':'xΦ','glu':'SwiGLU',
 'position-interpolation':'linearly','ruler':'effective','roformer':'relative',
 'attention':'positional encodings','gpt2':'probabilities','instructgpt-v1':'three steps',
 'reward-hacking':'optimizing','length-rlhf':'longer'}
cache=pathlib.Path('/tmp/p7-technical-d-source-cache')
url_cache={
 'cache_explanation.md':'cache-explanation.md','endpoint_request_func.py':'vllm-request.py',
 'scaled_dot_product_attention.html':'sdpa-2.14.html','checkpoint.html':'checkpoint-2.14.html',
 'cuda-gpus':'cuda-gpus.html','sdp_utils.cpp':'sdp_utils-pinned.cpp',
 'checkpoint.py#L435':'checkpoint-pinned.py','masking_utils.py':'masking_utils-v4.57.1.py',
 'flash-attention/blob/v2.8.3/README.md':'flash-readme-v2.8.3.md'}
torch_dir=pathlib.Path(inspect.getfile(torch)).parent
module_files={
 '/torch/__init__.py':'__init__.py','/torch/autograd/profiler_util.py':'autograd/profiler_util.py',
 '/torch/autograd/__init__.py':'autograd/__init__.py','/torch/nn/modules/module.py':'nn/modules/module.py',
 '/torch/nn/modules/batchnorm.py':'nn/modules/batchnorm.py','/torch/nn/modules/dropout.py':'nn/modules/dropout.py',
 '/torch/cuda/memory.py':'cuda/memory.py','/torch/amp/autocast_mode.py':'amp/autocast_mode.py',
 '/torch/amp/grad_scaler.py':'amp/grad_scaler.py','/torch/optim/adam.py':'optim/adam.py'}

index=json.loads((BASE/'receipt-timing-recheck-index-20261007T0228.json').read_text())
format_notes=[]
pages=[]
for pid in group['primary_page_ids']:
 p=copy.deepcopy(json.loads((BASE/'pages'/f'{pid}.json').read_text()))
 p['provisional_proof_ref']={'path':str((BASE/'pages'/f'{pid}.json').relative_to(ROOT)),'sha256':sha((BASE/'pages'/f'{pid}.json').relative_to(ROOT))}
 if not p['missing_visuals']:
  p['missing_visuals']=notes[pid][-1]['missing_visuals']
  format_notes.append(f'{pid}: empty missing_visuals filled with this owner final actual checkpoint judgment')
 if isinstance(p['variation'],str):
  p['variation']={'change':p['variation'],'prediction':notes[pid][0]['expected_change'],'reason':notes[pid][0]['understanding']}
  format_notes.append(f'{pid}: variation text structured using actual own checkpoint prediction/understanding')
 for check in p['checks'].values():
  if check['status']=='verified':
   check['status']='pass';format_notes.append(f'{pid}: technical check spelling verified→pass (claim verified retained)')
 for s in p['sources']:
  if s['kind']=='official_document':
   s['kind']='official_source' if s['url'].endswith('.py') else 'official_docs'
   format_notes.append(f'{pid}/{s["id"]}: official_document schema spelling corrected')
  if s['kind'] in {'official_docs','official_source'}:
   if not s.get('authority_reason'):
    s['authority_reason']='本人核對官方項目原文/原始碼或其安裝版本隨附原docstring，實際支持範圍依inspection_note。'
    format_notes.append(f'{pid}/{s["id"]}: official authority reason recorded explicitly')
   if not s.get('accessed_on'):
    s['accessed_on']='2026-10-07'
    format_notes.append(f'{pid}/{s["id"]}: actual UTC reading date recorded')
  if s['kind']=='repository_code' and not s.get('version'):
   s['version']='reviewed repository bytes SHA256 '+s['sha256']
   format_notes.append(f'{pid}/{s["id"]}: actual file SHA identifies version; no invented commit')
  if s['kind']=='paper' and not s.get('original_sha256'):
   key=next((k for k in paper_files if k in s['url']),None)
   name=paper_files[key] if key else ('gpt2' if 'language_models_are_unsupervised' in s['url'] else None)
   assert name,(pid,s)
   pdf=BASE/(name+'-paper.pdf');s['original_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
   s['original_format']='original PDF; complete PDF/text is local cache, not a required CI artifact'
   if name in needles:
    raw=(BASE/(name+'-paper.txt')).read_text();i=raw.lower().find(needles[name].lower());assert i>=0
    s['necessary_excerpt']={'locator':'original extracted text character offset '+str(max(0,i-50)),
      'text':raw[max(0,i-50):i+180].replace('\n',' '),
      'scope':'Only this small original excerpt is retained; claim evidence locators and inspected support scope remain authoritative.'}
  elif s['kind'] in {'official_docs','official_source'} and not s.get('original_sha256'):
   url=s['url'];local=next((cache/n for suffix,n in url_cache.items() if suffix in url),None)
   if local is None: local=next((torch_dir/n for suffix,n in module_files.items() if suffix in url),None)
   if local is not None:
    s['original_sha256']=hashlib.sha256(local.read_bytes()).hexdigest()
    s['original_format']='official original file bytes (local cache/installed version); not a required full-source CI artifact'
   else:
    docs=[('torch.topk.html',torch.topk.__doc__),('torch.Tensor.index_add_.html',torch.Tensor.index_add_.__doc__),('torch.allclose.html',torch.allclose.__doc__)]
    doc=next((d for suffix,d in docs if suffix in url),None)
    assert doc is not None,(pid,s)
    s['original_sha256']=hashlib.sha256(doc.encode()).hexdigest()
    s['original_format']='official installed 2.14.1+cpu docstring UTF-8 (inspection_note explicitly records installed docstring, not fetched HTML)'
 for c in p['claims']:
  v=c.get('verification',{})
  if 'tolerance' in v and not isinstance(v['tolerance'],str):
   value=v['tolerance'];v['tolerance']='exact equality' if value==0 else 'absolute tolerance '+str(value)
   format_notes.append(f'{pid}/{c["id"]}: numeric tolerance expressed as required text')
 for a in p['artifacts']:
  if a.get('description')=='本人執行當前教材code，保存命令、真stdout/stderr、code SHA及Python版本；沒有模型訓練。':
   related=[c for c in p['claims'] if a['id'] in c['artifact_ids']]
   a['description']='本人CPU執行 '+a['id']+'；實際用途：'+'；'.join(c['statement']+'（'+c['scope']+'）' for c in related)
   if not related:a['description']+='保存原probe實際結果；沒有用它替未執行的訓練/量測背書。'
 for entry in index['entries']:
  if entry['page_id']==pid:
   p.setdefault('review_process_notes',[]).append({'correction_ref':{'path':str((BASE/'receipt-timing-corrections-20261007T0225.json').relative_to(ROOT)),'sha256':sha((BASE/'receipt-timing-corrections-20261007T0225.json').relative_to(ROOT))},'actual_recheck_receipt':entry['actual_recheck_receipt'],'scope':'Original premature checkpoint retained; actual later reading/viewing supports final evidence.'})
   if entry.get('figure'):
    old=next(v for v in p['visual_checks'] if v['figure']==entry['figure'])
    new=copy.deepcopy(old);new['receipt']=entry['actual_recheck_receipt'];new['details']='本人後續獨立重新實看640/360工具完整返回後另記真receipt；原visual與checkpoint保留。'
    p['visual_checks'].append(new)
 if pid=='16.13':
  p.setdefault('review_process_notes',[]).append({'correction_ref':{'path':str((BASE/'16.13-source-timing-correction.json').relative_to(ROOT)),'sha256':sha((BASE/'16.13-source-timing-correction.json').relative_to(ROOT))},'scope':'README numerical-test source note preceded same exec return; later actual read completed at recorded time; no Flash test execution claim.'})
 pages.append(p)

errors=[]
for p in pages:
 e=[];a=schema._artifacts(ROOT,p['artifacts'],e);s=schema._sources(ROOT,p['sources'],a,e);schema._claims(p['claims'],s,a,e)
 errors.extend((p['page_id'],x) for x in e)
if errors:
 print(json.dumps({'metadata_errors':errors},ensure_ascii=False));raise SystemExit(1)

process_files=[]
for f in BASE.glob('*.json'):
 d=json.loads(f.read_text())
 if any(s in f.name for s in ('failed','correction','timing-recheck')) or ('exit_code' in d and d['exit_code']!=0):
  process_files.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f.relative_to(ROOT))})

report={
 'schema_version':1,'review_policy':'phase7_grouped','batch_id':'phase7','stage':'technical','group':'d',
 'reviewer_task':'/root/p7_technical_d','reviewer_context':'fresh','manifest_sha256':sha(MAN),'verdict':'pass',
 'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'trace_files':[{'path':TRACE,'sha256':sha(TRACE)}],
 'session':rows[0]['session'],'primary_pages_completed':58,
 'actual_context_pages':rows[0]['context_page_ids'],
 'issues':[],
 'review_scope':'本人以原session current-only逐段、自己的五欄checkpoint完成58primary+12.14–12.16必要context。原論文/官方原文件/原始碼與原measurementJSON核對；相稱CPU例/變化/核算與必要圖640/360及兩viewport實看。未讀他人review/作者研究摘要作答案，未改教材。',
 'review_process_limitations':[
  '多次同一owner恢復，初始fresh fork-none起點與原trace完整保留；最新恢復摘要停點落後實際trace，先current-only/必要past-unit-only及本人proof核實後接續，沒有重造舊checkpoint或猜測未保存toolhistory。',
  '13.17 unit1、14.1 unit0/unit2、14.2 unit2之圖觀看及14.8 unit0之PI原文措辭曾在同exec tool返回前提前。原記錄保留；後續真的重新看/讀並以獨立實際時間receipt記錄，final evidence依後續實際支持範圍。',
  '16.13 README Tests段的正式source note也先於同exec返回；後於03:47:24 UTC前實際讀完并保存更正。只支持容差語義，沒有實跑Flash測試。',
  '截斷的工具輸出/圖像不算已讀/看；必要圖像以後續完整返回的本人觀看及receipt為準，版面只聲稱保存截圖所涵蓋位置，不聲稱全頁所有scroll都看。',
  '自己的錯誤預測、欄位猜測、環境/import/metadata probe失敗均保留。13.2 byteIDs、13.6 failed parameters、13.11 bandit locator、13.15 action、14.2 median、15.13 8expert144/18等在後續自己checkpoint更正；正式claims引用真成功核算，不把審閱工具錯誤當教材問題。',
  '沒有重跑長訓練、每版實驗或GPUbenchmark；歷史成績僅核原JSON/程式/版本與可核算分母、ID/樣本/單位。未保存逐次latency的組，只核storedmedian和單位，不宣稱本人重算median。',
  'UF100人工作品源asset不可得時，證據限於實際程式/原統計，不代替人類原記錄；單seed、小模型、有限答案卡、短序列及分派等限制按各頁保存，不外推大型品質/速度。',
  '無圖且無必要layout頁明列unverified required:false；仍完成文字/程式核實。完整外部PDF/全文text只作localcache，正式不列為CI必需artifact；保留原URL/version/SHA/定位與必要短摘錄。',
  '單組group-preflight僅核metadata/evidencecontracts，不證明理解、圖片實看、scientifictruth或全stage完成。'],
 'process_record_refs':sorted(process_files,key=lambda x:x['path']),
 'format_normalizations':format_notes,
 'pages':pages}
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
path='docs/technical-reviews/artifacts/p7_technical_d/report-freeze03-'+stamp+'.json'
write(path,report)
print(json.dumps({'report':path,'sha256':sha(path),'pages':len(pages),'claims':sum(len(p['claims']) for p in pages),'verdict':report['verdict'],'process_refs':len(process_files),'metadata_lint':'passed (not truth)'},ensure_ascii=False))
