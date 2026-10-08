from pathlib import Path
import hashlib,json,math,statistics
from datetime import datetime,timezone
repo=Path('/workspace/tiny-perceptron-vlm');b=repo/'docs/course-repair-20261008/reviews/freeze-01';e=b/'evidence/technical-architecture';m=json.loads((b/'manifest.json').read_text());tm=json.loads((b/'checks/technical-inputs-manifest.json').read_text());out={'at':datetime.now(timezone.utc).isoformat(),'command':'.venv/bin/python /workspace/work/tutorial-repair-20261008/technical-architecture-sources/integrity.py','note':'從既有原始JSON重算欄位一致性；未重新跑歷史模型、資料或L4訓練。','inputs':{},'numeric_checks':{}}
for name in ['modern','moe','efficiency','precision','qat']:
 p=f'docs/course-experiments/results/{name}.json';f=b/'freeze/technical-data'/p;actual=hashlib.sha256(f.read_bytes()).hexdigest();assert actual==tm['files_sha256'][p];out['inputs'][p]={'sha256':actual,'revision':json.loads(f.read_text())['revision']}
for name,keys in [('modern',['baseline','rope','rmsnorm']),('efficiency',['mha','gqa']),('precision',['fp32','bf16','fp16'])]:
 j=json.loads((b/f'freeze/technical-data/docs/course-experiments/results/{name}.json').read_text())['results'];section=j.get('variants',j.get('models'));checks={}
 for key in keys:
  v=section[key];checks[key]={}
  for split,h in v['heldout'].items():
   delta=abs(h['nll_sum']/h['effective_tokens']-h['nll']);assert delta<1e-12
   d={'nll_ratio_error':delta,'nll':h['nll'],'denominator':h['effective_tokens']}
   if 'matches' in h:
    hits=[];eos=0;den=0;invalid=0
    for sample in h['samples']:
     ids=sample['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids;hit=raw==[x+8 for x in sample['expected'].encode()];assert hit==sample['exact'];hits.append(hit);eos+=2 in ids;den+=len(sample['expected'].encode())+1;invalid+=sum(x<8 for x in raw)
    assert sum(hits)==h['matches'];assert den==h['effective_tokens'];assert eos/len(hits)==h['eos_rate'];d.update(matches=sum(hits),examples=len(hits),eos=eos,invalid_control_ids=invalid,denominator_recounted=den)
   checks[key][split]=d
  if 'inference' in v:
   inf=v['inference'];assert statistics.median(inf['samples_seconds'])==inf['median_seconds'];checks[key]['forward_median_ms']=inf['median_seconds']*1000
  t=v['training'];checks[key]['updates']=[t['optimizer_updates'],t['skipped_updates']];checks[key]['training_targets']=t['effective_tokens']
  if name=='precision':checks[key]['memory_MiB']=[round(t[x]/2**20,3) for x in ['memory_allocated_before_bytes','peak_memory_allocated_bytes','peak_additional_allocated_bytes']]
  if name=='efficiency':
   c=v['cache'];assert c['generated_ids_full']==c['generated_ids_cached'];checks[key]['cache_bytes']=c['prefill_cache_bytes'];checks[key]['cached_median_ms']=c['cached_decode']['median_seconds']*1000
 out['numeric_checks'][name]=checks
j=json.loads((b/'freeze/technical-data/docs/course-experiments/results/moe.json').read_text())['results']['variants']['top1_aux0'];r=j['validation_routing']['layers'][1];counts=r['dispatch_counts'];f=[x/sum(counts) for x in counts];h=-sum(x*math.log(x) if x else 0 for x in f)/math.log(4);assert sum(counts)==39256;assert abs(h-r['normalized_load_entropy'])<1e-12
out['numeric_checks']['moe']={'second_layer_counts':counts,'denominator':sum(counts),'normalized_entropy':h,'before':j['router_gradients_before'],'after':j['router_gradients_after']}
j=json.loads((b/'freeze/technical-data/docs/course-experiments/results/qat.json').read_text())['results'];q={}
for key,v in j['runs'].items():
 q[key]={}
 for split in ['validation','test']:
  h=v[split];assert abs(h['nll_sum']/h['supervised_tokens']-h['answer_nll'])<1e-12;hit=0;eos=0;den=0;invalid=0
  for sample in h['generated_samples']:
   ids=sample['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids;exact=raw==[x+8 for x in sample['expected'].encode()];assert exact==sample['exact'];hit+=exact;eos+=2 in ids;den+=len(sample['expected'].encode())+1;invalid+=sum(x<8 for x in raw)
  assert hit==h['correct'];assert den==h['supervised_tokens'];q[key][split]={'nll':h['answer_nll'],'correct':hit,'examples':len(h['generated_samples']),'eos':eos,'denominator':den,'illegal_controls':invalid}
 q[key]['tensor_bytes']=v['storage']['tensor_bytes']
a,c=j['training']['fp_finetuning'],j['training']['qat'];assert a['batch_plan_sha256']==c['batch_plan_sha256'];assert a['initialization_sha256']==c['initialization_sha256'];q['matched_plan']={'batch_plan_sha256':a['batch_plan_sha256'],'initialization_sha256':a['initialization_sha256'],'updates':[a['optimizer_updates'],c['optimizer_updates']],'training_targets':[a['effective_supervised_tokens'],c['effective_supervised_tokens']],'fake_packed_logit_error':j['fake_deployed_max_logit_difference']};out['numeric_checks']['qat']=q
(e/'numeric-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
figures={}
for p in m['inventory']['pages']:
 if p['page_id'] in m['groups']['architecture']['pages']:
  assert hashlib.sha256((b/'freeze/sources'/f"{p['page_id']}.md").read_bytes()).hexdigest()==p['source_sha256']
  for path,expected in p['figures_sha256'].items():
   f=b/'freeze/original'/path;assert hashlib.sha256(f.read_bytes()).hexdigest()==expected
   stem=Path(path).stem;figures[path]={'svg_sha256':expected,'inspected':'view_image 實看靜態渲染；不是原位頁面、DOM或平台測試','renders':[{ 'path':str((b/'renders'/f'{stem}-{size}.png').relative_to(repo)),'width_px':size,'sha256':hashlib.sha256((b/'renders'/f'{stem}-{size}.png').read_bytes()).hexdigest()} for size in [640,360]]}
(e/'visual-inspection.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2)+'\n')
s=Path('/workspace/work/tutorial-repair-20261008/technical-architecture-sources');inputs=json.loads((s/'inputs.json').read_text());excerpts={}
ranges={'roformer':[(218,301)],'rmsnorm':[(177,197)],'yarn':[(235,335),(741,819)],'gqa':[(35,102)],'switch':[(279,320)],'jacob':[(255,322)],'ste':[(400,422)]}
for name,rs in ranges.items():
 lines=(s/f'{name}.txt').read_text().splitlines();excerpts[name]={'input':inputs[name],'excerpts':[{'locator':f'pdftotext -layout lines {lo}–{hi}','text':'\n'.join(lines[lo-1:hi])} for lo,hi in rs]}
for name,terms in {'amp':['You should not call','Gradient Scaling#'],'sdpa':['Grouped Query Attention (GQA)'],'normalize':['torch.nn.functional.normalize#'],'layernorm':['The mean and standard-deviation']}.items():
 t=(s/f'{name}-article.txt').read_text();chunks=[]
 for term in terms:
  pos=t.find(term);chunks.append({'locator':term,'text':t[max(0,pos-100):pos+1600]})
 excerpts[name]={'input':inputs[name],'excerpts':chunks}
(e/'source-excerpts.json').write_text(json.dumps(excerpts,ensure_ascii=False,indent=2)+'\n')
for name in ['run_checks.py','integrity.py']:(e/name).write_bytes((s/name).read_bytes())
print('原始五份結果 SHA、NLL 分母、逐題原始ID匹配、熵、計時中位數、QAT起點與抽題計畫核對一致。')
print('凍結十一頁正文與七張SVG指紋核對一致；保存十四張已實看渲染的指紋。')
print('artifact sha256',hashlib.sha256((e/'numeric-integrity.json').read_bytes()).hexdigest())
