from pathlib import Path
import json,hashlib,math,statistics,sys,torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT));D=Path(__file__).parent
from tiny_perceptron.tokenization import ByteTokenizer
T=ByteTokenizer();torch.set_num_threads(1)
a=json.loads((D/'record-audit.json').read_text());out={'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'},'record_scope':'original fixed records only, no GPU rerun','fp32_rate_diagnostics':[],'raw_id_checks':[],'derived_metric_checks':[],'metric_inventory':[]}
for x in a['failed_checks']:
 e=float(torch.tensor(x['observed'],dtype=torch.float32));out['fp32_rate_diagnostics'].append({**x,'observed_fp32':e,'fp32_equal':e==x['expected'],'source':'scripts/course_experiments/posttraining.py:217-231 correct.float().mean()'})
def walk(x,p,n):
 if isinstance(x,dict):
  ids=x.get('generated_ids');truth=x.get('expected',x.get('target'))
  flag=next((k for k in ['exact','exact_match'] if type(x.get(k)) is bool),None)
  if ids is not None and isinstance(truth,str) and flag is not None:
   raw=ids[:ids.index(T.eos_id)] if T.eos_id in ids else ids;v=raw==T.encode(truth)
   out['raw_id_checks'].append({'record':n,'locator':p+'/'+flag,'expected':x[flag],'observed':v,'passed':v==x[flag],'denominator':len(ids)})
  pairs=[('answer_nll','nll_sum','supervised_tokens'),('answer_bpb','nll_sum','answer_bytes'),('bits_per_byte','nll_sum','raw_utf8_bytes')]
  for metric,numerator,denom in pairs:
   if metric in x and numerator in x and x.get(denom,0)>0:
    v=x[numerator]/x[denom]/(math.log(2) if 'bpb' in metric or metric=='bits_per_byte' else 1)
    out['derived_metric_checks'].append({'record':n,'locator':p+'/'+metric,'expected':x[metric],'observed':v,'tolerance':1e-9,'passed':abs(v-x[metric])<1e-9,'denominator':x[denom]})
  for k,v in x.items():
   if k in ['history','loss_trace','samples','generated_samples','skipped','conversions','profiler_events','kernel_events','raw_samples_seconds','probabilities','step_seconds','latencies_seconds']:continue
   if k in ['nll','mean_token_nll','answer_nll','correct','matches','examples','records','effective_tokens','effective_supervised_tokens','effective_answer_tokens_both_sides','steps','optimizer_updates','parameters','trainable_parameters','tensor_bytes','file_bytes','total_parameters','active_parameter_proxy_per_token','warm_step_median_seconds','eos_count','eos_rate','eligible_rows','selected_rows','skipped_rows','successful_optimizer_updates','peak_memory_allocated_bytes','peak_additional_allocated_bytes','seconds','bpb'] and isinstance(v,(int,float)):
    out['metric_inventory'].append({'record':n,'locator':p+'/'+k,'value':v})
   walk(v,p+'/'+str(k),n)
 elif isinstance(x,list):
  for i,v in enumerate(x):walk(v,p+'/'+str(i),n)
for n in a['records']:
 r=json.loads((ROOT/a['records'][n]['path']).read_text());walk(r['results'],'/results',n)
# raw-ID samples require separate traversal despite excluding their text from metric inventory
for n in a['records']:
 r=json.loads((ROOT/a['records'][n]['path']).read_text())
 def samples(x,p):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in ('samples','generated_samples') and isinstance(v,list):
     for i,row in enumerate(v):walk(row,p+'/'+k+'/'+str(i),n)
    else:samples(v,p+'/'+k)
  elif isinstance(x,list):
   for i,v in enumerate(x):samples(v,p+'/'+str(i))
 samples(r['results'],'/results')
(D/'record-diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
inv='\n'.join(f"{x['record']} {x['locator']} = {x['value']}" for x in out['metric_inventory'])+'\n';(D/'record-metrics.txt').write_text(inv)
print(json.dumps({'fp32_rates':out['fp32_rate_diagnostics'],'raw_id_checks':len(out['raw_id_checks']),'raw_id_failures':[x for x in out['raw_id_checks'] if not x['passed']],'derived_metrics':len(out['derived_metric_checks']),'derived_failures':[x for x in out['derived_metric_checks'] if not x['passed']]},indent=2))
