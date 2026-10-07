import ast,hashlib,json,math,platform,random
from pathlib import Path
root=Path(__file__).resolve().parents[4];base=Path(__file__).resolve().parent
m=json.loads((root/'docs/course-experiments/results/tokenizer.json').read_text())
metrics=ast.parse((base/'sources/lm-eval-v0.4.9.1-metrics.py').read_text());selected=[]
for node in metrics.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['bits_per_byte','weighted_mean']:
  node.decorator_list=[];selected.append(node)
ns={'math':math};exec(compile(ast.Module(body=selected,type_ignores=[]),'official-metrics','exec'),ns)
raw='貓🙂';B=len(raw.encode('utf-8'));toy=[]
for nll in [5.0,4.0]:
 bpb=nll/(B*math.log(2));official=ns['bits_per_byte']([(-nll,B)]);assert bpb==official
 toy.append({'nll':nll,'raw_bytes':B,'bpb':bpb,'double_nll_bpb':2*nll/(B*math.log(2)),'double_raw_bpb':nll/(len((raw*2).encode())*math.log(2))})
assert B==7 and -math.log(0.5)/math.log(2)==1
chain=-math.log(0.5*0.25);assert math.isclose(chain,-math.log(0.5)-math.log(0.25))
rows=[]
for name,r in m['results']['runs'].items():
 for time in ['before','after']:
  for split in ['validation','test']:
   s=r[time][split];total=s['mean_token_nll']*s['effective_tokens'];computed=total/(s['raw_utf8_bytes']*math.log(2));observed=s['bpb_including_eos_boundary_targets'];assert math.isclose(computed,observed,abs_tol=1e-10)
   rows.append({'name':name,'time':time,'split':split,'documents':s['records'],'effective_targets_including_eos':s['effective_tokens'],'raw_utf8_bytes':s['raw_utf8_bytes'],'mean_token_nll':s['mean_token_nll'],'reconstructed_total_nll':total,'computed_bpb':computed,'recorded_bpb':observed})
 assert r['training']['steps']==400 and r['training_raw_utf8_bytes_exposed']==664759
exposure=[random.Random(42)]
rng=random.Random(42);schedule=[rng.choices(range(153),k=8) for _ in range(400)]
digest=hashlib.sha256(json.dumps(schedule,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest==m['results']['raw_document_schedule_sha256']
print(json.dumps({'python':platform.python_version(),'device':'cpu','toy':toy,'conditional_chain_nll':chain,'half_probability_bits':1,'ten_vs_five_total_nll':[10*0.6,5*1.0],'historical_readback':rows,'exposure_steps':400,'batch_documents':8,'schedule_sha256':digest,'raw_bytes_exposed_each':664759,'eos_convention':'each document includes EOS in numerator but not UTF-8 raw-byte denominator','no_retraining':True},ensure_ascii=False,indent=2))

