import ast,hashlib,json,math,platform,statistics,time
from decimal import Decimal
from pathlib import Path
import torch
from torch import nn
from torch._dynamo.backends.debugging import eager as eager_backend
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
print('environment',json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','threads':torch.get_num_threads()}))
for seed in [0,1,42]:
 torch.manual_seed(seed)
 layer=nn.Linear(4,3).eval()
 before={k:v.detach().clone() for k,v in layer.state_dict().items()}
 compiled=torch.compile(layer,backend='eager')
 for batch in [1,2,5]:
  x=torch.randn(batch,4)
  with torch.no_grad():reference,actual=layer(x),compiled(x)
  error=float((reference-actual).abs().max())
  assert actual.shape==(batch,3) and torch.allclose(reference,actual,atol=1e-7,rtol=1e-6)
  assert all(p.grad is None for p in layer.parameters())
  assert all(torch.equal(before[k],v) for k,v in layer.state_dict().items())
  print('eager_interface',json.dumps({'seed':seed,'input':[batch,4],'output':list(actual.shape),'max_abs_error':error,'parameters_unchanged':True,'no_gradients':True}))
torch._dynamo.reset()
backend_calls=[]
def counted_backend(graph,inputs):
 backend_calls.append(len(backend_calls)+1)
 return eager_backend(graph,inputs)
layer=nn.Linear(4,3).eval()
compiled=torch.compile(layer,backend=counted_backend,fullgraph=True,dynamic=False)
assert len(backend_calls)==0
counts=[len(backend_calls)]
with torch.no_grad():
 for batch in [2,2,5]:
  x=torch.ones(batch,4)
  actual=compiled(x)
  assert torch.equal(layer(x),actual)
  counts.append(len(backend_calls))
assert counts==[0,1,1,2]
print('lazy_and_shape_specialization',json.dumps({'stages':['after_wrap','after_first_call','same_shape_call','changed_batch_call'],'backend_compile_counts':counts,'backend':'counted eager','dynamic':False}))
for setup in [Decimal('2'),Decimal('4')]:
 plain,optimized=Decimal('0.01'),Decimal('0.006')
 break_even=setup/(plain-optimized)
 assert break_even in [Decimal(500),Decimal(1000)]
 print('hypothetical_break_even',json.dumps({'setup_seconds':str(setup),'saving_seconds_per_call':str(plain-optimized),'calls':str(break_even)}))
 for calls in [100,500,1000]:
  print('hypothetical_total',json.dumps({'setup_seconds':str(setup),'calls':calls,'plain_seconds':str(plain*calls),'compiled_seconds':str(setup+optimized*calls)}))
# Execute the actual original benchmark method in isolation; no GPU or training.
source=(BASE/'original-version'/'architecture.py').read_text();tree=ast.parse(source)
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_sync','_benchmark']]
ns={'torch':torch,'time':time,'statistics':statistics}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(BASE/'original-version'/'architecture.py'),'exec'),ns)
call_count=[0]
def inference():
 call_count[0]+=1
 with torch.no_grad():return layer(torch.ones(2,4))
measured=ns['_benchmark'](inference,'cpu')
assert call_count[0]==12 and len(measured['samples_seconds'])==9
assert measured['median_seconds']==sorted(measured['samples_seconds'])[4]
assert all(math.isfinite(x) and x>=0 for x in measured['samples_seconds'])
print('original_benchmark_cpu_contract',json.dumps({'warmup_calls':measured['warmup_calls'],'measured_calls':measured['measured_calls'],'total_invocations':call_count[0],'median_is_fifth_sorted':True,'device':'cpu','timing_scope':'fresh 2x4 linear inference, not L4 measurement'}))
# Read only the named original measurements and provenance.
raw=(BASE/'frozen/docs/course-experiments/results/efficiency.json').read_bytes();report=json.loads(raw)
c=report['results']['compile']
checked=[]
for name in ['eager','compiled']:
 d=c[name];samples=d['samples_seconds'];median=sorted(samples)[4]
 assert len(samples)==9 and d['measured_calls']==9 and d['warmup_calls']==3 and d['synchronized'] is True
 assert median==d['median_seconds']
 checked.extend(['/results/compile/'+name+'/'+field for field in ['samples_seconds','median_seconds','warmup_calls','measured_calls','synchronized']])
 print('recorded_L4_recalculated',json.dumps({'path':'/results/compile/'+name,'sample_count':len(samples),'median_seconds':median,'median_ms':median*1000,'median_is_fifth_sorted':True,'rounded_ms_6dp':format(median*1000,'.6f')}))
saving=c['eager']['median_seconds']-c['compiled']['median_seconds']
assert saving<0 and c['estimated_calls_to_amortize_first_call'] is None
assert c['shape']==[4,16,64] and c['dtype']=='torch.float32'
assert c['backend']=='inductor' and c['fullgraph'] is True and c['dynamic'] is False
assert c['unique_graphs']==1 and c['graph_breaks']=={} and c['inductor_generated_kernel_count']==1
assert format(c['first_compiled_call_seconds'],'.3f')=='7.717'
assert format(c['worker_wall_seconds'],'.3f')=='16.255'
assert format(c['output_max_error'],'.5e')=='7.15256e-07'
print('recorded_L4_compile_metadata',json.dumps({k:c[k] for k in ['status','backend','shape','dtype','fullgraph','dynamic','unique_graphs','graph_breaks','inductor_generated_kernel_count','first_compiled_call_seconds','worker_wall_seconds','output_max_error','estimated_calls_to_amortize_first_call']}))
print('recorded_L4_amortization',json.dumps({'saving_seconds_per_call':saving,'saving_ms_per_call':saving*1000,'finite_break_even_exists':False,'reason':'positive startup plus slower repeated execution cannot cross baseline'}))
config=report['results']['models']['mha']['model']['config']
training=report['results']['models']['mha']['training']
print('recorded_model_provenance',json.dumps({'config':config,'training':{k:training[k] for k in ['steps','optimizer_updates','status','requested_steps']},'revision':report['revision'],'device':report['device'],'gpu':report['gpu'],'torch_version':report['torch_version']}))
assert config['width']==64 and config['experts']==0 and training['optimizer_updates']>0
checked.extend(['/results/compile/'+key for key in ['status','backend','scope','shape','dtype','fullgraph','dynamic','first_compiled_call_seconds','output_max_error','unique_graphs','graph_breaks','inductor_generated_kernel_count','estimated_calls_to_amortize_first_call','worker_wall_seconds']])
checked.extend(['/revision','/device','/gpu','/seed','/torch_version','/python_version','/code_sha256','/results/source','/results/runtime','/results/models/mha/model/config']+['/results/models/mha/training/'+key for key in ['steps','optimizer_updates','status','requested_steps']])
(BASE/'checked-json-pointers.json').write_text(json.dumps({'original':'docs/course-experiments/results/efficiency.json','permanent_copy':'frozen/docs/course-experiments/results/efficiency.json','original_sha256':hashlib.sha256(raw).hexdigest(),'checked_pointers':checked,'not_read':['/results/compile/note','/results/gqa_note','/results/timing_note','/results/models/mha/training/initial','/results/models/mha/training/final']},ensure_ascii=False,indent=2)+'\n')
print('PASS all bounded checks; existing GPU measurements inspected only')
