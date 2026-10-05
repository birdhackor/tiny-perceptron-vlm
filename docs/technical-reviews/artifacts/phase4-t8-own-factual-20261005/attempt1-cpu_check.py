"""Independent T.8 bounded CPU audit; smoke behavior and hand calculations, not capability scores."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import ast, copy, contextlib, hashlib, importlib.util, json, math, os, statistics, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from scripts.train import parser,prepare_examples
from scripts.evaluate import evaluate,parse_limit,answer_sample
from tiny_perceptron.model import ModelConfig,TinyLM,loss_sum
from tiny_perceptron.modern import RMSNorm,DenseFFN,MoEFFN
from tiny_perceptron.data import ByteTokenizer,IGNORE,pad_batch,shifted
import scripts.course_experiments.architecture as arch
from scripts.course_experiments.common import Context
OUT=Path(__file__).parent
result={'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available())},'scope':'Short CPU checks only; no 200-step train, CUDA probe, external dataset or model download; fresh smoke losses are not capability evidence.'}
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
# Original shell fences are preserved byte for byte. Execute only original data preparation line.
spec=importlib.util.spec_from_file_location('sf',ROOT/'docs/review-tools/section_facts.py');sf=importlib.util.module_from_spec(spec);spec.loader.exec_module(sf)
body,whole,line=sf.original_section(ROOT/'course/training.md','T.8')
for i,f in enumerate(sf.fences(body,line),1):(OUT/f'original-fence-{i}.sh').write_bytes(f['raw'])
commands=[]
with tempfile.TemporaryDirectory(prefix='t8-own-cpu-') as tmp:
 w=Path(tmp);(w/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True);(w/'scripts').symlink_to(ROOT/'scripts',target_is_directory=True)
 def run(cmd,label):
  completed=subprocess.run(cmd,cwd=w,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},capture_output=True,text=True,timeout=45)
  (OUT/(label+'.stdout.txt')).write_text(completed.stdout);(OUT/(label+'.stderr.txt')).write_text(completed.stderr)
  commands.append({'label':label,'argv':cmd,'cwd':str(w),'exit_code':completed.returncode,'stdout':label+'.stdout.txt','stderr':label+'.stderr.txt'})
  assert completed.returncode==0,(label,completed.stderr)
 run(['.venv/bin/python','scripts/prepare_data.py','--kind','toy-text','--seed','42'],'prepare-original')
 manifest=json.loads((w/'data/generated/toy-text/manifest.json').read_text());result['original_prepare']={'splits':manifest['splits'],'seed':manifest['seed'],'original_fence':'original-fence-1.sh line 1'}
 defaults=vars(parser().parse_args([]));result['cli_defaults']={k:(str(v) if isinstance(v,Path) else v) for k,v in defaults.items()}
 assert defaults['width']==32 and defaults['norm']=='layer' and defaults['experts']==0 and defaults['batch_size']==4
 smoke={}
 for name,extra in [('baseline',[]),('rms',['--norm','rms']),('moe',['--experts','4','--top-k','2'])]:
  # Explicit bounded variation: original 200 updates -> 1; explicit CPU; temporary weights only.
  run(['.venv/bin/python','scripts/train.py','--task','text','--data','data/generated/toy-text/train.jsonl','--train','--steps','1','--seed','42','--device','cpu',*extra,'--output',f'checkpoints/{name}.pt'],name+'-bounded-train')
  d=json.loads((w/f'checkpoints/{name}.json').read_text());assert len(d['history'])==1 and d['history'][0]['step']==1 and d['config']['width']==32
  smoke[name]={'parameters':d['parameters'],'config':d['config'],'updates':len(d['history']),'batch_size':4,'counted_positions':d['history'][0]['effective_tokens']}
  run(['.venv/bin/python','scripts/evaluate.py',f'checkpoints/{name}.pt','--data','data/generated/toy-text/validation.jsonl','--mode','text','--tokens','32','--device','cpu','--limit','all','--output',f'{name}-eval.json'],name+'-bounded-evaluate')
  e=json.loads((w/f'{name}-eval.json').read_text());assert e['limit']=='all' and e['records_read']==1 and e['unselected_records']==0
  smoke[name]['evaluation_contract']={k:e[k] for k in ['effective_tokens','records_read','records_selected','generation_evaluated_records','generation_skipped_records','metric_denominators','limit']}
 result['bounded_cli_smokes']=smoke
 # Actual CPU no-run branch: reduce fixed fixture shape so the contract check remains bounded.
 oldshape=arch.FLASH_PROBE_SHAPE;arch.FLASH_PROBE_SHAPE=(1,1,4,3)
 try:
  flash=arch.run_flash_probe(Context('cpu',w,w,w));assert flash['status']=='not_run' and flash['verification_passed'] is False and not flash['routes']
  result['flash_cpu_no_run']={k:flash[k] for k in ['status','verification_passed','schedule_completed','reason','configuration']}
 finally:arch.FLASH_PROBE_SHAPE=oldshape
 # No fixtures or generated model weights are retained; TemporaryDirectory is destroyed.
result['commands']=commands
# Genuine model shapes/routing, no optimizer update.
counts={}
for name,cfg in [('baseline',ModelConfig()),('rms',ModelConfig(norm='rms')),('moe',ModelConfig(experts=4,top_k=2))]:
 m=TinyLM(cfg);x=torch.tensor([[1,73,74]]);o=m(x);assert o['logits'].shape==(1,3,264)
 counts[name]={'parameters':sum(p.numel() for p in m.parameters()),'norm1':type(m.blocks[0].norm1).__name__,'ffn':type(m.blocks[0].ffn).__name__,'logits_shape':list(o['logits'].shape)}
 if cfg.experts:
  xx=torch.randn(1,3,32,requires_grad=True);yy,aux,chosen=m.blocks[0].ffn(xx);assert chosen.shape==(3,2) and len(m.blocks[0].ffn.experts)==4
  (yy.square().sum()+aux).backward();counts[name]['routing_shape']=list(chosen.shape);counts[name]['all_expert_state_tables']=list(m.blocks[0].ffn.state_dict())
result['architecture_shapes']=counts
assert counts['moe']['parameters']>counts['baseline']['parameters']
x=torch.tensor([[1.,2.,3.,4.]]);r=RMSNorm(4)(x);expected=x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-5);assert torch.allclose(r,expected)
result['rms_formula']={'input':x.tolist(),'observed':r.tolist(),'expected':expected.tolist(),'atol':1e-6}
# AB table is a hand-designed two-position probability example; it is not CLI output.
tok=ByteTokenizer();ids=[tok.bos_id]+tok.encode('AB')+[tok.eos_id];_,y=shifted(ids);assert len(y)==3
p0,p1=math.exp(-0.8),math.exp(-0.6);result['AB_hand_calculation']={'assumed_targets':['B','EOS'],'effective_tokens':2,'probability_each_target_baseline':p0,'probability_each_target_other':p1,'nll_means':[-math.log(p0),-math.log(p1)],'actual_full_text_evaluate_targets':y.tolist(),'actual_full_text_effective_tokens':3,'distinction':'The hand example assumes only B and EOS scored; text evaluate also scores A after BOS.'}
# Verify weighted-token denominator and IGNORE/EOS directly on nonuniform target probabilities.
probs=torch.tensor([[[0.5,0.3,0.2],[0.1,0.6,0.3],[0.2,0.2,0.6]]],dtype=torch.float64);labels=torch.tensor([[0,IGNORE,2]])
summed,count=loss_sum(probs.log(),labels);expected=-math.log(0.5)-math.log(0.6)
assert count.item()==2 and abs(summed.item()-expected)<1e-12
result['ignored_eos_denominator']={'labels':labels.tolist(),'count':count.item(),'summed_nll':summed.item(),'mean_nll':summed.item()/count.item(),'expected_mean':expected/2,'tolerance':1e-12}
# Literal CLI AB evaluation: 3 targets; tiny max context demonstrates generation can skip selected records.
full=evaluate(TinyLM(ModelConfig()),[{'text':'AB'}],max_new_tokens=2);short=evaluate(TinyLM(ModelConfig(max_length=4)),[{'text':'ABCD'}],max_new_tokens=32)
assert full['effective_tokens']==3 and short['records_selected']==1 and short['generation_evaluated_records']==0 and short['generation_skipped_records']==1
result['actual_eval_denominators']={k:full[k] for k in ['effective_tokens','metric_denominators']};result['selected_but_skipped']={k:short[k] for k in ['records_selected','loss_evaluated_records','effective_tokens','generation_evaluated_records','generation_skipped_records','skipped']}
assert parse_limit('all') is None
ended=answer_sample(tok,tok.encode('B')+[tok.eos_id],'B');assert ended['completed_exact_match'] and ended['stop_reason']=='eos'
result['samples_eos']=ended
# Bounded original attention helpers on CPU; CPU SDPA does not prove CUDA Flash.
g=torch.Generator().manual_seed(42);inputs=tuple(torch.randn(1,1,4,3,generator=g,requires_grad=True) for _ in range(3));up=torch.randn(1,1,4,3,generator=g)
a=arch._flash_manual_attention(*inputs);b=arch._flash_sdpa_attention(*inputs);ga=torch.autograd.grad(a,inputs,up);gb=torch.autograd.grad(b,inputs,up)
assert torch.allclose(a,b,atol=1e-6) and all(torch.allclose(v,z,atol=1e-6) for v,z in zip(ga,gb))
result['bounded_attention']={'shape':[1,1,4,3],'dtype':'torch.float32','output_max_abs':float((a-b).abs().max()),'qkv_gradient_max_abs':[float((v-z).abs().max()) for v,z in zip(ga,gb)],'atol':1e-6,'backend_scope':'CPU SDPA; no CUDA backend claim'}
# CPU dtype API contract and FP16 exclusion branches, without running run_precision's training loop.
linear=torch.nn.Linear(3,3);inp=torch.ones(2,3)
with arch._amp('cpu',None):fp32=linear(inp)
with arch._amp('cpu',torch.bfloat16):bf16=linear(inp)
assert fp32.dtype==torch.float32 and bf16.dtype==torch.bfloat16 and linear.weight.dtype==torch.float32
result['precision_cpu_autocast']={'fp32_output':str(fp32.dtype),'bf16_output':str(bf16.dtype),'parameter_dtype':str(linear.weight.dtype),'fp16_branch_contract':'run_precision lines 1009-1011: non-CUDA -> not_run; not independently trained here'}
# Execute the original aggregation statements on constructed route reports, not fabricated GPU measurements.
src=(ROOT/'scripts/course_experiments/architecture.py').read_text();tree=ast.parse(src);fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_flash_probe')
aggregate=[n for n in fun.body if 1373<=n.lineno<=1388];module=ast.Module(body=copy.deepcopy(aggregate),type_ignores=[]);ast.fix_missing_locations(module);code=compile(module,'original-run_flash_probe-aggregation','exec')
cases=[]
for routes,supported in [({'fp16':{'status':'completed'},'bf16':{'status':'unsupported'}},['fp16']),({'fp16':{'status':'completed'},'bf16':{'status':'runtime_error'}},['fp16']),({'fp16':{'status':'completed'},'bf16':{'status':'backend_unverified'}},['fp16']),({'fp16':{'status':'numerical_mismatch'}},['fp16']),({'fp16':{'status':'unsupported'}},[])]:
 report={'routes':routes,'supported_routes':supported};exec(code,{'report':report});cases.append(report)
assert cases[1]['verification_passed'] and not cases[1]['schedule_completed'] and cases[1]['status']=='failed_verification'
result['flash_verification_semantics_cases']=cases
# Recalculate raw report predicates, medians and baseline arithmetic; no rerun of published GPU experiments.
rawchecks={}
for name in ['flash_probe','precision','efficiency']:
 d=json.loads((OUT/'raw-results'/f'{name}.json').read_text());r=d['results'];checks=[]
 if name=='flash_probe':
  expected=bool(r['supported_routes']) and all(r['routes'][s]['status']=='completed' for s in r['supported_routes']);assert r['verification_passed']==expected
  for route in r['routes'].values():
   comparisons=[route['correctness']['output'],*route['correctness']['gradients'].values()]
   assert route['all_checks_finite']==all(x['finite'] for x in comparisons)
   assert route['all_comparisons_within_tolerance']==all(x['within_declared_tolerance'] for x in comparisons)
   for mode,profile in route['profiles'].items():
    assert profile['flash_forward_operator_observed']==('aten::_scaled_dot_product_flash_attention' in profile['operator_names'])
    assert profile['flash_cuda_verified'] and profile['cuda_kernel_names']
    if mode=='forward_backward':assert 'aten::_scaled_dot_product_flash_attention_backward' in profile['operator_names']
   for mode,c in route['measurements'].items():
    for backend,m in c.items():
     assert len(m['samples_seconds'])==m['measured_calls']==9 and m['median_seconds']==statistics.median(m['samples_seconds']) and m['additional_peak_allocated_bytes']==m['peak_allocated_bytes']-m['allocated_before_bytes']
     checks.append({'mode':mode,'backend':backend,'median_recomputed':m['median_seconds'],'memory_subtraction_exact':True})
  rawchecks[name]={'verification_recomputed':expected,'routes_status':{k:v['status'] for k,v in r['routes'].items()},'checks':checks}
 elif name=='precision':
  for key,v in r['variants'].items():
   t=v['training'];m=v['inference'];assert t['steps']==t['optimizer_updates']+t['skipped_updates']==200
   assert t['peak_additional_allocated_bytes']==t['peak_memory_allocated_bytes']-t['memory_allocated_before_bytes']
   assert m['median_seconds']==statistics.median(m['samples_seconds']) and len(m['samples_seconds'])==9
   checks.append({'variant':key,'attempts':t['steps'],'updates':t['optimizer_updates'],'skips':t['skipped_updates'],'effective_tokens':t['effective_tokens'],'weights_dtype':v['weights_dtype'],'logits_dtype':v['observed_logits_dtype']})
  rawchecks[name]={'checks':checks}
 else:
  for key,v in r['models'].items():
   t=v['training'];assert t['peak_additional_allocated_bytes']==t['peak_memory_allocated_bytes']-t['memory_allocated_before_bytes']
  for k,m in r['manual_vs_sdpa']['timings'].items():assert m['median_seconds']==statistics.median(m['samples_seconds']) and len(m['samples_seconds'])==9
  rawchecks[name]={'selected_sdpa_backend':r['manual_vs_sdpa']['backend']['selected'],'flash_observed':r['manual_vs_sdpa']['backend']['flash_attention_observed'],'medians_verified':True,'memory_subtractions_verified':True}
result['raw_gpu_report_recalculations']=rawchecks
(OUT/'cpu_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
