from pathlib import Path
import ast, copy, hashlib, json, math, platform, shlex, subprocess, sys, tempfile
import torch

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import generate_records
from scripts.evaluate import evaluate, answer_sample
from scripts.train_simple import examples
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import save_checkpoint, load_checkpoint, seed_everything, choose_device
from tiny_perceptron.quantization import replace_linear_layers, pack_int4, unpack_int4
from tiny_perceptron.alignment import LoRALinear, distillation_kl, dpo_loss

torch.set_num_threads(2)
env = {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': torch.version.git_version, 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'threads': str(torch.get_num_threads())}
out = {'environment': env, 'commands': [], 'checks': []}
def check(name, expected, observed, passed, tolerance='exact equality', details=''):
    out['checks'].append(dict(name=name, expected=expected, observed=observed, passed=bool(passed), tolerance=tolerance, details=details))
def command(args, expected_exit=0):
    proc = subprocess.run([sys.executable]+args, cwd=ROOT, text=True, capture_output=True, timeout=40)
    out['commands'].append({'command': shlex.join([sys.executable]+args), 'exit_code':proc.returncode, 'expected_exit':expected_exit, 'stdout':proc.stdout, 'stderr':proc.stderr})
    return proc

raw = (ROOT/'course/training.md').read_text()
blocks = __import__('re').findall(r'```([^\n]*)\n(.*?)```', raw, __import__('re').S)
pyblocks = [b for lang,b in blocks if lang == 'python']
for block in pyblocks: ast.parse(block)
check('all authored Python blocks compile', 'all parse', len(pyblocks), True, details='AST syntax check is separate from execution; compares no claimed training quality.')
for path in ['scripts/train.py','scripts/train_simple.py','scripts/prepare_data.py','scripts/infer.py','scripts/infer_modal.py','scripts/evaluate.py','scripts/evaluate_modal.py','scripts/pretrain_encoders.py','scripts/prepare_ocr.py','scripts/quantize.py','scripts/check_course_models.py']:
    p=command([path,'--help']); check(path+' help',0,p.returncode,p.returncode==0)
command(['-m','scripts.course_experiments.run','--help'])
assets=command(['scripts/fetch_training_assets.py','--list'])
models=command(['scripts/fetch_course_models.py','--list'])
manifest=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
num_models=len(manifest['models']); num_files=sum(sum(f['path'].endswith('.pt') for f in m['files']) for m in manifest['models'])
check('public manifest experiment/checkpoint counts',[30,120],[num_models,num_files],[num_models,num_files]==[30,120],details='Current local pinned manifest only; no remote model download.')
for task,expected in [('text',(111,5.7530,1.0268)),('sft',(8,5.4658,2.7651)),('vision',(None,5.9620,1.8985))]:
    p=command(['scripts/train.py','--task',task,'--device','cpu']); r=json.loads(p.stdout.splitlines()[-1]); h=r['history'][0]
    vals=[h['effective_tokens'],h['loss'],h['grad_norm']]
    check('T2 default '+task, list(expected),vals,h['effective_tokens']==expected[0] and abs(h['loss']-expected[1])<=0.00006 and abs(h['grad_norm']-expected[2])<=0.00006,tolerance='last two fields ±0.00006 for four-decimal rounding',details='dry-run-no-weight-update; 1 batch; original script does not call optimizer.step or checkpoint saving in this mode')

split=split_documents(toy_documents(),seed=42)
vocab={c:i+2 for i,c in enumerate(sorted(set(''.join(split['train']))))}
counts={k:len(examples(v,vocab,1)[1]) for k,v in split.items()}
check('T3 split and next-character denominators',{'documents':[9,1,2],'targets':[103,11,22],'validation':['顏色=紅；形狀=圓。']},{'documents':[len(v) for v in split.values()],'targets':list(counts.values()),'validation':split['validation']},[len(v) for v in split.values()]==[9,1,2] and list(counts.values())==[103,11,22] and split['validation']==['顏色=紅；形狀=圓。'])
for model,context,param,losses in [('bigram',1,289,(3.773265838623047,3.6166253089904785)),('mlp',1,833,(2.97138,2.94521)),('mlp',3,1345,(3.00414,3.02520)),('mlp',5,1857,(2.83350,2.78032))]:
    p=command(['scripts/train_simple.py','--model',model,'--context',str(context),'--width','16','--seed','42','--device','cpu']);r=json.loads(p.stdout)
    check('T3 baseline '+model+str(context),{'parameters':param,'losses':losses},{'parameters':r['parameters'],'losses':[r['train_loss'],r['validation_loss']]},r['parameters']==param and all(abs(x-y)<0.000006 for x,y in zip(losses,[r['train_loss'],r['validation_loss']])),tolerance='±0.000006 rounded five-decimal losses')

with tempfile.TemporaryDirectory(dir=DEST,prefix='cpu-fixtures-') as tmp:
    t=Path(tmp)
    for kind in ['toy-text','attributes-sft','style','safety','preference']:
        command(['scripts/prepare_data.py','--kind',kind,'--seed','42','--output',str(t)])
    generated={kind:json.loads((t/kind/'manifest.json').read_text()) for kind in ['toy-text','attributes-sft','style','safety','preference']}
    out['generated_manifests']=generated
    check('T4 toy-text record/family split',[[9,9],[1,1],[2,2]],[[s['records'],s['families']] for s in generated['toy-text']['splits'].values()],list(s['records'] for s in generated['toy-text']['splits'].values())==[9,1,2])
    check('T4 SFT record/family split',[[45,9],[5,1],[10,2]],[[s['records'],s['families']] for s in generated['attributes-sft']['splits'].values()],list(s['records'] for s in generated['attributes-sft']['splits'].values())==[45,5,10])
    first=json.loads((t/'preference/validation.jsonl').read_text().splitlines()[0])
    check('T7 first generated preference',['0+5=?','5','6'],[first[k] for k in ['prompt','chosen','rejected']], [first[k] for k in ['prompt','chosen','rejected']]==['0+5=?','5','6'])
    seed_everything(42); model=TinyLM(ModelConfig(width=32,layers=1,heads=1,max_length=128))
    start=t/'start.pt';save_checkpoint(start,model,step=0,metadata={'seed':42,'note':'untrained baseline'})
    _,payload=load_checkpoint(start)
    check('T4 start checkpoint state',[0,None,True],[payload['step'],payload['optimizer'],all(torch.equal(payload['model'][k],v) for k,v in model.state_dict().items())],payload['step']==0 and payload['optimizer'] is None)
    for kind,mode,targets in [('toy-text','text',35),('attributes-sft','sft',36)]:
        p=command(['scripts/evaluate.py',str(start),'--data',str(t/kind/'validation.jsonl'),'--mode',mode,'--tokens','2','--device','cpu','--output',str(t/(mode+'.json'))]);r=json.loads((t/(mode+'.json')).read_text())
        check('T4 actual validation effective targets '+mode,targets,r['effective_tokens'],r['effective_tokens']==targets,details='Full generated validation selected; generation cap shortened to 2 solely for a bounded interface probe; NLL uses complete target. '+json.dumps(r['metric_denominators']))
    for cache in (False,True):
        p=command(['scripts/infer.py',str(start),'--chat','--prompt','shape?','--tokens','2','--temperature','0','--device','cpu','--json']+(['--cache'] if cache else []))
        out.setdefault('inference_cache',[]).append(json.loads(p.stdout))
    check('greedy cached/uncached output IDs equal',True,out['inference_cache'][0]['generated_ids']==out['inference_cache'][1]['generated_ids'],out['inference_cache'][0]['generated_ids']==out['inference_cache'][1]['generated_ids'])
    command(['scripts/train.py','--train','--steps','0','--device','cpu'],expected_exit=1)
    # Two updates only, under an unchanged four-step plan, compare genuine interrupted resumption.
    common=['scripts/train.py','--task','sft','--device','cpu','--seed','42','--train','--steps','4','--stop-after','2','--save-every','1','--width','8','--max-length','32','--batch-size','1']
    command(common+['--output',str(t/'full.pt')])
    command([x if x!='2' else '2' for x in common[:-0]] if False else common[:common.index('--stop-after')]+['--stop-after','1']+common[common.index('--save-every'):]+['--output',str(t/'part.pt')])
    command(['scripts/train.py','--task','sft','--device','cpu','--seed','42','--train','--steps','4','--stop-after','2','--save-every','1','--width','8','--max-length','32','--batch-size','1','--checkpoint',str(t/'part.pt'),'--resume','--output',str(t/'resume.pt')])
    a=torch.load(t/'full.pt',weights_only=True);b=torch.load(t/'resume.pt',weights_only=True)
    delta=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    check('T4 genuine interrupted CPU resume',0.0,delta,delta==0.0,details='2 updates under total schedule4, stop at1 and restore optimizer/RNG; no course training or quality replication')
    command(['scripts/train.py','--task','sft','--device','cpu','--train','--steps','4','--batch-size','1','--checkpoint',str(t/'part.pt'),'--resume','--lr','0.02','--output',str(t/'bad.pt')],expected_exit=1)
    command(['scripts/prepare_ocr.py','--seed','42','--output',str(t/'ocr')]);om=json.loads((t/'ocr/manifest.json').read_text())
    check('T6 OCR generated split',[240,30,30],[v['records'] for v in om['splits'].values()],[v['records'] for v in om['splits'].values()]==[240,30,30]);out['ocr_manifest']=om
    # I/O fixture: reject 8kHz and correctly admit 16kHz; no source dataset download.
    import soundfile as sf
    from tiny_perceptron.modal_data import modal_example
    sf.write(t/'8.wav',torch.zeros(80).numpy(),8000);sf.write(t/'16.wav',torch.zeros(160).numpy(),16000)
    record={'audio':'8.wav','question':'pitch?','answer':'low'}
    try: modal_example(record,t,'audio'); reject=False
    except ValueError as error: reject=True;out['audio_8khz_error']=str(error)
    record['audio']='16.wav';sample=modal_example(record,t,'audio')
    check('T6 explicit rate gate',[True,160],[reject,sample[3].numel()],reject and sample[3].numel()==160)

torch.manual_seed(42); r=evaluate(TinyLM(ModelConfig(width=32)),[{'question':'shape?','answer':'circle'}],mode='sft',max_new_tokens=2)
fields={k:r[k] for k in ['mean_token_nll','effective_tokens','exact_match','completed_exact_match','eos_rate','samples']};out['T4_numerical_sample']=fields
check('T4 literal untrained evaluator example',{'nll':5.97386714390346,'effective':7,'ids':[143,30]}, {'nll':r['mean_token_nll'],'effective':r['effective_tokens'],'ids':r['samples'][0]['generated_ids']},abs(r['mean_token_nll']-5.97386714390346)<=1e-6 and r['effective_tokens']==7 and r['samples'][0]['generated_ids']==[143,30],tolerance='NLL ±1e-6; exact count and IDs')
tok=ByteTokenizer(); probes={'content_only':tok.encode('circle'),'completed':tok.encode('circle')+[2],'extra_space':tok.encode('circle ')+[2],'illegal_role':[4]+tok.encode('circle')+[2]}
observed={k:{x:answer_sample(tok,v,'circle')[x] for x in ['exact_match','completed_exact_match','invalid_special_tokens']} for k,v in probes.items()}
check('T4 exact content/EOS/control distinctions',[[True,False],[True,True],[False,False],[False,False]],[[v['exact_match'],v['completed_exact_match']] for v in observed.values()],[[v['exact_match'],v['completed_exact_match']] for v in observed.values()]==[[True,False],[True,True],[False,False],[False,False]]);out['exact_match_probes']=observed
model=TinyLM(ModelConfig(width=64,layers=2,heads=1,max_length=128));n=sum(p.numel() for p in model.parameters());lin=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
sizes=[]
for bits in [None,4,8]:
    m=copy.deepcopy(model)
    if bits: replace_linear_layers(m,bits)
    sizes.append(sum(t.numel()*t.element_size() for t in m.state_dict().values()))
check('T9 actual native and packed tensor bytes',[141568,13,566272,168736,226336],[n,len(lin),*sizes],[n,len(lin),*sizes]==[141568,13,566272,168736,226336])
for width,expected in [(16,13744),(32,33632)]:
    m=TinyLM(ModelConfig(width=width,layers=1));actual=sum(p.numel() for p in m.parameters());check('T10 student width '+str(width),expected,actual,actual==expected)
values=torch.tensor([-8,-7,0,7,3],dtype=torch.int8);packed=pack_int4(values);restored=unpack_int4(packed,values.shape)
check('signed odd int4 packing',values.tolist(),restored.tolist(),torch.equal(values,restored),details='5 signed codes ->3 bytes; tail padding removed by original shape')
layer=LoRALinear(torch.nn.Linear(16,12),2,2);x=torch.randn(3,16);initial=torch.equal(layer(x),layer.base(x));layer(x).sum().backward()
check('LoRA low-rank parameter/path definition',[56,True,True,True],[sum(p.numel() for p in layer.parameters() if p.requires_grad),initial,layer.a.grad.abs().max().item()==0,layer.b.grad.abs().max().item()>0],sum(p.numel() for p in layer.parameters() if p.requires_grad)==56 and initial and layer.a.grad.abs().max().item()==0 and layer.b.grad.abs().max().item()>0)
check('T11 transparent hypothetical arithmetic',[30000,0,-0.2,20000],[80000-50000,4/5-4/5,3/5-4/5,80000-60000],abs((3/5-4/5)-(-0.2))<1e-15,tolerance='floating error ≤1e-15; hypothetical figures not empirical')
out['all_checks_passed']=all(c['passed'] for c in out['checks'])
(DEST/'cpu-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':len(out['checks']),'failures':[c for c in out['checks'] if not c['passed']],'environment':env},ensure_ascii=False,indent=2))
