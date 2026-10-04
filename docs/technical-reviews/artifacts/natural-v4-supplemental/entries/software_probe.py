from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys
import torch
from torch import nn
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[4];sys.path.insert(0,str(ROOT));torch.set_num_threads(2);torch.manual_seed(42)
from tiny_perceptron.alignment import LoRALinear,dpo_loss,distillation_loss
from tiny_perceptron.quantization import QuantizedLinear,fake_quantize
from tiny_perceptron.posttraining import FiniteResponsePolicy,ppo_clipped_objective
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.multimodal import MultiModalLM,scene,tone
from tiny_perceptron.data import ByteTokenizer,render_chat
from scripts.course_release import inference_payload
from scripts.check_course_models import check_payload
record={'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu'},'limits':'Small deterministic forwards/backwards and format tests. No training update, platform reinstall, full-book execution, model downloads, CUDA or MPS test.'}
tok=ByteTokenizer();x,y=render_chat([{'role':'user','content':'hello'},{'role':'assistant','content':'hi'}],tok)
lm=TinyLM(ModelConfig(width=8,max_length=64));logits=lm(x[None])['logits'];record['text']={'logits_shape':list(logits.shape),'ignored_prompt_positions':int((y==-100).sum()),'answer_positions':int((y!=-100).sum())}
modal=MultiModalLM(TinyLM(ModelConfig(width=8,max_length=64)))
ids=torch.tensor([tok.bos_id,tok.image_id,tok.audio_id,tok.assistant_id])
o=modal(ids,image=scene('red','square'),waveform=tone());record['modal']={'logits_shape':list(o['logits'].shape),'finite':str(torch.isfinite(o['logits']).all().item())}
base=nn.Linear(3,2);lora=LoRALinear(base,rank=1);inp=torch.tensor([[1.,2.,3.]])
assert torch.equal(lora(inp),base(inp));lora(inp).sum().backward();record['lora']={'initial_equal':True,'base_gradient':str(base.weight.grad),'adapter_b_gradient_nonzero':str(lora.b.grad.abs().sum().item()>0)}
pc=torch.tensor([-1.],requires_grad=True);pr=torch.tensor([-2.],requires_grad=True)
dl=dpo_loss(pc,pr,torch.tensor([-1.]),torch.tensor([-2.]));dl.backward();record['dpo']={'loss':dl.item(),'chosen_gradient':pc.grad.item(),'rejected_gradient':pr.grad.item()}
assert abs(dl.item()-0.6931471805599453)<1e-7
student=torch.randn(1,2,3,requires_grad=True);teacher=torch.randn(1,2,3);labels=torch.tensor([[-100,1]])
kl=distillation_loss(student,teacher,labels);kl.backward();record['distillation']={'loss':kl.item(),'ignored_grad_zero':bool((student.grad[:,0]==0).all()),'effective_positions':1}
q=QuantizedLinear(nn.Linear(4,2),bits=4);qy=q(torch.ones(1,4));record['quantization']={'packed_dtype':str(q.values.dtype),'packed_elements':q.values.numel(),'result_shape':list(qy.shape),'storage_bytes':q.storage_bytes()}
fq=torch.tensor([0.1,0.3,-0.2],requires_grad=True);fake_quantize(fq,4).sum().backward();record['qat']={'gradient':fq.grad.tolist()};assert torch.equal(fq.grad,torch.ones_like(fq))
policy=FiniteResponsePolicy();pl=policy(torch.zeros(2,4));record['finite_ppo']={'policy_logits_shape':list(pl.shape),'candidate_count':pl.shape[-1],'objective':ppo_clipped_objective(torch.tensor([0.]),torch.tensor([0.]),torch.tensor([1.]))['policy_loss'].item()}
saved={'format_version':1,'config':lm.description()['config'],'model':lm.state_dict(),'optimizer':{'state':1},'torch_rng':torch.get_rng_state(),'training_state':{'marker':1}}
clean=inference_payload(saved,{'experiment_id':'reviewer-format-probe','revision':'f'*40});check_payload(clean)
assert not any(k in clean for k in ['optimizer','torch_rng','training_state'])
record['inference_export']={'source_keys':list(saved),'clean_keys':list(clean),'check_payload_format':check_payload(clean),'weights_preserved':all(torch.equal(clean['model'][k],v) for k,v in saved['model'].items())}
bad=dict(clean,optimizer={})
try:check_payload(bad)
except ValueError as e:record['inference_export']['reject_training_state']=str(e)
else:raise AssertionError('Training state accepted')
# Complete bounded current implementation inventory for the asserted video helper.
files=sorted(set((ROOT/'tiny_perceptron').rglob('*.py'))|set((ROOT/'scripts').rglob('*.py'))|set((ROOT/'tests').rglob('*.py')))
inventory=[];video_definitions=[]
for path in files:
 b=path.read_bytes();tree=ast.parse(b)
 names=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
 matches=[n for n in names if any(x in n.lower() for x in ['video','frame','slice'])]
 inventory.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'video_frame_slice_definitions':matches})
 video_definitions.extend((str(path.relative_to(ROOT)),n) for n in matches)
record['video_helper_inventory']={'files_inspected':len(files),'matching_definitions':video_definitions,'inventory':inventory,'finding':'No video decoding or per-frame slicing helper in current implementation. Audio time-frame exercises and static-image n_frames validation are separate operations.'}
# One complete notebook execution and read-only CLI help confirm documented entry paths.
commands=[['scripts/check_notebooks.py','--mode','python','--lesson','1.1','--output',str(OUT/'notebook-one')],['-m','ipykernel','install','--help'],['-m','jupyterlab','--help'],['scripts/infer.py','--help'],['scripts/evaluate.py','--help']]
record['commands']=[]
for args in commands:
 cp=subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,text=True,capture_output=True);assert cp.returncode==0
 record['commands'].append({'command':args,'exit':cp.returncode,'stdout':cp.stdout[:6500],'stderr':cp.stderr})
(OUT/'software-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Passed mechanisms and export checks; video-related definitions:',video_definitions)
