from pathlib import Path
import argparse, contextlib, hashlib, io, json, math, re, runpy, shlex, subprocess, sys, tempfile
import torch
ROOT=Path(__file__).resolve().parents[5];DEST=Path(__file__).parent;sys.path.insert(0,str(ROOT))
out={'environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'device':'cpu'},'cli_parse_checks':[]}
class Parsed(BaseException):pass
parse=argparse.ArgumentParser.parse_args
def intercepted(parser,*a,**kw):
    args=parse(parser,*a,**kw);out['last_parsed']=vars(args);raise Parsed()
raw=(ROOT/'course/training.md').read_text()
blocks=re.findall(r'```bash\n(.*?)```',raw,re.S)
for block in blocks:
 for line in block.strip().splitlines():
  parts=shlex.split(line)
  if not parts or not parts[0].endswith('python'):continue
  parts=parts[:parts.index('>')] if '>' in parts else parts
  if parts[1]=='-m':module=parts[2];argv=parts[3:];is_module=True
  elif parts[1].startswith('scripts/'):module=parts[1];argv=parts[2:];is_module=False
  else:continue
  sys.argv=[module]+argv;argparse.ArgumentParser.parse_args=intercepted
  try:
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    if is_module:runpy.run_module(module,run_name='__main__')
    else:runpy.run_path(str(ROOT/module),run_name='__main__')
  except Parsed:out['cli_parse_checks'].append({'document_command':line,'parsed':out.pop('last_parsed'),'passed':True})
  except BaseException as e:out['cli_parse_checks'].append({'document_command':line,'passed':False,'error':repr(e)})
  finally:argparse.ArgumentParser.parse_args=parse
for entry in out['cli_parse_checks']:
 entry['parsed']={k:str(v) if isinstance(v,Path) else v for k,v in entry.get('parsed',{}).items()}

# Control the runtime thread count in the child process before entering unchanged original CLI.
# Training main preserves min(4, current_threads), so this probes single-thread resumption.
with tempfile.TemporaryDirectory(dir=DEST,prefix='resume-fixture-') as temp:
 t=Path(temp);calls=[]
 def run(extra):
  base=['--task','sft','--device','cpu','--seed','42','--train','--steps','4','--save-every','1','--width','8','--max-length','32','--batch-size','1']
  src='import sys,torch,runpy; torch.set_num_threads(1); sys.argv='+repr(['scripts/train.py']+base+extra)+"; runpy.run_path('scripts/train.py',run_name='__main__')"
  p=subprocess.run([sys.executable,'-c',src],cwd=ROOT,capture_output=True,text=True,timeout=30)
  calls.append({'command':shlex.join([sys.executable,'-c',src]),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 run(['--stop-after','2','--output',str(t/'full.pt')]);run(['--stop-after','1','--output',str(t/'part.pt')]);run(['--stop-after','2','--checkpoint',str(t/'part.pt'),'--resume','--output',str(t/'resumed.pt')])
 a=torch.load(t/'full.pt',weights_only=True);b=torch.load(t/'resumed.pt',weights_only=True)
 differences={k:float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model']}
 out['single_thread_resume']={'expected_max_weight_difference':0.0,'observed_max_weight_difference':max(differences.values()),'passed':max(differences.values())==0,'calls':calls,'details':'Original main, same total4 schedule;2 fixture updates; single-thread CPU; original four-thread initial difference3.5762786865234375e-7 preserved.'}

from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.alignment import distillation_kl,dpo_loss
from tiny_perceptron.quantization import fake_quantize
torch.set_num_threads(1)
model=MultiModalLM(TinyLM(ModelConfig(width=8,layers=2)))
model.requires_grad_(False);model.image_projector.requires_grad_(True);model.audio_projector.requires_grad_(True);model.language.blocks[0].requires_grad_(True);model.language.blocks[-1].requires_grad_(True)
names=[n for n,p in model.named_parameters() if p.requires_grad]
out['CLI_partial_freeze']={'language_blocks_enabled':sorted(set(n.split('.')[2] for n in names if n.startswith('language.blocks.'))),'embedding_enabled':model.language.embedding.weight.requires_grad,'output_enabled':model.language.output.weight.requires_grad,'image_projector_enabled':all(p.requires_grad for p in model.image_projector.parameters()),'audio_projector_enabled':all(p.requires_grad for p in model.audio_projector.parameters()),'encoder_enabled':any(p.requires_grad for p in list(model.vision.parameters())+list(model.audio.parameters()))}
x=torch.tensor([.1,.4,.9],requires_grad=True);y=fake_quantize(x,4);y.sum().backward();out['QAT_STE']={'forward':y.tolist(),'gradient':x.grad.tolist(),'expected_gradient':[1,1,1]}
student=torch.tensor([[[0.,0.],[20.,-20.]]],requires_grad=True);teacher=torch.tensor([[[.8,.2],[.2,.8]]]).log();labels=torch.tensor([[0,-100]])
v=distillation_kl(student,teacher,labels,temperature=1);v.backward();expected=.8*math.log(.8/.5)+.2*math.log(.2/.5)
out['teacher_KL_direction_mask']={'expected':expected,'observed':v.item(),'tolerance':1e-6,'passed':abs(v.item()-expected)<1e-6,'ignored_row_gradient':student.grad[0,1].tolist()}
out['units_derivations']={'MiB':2**20,'byte_bits':8,'rate_time_equality':[4591/8000,9182/16000],'PCM16_scale':-4096/32768,'T11_hypothetical':[80000-50000,4/5-4/5,80000-60000,3/5-4/5],'pass_k_n_equals_k':{str(k):[1-math.comb(k-c,k)/math.comb(k,k) if k-c>=k else 1 for c in range(k+1)] for k in [1,2,4,8]}}
(DEST/'additional-cpu-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'CLI_commands':len(out['cli_parse_checks']),'CLI_failures':[c for c in out['cli_parse_checks'] if not c['passed']],'single_thread_resume':{k:v for k,v in out['single_thread_resume'].items() if k!='calls'},'KL_probe':out['teacher_KL_direction_mask']},ensure_ascii=False,indent=2))
