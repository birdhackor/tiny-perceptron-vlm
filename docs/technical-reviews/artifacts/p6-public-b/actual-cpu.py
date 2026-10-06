from pathlib import Path
import hashlib, json, random, subprocess,sys, tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tiny_perceptron.selftrained.inference import verify_export,fetch_public_export,attach_public_metadata
from scripts.selftrained.train_local_stage import local_source,source_identity
from scripts.selftrained.train import rng_state,restore_rng,BalancedSampler
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cache=Path('/tmp/p5-native-public-cpu-smoke-actual/public-model')
pins={n:sha(cache/n) for n in ['model.safetensors','model-config.json','tokenizer.json','inference-manifest.json']}
snap=OUT/'public-model-metadata';snap.mkdir(exist_ok=True)
for n in ['model-config.json','tokenizer.json','inference-manifest.json']:(snap/n).write_bytes((cache/n).read_bytes())
manifest,actual=verify_export(cache,manifest_sha256='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e')
from safetensors import safe_open
with safe_open(cache/'model.safetensors',framework='pt',device='cpu') as f:
 keys=list(f.keys());meta=f.metadata()
assert all(not k.startswith(('optimizer','rng','sampler')) for k in keys)
runs=[]
for name,task in [('text','text'),('tool','tool_call')]:
 cmd=[sys.executable,str(ROOT/'scripts/selftrained/chat.py'),'--model-dir',str(cache),'--asset-dir',str(ROOT/'outputs/selftrained-v2/data'),'--manifest-sha256',actual,'--messages',str(ROOT/('docs/selftrained/examples/v2/'+task+'.messages.json')),'--task',task,'--device','cpu','--threads','1','--max-new-tokens','128']
 if task=='tool_call':cmd+=['--tools']
 p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
 (OUT/('current-'+name+'-stdout.json')).write_text(p.stdout);(OUT/('current-'+name+'-stderr.txt')).write_text(p.stderr)
 assert p.returncode==0,p.stderr
 d=json.loads(p.stdout);runs.append({'command':cmd,'cwd':str(ROOT),'returncode':p.returncode,'answer':d['answer'],'generations':len(d['generations']),'stdout_sha256':sha(OUT/('current-'+name+'-stdout.json'))})
 if task=='tool_call':assert d['tool_trace']['executed'] and d['tool_trace']['tool_result']['result']==416
contracts=[]
def rejects(name,fn):
 try:fn()
 except (ValueError,FileNotFoundError) as e:contracts.append({'case':name,'result':'rejected','exception':type(e).__name__,'message':str(e)})
 else:raise AssertionError(name+' unexpectedly accepted')
rejects('mutable revision rejected without fetching',lambda:fetch_public_export('birdhackor/tiny-perceptron-course-models','main','/tmp/uncreated-p6-b'))
rejects('wrong manifest SHA rejected before payload allocation',lambda:verify_export(cache,manifest_sha256='0'*64))
rejects('new audio without original message index in multiple-user history',lambda:attach_public_metadata([{'role':'user','content':'a'},{'role':'assistant','content':'b'},{'role':'user','content':'c'}],audio='audio/example.wav'))
rejects('resume selected best instead of latest',lambda:local_source(Path('/tmp/p5-local-wrapper-independent/final-runs/pretrain/best.pt'),'unused',True))
rejects('fresh init latest instead of selected best',lambda:local_source(Path('/tmp/p5-local-wrapper-independent/final-runs/pretrain/latest.pt'),'unused',False))
state=rng_state();a=(random.random(),float(np.random.random()),torch.rand(4));restore_rng(state);b=(random.random(),float(np.random.random()),torch.rand(4));assert a[:2]==b[:2] and torch.equal(a[2],b[2]);contracts.append({'case':'Python NumPy Torch RNG roundtrip','result':'exactly identical next values'})
records=[{'task':task,'group_id':task+str(i)} for task in ['text','tool_call','vision_clothing','ocr','voice_qa'] for i in range(2)]
s=BalancedSampler(records,20261006,mode='task-family');s.batch(16);state=s.state_dict();a=s.batch(16);t=BalancedSampler(records,20261006,mode='task-family');t.load_state_dict(state);b=t.batch(16);assert a==b
state1=s.state_dict();state2=t.state_dict();assert torch.equal(state1.pop('generator'),state2.pop('generator')) and state1==state2
contracts.append({'case':'task-family sampler state roundtrip','result':'same next 16 rows and subsequent state'})
rev,hashes=source_identity(ROOT);contracts.append({'case':'current wrapper source_identity','result':'tracked source bytes equal HEAD','revision':rev,'tracked_files':len(hashes)})
(OUT/'actual-cpu-summary.json').write_text(json.dumps({'runs':runs,'contracts':contracts,'public_cache_files':pins,'safetensors_metadata':meta,'tensor_key_count':len(keys),'tensor_keys':keys,'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'}},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'runs':runs,'contracts':contracts},ensure_ascii=False))
