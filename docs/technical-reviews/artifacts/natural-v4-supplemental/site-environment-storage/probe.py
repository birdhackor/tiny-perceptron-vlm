import ast,hashlib,json,platform,subprocess,sys,tomllib,tempfile
from pathlib import Path
from collections import Counter
import torch, numpy as np, soundfile as sf
from PIL import Image
root=Path.cwd();out=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-environment-storage'
sys.path.insert(0,str(root))
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.training import save_checkpoint,load_checkpoint
import importlib.util
spec=importlib.util.spec_from_file_location('fetch_data',root/'scripts/fetch_natural_data.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','cuda_runtime':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'os':platform.platform()}
record={'environment':env}
conf=tomllib.loads(Path('pyproject.toml').read_text());record['project']={'requires_python':conf['project']['requires-python'],'python_version_file':Path('.python-version').read_text().strip(),'extras':conf['project']['optional-dependencies'],'uv_conflicts':conf['tool']['uv']['conflicts'],'torch_indexes':conf['tool']['uv']['sources']['torch'],'requirements_natural':Path('requirements-natural.txt').read_text()}
a=json.loads(Path('assets/training/manifest.json').read_text())['assets'];v=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());files=[]
for group,items in [('foundation',a),('chapter20',v['archives'])]:
 rows=[]
 for s in items:
  path=s.get('archive',s.get('path'));expected=s.get('archive_bytes',s.get('bytes'));rows.append({'path':path,'manifest_bytes':expected,'stat_bytes':Path(path).stat().st_size,'equal':expected==Path(path).stat().st_size})
 files.append({'group':group,'archives':rows,'count':len(rows),'sum_manifest_bytes':sum(x['manifest_bytes'] for x in rows),'sum_stat_bytes':sum(x['stat_bytes'] for x in rows),'decimal_MB':sum(x['stat_bytes'] for x in rows)/1_000_000})
record['archive_size_recomputation']=files
plan=json.loads(Path('docs/course-experiments/plan.json').read_text());record['experiment_count']={'formal_sequence':len(plan['sequence']),'unique_formal_ids':len({x['id'] for x in plan['sequence']}),'supporting_experiments':len(plan['supporting_experiments'])}
record['natural_data_summary']=f.summary(f.validate_manifest(v),'9a61ecf524c9518f33f1501c28aa72997d4a82d0',digest('docs/natural-assistant/v4/manifest.json'))
record['audio_splits']=dict(Counter(x['split'] for x in v['audio_rows']));record['row_splits']=dict(Counter(x['split'] for x in v['rows']))
with tempfile.TemporaryDirectory(prefix='site-env-storage-probe-') as tmp:
 tmp=Path(tmp);im=Image.new('RGB',(2,3),(15,30,45));im.save(tmp/'image.png');record['pillow_decode']={'size':Image.open(tmp/'image.png').size,'pixel':Image.open(tmp/'image.png').getpixel((0,0))}
 wave=np.array([0.0,0.125,-0.25,0.5],dtype=np.float32);sf.write(tmp/'audio.wav',wave,16000,subtype='FLOAT');decoded,rate=sf.read(tmp/'audio.wav',dtype='float32');record['soundfile_decode']={'rate':rate,'samples':decoded.tolist(),'exact_equal':bool(np.array_equal(wave,decoded))}
 model=TinyLM(ModelConfig(vocab_size=264,width=16,heads=2,layers=1,max_length=16));optimizer=torch.optim.AdamW(model.parameters());x=torch.tensor([[1,2,3]]);model(x)["logits"].sum().backward();optimizer.step();save_checkpoint(tmp/'state.pt',model,optimizer,step=1,training_state={'data_cursor':3});loaded,payload=load_checkpoint(tmp/'state.pt');record['checkpoint']={'saved_keys':sorted(payload),'step':payload['step'],'training_state':payload['training_state'],'optimizer_nonempty':bool(payload['optimizer']['state']),'weights_exact_equal':all(torch.equal(t,loaded.state_dict()[k]) for k,t in model.state_dict().items())}
 z=torch.ones(3,4,requires_grad=True);(z@z.T).sum().backward();record['basic_compute']={'shape':list(z.shape),'gradient':z.grad.tolist(),'exact_equal_to_6':bool(torch.equal(z.grad,torch.full_like(z,6)))}
for cmd in [['.venv/bin/python','scripts/fetch_training_assets.py','--list'],['.venv/bin/python','scripts/fetch_natural_data.py','--manifest','docs/natural-assistant/v4/manifest.json','--revision','9a61ecf524c9518f33f1501c28aa72997d4a82d0','--manifest-sha256',digest('docs/natural-assistant/v4/manifest.json'),'--list'],['.venv/bin/python','scripts/fetch_natural_release.py','--manifest','docs/natural-assistant/v4/public-release.json','--list'],['uv','sync','--frozen','--extra','cpu','--group','notebook','--offline','--dry-run'],['.venv/bin/python','-m','jupyterlab','--version']]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);record.setdefault('commands',[]).append({'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
(out/'probe-output.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
