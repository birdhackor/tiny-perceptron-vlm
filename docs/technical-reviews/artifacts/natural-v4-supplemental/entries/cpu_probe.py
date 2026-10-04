from pathlib import Path
from dataclasses import asdict
import contextlib,gzip,hashlib,importlib.util,io,json,re,subprocess,sys,tarfile,tempfile,wave
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.capstone import CapstoneModel,default_config
from tiny_perceptron.assets import unpack_asset

record={'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'limits':'Offline CPU audit of existing fixed records plus one default dry run; no GPU, training updates, model/data downloads, or full-book rerun.'}
torch.set_num_threads(2)
text='貓看狗，狗看貓。';chars=sorted(set(text));ids=[chars.index(c) for c in text]
record['character_ids']={'chars':chars,'ids':ids,'restored':''.join(chars[i] for i in ids)}
assert ids==[3,2,1,4,1,2,3,0]
chapters=sorted((ROOT/'course/chapters').glob('*.md'))
headings=re.compile(r'^## ([A-Z\d]+\.\d+) ',re.M)
lesson_ids=[i for p in chapters for i in headings.findall(p.read_text())]
front={n:headings.findall((ROOT/'course'/n).read_text()) for n in ['README.md','first-steps.md','training.md','glossary.md']}
notebooks=list((ROOT/'notebooks').glob('*/*.ipynb'))
record['counts']={'chapter_files':len(chapters),'chapter_lessons':len(lesson_ids),'front':{k:len(v) for k,v in front.items()},'total_numbered':len(lesson_ids)+sum(len(v) for v in front.values()),'notebooks':len(notebooks)}
bad=[]; no_code=[]
for path in notebooks:
 nb=json.loads(path.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code']
 if not code:no_code.append(str(path.relative_to(ROOT)));continue
 first=code[0];src=''.join(first['source'])
 if not ('setup' in src or 'pip' in src):bad.append(str(path.relative_to(ROOT)))
record['notebook_setup_nonmatches']=bad
record['notebooks_without_code']=no_code
assert record['counts']['chapter_lessons']==266 and len(notebooks)==266 and record['counts']['total_numbered']==292 and not bad
model=CapstoneModel(default_config())
record['capstone']={'config':asdict(model.config),'parameters':sum(p.numel() for p in model.parameters()),'description':model.description()}
assert record['capstone']['parameters']==328128
public=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())['models']
cap=json.loads((ROOT/'docs/course-experiments/capstone-public.json').read_text())
record['release_counts']={'experiments':len(public),'weight_files':sum(f['path'].endswith(('.pt','.safetensors')) for m in public for f in m['files']),'capstone_models':len(cap['models']),'capstone_weight_files':sum(f['path'].endswith('.pt') for m in cap['models'] for f in m['files']),'capstone_repo':cap['repo'],'capstone_revision':cap['revision']}
assert record['release_counts']['experiments']==30 and record['release_counts']['weight_files']==120 and record['release_counts']['capstone_models']==11
manifest=json.loads((ROOT/'assets/training/manifest.json').read_text())
assets=[]
for asset in manifest['assets']:
 archive=ROOT/asset['archive'];data=archive.read_bytes()
 assert len(data)==asset['archive_bytes'] and hashlib.sha256(data).hexdigest()==asset['archive_sha256']
 expected={f['path']:f for f in asset['files']};entries={};rows={};rates=[];parquet_rows=None
 with tarfile.open(archive,'r:gz') as t:
  for member in t:
   content=t.extractfile(member).read();e=expected[member.name]
   assert len(content)==e['bytes'] and hashlib.sha256(content).hexdigest()==e['sha256']
   entries[member.name]=content
   if member.name.endswith('.jsonl'):
    js=[json.loads(l) for l in content.decode().splitlines() if l];rows[member.name]={'rows':len(js),'first_keys':list(js[0]) if js else []}
   if member.name.endswith('.wav'):
    with wave.open(io.BytesIO(content)) as w:rates.append(w.getframerate())
   if member.name.endswith('.parquet'):
    import pyarrow.parquet as pq
    parquet_rows=pq.read_metadata(io.BytesIO(content)).num_rows
 assert set(entries)==set(expected)
 rebuilt=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=rebuilt,mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT) as t:
   for name,content in sorted(entries.items()):
    m=tarfile.TarInfo(name);m.size=len(content);m.mode=0o644;m.mtime=0;t.addfile(m,io.BytesIO(content))
 assert rebuilt.getvalue()==data
 assets.append({'id':asset['id'],'training_records':asset['training_records'],'archive_bytes':len(data),'archive_sha256':asset['archive_sha256'],'members_verified':len(entries),'jsonl':rows,'wav_count':len(rates),'wav_rates':sorted(set(rates)),'parquet_rows':parquet_rows,'deterministic_repack_exact':True})
record['assets']={'packages':assets,'package_count':len(assets),'training_sum':sum(a['training_records'] for a in assets),'bytes_sum':sum(a['archive_bytes'] for a in assets),'MiB':sum(a['archive_bytes'] for a in assets)/2**20}
assert record['assets']['training_sum']==1447
# Real small extraction, idempotence and existing-file protection tests in ignored raw research scratch.
research=ROOT/'outputs/natural-v4/factual-research/entries';research.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=research) as temp:
 a=manifest['assets'][0];folder=Path(temp)/'data'
 first=unpack_asset(ROOT/a['archive'],a,folder);second=unpack_asset(ROOT/a['archive'],a,folder)
 changed=folder/a['files'][0]['path'];changed.write_bytes(b'reviewer conflict probe')
 try:unpack_asset(ROOT/a['archive'],a,folder)
 except ValueError as e: conflict=str(e)
 else:raise AssertionError('Different content accepted')
 record['asset_entry_behavior']={'first':first['files_written'],'second':second['files_written'],'files_verified':second['files_verified'],'conflict':conflict}
cmd=[str(ROOT/'.venv/bin/python'),'scripts/prepare_data.py','--kind','toy-text','--output',str(OUT/'generated')]
cp=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert cp.returncode==0
record['prepare']={'command':cmd,'exit':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
# Execute the real main while observing actual model tensors and AdamW.step.
spec=importlib.util.spec_from_file_location('reviewed_train',ROOT/'scripts/train.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
models=[];steps=[];orig=train.TinyLM
def observed_model(*args,**kwargs):
 m=orig(*args,**kwargs);models.append((m,{k:v.clone() for k,v in m.state_dict().items()}));return m
train.TinyLM=observed_model
origstep=torch.optim.AdamW.step
def observed_step(*args,**kwargs):steps.append(True);return origstep(*args,**kwargs)
torch.optim.AdamW.step=observed_step
argv=['scripts/train.py','--task','text','--data',str(OUT/'generated/toy-text/train.jsonl')]
sys.argv=argv;stream=io.StringIO()
with contextlib.redirect_stdout(stream):train.main()
m,before=models[0]
unchanged=all(torch.equal(before[k],v) for k,v in m.state_dict().items())
record['dry_run']={'argv':argv,'stdout':stream.getvalue(),'optimizer_step_calls':len(steps),'weights_exact_unchanged':unchanged,'gradient_tensors':sum(p.grad is not None for p in m.parameters())}
assert not steps and unchanged and record['dry_run']['gradient_tensors']>0
record['commands']={}
for args in [['scripts/check_env.py'],['scripts/fetch_training_assets.py','--list'],['scripts/fetch_course_models.py','--list'],['scripts/check_course_models.py','--help'],['scripts/check_notebooks.py','--help']]:
 cmd=[str(ROOT/'.venv/bin/python'),*args];cp=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert cp.returncode==0
 record['commands'][' '.join(args)]={'exit':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
(OUT/'cpu-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['environment','counts','capstone','release_counts','asset_entry_behavior']},ensure_ascii=False,indent=2))
print('assets totals',record['assets']['training_sum'],record['assets']['bytes_sum'],record['assets']['MiB'])
print('dry run',unchanged,len(steps))
