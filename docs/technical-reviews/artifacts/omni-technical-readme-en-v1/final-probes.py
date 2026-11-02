"""Inspect exact embedded command text, dry-run update guards and live read-only evidence."""
import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

import torch
import scripts.train as train
from ipykernel.kernelspec import InstallIPythonKernelSpecApp

ROOT = Path('/workspace/tiny-perceptron-vlm')
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-readme-en-v1/final-probes')
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(2)
result = {'created_at':datetime.now(UTC).isoformat(),'environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'device':'cpu','threads':'2'}}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = (ROOT/'docs/course-revision-20261029-omni/reviews/auxiliary-english-v2/source.md').read_text()
blocks = re.findall(r'```bash\n(.*?)```', source, re.S)
result['embedded_bash'] = []
for i,block in enumerate(blocks):
    p = CACHE/f'block-{i}.sh';p.write_text(block)
    check = subprocess.run(['bash','-n',str(p)],capture_output=True,text=True,timeout=10)
    result['embedded_bash'].append({'index':i,'sha256':sha(p),'syntax_returncode':check.returncode,'line_count':len(block.splitlines())})
bad=[]
for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',source):
    link=match[1]
    if '://' not in link and not (ROOT/link.split('#')[0]).is_file():bad.append(link)
result['readme_relative_links_missing']=bad

instances=[]
Original = train.TinyLM
def tensor_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
class Captured(Original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.before_hash=tensor_hash(self);instances.append(self)
def forbidden(*args,**kwargs):raise AssertionError('Dry run attempted optimizer update or checkpoint save')
argv=['scripts/train.py','--task','text','--data','/workspace/work/omni-technical-readme-en-v1/cpu-run2/command-cwd/data/generated/toy-text/train.jsonl']
stdout=io.StringIO()
with patch.object(sys,'argv',argv),patch.object(train,'TinyLM',Captured),patch.object(torch.optim.AdamW,'step',forbidden),patch.object(train,'save_checkpoint',forbidden),patch.object(torch,'save',forbidden),contextlib.redirect_stdout(stdout):train.main()
p=CACHE/'instrumented-dry-run.txt';p.write_text(stdout.getvalue())
result['dry_run_guards']={'argv':argv,'optimizer_step_and_checkpoint_save_forbidden':True,'before_hash':instances[0].before_hash,'after_hash':tensor_hash(instances[0]),'parameters_unchanged':instances[0].before_hash==tensor_hash(instances[0]),'raw_stdout_path':str(p),'raw_stdout_sha256':sha(p)}

result['read_only_fetches']=[]
for name,url in [
    ('live-1.1.html','https://birdhackor.github.io/tiny-perceptron-vlm/1.1/'),
    ('ci-run.json','https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/37071683289'),
    ('saving-models.html','https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html'),
    ('hf-license','https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/LICENSE'),
]:
    try:
        with urlopen(Request(url,headers={'User-Agent':'technical-readme-en-independent-review'}),timeout=20) as response:raw=response.read()
        p=CACHE/name;p.write_bytes(raw)
        entry={'url':url,'path':str(p),'bytes':len(raw),'sha256':sha(p),'accessed_at':datetime.now(UTC).isoformat(),'ok':True}
        text=raw.decode('utf-8',errors='replace')
        if name=='live-1.1.html':entry['markers']={key:key in text for key in ['CPU','output-label','colab.research.google.com','.ipynb','mathjax/tex-chtml.js']}
        if name=='ci-run.json':
            x=json.loads(text);entry['run']={k:x[k] for k in ['id','head_sha','status','conclusion','run_started_at']}
        if name=='saving-models.html':entry['checkpoint_markers']={key:key in text for key in ['optimizer_state_dict','general checkpoint','state_dict']}
        if name=='hf-license':entry['mit_license']='MIT License' in text
    except Exception as e:entry={'url':url,'ok':False,'error':str(e)}
    result['read_only_fetches'].append(entry)

path=Path(inspect.getsourcefile(InstallIPythonKernelSpecApp));lines,start=inspect.getsourcelines(InstallIPythonKernelSpecApp)
result['installed_ipykernel']={'source_path':str(path),'sha256':sha(path),'start_line':start,'sys_prefix_supported':'sys-prefix' in ''.join(lines)}
result['source_files']={p:sha(ROOT/p) for p in ['README_en.md','pyproject.toml','uv.lock','scripts/check_env.py','scripts/train.py','scripts/prepare_data.py','scripts/course_release.py','tiny_perceptron/model.py','tiny_perceptron/training.py','tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/selftrained/dataset.py','tiny_perceptron/lesson_products/inference.py','course/web/course.js','course/web/course.css','course/web/mathjax.js','course/web/mathjax-assets.json','zensical.toml','LICENSE','.gitattributes','.gitignore','.github/workflows/ci.yml','.github/workflows/pages.yml','.github/workflows/gpu-smoke.yml','docs/natural-assistant/v4/STUDENT.md','docs/natural-assistant/v4/public-release.json','docs/validation-artifacts/release-compatibility-ci.json','docs/validation.md']}
out=HERE/'final-probes-summary.json'
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'artifact':str(out),'sha256':sha(out),'dry_run_guards':result['dry_run_guards'],'read_only_fetches':result['read_only_fetches']},ensure_ascii=False))
