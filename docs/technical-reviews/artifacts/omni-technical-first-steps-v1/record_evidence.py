"""Store small primary-source excerpts and independently inspected mapping receipts."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import inspect
import json
import re

import torch

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
CACHE=Path('/workspace/work/reference/technical-first-steps-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Plain(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,t,a):
        if t in ['script','style']:self.skip+=1
    def handle_endtag(self,t):
        if t in ['script','style']:self.skip=max(0,self.skip-1)
    def handle_data(self,t):
        if not self.skip:self.parts.append(t)

receipts=[]
for name in ['fetch-receipt.json','fallback-fetch-receipt.json','additional-fetch-receipt.json','final-fetch-receipt.json']:
    if (CACHE/name).exists():receipts.extend(json.loads((CACHE/name).read_bytes()))
queries={
 'python-introduction':['Division (','used to assign','index'],
 'python-controlflow':['The :keyword:`for` statement','The :func:`range` Function','Defining Functions','Lambda Expressions'],
 'python-datastructures':['List Comprehensions','Tuples and Sequences','Sets\n','Dictionaries\n'],
 'python-errors':['Syntax Errors\n','Handling Exceptions\n'],
 'python-functions':['.. function:: enumerate','.. function:: next','.. function:: sorted','.. function:: zip','.. function:: round','.. function:: max','.. function:: sum'],
 'python-stdtypes':['always rounded towards minus infinity','Strings are immutable','A left shift by'],
 'python-pathlib':['.. classmethod:: Path.cwd','.. method:: Path.exists'],
 'python-math':['.. function:: log','invalid operations like'],
 'python-venv':['This will prepend','Activate.ps1','not need to activate'],
 'python-interpreter':['primary prompt'],
 'uv-sync-source':['To use the lockfile without','### Syncing optional dependencies','### Syncing development dependencies'],
 'uv-python-source':['By default, uv will automatically download'],
 'git-lfs-config':['* `GIT_LFS_SKIP_SMUDGE`'],
 'git-lfs-install':['`--local`::'],
 'git-mingw-source':['int are_long_paths_enabled'],
 'ipykernel-install-source':['The last command installs','The ``--name`` value'],
 'jupyterlab-interface-source2':['# The JupyterLab Interface'],
 'jupyterlab-notebook-source':['Shift','kernel','Restart'],
 'probability-authority':['The probability of an event is a number'],
 'openstax-independence':['Two events A','independent if','P(A AND B)'],
 'openstax-derivative':['derivative of a function','difference quotient'],
 'colab-faq':['Colab allows you','loaded from GitHub'],
 'powershell-policy':['PowerShell\'s execution policy','Restricted Users'],
 'course-website':['實際執行結果 · CPU','下載本節 .ipynb']
}
small=[]
for row in receipts:
    if not row.get('ok') or row['id'] not in queries:continue
    p=Path(row['path']);text=p.read_text()
    html=text.lstrip().startswith('<')
    if html:
        parser=Plain();parser.feed(text);text=re.sub(r'\s+',' ',' '.join(parser.parts))
    snippets=[]
    for term in queries[row['id']]:
        match=re.search(re.escape(term),text,re.I)
        if match:
            n=match.start();snippets.append({'query':term,'locator':('normalized HTML text character '+str(n)) if html else ('line '+str(text[:n].count('\n')+1)),
                                             'excerpt':text[max(0,n-30):n+650]})
    small.append({**row,'excerpts':snippets})

torch_receipts=[]
for name,obj in [('tensor_item',torch.Tensor.item),('tensor_tolist',torch.Tensor.tolist),('tensor_reshape',torch.Tensor.reshape),
                 ('tensor_mean',torch.mean),('matmul',torch.matmul),('softmax',torch.nn.functional.softmax),
                 ('cross_entropy',torch.nn.functional.cross_entropy),('CrossEntropyLoss',torch.nn.CrossEntropyLoss),
                 ('Linear',torch.nn.Linear),('no_grad',torch.no_grad),('backward',torch.Tensor.backward),
                 ('copy_',torch.Tensor.copy_),('requires_grad_',torch.Tensor.requires_grad_)]:
    row={'api':name,'version':torch.__version__,'git':torch.version.git_version,
         'excerpt':(obj.__doc__ or '')[:(2500 if name=='CrossEntropyLoss' else 1000)]}
    try:
        p=Path(inspect.getsourcefile(obj));row.update({'installed_source_path':str(p),'source_sha256':sha(p),
            'line':inspect.getsourcelines(obj)[1],'upstream_path':str(p).split('/site-packages/')[1]})
    except (TypeError,OSError):row['upstream_path']='torch/_tensor_docs.py' if name.startswith('tensor_') or name in ['copy_','requires_grad_'] else 'torch/_torch_docs.py'
    torch_receipts.append(row)

target=OUT/'primary-source-receipts-v1.json'
with target.open('x') as f:json.dump({'created_at':datetime.now(timezone.utc).isoformat(),'retention':'Third-party full texts remain outside Git; only bounded supporting quotations are stored here.',
                                  'official_sources':small,'fetch_failures':[x for x in receipts if not x.get('ok')], 'installed_pytorch':torch_receipts},f,ensure_ascii=False,indent=2);f.write('\n')

body=(ROOT/'docs/course-review-20261019/freeze-02/sources/first-steps.md').read_text()
mappings=[]
for line_number,line in enumerate(body.splitlines(),1):
    if not line.startswith('| `tiny_perceptron/') or line.split('`')[1]=='tiny_perceptron/':continue
    module=line.split('`')[1];entries=[]
    for label,ch,start,end in re.findall(r'\[([^\]]+)\]\((chapters/[^)#]+)#([^)]+)\)(?:[–-]([A-Z\d]+\.\d+))?',line):
        ids=[start]
        if end:
            prefix=start.rsplit('.',1)[0];ids=[prefix+'.'+str(n) for n in range(int(start.rsplit('.',1)[1]),int(end.rsplit('.',1)[1])+1)]
        p=ROOT/'course'/ch;t=p.read_text()
        for lesson in ids:
            match=re.search(r'^## '+re.escape(lesson)+r' .*?(?=^## |\Z)',t,re.M|re.S)
            section=match[0] if match else ''
            mentions=[{'section_line':i+1,'text':v[:600]} for i,v in enumerate(section.splitlines()) if module in v or module.rstrip('/').split('/')[-1] in v]
            entries.append({'lesson_id':lesson,'source':str(p.relative_to(ROOT)),'source_sha256':sha(p),
                            'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'heading':section.splitlines()[0] if section else None,'mentions':mentions})
    mappings.append({'source_line':line_number,'source_quote':line,'module':module,'entries':entries})
additional={'fstring':f'看到{"貓"}','assert_false':None}
try:assert False
except AssertionError:additional['assert_false']='AssertionError'
result={'created_at':datetime.now(timezone.utc).isoformat(),'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/record_evidence.py',
        'environment':{'python':__import__('platform').python_version(),'torch':torch.__version__,'device':'cpu'},
        'module_mappings':mappings,'additional_python_examples':additional,
        'independent_finding':'12.13 contains its own explicit STFT/mel calculation and never names or excerpts natural_concepts.py; W.7 range 12.13–12.14 is stale for 12.13.',
        'scope':'Linked section reading is limited to verifying this file-to-section lookup table, not a technical acceptance of those lessons.'}
target2=OUT/'mapping-inspection-v1.json'
with target2.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')

figures=[]
for name in ['rewrite-W-3-axes','rewrite-W-6-gradient-step']:
    source=ROOT/'docs/course-publish-20261008/freeze/current/figures/course/figures'/(name+'.svg')
    png=Path('/workspace/work/omni-technical-first-steps')/(name+'.png')
    figures.append({'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'render_path_outside_git':str(png),'render_sha256':sha(png),
        'command':'inkscape '+str(source.relative_to(ROOT))+' --export-type=png --export-filename='+str(png),'visually_viewed_with':'tools.view_image after rendering',
        'inspection': '2x3 scores and dim=1 / dim=0 arrows and means all agree with frozen W.3.' if name.endswith('axes') else 'w=1,L=4 and w=1.4,L=2.56 lie on the parabola; arrow goes right/down; minimum at w=3. SVG coordinates implement x=75+123w, y=430-34L (point residual zero). No axis unit distortion of the stated values.'})
target3=OUT/'figure-inspection-v1.json'
with target3.open('x') as f:json.dump({'created_at':datetime.now(timezone.utc).isoformat(),'figures':figures},f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [target,target2,target3]],
                  'mappings_without_explicit_name':[(x['module'],e['lesson_id']) for x in mappings for e in x['entries'] if not e['mentions']]},ensure_ascii=False))
