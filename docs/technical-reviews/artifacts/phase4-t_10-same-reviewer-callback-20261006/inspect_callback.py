"""Same-reviewer bounded T.10 callback; no full schedule or new saved weights."""
import ast
import contextlib
import hashlib
import html
import importlib.util
import io
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent
OLD = ROOT / 'docs/technical-reviews/artifacts/phase4-t_10-factual-independent-20261005-fresh'
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from torch.nn.modules import loss as torch_loss
from scripts.course_experiments import run, modalities
from scripts.course_experiments.common import Context
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def dump(name, value):
    OUT.joinpath(name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def section(path, lesson):
    raw = path.read_bytes()
    hs = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    ix = next(i for i, h in enumerate(hs) if h[0].startswith(f'## {lesson} '.encode()))
    return raw[hs[ix].start():hs[ix+1].start() if ix+1 < len(hs) else len(raw)]

torch.set_num_threads(1)
env = {'python': sys.version, 'python_executable': sys.executable, 'torch': str(torch.__version__),
       'torch_git_version': str(torch.version.git_version), 'device': 'cpu',
       'cuda_available': str(torch.cuda.is_available()), 'cwd': str(pathlib.Path.cwd()),
       'revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True, cwd=ROOT).strip()}
dump('environment.json', env)
body = section(ROOT / 'course/training.md', 'T.10')
assert sha(body) == '6e9fd45920966744b4332fff902c22f877a25d33574c01f4236be397ff54a6f0'
assert body == OUT.joinpath('current/T.10.md').read_bytes()
fences = re.findall(rb'```bash\n(.*?)```', body, re.S)
assert len(fences) == 3
prior_fences = re.findall(rb'```bash\n(.*?)```', OLD.joinpath('frozen/T.10.md').read_bytes(), re.S)
assert fences[:2] == prior_fences
OUT.joinpath('current/new-modal-recipe.sh').write_bytes(fences[2])
syntax = subprocess.run(['bash','-n',str(OUT / 'current/new-modal-recipe.sh')], capture_output=True, text=True)
assert syntax.returncode == 0

# Hash-only inspection of our own preceding proof metadata; no other report judgment read.
prior = json.loads(OUT.joinpath('prior-opaque/T.10.json').read_bytes())
artifact_checks = []
for item in prior['artifacts']:
    actual = sha(ROOT.joinpath(item['path']).read_bytes())
    artifact_checks.append({'id':item['id'], 'path':item['path'], 'expected_sha256':item['sha256'],
                            'actual_sha256':actual, 'same':actual == item['sha256']})
assert all(x['same'] for x in artifact_checks)
code_checks = json.loads(OUT.joinpath('initial-code-fingerprints.json').read_bytes())
code_live = []
for path in ('tiny_perceptron/model.py','tiny_perceptron/tokenization.py','tiny_perceptron/alignment.py',
             'tiny_perceptron/training.py','tiny_perceptron/data.py','scripts/fetch_training_assets.py',
             'scripts/infer.py','docs/course-experiments/README.md','scripts/course_experiments/run.py',
             'scripts/course_experiments/compression.py','scripts/course_experiments/common.py'):
    a = sha(ROOT.joinpath(path).read_bytes()); b = sha(OLD.joinpath('code',path).read_bytes())
    code_live.append({'path':path,'current_sha256':a,'prior_snapshot_sha256':b,'same':a == b})
assert all(x['same'] for x in code_live)
raw_result=json.loads(OLD.joinpath('raw/distillation.json').read_bytes())
extra_path='tiny_perceptron/multimodal.py'
current_extra_sha=sha(ROOT.joinpath(extra_path).read_bytes())
original_extra_sha=raw_result['code_sha256'][extra_path]
assert current_extra_sha==original_extra_sha
code_live.append({'path':extra_path,'current_sha256':current_extra_sha,'original_raw_code_sha256':original_extra_sha,
                  'original_pointer':'raw/distillation.json /code_sha256/tiny_perceptron~1multimodal.py','same':True})
context_checks = []
for lesson, path in [('T.4','course/training.md'),('T.5','course/training.md'),('T.8','course/training.md'),
                     ('18.8','course/chapters/18.md'),('18.9','course/chapters/18.md'),('18.13','course/chapters/18.md')]:
    a=section(ROOT/path,lesson); b=OLD.joinpath(f'frozen/{lesson}.md').read_bytes()
    context_checks.append({'lesson':lesson,'current_sha256':sha(a),'original_sha256':sha(b),'same':a == b})
    if a != b:
        OUT.joinpath(f'current/changed-context-{lesson}.md').write_bytes(a)
svg='course/figures/rewrite-18-modal-answer-rows.svg'
svg_entry = next(x for x in prior['artifacts'] if x['id'] == 'ancillary-svg')
oldsvg = ROOT / svg_entry['path']
assert sha(ROOT.joinpath(svg).read_bytes()) == sha(oldsvg.read_bytes())
sourceinspection=json.loads(OLD.joinpath('source-inspection.json').read_bytes())
official_checks=[]
for key, path in [('torch_functional', pathlib.Path(F.__file__)),('torch_loss',pathlib.Path(torch_loss.__file__))]:
    a=sha(path.read_bytes()); b=sourceinspection[key]['same_sha256']
    official_checks.append({'source':key,'installed_path':str(path),'current_sha256':a,'prior_original_sha256':b,'same':a==b})
assert all(x['same'] for x in official_checks)
assert str(torch.version.git_version) == '5c4886908584029761b579af026dcfb627c84070'
dump('reuse-fingerprints.json',{'own_prior_artifact_checks':artifact_checks,'live_implementation_checks':code_live,
    'context_checks':context_checks,'official_installed_checks':official_checks,
    'unchanged_18_13_svg':{'path':svg,'sha256':sha(oldsvg.read_bytes())},
    'unchanged_short_fences':[sha(x) for x in fences[:2]],
    'reuse_scope':'Only personally performed original source reading, original raw recount, original inference and bounded CPU contract. Exact input, code, source and output hashes checked; no prior verdict used as current proof. No original training schedule rerun.'})

# Save actual scoped code inspection ranges. Entire source snapshots retained unchanged.
ranges={
 'scripts/course_experiments/modalities.py':[(27,53),(73,92),(130,153),(219,283),(367,384),(656,675),(691,735),(913,963)],
 'scripts/course_experiments/capstone.py':[(37,42),(70,143),(280,305)],
 'tiny_perceptron/capstone.py':[(24,25),(644,652)],
 'scripts/train.py':[(37,71)],
 'tiny_perceptron/multimodal.py':[(109,153)],
 'scripts/course_experiments/run.py':[(58,88),(166,198)],
 'scripts/course_experiments/common.py':[(19,30)],
 'scripts/course_experiments/compression.py':[(55,67),(110,128),(991,1069)]}
code_inspection=[]
for path, intervals in ranges.items():
    raw=ROOT.joinpath(path).read_bytes(); text=raw.decode(); lines=text.splitlines(keepends=True)
    tree=ast.parse(text)
    target=OUT / 'current/code' / path
    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    excerpt=''.join(f'### {path}:{start}-{end}\n'+''.join(lines[start-1:end]) for start,end in intervals)
    inspect_path=OUT / ('read-'+path.replace('/','__')+'.txt'); inspect_path.write_text(excerpt)
    code_inspection.append({'path':path,'snapshot':str(target.relative_to(ROOT)),'whole_sha256':sha(raw),
        'read_ranges':intervals,'actual_read_excerpt':str(inspect_path.relative_to(ROOT)),
        'excerpt_sha256':sha(inspect_path.read_bytes()),'AST_method_positions':[
        {'name':n.name,'line':n.lineno,'end':n.end_lineno} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
        and any(start<=n.lineno<=end or n.lineno<=start<=n.end_lineno for start,end in intervals)]})
plan=json.loads(ROOT.joinpath('docs/course-experiments/plan.json').read_bytes())
selected=[]
for ident in ('sft','encoders','projector','vqa','joint','multimodal_distillation','capstone_joint'):
    spec=run.experiment_spec(ident)
    selected.append({k:spec[k] for k in ('id','module','function','dependencies','assets')})
assert next(x for x in selected if x['id']=='joint')['dependencies'] == ['sft','encoders']
assert next(x for x in selected if x['id']=='multimodal_distillation')['dependencies'] == ['vqa','joint']
dump('actual-source-inspection.json',{'reviewer_task':'/root/phase4_factual_coordinator/factual_t_10',
    'current_body_sha256':sha(body),'code_inspection':code_inspection,'named_plan_entries':selected,
    'original_source_reuse':'Primary paper/API original scopes were personally read on 2026-10-05 by this same reviewer; unchanged exact files/installed version confirmed in reuse-fingerprints.json, no re-fetch or other review used.',
    'actual_full_current_sections':['T.10','T.4','T.6','12.12'],
    'author_prose_policy':'Only necessary method ranges, native selected registry contracts, and current textbook claims; no extra result repair/summary or other reviewer body.'})

dispatch=[]
def stub_execute(*args,**kwargs):
    dispatch.append({'experiment':args[0],'device':args[1],'output':str(args[2]),'dependencies':str(args[3]),
                     'assets':str(args[4]),'step_scale':kwargs['step_scale']})
    return {'experiment_id':args[0],'elapsed_seconds':0,'evidence_status':'intercepted-parser-only','artifacts':[]}
for line in fences[2].decode().splitlines():
    argv=shlex.split(line)
    assert argv[:3] == ['.venv/bin/python','-m','scripts.course_experiments.run']
    with patch.object(sys,'argv',[argv[2]]+argv[3:]),patch.object(run,'execute',stub_execute),contextlib.redirect_stdout(io.StringIO()):
        run.main()
assert [x['experiment'] for x in dispatch] == ['joint','multimodal_distillation']
assert all(x['device']=='cuda' and x['step_scale']==1 and x['dependencies']==str(ROOT/'outputs/course-experiments/course-v1') for x in dispatch)

with tempfile.TemporaryDirectory(prefix='t10-callback-') as temp:
    temp=pathlib.Path(temp); output=temp/'joint'; output.mkdir(); deps=temp/'dependencies'
    ctx=Context('cpu',output,deps,temp/'assets',42)
    missing=deps/'vqa/model.pt'
    try:
        ctx.dependency('vqa')
        raise AssertionError('missing artifact must not be synthesized')
    except FileNotFoundError as e:
        missing_guard={'exception':str(e),'created':missing.exists()}
    gpuout=temp/'gpu-output'
    assert not torch.cuda.is_available()
    try:
        run.execute('joint','cuda',gpuout,deps,temp/'assets')
        raise AssertionError('must reject CUDA on this CPU environment')
    except RuntimeError as e:
        gpu_guard={'exception':str(e),'output_created':gpuout.exists()}
    assert not gpu_guard['output_created']
    # Input placeholders only; in-memory random models and encoder states stand in for trained teachers.
    for name in ('sft/model.pt','encoders/vision.pt','encoders/audio.pt'):
        target=deps/name; target.parent.mkdir(parents=True,exist_ok=True); target.touch()
    language=TinyLM(ModelConfig(width=8,layers=1,heads=2,max_length=128))
    donor=MultiModalLM(TinyLM(ModelConfig(width=8,layers=1,heads=2,max_length=128)))
    dependency_calls=[]; fit_calls=[]; held_model=[]
    original_dependency=Context.dependency
    def trace_dependency(self,experiment,filename='model.pt'):
        dependency_calls.append(f'{experiment}/{filename}')
        return original_dependency(self,experiment,filename)
    def fake_load(path,**kwargs):
        return {'encoder':getattr(donor,path.stem).state_dict()}
    def intercept_fit(model,calculate,ctx,steps,**kwargs):
        held_model.append(model)
        fit_calls.append({'steps':steps,'task':kwargs['task'],'save_intercepted':True})
        return {'intercepted':True}
    with patch.object(Context,'dependency',trace_dependency),patch.object(modalities,'load_checkpoint',lambda p,d:(language,{})),\
         patch.object(torch,'load',fake_load),patch.object(modalities,'_fit',intercept_fit),\
         patch.object(modalities,'_evaluate',lambda *a,**k:{}):
        modalities.run_joint(ctx)
    assert dependency_calls == ['sft/model.pt','encoders/vision.pt','encoders/audio.pt']
    assert not (deps/'vqa').exists()
    assert fit_calls == [{'steps':16,'task':'joint','save_intercepted':True}]
    data=json.loads(output.joinpath('dataset.json').read_bytes())
    assert {k:len(v) for k,v in data.items()} == {'train':24,'validation':12,'test':12}
    model=held_model[0]
    state_before=[x.detach().clone() for x in model.parameters()]
    loss, count=modalities._loss_fn(model,[data['train'][0]],ctx,batch=1)(0)
    loss.backward()
    gradient={name:bool(getattr(model,name).weight.grad is not None and torch.isfinite(getattr(model,name).weight.grad).all()
                        and getattr(model,name).weight.grad.abs().sum()>0) for name in ('image_projector','audio_projector')}
    assert all(gradient.values())
    assert all(torch.equal(a,b) for a,b in zip(state_before,model.parameters(),strict=True))
    assert not list(output.glob('*.pt'))
    real_joint={'dependency_calls':dependency_calls,'data_split_counts':{k:len(v) for k,v in data.items()},
        'fit_interception':fit_calls,'original_cuda_step_budget':modalities._steps(Context('cuda',output,deps,temp/'assets'),400,16),
        'vqa_absent':not (deps/'vqa').exists(),'finite_loss':bool(torch.isfinite(loss)),'effective_answer_targets':count,
        'projector_gradients_nonzero':gradient,'optimizer_updates':0,'new_saved_weight_files':0,
        'scope':'Original run_joint body, original _modal/_manifest/_loss_fn; fixture TinyLM width8 and ephemeral random encoder states replace trained inputs; _fit/save and every evaluation intercepted. One forward/backward only; no new quality score.'}

# Exact manual parser call with source input paths, no train() invocation.
train_spec=importlib.util.spec_from_file_location('t10_manual_train',ROOT/'scripts/train.py')
train_module=importlib.util.module_from_spec(train_spec);train_spec.loader.exec_module(train_module)
manual_argv=['scripts/train.py','--task','joint','--checkpoint','checkpoints/attributes.pt','--vision-encoder','checkpoints/vision-encoder.pt',
             '--audio-encoder','checkpoints/audio-encoder.pt','--freeze','partial','--train','--steps','500','--output','checkpoints/joint.pt']
with patch.object(sys,'argv',manual_argv):
    parsed=train_module.parser().parse_args()
manual={k:str(getattr(parsed,k)) for k in ('task','checkpoint','vision_encoder','audio_encoder','freeze','steps','output')}
dump('bounded-new-recipe-contract.json',{'environment':env,'bash_syntax':{'exit':syntax.returncode,'stderr':syntax.stderr},
    'current_fence_sha256':sha(fences[2]),'CLI_exact_arguments_dispatch_intercepted':dispatch,
    'missing_dependency_guard':missing_guard,'actual_CUDA_guard':gpu_guard,'original_joint_CPU_contract':real_joint,
    'manual_recipe_parser_only':manual,'capstone_method_scope':'Read actual stage-joint route and capstone-v1 loader; plan predecessor capstone_sft, default600/lr.0015/batch24; no capstone training run.',
    'excluded':'No original long shell executed, GPU, download, full training, mature score, or new serialized weights. No assertion that fixture model quality reproduces trained teacher.'})

with urllib.request.urlopen('http://127.0.0.1:8765/training.html') as response:
    site=response.read(); status=response.status; mime=response.headers.get('Content-Type')
local=ROOT.joinpath('outputs/site/training.html').read_bytes()
assert site==local
s=site.decode(); heads=list(re.finditer(r'<h2\b[^>]*>.*?</h2>',s,re.S))
ix=[i for i,h in enumerate(heads) if re.search(r'\bid=["\']T\.10["\']',h.group())]
assert len(ix)==1
ix=ix[0]; scope=s[heads[ix].start():heads[ix+1].start() if ix+1<len(heads) else s.index('</article>',heads[ix].end())]
OUT.joinpath('current/site-T.10-h2.html').write_text(scope)
sitefences=re.findall(r'<pre[^>]*>.*?<code[^>]*>(.*?)</code>\s*</pre>',scope,re.S)
decoded=[html.unescape(re.sub(r'<[^>]*>','',x)).encode() for x in sitefences]
assert decoded==fences
dump('site-response-inspection.json',{'URL':'http://127.0.0.1:8765/training.html','status':status,'content_type':mime,
    'actual_response_sha256':sha(site),'local_build_sha256':sha(local),'same_raw_bytes':site==local,
    'own_H2_id':'T.10','own_scope_sha256':sha(scope.encode()),'next_H2_id':'T.11',
    'codeblock_sha256':[sha(x) for x in decoded],'source_fences_equal':True,
    'initial_selector_failure_resolution':'Built H2 omits printed numeric prefix; select exact id T.10, not startswith text.',
    'scope':'Actual current HTTP bytes and exact own-H2 siblings only, not another output or parent export assertion.'})
inspection={'reviewer_task':'/root/phase4_factual_coordinator/factual_t_10','current_body_sha256':sha(body),
    'context_sha256':{n:sha(OUT.joinpath(f'current/context-{n}.md').read_bytes()) for n in ('T.4','T.6','12.12')},
    'script_sha256':sha(pathlib.Path(__file__).read_bytes()),'environment':env,'current_source_inspection':'actual-source-inspection.json',
    'bounded_contract':'bounded-new-recipe-contract.json','reuse_fingerprints':'reuse-fingerprints.json',
    'site_response':'site-response-inspection.json','exit_status':0}
dump('current-inspection.json',inspection)
print(json.dumps(inspection,ensure_ascii=False,indent=2))
