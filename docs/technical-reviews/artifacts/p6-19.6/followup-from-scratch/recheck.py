"""Owner's genuine wording/provenance follow-up. No neural execution or training."""
from pathlib import Path
import ast,datetime,hashlib,json,platform,shutil,sys,difflib
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
BASE=OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=json.loads((OUT/'initial-pass-report.opaque.json').read_text())
v=json.loads((BASE/'verification.json').read_text())
chapter=(ROOT/'course/chapters/19.md').read_bytes()
start=chapter.index('## 19.6 '.encode());end=chapter.index('## 19.7 '.encode(),start)
current=chapter[start:end];old=(BASE/'inputs/19.6.md').read_bytes()
before='所有神經權重都由隨機初始化開始'.encode()
after='所有神經零件都由本專案從零訓練'.encode()
assert old.count(before)==1 and current==old.replace(before,after)
(OUT/'19.6.current.md').write_bytes(current)
(OUT/'section.diff.txt').write_text(''.join(difflib.unified_diff(old.decode().splitlines(keepends=True),current.decode().splitlines(keepends=True),fromfile='initial independently-reviewed section',tofile='current owner-rechecked section')))
result={'reviewer_task':'/root/p6_fact_19_6','utc':datetime.datetime.now(datetime.UTC).isoformat(),'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU; hashes/AST/provenance only; no neural execution'},'old_section_sha256':sha(BASE/'inputs/19.6.md'),'new_section_sha256':sha(OUT/'19.6.current.md'),'initial_own_report_sha256':sha(OUT/'initial-pass-report.opaque.json'),'exact_change':'Single replacement above, all other section bytes identical','reused_artifact_hashes':{},'reused_original_hashes':{},'raw_pointers':{},'dependency_sources':{},'inspected_ranges':{}}
# Confirm every piece of previous cited evidence still has its original bytes.
for item in initial['artifacts']:
    p=ROOT/item['path'];actual=sha(p);assert actual==item['sha256']
    result['reused_artifact_hashes'][item['path']]=actual
for item in v['frozen']:
    p=Path(item['source']);p=p if p.is_absolute() else ROOT/p
    assert sha(p)==item['sha256']
    result['reused_original_hashes'][str(p)]=sha(p)
manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_text())
for item in manifest['records']:
    if item['path'].startswith(('vision-','ocr-','voice-')):
        p=ROOT/'outputs/selftrained-v2/data'/item['path'];assert sha(p)==item['sha256']
        result['reused_original_hashes'][str(p)]=sha(p)
for item in v['assets']:
    p=Path(item['source']);assert sha(p)==item['sha256']
    result['reused_original_hashes'][str(p)]=sha(p)
archive=ROOT/manifest['package']['path'];assert sha(archive)==manifest['package']['sha256']
result['archive_sha256']=sha(archive)
model=Path('/tmp/p5-native-public-cpu-smoke-actual/public-model/model.safetensors')
assert sha(model)==v['model_files']['model.safetensors']
result['public_weights_sha256']=sha(model)
for path,digest in initial['figure_sha256'].items():
    assert sha(ROOT/path)==digest
    result['reused_original_hashes'][path]=digest

# Reinspect exact own-stage initial/load branches and training contracts.
paths=['tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/dataset.py','tiny_perceptron/selftrained/inference.py','scripts/selftrained/train.py','tiny_perceptron/model.py','tiny_perceptron/modern.py','tiny_perceptron/attention.py']
for f in paths:
    p=ROOT/f;tree=ast.parse(p.read_text());target=OUT/'sources'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    assert sha(p)==sha(target)
    result['dependency_sources'][f]={'sha256':sha(p),'copy':str(target.relative_to(ROOT)),'ast_definitions':[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]}
result['inspected_ranges']={
 'tiny_perceptron/selftrained/model.py':['imports9–17','MaskedBlock.__init__109–116','SelftrainedLanguageModel.__init__134–151','VisionEncoder188–214','OCREncoder217–244','AudioEncoder247–282','LimitedAssistant.__init__296–302'],
 'tiny_perceptron/model.py':['imports3–11','Block.__init__32–41','TinyLM.__init__54–66'],
 'tiny_perceptron/modern.py':['RMSNorm8–16','DenseFFN.__init__36–43','MoEFFN.__init__59–65'],
 'tiny_perceptron/attention.py':['CausalAttention.__init__32–46'],
 'scripts/selftrained/train.py':['PERCEPTION_BRIDGE_PREFIXES48–61','set_trainable284–296','objective519–554','own checkpoint checks/main723–789','training loop904–932'],
 'tiny_perceptron/selftrained/dataset.py':['perception_loss244–271','audio _modality118–131'],
 'tiny_perceptron/selftrained/inference.py':['InferenceAssistant safe-tensor load218–264']}
modern=ast.parse((ROOT/'tiny_perceptron/modern.py').read_text())
rms=next(n for n in modern.body if isinstance(n,ast.ClassDef) and n.name=='RMSNorm')
rms_init=next(n for n in rms.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
weight=next(n for n in rms_init.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=='weight')
assert ast.unparse(weight.value)=='nn.Parameter(torch.ones(width))'
result['deterministic_initialization_example']={'source':'tiny_perceptron/modern.py:11','expression':ast.unparse(weight.value),'interpretation':'Trainable RMSNorm scale starts at ones. From-scratch project training does not require each scalar to be randomly initialized.'}

# Execute only the existing trainability rule on inert names, not model parameters.
train_tree=ast.parse((ROOT/'scripts/selftrained/train.py').read_text())
nodes=[n for n in train_tree.body if isinstance(n,ast.FunctionDef) and n.name=='set_trainable' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PERCEPTION_BRIDGE_PREFIXES' for t in n.targets)]
scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'scripts/selftrained/train.py'),'exec'),scope)
class InertParameter:
    requires_grad=False
class InertModel:
    def __init__(self,names):self.params={name:InertParameter() for name in names}
    def named_parameters(self):return self.params.items()
    def parameters(self):return self.params.values()
coverage={name:[] for name in v['frozen_trainability']}
for stage in ['pretrain','sft','vision','ocr','audio','joint']:
    inert=InertModel(coverage)
    scope['set_trainable'](inert,stage,stage=='joint')
    for name,p in inert.named_parameters():
        if p.requires_grad:coverage[name].append(stage)
assert all(coverage.values())
result['trainability_contract_coverage']=coverage
result['coverage_scope']='Static module/prefix coverage plus executed existing selection rule on inert names. Not a claim that every scalar received a nonzero gradient, and no new neural run.'

# Read only original argv and provenance pointers, not selection scores or result notes.
stages={}
for architecture in ['dense','moe']:
    for stage in ['pretrain','sft','vision','ocr','audio','joint']:
        p=ROOT/f'docs/selftrained/results/training-raw/{architecture}-{stage}/raw/execution.json'
        d=json.loads(p.read_text());command=d['command'];assert d['returncode']==0
        if stage=='pretrain':assert '--init-checkpoint' not in command and '--resume' not in command
        else:assert '--init-checkpoint' in command
        stages[architecture+'-'+stage]={'command':command,'returncode':d['returncode'],'revision':d['revision'],'sha256':sha(p)}
        result['raw_pointers'][str(p)]=['/command','/returncode','/revision']
for name in ['dense-vision','dense-ocr','dense-audio','dense-joint','moe-vision','moe-ocr','moe-audio','moe-joint','moe-weighted','moe-native']:
    p=ROOT/f'docs/selftrained/results/training-raw/{name}/raw/train-receipt.json';d=json.loads(p.read_text())
    assert d['origin']['kind']=='all-neural-weights-random' and d['completed_requested_steps']
    if name.endswith('joint') or name in ['moe-weighted','moe-native']:assert d['freeze_perception_backbones']
    result['raw_pointers'][str(p)]=['/origin/kind','/stage_history','/freeze_perception_backbones','/completed_requested_steps']
    result.setdefault('stage_provenance',{})[name]={'origin_kind':d['origin']['kind'],'completed_requested_steps':d['completed_requested_steps'],'freeze_perception_backbones':d['freeze_perception_backbones'],'stage_history':d['stage_history'],'sha256':sha(p)}
result['actual_original_commands']=stages
result['result']='verified: all neural components originate in locally constructed modules and the project training chain, including perception-stage losses and shared-core/bridge joint updates; no external pretrained neural model/ASR was introduced. Constant normalization initialization is compatible with the new wording.'
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'result':result['result'],'old_section_sha256':result['old_section_sha256'],'new_section_sha256':result['new_section_sha256'],'unchanged_prior_artifacts':len(result['reused_artifact_hashes']),'unchanged_original_files':len(result['reused_original_hashes']),'inert_parameter_names_covered':len(coverage),'RMSNorm_initialization':result['deterministic_initialization_example']},ensure_ascii=False,indent=2))
