"""Fresh bounded T.11 verification. No training, upload, or heldout evaluation."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import platform
import tarfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

copies = []
reads = []
def original(relative, pointers):
    p = ROOT / relative
    q = OUT / 'originals' / relative
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_bytes(p.read_bytes())
    assert sha(p) == sha(q)
    copies.append({'original': relative, 'copy': str(q.relative_to(ROOT)), 'sha256': sha(p)})
    reads.append({'path': relative, 'pointers': pointers})
    return json.loads(p.read_bytes())

result = {'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'device': 'cpu', 'cuda_available': torch.cuda.is_available()},
          'scope': 'bounded contract checks and archival measurements; no complete training or new heldout evaluation'}
result['numeric'] = {'old_bytes': 80000, 'new_bytes': 60000, 'old_correct': 4, 'new_correct': 3,
                     'denominator': 5, 'storage_smaller': 60000 < 80000,
                     'count_preserved': 3 >= 4, 'old_accuracy': 4/5, 'new_accuracy': 3/5}
assert result['numeric']['storage_smaller'] and not result['numeric']['count_preserved']
a = [True, True, True, True, False]
b = [False, True, True, True, True]
result['paired_counterexample'] = {'same_score': sum(a)==sum(b)==4,
                                   'different_error_ids': a != b,
                                   'full_score': sum(b)/len(b), 'drop_hard_score': sum(b[1:])/len(b[1:])}

manifest = original('docs/selftrained/v2-manifest.json', ['/initialization','/package','/model_config','/records','/assets'])
assert sha(ROOT/'docs/selftrained/v2-manifest.json') == '3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b'
package = manifest['package']
pointer = subprocess.check_output(['git','show',f"{package['revision']}:{package['path']}"],cwd=ROOT)
expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{package['sha256']}\nsize {package['bytes']}\n".encode()
assert pointer == expected
(OUT/'lfs-pointer.txt').write_bytes(pointer)
archive = ROOT/package['path']
assert archive.stat().st_size == package['bytes'] and sha(archive)==package['sha256']
data = ROOT/'outputs/selftrained-v2/data'
cache_mismatches=[]
with tarfile.open(archive,'r:gz') as tar:
    members={m.name:m for m in tar.getmembers() if m.isfile()}
    for item in manifest['records']+manifest['assets']:
        p=data/item['path']
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p)!=item['sha256']:
            cache_mismatches.append(item['path'])
        member=members[item['path']]
        assert member.size==item['bytes'] and hashlib.sha256(tar.extractfile(member).read()).hexdigest()==item['sha256'],item['path']
assert len(manifest['records'])==12 and len(manifest['assets'])==8950
result['data'] = {'manifest_sha256':sha(ROOT/'docs/selftrained/v2-manifest.json'), 'initialization':manifest['initialization'],
                  'package':package,'verified_records':len(manifest['records']),'verified_assets':len(manifest['assets']),
                  'archive_verified':True,'git_pointer_verified':True,'existing_data_root':str(data),
                  'cache_mismatches':cache_mismatches,'verified_material':'original pinned archive members; existing cache used only for no-asset text/tool examples'}

training = []
for stage in ['moe-pretrain','moe-sft','moe-vision','moe-ocr','moe-audio','moe-joint','moe-weighted','moe-native',
              'dense-pretrain','dense-sft','dense-vision','dense-ocr','dense-audio','dense-joint','dense-weighted']:
    base = f'docs/selftrained/results/training-raw/{stage}/'
    ex = original(base+'raw/execution.json', ['/status','/returncode','/command','/revision','/manifest_sha256','/job'])
    tr = original(base+'raw/train-receipt.json', ['/steps','/train_records','/validation_records','/test_used_for_selection',
           '/completed_requested_steps','/interrupted','/inference_exported','/origin','/stage_history','/selection'])
    outer = original(base+'receipt.json', ['/files','/status','/returncode'])
    assert ex['status']=='completed' and ex['returncode']==0
    assert tr['completed_requested_steps'] and not tr['interrupted'] and tr['inference_exported']
    assert not tr['test_used_for_selection'] and tr['origin']['kind']=='all-neural-weights-random'
    options = {k:ex['job'].get(k) for k in ['architecture','stage','steps','init_checkpoint','resume','device']}
    training.append({'stage':stage,'options':options,'steps':tr['steps'],'selection':tr['selection'],
       'train_records':tr['train_records'],'validation_records':tr['validation_records'],
       'origin_kind':tr['origin']['kind'],'history':[{k:h.get(k) for k in ['stage','step','checkpoint','checkpoint_sha256','selection']} for h in tr['stage_history']],
       'safe_exports':[{k:f[k] for k in ['path','bytes','sha256']} for f in outer['files'] if f['path'].endswith(('model.safetensors','inference-manifest.json'))]})
result['training'] = training

cpu = []
for p in sorted((ROOT/'docs/selftrained/results/public-cpu-raw').glob('*-result.json')):
    rel = str(p.relative_to(ROOT))
    obj = original(rel, ['/task','/chat','/argv','/returncode','/stdout_sha256','/actual_answer',
                         '/model','/public_source','/actual_tool_trace','/generation_count','/actual_generations'])
    assert obj['returncode']==0
    stem = p.name.removesuffix('-result.json')
    for suffix in ['-argv.json','-stdout.txt','-stderr.txt']:
        raw = p.parent/(stem+suffix)
        if raw.exists():
            q=OUT/'originals'/str(raw.relative_to(ROOT));q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw.read_bytes())
            copies.append({'original':str(raw.relative_to(ROOT)),'copy':str(q.relative_to(ROOT)),'sha256':sha(raw)})
            if suffix=='-stdout.txt':assert sha(raw)==obj['stdout_sha256']
    if obj['chat']:
        argv=obj['argv'];assert argv[argv.index('--device')+1]=='cpu'
        assert obj['public_source']['revision']=='979cdfacc588ad0536f1c64fff96f264571cf054'
        stdout=json.loads((p.parent/(stem+'-stdout.txt')).read_text())
        assert stdout['answer']==obj['actual_answer']
        cpu.append({'id':stem,'chat':True,'answer':stdout['answer'],'generation_count':len(stdout['generations']),
                    'generation_status':[x['generation_status'] for x in stdout['generations']],
                    'public_source':obj['public_source'],'manifest_sha256':obj['model']['manifest_sha256'],
                    'selected_step':obj['model']['selected_step'], 'tool_trace':stdout['tool_trace']})
    else:cpu.append({'id':stem,'chat':False,'argv':obj['argv'],'returncode':obj['returncode']})
assert sum(x['chat'] for x in cpu)==8 and len(cpu)==9
tool=next(x for x in cpu if x['id']=='tool_call-1')
assert tool['tool_trace']['executed'] and tool['generation_count']==2
result['archived_public_cpu']=cpu

final=[]
for arch in ['moe','dense']:
    base=f'docs/selftrained/results/public-raw/{arch}/test/'
    m=original(base+'metrics.json',['/split','/count','/expected_count','/evaluation_complete','/interrupted',
                  '/teacher_forcing_used_for_generation','/weight_source','/safe_weights_sha256','/inference_manifest_sha256'])
    r=original(base+'evaluation-receipt.json',['/status','/expected_count','/completed_count','/outputs_sha256','/protocol_sha256'])
    assert m['split']=='test' and m['evaluation_complete'] and not m['interrupted']
    assert m['count']==m['expected_count']==r['completed_count']==r['expected_count']==3734 and r['status']=='complete'
    final.append({k:m[k] for k in ['split','count','expected_count','evaluation_complete','teacher_forcing_used_for_generation','weight_source','inference_manifest_sha256']})
result['archived_final_completion']=final

def load_module(name, relative):
    s=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
trainer=load_module('t11_trainer','scripts/selftrained/train.py')
wrapper=load_module('t11_wrapper','scripts/selftrained/train_local_stage.py')
commands=[]
doc=(ROOT/'docs/selftrained/TRAINING.md').read_text()
for fence in re.findall(r'```bash\n(.*?)\n```',doc,re.S):
    if 'python scripts/selftrained/train_local_stage.py' not in fence:continue
    argv=shlex.split(fence.replace('\\\n',' '));separator=argv.index('--');args=trainer.parser().parse_args(argv[separator+1:]+['--records','contract-only.jsonl','--asset-dir',str(data)])
    commands.append({k:getattr(args,k) for k in ['stage','architecture','steps','init_checkpoint','resume','device']})
assert len(commands)==4
result['training_recipe_parser_contract']=commands

rejections=[]
for name,resume in [('model.safetensors',False),('best.pt',True),('latest.pt',False)]:
    try:wrapper.local_source(Path('/tmp')/name,'0'*64,resume)
    except ValueError as e:rejections.append({'input':name,'resume':resume,'error':str(e)})
    else:raise AssertionError('unexpected accepted wrong checkpoint kind')
result['wrong_checkpoint_kind_rejections']=rejections
result['source_identity']=wrapper.source_identity(ROOT)[0]

for relative in ['scripts/selftrained/train_local_stage.py','scripts/selftrained/train.py','scripts/selftrained/chat.py',
                 'scripts/selftrained/hf_transport.py','tiny_perceptron/selftrained/inference.py','tiny_perceptron/selftrained/tools.py',
                 'tiny_perceptron/selftrained/model.py','tiny_perceptron/natural_assistant.py','pyproject.toml']:
    p=ROOT/relative;q=OUT/'code'/relative;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());assert sha(p)==sha(q)
    copies.append({'original':relative,'copy':str(q.relative_to(ROOT)),'sha256':sha(p)})
result['actual_read_pointers']=reads
result['preserved_original_sha256']=copies
(OUT/'audit-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'numeric':result['numeric'],'data_files':8962,'completed_training_stages':len(training),
                  'archived_cpu_calls':8,'history_appends':1,'archived_final_counts':[x['count'] for x in final],
                  'training_recipes_parsed':len(commands),'wrong_checkpoint_rejections':len(rejections),
                  'environment':result['environment'],'source_git_head':result['source_identity']},ensure_ascii=False,indent=2))
