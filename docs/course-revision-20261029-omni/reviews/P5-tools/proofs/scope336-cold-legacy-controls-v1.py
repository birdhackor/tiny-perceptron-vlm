"""Own cold-process controls: true Hub and local legacy loader, no NN or network."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SNAPSHOT = Path('/workspace/work/omni-P5-full-tools-review-source-v4')
OLD_SNAPSHOT = Path('/workspace/work/omni-P5-full-tools-review-source-v3')
OUT = Path(__file__).with_name('scope336-cold-legacy-controls-result-v1.json')

# This starts in every fresh process before production modules. Hugging Face
# download functions and their offline request hook remain real. The last HTTP
# boundary supplies the exact old own-model/data bytes already outside Git.
HOOK = r'''
import atexit, hashlib, io, json, os, socket, sys, urllib.parse, urllib.request
from pathlib import Path
import httpx
import torch
assert not any(n == 'huggingface_hub' or n.startswith('huggingface_hub.') for n in sys.modules)
payloads = Path('/workspace/tiny-perceptron-vlm/outputs/omni-revision/baseline')
archive = Path('/workspace/tiny-perceptron-vlm/assets/training/selftrained-v2.tar.gz')
old_a = payloads / 'moe-joint'
old_b = payloads / 'b'
a_revision = '979cdfacc588ad0536f1c64fff96f264571cf054'
b_revision = '2c6a1062f3e66a602cf83eda6c703456ec18bca0'
a_base = 'https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/' + a_revision + '/selftrained/v2/moe-joint/'
b_base = 'https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/' + b_revision + '/lesson-products/v1-20261016/b/'
assert hashlib.sha256((old_a/'inference-manifest.json').read_bytes()).hexdigest() == 'f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
assert hashlib.sha256((old_b/'artifact-manifest.json').read_bytes()).hexdigest() == 'dd077fd7a0f11fa5d95bdcf76a73bf6e035819ab135fa3191ff94f4e3643a1a3'
trace = {'http': [], 'urllib': [], 'network_attempts': [], 'forward_attempts': []}
def no_socket(*a, **k):
    trace['network_attempts'].append('blocked')
    raise AssertionError('real sockets/DNS forbidden in review')
socket.socket.connect = no_socket
socket.getaddrinfo = no_socket
def no_forward(module, *a, **k):
    trace['forward_attempts'].append(type(module).__name__)
    raise AssertionError('NN forbidden in review')
torch.nn.Module._call_impl = no_forward
try:
    torch.nn.Identity()(torch.tensor(0.0))
except AssertionError:
    pass
assert trace['forward_attempts'] == ['Identity']
trace['forward_attempts'].clear()
def http(request):
    from huggingface_hub import constants
    assert constants.HF_HUB_OFFLINE is False and not constants.is_offline_mode()
    assert os.environ['HF_HUB_OFFLINE'] == '0'
    assert 'authorization' not in request.headers
    url = str(request.url)
    assert url.startswith(a_base), url
    name = urllib.parse.unquote(url[len(a_base):])
    assert Path(name).name == name
    raw = (old_a/name).read_bytes()
    trace['http'].append({'method': request.method, 'file': name, 'Hub_online': True, 'anonymous': True})
    assert request.method in ('HEAD', 'GET')
    return httpx.Response(200, headers={'Content-Length': str(len(raw)), 'ETag': '"'+hashlib.sha256(raw).hexdigest()+'"', 'X-Repo-Commit': a_revision}, content=raw if request.method=='GET' else b'')
original_client_init = httpx.Client.__init__
def client_init(self, *a, **k):
    k['transport'] = httpx.MockTransport(http)
    original_client_init(self, *a, **k)
httpx.Client.__init__ = client_init
def urlopen(request, timeout):
    if isinstance(request, str):
        assert request.startswith('https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/') and request.endswith('/assets/training/selftrained-v2.tar.gz')
        assert timeout == 60
        trace['urllib'].append({'kind':'data', 'anonymous':True})
        return archive.open('rb')
    assert request.full_url.startswith(b_base) and timeout == 30
    assert not any(k.lower() == 'authorization' for k, v in request.header_items())
    name = urllib.parse.unquote(request.full_url[len(b_base):])
    assert Path(name).name == name
    trace['urllib'].append({'kind':'B', 'file':name, 'anonymous':True})
    return io.BytesIO((old_b/name).read_bytes())
urllib.request.urlopen = urlopen
def save():
    from huggingface_hub import constants
    trace.update(argv0=sys.argv[0], environment_offline=os.environ.get('HF_HUB_OFFLINE'), Hub_offline=constants.HF_HUB_OFFLINE, formal_local_imported='scripts.omni_revision.formal_local' in sys.modules)
    Path(os.environ['P5_COLD_RECEIPTS'], str(os.getpid())+'.json').write_text(json.dumps(trace, indent=2)+'\n')
atexit.register(save)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(source, hook, cache, receipts):
    env = {k: v for k, v in os.environ.items() if k not in {'GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'}}
    env.update(PYTHONPATH=str(hook)+os.pathsep+str(source), HF_HOME=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_TOKEN='review-placeholder-must-not-be-sent', P5_COLD_RECEIPTS=str(receipts))
    return env


def run(code, cwd, env, timeout=60):
    result = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result


with tempfile.TemporaryDirectory(prefix='p5-cold-legacy-') as directory:
    private = Path(directory)
    hook = private/'hook'
    hook.mkdir()
    # Fail hard if the reviewer hook itself cannot initialize.
    guarded_hook = 'import os\ntry:\n' + '\n'.join('    '+line for line in HOOK.splitlines()) + '\nexcept BaseException:\n    os._exit(91)\n'
    compile(guarded_hook, 'p5-review-hook', 'exec')
    (hook/'sitecustomize.py').write_text(guarded_hook)
    receipts = private/'receipts'
    receipts.mkdir()
    # This reproduces the old cold failure before inspecting repair behavior.
    original_code = "from scripts.omni_revision import cpu_final_execution; from huggingface_hub import constants,hf_hub_download; from huggingface_hub.errors import LocalEntryNotFoundError; assert constants.HF_HUB_OFFLINE is True; import os; os.environ['HF_HUB_OFFLINE']='0'; assert constants.HF_HUB_OFFLINE is True;\ntry:\n hf_hub_download('birdhackor/tiny-perceptron-course-models','selftrained/v2/moe-joint/inference-manifest.json',revision='979cdfacc588ad0536f1c64fff96f264571cf054',token=False)\nexcept LocalEntryNotFoundError:\n print('OLD_COLD_OFFLINE_REJECTED')\nelse:\n raise AssertionError('old cold request unexpectedly succeeded')"
    old_cache = private/'old-empty-cache'
    old_cache.mkdir()
    old = run(original_code, OLD_SNAPSHOT, environment(OLD_SNAPSHOT, hook, old_cache, receipts))
    assert 'OLD_COLD_OFFLINE_REJECTED' in old.stdout
    old_reports = [json.loads(p.read_bytes()) for p in receipts.glob('*.json')]
    assert len(old_reports) == 1 and old_reports[0]['http'] == [] and old_reports[0]['Hub_offline'] is True
    for p in receipts.glob('*.json'):
        p.unlink()
    # Copy the immutable real99 source closure, avoiding author output paths.
    audit = json.loads((SNAPSHOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-audit-v2.json').read_bytes())
    source = private/'source'
    source.mkdir()
    assert len(audit['files']) == 99
    for name, digest in audit['files'].items():
        original = SNAPSHOT/name
        assert sha(original) == digest
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original,target)
    cache = private/'new-empty-cache'
    cache.mkdir()
    code = "from pathlib import Path; from scripts.omni_revision import cpu_final_execution as c; from huggingface_hub import constants; import os; assert os.environ['HF_HUB_OFFLINE']=='1' and constants.HF_HUB_OFFLINE; c.legacy_inputs(Path('controls')); assert os.environ['HF_HUB_OFFLINE']=='1' and constants.HF_HUB_OFFLINE"
    success = run(code, source, environment(source, hook, cache, receipts), 90)
    reports = [json.loads(p.read_bytes()) for p in receipts.glob('*.json')]
    assert len(reports) == 3
    download = next(v for v in reports if v['argv0'].endswith('legacy_download.py'))
    preflight = next(v for v in reports if v['argv0'].endswith('core_final.py'))
    parent = next(v for v in reports if v['argv0']=='-c')
    assert len(download['http']) == 8 and sum(v['method']=='HEAD' for v in download['http']) == 4
    assert len(download['urllib']) == 7 and sum(v['kind']=='B' for v in download['urllib']) == 6
    assert download['environment_offline']=='0' and download['Hub_offline'] is False and download['formal_local_imported'] is False
    for value in (parent, preflight):
        assert value['environment_offline']=='1' and value['Hub_offline'] is True and value['formal_local_imported'] is True
        assert value['http']==[] and value['urllib']==[]
    assert all(v['network_attempts']==[] and v['forward_attempts']==[] for v in reports)
    downloaded = json.loads((source/'controls/legacy-download.json').read_bytes())
    loaded = json.loads((source/'controls/legacy-preflight.json').read_bytes())
    assert downloaded['formal_local_not_imported'] is True and downloaded['anonymous_A_download']['authentication']=='disabled'
    assert loaded['data_files']==8962 and loaded['data_bytes']==127161811 and loaded['no_model_predictions'] is True and loaded['old_B_depth']==1
    # Negative cold clients must refuse before HTTP/model access; formal mixing
    # must refuse before changing the inherited offline environment.
    rejection_cases = [
        ('cached_offline', "import os; os.environ['HF_HUB_OFFLINE']='1'; from huggingface_hub import constants; assert constants.HF_HUB_OFFLINE; from scripts.omni_revision.legacy_download import configure_download_network;\ntry:\n configure_download_network()\nexcept ValueError as e:\n assert 'Hub已先快取離線設定' in str(e)\nelse:\n raise AssertionError('accepted cached offline')"),
        ('mixed_formal', "from scripts.omni_revision import formal_local; from scripts.omni_revision.legacy_download import configure_download_network; import os;\ntry:\n configure_download_network()\nexcept ValueError as e:\n assert '獨立新程序' in str(e) and os.environ['HF_HUB_OFFLINE']=='1'\nelse:\n raise AssertionError('accepted formal mixing')"),
    ]
    negatives=[]
    for name, code in rejection_cases:
        isolated = private/name
        isolated.mkdir()
        isolated_receipts = isolated/'receipts'
        isolated_receipts.mkdir()
        run(code, source, environment(source, hook, isolated/'empty-cache', isolated_receipts))
        report=json.loads(next(isolated_receipts.glob('*.json')).read_bytes())
        assert not report['http'] and not report['urllib'] and not report['network_attempts'] and not report['forward_attempts']
        negatives.append({'case':name,'rejected_before_HTTP_or_NN':True,'environment_offline_after':report['environment_offline'],'Hub_offline_after':report['Hub_offline']})

result = {'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),
          'old333_cold_failure_independently_reproduced':True,'old333_Hub_HTTP_calls':0,
          'real99_source_closure':True,'new_true_controller_two_bounded_children_pass':True,
          'download_Hub_HEAD':4,'download_Hub_GET':4,'anonymous_B_requests':6,'data_archive_requests':1,
          'true_hf_hub_download_and_offline_event_hook_unmodified':True,
          'empty_HF_cache':True,'initial_offline_environment':'1','parent_offline_preserved':True,'fresh_loader_offline':True,
          'data_files_actually_hashed_and_loaded':8962,'data_bytes':127161811,'A_B_actual_loader_preflight':loaded,
          'old_A_B_continuation_deserialization_not_performed':True,
          'forward_guard_selftested_in_every_child':True,'actual_forward_attempts':[],'actual_network_socket_or_DNS_attempts':[],
          'rejection_cases':negatives,'reports':reports,
          'limits':['HTTP boundary uses exact existing own-model/data bytes. No new provider network success is claimed; actual Hub download/request policy, extraction, hashes, constructors and loaders ran.','Old333 original/current remain unchanged; previous mocks did not cover this cold import policy. This new proof records the reproduced gap and R1 repair.','No final measurement/training/Modal/reservation/CAS/upload/publication or protected-source edit. Private data/weights/output trees are removed after verification.']}
with OUT.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','old_cold_failure_reproduced':True,'source_count':99,'data_files':8962,'true_Hub_HEAD':4,'true_Hub_GET':4,'forward_attempts':0,'network_socket_attempts':0,'rejection_cases':len(negatives)},ensure_ascii=False))
