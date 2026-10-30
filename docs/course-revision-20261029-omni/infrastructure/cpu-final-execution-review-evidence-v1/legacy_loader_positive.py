"""Actual legacy download/extract/load integration using verified local own-model bytes, no network or forward."""
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import urllib.parse

import huggingface_hub
import torch
from scripts.omni_revision import cpu_final_execution as runner
from tiny_perceptron.selftrained.inference import verify_export
from tiny_perceptron.lesson_products.inference import check_manifest

ROOT=Path('/workspace/work/omni-models-v1')
BASE=Path('/workspace/work/omni-cpu-final-execution-review-v1')
CACHE=Path('/workspace/tiny-perceptron-vlm/outputs/omni-revision/baseline')
ARCHIVE=Path('/workspace/tiny-perceptron-vlm/assets/training/selftrained-v2.tar.gz')
A=CACHE/'moe-joint';B=CACHE/'b'
assert runner.file_sha(A/'inference-manifest.json')==runner.OLD_A['manifest_sha256']
assert runner.file_sha(B/'artifact-manifest.json')==runner.OLD_B['manifest_sha256']
a_manifest,_=verify_export(A,manifest_sha256=runner.OLD_A['manifest_sha256'])
b_manifest=check_manifest(B,'b')
manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_bytes())
assert ARCHIVE.stat().st_size==manifest['package']['bytes'] and runner.file_sha(ARCHIVE)==manifest['package']['sha256']
calls=[]

def hf_download(**kw):
    assert kw['repo_id']==runner.OLD_A['repo'] and kw['revision']==runner.OLD_A['revision']
    assert kw['repo_type']=='model' and kw['token'] is False
    prefix,name=kw['filename'].rsplit('/',1)
    assert prefix==runner.OLD_A['prefix'] and name in {'inference-manifest.json',*a_manifest['files']}
    calls.append({'download':'A','filename':kw['filename'],'token':False})
    return str(A/name)

def urlopen(request,timeout):
    if isinstance(request,str):
        expected=f"https://media.githubusercontent.com/media/{runner.REPOSITORY}/{manifest['package']['revision']}/{manifest['package']['path']}"
        assert request==expected and timeout==60
        calls.append({'download':'GitLFS','url':request,'auth_headers':False})
        return ARCHIVE.open('rb')
    assert not any(k.lower()=='authorization' for k,_ in request.header_items())
    prefix=f"https://huggingface.co/{runner.OLD_B['repo']}/resolve/{runner.OLD_B['revision']}/{runner.OLD_B['prefix']}/"
    assert request.full_url.startswith(prefix) and timeout==30
    name=urllib.parse.unquote(request.full_url[len(prefix):])
    assert name in {'artifact-manifest.json',*b_manifest['files']}
    calls.append({'download':'B','filename':name,'auth_headers':False})
    return io.BytesIO((B/name).read_bytes())

with tempfile.TemporaryDirectory(prefix='legacy-all-real-',dir=BASE/'probes') as tmp:
    root=Path(tmp);(root/'docs/selftrained').mkdir(parents=True)
    shutil.copyfile(ROOT/'docs/selftrained/v2-manifest.json',root/'docs/selftrained/v2-manifest.json')
    out=root/'output';out.mkdir()
    with patch.object(runner,'ROOT',root),patch.object(huggingface_hub,'hf_hub_download',hf_download),patch.object(runner.urllib.request,'urlopen',urlopen),patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('model forward forbidden')):
        runner.legacy_inputs(out)
    receipt=json.loads((out/'legacy-preflight.json').read_bytes())
    assert receipt['no_model_predictions'] is True and receipt['data_files']==8962 and receipt['data_bytes']==127161811
    assert receipt['old_A']['manifest_sha256']==runner.OLD_A['manifest_sha256'] and receipt['old_B_manifest_sha256']==runner.OLD_B['manifest_sha256'] and receipt['old_B_depth']==1
    assert receipt['anonymous_A_download']['authentication']=='disabled'
    assert sum(c['download']=='A' for c in calls)==4
    assert sum(c['download']=='B' for c in calls)==len(b_manifest['files'])+1
    result={'reviewer_task':'/root/omni_cpu_final_execution_review','status':'pass_for_actual_download_decode_load_integration_with_mocked_network','no_network':True,'no_model_forward':True,'no_training':True,'no_final_opening_created':True,'actual_legacy_inputs_function':True,'actual_unpack_asset':True,'actual_fetch_public_export':True,'actual_download_product':True,'actual_legacy_preflight_and_A_B_loaders':True,'A_B_payload_source':'Known locally cached selfmade weights verified against pinned manifests before load; no new teacher/model download','download_calls':calls,'receipt':receipt}
    target=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-review-evidence-v1/legacy-loader-positive-results.json'
    assert not target.exists();target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'A_download_calls':4,'B_download_calls':len(b_manifest['files'])+1,'data_files':receipt['data_files'],'data_bytes':receipt['data_bytes'],'model_forward_calls':0},ensure_ascii=False))
