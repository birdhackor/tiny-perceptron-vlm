"""Independent delivery recheck: private fixtures, no live downloads/NN calls."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import contextlib
import copy
import datetime
import hashlib
import io
import json
import pickle
import sys
import urllib.error
import wave
import warnings
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path('/workspace/work/omni-models-v1')
E=Path('/workspace/work/omni-public-delivery-review-current-v1')
sys.path.insert(0,str(ROOT))
from scripts.omni_revision import public_download as loader,public_demo as demo
from tiny_perceptron.omni_products import inference as inf,training_runtime as runtime
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,FileExistsError):return True
    raise AssertionError('invalid input accepted')
def expected(files):
    # Expected identities come from declared fixture source bytes, independently
    # of the ZIP's possibly-corrupt own manifest. New API is explicit here.
    return {name:{'sha256':digest(raw),**({'bytes':len(raw)} if name.endswith('weights.safetensors') else {})} for name,raw in files.items() if name.endswith(('/training/weights.safetensors','/training/training-result.json','/job-result.json'))}
def package(folder,job='codec',extra=None,bad=None,duplicate=False):
    folder.mkdir(parents=True,exist_ok=True)
    files={f'core/{job}/training/weights.safetensors':b'opaque inference fixture',f'core/{job}/training/training-result.json':b'{"fixture":true}',f'core/{job}/job-result.json':b'{"fixture":true}',f'core/{job}/training/latest-continuation.pt':b'opaque optimizer/RNG fixture; no unpickle'}
    files.update(extra or {});manifest={n:{'bytes':len(v),'sha256':digest(v)} for n,v in files.items()}
    if bad:manifest[bad]['sha256']='0'*64
    path=folder/(job+'.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
        if duplicate:
            with warnings.catch_warnings():warnings.simplefilter('ignore');z.writestr(next(iter(files)),next(iter(files.values())))
        z.writestr('package-file-manifest.json',json.dumps(manifest))
    return path,{'job':job,'name':job+'.zip','bytes':path.stat().st_size,'sha256':sha(path)},files

fixtures=E/'fixtures';results={}
archive,asset,files=package(fixtures/'base',extra={'core/codec/training/version_1.2-log.json':b'ordinary valid artifact'})
destination=fixtures/'valid output 工作 空間'
value=loader.install(archive,asset,destination,expected_files=expected(files));assert loader.install(archive,asset,destination,expected_files=expected(files))==value
assert not (destination/'core/codec/training/latest-continuation.pt').exists()
with patch.object(torch,'load',side_effect=AssertionError('torch.load forbidden')),patch.object(pickle,'load',side_effect=AssertionError('pickle.load forbidden')),patch.object(pickle,'loads',side_effect=AssertionError('pickle.loads forbidden')):
    loader.install(archive,asset,fixtures/'full output',True,expected_files=expected(files))
assert (fixtures/'full output/core/codec/training/latest-continuation.pt').read_bytes()==files['core/codec/training/latest-continuation.pt']
results['opaque_continuation_and_valid_cache']={'default_excludes_pt':True,'full_state_copies_exact_bytes_without_deserialization':True,'valid_spaces_unicode_destination':True,'exact_reinstall_reuses_cache':True}

bad_names=['core/codec/C:/escaped.txt','core/codec/C:escaped.txt','core/codec/name:stream','core/codec/CON','core/codec/NUL.txt','core/codec/LPT9.log','core/codec/../x','core/codec/x/./y','core/codec/x//y','core/codec/tail.','core/codec/folder\\x','/core/codec/absolute','core/talker/other-job']
for i,name in enumerate(bad_names):
    path,row,source=package(fixtures/f'bad-path-{i}',extra={name:b'bad path'})
    target=fixtures/f'rejected-path-{i}';assert reject(lambda:loader.install(path,row,target,expected_files=expected(source))) and not target.exists()
assert all(not loader.portable_member(name) for name in bad_names[:-1])
windows_root=PureWindowsPath('D:/downloads/工作 空間')
for name in files:
    assert loader.portable_member(name) and windows_root.joinpath(*PurePosixPath(name).parts).is_relative_to(windows_root)
path,row,source=package(fixtures/'case-collision',extra={'core/codec/training/WEIGHTS.safetensors':b'case collision'})
assert reject(lambda:loader.install(path,row,fixtures/'bad-case',expected_files=expected(source)))
path,row,source=package(fixtures/'duplicate',duplicate=True);assert reject(lambda:loader.install(path,row,fixtures/'bad-duplicate',expected_files=expected(source)))
assert reject(lambda:loader.install(archive,{**asset,'sha256':'0'*64},fixtures/'bad-outer',expected_files=expected(files)))
results['PD01']={'original_Windows_drive_escape_rejected':True,'unsafe_member_cases':len(bad_names),'casefold_and_duplicate_rejected':True,'valid_members_remain_under_PureWindows_destination':True,'valid_ASCII_versioned_artifact_and_unicode_output_not_blocked':True}

for suffix in ('weights.safetensors','training-result.json','job-result.json'):
    identity=copy.deepcopy(expected(files));key=next(n for n in identity if n.endswith('/'+suffix));identity[key]['sha256']='f'*64
    target=fixtures/('wrong-'+suffix);assert reject(lambda:loader.install(archive,asset,target,expected_files=identity)) and not target.exists()
for label,bad in [('late-inner','core/codec/training/last.json'),('first-inner','core/codec/training/weights.safetensors')]:
    path,row,source=package(fixtures/label,extra={'core/codec/training/last.json':b'late bytes'},bad=bad)
    target=fixtures/('no-write-'+label);assert reject(lambda:loader.install(path,row,target,expected_files=expected(source))) and not target.exists()
path,row,source=package(fixtures/'late-conflict',extra={'core/codec/training/last.json':b'new bytes'})
target=fixtures/'existing';last=target/'core/codec/training/last.json';last.parent.mkdir(parents=True,exist_ok=True);last.write_bytes(b'keep user bytes')
assert reject(lambda:loader.install(path,row,target,expected_files=expected(source)))
assert last.read_bytes()==b'keep user bytes' and not (target/'core/codec/training/weights.safetensors').exists()
results['PO01']={'late_inner_SHA_and_existing_conflict_detected_before_first_new_weight_write':True,'required_3_file_pins_each_checked':True,'existing_regular_data_preserved':True}

# Construct one complete five-kind selection/descriptor/archive fixture. These
# are synthetic future metadata and bytes, not real publication artifacts.
repo=fixtures/'repo';selection={'schema':'omni-core-model-selection-v1','final_opened':False,'selected':{'av':'av-time-rope-s11','choice':'c-augmented','choice_control':'c-compact','codec':'codec','talker':'talker'},'temperature':1.,'threshold':None,'policy_feasible':False,'model_files':{},'all_formal_jobs':{}}
release={'schema':'omni-public-model-download-v1','repository':loader.REPOSITORY,'release_tag':'explicit-fixture-v1','model_selection':{'path':'docs/omni-products/v1/model-selection.json'},'components':{},'_model_selection':{'untrusted':'must be replaced'}}
archives={};members={}
for kind,job in selection['selected'].items():
    weight=b'selected opaque fixture bytes: '+job.encode();config={'codes':128,'width':96,'hop':80} if kind=='talker' else {}
    training={'final_opened':False,'optimizer_updates_observed':True,'training_revision':runtime.TRAINING_REVISION,'implementation_sha256':runtime.source_binding(),'source_unchanged_during_training':True,'config':{'model':config},'files':{'weights.safetensors':{'bytes':len(weight),'sha256':digest(weight)}}}
    training_raw=json.dumps(training).encode();job_raw=json.dumps({'explicit_fixture':True,'job':job}).encode()
    path,row,source=package(fixtures/('selected-'+kind),job,extra={f'core/{job}/training/weights.safetensors':weight,f'core/{job}/training/training-result.json':training_raw,f'core/{job}/job-result.json':job_raw})
    archives[row['name']]=path;members[kind]=source;release['components'][kind]=row
    selection['model_files'][kind]={'job':job,'weights':{'bytes':len(weight),'sha256':digest(weight)},'training_result_sha256':digest(training_raw)}
    selection['all_formal_jobs'][job]={'job_result_sha256':digest(job_raw)}
selection_path=repo/'docs/omni-products/v1/model-selection.json';write(selection_path,selection);release['model_selection']['sha256']=sha(selection_path)
release_path=repo/'docs/omni-products/v1/public-release.json';write(release_path,release)
fetches=[]
def fixture_fetch(value,row,target):fetches.append(row['name']);Path(target).write_bytes(archives[row['name']].read_bytes())
with patch.object(loader,'ROOT',repo),patch.object(demo,'ROOT',repo),patch.object(loader,'fetch',fixture_fetch):
    checked=loader.descriptor(release_path);assert checked['_model_selection']==selection and checked['_model_selection']!=release['_model_selection']
    outputs={}
    for label,control,full in [('default',False,False),('all5',True,False),('full5',True,True)]:
        target=fixtures/('download-'+label);receipt=loader.download(release_path,target,control,full);outputs[label]=target
        assert len(receipt['assets'])==(5 if control else 4)
        assert loader.download(release_path,target,control,full)==receipt
        for kind in receipt['assets']:
            loader.verify_installed(target,selection,kind)
            continuation=target/f"core/{selection['selected'][kind]}/training/latest-continuation.pt"
            assert continuation.exists() is full
    # All4 must verify before any model loader. Real load_trained metadata checks
    # run on valid fixtures; only model construction / safetensors parsing mock.
    parsed=[]
    class FakeModel:
        training=False
        def __init__(self,config=None,codes=128,width=96,hop=80):self.config=config;self.codes=codes;self.width=width;self.hop=hop
        def parameters(self):yield torch.zeros(1)
        def load_state_dict(self,value,strict):assert strict is True
        def to(self,device):return self
        def eval(self):return self
        def requires_grad_(self,value):assert value is False;return self
    def parser(path,device):parsed.append(path);return {'fixture':torch.ones(1)}
    with patch.object(inf,'AVModel',lambda config:FakeModel(config)),patch.object(inf,'ChoiceModel',lambda config:FakeModel(config)),patch.object(inf,'SoundCodec',lambda config:FakeModel(config)),patch.object(inf,'ProgramTalker',lambda codes,width,hop:FakeModel(codes=codes,width=width,hop=hop)),patch.object(inf,'load_file',parser):
        product,published=demo.assistant(outputs['default'],release_path)
    assert isinstance(product,inf.OmniAssistant) and len(parsed)==4
    changed_paths=[]
    for relative in ('training/weights.safetensors','training/training-result.json','job-result.json'):
        job=selection['selected']['talker'];path=outputs['default']/f'core/{job}'/relative;old=path.read_bytes();path.write_bytes(b'other internally consistent checkpoint or receipt')
        loaded=[]
        with patch.object(demo,'load_trained',lambda *args:loaded.append(args)):
            assert reject(lambda:demo.assistant(outputs['default'],release_path))
        assert not loaded;path.write_bytes(old);changed_paths.append(relative)
    # A self-consistent archive that is not selected must fail before installing.
    job=selection['selected']['codec'];path,row,source=package(fixtures/'alternate-codec',job,extra={f'core/{job}/training/weights.safetensors':b'alternate valid-looking checkpoint'})
    assert reject(lambda:loader.install(path,row,fixtures/'alternate-not-installed',expected_files=loader.selected_files(selection,'codec')))
    assert not (fixtures/'alternate-not-installed').exists()
    # Receipt links fail before fetch, even though descriptor is complete/valid.
    target=fixtures/'receipt-link';target.mkdir();outside=fixtures/'outside-receipt.json';(target/'download-receipt.json').symlink_to(outside)
    before=len(fetches);assert reject(lambda:loader.download(release_path,target)) and len(fetches)==before and not outside.exists()
results['PD02']={'valid5kind_descriptor_checked':True,'untrusted_embedded_model_selection_replaced':True,'default4_control5_fullstate5_and_cache_reuse_succeed':True,'actual_load_trained_metadata_guard_reaches4mock_weight_parsers_for_correct_fixtures':True,'demo_4model_verification_precedes_all_loads':True,'3_late_talker_identity_drifts_rejected_before_any_model_loader':changed_paths,'original_unselected_self_consistent_archive_rejected':True,'real_weights_or_model_predictions':False}

wav=fixtures/'valid.wav';saved=demo.save_wave(wav,torch.zeros(80))
with wave.open(str(wav),'rb') as reader:assert (reader.getframerate(),reader.getnframes(),reader.getsampwidth(),reader.getnchannels())==(8000,80,2,1)
original=wav.read_bytes();assert reject(lambda:demo.save_wave(wav,torch.zeros(80))) and wav.read_bytes()==original
link=fixtures/'dangling.wav';outside_wav=fixtures/'outside.wav';link.symlink_to(outside_wav);assert reject(lambda:demo.save_wave(link,torch.zeros(80))) and not outside_wav.exists()
real_exists=Path.exists
with patch.object(Path,'exists',lambda path:False if path==wav else real_exists(path)):
    assert reject(lambda:demo.save_wave(wav,torch.zeros(80)))
assert wav.read_bytes()==original
assert reject(lambda:demo.save_wave(fixtures/'nonfinite.wav',torch.tensor([float('nan')])))
results['PD03']={'receipt_dangling_link_rejected_before_network':True,'WAV_dangling_link_rejected':True,'regular_WAV_preserved':True,'exclusive_create_rejects_simulated_post_check_collision':True,'valid_PCM16_8k_mono_write_succeeds':True,'nonfinite_rejected':True}

# Replay unchanged bare-URL fetch and malformed-network controls with no network.
class Response(io.BytesIO):
    def __init__(self,raw,url='https://release-assets.githubusercontent.com/fixture',length=None):super().__init__(raw);self.url=url;self.headers={} if length is None else {'Content-Length':str(length)}
raw=b'anonymous fixture';row={'name':'codec.zip','bytes':len(raw),'sha256':digest(raw)};requests=[]
def opener(body,url='https://fixture.invalid',length=None):
    def call(request,timeout):requests.append(request);return Response(body,url,length)
    return call
with patch.object(loader.urllib.request,'urlopen',opener(raw,length=len(raw))):loader.fetch({'release_tag':'fixture-v1'},row,fixtures/'fetch-valid')
assert isinstance(requests[0],str) and requests[0].startswith('https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/')
for label,body,url,length in [('scheme',raw,'http://fixture.invalid',None),('header',raw,'https://fixture.invalid',999),('overflow',raw+b'x','https://fixture.invalid',None),('short',raw[:-1],'https://fixture.invalid',None)]:
    with patch.object(loader.urllib.request,'urlopen',opener(body,url,length)):assert reject(lambda:loader.fetch({'release_tag':'fixture-v1'},row,fixtures/('fetch-'+label)))
results['anonymous_HTTP']={'bare_fixed_HTTPS_URL_no_supplied_auth':True,'scheme_header_stream_size_SHA_negative_controls':4,'real_network_requests':0}

# Demo still shows actual returned wrong answers/refusals. Timing is now named
# internal readiness and requested/condition programs are not claimed as output.
def judge(question,information):
    assert question.gold is None and information.gold is None
    return {'prediction':{'answer':'stop','ids':['record','stop','none'],'logits':[0.,2.,0.],'probabilities':[.1,.8,.1],'p_max':.8},'accepted':False,'decision':'handoff'}
stub=SimpleNamespace(c=SimpleNamespace(judge=judge),video=lambda item,with_sound:{'answer':'unknown','logits':[0.,0.,2.],'source':'explicit refusal fixture'},sound_chunks=lambda program:iter([{'samples':torch.zeros(80),'ready_seconds':.3,'whole_code_generation_seconds':.2},{'samples':torch.zeros(80),'ready_seconds':.4,'whole_code_generation_seconds':.2}]))
reports={}
for mode in ('choice','av','sound'):
    capture=io.StringIO()
    with patch.object(demo,'assistant',lambda *args:(stub,{'release_tag':'presentation fixture'})),patch.object(sys,'argv',['public_demo',mode,'--model-root',str(outputs['default'])]),contextlib.redirect_stdout(capture):demo.main()
    reports[mode]=json.loads(capture.getvalue())
assert reports['choice']['prediction']['answer']=='stop' and reports['choice']['decision']=='handoff'
assert reports['av']['answer']=='unknown' and 'gold' not in reports['av']
assert reports['sound']['internal_first_chunk_ready_seconds']==.3 and reports['sound']['whole_code_generation_seconds']==.2 and 'requested_program' in reports['sound'] and 'excludes CLI WAV writing and playback' in reports['sound']['timing_scope']
stub.sound_chunks=lambda program:(_ for _ in ()).throw(ValueError('fixture generation failure'))
with patch.object(demo,'assistant',lambda *args:(stub,{'release_tag':'fixture'})),patch.object(sys,'argv',['public_demo','sound','--model-root',str(outputs['default'])]):assert reject(demo.main)
assert reject(lambda:demo.public_question({**demo.choice_example(),'gold':'record'}))
results['PO02_and_demo_truth']={'wrong_answer_and_unknown_refusal_preserved':True,'gold_field_not_sent':True,'requested_program_distinguished_from_output':True,'internal_latency_excludes_CLI_and_playback_explicit':True,'generation_failure_not_replaced_by_reference_or_silence':True}

out={'reviewer_task':'/root/omni_final_definition_review','completed_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),'method':'Independent new expected_files API fixtures; synthetic5kind selected artifacts, no-NN mocked parser/constructors, actual wrapper composer, private ZIP/symlink/HTTP/presentation controls','recheck':results,'actual_model_selection_exists':(ROOT/'docs/omni-products/v1/model-selection.json').exists(),'actual_public_release_exists':(ROOT/'docs/omni-products/v1/public-release.json').exists(),'actual_live_download_or_real_model_integration_verified':False,'no_network_paid_teacher_or_final_predictions':True}
p=E/'probe-evidence-current-v1.json'
with p.open('x') as stream:stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'rechecked':list(results),'real_public_artifacts_present':[out['actual_model_selection_exists'],out['actual_public_release_exists']],'no_live_or_NN':True},ensure_ascii=False,indent=2))
