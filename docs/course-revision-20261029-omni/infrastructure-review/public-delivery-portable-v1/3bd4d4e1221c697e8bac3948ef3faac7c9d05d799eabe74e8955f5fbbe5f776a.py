"""Independent offline delivery review. No network or real model predictions."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import contextlib
import copy
import datetime
import hashlib
import io
import json
import sys
import urllib.error
import wave
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path('/workspace/work/omni-models-v1')
E=Path('/workspace/work/omni-public-delivery-review-original-v1')
sys.path.insert(0,str(ROOT))
from scripts.omni_revision import public_download as loader,public_demo as demo
from tiny_perceptron.omni_products import inference as inf,training_runtime as runtime

def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def reject(fn,expected=(ValueError,RuntimeError,FileExistsError)):
    try:fn()
    except expected:return True
    raise AssertionError('expected rejection')
def package(folder,job='codec',extra=None,bad_member=None,duplicate=False):
    folder.mkdir(parents=True,exist_ok=True)
    files={f'core/{job}/training/weights.safetensors':b'opaque inference weight fixture',f'core/{job}/training/training-result.json':b'{"fixture":true}',f'core/{job}/job-result.json':b'{"fixture":true}',f'core/{job}/training/latest-continuation.pt':b'opaque optimizer/RNG bytes; never deserialized'}
    files.update(extra or {})
    manifest={name:{'bytes':len(raw),'sha256':sha_bytes(raw)} for name,raw in files.items()}
    if bad_member:manifest[bad_member]['sha256']='0'*64
    path=folder/(job+'.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in files.items():archive.writestr(name,raw)
        if duplicate:
            with __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore');archive.writestr(next(iter(files)),next(iter(files.values())))
        archive.writestr('package-file-manifest.json',json.dumps(manifest))
    return path,{'job':job,'name':job+'.zip','bytes':path.stat().st_size,'sha256':sha(path)},files

positive={};findings={};fixtures=E/'fixtures'
archive,asset,files=package(fixtures/'normal')
target=fixtures/'install'
first=loader.install(archive,asset,target);assert loader.install(archive,asset,target)==first
assert not (target/'core/codec/training/latest-continuation.pt').exists()
with patch('pickle.load',side_effect=AssertionError('must not load')),patch('pickle.loads',side_effect=AssertionError('must not loads')),patch.object(torch,'load',side_effect=AssertionError('must not torch.load')):
    full=loader.install(archive,asset,fixtures/'full',True)
assert (fixtures/'full/core/codec/training/latest-continuation.pt').read_bytes()==files['core/codec/training/latest-continuation.pt']
positive['continuation_bytes_only']={'default_excluded':True,'full_requested_copied_exactly':True,'pickle_and_torch_load_not_called':True,'exact_cache_reuse':True}

rejections=[]
bad={**asset,'sha256':'0'*64};assert reject(lambda:loader.install(archive,bad,fixtures/'wrong-outer'))
rejections.append('outer SHA')
for label,extra,kwargs in [
 ('parent traversal',{'core/codec/../../escape':b'x'},{}),
 ('absolute',{'/core/codec/escape':b'x'},{}),
 ('backslash',{'core/codec/training\\escape':b'x'},{}),
 ('other job',{'core/talker/training/escape':b'x'},{}),
 ('duplicate',None,{'duplicate':True}),
 ('inner SHA',None,{'bad_member':'core/codec/training/weights.safetensors'}),
 ('too many members',{f'core/codec/extra-{i}':b'x' for i in range(61)},{}),
]:
    path,row,_=package(fixtures/label.replace(' ','-'),extra=extra,**kwargs)
    assert reject(lambda:loader.install(path,row,fixtures/('reject-'+label.replace(' ','-'))));rejections.append(label)
existing=fixtures/'existing';weight=existing/'core/codec/training/weights.safetensors';weight.parent.mkdir(parents=True,exist_ok=True);weight.write_bytes(b'user data')
assert reject(lambda:loader.install(archive,asset,existing)) and weight.read_bytes()==b'user data'
symlink_root=fixtures/'symlink-root';symlink_root.mkdir();outside=fixtures/'outside';outside.mkdir();(symlink_root/'core').symlink_to(outside,target_is_directory=True)
assert reject(lambda:loader.install(archive,asset,symlink_root))
positive['ZIP_and_existing_file_rejections']={'cases':rejections,'existing_regular_file_preserved':True,'asset_subdirectory_symlink_rejected':True}

# PureWindowsPath follows the exact joinpath used by install after the member
# passed all existing checks. Actual Linux installation also accepts the name.
danger='core/codec/C:/escaped.txt';path,row,_=package(fixtures/'windows-drive',extra={danger:b'path fixture'})
loader.install(path,row,fixtures/'windows-drive-linux')
windows_destination=PureWindowsPath('D:/downloads/models')
joined=windows_destination.joinpath(*PurePosixPath(danger).parts)
assert not joined.is_relative_to(windows_destination)
findings['Windows_drive_member_accepted']={'member':danger,'accepted_by_actual_install_on_Linux':True,'exact_install_join_on_PureWindowsPath':str(joined),'chosen_destination':str(windows_destination),'joined_path_remains_inside_destination':False,'method':'Path semantics only on Windows; no actual Windows filesystem write attempted'}

# Late corrupt member leaves earlier files installed (non-atomic behavior).
late='core/codec/training/last.json';path,row,_=package(fixtures/'late-corrupt',extra={late:b'late corrupt fixture'},bad_member=late)
partial=fixtures/'partial';assert reject(lambda:loader.install(path,row,partial))
findings['late_inner_failure_partial_install']={'weight_written_before_rejection':(partial/'core/codec/training/weights.safetensors').exists(),'no_complete_download_receipt':not (partial/'download-receipt.json').exists(),'classification':'installation atomicity limitation, not existing-file overwrite'}

# Simulated public HTTP uses a fixed HTTPS URL and no supplied auth headers.
class Response(io.BytesIO):
    def __init__(self,raw,url='https://release-assets.githubusercontent.com/fixture',length=None):
        super().__init__(raw);self.url=url;self.headers={} if length is None else {'Content-Length':str(length)}
requests=[]
def response_for(raw,url='https://release-assets.githubusercontent.com/fixture',length=None):
    def open_fixture(request,timeout):requests.append((request,timeout));return Response(raw,url,length)
    return open_fixture
payload=b'anonymous download fixture';fetch_asset={'name':'codec.zip','bytes':len(payload),'sha256':sha_bytes(payload)}
with patch.object(loader.urllib.request,'urlopen',response_for(payload,length=len(payload))):loader.fetch({'release_tag':'fixture-v1'},fetch_asset,fixtures/'fetch-ok')
assert isinstance(requests[0][0],str) and requests[0][0]=='https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/fixture-v1/codec.zip'
for label,raw,url,length in [('HTTPS downgrade',payload,'http://fixture.invalid',None),('wrong length header',payload,'https://fixture.invalid',99),('too many bytes',payload+b'x','https://fixture.invalid',None),('short',payload[:-1],'https://fixture.invalid',None)]:
    with patch.object(loader.urllib.request,'urlopen',response_for(raw,url,length)):
        assert reject(lambda:loader.fetch({'release_tag':'fixture-v1'},fetch_asset,fixtures/('fetch-'+label.replace(' ','-'))))
with patch.object(loader.urllib.request,'urlopen',side_effect=urllib.error.URLError('https://signed.example/?private=query')):
    try:loader.fetch({'release_tag':'fixture-v1'},fetch_asset,fixtures/'fetch-error')
    except RuntimeError as err:assert 'private' not in str(err) and 'signed.example' not in str(err)
positive['anonymous_fetch_fixture']={'request_is_bare_fixed_HTTPS_URL':True,'explicit_auth_headers_supplied':False,'HTTPS_length_and_stream_bounds_checked':True,'redirect_query_not_logged':True,'real_network_requests':0}

# An archive and record can be self-consistent yet differ from frozen selection.
fake_repo=fixtures/'repo';selected={'selected':{'av':'av-time-rope-s11','choice':'c-augmented','choice_control':'c-compact','codec':'codec','talker':'talker'},'temperature':1.0,'threshold':None,'model_files':{}}
release={'schema':'omni-public-model-download-v1','repository':loader.REPOSITORY,'release_tag':'explicit-fixture-v1','model_selection':{'path':'docs/omni-products/v1/model-selection.json'},'components':{}}
archives={}
for kind,job in selected['selected'].items():
    weight=b'alternate unselected checkpoint bytes: '+job.encode()
    config={'codes':128,'width':96,'hop':80} if kind=='talker' else {}
    record={'final_opened':False,'optimizer_updates_observed':True,'training_revision':runtime.TRAINING_REVISION,'implementation_sha256':runtime.source_binding(),'source_unchanged_during_training':True,'config':{'model':config},'files':{'weights.safetensors':{'bytes':len(weight),'sha256':sha_bytes(weight)}}}
    directory=fixtures/('alternate-'+kind);path,assetrow,_=package(directory,job,extra={f'core/{job}/training/weights.safetensors':weight,f'core/{job}/training/training-result.json':json.dumps(record).encode()})
    archives[assetrow['name']]=path;release['components'][kind]=assetrow
    chosen=b'actually selected different bytes: '+job.encode()
    selected['model_files'][kind]={'job':job,'weights':{'bytes':len(chosen),'sha256':sha_bytes(chosen)},'training_result_sha256':sha_bytes(b'actual selected training receipt '+job.encode())}
selection_path=fake_repo/'docs/omni-products/v1/model-selection.json';write(selection_path,selected)
release['model_selection']['sha256']=sha(selection_path);release_path=fake_repo/'docs/omni-products/v1/public-release.json';write(release_path,release)
def copy_fetch(value,asset,target):Path(target).write_bytes(archives[asset['name']].read_bytes())
model_root=fixtures/'downloaded-alternate'
with patch.object(loader,'ROOT',fake_repo),patch.object(demo,'ROOT',fake_repo),patch.object(loader,'fetch',copy_fetch):
    assert loader.descriptor(release_path)==release
    downloaded=loader.download(release_path,model_root)
    assert set(downloaded['assets'])=={'av','choice','codec','talker'}
    mismatch=[]
    for kind in downloaded['assets']:
        job=selected['selected'][kind];weight_path=model_root/f'core/{job}/training/weights.safetensors'
        assert sha(weight_path)!=selected['model_files'][kind]['weights']['sha256'];mismatch.append(kind)
    parsed=[]
    class FakeModel:
        def load_state_dict(self,value,strict):assert strict is True
        def to(self,device):return self
        def eval(self):return self
        def requires_grad_(self,value):assert value is False;return self
    def parser(path,device):parsed.append(path);return {'fixture':torch.ones(1)}
    with patch.object(inf,'AVModel',lambda config:FakeModel()),patch.object(inf,'ChoiceModel',lambda config:FakeModel()),patch.object(inf,'SoundCodec',lambda config:FakeModel()),patch.object(inf,'ProgramTalker',lambda *args:FakeModel()),patch.object(inf,'load_file',parser),patch.object(demo,'OmniAssistant',lambda *args,**kwargs:SimpleNamespace(models=args)):
        product,published=demo.assistant(model_root,release_path)
    assert len(parsed)==4 and len(product.models)==4
    findings['selected_weight_binding_not_enforced']={'descriptor_references_valid_selection_SHA':True,'archive_outer_inner_self_SHA_checks_pass':True,'unselected_checkpoint_kinds':mismatch,'demo_reached_model_constructor_and_composer':True,'selected_training_result_SHA_also_not_enforced':True,'weight_parsers_and_constructors_mocked':True,'real_weights_or_predictions':False}
    # A dangling receipt symlink is lexically occupied, but .exists() is false.
    symlink_destination=fixtures/'receipt-link';symlink_destination.mkdir();outside_receipt=fixtures/'outside-receipt.json';(symlink_destination/'download-receipt.json').symlink_to(outside_receipt)
    assert not outside_receipt.exists()
    loader.download(release_path,symlink_destination)
    assert outside_receipt.exists()
    findings['download_receipt_dangling_symlink_followed']={'outside_fixture_target_created':True,'lexically_existing_receipt_link_not_rejected':True,'real_user_files_touched':False}

# WAV output path is also not protected against a dangling symlink.
wave_link=fixtures/'output-link.wav';outside_wave=fixtures/'outside-output.wav';wave_link.symlink_to(outside_wave)
demo.save_wave(wave_link,torch.zeros(80));assert outside_wave.exists()
with wave.open(str(outside_wave),'rb') as wav:assert wav.getframerate()==8000 and wav.getnframes()==80
findings['WAV_dangling_symlink_followed']={'outside_fixture_target_created':True,'existing_link_not_rejected':True,'PCM_sample_rate':8000,'samples':80}
existing_wave=fixtures/'existing.wav';existing_wave.write_bytes(b'original user file')
assert reject(lambda:demo.save_wave(existing_wave,torch.zeros(80))) and existing_wave.read_bytes()==b'original user file'
assert reject(lambda:demo.save_wave(fixtures/'bad.wav',torch.tensor([float('nan')])))
positive['WAV_regular_existing_and_nonfinite_rejected']=True

# Presentation probes use explicit wrong-answer/refusal stubs, not actual models.
question_seen=[]
def judge(question,information):
    assert question.gold is None and information.gold is None;question_seen.append(question)
    return {'prediction':{'answer':'stop','ids':['record','stop','none'],'logits':[0.,2.,0.],'probabilities':[.1,.8,.1],'p_max':.8},'accepted':False,'decision':'handoff'}
public_product=SimpleNamespace(c=SimpleNamespace(judge=judge),video=lambda item,with_sound:{'answer':'unknown','logits':[0.,0.,2.],'source':'explicit refusal fixture'},sound_chunks=lambda program:iter([{'samples':torch.zeros(80),'ready_seconds':.3,'whole_code_generation_seconds':.2},{'samples':torch.zeros(80),'ready_seconds':.4,'whole_code_generation_seconds':.2}]))
shown={}
for mode in ('choice','av','sound'):
    output=io.StringIO()
    with patch.object(demo,'assistant',lambda *args:(public_product,{'release_tag':'explicit presentation fixture'})),patch.object(sys,'argv',['public_demo',mode,'--model-root',str(model_root)]),contextlib.redirect_stdout(output):demo.main()
    shown[mode]=json.loads(output.getvalue())
assert shown['choice']['prediction']['answer']=='stop' and shown['choice']['decision']=='handoff'
assert shown['av']['answer']=='unknown' and 'gold' not in shown['av']
assert shown['sound']['samples']==160 and shown['sound']['chunks']==2 and shown['sound']['first_playable_seconds']==.3 and shown['sound']['whole_code_generation_seconds']==.2
assert reject(lambda:demo.public_question({**demo.choice_example(),'gold':'record'}))
assert reject(lambda:demo.public_question({**demo.choice_example(),'state':'\ud800'}))
public_product.sound_chunks=lambda program:(_ for _ in ()).throw(ValueError('explicit actual generation failure'))
with patch.object(demo,'assistant',lambda *args:(public_product,{'release_tag':'fixture'})),patch.object(sys,'argv',['public_demo','sound','--model-root',str(model_root)]):assert reject(demo.main)
positive['demo_actual_value_and_failure_presentation']={'question_gold_not_sent':True,'intentionally_wrong_answer_kept':'stop','unknown_refusal_kept':True,'sound_fields_match_generator_chunks':True,'generation_error_not_replaced_by_reference_or_silence':True,'default_AV_uses_declared_train_pattern':True}

report={'reviewer_task':'/root/omni_final_definition_review','completed_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),'method':'Offline private ZIP/HTTP/path/selection fixtures, mocked weight parsing/constructors and explicit output stubs only; no live download, actual weight load, model final, teacher or paid execution','positive_controls':positive,'findings_reproduced':findings,'actual_public_release_present':(ROOT/'docs/omni-products/v1/public-release.json').exists(),'actual_model_selection_present':(ROOT/'docs/omni-products/v1/model-selection.json').exists(),'demo_timing_scope_limit':'CLI collects all chunks before printing/export; first_playable_seconds is internal generator sample readiness, not measured CLI file/player delivery time'}
p=E/'probe-evidence-original-v1.json'
with p.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'findings':list(findings),'positive_groups':list(positive),'actual_release_present':report['actual_public_release_present'],'no_live_or_model_predictions':True},ensure_ascii=False,indent=2))
