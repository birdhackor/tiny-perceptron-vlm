"""Actual bounded snapshot helper against small private LFS/Git metadata/file fixtures, no weight downloads."""
import copy,hashlib,json,sys,types
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,'/workspace/work/omni-models-v1')
from scripts.omni_revision import research_model_files as mf
root=BASE/'probes/current-v3-model-files';root.mkdir(parents=True,exist_ok=False);cases=[]
for mode in ('Qwen_valid','Whisper_valid','wrong_commit','missing_speaker','bad_LFS','bad_Git_blob','size_mismatch','download_total_bound','file_count_bound','unsafe_path','wrong_snapshot_root','wrong_requested_revision'):
 model='openai/whisper-small' if mode=='Whisper_valid' else 'Qwen/Qwen2.5-Omni-3B';revision=mf.MODELS[model][0];folder=root/mode;folder.mkdir();snapshot=folder/revision;snapshot.mkdir();contents={'config.json':b'{"private_fixture":true}','model.safetensors':b'own fake weight bytes, never parsed or loaded'}
 if model.startswith('Qwen/'):contents['spk_dict.pt']=b'own speaker bytes, never parsed as checkpoint'
 rows=[]
 for name,raw in contents.items():
  p=snapshot/name;p.write_bytes(raw);digest,git=mf.file_digests(p)
  lfs=None
  if name.endswith(('.safetensors','.pt')):
   pointer=('version https://git-lfs.github.com/spec/v1\noid sha256:'+digest+'\nsize '+str(len(raw))+'\n').encode();git=hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest();lfs=types.SimpleNamespace(sha256=digest)
  rows.append(types.SimpleNamespace(rfilename=name,size=len(raw),blob_id=git,lfs=lfs))
 selected=set(contents)
 if mode=='missing_speaker':rows=[r for r in rows if r.rfilename!='spk_dict.pt']
 if mode=='bad_LFS':next(r for r in rows if r.rfilename=='model.safetensors').lfs.sha256='0'*64
 if mode=='bad_Git_blob':next(r for r in rows if r.rfilename=='config.json').blob_id='0'*40
 if mode=='size_mismatch':rows[0].size+=1
 if mode=='download_total_bound':rows[0].size=mf.MODELS[model][1]+1
 if mode=='file_count_bound':rows.extend(types.SimpleNamespace(rfilename=f'extra-{i}.json',size=1,blob_id='0'*40,lfs=None) for i in range(126))
 if mode=='unsafe_path':rows.append(types.SimpleNamespace(rfilename='../outside.json',size=1,blob_id='0'*40,lfs=None))
 calls=[]
 class API:
  def __init__(self,**kwargs):assert kwargs=={'token':False};calls.append('anonymous_metadata_client')
  def model_info(self,which,**kwargs):assert which==model and kwargs=={'revision':revision,'files_metadata':True};calls.append('fixed_revision_metadata');return types.SimpleNamespace(sha='0'*40 if mode=='wrong_commit' else revision,siblings=rows)
 def download(**kwargs):
  calls.append('snapshot_download');assert kwargs=={'repo_id':model,'revision':revision,'token':False,'allow_patterns':sorted(r.rfilename for r in rows),'max_workers':4};return str(folder if mode=='wrong_snapshot_root' else snapshot)
 hub=types.ModuleType('huggingface_hub');hub.HfApi=API;hub.snapshot_download=download
 with patch.dict(sys.modules,{'huggingface_hub':hub}):
  try:directory,note=mf.frozen_snapshot(model,'wrong' if mode=='wrong_requested_revision' else revision);entry={'case':mode,'accepted':True,'files':{n:rec for n,rec in note['files'].items()},'local_root_name':directory.name}
  except Exception as e:entry={'case':mode,'accepted':False,'error_type':type(e).__name__,'reason':str(e)}
 expected=mode in ('Qwen_valid','Whisper_valid');assert entry['accepted'] is expected,entry;entry['events']=calls;cases.append(entry)
 if mode in ('wrong_commit','missing_speaker','download_total_bound','file_count_bound','unsafe_path','wrong_requested_revision'):assert 'snapshot_download' not in calls,entry
record={'scope':'Actual research_model_files helper, with anonymous API/snapshot transport fixtures and tiny own file bytes. All LFS SHA256/Git blob calculations are real; none of the files are neural weights and none were parsed by model loaders.','cases':cases,'all_expected_outcomes':True,'real_weight_downloads_or_NN_or_paid_or_final_calls':0}
p=BASE/'current-v3-model-files-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
