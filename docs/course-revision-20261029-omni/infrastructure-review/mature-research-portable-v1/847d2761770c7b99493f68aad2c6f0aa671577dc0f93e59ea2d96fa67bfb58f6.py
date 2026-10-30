"""Anonymous fixed-revision HF metadata only; never fetch a weight/config/blob body."""
import concurrent.futures,datetime,fnmatch,hashlib,json,sys,urllib.request
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,'/workspace/work/omni-models-v1')
from scripts.omni_revision.research_model_files import MODELS,PATTERNS
folder=BASE/'external-metadata-current-v3';folder.mkdir(exist_ok=False)
def get(item):
 model,(revision,bound)=item;url='https://huggingface.co/api/models/'+model+'/revision/'+revision+'?blobs=true'
 request=urllib.request.Request(url,headers={'User-Agent':'independent-mature-source-review','Accept':'application/json'})
 with urllib.request.urlopen(request,timeout=30) as response:
  raw=response.read(2*1024**2+1);final_url=response.url
 if len(raw)>2*1024**2:raise ValueError('metadata body exceeds2MiB')
 name=model.replace('/','--')+'.metadata.json.raw';(folder/name).write_bytes(raw);info=json.loads(raw);assert info['sha']==revision
 rows=[r for r in info['siblings'] if any(fnmatch.fnmatchcase(r['rfilename'],p) for p in PATTERNS)];total=sum(r['size'] for r in rows);names={r['rfilename'] for r in rows};assert len(rows)==len(names) and 0<len(rows)<=128 and total<=bound and 'config.json' in names and any(n.endswith('.safetensors') for n in names)
 if model.startswith('Qwen/'):assert 'spk_dict.pt' in names
 return {'model':model,'revision':revision,'actual_commit':info['sha'],'source_url':url,'final_url':final_url,'observed_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),'raw_metadata_path':str(folder/name),'raw_metadata_sha256':hashlib.sha256(raw).hexdigest(),'raw_metadata_bytes':len(raw),'selected_files':len(rows),'selected_total_bytes':total,'bound_bytes':bound,'filenames_have_no_glob_metacharacters':all(not any(c in n for c in '*?[]') for n in names),'files':[{'path':r['rfilename'],'bytes':r['size'],'source_blob_id':r.get('blobId'),'source_lfs_sha256':r.get('lfs',{}).get('sha256')} for r in rows]}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(get,MODELS.items()))
record={'scope':'Independent anonymous HTTPS metadata endpoints only, exact whitelisted commits. No model config/weight/speaker/blob body download or neural/provider/paid call. Raw external responses stay outside Git.','models':rows,'weight_downloads':0,'neural_loads_or_predictions':0,'paid_calls':0}
p=BASE/'current-v3-metadata-only-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({**record,'models':[{k:v for k,v in r.items() if k!='files'} for r in rows]},ensure_ascii=False,indent=2))
