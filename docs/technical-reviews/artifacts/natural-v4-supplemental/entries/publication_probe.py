from pathlib import Path,PurePosixPath
import concurrent.futures,hashlib,json,urllib.request
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[4];RAW=ROOT/'outputs/natural-v4/factual-research/entries'
manifest=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
jobs=[]
for model in manifest['models']:
 weights=[f for f in model['files'] if f['path'].endswith(('.pt','.safetensors'))]
 dirs=sorted(set(str(PurePosixPath(f['path']).parent) for f in weights))
 for directory in dirs:jobs.append((model['id'],model['repo'],model['revision'],directory,weights))
def fetch(job):
 id,repo,revision,directory,weights=job
 url=f'https://huggingface.co/api/models/{repo}/tree/{revision}/{directory}?recursive=true&expand=false'
 with urllib.request.urlopen(url,timeout=20) as r:raw=r.read();assert r.status==200
 rows=json.loads(raw);remote={f['path']:f for f in rows};matches=[]
 for weight in weights:
  if weight['path'] not in remote:continue
  source=remote[weight['path']];assert source['size']==weight['bytes']
  lfs=source.get('lfs',{});digest=lfs.get('oid')
  if digest:assert digest==weight['sha256']
  matches.append({'path':weight['path'],'bytes':weight['bytes'],'lfs_sha256':digest,'matches_fixed_manifest':True})
 assert matches
 name='published-'+id+'-'+hashlib.sha256(directory.encode()).hexdigest()[:8]+'.json';(RAW/name).write_bytes(raw)
 return {'id':id,'url':url,'anonymous':True,'response_sha256':hashlib.sha256(raw).hexdigest(),'response_bytes':len(raw),'raw_path':str((RAW/name).relative_to(ROOT)),'weights':matches}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,jobs))
assert len(set(f['path'] for row in rows for f in row['weights']))==120
cap=json.loads((ROOT/'docs/course-experiments/capstone-public.json').read_text());remote={f['path']:f for f in json.loads((RAW/'hf-course-tree.json').read_text())}
checks=[]
for model in cap['models']:
 for weight in model['files']:
  if weight['path'].endswith('.pt'):
   f=remote[weight['path']];assert f['size']==weight['bytes'];assert f['lfs']['oid']==weight['sha256']
   checks.append({'id':model['id'],'path':weight['path'],'bytes':weight['bytes'],'sha256':weight['sha256']})
result={'limits':'Anonymous fixed-version remote file metadata inspection only; no weights downloaded or model inference run. Public metadata confirms sizes and LFS content hashes, not model quality or exact training resume.','course':rows,'unique_course_weight_files':120,'capstone':checks,'capstone_repo':cap['repo'],'capstone_revision':cap['revision']}
(OUT/'publication-record.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified remote fixed metadata for 120 course weights and',len(checks),'capstone weights; no weight download.')
