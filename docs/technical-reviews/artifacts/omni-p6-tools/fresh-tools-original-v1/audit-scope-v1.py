"""Complete mechanical inspection of the neutral 460-file scope; no peer answers printed."""
from pathlib import Path
import ast,collections,datetime,hashlib,json,tomllib
ROOT=Path('/workspace/work/omni-book-integrated-v1')
OUT=Path('/workspace/work/omni-p6-tools-independent-v1')
NEUTRAL=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-tools-neutral-inputs-v1.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def references(value,pointer=''):
 if isinstance(value,dict):
  if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and len(value['sha256'])==64:
   yield pointer,value['path'],value['sha256']
  for key,child in value.items():yield from references(child,pointer+'/'+str(key).replace('~','~0').replace('/','~1'))
 elif isinstance(value,list):
  for i,child in enumerate(value):yield from references(child,pointer+'/'+str(i))
manifest=json.loads(NEUTRAL.read_bytes())
files=[];references_checked=[];reference_counts=collections.Counter();errors=[]
for entry in manifest['files']:
 path=ROOT/entry['path'];raw=path.read_bytes();snapshot=Path(entry['snapshot']).read_bytes()
 record={'path':entry['path'],'sha256':sha(raw),'bytes':len(raw),'line_count':len(raw.splitlines()),'snapshot':entry['snapshot'],'snapshot_sha256':sha(snapshot),'neutral_sha256':entry['sha256'],'current_matches_neutral':sha(raw)==entry['sha256'],'raw_matches_current':raw==snapshot}
 if not record['current_matches_neutral'] or not record['raw_matches_current']:errors.append({'path':entry['path'],'kind':'scope_mismatch'})
 source=raw.decode('utf-8');record['utf8']='valid';suffix=path.suffix
 if suffix=='.json':
  value=json.loads(source);types=collections.Counter();keys=collections.Counter()
  def shape(value):
   types[type(value).__name__]+=1
   if isinstance(value,dict):
    for key,child in value.items():keys[key]+=1;shape(child)
   elif isinstance(value,list):
    for child in value:shape(child)
  shape(value)
  record['structure']={'root_type':type(value).__name__,'node_types':dict(types),'distinct_key_count':len(keys),'key_schema_sha256':sha(json.dumps(sorted(keys.items()),ensure_ascii=False).encode())}
  refs=list(references(value));record['path_sha_reference_count']=len(refs)
  for pointer,name,expected in refs:
   target=Path(name) if name.startswith('/') else ROOT/name
   ref={'source':entry['path'],'pointer':pointer,'target':name,'expected_sha256':expected}
   if target.is_file():
    actual=sha(target.read_bytes());ref.update(actual_sha256=actual,status='match' if actual==expected else 'mismatch')
   else:ref['status']='not_present_here'
   reference_counts[ref['status']]+=1;references_checked.append(ref)
 elif suffix=='.py':
  tree=ast.parse(source,filename=entry['path']);nodes=list(ast.walk(tree))
  record['structure']={'ast_nodes':len(nodes),'function_count':sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes),'class_count':sum(isinstance(n,ast.ClassDef) for n in nodes),'imports':[ast.unparse(n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom))]}
 elif suffix in {'.toml','.lock'}:
  value=tomllib.loads(source);record['structure']={'top_level_keys':sorted(value),'parsed':True}
 else:record['structure']={'full_utf8_line_enumeration':True}
 files.append(record)
result={'reviewer_task':'/root/omni_p6_technical_tools','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'neutral_sha256':sha(NEUTRAL.read_bytes()),'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'errors':errors,'reference_counts':dict(reference_counts),'files':files,'reference_resolution_limit':'Generic path+sha pairs use repository-root/absolute resolution only. Relative paths within packages, logical records, remote evidence and historical mutable locations require context and are not inferred current-source failures. No old report verdict/issue values are emitted.'}
full=OUT/'full-scope-mechanical-inventory-v2.json';refs=OUT/'path-sha-reference-results-v2.json'
for path,value in [(full,result),(refs,references_checked)]:
 with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
summary={key:value for key,value in result.items() if key!='files'}
summary['large_evidence']=[{'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in [full,refs]]
with Path(__file__).with_name('full-scope-summary-v1.json').open('x') as stream:stream.write(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({key:value for key,value in summary.items() if key not in {'reference_resolution_limit','large_evidence'}},ensure_ascii=False))
