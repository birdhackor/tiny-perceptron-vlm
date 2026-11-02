"""Fresh CPU-only adversarial contract probes using current first-party fixtures.

Fixtures are synthetic; baseline replay/inventory/build are the fixture's declared
stubs. No actual reader result, training, model asset or formal final is executed.
"""
from pathlib import Path
import copy,datetime,hashlib,importlib.util,json,sys
ROOT=Path('/workspace/work/omni-book-integrated-v1')
OUT=Path('/workspace/work/omni-p6-tools-independent-v1/negative-control-fixtures-v2')
OUT.mkdir(exist_ok=False)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path)
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
base=load('fresh_tool_history_fixture','tests/test_historical_rechecks.py')
inc=load('fresh_incremental_fixture','tests/test_incremental_revision_review.py')
pre=load('fresh_preflight','docs/course-review-20261019/tools/preflight.py')
results=[]
def outcome(name,fn,expected):
 try:
  value=fn(); observed='accepted'; details=value
 except Exception as e:
  observed='rejected'; details={'type':type(e).__name__,'message':str(e)}
 results.append({'name':name,'expected':expected,'observed':observed,'details':details})

def tool_case(variant):
 root=OUT/('tool-'+variant); root.mkdir()
 built=base.fixture(root,only_tool=True)
 manifest,live,tasks,sealed=built
 sidecar=base.review.document(root,manifest['historical_fix_rechecks'])
 if variant!='valid-fixture':
  group=sidecar['rechecks'][0]
  full=base.review.document(root,group['full_review'])
  original={'reviewer_task':base.NEW_TOOL,'verdict':'pass','checked_files':copy.deepcopy(full['checked_files'])}
  if variant=='original-fake-digests':
   for row in original['checked_files']:row['sha256']='0'*64
  ref=base.save(root,'current/fresh-original-shell.json',original)
  group['original_review']=ref
  phase=sealed['new']['technical']
  phase['report_sha256s'].append(ref['sha256']);phase['collection_index'].append({**ref,'received_at':base.at(22)})
  manifest['issue_reports'].append(ref)
  manifest['historical_fix_rechecks']=base.save(root,'current/fresh-child-chain.json',sidecar)
 manifest['historical_fix_rechecks']=base.save(root,'current/fresh-v2-chain.json',{'schema_version':2,'chains':[manifest['historical_fix_rechecks']]})
 return base.check(root,built)
for variant in ['valid-fixture','original-shell','original-fake-digests']:
 outcome('schema2-tool-'+variant,lambda v=variant:tool_case(v),'accepted' if variant=='valid-fixture' else 'rejected')

def replace_ref(data,old,new):
 if isinstance(data,dict):
  if data.get('path')==old['path'] and data.get('sha256')==old['sha256']:
   data.update(new)
  for value in data.values():replace_ref(value,old,new)
 elif isinstance(data,list):
  for value in data:replace_ref(value,old,new)

def gate_case(variant):
 root=OUT/('gate-'+variant);root.mkdir()
 path=root/inc.gate_round(root,**({'listed':False,'reader_on':inc.PATCHED,'technical_on':inc.PATCHED,'chapter_cited':inc.FILE_PATCHED} if variant=='basis-without-current-trace' else {}))
 manifest=json.loads(path.read_bytes())
 if variant=='future-technical-report':
  old=copy.deepcopy(manifest['pages'][0]['technical']['report'])
  report=json.loads((root/old['path']).read_bytes());report['created_at']='2035-01-01T00:00:00+00:00'
  new=inc.save(root,old['path'],report)
  replace_ref(manifest,old,new)
  for wave in manifest['review_waves']:
   ref=wave['seals']['technical'];seal=json.loads((root/ref['path']).read_bytes())
   seal['report_sha256s']=[new['sha256'] if x==old['sha256'] else x for x in seal['report_sha256s']]
   wave['seals']['technical']=inc.save(root,ref['path'],seal)
 if variant=='uncollected-technical-report':
  for wave in manifest['review_waves']:
   ref=wave['seals']['technical'];seal=json.loads((root/ref['path']).read_bytes());seal['collection_index']=[]
   wave['seals']['technical']=inc.save(root,ref['path'],seal)
 inc.save(root,path.relative_to(root).as_posix(),manifest)
 stage='reader' if variant=='basis-without-current-trace' else 'technical'
 result=inc.review.audit_manifest(root,path,stage)
 if result['failures']:raise ValueError(result['failures'])
 basis_errors=pre.basis_read(root,manifest['pages'][0]['reader'])
 return {'gate_status':result['status'],'stage':stage,'gate_failures':result['failures'],'coordinator_basis_read_errors':basis_errors,'fixture_scope':'current real incremental audit; fixture baseline replay/current inventory/Notebook build are explicit stubs'}
for variant in ['valid-fixture','basis-without-current-trace','future-technical-report','uncollected-technical-report']:
 outcome('ordinary-section-'+variant,lambda v=variant:gate_case(v),'accepted' if variant=='valid-fixture' else 'rejected')
output={'schema_version':1,'reviewer_task':'/root/omni_p6_technical_tools','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'device':'cpu','scope_note':'Independent contract negative controls. Expected outcomes express the neutral source/receipt/original-preservation contract, not scientific model quality. No actual reports or source files were mutated. Fixtures remain outside Git.','results':results}
print(json.dumps(output,ensure_ascii=False,indent=2))
