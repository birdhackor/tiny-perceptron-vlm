"""Read-only bounded identity/delta probe. Output is mechanical evidence, never an AI verdict."""
import argparse,hashlib,json,re,sys
from datetime import datetime,UTC
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def readref(ref):
 p=(ROOT/ref['path']).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('outside repo')
 raw=p.read_bytes()
 if digest(raw)!=ref['sha256']:raise ValueError('stale ref: '+ref['path'])
 return raw
p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();j=json.loads(a.job.read_bytes());page=j['canonical_or_auxiliary_page'];rows=[]
old=json.loads(readref(j['prior_current_report']));body=readref({'path':page['snapshot'],'sha256':page['source_sha256']});rows.append({'check':'actual_frozen_page_identity','page_id':j['lesson_id'],'source_sha256':digest(body),'prior_source_sha256':old['source_sha256'],'same_body':digest(body)==old['source_sha256']})
for update in j['required_repository_source_sha_updates']:
 actual=digest((ROOT/update['path']).read_bytes())
 if actual!=update['new_sha256']:raise ValueError('current source moved')
 rows.append({'check':'whole_file_dependency_identity',**{k:update[k] for k in ['source_id','path','old_sha256','new_sha256']}})
for change in j['authorized_exact_changes']:
 raw=(ROOT/change['path']).read_bytes();text=raw.decode();restored=text
 if change.get('index_addition'):
  idx=change['index_addition']
  if restored.count(idx['new'])!=1:raise ValueError('index change no longer unique')
  restored=restored.replace(idx['new'],idx['old'],1)
 if restored.count(change['new'])!=1:raise ValueError('authorized change no longer unique')
 restored=restored.replace(change['new'],change['old'],1)
 if digest(restored.encode())!=change['before_sha256'] or digest(raw)!=change['after_sha256']:raise ValueError('source delta outside authorization')
 rows.append({'check':'only_authorized_exact_change','path':change['path'],'before_sha256':digest(restored.encode()),'after_sha256':digest(raw)})
 for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',change['new']):
  if '://' in target:continue
  name,_,anchor=target.partition('#');dest=((ROOT/change['path']).parent/name).resolve()
  if not dest.is_file():raise ValueError('missing link target '+target)
  if anchor and not re.search(r'^## '+re.escape(anchor)+r'\s',dest.read_text(),re.M):raise ValueError('missing numbered anchor '+target)
  rows.append({'check':'added_link_target_and_numbered_anchor','label':label,'target':dest.relative_to(ROOT).as_posix(),'anchor':anchor,'target_sha256':digest(dest.read_bytes())})
bind=j.get('required_new_bundle_artifact')
if bind:
 path=ROOT/bind['path']
 if path.exists():
  raw=path.read_bytes();bundle=json.loads(raw);canonical={'path':'docs/course-small-patch-20261102/source-freeze.json','sha256':digest((ROOT/'docs/course-small-patch-20261102/source-freeze.json').read_bytes())}
  if bundle.get('canonical_source_freeze')!=canonical:raise ValueError('new numbered bundle has wrong canonical freeze')
  rows.append({'check':'numbered_bundle_reference_and_scope','path':bind['path'],'sha256':digest(raw),'lesson_ids':[e['lesson_id'] for e in bundle['sections']],'canonical_source_freeze':canonical,'note':'Does not assert reviewer comprehension or collection chronology.'})
 else:rows.append({'check':'numbered_bundle_pending','path':bind['path'],'blocking_for_collector':True})
result={'kind':'mechanical_bounded_identity_evidence_not_ai_verdict','lesson_id':j['lesson_id'],'created_at':datetime.now(UTC).isoformat(),'checks':rows,'unverified':['New reviewer judgment, old scientific evidence retention, added claim semantics, actual collection receipts and phase closure.'],'command':' '.join(sys.argv),'environment':{'python':sys.version.split()[0],'device':'CPU only; hashes/text parsing; no model import, training, or deployment'}}
raw=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
if a.output:
 output=a.output.resolve()
 if not output.is_relative_to(ROOT/'docs/course-small-patch-20261102'):raise ValueError('output must stay in authorized candidate directory')
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:f.write(raw)
print(raw)
