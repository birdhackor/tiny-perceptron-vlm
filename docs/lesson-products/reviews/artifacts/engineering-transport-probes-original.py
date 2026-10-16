"""Isolated-worktree transport audit: real local hashes, explicit Git/gh fixture replies only."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
TREE=Path('/workspace/work/lesson-products-transport')
ART=Path(__file__).parent
source=TREE/'scripts/lesson_products/upload_transport.py'
spec=importlib.util.spec_from_file_location('isolated_transport_under_review',source)
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 started=datetime.now(UTC).isoformat()
 names=['scripts/lesson_products/upload_transport.py','.github/workflows/lesson-products-transport.yml','docs/lesson-products/jobs/transport-v1.json','.gitattributes']
 before={name:sha(TREE/name) for name in names}
 actual=json.loads((TREE/names[2]).read_bytes());asset=TREE/actual['asset_path']
 assert transport.digest(asset)==actual['sha256'] and asset.stat().st_size==actual['bytes']==20761964
 git={}
 for name,command in [('branch',['git','branch','--show-current']),('head',['git','rev-parse','HEAD']),('attributes',['git','check-attr','filter','diff','merge','text','--',actual['asset_path']]),('lfs_pointer',['git','lfs','pointer','--file',actual['asset_path']]),('head_pointer',['git','show','HEAD:'+actual['asset_path']])]:
  run=subprocess.run(command,cwd=TREE,capture_output=True,text=True)
  git[name]={'command':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
 expected=f"version https://git-lfs.github.com/spec/v1\noid sha256:{actual['sha256']}\nsize {actual['bytes']}\n"
 assert git['lfs_pointer']['exit_code']==0 and git['lfs_pointer']['stdout']==expected
 assert git['branch']['stdout'].strip()=='lesson-products-transport-20261016'
 results=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='transport-audit-') as temporary:
  work=Path(temporary)
  for kind in ('positive','dry_run','file_tamper','pointer_oid','pointer_is_full_blob','source_symlink','unsafe_path','wrong_repo','wrong_environment_repo','bytes_bool','size_over_limit','release_id_bool','parent_not_draft','parent_wrong_commit','parent_duplicate','upload_error','after_parent_wrong_commit','after_extra_asset','after_wrong_name','after_wrong_size','roundtrip_badsha','roundtrip_badsize'):
   case=work/kind;case.mkdir();fixture=copy.deepcopy(actual);raw=b'explicit synthetic archive fixture, not a model or real release'
   fixture['sha256']=hashlib.sha256(raw).hexdigest();fixture['bytes']=len(raw)
   package=case/fixture['asset_path'];package.parent.mkdir(parents=True);package.write_bytes(raw)
   pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{fixture['sha256']}\nsize {fixture['bytes']}\n".encode()
   descriptor=case/'descriptor.json'
   parent={'draft':True,'target_commitish':fixture['release_commit'],'tag_name':'synthetic-transport-tag','assets':[]}
   asset_info={'id':123456,'name':fixture['asset_name'],'size':fixture['bytes']}
   events=[];queries=0;uploads=0;downloads=0
   current_kind='positive'
   def check_output(command,cwd=None):
    nonlocal queries
    if command[:2]==['git','show']:
     assert command==['git','show','HEAD:'+fixture['asset_path']] and cwd==case
     events.append('read_HEAD_pointer');return pointer
    assert command==['gh','api',f"repos/{fixture['repo']}/releases/{fixture['release_id']}"]
    events.append('read_draft_parent');queries+=1
    value=copy.deepcopy(parent)
    if queries>=2:
     value['assets']=[copy.deepcopy(asset_info)]
     if current_kind=='after_parent_wrong_commit':value['target_commitish']='f'*40
     elif current_kind=='after_extra_asset':value['assets'].append({'id':999,'name':'other','size':1})
     elif current_kind=='after_wrong_name':value['assets'][0]['name']='other.tar.gz'
     elif current_kind=='after_wrong_size':value['assets'][0]['size']+=1
    return json.dumps(value).encode()
   def run(command,check,stdout=None):
    nonlocal uploads,downloads
    assert check is True
    if command[:3]==['gh','release','upload']:
     assert command==['gh','release','upload',parent['tag_name'],str(package),'--repo',fixture['repo']]
     assert '--clobber' not in command
     assert events[-1]=='read_draft_parent'
     events.append('upload_without_clobber');uploads+=1
     if current_kind=='upload_error':raise subprocess.CalledProcessError(1,command)
     return
    assert command==['gh','api',f"repos/{fixture['repo']}/releases/assets/123456",'-H','Accept: application/octet-stream']
    events.append('download_by_asset_id');downloads+=1
    if current_kind=='roundtrip_badsha':stdout.write(b'x'*len(raw))
    elif current_kind=='roundtrip_badsize':stdout.write(raw+b'extra')
    else:stdout.write(raw)
   with patch.object(transport,'ROOT',case),patch.object(transport.subprocess,'check_output',side_effect=check_output):
    assert transport.verify_local(fixture)==package
    assert transport.parent(fixture)['draft'] is True
   events.clear();queries=0;current_kind=kind
   if kind=='file_tamper':package.write_bytes(b'x'*len(raw))
   elif kind=='pointer_oid':pointer=pointer.replace(fixture['sha256'].encode(),b'0'*64)
   elif kind=='pointer_is_full_blob':pointer=raw
   elif kind=='source_symlink':target=case/'outside';target.write_bytes(raw);package.unlink();package.symlink_to(target)
   elif kind=='unsafe_path':fixture['asset_path']='../outside.tar.gz'
   elif kind=='wrong_repo':fixture['repo']='other/repo'
   elif kind=='bytes_bool':fixture['bytes']=True
   elif kind=='size_over_limit':fixture['bytes']=64*1024*1024+1
   elif kind=='release_id_bool':fixture['release_id']=True
   elif kind=='parent_not_draft':parent['draft']=False
   elif kind=='parent_wrong_commit':parent['target_commitish']='e'*40
   elif kind=='parent_duplicate':parent['assets']=[copy.deepcopy(asset_info)]
   descriptor.write_text(json.dumps(fixture))
   argv=['upload_transport','--descriptor',str(descriptor)]+(['--dry-run'] if kind=='dry_run' else [])
   env={'GITHUB_REPOSITORY':'wrong/repo' if kind=='wrong_environment_repo' else fixture['repo'],'GITHUB_SHA':'b'*40}
   caught=None;console=io.StringIO()
   with patch.dict(os.environ,env),patch.object(sys,'argv',argv),patch.object(transport,'ROOT',case),patch.object(transport.subprocess,'check_output',side_effect=check_output),patch.object(transport.subprocess,'run',side_effect=run),contextlib.redirect_stdout(console):
    try:transport.main()
    except (ValueError,subprocess.CalledProcessError) as error:caught={'type':type(error).__name__,'message':str(error)}
   good=kind in ('positive','dry_run')
   assert (caught is None)==good,kind
   if kind=='positive':
    value=json.loads((case/'outputs/lesson-products-transport/receipt.json').read_bytes())
    assert uploads==downloads==1 and value['status']=='completed' and value['modal_started'] is False
    assert value['asset_id']==123456 and value['transport_sha256']==value['download_sha256']==fixture['sha256']
   elif kind=='dry_run':assert uploads==downloads==0 and not (case/'outputs').exists()
   elif kind in ('upload_error','after_parent_wrong_commit','after_extra_asset','after_wrong_name','after_wrong_size','roundtrip_badsha','roundtrip_badsize'):
    assert uploads==1
    assert not (case/'outputs/lesson-products-transport/receipt.json').exists()
   else:assert uploads==downloads==0
   results.append({'id':kind,'valid_local_and_parent_base_passed':True,'passed':True,'mock_upload_calls':uploads,'mock_asset_id_downloads':downloads,'events':events,'error':caught,'stdout':console.getvalue().strip(),'scope':'Real verify_local/parent/main; Git HEAD pointer/gh are explicit fixture replies, all files under isolated temp root.'})
 assert before=={name:sha(TREE/name) for name in names}
 report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed engineering original','started_at':started,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'source_worktree':str(TREE),'source_branch':git['branch']['stdout'].strip(),'source_head_at_review':git['head']['stdout'].strip(),'checked_files':[{'path':str(TREE/name),'repo_relative_path':name,'sha256':digest} for name,digest in before.items()],'actual_asset':{'path':str(asset),'sha256':actual['sha256'],'bytes':asset.stat().st_size,'content_not_copied_to_audit':True},'actual_git_reads':git,'results':results,'all_expected':True,'limits':['No actual gh upload/download or remote API, Modal/GPU/HF/ledger/paid calls. Only local Git metadata and LFS pointer calculation read actual worktree.','Actual pending source package20.7MB hashed/sizechecked; fixtures are small synthetic bytes. No tar/model quality evaluation.','Current HEAD821 lacks pending LFS file; postcommit HEAD pointer still must be verified before real operation. git lfs pointer computed from current actual bytes matches expected pointer exactly, but is not called an existing committed blob.']}
 path=ART/'engineering-transport-probes-results-original.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'cases':len(results),'all_expected':True,'sha256':sha(path),'actual_lfs_pointer_matches':True,'precommit_head_pointer_exit':git['head_pointer']['exit_code']}))


if __name__=='__main__':main()
