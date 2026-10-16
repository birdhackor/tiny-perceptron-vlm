"""Read actual isolated commit and pinned LFS blob; no Modal/HF/remote operation."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
TREE=Path('/workspace/work/lesson-products-transport')
ART=Path(__file__).parent
REVISION='dddf58cde65cc5aeb6ea24b67fd07eefcc0199a3'
sys.path.insert(0,str(TREE))
from scripts.lesson_products import publish


def main():
 start=datetime.now(UTC).isoformat()
 assert publish.ROOT==TREE
 spec,hashes=publish.inputs(REVISION,'docs/lesson-products/jobs/release-v1.json')
 assert len(hashes)==10 and len(spec['files'])==32
 assert spec['release_id']==414676375 and spec['asset_id']==646576897
 assert spec['release_commit']=='821a2c717f05ad7b57e08e88cd965e92ff0c1200'
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip();assert head==REVISION
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=TREE,text=True).strip()
 asset_path='assets/lesson-products/'+spec['asset_name']
 pointer=subprocess.check_output(['git','show',f'{REVISION}:{asset_path}'],cwd=TREE)
 expected=f"version https://git-lfs.github.com/spec/v1\noid sha256:{spec['transport_sha256']}\nsize {spec['asset_bytes']}\n".encode()
 assert pointer==expected and len(pointer)==len(expected)
 prior_pointer=subprocess.check_output(['git','show',f'c10a86256d4b936c7c9833d33b8a2f3c87f36d28:{asset_path}'],cwd=TREE)
 assert prior_pointer==pointer
 report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed actual-input engineering check','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'source_worktree':str(TREE),'source_branch':branch,'actual_head':head,'command':'scripts.lesson_products.publish.inputs(dddf58cde65cc5aeb6ea24b67fd07eefcc0199a3, docs/lesson-products/jobs/release-v1.json)','actual_inputs_accepted':True,'checked_files':[{'path':str(TREE/name),'repo_relative_path':name,'sha256':digest} for name,digest in hashes.items()],'spec_sha256':hashes['docs/lesson-products/jobs/release-v1.json'],'spec_file_count':len(spec['files']),'actual_head_lfs_pointer':{'commit':REVISION,'prior_commit':'c10a86256d4b936c7c9833d33b8a2f3c87f36d28','path':asset_path,'bytes':len(pointer),'sha256':hashlib.sha256(pointer).hexdigest(),'raw_pointer':pointer.decode(),'both_commit_blobs_identical':True},'all_expected':True,'limits':['inputs is actual current committed tree validation, not mocked Git. No publish main, registration, reserve, Modal/HF API, GPU, paid operation or remote write.','No repeat real GitHub upload/download; Actions roundtrip receipt separately attributed.']}
 path=ART/'engineering-publication-committed-inputs-results-original.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'accepted':True,'head':head,'source_files':len(hashes),'lfs_pointer_bytes':len(pointer),'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
