from pathlib import Path
import datetime,json,os,socket,sys
import pytest,torch
root=Path('/workspace/work/omni-book-integrated-v1')
m=json.loads((root/'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-tools-neutral-inputs-v1.json').read_text())
files=[r['path'] for r in m['files'] if r['path'].startswith('tests/') and r['path'].endswith('.py') and r['path']!='tests/test_omni_training_runtime.py']
skips=['test_actual_fresh_training_roundtrip_and_source_revision_rejection','test_new_directory_resume_keeps_original_best_without_dev_improvement','test_codec_gradients_update_all_three_modules_and_eval_never_resets']
def forbidden_network(*args,**kwargs):
 raise RuntimeError('independent tool audit permits offline CPU fixtures only')
socket.create_connection=forbidden_network
socket.getaddrinfo=forbidden_network
socket.socket.connect=forbidden_network
torch.set_num_threads(1)
os.chdir(root)
args=['-q','-ra','--basetemp=/workspace/work/omni-p6-tools-independent-v1/pytest-tmp-v1','-o','cache_dir=/workspace/work/omni-p6-tools-independent-v1/pytest-cache-v1',*files,'-k',' and '.join('not '+s for s in skips)]
print(json.dumps({'reviewer_task':'/root/omni_p6_technical_tools','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'torch':torch.__version__,'device':'cpu','socket_network':'blocked','test_files':files,'excluded_files':['tests/test_omni_training_runtime.py'],'excluded_training_cases':skips,'pytest_args':args},ensure_ascii=False,indent=2),flush=True)
code=pytest.main(args)
print('ACTUAL_COMPLETED_AT',datetime.datetime.now(datetime.timezone.utc).isoformat(),'PYTEST_EXIT_CODE',code,flush=True)
raise SystemExit(code)
