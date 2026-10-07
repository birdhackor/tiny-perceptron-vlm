"""Run the displayed current fence verbatim against temporary CPU-only trainer stubs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'docs/review-tools'))
import phase7_review as review
display=review.display(review.read_json(R/'outputs/grouped-review/f6f5240dffca4aeca152b6afb8dca93d/state.json'),R)
assert (display['page_id'],display['unit_index'])==('natural-v4-training',13)
text=display['text']
assert text.startswith('```bash\n')and text.endswith('```\n')
fence=text[len('```bash\n'):-len('```\n')]
folder=R/'outputs/p7_technical_f/recheck04-resume-fixtures'
folder.mkdir(parents=True,exist_ok=False)
script=folder/'exact-fence.sh';script.write_text(fence)
p=subprocess.run(['bash','-n',str(script)],text=True,capture_output=True)
assert p.returncode==0
cases=[('n0',{'completed_steps':0},True,'1039,2077'),('n1038',{'completed_steps':1038},True,'1039,2077'),
 ('n1039',{'completed_steps':1039},True,'2077'),('n1040',{'completed_steps':1040},True,'2077'),
 ('n2076',{'completed_steps':2076},True,'2077'),('n2077',{'completed_steps':2077},False,None),
 ('negative',{'completed_steps':-1},False,None),('missing_file',None,False,None),
 ('missing_key',{},False,None),('malformed_json','{broken',False,None),('string_steps',{'completed_steps':'1039'},False,None)]
results=[]
for name,data,expected_marker,pending in cases:
    root=folder/name;root.mkdir()
    bin=root/'.venv-natural/bin';bin.mkdir(parents=True);(bin/'python').symlink_to(Path(sys.executable).resolve())
    stub=root/'scripts/natural_assistant.py';stub.parent.mkdir()
    stub.write_text('import json,sys\nfrom pathlib import Path\nassert sys.argv[1]=="train"\nPath("trainer-marker.json").write_text(json.dumps({"argv":sys.argv[1:],"scope":"temporary CPU stub only; no model/GPU/network"}))\nprint("CPU_TRAINER_STUB")\n')
    state=root/'outputs/natural-my-v4/train/adapter/training.json';state.parent.mkdir(parents=True)
    if data is not None:state.write_text(data if isinstance(data,str)else json.dumps(data))
    p=subprocess.run(['bash',str(script)],cwd=root,text=True,capture_output=True)
    marker=root/'trainer-marker.json';called=marker.exists();args=json.loads(marker.read_text())['argv']if called else []
    observed=args[args.index('--checkpoint-steps')+1]if called else None
    row=dict(case=name,training_json=data,bash_exit_code=p.returncode,trainer_marker=called,pending_checkpoint_steps=observed,
        stdout=p.stdout,stderr=p.stderr,exact_fence_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest()if called else None)
    results.append(row)
    assert called==expected_marker,(name,row)
    if called:assert observed==pending and p.returncode==0,(name,row)
    else:assert p.returncode!=0,(name,row)
print(json.dumps(dict(scope='Unmodified displayed current unit13 Bash fence; only trainer file and relative Python alias are temporary CPU stubs. No actual train/model/weights/GPU/network operation.',
    displayed_unit_sha256=display['unit_sha256'],fence_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    fixture_python=sys.executable,bash_version=subprocess.run(['bash','--version'],text=True,capture_output=True).stdout.splitlines()[0],
    bash_syntax_exit=0,cases=results),ensure_ascii=False))
