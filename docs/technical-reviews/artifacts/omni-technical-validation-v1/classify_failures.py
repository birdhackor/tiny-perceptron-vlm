"""保留五個精確node、實際錯誤，以及同環境隔離原版結果。"""
from pathlib import Path
import subprocess,os,sys,re,json,hashlib,datetime

root=Path(__file__).resolve().parents[4]
cache=Path('/workspace/work/reference/omni-technical-validation-v1');base=cache/'baseline-head'
rawpath=cache/'command-output/page-7.log';raw=rawpath.read_text();nodes=re.findall(r'(?m)^FAILED (\S+)',raw)
sections=list(re.finditer(r'(?m)^_+ (test[^\n]*?) _+$',raw));failures=[]
for node in nodes:
    name=node.split('::')[-1];m=next(x for x in sections if x.group(1)==name)
    end=next((x.start() for x in sections if x.start()>m.start()),len(raw));sec=raw[m.end():end]
    failures.append(dict(node_id=node,actual_errors=[x for x in sec.splitlines() if x.startswith('E ')]))
env={**os.environ,'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],'PYTHONPATH':str(base),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
argv=['pytest','-ra',*nodes];result=subprocess.run(argv,cwd=base,env=env,capture_output=True,text=True,timeout=45)
log=cache/'command-output/baseline-five-node-check.log';log.write_text(result.stdout+result.stderr)
comparisons=[]
for name in ['scripts/export_course.py','scripts/check_site.py','.github/workflows/windows-student-commands.yml','README.md','README_en.md','course/training.md','tests/test_reading_time.py','tests/test_windows_student_commands.py']:
    a=(root/name).read_bytes();b=(base/name).read_bytes()
    comparisons.append(dict(path=name,current_sha256=hashlib.sha256(a).hexdigest(),head_sha256=hashlib.sha256(b).hexdigest(),current_equals_head=a==b))
value=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(root)),environment=dict(python=sys.version,device='CPU',pytest='9.1.1'),current_worktree_failures=failures,current_full_pytest=dict(command='pytest -ra',returncode=1,passed=2432,failed=5,skipped=1,raw_path=str(rawpath),raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest()),baseline_full_command_reference='baseline-comparison.json',baseline_exact_nodes=dict(command=argv,returncode=result.returncode,output_path=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),output=log.read_text()),current_vs_head_inputs=comparisons,classification='The same full pytest command is green at isolated HEAD; all five named current failures also pass individually at that HEAD. They are integration failures in the current modified worktree relative to HEAD, not baseline-known failures. The current reviewer changed no lesson/implementation/checker. The baseline lacks some local natural assets and therefore has three extra asset skips; the five named tests are not among those skips.',gate_classification='All three exact python gate commands pass at isolated HEAD and fail at the current working tree at preserved baseline replay, whose first reported blocker is current implementation drift: scripts/check_site.py. Active accepted metadata has not been advanced by this reviewer. This is no claim that removing that first blocker would make the full new review graph pass.',publication_status='Current global release checks remain failed; no claim of repository-wide or publication pass.')
p=root/'docs/technical-reviews/artifacts/omni-technical-validation-v1/execution-failure-classification.json'
p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(result.stdout[-1400:]);print('errors recorded',len(failures))
