"""Bounded evaluator source-dependency callback; no course model or evaluator run."""
from pathlib import Path, PureWindowsPath
import ast
import difflib
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
CALLBACK=BASE/'evaluator-dependency-callback'
CALLBACK.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def only_function(n,scope):
    exec(compile(ast.Module(body=[n],type_ignores=[]),'personally inspected evaluator dependency','exec'),scope)
    return scope[n.name]

report_path=ROOT/'docs/technical-reviews/19.5.json'
prior_bytes=report_path.read_bytes()
prior=json.loads(prior_bytes)
assert prior['reviewer_task']=='/root/p6_fact_19_5' and prior['verdict']=='pass'
saved=CALLBACK/'prior-report.json'
if saved.exists():assert saved.read_bytes()==prior_bytes
else:saved.write_bytes(prior_bytes)
old_path=BASE/'frozen-source/scripts/selftrained/evaluate.py'
new_path=ROOT/'scripts/selftrained/evaluate.py'
old=old_path.read_text();new=new_path.read_text()
old_tree=ast.parse(old);new_tree=ast.parse(new)
old_line='names = [str(path.relative_to(root)) for path in sorted((root / "tiny_perceptron").rglob("*.py"))]'
new_line='names = [path.relative_to(root).as_posix() for path in sorted((root / "tiny_perceptron").rglob("*.py"))]'
old_mode='with journal.receipt_path.open("rb") as handle:'
new_mode='with journal.receipt_path.open("r+b") as handle:'
assert old.count(old_line)==old.count(old_mode)==1
assert new==old.replace(old_line,new_line).replace(old_mode,new_mode)
(CALLBACK/'current-evaluate.py').write_bytes(new_path.read_bytes())
(CALLBACK/'evaluator-exact-diff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='prior-personally-reviewed-evaluator',tofile='current-evaluator',n=4)))
unchanged_contracts={}
for name in ['normalize','format_pass','score_reply','summarize']:
    a=function(old_tree,name);b=function(new_tree,name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    unchanged_contracts[name]={'old_lines':[a.lineno,a.end_lineno],'current_lines':[b.lineno,b.end_lineno],'AST_identical':True}
old_generator=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
new_generator=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
a=next(n for n in old_generator.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
b=next(n for n in new_generator.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
unchanged_contracts['Generator.generate']={'old_lines':[a.lineno,a.end_lineno],'current_lines':[b.lineno,b.end_lineno],'AST_identical':True}

artifact_checks=[]
for a in prior['artifacts']:
    assert sha(ROOT/a['path'])==a['sha256'],a['id']
    artifact_checks.append({'id':a['id'],'path':a['path'],'sha256':a['sha256']})
source_checks=[]
for s in prior['sources']:
    if s['kind']=='repository_code':
        actual=sha(ROOT/s['path'])
        if s['id']=='s_evaluate':
            assert sha(old_path)==s['sha256']
        else:assert actual==s['sha256'],s['id']
        source_checks.append({'id':s['id'],'current_sha256':actual,'previous_sha256':s['sha256'],'unchanged':actual==s['sha256']})
chapter=(ROOT/'course/chapters/19.md').read_bytes()
start=chapter.index(b'## 19.5 ');end=chapter.index(b'## 19.6 ',start)
current_section=chapter[start:end]
assert hashlib.sha256(current_section).hexdigest()==prior['source_sha256']
assert current_section==(BASE/'current-section.md').read_bytes()

with tempfile.TemporaryDirectory(prefix='p6-19-5-evaluator-fixture-') as task_dir:
    task_root=Path(task_dir)
    names=['tiny_perceptron/a.py','tiny_perceptron/nested/b.py','scripts/selftrained/train.py','scripts/selftrained/evaluate.py','scripts/selftrained/chat.py']
    for name in names:
        p=task_root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fixture-only '+name+'\n').encode())
    expected={name:sha(task_root/name) for name in sorted(names)}
    actual_fn=only_function(function(new_tree,'code_fingerprints'),{'Path':Path,'__file__':str(task_root/'scripts/selftrained/evaluate.py'),'file_sha256':sha})
    linux_map=actual_fn();assert linux_map==expected

    class FixtureWindowsPath(PureWindowsPath):
        def resolve(self):return self
        def rglob(self,pattern):
            assert pattern=='*.py' and self.as_posix()=='C:/fixture/tiny_perceptron'
            return [self/'a.py',self/'nested'/'b.py']
    win_root=FixtureWindowsPath('C:/fixture')
    def windows_fixture_sha(path):return sha(task_root/path.relative_to(win_root).as_posix())
    new_fn=only_function(function(new_tree,'code_fingerprints'),{'Path':FixtureWindowsPath,'__file__':'C:/fixture/scripts/selftrained/evaluate.py','file_sha256':windows_fixture_sha})
    new_windows_map=new_fn();assert new_windows_map==expected
    old_fn=only_function(function(old_tree,'code_fingerprints'),{'Path':FixtureWindowsPath,'__file__':'C:/fixture/scripts/selftrained/evaluate.py','file_sha256':windows_fixture_sha})
    old_windows_map=old_fn()
    assert any('\\' in k for k in old_windows_map)
    assert all('\\' not in k for k in new_windows_map)
    assert sorted(old_windows_map.values())==sorted(new_windows_map.values())

    old_main=function(old_tree,'main');new_main=function(new_tree,'main')
    def receipt_fsync_with(main):
        return next(n for n in ast.walk(main) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute) and i.context_expr.func.attr=='open' and isinstance(i.context_expr.func.value,ast.Attribute) and i.context_expr.func.value.attr=='receipt_path' for i in n.items))
    old_with=receipt_fsync_with(old_main);new_with=receipt_fsync_with(new_main)
    assert old_with.items[0].context_expr.args[0].value=='rb'
    assert new_with.items[0].context_expr.args[0].value=='r+b'
    assert ast.dump(old_with.body[0],include_attributes=False)==ast.dump(new_with.body[0],include_attributes=False)
    receipt=task_root/'disposable-receipt.json';before=b'{"fixture_only":true,"rows":0}\n';receipt.write_bytes(before)
    fstat_modes=[]
    def real_fsync(fileno):
        assert os.get_inheritable(fileno) is False
        os.fsync(fileno);fstat_modes.append(os.fstat(fileno).st_size)
    exec(compile(ast.Module(body=[new_with],type_ignores=[]),'current evaluator exact receipt fsync block','exec'),{'journal':SimpleNamespace(receipt_path=receipt),'os':SimpleNamespace(fsync=real_fsync)})
    assert fstat_modes==[len(before)] and receipt.read_bytes()==before
    # Isolate the actual branch on a fixture path to inspect open mode, not a Windows syscall claim.
    class CaptureOpen:
        def __init__(self):self.modes=[]
        def open(self,mode):
            self.modes.append(mode)
            return receipt.open(mode)
    captured=CaptureOpen()
    exec(compile(ast.Module(body=[new_with],type_ignores=[]),'current evaluator captured fsync block','exec'),{'journal':SimpleNamespace(receipt_path=captured),'os':os})
    assert captured.modes==['r+b'] and receipt.read_bytes()==before

    score_scope={'re':re}
    for name in ['normalize','format_pass','score_reply']:only_function(function(new_tree,name),score_scope)
    synthetic={'id':'software-fixture-only','task':'text','messages':[{'role':'user','content':'請用兩點回答App問題。'},{'role':'assistant','content':'1. 先重新啟動App。\n2. 再詢問官方客服。'}],'supervision':{'semantic_all':['App','客服'],'format':'two_points'}}
    trace={'tool_call':None,'status':'no_call'}
    exact=score_scope['score_reply'](synthetic,synthetic['messages'][-1]['content'],trace)
    wrong=score_scope['score_reply'](synthetic,'不知道。',trace)
    assert exact['semantic'] and exact['format']
    assert not wrong['semantic'] and not wrong['format']
    exhausted=score_scope['score_reply'](synthetic,'',dict(trace,generation_failure='context_budget_exhausted'))
    assert not exhausted['semantic'] and not exhausted['format']

payload={'reviewer_task':'/root/p6_fact_19_5','python':sys.version,'platform':platform.platform(),'device':'cpu; disposable filesystem/software fixtures only','prior_report_sha256':sha(saved),'current_section_sha256':prior['source_sha256'],'section_bytes_unchanged':True,'previous_evaluator_sha256':sha(old_path),'current_evaluator_sha256':sha(new_path),'exact_changes':['code_fingerprints: relative_to(root).as_posix() instead of str(...) at line313','main: receipt fsync open mode r+b instead of rb at line839'],'cited_claim':'c13','cited_contracts_AST_unchanged':unchanged_contracts,'POSIX_fixture_map':linux_map,'Windows_path_fixture_old_keys':list(old_windows_map),'Windows_path_fixture_current_keys':list(new_windows_map),'Windows_path_fixture_digests_preserved':True,'receipt_fixture_actual_current_block_fsync_succeeded':True,'receipt_fixture_bytes_preserved':True,'receipt_open_mode_captured':captured.modes,'synthetic_score_checks':{'correct':exact,'wrong':wrong,'context_exhausted':exhausted},'artifact_hash_checks':artifact_checks,'repository_source_hash_checks':source_checks,'new_model_generations':0,'training_steps_run':0,'heldout_model_or_record_scoring':0,'limitations':'Linux filesystem fsync executed; Windows path normalization exercised through PureWindowsPath with disposable fixture content. Windows OS syscall execution is not claimed. Existing c13 scoring/stratification/EOS contracts unchanged.','result':'Source dependency change does not invalidate c13 or other 19.5 claims; actual verification passed without any course model/evaluation/training invocation.'}
(CALLBACK/'output.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k not in ['artifact_hash_checks','repository_source_hash_checks','POSIX_fixture_map']},ensure_ascii=False,indent=2))
