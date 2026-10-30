"""Evaluate the actual old/new three-replacement AST under drive/UNC/Unicode-space paths; compile only."""
import ast,hashlib,json,sys,warnings
from pathlib import Path,PureWindowsPath
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'current-source'))
from scripts.omni_revision.staged_execution import staged_bootstrap

def expression(path):
 tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='test_short_bootstrap_checks_archive_identity_and_paths')
 found=[x.value for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in x.targets) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='replace'];assert len(found)==1
 return compile(ast.Expression(found[0]),str(path),'eval')
old=expression(BASE/'current-source/tests/test_omni_mature_research.py');new=expression(BASE/'current-v2-source/tests/test_omni_mature_research.py');bootstrap=staged_bootstrap(b'explicit compile-only input fixture')
paths=[('drive',PureWindowsPath(r'C:\Users\reviewer\AppData\Local\Temp\pytest-run')),('drive_unicode_spaces',PureWindowsPath(r'D:\課程 空白\審閱 測試')),('UNC',PureWindowsPath(r'\\server\share\pytest-run')),('UNC_unicode_spaces',PureWindowsPath(r'\\server\共用 空間\課程 測試'))]
cases=[]
for name,path in paths:
 value=eval(new,{'code':bootstrap,'tmp_path':path});compiled=compile(value,'current-v2-test-'+name,'exec');tree=ast.parse(value);decoded=[x.args[0].value for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='Path' and x.args and isinstance(x.args[0],ast.Constant)];expected=[(path/n).as_posix() for n in ('input.zip','input.sha256','root')];assert decoded==expected,(decoded,expected)
 cases.append({'case':name,'windows_path':str(path),'compiled':True,'decoded_paths_exact':True,'fixture_code_sha256':hashlib.sha256(value.encode()).hexdigest()})
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 try:compile(eval(old,{'code':bootstrap,'tmp_path':paths[0][1]}),'old-test-drive-control','exec');old_rejected=False;old_error=None
 except SyntaxError as e:old_rejected=True;old_error=e.msg
assert old_rejected
result={'scope':'Actual test replacement expressions extracted via AST; simulated PureWindowsPath drive/UNC/Chinese/space cases. compile and exact decoded path comparison only, no Windows OS/IO, worker/model/provider/final execution.','old_original_counterexample_rejected':old_rejected,'old_error':old_error,'current_v2_controls':cases,'all_new_compile_and_preserve_paths':True,'production_bootstrap_unchanged':True,'real_neural_or_paid_or_final_calls':0}
p=BASE/'current-v2-bootstrap-windows-result.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
