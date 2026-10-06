"""Source-only callback: disposable software fixtures, never course model/test data."""
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import ast, hashlib, json, platform, tempfile, os, fcntl, re

R=Path.cwd();A=R/'docs/technical-reviews/artifacts/p6-19.7/source-dependency-callback-20261006'
old=R/'docs/technical-reviews/artifacts/p6-19.7/inputs/scripts/selftrained/evaluate.py'
current=R/'scripts/selftrained/evaluate.py'
oldtree=ast.parse(old.read_text());newtree=ast.parse(current.read_text())
expected_current=old.read_text().replace('str(path.relative_to(root))','path.relative_to(root).as_posix()').replace('journal.receipt_path.open("rb")','journal.receipt_path.open("r+b")')
assert expected_current==current.read_text()
def named(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def ahash(n):return hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
def select(t,names,ns):
 module=ast.Module(body=[named(t,n) for n in names],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(current),'exec'),ns)
print('ENVIRONMENT',json.dumps({'python':platform.python_version(),'OS':platform.system(),'device':'CPU; no model loaded','Windows_native_runtime':'not available; PureWindowsPath software fixture only'}))
print('SOURCE_HASHES',hashlib.sha256(old.read_bytes()).hexdigest(),hashlib.sha256(current.read_bytes()).hexdigest())
for name in ('score_reply','Generator','summarize','normalize','numeric_reply_value','format_pass'):
 a=named(oldtree,name);b=named(newtree,name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 assert old.read_text().splitlines()[a.lineno-1:a.end_lineno]==current.read_text().splitlines()[b.lineno-1:b.end_lineno]
 print('CITED_DEPENDENCY_IDENTICAL',name,b.lineno,b.end_lineno,ahash(b))

# Run the actual new and old code_fingerprints bodies on a disposable POSIX tree.
data=ast.parse((R/'tiny_perceptron/selftrained/dataset.py').read_text())
with tempfile.TemporaryDirectory(prefix='p6-19.7-source-fixture-') as temp:
 root=Path(temp)
 contents={'tiny_perceptron/a.py':b'fixture A\n','tiny_perceptron/nested/b.py':b'fixture B\n',
 'scripts/selftrained/train.py':b'train fixture, not executable\n',
 'scripts/selftrained/evaluate.py':b'evaluate fixture, not executable\n',
 'scripts/selftrained/chat.py':b'chat fixture, not executable\n'}
 for path,content in contents.items():
  p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
 expected={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()}
 results=[]
 for t in (oldtree,newtree):
  ns={'Path':Path,'hashlib':hashlib,'__file__':str(root/'scripts/selftrained/evaluate.py')}
  exec(compile(ast.Module(body=[named(data,'file_sha256')],type_ignores=[]),'original-dataset-helper','exec'),ns)
  select(t,['code_fingerprints'],ns)
  result=ns['code_fingerprints']();assert result==expected;results.append(result)
 assert results[0]==results[1]
 print('POSIX_ACTUAL_FUNCTION_FIXTURE',json.dumps(results[1],sort_keys=True))

 # Run the actual function body with PureWindowsPath's separator semantics.
 class WinFixturePath(PureWindowsPath):
  def resolve(self):return self
  def rglob(self,pattern):
   assert self==WinFixturePath('C:/fixture/tiny_perceptron') and pattern=='*.py'
   return [WinFixturePath('C:/fixture/tiny_perceptron/a.py'),WinFixturePath('C:/fixture/tiny_perceptron/nested/b.py')]
 win_results=[]
 for t in (oldtree,newtree):
  ns={'Path':WinFixturePath,'__file__':'C:/fixture/scripts/selftrained/evaluate.py',
      'file_sha256':lambda p:hashlib.sha256(contents[p.relative_to(WinFixturePath('C:/fixture')).as_posix()]).hexdigest()}
  select(t,['code_fingerprints'],ns);win_results.append(ns['code_fingerprints']())
 assert 'tiny_perceptron\\a.py' in win_results[0]
 assert win_results[1]==expected
 assert all('\\' not in key for key in win_results[1])
 print('PURE_WINDOWS_PATH_FUNCTION_FIXTURE',json.dumps({'old_keys':list(win_results[0]),'new_keys':list(win_results[1]),'hash_values_preserved':sorted(win_results[0].values())==sorted(win_results[1].values())}))

 # Execute only the exact new main receipt open/fsync AST block, not main.
 main=named(newtree,'main')
 candidates=[n for n in ast.walk(main) if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call)
             and isinstance(n.items[0].context_expr.func,ast.Attribute)
             and ast.unparse(n.items[0].context_expr.func)=='journal.receipt_path.open']
 assert len(candidates)==1;block=candidates[0]
 assert ast.unparse(block.items[0].context_expr)=='journal.receipt_path.open(\'r+b\')'
 receipt=root/'receipt.json';before=b'{"fixture":true,"completed_count":0}\n';receipt.write_bytes(before)
 calls=[]
 def fsync_spy(fd):
  mode=fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE
  assert mode==os.O_RDWR
  os.fsync(fd);calls.append({'fd_access':'O_RDWR','real_os_fsync':'success'})
 ns={'journal':SimpleNamespace(receipt_path=receipt),'os':SimpleNamespace(fsync=fsync_spy)}
 exec(compile(ast.Module(body=[block],type_ignores=[]),str(current),'exec'),ns)
 assert len(calls)==1 and receipt.read_bytes()==before
 print('ACTUAL_RECEIPT_FSYNC_AST_BLOCK',block.lineno,block.end_lineno,json.dumps(calls),'receipt_bytes_unchanged',True)

 # Pure scoring fixtures verify the cited legality/correctness distinctions.
 ns={'re':re,'json':json};select(newtree,['normalize','numeric_reply_value','format_pass','score_reply'],ns)
 expected_call={'tool':'calculator','operation':'multiply','a':26,'b':16}
 record={'task':'tool_call','messages':[{'role':'assistant','content':json.dumps(expected_call)}],
         'supervision':{'expected_call':expected_call,'format':'one_sentence'}}
 cases=[('correct',expected_call,416,'結果是416。',True,True,True),
        ('legal_wrong_parameters',{**expected_call,'a':25},400,'結果是400。',False,True,False),
        ('wrong_final_text',expected_call,416,'結果是417。',True,False,False)]
 for name,call,result,text,params,final,roundtrip in cases:
  trace={'tool_call':call,'tool_result':{'tool':'calculator','ok':True,'result':result},'executed':True,'status':'executed'}
  score=ns['score_reply'](record,text,trace)
  assert score['tool_valid'] and score['tool_parameters']==params and score['tool_final']==final and score['tool_roundtrip']==roundtrip
  print('CURRENT_SCORER_SOFTWARE_FIXTURE',name,json.dumps({k:score[k] for k in ('tool_valid','tool_parameters','tool_executed','tool_final','tool_roundtrip')}))
print('RESULT PASS: exact two-source-line changes, cited methods byte/AST identical, POSIX and PureWindowsPath keys, writable receipt fsync, and synthetic scorer verified. No model/test inference, training, generation, or raw test reread.')
