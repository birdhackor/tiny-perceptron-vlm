"""抽取真生產f-string AST，用PureWindows/POSIX path求值；不是native Windows執行。"""
import ast
import copy
import hashlib
import json
import platform
import sys
from datetime import UTC,datetime
from pathlib import Path,PurePosixPath,PureWindowsPath

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 start=datetime.now(UTC).isoformat()
 source=ROOT/'scripts/check_lesson_order.py';original=ART/'order-guard-source-snapshots/02fac372ddbf9765b2497d1292ce1b313810ee3997a7150b31367595ea633272.py'
 assert sha(original)=='02fac372ddbf9765b2497d1292ce1b313810ee3997a7150b31367595ea633272'
 old_tree=ast.parse(original.read_text());new_tree=ast.parse(source.read_text())
 class RemoveOnlyAsPosix(ast.NodeTransformer):
  def __init__(self):self.count=0
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and node.func.attr=='as_posix':
    assert not node.args and not node.keywords and isinstance(node.func.value,ast.Call) and isinstance(node.func.value.func,ast.Attribute) and node.func.value.func.attr=='relative_to'
    self.count+=1;return node.func.value
   return node
 transformer=RemoveOnlyAsPosix();normalized=transformer.visit(copy.deepcopy(new_tree))
 assert transformer.count==1 and ast.dump(normalized,include_attributes=False)==ast.dump(old_tree,include_attributes=False)
 def expression(tree):
  found=[node for node in ast.walk(tree) if isinstance(node,ast.JoinedStr) and any(isinstance(value,ast.FormattedValue) and 'relative_to' in ast.dump(value.value) for value in node.values)]
  assert len(found)==1
  return found[0]
 old_expr=expression(old_tree);new_expr=expression(new_tree)
 values={}
 for label,pathclass,base in [('windows',PureWindowsPath,'C:/audit-root'),('posix',PurePosixPath,'/audit-root')]:
  root=pathclass(base);path=root/'course/chapters/19.md';environment={'path':path,'root':root,'ids':['19.3','19.1','19.2']}
  values[label]={}
  for version,node in [('original',old_expr),('current',new_expr)]:
   values[label][version]=eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),'actual_production_fstring_'+version,'eval'),{'__builtins__':{'len':len}},environment)
 assert values['windows']['current']==values['posix']['current']==values['posix']['original']
 assert values['windows']['current'].startswith('course/chapters/19.md:')
 assert values['windows']['original'].startswith('course\\chapters\\19.md:')
 report={'reviewer_task':'/root/tech_20261016_review_tools','initial_context':'informed','current_context':'informed','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'checked_files':[{'path':'scripts/check_lesson_order.py','sha256':sha(source)}],'original_source_sha256':sha(original),'only_ast_change':'Add .as_posix() to the actual path.relative_to(root) in the rejection f-string; removing exactly this call restores identical original AST.','actual_production_fstring_ast':{'original':ast.unparse(old_expr),'current':ast.unparse(new_expr)},'evaluated_outputs':values,'all_expected':True,'limits':['Real production AST expression evaluated with pathlib.PureWindowsPath/PurePosixPath on Linux; not native Windows file traversal or a native Windows CI rerun.','No synthetic replacement of the production formatting expression; original snapshot and current script AST are parsed directly. No code/test/assertion/source modifications.']}
 p=ART/'order-windows-path-expression-results-02.json'
 with p.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'all_expected':True,'only_ast_delta':True,'windows_current':values['windows']['current'],'windows_original':values['windows']['original'],'sha256':sha(p)},ensure_ascii=False))


if __name__=='__main__':main()
