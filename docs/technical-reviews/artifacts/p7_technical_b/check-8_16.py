import ast,json
from pathlib import Path
p=Path('docs/technical-reviews/artifacts/p7_technical_b/sources/ifeval-instructions-e49bbfe.py');tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JsonFormat');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_following');ns={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
inputs=['{"text":["今日休館","明日開放"]}','```json\n{"text":["今日休館","明日開放"]}\n```','{"answer":4}','7','今日休館\n明日開放','{"text":["今日休館","明日開放"]}\n以下是中文']
out=[]
for value in inputs:
 official=ns['check_following'](None,value)
 try:o=json.loads(value);bare_schema=isinstance(o,dict) and set(o)=={'text'} and isinstance(o['text'],list) and all(isinstance(s,str) for s in o['text'])
 except ValueError:bare_schema=False
 out.append({'input':value,'official_JsonFormat':official,'task_bare_onlytext_list_schema':bare_schema})
assert [x['official_JsonFormat'] for x in out]==[True,True,True,True,False,False]
assert [x['task_bare_onlytext_list_schema'] for x in out]==[True,False,False,False,False,False]
# Original page labels are given paper conditions; this is aggregation, not model measurements.
given_flags=[[1,1,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
aggregate={'denominator_original_cases':5,'dimension_pass_counts':[sum(row[c] for row in given_flags) for c in range(4)],'joint_pass_count':sum(all(row) for row in given_flags),'forced_limit_case_retained':True}
assert aggregate['dimension_pass_counts']==[4,4,4,4] and aggregate['joint_pass_count']==1
print(json.dumps({'official_checker_executed':out,'given_condition_aggregate':aggregate,'English_target':{'text':['Closed today','Open tomorrow']},'old_Chinese_for_English_given_flags':[0,0,1,1]},ensure_ascii=False,indent=2))
