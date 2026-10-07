import json,subprocess,ast,hashlib
j=json.load(open('docs/course-experiments/results/dpo.json'));rev=j['revision'];p='scripts/course_experiments/behavior.py'
r=subprocess.run(['git','show',rev+':'+p],text=True,capture_output=True);assert r.returncode==0
assert hashlib.sha256(r.stdout.encode()).hexdigest()==j['code_sha256'][p]
newtext=open(p).read();old=ast.parse(r.stdout);new=ast.parse(newtext)
print('historical revision',rev,'recorded hash matches',True)
for name in ('run_dpo','_dpo_train','_preference_parts'):
 get=lambda tree,text:ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
 same=get(old,r.stdout)==get(new,newtext);print(name,'unchanged',same);assert same
