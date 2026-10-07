import json,ast,subprocess,hashlib
from tiny_perceptron.data import render_chat
from scripts.course_experiments.text import _utf8_prefix
p='docs/course-experiments/results/dpo.json';j=json.load(open(p));q=j['results']['ultrafeedback_pilot']
counts={s:r['records'] for s,r in q['data'].items() if isinstance(r,dict)}
print('raw_result_sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
print('source_records',q['source_records'],'split_counts',counts,'sum',sum(counts.values()));assert q['source_records']==100 and counts=={'train':80,'validation':10,'test':10}
old=subprocess.check_output(['git','show',j['revision']+':scripts/course_experiments/behavior.py'],text=True)
assert hashlib.sha256(old.encode()).hexdigest()==j['code_sha256']['scripts/course_experiments/behavior.py']
current=open('scripts/course_experiments/behavior.py').read()
def get(text,name):return ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name))
print('historical_natural_pilot_identical',get(old,'_natural_dpo_pilot')==get(current,'_natural_dpo_pilot'));assert get(old,'_natural_dpo_pilot')==get(current,'_natural_dpo_pilot')
s='中'*41;out=_utf8_prefix(s,120);print('120-byte clip',len(out),len(out.encode()),out==s[:40]);assert out==s[:40]
s='a'+'中'*40;out=_utf8_prefix(s,120);print('partial-char clip',len(out.encode()),out==s[:40]);assert len(out.encode())<=120
x,y=render_chat([{'role':'user','content':'1+1=?'},{'role':'assistant','content':'4'}]);print('variation_4',x.tolist(),y.tolist());assert x[-1]==60 and y[-2]==60 and y[-1]==2
