import json,ast,hashlib
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.tokenization import generation_report
raw=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text());r=raw['results']['tasks']['style_transfer'];tok=ByteTokenizer()
for n,t in [('teacher',r['teacher_test'])]+[(n,v['test']) for n,v in r['runs'].items() if 'packed' not in n]:
 s=next(v for v in t['generated_samples'] if v['question']=='style=json; 2+2=?');val=json.loads(s['generated'])['answer'];print(n,'generated',s['generated'],'int',type(val) is int,'correct',val==4,'EOS',s['generated_ids'][-1]==2)
for f in ['tiny_perceptron/data.py','tiny_perceptron/tokenization.py']:
 h=hashlib.sha256(Path(f).read_bytes()).hexdigest();print('hashmatches',f,h==raw['code_sha256'][f]);assert h==raw['code_sha256'][f]
p=Path('docs/technical-reviews/artifacts/p7_technical_e/sources/behavior-5af615e.py');tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_style_metrics');ns={'ByteTokenizer':ByteTokenizer,'generation_report':generation_report,'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
for text in ['{"answer": true}','{"answer": 3}','{"answer": 4}']:
 row={'a':2,'b':2,'style':'json','family':'2+2'};sample={'generated':text,'generated_ids':tok.encode(text)+[2],'exact':text=='{"answer": 4}'};result=ns['_style_metrics']({'samples':[sample]},[row]);print('rubric',text,result['rubric'])
