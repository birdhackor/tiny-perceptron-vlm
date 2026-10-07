import ast,copy,json,hashlib,random
from pathlib import Path
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.tokenization import ByteTokenizer
from scripts.course_experiments.common import split_records,text_examples
from scripts.course_experiments.text import arithmetic_records
p=Path('docs/technical-reviews/artifacts/p7_technical_b/sources/behavior-a7cdffec.py');tree=ast.parse(p.read_text());ns={'LoRALinear':LoRALinear,'json':json}
for name in ['_conversation','_style_record','_add_lora']:
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
m=TinyLM(ModelConfig(width=64,layers=2,heads=2));base=sum(p.numel() for p in m.parameters());layers=ns['_add_lora'](m);trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
assert base==141568 and trainable==9504 and len(layers)==11
out={'parameters':{'base':base,'adapter':trainable,'percentage':100*trainable/base},'modules':layers,'styles':{}}
r=json.load(open('docs/technical-reviews/artifacts/p7_technical_b/sources/lora-result.json'))['results'];parts=split_records(arithmetic_records(),seed=42);tok=ByteTokenizer()
for style in ['concise','vivid']:
 rows=[ns['_style_record'](x,style,False) for x in parts['train']];ex=text_examples(rows,mode='sft',max_length=128);sampler=random.Random(42)
 count=sum(sum(int(y!=-100) for y in e[1]) for _ in range(450) for e in sampler.choices(ex,k=16))
 assert count==r['runs'][style]['training']['effective_tokens']
 assert len(rows)==49
 vals=[]
 for name,e in [('lora',r['runs'][style]['evaluation']['validation'])]+([('full',r['full_sft']['evaluation']['validation'])] if style=='vivid' else []):
  correct=0;samples=[]
  for s in e['samples']:
   assert tok.decode(s['generated_ids'])==s['generated'];g=s['generated'];v='像把兩組積木合在一起再數。' in g if style=='vivid' else g.isdigit();correct+=v;assert v==s['style_correct'];samples.append({'question':s['messages'][0]['content'],'generated':g,'style_correct':v})
  assert correct==e['rubric'][style]['style_correct'];vals.append({'method':name,'style_correct':correct,'records':len(e['samples']),'samples':samples})
 out['styles'][style]={'records':49,'steps':450,'targets':count,'validation':vals}
assert r['full_sft']['training']['effective_tokens']==out['styles']['vivid']['targets'] and r['full_sft']['training']['records']==49 and r['full_sft']['training']['steps']==450
print(json.dumps(out,ensure_ascii=False,indent=2))
