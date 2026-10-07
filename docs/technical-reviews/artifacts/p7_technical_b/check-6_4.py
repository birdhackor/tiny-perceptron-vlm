import inspect,json,platform,hashlib
from pathlib import Path
import torch
from tiny_perceptron.model import TinyLM,ModelConfig
root=Path(__file__).resolve().parents[4];base=Path(__file__).resolve().parent
m=json.loads((root/'docs/course-experiments/results/tokenizer.json').read_text())
rows=[]
for width,vocabs in [(32,[300,1000,3000]),(64,[300,1000,3000]),(32,[600])]:
 for vocab in vocabs:
  embedding=vocab*width;twotables=2*embedding
  actual=torch.nn.Embedding(vocab,width)
  assert actual.weight.numel()==embedding and actual.weight.element_size()==4
  rows.append({'V':vocab,'D':width,'embedding_parameters':embedding,'fp32_bytes':embedding*4,'unshared_input_output':twotables})
counts=[]
for name,vocab in [('byte256',264),('bpe512',512)]:
 r=m['results']['runs'][name];lengths=r['validation_text_token_lengths'];model=TinyLM(ModelConfig(width=32,layers=1,max_length=384,vocab_size=vocab))
 parameters=sum(p.numel() for p in model.parameters());assert parameters==r['parameters'];assert model.output.bias is None
 counts.append({'name':name,'vocab':r['vocab_size_including_8_specials'],'lengths':lengths,'documents':len(lengths),'plain_text_tokens':sum(lengths),'mean_tokens':sum(lengths)/len(lengths),'parameters':parameters,'embedding_parameters':model.embedding.weight.numel(),'output_parameters':model.output.weight.numel(),'unshared':model.output.weight is not model.embedding.weight})
assert [c['plain_text_tokens'] for c in counts]==[3894,2112]
assert [c['parameters'] for c in counts]==[41824,57696]
(base/'sources/torch-linear.py').write_text(inspect.getsource(torch.nn.Linear))
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','arithmetic':rows,'historical':counts,'parameter_increase':57696-41824,'prediction':{'width64_all_double':True,'V600_D32_double_first':True},'source_hash_matches':{p:hashlib.sha256((root/p).read_bytes()).hexdigest()==m['code_sha256'][p] for p in ['tiny_perceptron/model.py','scripts/course_experiments/text.py']}},ensure_ascii=False,indent=2))

