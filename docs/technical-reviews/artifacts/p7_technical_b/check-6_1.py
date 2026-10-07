import hashlib,inspect,json,platform,sys
from pathlib import Path
import torch
from tiny_perceptron.data import ByteTokenizer
root=Path(__file__).resolve().parents[4]
raw_path=root/'docs/course-experiments/results/real_text.json'
m=json.loads(raw_path.read_text())
s=m['results']['runs']['chinese-poetry']['after']['validation']['samples'][7]
tok=ByteTokenizer();b=bytes(i-8 for i in s['generated_ids'] if i>=8)
rows=[]
for text in ['小小貓','cat','🙂','cat🙂','貓🙂']:
 rows.append({'text':text,'codepoints':len(text),'utf8_bytes':len(text.encode('utf-8')),'ids':tok.encode(text)})
assert [(r['codepoints'],r['utf8_bytes']) for r in rows]==[(3,9),(3,3),(1,4),(4,7),(2,7)]
assert len(s['prompt'].encode('utf-8'))==23 and len(s['generated_ids'])==32
assert tok.decode(s['generated_ids'])==s['generated']=='孟沙，天夜天天天天春�'
assert b[:-2].decode('utf-8')=='孟沙，天夜天天天天春' and b[-2:].hex()=='e5a4'
try:b.decode('utf-8');raise AssertionError('unexpected strict decode')
except UnicodeDecodeError as e:invalid={'start':e.start,'end':e.end,'reason':e.reason}
code={p:{'recorded':m['code_sha256'][p],'current':hashlib.sha256((root/p).read_bytes()).hexdigest()} for p in ['scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/data.py','tiny_perceptron/model.py']}
assert all(v['recorded']==v['current'] for v in code.values())
embedding_source=inspect.getsource(torch.nn.Embedding)
(root/'docs/technical-reviews/artifacts/p7_technical_b/sources/torch-embedding.py').write_text(embedding_source)
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','text_counts':rows,'attention_cells':[3*3,9*9],'byte_alphabet':len(bytes(range(256))),'embedding_shape':list(torch.nn.Embedding(264,8).weight.shape),'sample_locator':'results.runs.chinese-poetry.after.validation.samples[7]','prompt_bytes':len(s['prompt'].encode('utf-8')),'generated_ids_count':len(s['generated_ids']),'generated_utf8_bytes':b.hex(),'strict_decode_error':invalid,'replacement_decode':tok.decode(s['generated_ids']),'historical_environment':{k:m[k] for k in ['revision','device','torch_version','python_version','seed','gpu']},'raw_result_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'code_hash_match':code},ensure_ascii=False,indent=2))

