import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import tokenizers
from tokenizers import Tokenizer
from tiny_perceptron.data import ByteTokenizer, IGNORE, SPECIALS, shifted, pad_batch

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment = {'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'tokenizers':tokenizers.__version__,'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'platform':platform.platform(),'cwd':str(Path.cwd()),'scope':'Original fences, bounded slice/mask checks, saved original tokenizer/input re-encoding; no model weights, evaluation, training, tokenizer retraining, or dataset preparation.'}
(OUT/'probe.environment.json').write_text(json.dumps(environment,indent=2)+'\n')

code = (OUT/'original-run/fence-1.py').read_text()
variants = {}
for name, altered in [('original',code), ('context4_original_starts',code.replace('context = 3','context = 4')), ('context4_starts0_4',code.replace('context = 3','context = 4').replace('[0, 3, 6]','[0, 4]'))]:
 stdout = io.StringIO()
 error = None
 try:
  with contextlib.redirect_stdout(stdout): exec(compile(altered,name,'exec'),{})
 except Exception as exc:
  error = type(exc).__name__
 (OUT/f'variant-{name}.py').write_text(altered)
 variants[name] = {'stdout':stdout.getvalue(),'error':error}
assert variants['original']['error'] is None
assert variants['context4_original_starts']['error'] == 'AssertionError'
assert variants['context4_starts0_4']['error'] is None

namespace = {'torch':torch,'Tokenizer':Tokenizer,'json':json,'SPECIALS':SPECIALS,'shifted':shifted,'ByteTokenizer':ByteTokenizer}
for source, names in [('scripts/course_experiments/text.py', {'_BPE','_utf8_prefix'}), ('scripts/course_experiments/common.py', {'text_examples'})]:
 tree = ast.parse((OUT/'inputs/historical'/source).read_text())
 nodes = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
 assert {node.name for node in nodes} == names
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'inputs/historical'/source),'exec'),namespace)
text_examples = namespace['text_examples']
_BPE = namespace['_BPE']

pairs = []
windows = []
for start in (0,3,6):
 x,y = shifted(list(range(10))[start:start+4])
 pairs.extend(zip(x.tolist(),y.tolist()))
 windows.append({'start':start,'x':x.tolist(),'y':y.tolist(),'targets':len(y)})
assert pairs == list(zip(range(9),range(1,10)))
assert len(pairs) == len(set(pairs)) == 9
tail_results = {}
for length in range(5):
 try:
  x,y = shifted(list(range(length)))
  tail_results[str(length)]={'x':x.tolist(),'y':y.tolist(),'positions':len(y)}
  assert len(y) == length-1
 except ValueError as exc:
  assert length < 2
  tail_results[str(length)]={'error':str(exc),'available_adjacent_pairs':max(length-1,0)}

# A genuine overlap can duplicate a next-token target. Masks distinguish context-only positions.
first=shifted([0,1,2,3]); second=shifted([1,2,3,4])
duplicate_targets=sorted(set(first[1].tolist()) & set(second[1].tolist()))
assert duplicate_targets == [2,3]
masked_second = second[1].clone(); masked_second[:2] = IGNORE
assert masked_second.tolist() == [IGNORE,IGNORE,4]
batch_x,batch_y,valid = pad_batch([first,(second[0],masked_second)])
assert int((batch_y != IGNORE).sum()) == 4 and int(valid.sum()) == 6
short = shifted([7,8])
pad_x,pad_y,pad_valid = pad_batch([first,short])
assert pad_x.shape == pad_y.shape == pad_valid.shape == (2,3)
assert pad_x[1].tolist() == [7,0,0]
assert pad_y[1].tolist() == [8,IGNORE,IGNORE]
assert int((pad_y != IGNORE).sum()) == int(pad_valid.sum()) == 4

tok=ByteTokenizer()
doc_examples=text_examples([{'text':'ABC'},{'text':'DE'}],mode='text',max_length=2,tokenizer=tok)
doc_pairs=[p for x,y in doc_examples for p in zip(x.tolist(),y.tolist())]
expected_pairs=[]
for text in ('ABC','DE'):
 ids=[tok.bos_id]+tok.encode(text)+[tok.eos_id]
 expected_pairs.extend(zip(ids[:-1],ids[1:]))
assert doc_pairs == expected_pairs and len(doc_pairs) == 7
assert (tok.eos_id,tok.bos_id) not in doc_pairs
assert [y.tolist() for x,y in doc_examples] == [[73,74],[75,2],[76,77],[2]]

result=json.loads((OUT/'inputs/current/docs/course-experiments/results/tokenizer.json').read_bytes())
exp=OUT/'inputs/original-experiment'
bpe=_BPE(Tokenizer.from_file(str(exp/'tokenizer-bpe512.json')))
rows_by_split={name:[json.loads(line) for line in (exp/'data'/f'{name}.jsonl').read_text().splitlines()] for name in ('train','validation','test')}
families={name:{r['family'] for r in rows} for name,rows in rows_by_split.items()}
assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
input_checks={}
for name,rows in rows_by_split.items():
 assert len(rows) == result['results']['data'][name]['records']
 encoded={}
 for label,encoder in [('byte256',tok),('bpe512',bpe)]:
  examples=text_examples(rows,mode='text',max_length=384,tokenizer=encoder)
  lengths=[len(encoder.encode(r['text'])) for r in rows]
  assert len(examples)==len(rows)
  assert all(len(x)==len(y)==n+1 for (x,y),n in zip(examples,lengths,strict=True))
  assert all(encoder.decode(encoder.encode(r['text']))==r['text'] for r in rows)
  assert max(n+1 for n in lengths)<=384
  encoded[label]={'ordinary_tokens':sum(lengths),'max_ordinary_tokens':max(lengths),'max_input_positions_including_BOS':max(n+1 for n in lengths),'windows':len(examples),'effective_targets_including_EOS':sum(len(y) for x,y in examples)}
 raw_bytes=[len(r['text'].encode()) for r in rows]
 assert max(raw_bytes)<=256
 assert all(len(bpe.encode(r['text']))<=len(r['text'].encode()) for r in rows)
 input_checks[name]={'records':len(rows),'families':len(families[name]),'raw_utf8_bytes':sum(raw_bytes),'max_raw_utf8_bytes':max(raw_bytes),'encoders':encoded}
sampler=random.Random(result['seed'])
exposure=[sampler.choices(range(len(rows_by_split['train'])),k=8) for _ in range(400)]
exposure_bytes=sum(len(rows_by_split['train'][index]['text'].encode()) for batch in exposure for index in batch)
schedule_bytes=json.dumps(exposure,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
schedule_sha=hashlib.sha256(schedule_bytes).hexdigest()
assert schedule_sha==result['results']['raw_document_schedule_sha256']
assert exposure_bytes==664759
assert all(run['training_raw_utf8_bytes_exposed']==exposure_bytes for run in result['results']['runs'].values())

observations={'variants':variants,'c3_windows':windows,'unique_adjacent_pairs':len(set(pairs)),'tail_lengths':tail_results,'overlap_check':{'duplicate_targets_without_mask':duplicate_targets,'masked_second_y':masked_second.tolist(),'valid_attention_positions':int(valid.sum()),'effective_loss_targets':int((batch_y!=IGNORE).sum())},'padding':{'x':pad_x.tolist(),'y':pad_y.tolist(),'valid':pad_valid.tolist(),'effective_targets':int((pad_y!=IGNORE).sum())},'separate_documents_with_BOS_EOS':{'windows':[{'x':x.tolist(),'y':y.tolist()} for x,y in doc_examples],'targets':len(doc_pairs),'cross_document_pairs':0},'original_input_checks':input_checks,'original_schedule':{'steps':400,'documents_per_step':8,'document_draws':3200,'raw_utf8_bytes_exposed':exposure_bytes,'schedule_sha256':schedule_sha},'scope':'Original run/input audit and bounded independent CPU variations; no language-model training or evaluation.'}
(OUT/'probe.results.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observations,ensure_ascii=False,indent=2))
