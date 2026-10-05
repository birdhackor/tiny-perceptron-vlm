"""Bounded CPU verification; audit original results, never train a model."""
import ast
import hashlib
import json
import random
import sys
import tarfile
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = Path(__file__).parent
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, CharTokenizer
from tiny_perceptron.attention import manual_attention

torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None
digest = lambda b: hashlib.sha256(b).hexdigest()
tok = ByteTokenizer()
rows = []
for text in ['小小貓','cat','🙂','cat🙂','貓🙂','e\u0301','👨\u200d👩\u200d👧\u200d👦','未見字🦊']:
    raw=text.encode('utf-8'); ids=tok.encode(text)
    assert tok.decode(ids)==text
    rows.append({'text':text,'codepoints':len(text),'utf8_bytes':len(raw),
                 'codepoint_values':[f'U+{ord(c):04X}' for c in text],
                 'raw_bytes':list(raw),'byte_token_count':len(ids),'roundtrip':True})
attention=[]
for length in [3,9]:
    q=torch.zeros(1,1,length,4)
    _,weights=manual_attention(q,q,q,torch.ones(1,1,length,length,dtype=torch.bool))
    assert weights.shape==(1,1,length,length)
    attention.append({'T':length,'batch':1,'heads':1,'dense_grid_shape':list(weights.shape),
                      'table_entries':weights.numel(),'row_sums':weights.sum(-1).tolist()})
embedding=torch.nn.Embedding(256,32)
assert embedding.weight.numel()==256*32
char=CharTokenizer('cat')
char_unseen={'input':'貓','ids':char.encode('貓'),'decoded':char.decode(char.encode('貓')),
             'equal_original':char.decode(char.encode('貓'))=='貓'}

original=json.loads((OUT/'inputs/docs/course-experiments/results/real_text.json').read_text())
archive=ROOT/'assets/training/chinese-poetry-v1.tar.gz'
archive_raw=archive.read_bytes()
asset=next(a for a in original['assets'] if a['id']=='chinese-poetry')
assert digest(archive_raw)==asset['archive_sha256']
with tarfile.open(archive,'r:gz') as tar:
    member=next(m for m in tar.getmembers() if m.name.endswith('chinese-classical-train-365.jsonl'))
    raw=tar.extractfile(member).read()
file_info=next(f for f in asset['files'] if f['path'].endswith('chinese-classical-train-365.jsonl'))
assert digest(raw)==file_info['sha256']
corpus=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
# Execute just the original historical pure functions; no training entrypoint/import.
namespace={'json':json,'hashlib':hashlib,'random':random}
function_inputs=[]
for filename,function in [('scripts/course_experiments/text.py','_deduplicate_text'),
                          ('scripts/course_experiments/common.py','split_records')]:
    path=OUT/'historical-code'/filename
    tree=ast.parse(path.read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    function_inputs.append({'file':filename,'sha256':digest(path.read_bytes()),
                            'function':function,'line_start':node.lineno,'line_end':node.end_lineno})
parts=namespace['split_records'](namespace['_deduplicate_text'](corpus),seed=42)
run=original['results']['runs']['chinese-poetry']
split_proof={}
for name,records in parts.items():
    data=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records).encode('utf-8')
    expected=run['data'][name]
    assert digest(data)==expected['sha256'],name
    split_proof[name]={'records':len(records),'sha256':digest(data),'matches_result':True}
sample=run['after']['validation']['samples'][7]
record=parts['validation'][7]
assert record['text'].startswith(sample['prompt'])
prefix=''
for c in record['text']:
    candidate=prefix+c
    if len(tok.encode(candidate))>24:break
    prefix=candidate
assert prefix==sample['prompt']
ids=sample['generated_ids']; raw=bytes(i-8 for i in ids)
assert all(8<=i<264 for i in ids) and len(ids)==32 and 2 not in ids
assert tok.decode(ids)==sample['generated']=='孟沙，天夜天天天天春�'
try:
    raw.decode('utf-8',errors='strict')
    raise AssertionError('Expected incomplete final code point')
except UnicodeDecodeError as error:
    failure={'start':error.start,'end':error.end,'reason':error.reason}
assert failure=={'start':30,'end':32,'reason':'unexpected end of data'}
assert raw[-2:]==bytes([0xe5,0xa4])
(OUT/'inputs/poem-selected-original-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
poem={
 'original_result_sha256':digest((OUT/'inputs/docs/course-experiments/results/real_text.json').read_bytes()),
 'historical_revision':original['revision'],'source_archive':str(archive.relative_to(ROOT)),
 'archive_sha256':digest(archive_raw),'archive_member':member.name,'member_sha256':file_info['sha256'],
 'source_revision':record.get('source_revision'), 'split_proof':split_proof,'functions_executed':function_inputs,
 'split':'validation','validation_records':len(parts['validation']),'displayed_samples':8,
 'sample_zero_based_index':7,'prompt':prefix,'prompt_codepoints':len(prefix),
 'prompt_utf8_bytes':len(prefix.encode()),'next_codepoint':record['text'][len(prefix)],
 'next_codepoint_bytes':len(record['text'][len(prefix)].encode()),
 'prompt_limit':24,'generation_limit':32,'context_limit':128,
 'total_sequence_positions':1+len(prefix.encode())+len(ids),
 'generated_ids':ids,'generated_raw_bytes':list(raw),'generated_raw_hex':raw.hex(),
 'generated_display':tok.decode(ids),'complete_utf8_prefix':raw[:30].decode(),
 'complete_codepoints':len(raw[:30].decode()),'final_partial_byte_count':2,
 'strict_decode_error':failure,'contains_eos':False,
 'original_training_steps':run['training']['steps'],'original_training_records':run['training']['records'],
 'scope':'Reconstruct original archive/split/prompt and decode recorded IDs, not rerun model inference or training.'}
output={'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu',
                       'cuda_build':str(torch.version.cuda),'cwd':str(Path.cwd()),'threads':torch.get_num_threads()},
        'length_rows':rows,'attention':attention,'embedding_shape':list(embedding.weight.shape),
        'embedding_parameters':embedding.weight.numel(),'char_unseen_counterexample':char_unseen,'poem':poem}
(OUT/'probe.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(output,ensure_ascii=False,indent=2))
