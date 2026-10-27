"""1.14 獨立 CPU 審查：只執行原頁程式、練習及既有權重推論；不訓練、不連網。"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import re
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import BigramLM
from scripts.course_experiments.text import _simple_examples, _simple_sample
from tiny_perceptron.model import generate

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None
freeze = json.loads(Path('docs/course-review-20261027-tokenizer/freeze-01.json').read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == '1.14')
raw = Path(page['source']).read_bytes()
body = re.search(rb'(?ms)^## 1\.14 .*?(?=^## |\Z)', raw).group(0)
assert body == Path(page['snapshot']).read_bytes()
assert hashlib.sha256(body).hexdigest() == page['source_sha256']
code = re.search(r'```python\n(.*?)\n```', body.decode(), re.S).group(1)

def execute(source):
    namespace, captured = {}, io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, 'course/chapters/01.md#1.14:original-or-stated-exercise', 'exec'), namespace)
    return namespace, {'stdout': captured.getvalue().strip(), 'length': len(namespace['output']),
                       'ids': [namespace['chars'].index(x) for x in namespace['output']]}

ns, original = execute(code)
_, repeat = execute(code)
_, seed7 = execute(code.replace('manual_seed(42)', 'manual_seed(7)'))
_, short = execute(code.replace('range(12)', 'range(3)').replace('len(output) == 13', 'len(output) == 4'))
_, period_start = execute(code.replace('current = 0', 'current = 3'))
assert original == repeat
assert original['length'] == seed7['length'] == period_start['length'] == 13
assert short['length'] == 4
assert original['stdout'] != seed7['stdout']
assert isinstance(ns['current'], int)
assert ns['p'].dtype == torch.float32
assert ns['p'].shape == (4, 4)
assert torch.all(ns['p'] > 0)
assert torch.allclose(ns['p'].sum(-1), torch.ones(4), atol=1e-7, rtol=0)
# 正文沒有固定抽樣文字保證；額外觀察各步選用的是上一輪的ID那列。
trace=[]
g=torch.Generator().manual_seed(42)
current=0
for step in range(12):
    following=torch.multinomial(ns['p'][current],1,generator=g).item()
    trace.append({'step':step+1,'row':current,'selected':following,'weight':float(ns['p'][current,following])})
    current=following
assert [trace[0]['row']]+[x['selected'] for x in trace] == original['ids']

record_path=Path('docs/course-experiments/results/simple_models.json')
record=json.loads(record_path.read_text())
run=record['results']['runs']['bigram']
base=Path('outputs/course-experiments/course-v1/simple_models')
checkpoint=base/'bigram.pt'
expected=next(a['sha256'] for a in record['artifacts'] if a['path']=='bigram.pt')
checkpoint_matches_record = sha(checkpoint)==expected
local_record_path=base/'result.json'
local_record=json.loads(local_record_path.read_text())
assert sha(checkpoint)==next(a['sha256'] for a in local_record['artifacts'] if a['path']=='bigram.pt')
saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert saved['format_version']=='simple-v1' and saved['kind']=='bigram' and saved['context']==1
vocabulary=saved['vocabulary']
parts=split_documents(toy_documents(),42)
assert len(toy_documents())==12
assert {k:len(v) for k,v in parts.items()} == {'train':9,'validation':1,'test':2}
assert not(set(parts['train'])&set(parts['validation']) or set(parts['train'])&set(parts['test']) or set(parts['test'])&set(parts['validation']))
model=BigramLM(len(vocabulary)+2)
model.load_state_dict(saved['model'])
model.eval()
checks={}
with torch.no_grad():
    for split,documents in parts.items():
        rows=[{'text':text,'family':text} for text in documents]
        generated=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
        assert hashlib.sha256(generated).hexdigest()==record['results']['data'][split]['sha256']
        assert sha(base/'data'/f'{split}.jsonl')==record['results']['data'][split]['sha256']
        x,y=_simple_examples(documents,vocabulary,1)
        nll=float(F.cross_entropy(model(x),y))
        assert abs(nll-run['after_nll_same_post_update_time'][split])<1e-6
        checks[split]={'documents':len(documents),'effective_next_character_targets_including_0_end':int(y.numel()),'post_update_mean_nll':nll}
    samples=[_simple_sample(model,vocabulary,1,prompt,'cpu') for prompt in ('顏色=','顏色=紅；形狀=')]
    assert samples==run['samples']
    contexts=torch.tensor([[vocabulary[c] for c in '顏色='],[vocabulary[c] for c in '形狀=']])
    same_last=torch.equal(model(contexts)[0],model(contexts)[1])
    assert same_last
    torch.manual_seed(42)
    start_model=BigramLM(len(vocabulary)+2)
    before={}
    for split,documents in parts.items():
        x,y=_simple_examples(documents,vocabulary,1)
        before[split]=float(F.cross_entropy(start_model(x),y))
        assert abs(before[split]-run['before_nll'][split])<1e-6
# 實際生成工具遇EOS停止的最小CPU契約核對；不需訓練模型。
class AlwaysEOS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.config=SimpleNamespace(max_length=32)
    def forward(self,ids,cache=None):
        logits=torch.zeros((ids.shape[0],ids.shape[1],264))
        logits[...,2]=1
        return {'logits':logits,'cache':None}
eos_model=AlwaysEOS()
end=generate(eos_model,torch.tensor([[1,73]]),max_new_tokens=12)
assert end.tolist()==[[1,73,2]] and eos_model.training
from tiny_perceptron.data import ByteTokenizer, SPECIALS
assert ByteTokenizer.eos_id==2 and SPECIALS[2]=='<eos>'

result={
 'command':'CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/1-14/verify_cpu.py',
 'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'cuda_build':str(torch.version.cuda),'device':'cpu','threads':str(torch.get_num_threads())},
 'source':{'canonical_slice_sha256':hashlib.sha256(body).hexdigest(),'original_fence_sha256':hashlib.sha256(code.encode()).hexdigest(),'freeze_match':True,'has_svg':'.svg' in body.decode()},
 'manual_table':{'chars':ns['chars'],'shape':list(ns['p'].shape),'dtype':str(ns['p'].dtype),'row_sums':ns['p'].sum(-1).tolist(),'cat_to_look':float(ns['p'][0,1]),'cat_self':float(ns['p'][0,0]),'look_self':float(ns['p'][1,1])},
 'original':original,'repeat_seed42_matches':original==repeat,'seed7':seed7,'range3':short,'ordinary_period_start':period_start,'trace_seed42':trace,
 'fixed_experiment':{'cloud_record_sha256':sha(record_path),'cloud_record_before_nll':run['before_nll'],'cloud_record_after_nll':run['after_nll_same_post_update_time'],'current_code_matches_cloud_record':{p:sha(p)==record['code_sha256'][p] for p in ['tiny_perceptron/simple.py','tiny_perceptron/data.py','scripts/course_experiments/text.py']},'checkpoint_sha256':sha(checkpoint),'checkpoint_matches_cloud_record':checkpoint_matches_record,'cloud_record_checkpoint_sha256':expected,'local_record_path':str(local_record_path),'local_record_sha256':sha(local_record_path),'checkpoint_matches_local_record':True,'parts':checks,'untrained_seed42_mean_nll':before,'samples':samples,'same_last_equals_same_logits':same_last,'recorded_training_steps':run['steps'],'recorded_seed':record['seed'],'checkpoint_provenance':'existing local CPU rerun, not hash-identical cloud checkpoint','recorded_device':record['device'],'recorded_torch':record['torch_version']},
 'eos':{'byte_tokenizer_id':ByteTokenizer.eos_id,'generated_ids':end.tolist(),'new_tokens':end.shape[1]-2,'budget':12},
 'scope':'實際執行原始1.14與兩項練習；額外句號起始、同尾字logits、既有checkpoint推論及EOS停止。沒有重訓、下載或GPU執行；既有紀錄品質只核對這個seed/小資料。',
 'status':'all assertions passed'
}
print(json.dumps(result,ensure_ascii=False,indent=2))
