"""Actual original-owner callback: current-source assertions and small canonical CPU unit probe.

No original fence is rerun: its bytes and all relevant original execution evidence
are verified unchanged. This executes only new byte/token/EOS boundary cases using
the real repository ByteTokenizer and generate, not any trained model.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import re
import sys
import torch

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=OUT.parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import generate

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device('cpu')
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def raw_section(path,identifier):
    raw=path.read_bytes()
    heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    i=next(i for i,h in enumerate(heads) if h[0].startswith(('## '+identifier+' ').encode()))
    return raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)]
identity=json.loads((OUT/'callback-identity.json').read_bytes())
prior_bytes=(ROOT/identity['prior_history_path']).read_bytes()
assert sha(prior_bytes)==identity['prior_history_sha256']
prior=json.loads(prior_bytes)
assert prior['verdict']=='pass' and prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_5_8'
assert prior['source_sha256']==sha((BASE/'original-run/section.md').read_bytes())
preserved=[]
for artifact in prior['artifacts']:
    raw=(ROOT/artifact['path']).read_bytes()
    assert sha(raw)==artifact['sha256'],artifact['id']
    preserved.append(artifact['id'])
old=(BASE/'original-run/section.md').read_bytes()
current=(OUT/'inputs/5.8.md').read_bytes()
assert current==raw_section(ROOT/'course/chapters/05.md','5.8')
old_clause='每次選最高分候選、往後寫最多32個byte，一筆實際結果是：'
new_clause='每次選最高分候選、最多新增32個token（這份模型把文字拆成byte，也就是位元組；本例的英文字母與空格各占一byte，結束符號另占一個token，拆分方法會在[6.1](06.md#6.1)展開），一筆實際結果是：'
assert old.count(old_clause.encode())==1
assert old.replace(old_clause.encode(),new_clause.encode())==current
old_fences=re.findall(rb'(?ms)^```python\r?\n(.*?)^```\s*$',old)
new_fences=re.findall(rb'(?ms)^```python\r?\n(.*?)^```\s*$',current)
assert len(old_fences)==len(new_fences)==1 and old_fences==new_fences
assert old_fences[0]==(BASE/'original-run/fence-1.py').read_bytes()
assert not re.findall(rb'!\[[^]]*\]\([^)]+\)',current)
dependencies={}
for identifier,path in [('6.1','course/chapters/06.md'),('T.4','course/training.md')]:
    raw=(OUT/f'inputs/{identifier}.md').read_bytes()
    assert raw==raw_section(ROOT/path,identifier)
    dependencies[identifier]={'source':path+'#'+identifier,'sha256':sha(raw)}

# Verify the exact used method contracts rather than assuming whole-file identity.
method_equivalence={}
methods={'tiny_perceptron/data.py':['ByteTokenizer'],
         'tiny_perceptron/model.py':['loss_sum','masked_loss','generate'],
         'scripts/course_experiments/common.py':['text_examples','_nll','evaluate_lm'],
         'scripts/course_experiments/text.py':['_evaluations','run_real_text']}
for path,names in methods.items():
    def functions(raw):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body
          if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    original=functions((BASE/'inputs/current'/path).read_bytes())
    new=functions((OUT/'inputs'/path).read_bytes())
    assert original==new,(path,'relevant method changed')
    assert (ROOT/path).read_bytes()==(OUT/'inputs'/path).read_bytes()
    method_equivalence[path]={'methods':names,'unchanged_ast':True,'current_file_sha256':sha((ROOT/path).read_bytes())}

tokenizer=ByteTokenizer()
encoding_examples={}
for text,expected_bytes in [('cat',3),('小小貓',9),('🙂',4),(' ',1)]:
    ids=tokenizer.encode(text)
    assert len(ids)==len(text.encode('utf-8'))==expected_bytes
    assert tokenizer.decode(ids)==text
    assert ids==[b+8 for b in text.encode('utf-8')]
    encoding_examples[text]={'unicode_codepoints':len(text),'utf8_bytes':len(ids),'content_ids':ids}
assert tokenizer.eos_id==2 and tokenizer.decode([2])==''

class BudgetProbe(torch.nn.Module):
    """Explicit scores only; a control-flow probe with no learned parameters."""
    def __init__(self,eos_at=None,max_length=128):
        super().__init__()
        self.config=SimpleNamespace(max_length=max_length)
        self.eos_at=eos_at
        self.calls=0
    def forward(self,ids,cache=None):
        self.calls+=1
        selected=2 if self.calls==self.eos_at else ord('a')+8
        scores=torch.full((1,ids.shape[1],264),-100.)
        scores[0,-1,selected]=100.
        return {'logits':scores,'cache':None}
probes={}
for name,eos_at,max_length,expected_count,expected_payload in [
    ('32-content-tokens',None,128,32,32),
    ('31-content-plus-EOS',32,128,32,31),
    ('immediate-EOS',1,128,1,0),
    ('context-limit',None,2,1,1),
]:
    model=BudgetProbe(eos_at,max_length)
    raw=generate(model,torch.tensor([[tokenizer.bos_id]]),max_new_tokens=32,eos_id=2)[0,1:].tolist()
    decoded=tokenizer.decode(raw)
    assert len(raw)==expected_count and len(decoded.encode())==expected_payload
    assert model.calls==expected_count and model.training
    probes[name]={'generated_ids':raw,'new_token_count':len(raw),'content_byte_count':len(decoded.encode()),
      'eos_count':raw.count(2),'forward_calls':model.calls,'training_flag_restored':model.training}

# Read only named raw metrics/provenance/sample fields, never attached author notes.
metrics={}
pointers=[]
old_audit=json.loads((BASE/'bounded-result.json').read_bytes())['existing_result_audits']
for experiment in ('text_foundation','real_text'):
    relative=f'docs/course-experiments/results/{experiment}.json'
    raw=(OUT/'inputs'/relative).read_bytes()
    assert raw==(ROOT/relative).read_bytes()
    assert sha(raw)==old_audit[experiment]['result_sha256']
    obj=json.loads(raw)
    run=obj['results'] if experiment=='text_foundation' else obj['results']['runs']['tinystories']
    prefix='/results' if experiment=='text_foundation' else '/results/runs/tinystories'
    read={'revision':obj['revision'],'seed':obj['seed'],
      'training_records':run['training']['records'],'steps':run['training']['steps'],
      'validation_records':run['data']['validation']['records']}
    pointers.extend([relative+'#/revision',relative+'#/seed',relative+prefix+'/training/records',
      relative+prefix+'/training/steps',relative+prefix+'/data/validation/records'])
    for stage in ('before','after'):
        values=run[stage]['validation']
        read[stage]={k:values[k] for k in ('nll','nll_sum','effective_tokens','raw_utf8_bytes','examples','records')}
        assert math.isclose(values['nll_sum']/values['effective_tokens'],values['nll'],abs_tol=1e-12,rel_tol=0)
        assert values['effective_tokens']==values['raw_utf8_bytes']+values['records']
        pointers.extend(relative+prefix+'/'+stage+'/validation/'+k for k in read[stage])
    index=0 if experiment=='text_foundation' else 1
    sample=run['after']['validation']['samples'][index]
    read['quoted_sample']={k:sample[k] for k in ('prompt','generated','generated_ids')}
    pointers.extend(relative+prefix+'/after/validation/samples/'+str(index)+'/'+k for k in read['quoted_sample'])
    assert tokenizer.decode(sample['generated_ids'])==sample['generated']
    assert len(sample['generated_ids'])<=32
    if experiment=='real_text':
        assert sample['generated']=='as a big was a she was a ber the'
        assert sample['generated'].isascii() and len(sample['generated'].encode())==32
        assert len(sample['generated_ids'])==32 and all(i>=8 for i in sample['generated_ids'])
        assert 2 not in sample['generated_ids']
        assert read['training_records']==409 and read['validation_records']==51
        assert read['after']['effective_tokens']==42453 and read['after']['examples']==358
        assert round(read['before']['nll'],5)==5.76091 and round(read['after']['nll'],5)==1.80083
    else:
        assert sample['generated']=='side=right.' and sample['generated_ids'][-1]==2
        assert read['validation_records']==1 and read['after']['effective_tokens']==35
        assert round(read['before']['nll'],5)==5.73454 and round(read['after']['nll'],5)==0.96282
    metrics[experiment]={'sha256':sha(raw),'pointers_read_only_raw_measurements':read}

environment={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),
 'torch_git_version':str(torch.version.git_version),'cuda_build':str(torch.version.cuda),
 'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads()),
 'scope':'Canonical ByteTokenizer/generate boundary probes; no model/data download, training or old fence rerun.'}
result={'source_sha256':sha(current),'figure_sha256':{},'dependency_raw_versions':dependencies,
 'prior_history':identity,'preserved_prior_artifact_ids':preserved,
 'old_fence_sha256':sha(old_fences[0]),'current_fence_sha256':sha(new_fences[0]),
 'proof_reuse_reason':'Only one explanatory generation-unit sentence changed. Original fence, used method ASTs and raw result JSONs are unchanged and original evidence hashes were verified.',
 'method_equivalence':method_equivalence,'encoding_examples':encoding_examples,
 'canonical_generate_boundary_probes':probes,'original_metrics_pointer_audit':metrics,
 'json_pointers_inspected':pointers,'environment':environment,'assertions':'All completed successfully'}
(OUT/'callback-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
