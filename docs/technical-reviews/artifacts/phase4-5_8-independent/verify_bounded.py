"""A short CPU check of 5.8, original-fence variations and existing result denominators.

This performs no training or data/model acquisition. JSON scores are audited,
not remeasured; the tiny deterministic model below is solely a control-flow probe.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextlib
import hashlib
import io
import json
import math
import random
import sys
import torch

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import ByteTokenizer, shifted, pad_batch, IGNORE
from tiny_perceptron.model import generate, masked_loss

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device('cpu')
environment={'python':sys.version, 'executable':sys.executable, 'torch':str(torch.__version__),
 'torch_git_version':str(torch.version.git_version), 'device':'cpu', 'cuda_build':str(torch.version.cuda),
 'cuda_available':str(torch.cuda.is_available()), 'threads':str(torch.get_num_threads()),
 'scope':'No training, no imported experiment runner, no network, no model load.'}
(OUT/'bounded-environment.json').write_text(json.dumps(environment,indent=2)+'\n')

original=(OUT/'original-run/fence-1.py').read_bytes()
def run_fence(code):
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code,'original 5.8 fence or explicit one-line variation','exec'),{})
    return output.getvalue()
baseline=run_fence(original)
assert baseline=='正確前文下 逐字正確率 0.75 整串相同 False\n自由生成 逐字正確率 0.5 整串相同 False\n'
variation=original.replace(b'teacher_forced = [1, 2, 0, 4]',b'teacher_forced = [1, 2, 3, 4]')
(OUT/'variation-third-correct.py').write_bytes(variation)
corrected=run_fence(variation)
assert corrected=='正確前文下 逐字正確率 1.0 整串相同 True\n自由生成 逐字正確率 0.5 整串相同 False\n'
length_checks=[]
for replacement in (b'[1, 2, 0]',b'[1, 2, 0, 0, 4]'):
    code=original.replace(b'generated = [1, 2, 0, 0]',b'generated = '+replacement)
    try:
        run_fence(code)
    except ValueError as error:
        length_checks.append({'prediction':replacement.decode(),'error':str(error)})
    else:
        raise AssertionError('strict zip did not reject unequal lengths')
assert [1,2,3,4] != [1,3,2,4] and [1,2,3,4] != [1,2,3]
assert sum([True,False,True])==2

probability_checks=[]
for p in (0.6,0.9):
    logits=torch.tensor([[[math.log(p),math.log(1-p)]]],dtype=torch.float64)
    loss=float(masked_loss(logits,torch.tensor([[0]])))
    assert int(logits.argmax(-1))==0
    assert math.isclose(loss,-math.log(p),rel_tol=0,abs_tol=1e-12)
    probability_checks.append({'p_truth':p, 'top1_id':int(logits.argmax(-1)), 'nll':loss})
assert probability_checks[1]['nll'] < probability_checks[0]['nll']

class ConditionalProbe(torch.nn.Module):
    """Specified conditional choices, no learned parameters or quality conclusion."""
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(max_length=8)
        self.inputs=[]
    def forward(self,ids,cache=None):
        self.inputs.append(ids.tolist()[0])
        next_id={1:1,2:2,3:0}.get(ids.shape[1],4 if ids[0,-1].item()==3 else 0)
        logits=torch.full((1,ids.shape[1],5),-100.)
        logits[0,-1,next_id]=100.
        return {'logits':logits,'cache':None}
probe=ConditionalProbe()
free=generate(probe,torch.tensor([[4]]),max_new_tokens=4,eos_id=-1)[0,1:].tolist()
assert free==[1,2,0,0] and probe.inputs[-1]==[4,1,2,0]
forced=ConditionalProbe()(torch.tensor([[4,1,2,3]]))['logits'][0,-1].argmax().item()
assert forced==4

def historical_functions(experiment,names,relative,globals_dict):
    path=OUT/f'inputs/historical/{experiment}'/relative
    tree=ast.parse(path.read_bytes())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    assert len(selected)==len(names)
    namespace=dict(globals_dict)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

common_globals={'random':random,'json':json,'hashlib':hashlib,'ByteTokenizer':ByteTokenizer,
 'shifted':shifted,'torch':torch,'IGNORE':IGNORE}
contract_comparison={}
for experiment in ('text_foundation','real_text'):
    comparison={}
    for relative,names in {
        'tiny_perceptron/data.py':{'ByteTokenizer','shifted','pad_batch'},
        'tiny_perceptron/model.py':{'loss_sum','masked_loss','generate'},
        'scripts/course_experiments/common.py':{'split_records','text_examples','_nll','evaluate_lm'},
    }.items():
        def nodes(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
        a=nodes(OUT/f'inputs/historical/{experiment}'/relative)
        b=nodes(OUT/'inputs/current'/relative)
        assert set(a)==set(b)==names
        for name in sorted(names):
            same=a[name]==b[name]
            if same:
                comparison[relative+':'+name]='AST-identical to recorded experiment revision'
            else:
                assert (experiment,relative,name)==('text_foundation','scripts/course_experiments/common.py','evaluate_lm')
                comparison[relative+':'+name]='Historical text_foundation slices 24 byte IDs; current uses character-safe 24-byte prefix. All audited toy prompts are ASCII, giving identical prefixes; original source read and used below.'
    contract_comparison[experiment]=comparison
metrics={}
for experiment in ('text_foundation','real_text'):
    result_path=OUT/f'inputs/current/docs/course-experiments/results/{experiment}.json'
    outer=json.loads(result_path.read_bytes())
    common=historical_functions(experiment,{'split_records','text_examples'},'scripts/course_experiments/common.py',common_globals)
    if experiment=='text_foundation':
        # Independently specify exactly the 3 x 2 x 2 source combinations.
        rows=[{'text':f'color={color};shape={shape};side={side}.','family':f'color={color};shape={shape};side={side}.',
               'source':'course-generated','license':'MIT'} for color in ('red','green','blue')
              for shape in ('circle','square') for side in ('left','right')]
        run=outer['results']
    else:
        rows=[json.loads(line) for line in (OUT/'inputs/current/data/training/text-initial/tinystories-train-512.jsonl').read_text().splitlines()]
        dedup=historical_functions(experiment,{'_deduplicate_text'},'scripts/course_experiments/text.py',{'hashlib':hashlib})
        rows=dedup['_deduplicate_text'](rows)
        run=outer['results']['runs']['tinystories']
    parts=common['split_records'](rows,seed=outer['seed'])
    counts={}
    for split,records in parts.items():
        raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records).encode()
        observed_sha=hashlib.sha256(raw).hexdigest()
        assert observed_sha==run['data'][split]['sha256'],(experiment,split,observed_sha)
        dest=OUT/f'reconstructed-splits/{experiment}/{split}.jsonl'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
        examples=common['text_examples'](records,'text',128)
        tokens=sum(int((y != IGNORE).sum()) for x,y in examples)
        byte_count=sum(len(row['text'].encode('utf8')) for row in records)
        assert tokens==byte_count+len(records)
        assert len(records)==run['data'][split]['records']
        counts[split]={'records':len(records),'windows':len(examples),'effective_tokens':tokens,
                       'raw_utf8_bytes':byte_count,'sha256':observed_sha,'boundary_targets':len(records)}
        if split=='train':
            continue
        for stage in ('before','after'):
            report=run[stage][split]
            assert report['records']==len(records) and report['examples']==len(examples)
            assert report['effective_tokens']==tokens and report['raw_utf8_bytes']==byte_count
            assert math.isclose(report['nll_sum']/tokens,report['nll'],rel_tol=0,abs_tol=1e-12)
            assert len(report['samples'])==min(8,len(records))
            for row,sample in zip(records[:8],report['samples'],strict=True):
                if experiment=='text_foundation':
                    prefix=ByteTokenizer().decode(ByteTokenizer().encode(row['text'])[:24])
                    assert row['text'].isascii()
                else:
                    prefix=''
                    for character in row['text']:
                        if len((prefix+character).encode())>24:
                            break
                        prefix+=character
                assert sample['prompt']==prefix
                assert ByteTokenizer().decode(sample['generated_ids'])==sample['generated']
                assert len(sample['generated_ids'])<=32
    metrics[experiment]={'revision':outer['revision'],'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
       'gpu_run_environment':{k:outer[k] for k in ('seed','device','torch_version','python_version','gpu','evidence_status')},
       'splits':counts,'validation_before':run['before']['validation']['nll'],
       'validation_after':run['after']['validation']['nll'],'steps':run['training']['steps'],
       'samples_per_held_out_side':{s:len(run['after'][s]['samples']) for s in ('validation','test')}}
    if experiment=='text_foundation':
        row=parts['validation'][0]
        sample=run['after']['validation']['samples'][0]
        assert row['text']=='color=blue;shape=circle;side=left.'
        assert sample['prompt']=='color=blue;shape=circle;' and sample['generated']=='side=right.'
        assert any(r['text']=='color=blue;shape=circle;side=right.' for r in rows)
        assert round(run['before']['validation']['nll'],5)==5.73454
        assert round(run['after']['validation']['nll'],5)==0.96282
        metrics[experiment]['quoted_sample']={'full_reference':row['text'],**sample,
           'reference_suffix':row['text'][len(sample['prompt']):],'ambiguity':'Both sides occur for the same prefix in the generated source pool.'}
    else:
        assert counts['train']['records']==409 and counts['validation']['records']==51
        assert counts['validation']['effective_tokens']==42453
        assert round(run['before']['validation']['nll'],5)==5.76091
        assert round(run['after']['validation']['nll'],5)==1.80083
        sample=run['after']['validation']['samples'][1]
        assert sample['prompt']=='Once upon a time there w'
        assert sample['generated']=='as a big was a she was a ber the'
        metrics[experiment]['quoted_sample']={'sample_index_zero_based':1,**sample,
          'visible_excerpt_note':'Original generated field equals the textbook text fence exactly, with no trailing space.'}

result={'environment':environment,'original_stdout':baseline,'third_correct_stdout':corrected,
 'unequal_length_checks':length_checks,'probability_example':probability_checks,
 'conditional_probe':{'generated':free,'actual_input_prefixes':probe.inputs,'correct_prefix_fourth_choice':forced,
   'scope':'Deterministic control-flow example, not a trained model or empirical error-rate comparison.'},
 'historical_current_contract_comparison':contract_comparison,
 'existing_result_audits':metrics,'figure_check':{'svg_references':0,'figure_render':'not_applicable'}}
(OUT/'bounded-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
