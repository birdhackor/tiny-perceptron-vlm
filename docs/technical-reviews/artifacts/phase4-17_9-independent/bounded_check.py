"""Fresh 17.9 reviewer: small CPU conversion + original raw-record arithmetic only."""
import ast
import copy
import hashlib
import inspect
import json
import os
import platform
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers, unpack_int4
from tiny_perceptron.training import load_checkpoint

BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()

def emit(name, value):
    print(json.dumps({'check': name, 'result': value}, ensure_ascii=False))

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

emit('environment', {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'torch_git': str(torch.version.git_version), 'device': 'cpu',
                     'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})

torch.manual_seed(0)
original = TinyLM(ModelConfig(width=8, tied=False)).eval()
ids = torch.tensor([[1, 2, 3]])
initial = state_sha(original)
with torch.no_grad():
    before = original(ids)['logits']
    for bits in (4, 8):
        independent = copy.deepcopy(original)
        assert all(dict(independent.named_parameters())[n].data_ptr() != p.data_ptr()
                   for n, p in original.named_parameters())
        candidate = replace_linear_layers(independent, bits)
        assert candidate is independent
        after = candidate(ids)['logits']
        repeated = original(ids)['logits']
        assert tuple(after.shape) == (1, 3, 264)
        assert torch.equal(before, repeated) and state_sha(original) == initial
        assert not before.requires_grad and not after.requires_grad
        assert all(p.grad is None for p in original.parameters())
        qlayers = [(n, m) for n, m in candidate.named_modules() if isinstance(m, QuantizedLinear)]
        for name, quant in qlayers:
            ref = original.get_submodule(name)
            expected_scale = ref.weight.abs().amax(-1, keepdim=True).clamp(min=1e-8) / (2**(bits-1)-1)
            assert torch.equal(quant.scale, expected_scale)
            values = unpack_int4(quant.values, quant.shape) if bits == 4 else quant.values
            assert tuple(values.shape) == tuple(ref.weight.shape)
            assert int(values.min()) >= -(2**(bits-1)-1) and int(values.max()) <= 2**(bits-1)-1
        for name, module in candidate.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
                ref = original.get_submodule(name)
                assert all(torch.equal(dict(ref.named_parameters())[k], v)
                           for k, v in module.named_parameters())
        emit('conversion_'+str(bits), {'output_shape': list(after.shape), 'output_dtype': str(after.dtype),
            'mae_denominator': before.numel(), 'original_max_difference': float((before-repeated).abs().max()),
            'mae': float((before-after).abs().mean()), 'packed_linear_count': len(qlayers),
            'retained_parameter_count': sum(p.numel() for p in candidate.parameters()),
            'buffer_bytes': sum(t.numel()*t.element_size() for t in candidate.buffers()),
            'values_bytes': sum(m.values.numel()*m.values.element_size() for _,m in qlayers),
            'scale_bytes': sum(m.scale.numel()*m.scale.element_size() for _,m in qlayers),
            'bias_bytes': sum(m.bias.numel()*m.bias.element_size() for _,m in qlayers if m.bias is not None),
            'original_state_sha256': initial})

torch.manual_seed(0)
tied = TinyLM(ModelConfig(width=8, tied=True)).eval()
assert tied.output.weight is tied.embedding.weight
tied_copy = copy.deepcopy(tied)
assert tied_copy.output.weight is tied_copy.embedding.weight
saved_embedding = tied_copy.embedding.weight.detach().clone()
replace_linear_layers(tied_copy, 4)
assert isinstance(tied_copy.output, QuantizedLinear)
assert torch.equal(tied_copy.embedding.weight, saved_embedding)
emit('tied_relation', {'before_same_parameter': True, 'deepcopy_preserves_internal_sharing': True,
                      'after_output_is_buffer_layer': True, 'embedding_unchanged': True})

original_code = BASE / 'original-scripts--course_experiments--compression.py'
tree = ast.parse(original_code.read_bytes())
names = {'_packed', '_chat', '_example', '_examples'}
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
namespace = dict(torch=torch, copy=copy, Path=Path, asdict=asdict, ByteTokenizer=ByteTokenizer,
                 replace_linear_layers=replace_linear_layers, load_checkpoint=load_checkpoint,
                 render_chat=render_chat)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(original_code), 'exec'), namespace)
temp_files = []
with tempfile.TemporaryDirectory(prefix='phase4-17_9-small-') as tempdir:
    ctx = SimpleNamespace(output=tempdir, device='cpu')
    for bits in (4,8):
        converted = replace_linear_layers(copy.deepcopy(original).eval(), bits)
        reloaded, path = namespace['_packed'](ctx, original, 'bounded'+str(bits), bits, {'source': 'reviewer random width8 model'})
        with torch.no_grad():
            expected = converted(ids)['logits']
            observed = reloaded(ids)['logits']
        assert torch.equal(expected, observed)
        assert state_sha(original) == initial
        temp_files.append(path)
        emit('save_load_'+str(bits), {'exact_logits_equal': True, 'shape': list(observed.shape),
              'file_bytes': path.stat().st_size, 'ephemeral_file_sha256': sha(path.read_bytes()),
              'source_parameter_hash_unchanged': True})
assert all(not p.exists() for p in temp_files)
emit('ephemeral_cleanup', {'retained_neural_weight_files': 0})

raw = json.loads((BASE/'raw-quantization.json').read_bytes())
dataset = json.loads((BASE/'raw-sft-dataset.json').read_bytes())
assert sha((BASE/'raw-sft-dataset.json').read_bytes()) == raw['results']['data']['sha256']
counts = {k: len(v) for k,v in dataset.items()}
families = {k:set(r['family'] for r in rows) for k,rows in dataset.items()}
assert counts == {'train':45, 'validation':5, 'test':10}
assert {k:len(v) for k,v in families.items()} == {'train':9,'validation':1,'test':2}
assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
examples = namespace['_examples'](dataset['train'], 128)
target_counts = [int((y!=IGNORE).sum()) for _,y in examples]
rng = random.Random(raw['seed'])
plan = [[rng.randrange(len(examples)) for _ in range(16)] for _ in range(120)]
plan_hash = sha(json.dumps(plan).encode())
effective = sum(target_counts[i] for batch in plan for i in batch)
assert plan_hash == raw['results']['training']['batch_plan_sha256']
assert effective == raw['results']['training']['effective_supervised_tokens'] == 13610
emit('data_and_training_arithmetic', {'counts':counts, 'families':{k:len(v) for k,v in families.items()},
     'intersections':0, 'sample_draws':120*16, 'steps':120, 'batch_size':16,
     'learning_rate':raw['results']['training']['learning_rate'], 'seed':raw['seed'],
     'batch_plan_sha256':plan_hash, 'effective_answer_targets':effective,
     'includes_assistant_answer_bytes_and_eos':True, 'max_training_sequence_length':max(len(x) for x,_ in examples)})

tok = ByteTokenizer()
def checked_samples(samples, eos_key):
    hits=[]
    for s in samples:
        generated = s['generated_ids']
        ended = tok.eos_id in generated
        answer = generated[:generated.index(tok.eos_id)] if ended else generated
        hit = answer == tok.encode(s['expected'])
        assert s['exact'] == hit and s[eos_key] == ended
        hits.append(hit)
    return sum(hits)

for variant in ('fp32','packed4','packed8'):
    result = raw['results']['runs'][variant]['test']
    samples=result['generated_samples']
    total=checked_samples(samples,'ended_with_eos')
    assert len(samples)==result['examples']==10 and total==result['correct']==6
    assert result['exact_match']==total/len(samples)==0.6
    emit('raw_test_'+variant, {'raw_token_exact':total,'denominator':len(samples), 'exact_match':total/len(samples)})

sft=json.loads((BASE/'raw-sft.json').read_bytes())
result=sft['results']['after']['test']
hits=checked_samples(result['samples'],'eos')
assert hits==result['matches']==5 and len(result['samples'])==result['examples']==10
assert result['exact_match']==0.5
assert [(s['messages'][-1]['content'],s['expected']) for s in result['samples']] == [
       (s['question'],s['expected']) for s in raw['results']['runs']['fp32']['test']['generated_samples']]
emit('original_sft_and_updated_fp32', {'source_sft_correct':hits,'updated_fp32_correct':6,'same_test_denominator':10,
      'intervening_optimizer_updates':120,'training_initial_final_hashes_different':
      raw['results']['training']['initialization_sha256']!=raw['results']['training']['final_sha256']})

for name, obj in [('installed-eval.py',torch.nn.Module.eval),('installed-no_grad.py',torch.no_grad),
                  ('installed-deepcopy.py',copy.deepcopy)]:
    source=inspect.getsource(obj).encode()
    (BASE/name).write_bytes(source)
    emit('installed_api_snapshot',{'file':name,'sha256':sha(source),'origin':inspect.getsourcefile(obj)})
emit('completed', {'assertions_passed':True,'full_training_executed':False,'existing_model_reevaluation_executed':False})
