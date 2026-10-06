"""Independent, bounded CPU checks of chapter 19.10; no training or held-out inference."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]
sys.path.insert(0, str(REPO))
import torch
import safetensors
from safetensors.torch import load_file
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig, PaddingSafeMoE
from tiny_perceptron.selftrained.inference import verify_export, InferenceAssistant

torch.set_num_threads(1)
torch.manual_seed(1061910)
reads = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path, ptr):
    raw = path.read_bytes()
    value = json.loads(raw)
    for key in ptr.strip('/').split('/'):
        assert key not in ('notes', 'review') and not key.endswith('_scope_correction')
        value = value[int(key)] if isinstance(value, list) else value[key]
    reads.append({'path': str(path.relative_to(REPO)), 'sha256': hashlib.sha256(raw).hexdigest(), 'pointer': ptr})
    return value


section = (BASE / 'inputs/19.10.md').read_bytes()
fence = re.search(rb'```python\n(.*?)\n```', section, re.S).group(1)
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(fence, 'course/chapters/19.md#19.10:fence1', 'exec'), {})
models = []
for name, expected in [('moe-joint', 5447107), ('dense-joint', 2288067)]:
    local = Path('/tmp/p6-19.10-exports') / name
    snap = BASE / 'public' / name
    freeze = REPO / f"docs/selftrained/results/public-raw/{name.split('-')[0]}/freeze/frozen.json"
    safe_hash = pointer(freeze, '/safe_weights_sha256')
    manifest_hash = pointer(freeze, '/inference_manifest_sha256')
    assert digest(local / 'model.safetensors') == safe_hash
    assert digest(snap / 'inference-manifest.json') == manifest_hash
    manifest, verified_hash = verify_export(local, manifest_sha256=manifest_hash)
    cfg = json.loads((snap / 'model-config.json').read_bytes())
    header = json.loads((snap / 'model.safetensors.header.json').read_bytes())
    entries = {k:v for k,v in header.items() if k != '__metadata__'}
    assert {v['dtype'] for v in entries.values()} == {'F32'}
    stored_count = sum(math.prod(v['shape']) for v in entries.values())
    stored_bytes = sum(v['data_offsets'][1]-v['data_offsets'][0] for v in entries.values())
    state = load_file(local / 'model.safetensors', device='cpu')
    original_loader = InferenceAssistant(local, Path('/tmp/p6-19.10-empty-assets'), device='cpu', manifest_sha256=manifest_hash)
    model = original_loader.model
    expected_state = model.state_dict()
    assert set(state) == set(expected_state)
    assert all(state[n].shape == expected_state[n].shape and state[n].dtype == expected_state[n].dtype for n in state)
    assert all(torch.isfinite(v).all() for v in state.values())
    assert torch.equal(state['lm.embedding.weight'], state['lm.output.weight'])
    unique_count = sum(p.numel() for p in model.parameters())
    unique_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    alias_count = model.lm.embedding.weight.numel()
    assert unique_count == expected
    assert model.lm.embedding.weight is model.lm.output.weight
    assert model.lm.embedding.weight.data_ptr() != state['lm.embedding.weight'].data_ptr()
    assert state['lm.embedding.weight'].data_ptr() != state['lm.output.weight'].data_ptr()
    assert unique_bytes == expected * 4 and stored_count == unique_count + alias_count
    assert stored_bytes == stored_count * 4
    assert {str(p.dtype) for p in model.parameters()} == {'torch.float32'}
    pointers_before = {n:p.data_ptr() for n,p in model.named_parameters()}
    with torch.no_grad():
        out = model(torch.tensor([[1,10,50,100]], dtype=torch.long))
    assert pointers_before == {n:p.data_ptr() for n,p in model.named_parameters()}
    cache_bytes = sum(t.numel()*t.element_size() for kv in out['cache'] for t in kv)
    components = {n:sum(p.numel() for p in module.parameters()) for n,module in model.named_children()}
    routes = []
    if cfg['architecture'] == 'moe':
        expert_capacity = sum(p.numel() for p in model.lm.blocks[0].ffn.experts[0].parameters())
        router_capacity = sum(p.numel() for p in model.lm.blocks[0].ffn.router.parameters())
        assert expert_capacity == 262912 and router_capacity == 1024
        for layer,route in enumerate(out['routing']):
            assert route['chosen'].shape == (1,4,2)
            assert route['counts'].sum().item() == 8
            routes.append({'layer':layer, 'counts':route['counts'].tolist(), 'chosen':route['chosen'].tolist()})
    models.append({'name':name, 'weights_sha256':safe_hash, 'manifest_sha256':verified_hash,
                   'config':cfg, 'unique_parameters':unique_count, 'unique_FP32_bytes':unique_bytes,
                   'unique_MiB':unique_bytes/1024**2, 'rounded_MiB':round(unique_bytes/1024**2,2),
                   'logical_FP16_bytes':unique_count*2, 'logical_FP16_MiB':unique_count*2/1024**2,
                   'state_tensor_count':len(state), 'serialized_parameter_occurrences':stored_count,
                   'duplicate_tied_embedding_occurrences':alias_count, 'serialized_data_bytes':stored_bytes,
                   'safetensors_file_bytes':(local/'model.safetensors').stat().st_size,
                   'format_bytes':(local/'model.safetensors').stat().st_size-stored_bytes,
                   'components':components, 'dtype':'torch.float32', 'tied_after_load':True,
                   'original_InferenceAssistant_constructor_completed':True,
                   'state_and_live_model_have_distinct_storages':True,
                   'serialized_embedding_and_output_have_distinct_storages':True,
                   'all_weight_addresses_retained_after_forward':True, 'synthetic_forward_tokens':4,
                   'KV_cache_bytes_at_4_tokens':cache_bytes, 'routes':routes})
    if cfg['architecture'] == 'moe':
        models[-1]['FFN_one_expert_parameters_per_layer'] = expert_capacity
        models[-1]['FFN_resident_experts_per_layer'] = len(model.lm.blocks[0].ffn.experts)
        models[-1]['FFN_selected_experts_per_token_per_layer'] = cfg['top_k']
        models[-1]['FFN_router_parameters_per_layer'] = router_capacity

assert models[0]['unique_parameters']-models[1]['unique_parameters'] == 4*(3*262912+1024)

variants = []
for top_k in [1,2]:
    module = PaddingSafeMoE(4,4,top_k,8)
    with torch.no_grad():
        module.router.weight.zero_()
        module.router.weight[:,0] = torch.tensor([4.,3.,-3.,-4.])
    calls = []
    hooks = [e.register_forward_pre_hook(lambda m,args,index=i:calls.append({'expert':index,'rows':len(args[0])}))
             for i,e in enumerate(module.experts)]
    stored = {n:p.data_ptr() for n,p in module.named_parameters()}
    passes = []
    for value in [1.,-1.]:
        calls.clear()
        with torch.no_grad():
            output,aux,route = module(torch.tensor([[[value,0.,0.,0.]]]))
        assert len(calls) == top_k and route['counts'].sum().item() == top_k
        assert stored == {n:p.data_ptr() for n,p in module.named_parameters()}
        passes.append({'input_first_value':value,'chosen':route['chosen'].tolist(),
                       'calls':list(calls),'all_four_expert_weights_retained':True})
    for h in hooks:h.remove()
    assert passes[0]['chosen'] != passes[1]['chosen']
    variants.append({'top_k':top_k,'resident_parameters':sum(p.numel() for p in module.parameters()),'passes':passes})

aggregates = []
for arch in ['moe','dense']:
    p=REPO/f'docs/selftrained/results/public-raw/{arch}/test/metrics.json'
    count=pointer(p,'/count');expected_count=pointer(p,'/expected_count')
    complete=pointer(p,'/evaluation_complete');smoke=pointer(p,'/limited_smoke')
    # Inspect topkeys/task keys first, then only the already-produced aggregate denominators.
    tasks=json.loads(p.read_bytes())['per_task_final_reply'].keys()
    counts={t:pointer(p,f'/per_task_final_reply/{t}/count') for t in tasks}
    weight_hash=pointer(p,'/safe_weights_sha256')
    assert count == expected_count == sum(counts.values()) == 3734 and complete and not smoke
    ep=REPO/f'docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json'
    completed=pointer(ep,'/completed_count');expected_receipt=pointer(ep,'/expected_count');status=pointer(ep,'/status')
    assert completed == count == expected_receipt
    aggregates.append({'architecture':arch,'existing_aggregate_count':count,'per_task_counts':counts,
                       'weights_sha256':weight_hash,'completed_receipt_count':completed,'receipt_status':status})

lineage = []
for arch, stage in [('moe','moe-native'),('dense','dense-weighted')]:
    p=REPO/f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json'
    values={k:pointer(p,'/'+k) for k in ['stage','architecture','config','steps','tokens','total_parameters','tool_loss_weight','numeric_run_loss_weight']}
    values['origin_kind']=pointer(p,'/origin/kind')
    history=json.loads(p.read_bytes())['stage_history']
    values['stage_history']=[{k:pointer(p,f'/stage_history/{i}/{k}') for k in ['stage','step','tokens']} for i in range(len(history))]
    if arch=='moe':values['native_voice_loss_weight']=pointer(p,'/native_voice_loss_weight')
    else:values['native_voice_loss_weight']=1.0
    assert values['architecture']==arch and values['total_parameters']==models[0 if arch=='moe' else 1]['unique_parameters']
    lineage.append(values)

report={'environment':{'python':platform.python_version(),'torch':torch.__version__,
                      'safetensors':safetensors.__version__,'device':'cpu','threads':'1','seed':'1061910'},
        'section_sha256':hashlib.sha256(section).hexdigest(),'chapter_fence_stdout':stdout.getvalue(),
        'scalar_element_sizes':{str(t):torch.zeros(1,dtype=t).element_size() for t in [torch.float32,torch.float16,torch.bfloat16]},
        'models':models,'controlled_routing_variants':variants,
        'existing_aggregate_checks':aggregates,'raw_lineage_measurements':lineage,
        'inspected_json_pointers':reads,
        'scope':'Read-only original aggregates; only four synthetic tokens per loaded model; no task evaluation, training, paid compute, quantization, or distillation.'}
(BASE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['inspected_json_pointers','raw_lineage_measurements']},ensure_ascii=False,indent=2))
