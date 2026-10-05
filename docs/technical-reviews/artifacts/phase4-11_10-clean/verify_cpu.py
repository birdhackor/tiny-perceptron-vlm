"""Independent bounded CPU checks for section 11.10; no training or model files."""
import ast
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, expand_modalities, patchify, scene

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(42)
out = Path(__file__).resolve().parent
checks = {"environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)}, "scope": "Original fence separately executed; this script checks coordinate axes, sequence counts, attention table cells, encoder shapes, max_length, and audits historical raw records. It performs zero optimizer steps, no checkpoint reads, no model retest, no downloads."}

numbers = []
for h in (16, 32):
    image = torch.arange(3 * h * h).reshape(1, 3, h, h).float()
    for p in (8, 4, 2):
        patches = patchify(image, p)
        n = (h // p) ** 2
        assert list(patches.shape) == [1, n, 3 * p * p]
        # Direct coordinate check, rather than a mutually compatible round trip.
        for r in range(h // p):
            for c in range(h // p):
                assert torch.equal(patches[0, r * (h // p) + c], image[0, :, r*p:(r+1)*p, c*p:(c+1)*p].flatten())
        total = 20 + n
        numbers.append({"height_width": [h, h], "patch_size": p, "visual_tokens": n, "text_positions": 20, "total": total, "full_attention_cells": total * total, "patch_values": 3*p*p})
assert [(x["visual_tokens"], x["total"], x["full_attention_cells"]) for x in numbers[:3]] == [(4,24,576),(16,36,1296),(64,84,7056)]
assert [(x["visual_tokens"], x["total"], x["full_attention_cells"]) for x in numbers[3:]] == [(16,36,1296),(64,84,7056),(256,276,76176)]
assert list(patchify(torch.zeros(1,3,16,32),4).shape) == [1,32,48]
try:
    patchify(torch.zeros(1,3,17,16),4)
except ValueError as error:
    checks["nondivisible_rejected"] = str(error)
else:
    raise AssertionError("nondivisible height accepted")
checks["arithmetic_and_axes"] = numbers
checks["rectangular_control"] = {"image_shape": [1,3,16,32], "patch_size":4, "shape":[1,32,48], "scope":"HW/P^2 generalization; lesson height-only formula assumes its explicitly square images."}

attention_rows = []
for n in (24,36,84):
    q = torch.randn(1,1,n,4)
    k = torch.randn(1,1,n,4)
    v = torch.randn(1,1,n,3)
    allowed = attention_mask(torch.arange(n), torch.arange(n))
    values, weights = manual_attention(q,k,v,allowed)
    assert list(weights.shape) == [1,1,n,n]
    assert weights.numel() == n*n
    assert int(allowed.sum()) == n*(n+1)//2
    assert torch.allclose(weights.sum(-1), torch.ones(1,1,n), atol=2e-7, rtol=1e-6)
    assert not weights[~allowed].any()
    attention_rows.append({"N":n, "weight_shape":list(weights.shape), "full_cells_per_head":weights.numel(), "causal_allowed_cells":int(allowed.sum()), "axis_check":"each query row sums to one across source key columns"})
# A nonsquare attention table verifies that rows count queries and columns keys.
_, w = manual_attention(torch.zeros(1,1,2,4),torch.zeros(1,1,3,4),torch.arange(6).reshape(1,1,3,2).float(),torch.ones(1,1,2,3,dtype=torch.bool))
assert list(w.shape)==[1,1,2,3] and torch.allclose(w,torch.full_like(w,1/3))
checks["attention"]={"square":attention_rows,"axis_control":{"queries":2,"keys":3,"table_cells":6},"cost_scope":"full dense logical table for one head, without cache; masked allowed cells differ. No timing or memory benchmark."}

encoders={p:VisionEncoder(patch_size=p) for p in (2,4,8)}
checks["entry_matrices"]=[]
for p,e in encoders.items():
    features=e(scene()[None])
    assert list(features.shape)==[1,(16//p)**2,16]
    assert list(e.projection.weight.shape)==[16,3*p*p]
    checks["entry_matrices"].append({"patch_size":p,"weight_shape":list(e.projection.weight.shape),"weight_parameters":e.projection.weight.numel(),"position_parameters":e.position.numel(),"encoder_total_parameters":sum(x.numel() for x in e.parameters()),"output_shape":list(features.shape)})
try:
    encoders[4].projection(patchify(scene()[None],2))
except RuntimeError as error:
    checks["old_projection_rejected_new_patch"] = str(error)
else:
    raise AssertionError("48-column projection accepted 12-value patches")

lm=TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=64))
assert list(lm(embeddings=torch.zeros(1,64,8))["logits"].shape)==[1,64,264]
try:
    lm(embeddings=torch.zeros(1,84,8))
except ValueError as error:
    checks["max_length"]={"64":"accepted","84":"rejected","error":str(error)}
else:
    raise AssertionError("84 positions accepted by max_length=64")
larger=TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=128))
assert list(larger(embeddings=torch.zeros(1,84,8))["logits"].shape)==[1,84,264]
checks["max_length"]["larger_128"]="84 positions accepted structurally; no task accuracy was measured"
tok=ByteTokenizer()
ids=torch.tensor([tok.image_id]+[20]*20)
labels=torch.tensor([-100]*17+[30]*4)
expanded,y=expand_modalities(ids,labels,lm.embedding,{tok.image_id:torch.zeros(64,8)},{tok.image_id})
assert expanded.shape[1]==83 and y.shape[1]==83
assert int((y!=-100).sum())==4 and int((y[:,:64]!=-100).sum())==0
checks["truncation_control"]={"unshifted_total_positions":84,"shifted_input_positions":83,"full_effective_targets":4,"effective_targets_in_first_64_shifted_positions":0,"scope":"one synthetic sequence illustrates that prefix truncation can remove all answer targets; original arithmetic counts unshifted material positions, not this helper's shifted training sequence."}

path=out/'inputs/docs/course-experiments/results/vision_ablation.json'
j=json.loads(path.read_bytes())
inspection_pointers=["/revision","/seed","/device","/torch_version","/python_version","/step_scale","/code_sha256","/results/data/seed","/results/data/split_policy"]
def original_function(name):
    source=out/'inputs/historical/scripts/course_experiments/modalities.py'
    tree=ast.parse(source.read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=ast.Module(body=[node],type_ignores=[])
    namespace={"ByteTokenizer":ByteTokenizer,"torch":torch}
    exec(compile(code,str(source)+':'+name,'exec'),namespace)
    return namespace[name]
build_records=original_function('_vision_records')
sequence=original_function('_sequence')
expected_splits=build_records(("shape?","color?"))
ctx=SimpleNamespace(device='cpu')
families={};split_audit={}
for split,v in j['results']['data']['splits'].items():
    inspection_pointers.append('/results/data/splits/'+split)
    records=v['records']
    assert records==expected_splits[split]
    actual_hash=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert v['count']==len(records) and actual_hash==v['sha256']
    families[split]={r['family'] for r in records}
    split_audit[split]={"records":len(records),"families":len(families[split]),"offsets":sorted({r['offset'] for r in records}),"sha256_verified":actual_hash}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not families[a]&families[b]
assert set(j['results']['patch_variants'])=={'4','8'}
variants={}
for patch,v in j['results']['patch_variants'].items():
    prefix='/results/patch_variants/'+patch
    t=v['training'];te=v['test'];records=j['results']['data']['splits']['test']['records']
    fields=['parameters','trainable_parameters','config','modal_config','steps','history','effective_tokens','effective_targets','weights_changed','nonzero_gradient_seen','cpu_smoke']
    inspection_pointers.extend([prefix+'/visual_tokens']+[prefix+'/training/'+f for f in fields]+[prefix+'/test/'+f for f in ['examples','correct','exact_match','effective_tokens','eos_rate','generation_errors','invalid_special_tokens','skipped','groups','macro_accuracy','samples']])
    assert v['visual_tokens']==(16//int(patch))**2
    assert t['steps']==200==len(t['history'])
    per_step=[]
    for step in range(200):
        rng=random.Random(j['seed']+step)
        counts=sum(sequence(rng.choice(expected_splits['train']),ctx)[2] for _ in range(4))
        per_step.append(counts)
        assert t['history'][step]['step']==step+1 and t['history'][step]['effective_targets']==counts
    assert sum(per_step)==4765==t['effective_targets']==t['effective_tokens']
    assert t['weights_changed'] and t['nonzero_gradient_seen'] and not t['cpu_smoke']
    candidate=MultiModalLM(TinyLM(ModelConfig(**t['config'])))
    candidate.vision=VisionEncoder(patch_size=int(patch))
    assert sum(p.numel() for p in candidate.parameters())==t['parameters']==t['trainable_parameters']
    group_count=Counter();group_correct=Counter();correct=0;token_count=0;eos=0
    sample_receipts=[]
    assert len(te['samples'])==12==te['examples']==len(records)
    for index,(s,r) in enumerate(zip(te['samples'],records,strict=True)):
        assert s['row']==index and s['family']==r['family'] and s['question']==r['question'] and s['target']==r['answer']
        generated=s['generated_ids'];raw=generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        exact=raw==tok.encode(r['answer'])
        assert exact==s['exact_match'] and tok.decode(raw)==s['generated']
        assert (tok.eos_id in generated)==s['eos']
        assert s['generation_error'] is None and s['invalid_special_tokens']==sum(x<8 for x in raw)==0
        assert s['donor_row'] is None
        correct+=int(exact);eos+=int(s['eos']);token_count+=sequence(r,ctx)[2]
        group_count[r['question']]+=1;group_correct[r['question']]+=int(exact)
        sample_receipts.append({"row":index,"family":s['family'],"question":s['question'],"target":r['answer'],"generated":s['generated'],"recomputed_exact":exact,"eos":s['eos']})
    assert correct==te['correct']==9 and correct/12==te['exact_match']==0.75
    assert token_count==te['effective_tokens']==72
    assert eos/12==te['eos_rate']==1 and te['skipped']==[]
    assert te['generation_errors']==0==te['invalid_special_tokens']
    groups={k:{'correct':group_correct[k],'count':group_count[k]} for k in group_count}
    assert groups==te['groups'] and sum(x['correct']/x['count'] for x in groups.values())/len(groups)==te['macro_accuracy']==0.75
    variants[patch]={"visual_tokens":v['visual_tokens'],"config":t['config'],"modal_config":t['modal_config'],"parameters_recomputed":t['parameters'],"steps":t['steps'],"effective_training_target_tokens_recomputed":sum(per_step),"test_questions":12,"test_image_families":6,"test_correct_recomputed":correct,"test_effective_target_tokens_recomputed":token_count,"groups_recomputed":groups,"eos_correct":eos,"samples_recomputed":sample_receipts}
checks['historical_audit']={"input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"original_revision":j['revision'],"original_device":j['device'],"original_torch":j['torch_version'],"seed":j['seed'],"splits":split_audit,"variants":variants,"method_contract":"same inherited SFT language initialization, fresh vision encoders, seed 42, all parameters trainable, AdamW lr .003, four sampled examples per step, 200 updates, exact byte-token content before first EOS; not equal FLOP, parameter, or wall time budget","scope":"audit of existing recorded samples/config/history only; historical GPU training is not reproduced; no small text targets; score tie cannot establish universal equivalence or superiority."}
checks['json_inspection_pointers']=inspection_pointers
print(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False))
