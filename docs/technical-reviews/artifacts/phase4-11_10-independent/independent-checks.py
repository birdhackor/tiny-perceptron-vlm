"""Bounded CPU checks and pointer-scoped historical-data verification; no training."""
from pathlib import Path
from datetime import datetime, UTC
from collections import Counter
import hashlib
import json
import sys
import contextlib
import io

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import VisionEncoder, patchify, unpatchify, scene
from tiny_perceptron.attention import manual_attention, attention_mask
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.data import ByteTokenizer, pad_batch, IGNORE

torch.set_num_threads(1)
torch.manual_seed(1010)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": "1", "seed": "1010"}
result = {"checked_at": datetime.now(UTC).isoformat(), "environment": environment, "scope": "Exact arithmetic, random untrained CPU forwards and historical JSON arithmetic only. No optimizer, training, saved neural weights or historical-model inference."}

geometry = []
for height in (16, 32):
    for patch_size in (8, 4, 2):
        image = torch.zeros(1, 3, height, height)
        patches = patchify(image, patch_size)
        visual_tokens = (height // patch_size) ** 2
        total = visual_tokens + 20
        assert patches.shape == (1, visual_tokens, 3 * patch_size**2)
        assert torch.equal(image, unpatchify(patches, 3, height, height, patch_size))
        geometry.append({"image_height_width_pixels": height, "patch_edge_pixels": patch_size, "patch_tensor_B_N_Cpp": list(patches.shape), "text_positions": 20, "total_positions": total, "full_cells_per_head_per_batch": total**2})
assert [(x["total_positions"],x["full_cells_per_head_per_batch"]) for x in geometry[:3]] == [(24,576),(36,1296),(84,7056)]
assert geometry[-1]["total_positions"] == 276 and geometry[-1]["full_cells_per_head_per_batch"] == 76176
result["geometry"] = geometry
result["halving_edge_4_to_2"] = {"patch_count_ratio": 64/16, "note": "Each spatial axis doubles; patch count quadruples. N^2 includes unchanged 20 text positions, so cells do not simply increase sixteenfold."}
assert patchify(torch.zeros(1,3,16,8),4).shape == (1,8,48)
try:
    patchify(torch.zeros(1,3,15,16),4)
except ValueError as e:
    result["geometry_boundary"] = {"rectangle_16x8_p4": [1,8,48], "nondivisible_15x16_p4": str(e), "scope": "Original (height//p)^2 formula is for the stated square divisible images."}
else:
    raise AssertionError("Nondivisible input was not rejected")

attention = []
for total in (24,36,84):
    q,k,v = [torch.randn(1,1,total,4) for _ in range(3)]
    positions = torch.arange(total)
    allowed = attention_mask(positions,positions)
    y, weights = manual_attention(q,k,v,allowed)
    assert weights.shape == (1,1,total,total)
    assert weights.numel() == total**2
    assert int(allowed.sum()) == total*(total+1)//2
    assert torch.count_nonzero(weights[0,0].triu(1)) == 0
    assert torch.allclose(weights.sum(-1),torch.ones(1,1,total),atol=1e-6)
    attention.append({"positions":total,"axis_order":"batch,head,query_row,key_column","weights_shape":list(weights.shape),"full_cells":weights.numel(),"causally_allowed_cells":int(allowed.sum()),"upper_triangle_weights":0,"output_shape":list(y.shape)})
q,k,v = [torch.randn(2,3,5,4) for _ in range(3)]
_,w=manual_attention(q,k,v,attention_mask(torch.arange(5),torch.arange(5)))
assert w.shape == (2,3,5,5) and w.numel() == 150
result["attention"] = attention
result["attention_boundary"] = {"B_H_N": [2,3,5], "full_cells_all_batches_heads": 150, "scope": "N^2 counts one full query-key table, not all heads/layers/batches and not allowed-edge count, bytes or seconds."}

entry=[]
for p in (2,4,8):
    enc=VisionEncoder(width=16,patch_size=p)
    with torch.no_grad(): features=enc(scene()[None])
    entry.append({"patch_size":p,"input_pixel_values_per_patch":3*p*p,"weight_shape_out_in":list(enc.projection.weight.shape),"projection_parameters_with_bias":sum(x.numel() for x in enc.projection.parameters()),"position_parameters":enc.position.numel(),"encoder_parameters":sum(x.numel() for x in enc.parameters()),"features_B_N_D":list(features.shape)})
old=VisionEncoder(width=16,patch_size=4)
new_patches=patchify(scene()[None],2)
try:
    old.projection(new_patches)
except RuntimeError as e: result["old_projection_new_patch_error"] = str(e)
else: raise AssertionError("48-column projection accepted 12 values")
old.patch_size=2
try:
    old(scene()[None])
except RuntimeError as e: result["mutated_inference_patch_error"] = str(e)
else: raise AssertionError("Mutation alone accepted new patch shape")
result["entry_projection"] = entry

cap=TinyLM(ModelConfig(vocab_size=16,width=8,layers=1,heads=1,max_length=64))
capacity=[]
for p in (8,4,2):
    enc=VisionEncoder(width=8,patch_size=p)
    with torch.no_grad(): x=torch.cat([enc(scene()[None]),torch.zeros(1,20,8)],dim=1)
    try:
        with torch.no_grad(): y=cap(embeddings=x)
        assert x.shape[1] <= 64
        capacity.append({"patch_size":p,"length":x.shape[1],"accepted_logits_shape":list(y["logits"].shape)})
    except ValueError as e:
        assert x.shape[1] == 84
        capacity.append({"patch_size":p,"length":x.shape[1],"rejected":str(e)})
large=TinyLM(ModelConfig(vocab_size=16,width=8,layers=1,heads=1,max_length=84))
with torch.no_grad(): large_y=large(embeddings=torch.zeros(1,84,8))
assert large_y["logits"].shape == (1,84,16)
result["capacity"] = capacity
result["larger_limit"] = {"max_length":84,"accepted_logits_shape":list(large_y["logits"].shape),"scope":"Random untrained forward verifies interface capacity only; no task competence test."}
labels=torch.full((84,),IGNORE);labels[64:]=1
try:
    pad_batch([(torch.zeros(84,dtype=torch.long),labels)],max_length=64)
except ValueError as e: result["truncation_loses_all_answer_targets"]={"original_active_targets":20,"kept_active_targets":0,"error":str(e)}
else: raise AssertionError("All-answer truncation was not rejected")

tiny=torch.zeros(1,3,16,16);tiny[:,0,:2,:2]=1
granularity=[]
for p in (4,2):
    patches=patchify(tiny,p)
    red_mean=patches[:,:,:p*p].mean(-1)
    granularity.append({"patch_edge_pixels":p,"highest_red_mean":float(red_mean.max()),"positions_above_half":int((red_mean>0.5).sum())})
assert granularity == [{"patch_edge_pixels":4,"highest_red_mean":0.25,"positions_above_half":0},{"patch_edge_pixels":2,"highest_red_mean":1.0,"positions_above_half":1}]
result["bounded_small_object_derivation"]={"input":"A 2x2 red region on otherwise black 16x16 image","projection":"Hand-selected mean of red pixels in each patch; not the trained VisionEncoder","observed":granularity,"scope":"Illustrates possible dilution of a small region by coarse fixed pooling. Both raw patch arrays retain every pixel; learned small-object recognition advantage is not established."}

raw_path=ROOT/'docs/course-experiments/results/vision_ablation.json'
raw=raw_path.read_bytes();(OUT/'vision_ablation.original.json').write_bytes(raw)
data=json.loads(raw)
pointers=[]
def read(ptr):
    value=data
    for part in ptr.strip('/').split('/'):value=value[part]
    pointers.append(ptr)
    return value
historical={"original_path":str(raw_path.relative_to(ROOT)),"full_sha256":hashlib.sha256(raw).hexdigest(),"snapshot":"vision_ablation.original.json","provenance":{p[1:]:read(p) for p in ['/revision','/device','/seed','/torch_version','/python_version','/gpu','/step_scale']}}
tokenizer=ByteTokenizer()
split_summaries={}
families={}
test_records=None
for split in ('train','validation','test'):
    base=f'/results/data/splits/{split}'
    count=read(base+'/count'); records=read(base+'/records');recorded_hash=read(base+'/sha256')
    assert count==len(records)
    hash_=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest();assert hash_==recorded_hash
    families[split]={r['family'] for r in records}
    split_summaries[split]={'count':count,'records_sha256':hash_,'family_count':len(families[split]),'questions':dict(Counter(r['question'] for r in records)),'offsets':sorted({r['offset'] for r in records}),'modalities':sorted({r['modality'] for r in records})}
    assert all(r['question'] in ('shape?','color?') for r in records)
    assert all(r['answer']==(r['shape'] if r['question']=='shape?' else r['color']) for r in records)
    assert all('digits' not in r and 'image' not in r for r in records)
    if split=='test':test_records=records
assert not any(families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
historical['splits']=split_summaries
variants={}
for p in (4,8):
    base=f'/results/patch_variants/{p}'
    values=read(base+'/visual_tokens');assert values==(16//p)**2
    trainbase=base+'/training'
    selected_training={key:read(trainbase+'/'+key) for key in ['parameters','trainable_parameters','config','modal_config','steps','effective_tokens','effective_targets','weights_changed','nonzero_gradient_seen','cpu_smoke']}
    history=read(trainbase+'/history');assert len(history)==selected_training['steps'];assert [h['step'] for h in history]==list(range(1,len(history)+1))
    assert sum(h['effective_targets'] for h in history)==selected_training['effective_targets']
    assert selected_training['modal_config']['patch_size']==p
    samples=read(base+'/test/samples'); examples=read(base+'/test/examples');correct=read(base+'/test/correct');accuracy=read(base+'/test/exact_match'); effective=read(base+'/test/effective_tokens');skipped=read(base+'/test/skipped')
    assert len(samples)==examples==len(test_records)==12 and not skipped
    checked=[]
    for s,r in zip(samples,test_records,strict=True):
        assert s['family']==r['family'] and s['question']==r['question'] and s['target']==r['answer']
        raw_ids=s['generated_ids']; raw_answer=raw_ids[:raw_ids.index(tokenizer.eos_id)] if tokenizer.eos_id in raw_ids else raw_ids
        exact=raw_answer==tokenizer.encode(s['target'])
        assert exact==s['exact_match'];assert tokenizer.decode(raw_answer)==s['generated']
        checked.append({'row':s['row'],'family':s['family'],'question':s['question'],'target':s['target'],'generated':s['generated'],'exact_match_recomputed':exact,'eos':tokenizer.eos_id in raw_ids})
    recomputed=sum(x['exact_match_recomputed'] for x in checked);assert recomputed==correct==9 and accuracy==correct/examples==0.75
    computed_targets=sum(len(tokenizer.encode(r['answer']))+1 for r in test_records);assert computed_targets==effective
    groups={q:{'count':sum(x['question']==q for x in checked),'correct':sum(x['question']==q and x['exact_match_recomputed'] for x in checked)} for q in ('shape?','color?')}
    variants[str(p)]={'visual_tokens':values,'training':selected_training,'history_steps_checked':len(history),'test':{'examples':examples,'correct_recomputed':correct,'exact_match':accuracy,'target_tokens_recomputed':effective,'groups_recomputed':groups,'samples_checked':checked}}
historical['variants']=variants
historical['selected_code_hashes']={}
for file in ['tiny_perceptron/data.py','tiny_perceptron/multimodal.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','scripts/course_experiments/modalities.py']:
    reported=read('/code_sha256/'+file.replace('~','~0').replace('/','~1')) if False else data['code_sha256'][file]
    pointers.append('/code_sha256/'+file.replace('/','~1'))
    current=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    historical['selected_code_hashes'][file]={'reported':reported,'inspected_current_sha256':current,'same_bytes':reported==current}
historical['inspected_pointers']=pointers
historical['scope']='Read and recomputed original raw samples, configs, histories and provenance. Historical CUDA run is not repeated; no model was loaded. Dataset is shape/color only and cannot establish a small-text or small-object benefit.'
result['historical']=historical
(OUT/'independent-checks-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
