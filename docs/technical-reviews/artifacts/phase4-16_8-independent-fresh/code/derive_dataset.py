import ast,hashlib,json,random,sys
from pathlib import Path
import torch

ROOT=Path('/workspace/tiny-perceptron-vlm');OUT=ROOT/'docs/technical-reviews/artifacts/phase4-16_8-independent-fresh'
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import ByteTokenizer,render_chat,shifted,pad_batch,IGNORE
eff=json.loads((OUT/'inputs/efficiency.json').read_bytes())
assert hashlib.sha256((ROOT/'tiny_perceptron/data.py').read_bytes()).hexdigest()==eff['code_sha256']['tiny_perceptron/data.py']
origin=ROOT/'docs/technical-reviews/artifacts/phase4-16_5-independent/input/dataset-original.json'
raw=origin.read_bytes();expected=next(x['sha256'] for x in eff['artifacts'] if x['path']=='dataset.json')
assert hashlib.sha256(raw).hexdigest()==expected
copy=OUT/'inputs/dataset-original.json';assert copy.read_bytes()==raw
data=json.loads(raw)
train_hash=hashlib.sha256(json.dumps(data['train'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
assert train_hash==eff['results']['dataset']['train']['sha256']
tree=ast.parse((OUT/'inputs/efficiency__scripts__course_experiments__common.py').read_bytes())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='text_examples')
ns=dict(ByteTokenizer=ByteTokenizer,render_chat=render_chat,shifted=shifted)
exec(compile(ast.Module(body=[function],type_ignores=[]),'original:text_examples','exec'),ns)
examples=ns['text_examples'](data['train'],mode='sft',max_length=eff['results']['models']['mha']['model']['config']['max_length'])
x,y,valid=pad_batch(examples[:4]);assert list(x.shape)==eff['results']['manual_vs_sdpa']['shape']==[4,53]
sampler=random.Random(eff['seed']);targets=[]
for _ in range(40):
 batch=sampler.choices(examples,k=8)
 targets.append(sum(int((label!=IGNORE).sum()) for _,label in batch))
assert sum(targets)==2328
for route in ['ordinary','sdpa']:
 assert eff['results']['update_variants'][route]['training']['effective_tokens']==sum(targets)

flash=json.loads((OUT/'inputs/flash_probe.json').read_bytes())
shape=flash['results']['configuration']['shape_B_H_T_D']
generator=torch.Generator().manual_seed(flash['seed'])
reconstructed={name:torch.randn(shape,generator=generator) for name in ['q','k','v','upstream']}
reconstructed['upstream']*=0.125
assert all(list(x.shape)==shape and torch.isfinite(x).all() for x in reconstructed.values())
result={
 'dataset_original_locator':str(origin.relative_to(ROOT)),
 'dataset_permanent_own_copy':str(copy.relative_to(ROOT)),
 'dataset_original_and_copy_sha256':expected,
 'dataset_bytes':len(raw),
 'train_records':len(examples),'train_records_sha256':train_hash,
 'read_dataset_pointers':['/train'],
 'batch_first_four_shape':list(x.shape),'first_four_input_lengths':[len(a) for a,_ in examples[:4]],
 'effective_targets_per_update':targets,'total_effective_targets_40_updates':sum(targets),
 'sampler_seed':eff['seed'],'batch_size':8,'ignore_index':IGNORE,
 'dtype_width_bits':{str(t):torch.finfo(t).bits for t in [torch.float32,torch.float16,torch.bfloat16]},
 'flash_fixture_reconstruction':{
  'kind':'CPU reconstruction of recorded recipe; not recovered original fixture',
  'shape':shape,'seed':flash['seed'],'contains_model_weights':False,
  'original_fixture_declared_sha256':next(a['sha256'] for a in flash['artifacts'] if a['path']=='fixture.pt'),
  'original_fixture_bytes_available':False,
  'reconstructed_tensors_finite':True,
  'tensor_dtype':str(reconstructed['q'].dtype),
  'tensor_bytes_sha256':{name:hashlib.sha256(t.numpy().tobytes()).hexdigest() for name,t in reconstructed.items()},
  'saved_tensor_files':False,'reran_cuda':False,
 },
 'scope':'Replays only original data rendering, seeded sampling and input tensor recipe. No model loading, model scoring, optimizer, training, or CUDA execution.'
}
(OUT/'dataset_derivation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'dataset_stdout.txt').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
