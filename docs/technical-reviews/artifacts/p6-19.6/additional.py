"""Check original-recording counts, fonts, and stage provenance without result notes."""
from pathlib import Path
import json,hashlib,sys,gzip,ast,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[4];out=Path(__file__).resolve().parent
data=root/'outputs/selftrained-v2/data';sys.path.insert(0,str(root))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'read_pointers':{},'voice':{},'font_sha256':{},'stage_commands':{},'source_sha256':{}}
parquet=data/'voice-sources/minds14-zh-CN.parquet';table=pq.read_table(parquet,columns=['audio','intent_class','lang_id','path'])
source={hashlib.sha256(r['audio']['bytes']).hexdigest():r for r in table.to_pylist()}
result['original_parquet_sha256']=sha(parquet)
all_original=set()
for split in ['train','validation','test']:
    path=data/f'voice-{split}.jsonl';rows=[json.loads(l) for l in path.read_bytes().splitlines() if l]
    originals={r['audio'] for r in rows if 'augmentation' not in r}
    parents={r.get('augmentation',{}).get('parent_audio',r['audio']) for r in rows}
    assert originals==parents
    for a in originals:
        h=sha(data/a);assert h in source
        assert source[h]['intent_class'] in [1,2,6] and source[h]['lang_id']==13
    result['voice'][split]={'jsonl_sha256':sha(path),'rows':len(rows),'processed_assets':len({r['audio'] for r in rows}),'original_source_recordings':len(originals),'derived_assets':len({r['audio'] for r in rows})-len(originals),'source_group_count':len({r['group_id'] for r in rows})}
    result['read_pointers'][str(path)]=['/*/audio','/*/group_id','/*/augmentation/parent_audio','/*/augmentation/parent_audio_sha256','/*/split']
    all_original |= originals
result['original_source_recordings_total']=len(all_original)
assert len(all_original)==107
assert [result['voice'][s]['original_source_recordings'] for s in ['train','validation','test']]==[62,15,30]

expected={'Sans':'dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4','Serif':'234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798'}
for name,digest in expected.items():
    path=data/'fonts'/f'Noto{name}CJKtc-Regular.otf';assert sha(path)==digest
    result['font_sha256'][name]=digest
from scripts.selftrained.prepare_vision_ocr import draw_line, PINNED_SHA256
for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
    path=data/'source-cache/fashion-mnist'/name
    assert sha(path)==PINNED_SHA256[name]
    result['source_sha256']['official-fashion/'+name]=sha(path)
rows=json.loads((out/'selected-fixed-records.json').read_text());cards=[r for r in rows if r.get('image')=='images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png']
canvas=Image.new('L',(192,96),255);draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(data/'fonts/NotoSerifCJKtc-Regular.otf'),24)
# The supplied glyph boxes identify jitter = 0 among the producer's -1/0/+1 variants.
for r in cards:
    x,y,_,_=r['roi'];boxes=draw_line(draw,r['supervision']['ocr_text'],font,x,y,0)
    assert boxes==r['supervision']['glyph_boxes']
assert np.array_equal(np.asarray(canvas),np.asarray(Image.open(data/cards[0]['image'])))
canvas.save(out/'assets/recreated-ocr-card.png');result['ocr_pixels_recreated_exactly']=True
first=np.array([1.,0.,0.]);second=np.array([0.,1.,0.])
result['positionless_mean']={'forward':((first+second)/2).tolist(),'swapped':((second+first)/2).tolist(),'equal':bool(np.array_equal((first+second)/2,(second+first)/2))}
from tiny_perceptron.selftrained.inference import InferenceAssistant
import torch
torch.set_num_threads(2)
assistant=InferenceAssistant('/tmp/p5-native-public-cpu-smoke-actual/public-model',data,device='cpu',manifest_sha256='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e')
result['history_token_checks']={}
for name in ['voice_topic_continuation-1','voice_topic_continuation-3']:
    original=json.loads((out/'inputs/public-cpu-raw'/f'{name}-stdout.txt').read_text());g=original['generations'][0];history=g['prompt_messages']
    row=assistant.encoder.encode({'id':'fresh-history','task':'voice_topic_continuation','messages':history},generation=True,messages=history)
    assert row['input_ids']==g['prompt_ids']
    assert assistant.encoder.tokenizer.decode(g['generated_ids'],skip_special_tokens=True)==original['answer']
    assert assistant.receipt['files']['model.safetensors']==original['model']['files']['model.safetensors']
    result['history_token_checks'][name]={'prompt_ids_exact':True,'generated_ids_decode_exact':True,'model_hash_exact':True}
rawfiles=['scripts/selftrained/augment_voice_v2.py']
for architecture in ['dense','moe']:
    for stage in ['pretrain','sft','vision','ocr','audio','joint']:
        path=root/f'docs/selftrained/results/training-raw/{architecture}-{stage}/raw/execution.json';d=json.loads(path.read_text())
        result['stage_commands'][architecture+'-'+stage]={k:d[k] for k in ['stage','revision','command','returncode','started_at','finished_at']}
        result['read_pointers'][str(path)]=['/stage','/revision','/command','/returncode','/started_at','/finished_at']
        target=out/'inputs/training-raw'/f'{architecture}-{stage}-execution.json';shutil.copyfile(path,target);assert sha(path)==sha(target)
        result['source_sha256'][str(path.relative_to(root))]=sha(path)
        if stage=='pretrain':assert '--init-checkpoint' not in d['command'] and '--resume' not in d['command']
        if stage in ['vision','ocr','audio','joint']:assert '--init-checkpoint' in d['command']
for f in rawfiles:
    p=root/f;t=out/'inputs'/f;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);result['source_sha256'][f]=sha(p)
    result['ast_locations']=[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)]
(out/'additional-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['voice','original_source_recordings_total','font_sha256','ocr_pixels_recreated_exactly']},ensure_ascii=False,indent=2))
