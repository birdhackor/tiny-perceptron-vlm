"""Fresh bounded CPU verification for section 19.6. No training or evaluation sweep."""
from pathlib import Path
import ast, base64, collections, gzip, hashlib, io, json, platform, shutil, struct, subprocess, sys, tarfile
import importlib.metadata as md
import numpy as np
import soundfile as sf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'outputs/selftrained-v2/data'
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    p=OUT/name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def copy(p, relative):
    t=OUT/relative; t.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,t)
    assert sha(p)==sha(t)
    return {'source':str(p), 'copy':str(t.relative_to(ROOT)), 'sha256':sha(p)}

result={'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu','platform':platform.platform()},'frozen':[], 'read_pointers':{}}
for package in ['numpy','Pillow','soundfile','pyarrow','safetensors']:
    result['environment'][package]=md.version(package)
chapter=(ROOT/'course/chapters/19.md').read_bytes()
start=chapter.index('## 19.6 '.encode());end=chapter.index('## 19.7 '.encode(),start)
(OUT/'inputs/19.6.md').write_bytes(chapter[start:end])
result['section_sha256']=sha(OUT/'inputs/19.6.md')
result['frozen_chapter']=copy(ROOT/'course/chapters/19.md','inputs/19.frozen.md')
codefiles=['tiny_perceptron/selftrained/dataset.py','tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/inference.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/multimodal.py','scripts/selftrained/train.py','scripts/selftrained/chat.py','scripts/selftrained/evaluate.py','scripts/selftrained/prepare_vision_ocr.py','scripts/selftrained/prepare_voice.py','scripts/selftrained/augment_ocr_v2.py','scripts/selftrained/prepare_v2.py','docs/selftrained/examples/v2/append-voice-continuation.py','docs/selftrained/v2-manifest.json']
for f in codefiles: result['frozen'].append(copy(ROOT/f,'inputs/'+f))
result['ast_locations']={}
for f in codefiles:
    if f.endswith('.py'):
        result['ast_locations'][f]=[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(ast.parse((ROOT/f).read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_text())
result['read_pointers']['docs/selftrained/v2-manifest.json']=['/package','/records','/assets','/initialization','/model_config']
result['package']=manifest['package']
archive=ROOT/manifest['package']['path']
assert sha(archive)==manifest['package']['sha256'] and archive.stat().st_size==manifest['package']['bytes']
result['archive_verified']={'path':manifest['package']['path'],'sha256':sha(archive),'bytes':archive.stat().st_size}
with tarfile.open(archive,'r:gz') as tar:
    result['archive_metadata']=[]
    for name in ['DATA-NOTICE.md','licenses/minds14/PROJECT-NOTICE.md','licenses/minds14/README.md','licenses/fashion-mnist/LICENSE','licenses/fashion-mnist/README.md','licenses/noto-cjk/Sans-LICENSE.txt','licenses/noto-cjk/Serif-LICENSE.txt']:
        raw=tar.extractfile(name).read();target=OUT/'inputs/archive'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        result['archive_metadata'].append({'member':name,'saved':str(target.relative_to(ROOT)),'sha256':sha(target),'matches_cache':(DATA/name).exists() and sha(DATA/name)==sha(target)})
assets=['images/vision/c98cb897d639c880fde261eafc6d953d.png','images/vision/13913fb756b87b5c3e0f52e4ba194fec.png','images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png','audio/ef25d9a7a3a6ce3790a040ba.wav']
result['assets']=[]
for a in assets:
    m=next(x for x in manifest['assets'] if x['path']==a)
    assert sha(DATA/a)==m['sha256']
    result['assets'].append({**copy(DATA/a,'assets/'+a),'manifest_entry':m})

result['data_audit']={};selected=[];voice_sets={};ocr_words={}
for family in ['vision','ocr','voice']:
    for split in ['train','validation','test']:
        f=f'{family}-{split}.jsonl';raw=(DATA/f).read_bytes();m=next(x for x in manifest['records'] if x['path']==f)
        assert hashlib.sha256(raw).hexdigest()==m['sha256']
        rows=[json.loads(line) for line in raw.splitlines() if line]
        audit={'sha256':m['sha256'],'count':len(rows),'first_keys':list(rows[0]),'supervision_keys':list(rows[0]['supervision'])}
        if family=='vision': audit['class_ids']=sorted({v for r in rows for v in r['supervision']['vision_labels']})
        if family=='ocr':
            texts=[r['supervision']['ocr_text'] for r in rows];ocr_words[split]=set(texts)
            audit.update(font_families=dict(collections.Counter(r['supervision']['font_family'] for r in rows)),min_chars=min(map(len,texts)),max_chars=max(map(len,texts)),characters=''.join(sorted(set(''.join(texts)))))
            for font in ['Sans','Serif']:
                selected.append(next(r for r in rows if r['supervision']['font_family']==font))
        if family=='voice':
            voice_sets[split]={r['audio'] for r in rows}
            audit.update(recording_assets=len(voice_sets[split]),intents=sorted({r['supervision']['intent'] for r in rows}))
        selected.extend(r for r in rows if r.get('image',r.get('audio')) in assets)
        result['data_audit'][f]=audit
        result['read_pointers']['outputs/selftrained-v2/data/'+f]=['/*/id','/*/split','/*/task','/*/messages','/*/image','/*/audio','/*/roi','/*/image_layout','/*/supervision']
result['voice_unique_recordings']=len(set.union(*voice_sets.values()))
result['voice_split_overlap']={a+'-'+b:len(voice_sets[a]&voice_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
result['ocr_test_unseen_strings']=len(ocr_words['test']-ocr_words['train'])
save('selected-fixed-records.json',selected)

result['public_cpu']={}
names=['vision_clothing-1','vision_relation-1','ocr-1','voice_qa-1','voice_topic_continuation-1','voice_topic_continuation-2','voice_topic_continuation-3']
for n in names:
    folder=ROOT/'docs/selftrained/results/public-cpu-raw'
    for suffix in ['argv.json','result.json','stdout.txt','stderr.txt']:
        result['frozen'].append(copy(folder/f'{n}-{suffix}','inputs/public-cpu-raw/'+f'{n}-{suffix}'))
    av=json.loads((folder/f'{n}-argv.json').read_text());r=json.loads((folder/f'{n}-result.json').read_text())
    stdout=(folder/f'{n}-stdout.txt').read_text()
    assert sha(folder/f'{n}-stdout.txt')==r['stdout_sha256']
    record={'argv':av,'returncode':r['returncode']}
    if stdout.strip():
        o=json.loads(stdout)
        record.update({k:o[k] for k in ['answer','messages','generations']})
        assert o['answer']==r['actual_answer']==r['actual_generations'][0]['raw_output']
        record['model_sha256']=o['model']['files']['model.safetensors']
        record['model_origin']=o['model']['origin']
        result['read_pointers'][str(folder/f'{n}-stdout.txt')]=['/answer','/messages','/generations','/model/files','/model/origin']
    result['public_cpu'][n]=record
    result['read_pointers'][str(folder/f'{n}-result.json')]=['/returncode','/stdout_sha256','/actual_answer','/actual_generations']
first=result['public_cpu']['voice_topic_continuation-1'];last=result['public_cpu']['voice_topic_continuation-3']
assert last['argv']['public_messages_before_execution'][:-1]==first['messages']
assert last['generations'][0]['prompt_messages'][:-1]==first['messages']
assert first['messages'][-1]['content']==first['answer']
assert 'audio' in first['messages'][3] and 'audio' not in last['generations'][0]['prompt_messages'][-1]
result['history_verified']=True

from tiny_perceptron.selftrained.inference import InferenceAssistant
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.dataset import OCR_CHARACTERS
MODEL=Path('/tmp/p5-native-public-cpu-smoke-actual/public-model')
result['model_files']={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()}
assert result['model_files']['model.safetensors']==first['model_sha256']
for n in ['inference-manifest.json','model-config.json','tokenizer.json']: copy(MODEL/n,'inputs/public-model/'+n)
assistant=InferenceAssistant(MODEL,DATA,device='cpu',manifest_sha256='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e')
result['payloads']={}
with torch.no_grad():
    for n in ['vision_clothing-1','vision_relation-1','ocr-1','voice_qa-1']:
        h=result['public_cpu'][n]['generations'][0]['prompt_messages']
        row=assistant.encoder.encode({'id':'fresh-check','task':n.rsplit('-',1)[0],'messages':h},generation=True,messages=h)
        assert row['input_ids']==result['public_cpu'][n]['generations'][0]['prompt_ids']
        generated=result['public_cpu'][n]['generations'][0]['generated_ids']
        assert assistant.encoder.tokenizer.decode(generated,skip_special_tokens=True)==result['public_cpu'][n]['answer']
        p=row['modalities'][0];kind=p['kind'];values=p['values']
        args=(values,p['coordinates']) if kind=='image' else (values,p.get('valid')) if kind=='audio' else (values,)
        tokens,logits=getattr(assistant.model, {'image':'vision_encoder','ocr':'ocr_encoder','audio':'audio_encoder'}[kind])(*args)
        result['payloads'][n]={'kind':kind,'input_shape':list(values.shape),'tokens_shape':list(tokens.shape),'logits_shape':list(logits.shape),'reserved_slots':row['input_ids'].count({'image':7,'ocr':8,'audio':9}[kind]),'finite':bool(torch.isfinite(values).all())}
        if kind=='image': result['payloads'][n]['coordinates']=p['coordinates'].tolist()
        if kind=='audio':
            wave,rate=sf.read(DATA/assets[-1],dtype='float32')
            result['payloads'][n].update(sample_rate=rate,full_samples=len(wave),duration_seconds=len(wave)/rate,logits=logits.tolist(),softmax=logits.softmax(-1).tolist(),argmax=int(logits.argmax(-1)))
            result['voice_expected_intent_id']=next(r['supervision']['intent_id'] for r in selected if r.get('audio')==assets[-1])
            assert int(logits.argmax(-1))!=result['voice_expected_intent_id']
        if kind=='ocr':
            result['payloads'][n]['roi']=h[-1]['roi']
            result['payloads'][n]['resized_width']=round((90-18)*32/(86-50))
            assert torch.all(values[:,:,64:]==1)
            result['payloads'][n]['right_padding_white']=True

# Verify actual original WAV bytes and absence of speaker metadata in the fixed original parquet.
import pyarrow.parquet as pq
parquet=DATA/'voice-sources/minds14-zh-CN.parquet'
table=pq.read_table(parquet)
assert sha(parquet)=='cea0246e1a54afe5a3ab9d9542baa53c7cb45fc8f4e28e7872ebafefb25ba99d'
target_hash=sha(DATA/assets[-1])
orig=next(r for r in table.to_pylist() if hashlib.sha256(r['audio']['bytes']).hexdigest()==target_hash)
result['original_voice']={'parquet_sha256':sha(parquet),'columns':table.column_names,'count':table.num_rows,'source_path':orig['path'],'intent_class':orig['intent_class'],'lang_id':orig['lang_id'],'audio_bytes_sha256':target_hash,'project_wav_matches_original':True}
# Do not read source ASR transcript values; they are neither prompt input nor listening gold.
for p in (DATA/'voice-sources').glob('minds14-*'):
    if p.suffix!='.parquet': result['frozen'].append(copy(p,'official/'+p.name))

# Original Fashion-MNIST pixel patches match selected source IDs, not an artist approximation.
source=DATA/'source-cache/fashion-mnist'
source_arrays={}
for original_split in ['train','t10k']:
    raw=gzip.decompress((source/f'{original_split}-images-idx3-ubyte.gz').read_bytes())
    pixels=np.frombuffer(raw[16:],dtype=np.uint8).reshape(-1,28,28)
    labels=np.frombuffer(gzip.decompress((source/f'{original_split}-labels-idx1-ubyte.gz').read_bytes())[8:],dtype=np.uint8)
    source_arrays[original_split]=(pixels,labels)
result['fashion_source_sha256']={p.name:sha(p) for p in source.glob('*idx*')}
result['fashion_pixel_matches']=[]
for a in assets[:2]:
    r=next(x for x in selected if x.get('image')==a)
    image=np.asarray(Image.open(DATA/a))
    for i,sid in enumerate(r['supervision']['source_ids']):
        index=int(sid.rsplit(':',1)[-1]);pixels,labels=source_arrays[sid.split(':')[1]];x,y=(14,42) if len(r['supervision']['source_ids'])==1 else (14+56*i,42)
        assert np.array_equal(image[y:y+28,x:x+28],pixels[index])
        result['fashion_pixel_matches'].append({'image':a,'source_id':sid,'source_label':int(labels[index]),'patch_xy':[x,y],'pixel_sha256':hashlib.sha256(pixels[index].tobytes()).hexdigest(),'equal':True})
        Image.fromarray(pixels[index]).save(OUT/'assets'/f'fashion-original-{index}.png')

# Training contract and actual stage metadata (raw measurements only).
result['training_stages']={}
for architecture in ['dense','moe']:
    for stage in ['vision','ocr','audio','joint']:
        f=ROOT/f'docs/selftrained/results/training-raw/{architecture}-{stage}/raw/train-receipt.json'
        r=json.loads(f.read_text());allowed=['stage','architecture','steps','trainable_parameters','total_parameters','freeze_perception_backbones','origin','stage_history','completed_requested_steps','interrupted']
        result['training_stages'][f'{architecture}-{stage}']={k:r[k] for k in allowed}
        result['read_pointers'][str(f)]=['/'+k for k in allowed]
        result['frozen'].append(copy(f,'inputs/training-raw/'+architecture+'-'+stage+'-train-receipt.json'))
from scripts.selftrained.train import set_trainable
fresh=LimitedAssistant(SelftrainedConfig(width=32,layers=1,heads=4,kv_heads=2,ffn_hidden=64))
set_trainable(fresh,'joint',True)
result['frozen_trainability']={name:parameter.requires_grad for name,parameter in fresh.named_parameters()}
assert all(not p.requires_grad for name,p in fresh.named_parameters() if any(name.startswith(k) for k in ['vision_encoder.cnn.','vision_encoder.projection.','vision_encoder.head.','ocr_encoder.cnn.','ocr_encoder.sequence.','ocr_encoder.head.','audio_encoder.temporal.','audio_encoder.sequence.','audio_encoder.head.']))

# Only named aggregate methods/counts, without notes or author interpretations.
result['test_aggregates']={}
for architecture in ['dense','moe']:
    f=ROOT/f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json';r=json.loads(f.read_text())
    allowed=['split','count','expected_count','evaluation_complete','teacher_forcing_used_for_generation','voice_topic_continuation','perception']
    result['test_aggregates'][architecture]={k:r[k] for k in allowed}
    result['read_pointers'][str(f)]=['/'+k for k in allowed]
    result['frozen'].append(copy(f,'inputs/public-raw/'+architecture+'-test-metrics.json'))

# Compare embedded source image bytes; separate Playwright renderer uses these frozen SVGs.
import xml.etree.ElementTree as ET
result['figures']={}
for name in ['fashion','ocr','voice']:
    f=ROOT/f'course/figures/p6-19-modal-{name}.svg';result['frozen'].append(copy(f,'inputs/'+f.name))
    svg=ET.fromstring(f.read_bytes());embedded=[]
    for n in svg.iter():
        if n.tag.endswith('image'):
            url=n.attrib.get('href',n.attrib.get('{http://www.w3.org/1999/xlink}href',''))
            if url.startswith('data:image/png;base64,'):
                data=base64.b64decode(url.split(',',1)[1]);digest=hashlib.sha256(data).hexdigest()
                match=next(a for a in assets[:3] if sha(DATA/a)==digest)
                assert np.array_equal(np.asarray(Image.open(io.BytesIO(data))),np.asarray(Image.open(DATA/match)))
                embedded.append({'sha256':digest,'matches_asset':match,'image_attributes':{k:v for k,v in n.attrib.items() if 'href' not in k}})
    rendered=[]
    for width in [640,360]:
        bodywidth=min(width-32,560);height=int(float(svg.attrib['height'])*bodywidth/560)+34
        html=OUT/'renders'/f'{name}-{width}.html';html.write_text(f'<html><head><meta charset="utf-8"></head><body style="margin:16px;background:white"><img style="display:block;width:100%;max-width:560px;height:auto" src="../inputs/{f.name}"></body></html>')
        png=OUT/'renders'/f'{name}-{width}.png'
        rendered.append({'viewport':[width,height],'png':str(png.relative_to(ROOT)),'html':str(html.relative_to(ROOT))})
    result['figures'][name]={'sha256':sha(f),'embedded':embedded,'renders':rendered}
save('verification.json',result)
print(json.dumps({'section_sha256':result['section_sha256'],'environment':result['environment'],'data_audit':result['data_audit'],'voice_unique_recordings':result['voice_unique_recordings'],'voice_split_overlap':result['voice_split_overlap'],'history_verified':result['history_verified'],'payloads':result['payloads'],'original_voice':result['original_voice'],'fashion_pixel_matches':result['fashion_pixel_matches'],'planned_render_count':6},ensure_ascii=False,indent=2))
