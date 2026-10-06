"""Bounded CPU verification of chapter 19.3's fixed data; no model runs."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from difflib import SequenceMatcher
import ast, copy, gzip, hashlib, importlib.util, json, platform, shutil, sys, tarfile
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
ART = Path(__file__).resolve().parent
DATA = ROOT / 'outputs/selftrained-v2/data'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_bytes())
def output(tag, value): print(tag, json.dumps(value, ensure_ascii=False, sort_keys=True))
def disjoint(registry):
    return {f'{a}/{b}': len(registry[a] & registry[b]) for a,b in combinations(['train','validation','test'],2)}
def registry(rows, key):
    out=defaultdict(set)
    for row in rows:
        value=key(row)
        if value is not None: out[row['split']].add(value)
    return out
manifest=load(ROOT/'docs/selftrained/v2-manifest.json')
base=load(ROOT/'docs/selftrained/manifest.json')
output('environment', {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'})
output('manifest', {'sha256':sha(ROOT/'docs/selftrained/v2-manifest.json'),'records':len(manifest['records']),'assets':len(manifest['assets']),'package':manifest['package']})
from tiny_perceptron.selftrained.dataset import read_records, train_tokenizer, RecordEncoder
from scripts.selftrained.hf_transport import verify_file
paths=[DATA/item['path'] for item in manifest['records']]
mismatches=[]
for item in manifest['records']+manifest['assets']:
    try: verify_file(DATA,item)
    except ValueError as e: mismatches.append({'path':item['path'],'reason':str(e)})
output('local_cache_mismatches',mismatches)
assert all(e['path'] not in {r['path'] for r in manifest['records']} for e in mismatches)
archive=ROOT/manifest['package']['path']
assert sha(archive)==manifest['package']['sha256'] and archive.stat().st_size==manifest['package']['bytes']
expected={item['path']:item for item in manifest['records']+manifest['assets']}
archive_checked=[]
with tarfile.open(archive,'r:gz') as tar:
    for entry in tar:
        if not entry.isfile(): continue
        relative=entry.name.removeprefix('./')
        assert relative in expected, relative
        item=expected[relative];raw=tar.extractfile(entry).read()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'], relative
        archive_checked.append(relative)
assert set(archive_checked)==set(expected) and len(archive_checked)==len(expected)
output('all_manifest_files_verified',{'authoritative_archive_files':len(archive_checked),'matching_local_cache_files':len(expected)-len(mismatches),'archive_sha256':sha(archive),'archive_member_reads_only':True})
rows=read_records(paths)
output('record_counts',dict(Counter(r['split'] for r in rows)))
output('record_files', [{**item,'records':sum(1 for x in (DATA/item['path']).read_bytes().splitlines() if x.strip())} for item in manifest['records']])
output('task_counts',{s:dict(Counter(r['task'] for r in rows if r['split']==s)) for s in ['train','validation','test']})
output('group_cross_split',disjoint(registry(rows,lambda r:r['group_id'])))
text=[r for r in rows if r['task'].startswith(('text','tool'))]
output('template_family_cross_split',disjoint(registry(text,lambda r:r['supervision'].get('template_family'))))
output('operand_cross_split',disjoint(registry(text,lambda r:tuple(sorted((r['supervision']['expected_call']['a'],r['supervision']['expected_call']['b']))) if 'expected_call' in r['supervision'] else None)))
output('text_exact_public_prompt_cross_split',disjoint(registry(text,lambda r:hashlib.sha256(json.dumps(r['messages'][:-1],sort_keys=True,ensure_ascii=False).encode()).hexdigest())))
# V1/V2 identity is independently checked from the two fixed manifests.
v2={item['path']:item for item in manifest['records']+manifest['assets']}
media=[item for item in base['assets'] if item['path'].startswith(('images/','audio/','fonts/'))]
immutable=[item for item in base['records'] if not item['path'].endswith('-train.jsonl')]+media
assert all(v2[item['path']]==item for item in immutable)
prefixes={}
for item in base['records']:
    if item['path'].endswith('-train.jsonl'):
        prefix=(DATA/item['path']).read_bytes()[:item['bytes']]
        prefixes[item['path']]={'v1_bytes':item['bytes'],'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'matches_v1':hashlib.sha256(prefix).hexdigest()==item['sha256']}
assert all(v['matches_v1'] for v in prefixes.values())
output('v1_v2_preservation',{'unchanged_heldout_record_files':8,'unchanged_original_media_files':len(media),'train_prefixes':prefixes,'changed_non_media_notices':[item['path'] for item in base['assets'] if v2.get(item['path'])!=item]})
# Verify original Fashion-MNIST pixels, source assignment and every canvas slot.
vision=[r for r in rows if r['task'].startswith('vision')]
prov=load(ROOT/'docs/selftrained/provenance/vision-sources.json')
sources={s['source_id']:s for s in prov['original_sources']}
pixels={}
for split in ['train','t10k']:
    b=gzip.decompress((DATA/f'source-cache/fashion-mnist/{split}-images-idx3-ubyte.gz').read_bytes())
    pixels[split]=np.frombuffer(b, dtype=np.uint8,offset=16).reshape(-1,28,28)
for source in sources.values():
    assert hashlib.sha256(pixels[source['official_split']][source['original_index']].tobytes()).hexdigest()==source['pixel_sha256']
ids=defaultdict(set); hashes=defaultdict(set); checked=0
for r in vision:
    with Image.open(DATA/r['image']) as im:
        for source_id,label,box in zip(r['supervision']['source_ids'],r['supervision']['vision_labels'],r['image_layout']['slots']):
            s=sources[source_id]; assert s['split']==r['split'] and s['local_label']==label
            expected=Image.new('L',(box[2]-box[0],box[3]-box[1]),0)
            crop=pixels[s['official_split']][s['original_index']]
            expected.paste(Image.fromarray(crop),((expected.width-28)//2,(expected.height-28)//2))
            assert im.crop(box).tobytes()==expected.tobytes()
            checked+=1; ids[r['split']].add(source_id); hashes[r['split']].add(s['pixel_sha256'])
output('vision_source_check',{'sources_per_split':{s:len(v) for s,v in ids.items()},'source_id_cross_split':disjoint(ids),'original_pixel_cross_split':disjoint(hashes),'verified_slots':checked})
controls=defaultdict(list)
for r in vision:
    if 'pair_id' in r['supervision']: controls[r['supervision']['pair_id']].append(r)
assert all(len(v)==2 and len({r['split'] for r in v})==1 and len({r['messages'][-1]['content'] for r in v})==2 for v in controls.values())
output('vision_controls',{'swap_pairs':len(controls),'both_orders_change_target':True,'targets':{s:dict(Counter(r['supervision']['vision_labels'][r['supervision']['query_slot']] for r in vision if r['split']==s)) for s in ['train','validation','test']}})
# OCR target strings, known single characters, actual ROI geometry and fonts.
ocr=[r for r in rows if r['task']=='ocr']
compositions=registry(ocr,lambda r:r['supervision']['ocr_text'] if len(r['supervision']['ocr_text'])>1 else None)
singles=registry(ocr,lambda r:r['supervision']['ocr_text'] if len(r['supervision']['ocr_text'])==1 else None)
for r in ocr:
    roi=r['roi']; target=r['supervision']['ocr_text']
    assert 1<=len(target)<=4 and (roi[2]-roi[0],roi[3]-roi[1])==(len(target)*36,36)
    assert 0<=roi[0]<roi[2]<=192 and 0<=roi[1]<roi[3]<=96
    assert target==r['messages'][-1]['content']
    assert all(roi[0]<=b[0]<b[2]<=roi[2] and roi[1]<=b[1]<b[3]<=roi[3] for b in r['supervision']['glyph_boxes'])
output('ocr',{'records_verified':len(ocr),'composition_strings':{s:len(v) for s,v in compositions.items()},'composition_cross_split':disjoint(compositions),'shared_single_characters':''.join(sorted(set.intersection(*singles.values()))),'single_counts':{s:len(v) for s,v in singles.items()},'font_families':{s:dict(Counter(r['supervision']['font_family'] for r in ocr if r['split']==s)) for s in ['train','validation','test']},'image_cross_split':disjoint(registry(ocr,lambda r:r['image'])),'render_source_cross_split':disjoint(registry(ocr,lambda r:r['supervision']['render_source_id']))})
ocr_images=registry(ocr,lambda r:r['image'])
image_pixels=defaultdict(set)
for s,images in ocr_images.items():
    for f in images:
        with Image.open(DATA/f) as im: image_pixels[s].add(hashlib.sha256(im.tobytes()).hexdigest())
output('ocr_pixel_cross_split',disjoint(image_pixels))
# Raw source audit is small enough to retain unchanged, with an exact copy check.
audit_path=DATA/'voice-audit.jsonl'
copy_path=ART/'voice-audit.jsonl'
shutil.copyfile(audit_path,copy_path)
assert sha(audit_path)==sha(copy_path)
audit=[json.loads(x) for x in audit_path.read_bytes().splitlines()]
selected=[r for r in audit if r['selected']]
output('voice_originals',{'source_audit_sha256':sha(audit_path),'per_split':dict(Counter(r['prepared_split'] for r in selected)),'selected_recordings':len(selected),'speaker_ids_all_null':all(r['speaker_id'] is None for r in audit),'session_ids_all_null':all(r['session_id'] is None for r in audit),'source_revision':selected[0]['source_revision']})
voice=[r for r in rows if r['task'].startswith('voice')]
lookup={r['audio']:r for r in selected}; parent_ids={r['id']:r for r in voice if 'augmentation' not in r}
for r in voice:
    parent_audio=r.get('augmentation',{}).get('parent_audio',r['audio']); a=lookup[parent_audio]
    assert r['split']==a['prepared_split'] and r['group_id']==a['group_id']
    if 'augmentation' in r:
        parent=parent_ids[r['augmentation']['parent_id']]
        assert all(r[k]==parent[k] for k in ['group_id','split','task','messages','supervision']) and r['split']=='train'
        assert not r['augmentation']['new_independent_human_recording']
original_voice=[dict(r,split=r['prepared_split']) for r in selected]
output('voice_family_cross_split',disjoint(registry(original_voice,lambda r:r['group_id'])))
output('voice_original_audio_cross_split',disjoint(registry(original_voice,lambda r:r['audio_sha256'])))
source_file=DATA/'voice-sources/minds14-zh-CN.parquet'
output('source_parquet',{'sha256':sha(source_file),'columns':pq.read_schema(source_file).names})
spec=importlib.util.spec_from_file_location('prepare_voice_raw',ROOT/'scripts/selftrained/prepare_voice.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
assert sha(source_file)==mod.PARQUET_SHA256 and source_file.stat().st_size==mod.PARQUET_SIZE
violations=[]; edges=0
for a,b in combinations(selected,2):
    x,y=mod.normalized_transcript(a['transcription']),mod.normalized_transcript(b['transcription'])
    exact=x==y; near=min(len(x),len(y))>=6 and min(len(x),len(y))/max(len(x),len(y))>=.8
    duplicate=exact or (near and SequenceMatcher(None,x,y,autojunk=False).ratio()>=.9)
    if duplicate:
        edges+=1
        if a['group_id']!=b['group_id'] or a['prepared_split']!=b['prepared_split']: violations.append([a['source_row'],b['source_row']])
assert not violations
output('voice_lexical_duplicate_verification',{'selected_pair_checks':len(selected)*(len(selected)-1)//2,'duplicate_edges':edges,'group_or_split_violations':violations})
import soundfile as sf
waves=[sf.read(DATA/a['audio'],dtype='float32')[0] for a in selected]
acoustic_edges=mod.acoustic_duplicate_edges(list(range(len(selected))),waves)
assert all(selected[e['rows'][0]]['group_id']==selected[e['rows'][1]]['group_id'] and selected[e['rows'][0]]['prepared_split']==selected[e['rows'][1]]['prepared_split'] for e in acoustic_edges)
output('voice_acoustic_duplicates',{'pair_candidates':len(selected)*(len(selected)-1)//2,'near_identical_edges':len(acoustic_edges),'group_or_split_violations':0,'threshold':0.995})
spec=importlib.util.spec_from_file_location('prepare_text_raw',ROOT/'scripts/selftrained/prepare_text_tools.py'); textmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(textmod)
filtered,overlap=textmod.audit_voice_overlap(text,audit_path)
assert len(filtered)==len(text)
output('text_voice_lexical_overlap', {k:overlap[k] for k in ['status','voice_audit_sha256','voice_test_utterances','train_validation_unique_user_queries','near_duplicate_threshold','maximum_similarity_before_exclusion','excluded_records','collisions']})
# Execute the reader's rejection and metadata exclusion contracts on CPU.
negative=copy.deepcopy(rows[:2]); negative[1]['group_id']=negative[0]['group_id']; negative[1]['split']='test' if negative[0]['split']!='test' else 'train'
negative_path=ART/'cross-split-negative.jsonl'; negative_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in negative))
try: read_records([negative_path])
except ValueError as e: output('reader_negative',str(e)); assert 'group 跨 split' in str(e)
else: raise AssertionError('reader accepted cross-split group')
tokenizer=train_tokenizer(rows)
encoder=RecordEncoder(tokenizer,DATA,context=8192,device='cpu')
checks=[]
for r in [next(x for x in ocr if x['split']=='test'),next(x for x in voice if x['split']=='test' and x['task']=='voice_qa')]:
    before=encoder.encode(r,generation=True)
    altered=copy.deepcopy(r); altered['supervision']={'intent':'NEVER_PUBLIC_LABEL','ocr_text':'NEVER_PUBLIC_TARGET'}; altered['transcription']='NEVER_PUBLIC_TRANSCRIPT'; altered['messages'][-1]['content']='NEVER_PUBLIC_CURRENT_ANSWER'
    after=encoder.encode(altered,generation=True)
    assert before['input_ids']==after['input_ids']
    assert all(torch.equal(a['values'],b['values']) for a,b in zip(before['modalities'],after['modalities']))
    checks.append({'task':r['task'],'prompt_tokens':len(before['input_ids']),'modality_shape':list(before['modalities'][0]['values'].shape),'metadata_and_current_target_excluded':True})
output('encoder_checks',checks)
# Execute the unmodified continuation branch with a deterministic stub, not a model.
evtree=ast.parse((ROOT/'scripts/selftrained/evaluate.py').read_text())
branch=next(n for n in ast.walk(evtree) if isinstance(n,ast.If) and ast.unparse(n.test)=="record.get('evaluation', {}).get('mode') == 'voice_topic_continuation'")
class StubGenerator:
    def __init__(self): self.calls=[]; self.messages=[]
    def generate(self,record,messages):
        self.messages.append(copy.deepcopy(messages)); self.calls.append({'generation_status':'generated'})
        return 'SELF_GENERATED_FIRST' if len(self.messages)==1 else 'SELF_GENERATED_FINAL'
record=copy.deepcopy(next(x for x in voice if x['task']=='voice_topic_continuation' and x['split']=='train'))
gold_index=record['evaluation']['first_target_message_index'];record['messages'][gold_index]['content']='GOLD_NEVER_PROMPT'
record['messages'][-1]['content']='FINAL_GOLD_NEVER_PROMPT'
stub=StubGenerator();namespace={'record':record,'generator':stub,'score_reply':lambda *x:{'semantic':False,'format':False}}
exec(compile(ast.fix_missing_locations(ast.Module(body=branch.body,type_ignores=[])),'evaluate.py:voice-continuation','exec'),namespace)
assert len(stub.messages)==2 and all('GOLD_NEVER_PROMPT' not in json.dumps(m) for m in stub.messages)
assert stub.messages[1][-2]=={'role':'assistant','content':'SELF_GENERATED_FIRST'}
output('continuation_contract',{'executed_original_branch_lines':[branch.lineno,branch.end_lineno],'calls':2,'second_turn_uses_actual_generated_first_reply':True,'gold_reference_excluded_from_both_prompts':True,'model_evaluation':False})
from tiny_perceptron.selftrained.inference import _public_history
try: _public_history([{'role':'user','content':'hello','intent':'private'}])
except ValueError as e: output('public_input_rejection',str(e))
else: raise AssertionError('public history accepted private intent')
# Original training provenance: read only measurement/identity pointers.
stage_evidence=[]
for p in sorted((ROOT/'docs/selftrained/results/training-raw').glob('*/raw/execution.json')):
    d=load(p); readiness=d['data_readiness']
    stage_evidence.append({'path':str(p.relative_to(ROOT)),'source_sha256':sha(p),'pointers':['/manifest_sha256','/status','/returncode','/data_readiness/status','/data_readiness/record_files_verified','/data_readiness/asset_files_verified','/data_readiness/archive_sha256'],'manifest_sha256':d['manifest_sha256'],'status':d['status'],'returncode':d['returncode'],'data_ready_status':readiness['status'],'record_files':readiness['record_files_verified'],'asset_files':readiness['asset_files_verified'],'archive_sha256':readiness['archive_sha256']})
assert len(stage_evidence)==15 and all(e['manifest_sha256']==sha(ROOT/'docs/selftrained/v2-manifest.json') and e['record_files']==12 and e['asset_files']==len(manifest['assets']) for e in stage_evidence)
output('training_stage_frozen_data',stage_evidence)
receipts=[]
for p in sorted((ROOT/'docs/selftrained/results/training-raw').glob('*/raw/train-receipt.json')):
    d=load(p); assert d['test_used_for_selection'] is False
    receipts.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'pointers':['/test_used_for_selection','/data_sha256','/selection','/train_records','/validation_records'],'test_used_for_selection':d['test_used_for_selection'],'record_fingerprints_match':all(d['data_sha256'][item['path']]==item['sha256'] for item in manifest['records']),'selection':d['selection'],'train_records':d['train_records'],'validation_records':d['validation_records']})
assert all(r['record_fingerprints_match'] for r in receipts)
output('training_selection_receipts',receipts)
for p in sorted((ROOT/'docs/selftrained/results/public-raw').glob('*/freeze/frozen.json')):
    d=load(p); assert all(d['data_sha256'][item['path']]==item['sha256'] for item in manifest['records'])
    output('frozen_test_protocol',{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'pointers':['/data_sha256','/selection','/test_once','/prompt_policy'],'fixed_record_fingerprints_match':True,'selection':d['selection'],'test_once':d['test_once'],'prompt_policy':d['prompt_policy']})
for p in sorted((ROOT/'docs/selftrained/results/public-raw').glob('*/test/metrics.json')):
    d=load(p); output('original_test_count',{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'pointers':['/split','/count','/expected_count','/limited_smoke','/evaluation_complete'],'split':d['split'],'count':d['count'],'expected_count':d['expected_count'],'limited_smoke':d['limited_smoke'],'evaluation_complete':d['evaluation_complete']})
output('verification_complete',True)
