"""Independent CPU review of lesson 20.3 and its fixed data snapshot."""
import collections
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

from PIL import Image
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/technical-reviews/artifacts'
EXPECTED = '7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91'
splits = ('train', 'validation', 'test')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return json.loads((ROOT / path).read_text())
def save(name,value):
    (OUT / ('natural-20.3-'+name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def intersections(groups):
    return {a+'/'+b:sorted(groups[a] & groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
def safe(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts

chapter=(ROOT/'course/chapters/20.md').read_bytes().decode('utf-8')
start=re.search(r'^## 20\.3 ',chapter,re.M).start()
next_heading=re.search(r'^## ',chapter[start+1:],re.M)
body=chapter[start:start+1+next_heading.start()] if next_heading else chapter[start:]
(OUT/'natural-20.3-section.md').write_bytes(body.encode('utf-8'))
block=re.findall(r'```python\n(.*?)\n```',body,re.S)
assert len(block)==1
code=block[0]+'\n'
exercise=code.replace('{"family": "公園照片A", "split": "train", "question": "圖裡有腳踏車嗎？"}', '{"family": "公園照片A", "split": "test", "question": "圖裡有腳踏車嗎？"}')
assert exercise!=code
env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
executions=[]
for label,script,expected in [
    ('example',code,"訓練家族 ['公園照片A']\n測試家族 ['招牌照片B']\n跨份重疊 []\n"),
    ('exercise',exercise,"訓練家族 ['公園照片A']\n測試家族 ['公園照片A', '招牌照片B']\n跨份重疊 ['公園照片A']\n")]:
    p=OUT/f'natural-20.3-{label}.py'; p.write_text(script)
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(p)],cwd=ROOT,env=env,text=True,capture_output=True)
    assert run.returncode==0 and run.stdout==expected and run.stderr=='',(label,run)
    executions.append({'label':label,'command':'.venv/bin/python '+str(p.relative_to(ROOT)),'source_sha256':sha(p.read_bytes()),'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'expected':expected,'exact_output_match':True})

manifest_path=ROOT/'docs/natural-assistant/manifest.json'
raw=manifest_path.read_bytes(); assert sha(raw)==EXPECTED
m=json.loads(raw)
specs={x['path']:x for x in m['files']}; assert len(specs)==len(m['files'])==345
disk=[]
for name,spec in sorted(specs.items()):
    safe(name); content=(ROOT/'data/natural'/name).read_bytes()
    assert len(content)==spec['bytes'] and sha(content)==spec['sha256'],name
    disk.append({'path':name,'bytes':len(content),'sha256':sha(content)})
archives=[]; archive_member_names=[]
for a in m['archives']:
    path=ROOT/a['path']; content=path.read_bytes()
    assert len(content)==a['bytes'] and sha(content)==a['sha256']
    declared={x['path']:x for x in a['files']}; assert len(declared)==len(a['files'])
    actual={}
    with tarfile.open(path,'r:gz') as tar:
        for member in tar:
            safe(member.name)
            if member.isdir(): continue
            assert member.isfile() and member.name not in actual and member.name in declared
            content=tar.extractfile(member).read(); spec=declared[member.name]
            assert len(content)==spec['bytes'] and sha(content)==spec['sha256']
            assert specs[member.name]==spec
            actual[member.name]={'path':member.name,'bytes':len(content),'sha256':sha(content)}
    assert set(actual)==set(declared)
    archives.append({'path':a['path'],'archive_sha256':a['sha256'],'bytes':a['bytes'],'regular_members':len(actual),'all_member_hashes_match':True})
    archive_member_names.extend(actual)
assert len(archive_member_names)==len(set(archive_member_names))==345 and set(archive_member_names)==set(specs)

groups={s:{r['family'] for r in m['rows']+m['audio_rows'] if r['split']==s} for s in splits}
family_overlaps=intersections(groups); assert not any(family_overlaps.values())
tasks={s:dict(collections.Counter(r['task'] for r in m['rows'] if r['split']==s)) for s in splits}
photo_rows=[r for r in m['rows'] if r['task']=='scene']
ocr_rows=[r for r in m['rows'] if r['task'] in ('ocr','text_presence')]
image_groups={s:{r['image'] for r in photo_rows+ocr_rows if r['split']==s} for s in splits}
image_path_overlaps=intersections(image_groups); assert not any(image_path_overlaps.values())
byte_groups={s:{specs[n]['sha256'] for n in image_groups[s]} for s in splits}
byte_overlaps=intersections(byte_groups); assert not any(byte_overlaps.values())
pixel_groups={s:set() for s in splits}; pixels=[]
for split,names in image_groups.items():
    for name in sorted(names):
        with Image.open(ROOT/'data/natural'/name) as im:
            rgb=im.convert('RGB'); pixel_digest=sha((str(rgb.size)+'RGB').encode()+rgb.tobytes())
            pixel_groups[split].add(pixel_digest)
            pixels.append({'path':name,'split':split,'dimensions':list(rgb.size),'rgb_sha256':pixel_digest})
pixel_overlaps=intersections(pixel_groups); assert not any(pixel_overlaps.values())

photos={s:len({r['image'] for r in photo_rows if r['split']==s}) for s in splits}
photo_questions={s:sum(r['split']==s for r in photo_rows) for s in splits}
ocr_images={s:len({r['image'] for r in ocr_rows if r['split']==s}) for s in splits}
ocr_questions={s:sum(r['split']==s for r in ocr_rows) for s in splits}
assert photos==dict(zip(splits,[120,12,12])) and photo_questions==dict(zip(splits,[120,36,36]))
assert ocr_images==dict(zip(splits,[108,10,21])) and ocr_questions==dict(zip(splits,[120,12,24]))
originals={r['example_id']:r for r in map(json.loads,(ROOT/'data/natural/vision/sources/docci-descriptions.jsonl').read_text().splitlines())}
train_caption_checks=[]
sentence_wording_exceptions=[]
for r in photo_rows:
    orig=originals[r['source']['original_id']]
    expected_split={'train':'train','validation':'qual_dev','test':'test'}[r['split']]
    assert orig['split']==r['source']['official_split']==expected_split
    if r['split']=='train':
        description=orig['description'].strip()
        # First end-of-sentence punctuation followed by whitespace, or the end.
        boundary=re.search(r'[.!?](?=\s|$)',description)
        assert boundary is not None
        expected=description[:boundary.end()]
        assert r['answer']==expected,(r['id'],r['answer'],expected)
        quoted_boundary=re.search(r'''[.!?]["”']?(?=\s|$)''',description)
        first_complete=description[:quoted_boundary.end()]
        if r['answer'] != first_complete:
            sentence_wording_exceptions.append({'id':r['id'],'first_complete_sentence':first_complete,'actual_target':r['answer'],'cause':'Generator splits only punctuation immediately followed by whitespace; punctuation followed by a closing quote is skipped.'})
        train_caption_checks.append({'id':r['id'],'answer':r['answer'],'matches_original_punctuation_whitespace_prefix':True,'matches_quote_aware_first_sentence':r['answer']==first_complete})
    else:
        assert r['source']['target_author']=='AI reader after direct image inspection and complete official caption reading'
        assert r['source']['target_frozen_before_model_inference'] is True
counts_per_photo=collections.Counter(r['image'] for r in photo_rows)
assert all(n==1 for r,n in counts_per_photo.items() if r.startswith('vision/images/train_'))
assert all(n==3 for r,n in counts_per_photo.items() if not r.startswith('vision/images/train_'))
ocr=load('data/natural/ocr/manifest.json'); image_specs={x['image']:x for x in ocr['images']}
for r in ocr['rows']:
    info=image_specs[r['image']]
    if r['task']=='ocr': assert r['answer']=='\n'.join(b['text'] for b in info['boxes'])
    else: assert r['answer']==('有' if info['boxes'] else '沒有')
    if info['boxes']: assert info['font_source']==('noto-serif-tc' if r['split']=='test' else 'noto-sans-tc')
chat=[r for r in m['rows'] if r['task']=='text_chat' and r['split']=='train']
chat_families=collections.Counter(r['family'] for r in chat)
assert len(chat)==30 and len(chat_families)==10 and set(chat_families.values())=={3}
assert len({(r['user'],r['answer']) for r in chat})==10
for family in chat_families:
    assert len({json.dumps({k:v for k,v in r.items() if k!='id'},sort_keys=True,ensure_ascii=False) for r in chat if r['family']==family})==1
totals={s:sum(r['split']==s for r in m['rows']) for s in splits}; assert totals==dict(zip(splits,[272,52,66]))
audio_counts={s:sum(r['split']==s for r in m['audio_rows']) for s in splits}; assert audio_counts==dict(zip(splits,[24,6,12]))
audio_hashes={s:set() for s in splits}; audio_checks=[]
tables={}
for source_split in ('train','dev','test'):
    table={}
    for line in (ROOT/f'data/natural/speech/official-sources/{source_split}.tsv').read_text().splitlines():
        parts=line.split('\t'); assert len(parts)==7; table[parts[1]]=parts
    tables[source_split]=table
for r in m['audio_rows']:
    data=(ROOT/'data/natural'/r['audio']).read_bytes(); header=sf.info(io.BytesIO(data))
    assert r['speaker'] is None and r['synthetic'] is False and r['answer'] is None
    assert r['source_split']=={'train':'train','validation':'dev','test':'test'}[r['split']]
    p=tables[r['source_split']][Path(r['audio']).name]
    assert r['raw_transcription']==p[2] and r['normalized_transcription']==p[3]
    assert r['family']=='speech:fleurs-cmn-sentence-'+p[0]
    assert header.frames==int(p[5])==r['num_samples'] and header.channels==1 and header.samplerate==16000
    assert sha(data)==r['sha256'] and len(data)==r['bytes']
    audio_hashes[r['split']].add(sha(data))
    audio_checks.append({'id':r['id'],'source_split':r['source_split'],'frames':header.frames,'sample_rate':header.samplerate,'channels':header.channels,'wav_sha256':sha(data),'matches_original_transcript':True})
audio_overlaps=intersections(audio_hashes); assert not any(audio_overlaps.values())

receipts=[]
for kind,path,key,prefix,expected_generator in [
    ('vision','docs/natural-assistant/evidence/data/vision/original-reconstruction-proof.json','file_digest_map','vision/','scripts/prepare_natural_vision.py'),
    ('ocr','docs/natural-assistant/evidence/data/ocr/original-generator-verification.json','full_file_sha256','ocr/','scripts/prepare_natural_ocr.py'),
    ('speech','docs/natural-assistant/evidence/data/speech/original-reproducibility-check.json','files','speech/',None)]:
    proof=load(path); hashes=proof[key]
    for name,digest in hashes.items(): assert specs[prefix+name]['sha256']==digest,(kind,name)
    assert set(prefix+n for n in hashes)=={n for n in specs if n.startswith(prefix)}
    if expected_generator:
        recorded=proof.get('generator_sha256',proof.get('source_sha256'))
        assert recorded==sha((ROOT/expected_generator).read_bytes())
    receipts.append({'kind':kind,'receipt':path,'receipt_sha256':sha((ROOT/path).read_bytes()),'all_declared_receipt_hashes_match_current_snapshot':True,'matched_files':len(hashes),'scope':'Existing reconstruction receipt inspected and matched; independent current audit does not claim to have rerun that old network operation.'})

environment={'python':sys.version,'pillow':importlib.metadata.version('pillow'),'soundfile':importlib.metadata.version('soundfile'),'torch':importlib.metadata.version('torch'),'platform':platform.platform(),'device':'CPU; CUDA_VISIBLE_DEVICES empty; no GPU operations or model inference'}
result={'reviewer_task':'/root/natural_factual_20_3','manifest_sha256':sha(raw),'source_sha256':sha(body.encode()),'environment':environment,'example_executions':executions,'files_read_and_hashed':len(disk),'archive_results':archives,'row_totals':totals,'tasks_by_split':tasks,'photo_images':photos,'photo_questions':photo_questions,'ocr_images':ocr_images,'ocr_questions':ocr_questions,'audio_counts':audio_counts,'train_text_chat_families':dict(chat_families),'distinct_train_text_chat_pairs':10,'family_overlaps':family_overlaps,'image_path_overlaps':image_path_overlaps,'image_byte_overlaps':byte_overlaps,'image_pixel_overlaps':pixel_overlaps,'audio_byte_overlaps':audio_overlaps,'train_original_punctuation_whitespace_prefix_matches':len(train_caption_checks),'sentence_wording_exceptions':sentence_wording_exceptions,'heldout_questions_with_explicit_AI_authorship':72,'audio_seconds':sum(r['duration_seconds'] for r in m['audio_rows']),'existing_reconstruction_receipts':receipts,'limits':['No model quality, GPU speed, memory, completed training, or chapter-wide results assessed.','Exact family IDs, asset bytes, and decoded RGB pixels checked; near-duplicate scenes and unseen upstream pretraining cannot be ruled out.','AI authorship and before-inference freezing are supported by preserved documentary provenance and static generators, not a new human annotation audit.','Individual FLEURS speaker identities remain unavailable; sentence and WAV disjointness is not dev/test speaker disjointness.']}
save('execution',result)
save('all-files',{'disk_files':disk,'pixels':pixels,'audio_checks':audio_checks,'train_caption_checks':train_caption_checks})
save('derivation',{'original_example':'Train families={公園照片A}; test families={招牌照片B}; intersection={} => sorted=[]','exercise':'Second row moved to test; train={公園照片A}; test={公園照片A,招牌照片B}; intersection={公園照片A}','photo_counts':'120+12+12=144 images; 120×1=120 train questions; 12×(1+2)=36 validation and 36 test questions','ocr_counts':'108+10+21=139 images; 120+12+24=156 questions, because positive text_presence rows reuse an OCR image. Questions are not independent image counts.','chat_replay':'10 independent train (user,answer) pairs ×3 identical copies =30 rows; plus two multiround dialogues =32 text train rows','totals':'train=120 photo+96 OCR+24 text_presence+30 chat+2 dialogue=272; validation=36+9+3+3+1=52; test=36+18+6+4+2=66','audio_separate':'24+6+12=42 recordings; answer=null; excluded from m.rows. run_train selects only m.rows where split=train; no audio_rows are fed to the LoRA loop.','archive_counts':'154 vision+143 OCR+48 speech=345 distinct regular archive members; equals 345 declared disk files','leakage_boundaries':'Family-name intersection is an identity constraint. File SHA checks exact serialized content; dimensions plus RGB SHA check exact decoded pixel content. None is a near-duplicate detector or proof of absent unknown upstream pretraining.'})
print(json.dumps(result,ensure_ascii=False,indent=2))
