"""Bounded CPU review of current lesson 20.4 and its necessary prerequisites."""
from pathlib import Path
import base64
import hashlib
import io
import json
import platform
import sys
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from tiny_perceptron.natural_concepts import picture_order_report, audio_order_report
from tiny_perceptron.multimodal import tone, log_mel

def read(name):
    return json.loads((ROOT / name).read_text())

def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

torch.set_num_threads(1)
manifest = read('docs/natural-assistant/v4/manifest.json')
raw = ROOT / 'outputs/natural-v4/factual-research/20.4'
cat_rows = [r for r in manifest['rows'] if r.get('image') == 'vision/images/train_01827.jpg']
assert len(cat_rows) == 2 and all(r['split'] == 'train' for r in cat_rows)
captions = [json.loads(l) for l in (raw / 'docci-descriptions.raw').read_text().splitlines()]
original = next(r for r in captions if r['example_id'] == 'train_01827')
assert original['split'] == 'train'
assert all(r['references']['official_description'] == original['description'] for r in cat_rows)
assert all(r['source']['label_kind'].startswith('AI-authored') for r in cat_rows)
assert all(r['source']['image_review']['label_author'] == r['source']['label_author'] for r in cat_rows)
assert all('train_01827' in r['source']['image_review']['viewed_image_ids'] for r in cat_rows)
view_file = ROOT / cat_rows[0]['source']['image_review']['artifact']
assert hashlib.sha256(view_file.read_bytes()).hexdigest() == cat_rows[0]['source']['image_review']['artifact_sha256']
svg = ET.parse(ROOT / 'course/figures/natural-v4-training-cat.svg').getroot()
im = svg.find('{http://www.w3.org/2000/svg}image')
href = next(v for k,v in im.attrib.items() if k.endswith('href'))
jpeg = base64.b64decode(href.split(',',1)[1])
assert jpeg == (raw / 'docci-cat.jpg').read_bytes()
assert hashlib.sha256(jpeg).hexdigest() == cat_rows[0]['source']['image_sha256']
dimensions = Image.open(io.BytesIO(jpeg)).size
assert dimensions[0] / dimensions[1] == float(im.attrib['width']) / float(im.attrib['height'])

# Independent exact construction, separate from the prerequisite report function.
a = torch.zeros(3,32,32)
a[0,8:16,:4] = 1
a[2,8:16,4:8] = 1
b = a.clone()
b[:,:,:8] = a[:,:,:8].flip(-1)
manual_pool = lambda x: x.reshape(3,4,8,4,8).mean((2,4))
picture = picture_order_report()
assert picture == {'different_pixels':128,'same_4_by_4_summary':True,'same_pixel_sequence':False}
assert int((a != b).sum()) == 8*8*2 == 128
assert torch.equal(manual_pool(a),manual_pool(b))
audio = audio_order_report()
assert audio['time_frames'] == 21 and audio['bands'] == 16
assert not audio['same_time_sequence'] and audio['same_mean_with_roundoff_tolerance']
sequence = log_mel(torch.cat([tone(440,seconds=.1),tone(880,seconds=.1)])).T
roundoff = float((sequence.mean(0)-sequence.flip(0).mean(0)).abs().max())
assert sequence.shape == (21,16)

ocr = read('docs/natural-assistant/v4/data/ocr-sources.json')
sources = {r['source_id']:r for r in ocr['sources']}
labels = [json.loads(l) for l in (ROOT/'docs/natural-assistant/v4/data/ocr-labels.jsonl').read_text().splitlines()]
nvidia = [r for r in labels if r['source_id'].startswith('nvidia-')]
annotation_receipts = {}
for row in nvidia:
    sid = row['source_id']
    file = ROOT / 'docs/natural-assistant/evidence/v4-research/ocr/nvidia' / (sid+'.annotations.json')
    annotation = json.loads(file.read_text())
    annotation_receipts[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    texts = [annotation['line_bboxes'][i]['text'] for i in row['source_line_indices']]
    assert '\n'.join(texts) == row['answer'] == row['text'], row['id']
    assert row['synthetic'] is True and row['split'] == 'train'
    assert row['source_dataset_revision'] == ocr['nvidia']['revision']
    assert row['source_image_sha256'] == sources[sid]['image_sha256']
assert len(nvidia) == 827

fleurs = [r for r in manifest['audio_rows'] if r['source'] == 'fleurs-mandarin-v4']
aishell = [r for r in manifest['audio_rows'] if r['source'] == 'aishell1-human-read-questions-v4']
assert len(fleurs) == 30 and len(aishell) == 8
assert all(r['task'] == 'speech_transcription' and r['answer'] is None and not r['synthetic'] for r in fleurs)
assert all(r['task'] == 'speech_chat' and r['answer'] is None and r['semantic_rubric'] and not r['synthetic'] for r in aishell)
assert all(r['user'] == ''.join(r['original_spaced_source_transcription'].split()) for r in aishell)
assert not any(r['split'] == 'train' for r in manifest['audio_rows'])
oai = [r for r in manifest['rows'] if r['task'] == 'chat' and r['family'].startswith('oasst2:')]
if not oai:
    oai = [r for r in manifest['rows'] if r['task'] == 'chat' and 'oasst' in str(r.get('source',{})).lower()]
family_splits = {}
for row in oai:
    family_splits.setdefault(row['family'],set()).add(row['split'])
assert len(oai) == 64 and len(family_splits) == 60
assert all(len(splits) == 1 for splits in family_splits.values())
assert all(r['user'] and r['answer'] for r in oai)
message_file = ROOT / 'docs/natural-assistant/evidence/v4-research/chat-speech/oasst2-selected-original-messages.jsonl'
messages = {r['message_id']:r for r in map(json.loads,message_file.read_text().splitlines())}
for row in oai:
    path = [messages[i] for i in row['source_message_ids']]
    assert path[0]['parent_id'] is None and path[-1]['role'] == 'assistant'
    assert all(m['lang'] == 'zh' and m['text'] for m in path)
    assert all(path[i]['parent_id'] == path[i-1]['message_id'] for i in range(1,len(path)))
    assert all(m['message_tree_id'] == row['source_tree_id'] for m in path)

# Directly inspect original project model-generation records only for task routing.
# No scores/verdicts or prior review reports are consumed and no inference is run.
fixed_routes = []
for split, directory, generations in [
    ('validation','outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review','generations-base.json'),
    ('test','outputs/natural-v4/modal-runs/evaluate-37221188153/natural-natural-v4-evaluate-37221188153-1/review','generations.json')]:
    gen_path = directory + '/' + generations
    trans_path = directory + '/transcripts.json'
    records = read(gen_path)
    transcripts = {r['id']:r for r in read(trans_path)}
    selected = [r for r in aishell if r['split'] == split]
    for row in selected:
        typed = next(r for r in records if r['id'] == row['id'] and r['task'] == 'typed_chat')
        spoken = next(r for r in records if r['id'] == row['id'] and r['task'] == 'speech_chat')
        assert typed['user'] == row['user']
        assert spoken['user'] == spoken['transcript'] == transcripts[row['id']]['transcript']
        assert spoken['reference_user'] == row['user']
    assert not any(r['id'] in {q['id'] for q in fleurs} for r in records)
    fixed_routes.append({'split':split,'aishell_recordings':len(selected),'typed_and_actual_asr_routes':2*len(selected),'generation_source':gen_path,'generation_source_sha256':digest(gen_path),'transcription_source':trans_path,'transcription_source_sha256':digest(trans_path),'scope':'Original project records establish the claimed route use, not response correctness or fresh model/ASR reproduction.'})

receipt = {
    'environment': {'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':str(torch.get_num_threads())},
    'cat': {'photos':1,'project_rows':2,'official_and_project_split':'train','embedded_jpeg_sha256':hashlib.sha256(jpeg).hexdigest(),'dimensions':list(dimensions),'jpeg_exactly_equal_to_fixed_original':True,'original_caption_exactly_matches_both_rows':True,'authorship':'Both rows record AI-authored Traditional-Chinese adaptations with matching source-view provenance; no independent proof of historical mental acts is claimed.'},
    'prerequisites': {'picture':picture,'independent_difference_derivation':'8 rows * 8 columns * 2 changed RGB channels = 128 channel values; 64 color positions. Each pooled 8x8 cell has unchanged red/blue sums.','audio':audio,'independent_frames_derivation':'Two 0.1s tones at 16000Hz give 3200 samples. Centered STFT with hop160 gives floor(3200/160)+1=21 frames. Filter bank has16 bands.','maximum_mean_roundoff':roundoff},
    'nvidia': {'matched_original_line_transcriptions':len(nvidia),'selected_pages':len(annotation_receipts),'single_line_crops':sum(len(r['source_line_indices']) == 1 for r in nvidia),'multiline_crops':sum(len(r['source_line_indices']) > 1 for r in nvidia),'all_synthetic_and_training_only':True,'annotation_file_receipts':annotation_receipts,'scope':'Frozen original annotation snapshots, not fresh HDF5 acquisition or full-shard verification.'},
    'source_use': {'fleurs':dict(Counter(r['split'] for r in fleurs)),'aishell':dict(Counter(r['split'] for r in aishell)),'fleurs_no_assistant_answers':True,'aishell_project_semantic_rubrics':True,'audio_training_rows':0,'oasst_rows':len(oai),'oasst_tree_families':len(family_splits),'tree_family_split_isolation':True,'original_zh_paths_complete':True,'original_messages_snapshot_sha256':hashlib.sha256(message_file.read_bytes()).hexdigest(),'original_project_paired_routes':fixed_routes},
    'duplication': {'derivation':'Repeating one identical request/answer three times gives three occurrences and one unique scenario. If the original is kept and three additional copies are added, four occurrences still give one unique scenario. Under uniform sampling this raises this scenario weight relative to unrepeated examples.','three_occurrences':len(['same']*3),'unique_scenarios':len(set(['same']*3))},
    'limits':'No model loaded, training, ASR inference, GPU, download of audio/model/shards, package installation, environment edits or wholebook rerun. The lesson has no measured quality/speed claims. Corpus role audits do not measure assistant competence.'
}
print(json.dumps(receipt,ensure_ascii=False,indent=2))
