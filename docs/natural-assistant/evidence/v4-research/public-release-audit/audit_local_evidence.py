"""Read-only release evidence audit. Does not infer, grade, select, publish or invoke Git."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from functools import reduce
import ast
import hashlib
import json
import math
import re
import struct
import unicodedata
import zlib
import datetime

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
E = ROOT / 'docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153'
V = ROOT / 'docs/natural-assistant/v4'
checks = []
receipts = []

def sha(b):
    return hashlib.sha256(b).hexdigest()

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    b = p.read_bytes()
    receipts.append({'path': str(p.relative_to(ROOT)), 'bytes': len(b), 'sha256': sha(b)})
    return json.loads(b)

def check(name, actual, expected):
    ok = actual == expected
    checks.append({'name': name, 'passed': ok, 'actual': actual, 'expected': expected})
    if not ok:
        raise ValueError(f'{name}: {actual!r} != {expected!r}')

def git_object(oid):
    # Read/hash actual local immutable Git objects without invoking Git or changing it.
    b = zlib.decompress((ROOT / '.git/objects' / oid[:2] / oid[2:]).read_bytes())
    assert hashlib.sha1(b).hexdigest() == oid
    header, data = b.split(b'\0', 1)
    kind, length = header.decode().split(' ')
    assert len(data) == int(length)
    return kind, data

def git_path(commit, path):
    kind, data = git_object(commit)
    assert kind == 'commit'
    oid = data.split(b'\n')[0].decode().removeprefix('tree ')
    for part in Path(path).parts:
        kind, data = git_object(oid)
        assert kind == 'tree'
        entries = {}
        pos = 0
        while pos < len(data):
            end = data.index(b'\0', pos)
            mode, name = data[pos:end].split(b' ', 1)
            entries[name.decode()] = data[end+1:end+21].hex()
            pos = end + 21
        oid = entries[part]
    kind, data = git_object(oid)
    assert kind == 'blob'
    return oid, data

def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]

def normalized(s):
    return ''.join(unicodedata.normalize('NFKC', s).split())

candidate_path = ROOT / 'outputs/natural-v4/release-preflight/assistant-2b-v4.candidate.json'
c = read(candidate_path)
check('Exact final candidate SHA', digest(candidate_path), 'c529b559bcb71840bb954b5164388ad1c0ed16e498ba4e02bc1b9e42200119ab')
card = c['model_card'].encode()
check('Exact final card SHA', sha(card), 'c319a97d90aca2a074410e3ece8c6325515e0c3a336660e4ca1d00ba6fab65fd')
check('Candidate card standalone bytes', card, (candidate_path.parent/'model-card-v4.candidate.md').read_bytes())
# Replace bytes in the receipt with the already verified digest to keep JSON serializable.
checks[-1]['actual'] = checks[-1]['expected'] = sha(card)
for key, expected in [('approved', False), ('reviewed', False), ('selected_variant', 'base'), ('adapter_parameters', 0), ('source', None), ('files', [])]:
    check('Candidate ' + key, c[key], expected)
check('Repository complete MIT text in card', (ROOT/'LICENSE').read_text().strip() in c['model_card'], True)

selection = read(V/'selection.json')
manifest = read(V/'manifest.json')
summary = read(E/'selected-final-test-audit-summary.json')
scores = read(E/'blind-review/scored/scores.json')
subsets = read(E/'blind-review/descriptive-subsets.json')
closure = read(E/'blind-review/test-closure.json')
grades = read(E/'blind-review/combined/grades.json')
generations = read(E/'blind-review/generations-base.json')
transcripts = read(E/'blind-review/transcripts.json')
result = read(E/'blind-review/result.json')
commit = '1e71b8abffb34eccd297747d39827135a627107a'
committed = []
for item in summary['artifact_binding']['committed_inputs']:
    oid, b = git_path(commit, item['path'])
    check('Pretest Git bytes '+item['path'], sha(b), item['sha256'])
    check('Current bytes remain pretest '+item['path'], (ROOT/item['path']).read_bytes() == b, True)
    committed.append({'path': item['path'], 'blob_oid': oid, 'bytes': len(b), 'sha256': sha(b)})
check('Selection SHA in card', digest(V/'selection.json') in c['model_card'], True)
check('Manifest SHA in card', digest(V/'manifest.json') in c['model_card'], True)
check('Selection remained BASE', selection['selected_variant'], 'base')
check('Final test remained BASE', scores['selected_variant'], 'base')
check('Actual test selected-only BASE', (result['selected_only'], set(result['variants'])), (True, {'base'}))
checks[-1]['actual'] = checks[-1]['expected'] = [True, ['base']]
check('Actual base has no trainable parameters', result['trainable_parameters'], 0)
check('Actual test source Git', scores['artifact_binding']['execution_revision'], commit)
for name, p, expected in [
    ('scores', E/'blind-review/scored/scores.json', closure['scores_sha256']),
    ('grades', E/'blind-review/combined/grades.json', scores['manual_grades_sha256']),
    ('generations', E/'blind-review/generations-base.json', scores['artifact_binding']['generation_file']['sha256']),
    ('transcripts', E/'blind-review/transcripts.json', scores['artifact_binding']['transcripts_sha256']),
    ('result', E/'blind-review/result.json', scores['artifact_binding']['test_result_sha256'])]:
    check('Actual evidence binding '+name, digest(p), expected)
check('Descriptive subset source scores', subsets['scores_sha256'], digest(E/'blind-review/scored/scores.json'))
check('Canonical 178 / 22 / 147', [len(generations), len(transcripts), len(grades['grades'])], [178, 22, 147])
expected_den = {'photo_summary':42, 'photo_fact':84, 'text_presence':18, 'single_ocr':10, 'ordered_ocr':3, 'text_chat':13, 'voice_typed_reference_chat':4, 'voice_actual_asr_chat':4}
expected_correct = {'photo_summary':25, 'photo_fact':58, 'text_presence':18, 'single_ocr':8, 'ordered_ocr':1, 'text_chat':2, 'voice_typed_reference_chat':2, 'voice_actual_asr_chat':2}
check('Actual decision group denominators', dict(Counter(d['group'] for d in scores['case_decisions'])), expected_den)
check('Actual decisions correct counts', dict(Counter(d['group'] for d in scores['case_decisions'] if d['passed'])), expected_correct)
check('Stored correct counts', scores['correct_counts'], expected_correct)
check('Stored original denominators', {k:scores['denominators'][k] for k in expected_den}, expected_den)
decision = {d['case_id']:d for d in scores['case_decisions']}
check('Independent original grade declarations', [grades['independent_of_training_and_selection'], grades['private_mapping_consulted']], [True, False])
check('Manual grades all have decisions and nonempty reasons', all(g['passed']==decision[g['case_id']]['rubric_passed'] and isinstance(g['passed'],bool) and bool(g['reason'].strip()) for g in grades['grades']), True)
check('Four grade owners equal closure owners', sorted({g['grader'] for g in grades['grades']}), sorted(closure['owners']))
scene_grades = [g for g in grades['grades'] if g['case_id'].startswith('scene:')]
check('Original grade owners declared actual source image inspection', [len(scene_grades), all(g['source_image_inspected'] is True for g in scene_grades)], [126, True])
check('Grades contain exactly 42 unique original photographs', len({g['case_id'].rsplit('/',1)[0] for g in scene_grades}), 42)
check('Original token EOS count', sum(bool(g['generated_token_ids']) and g['generated_token_ids'][-1] in g['eos_token_ids'] for g in generations), 171)
incomplete = [g for g in generations if g['truncated'] or g['completion_unknown']]
check('Original truncated chat count', [len(incomplete), dict(Counter(g['task'] for g in incomplete))], [7, {'chat':7}])
check('Incomplete answers remain incorrect in original denominators', [len(scores['incomplete_case_ids']), all(not decision[k]['passed'] for k in scores['incomplete_case_ids'])], [7, True])
rows = manifest['rows']; audio = manifest['audio_rows']; row_by_id = {d['id']:d for d in rows}
test_facts = [d for d in rows if d['split']=='test' and d['task']=='scene' and d['references']['qa_type']!='scene']
for label, den, correct in [('action',11,3),('relation',29,21)]:
    sub = [d for d in test_facts if d['references']['qa_type']==label]
    check('Original manifest '+label+' subset', [len(sub),sum(decision['scene:'+d['id']]['passed'] for d in sub)], [den,correct])
    check('Descriptive '+label+' subset', [subsets['photo_fact_subsets'][label]['denominator'], subsets['photo_fact_subsets'][label]['correct']], [den,correct])
check('One context-dependent test chat', len([d for d in rows if d['split']=='test' and d['task']=='chat' and d.get('history')]), 1)
check('Context subset original denominator', subsets['context_dependent_text_chat']['denominator'], 1)
asr_group = defaultdict(lambda: [0,0,0,0,0])
for t in transcripts:
    ref = t['reference_transcript']; hyp=t['transcript']
    values = [1,distance(ref,hyp),len(ref),distance(normalized(ref),normalized(hyp)),len(normalized(ref))]
    for k in ['all',t['task']]:
        asr_group[k] = [x+y for x,y in zip(asr_group[k],values)]
    source = next(x for x in audio if x['id']==t['id'])
    check('Genuine unchanged audio/reference '+t['id'], [source['sha256']==t['audio_sha256'], source['user']==ref], [True,True])
check('Recomputed actual ASR totals', dict(asr_group), {'all':[22,112,674,96,661], 'speech_transcription':[18,112,632,96,619], 'speech_chat':[4,0,42,0,42]})
check('Dataset text/image splits', dict(Counter(d['split'] for d in rows)), {'train':2077,'validation':124,'test':170})
check('Dataset audio splits', dict(Counter(d['split'] for d in audio)), {'validation':16,'test':22})
families=defaultdict(set)
for d in rows+audio:families[d['family']].add(d['split'])
check('No source family crosses splits', all(len(s)==1 for s in families.values()), True)
photos=defaultdict(set)
for d in rows:
    if d['task']=='scene':photos[d['split']].add(d['image'])
check('DOCCI unique source photos', {k:len(v) for k,v in photos.items()}, {'train':439,'validation':28,'test':42})
check('NVIDIA retained file count', len([f for f in manifest['files'] if f['path'].startswith('ocr/nvidia/')]),827)
check('NVIDIA rows exclusively training', all(d['split']=='train' for d in rows if d.get('image','').startswith('ocr/nvidia/')),True)
ocr = read(V/'data/ocr-sources.json')
commons = [s for s in ocr['sources'] if s['source_id'].startswith('commons-')]
check('Commons 70 source images', len(commons),70)
check('Commons 62 retained training crops', len([f for f in manifest['files'] if f['path'].startswith('ocr/commons/crops/')]),62)
commons_receipts=[]
for s in commons:
    q=ROOT/s['metadata_query']; d=json.loads(q.read_text()); pages=d['query']['pages']
    pages=list(pages.values()) if isinstance(pages,dict) else pages
    page=next(x for x in pages if x['pageid']==s['source_page_id'])
    image=page['imageinfo'][0]; ext=image['extmetadata']
    check('Commons license '+s['source_id'], ext['LicenseShortName']['value'],s['license'])
    check('Commons author '+s['source_id'], ext['Artist']['value'],s['artist_html'])
    check('Commons preserved source restrictions '+s['source_id'], ext.get('Restrictions',{}).get('value',''),s['restrictions'])
    commons_receipts.append({'source_id':s['source_id'],'source_url':s['source_page'],'license':s['license'],'artist_html':s['artist_html'],'license_url':s['license_url'],'modification':s['transform'],'restrictions':s['restrictions'],'saved_official_metadata':str(q.relative_to(ROOT)),'saved_official_metadata_sha256':digest(q)})
asset_count=0
for f in manifest['files']:
    p=ROOT/'outputs/natural-v4/data'/f['path']
    check('Fixed asset bytes/hash '+f['path'], [p.stat().st_size,digest(p)], [f['bytes'],f['sha256']])
    asset_count+=1
for s in manifest['sources']+manifest['annotation_sources']:
    p=ROOT/s['path'];check('Manifest source fingerprint '+s['path'],digest(p),s['sha256'])
train=[]
for p in sorted((ROOT/'outputs/natural-v4/modal-runs').glob('train-*/natural-*/review/training.json')):
    d=read(p)
    check('Complete candidate train '+p.parts[-4], [d['status'],d['completed_steps'],d['trained_rows'],d['trainable_parameters'],len(d['history'])], ['completed',2077,4154,1605632,2077])
    train.append({'path':str(p.relative_to(ROOT)),'learning_rate':d['learning_rate'],'completed_steps':d['completed_steps'],'trained_rows':d['trained_rows'],'trainable_parameters':d['trainable_parameters'],'base_parameters':d['total_parameters']-d['trainable_parameters']})
check('Two actual complete candidate training runs',len(train),2)
validation=read(ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json')
check('Validation selection BASE',validation['selected_variant'],'base')
base=validation['variants']['base']
check('Candidate photo improvement with failed completeness and speech-chat gates',all(x['correct_counts']['photo_summary']>base['correct_counts']['photo_summary'] and x['correct_counts']['photo_fact']>base['correct_counts']['photo_fact'] and not x['all_generations_complete'] and not x['eligible'] and not x['gates']['voice_typed_reference_chat']['passed'] for k,x in validation['variants'].items() if k!='base'),True)
asr_selection=read(V/'asr-selection.json')
comparison=read(ROOT/'docs/natural-assistant/evidence/v4-research/asr-cpu-validation/comparison.json')
check('ASR validation same original denominator/count',[[comparison['variants'][k]['metrics'][f] for f in ['count','errors','reference_characters']] for k in ['small','turbo']],[[16,117,510],[16,50,510]])
for f in asr_selection['evidence']:check('Frozen ASR selection evidence '+f['path'],digest(ROOT/f['path']),f['sha256'])
parameters=[]
upstream=json.loads((OUT/'upstream-source-receipts.json').read_text())
for model,key,expected in [(c['base_model'],'qwen-api',2127532032),(c['asr_model'],'whisper-api',808878080)]:
    api=json.loads((OUT/'research'/f'{key}.raw').read_text())
    check('Actual anonymous upstream immutable pin '+model['repo'], [api['sha'],api['private'],api['gated'],api['disabled']], [model['revision'],False,False,False])
    license_name='apache-2.0' if key=='qwen-api' else 'mit'
    check('Actual official model card license '+model['repo'],api['cardData']['license'],license_name)
    p=ROOT/'outputs/natural-v4/student-base-cache/hf'/('models--'+model['repo'].replace('/','--'))/'snapshots'/model['revision']/'model.safetensors'
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; header=f.read(n)
    h=json.loads(header);count=sum(math.prod(t['shape']) for k,t in h.items() if k!='__metadata__')
    check('Cached exact tensor parameter count '+model['repo'],count,expected)
    sibling=next(s for s in api['siblings'] if s['rfilename']=='model.safetensors')
    model_sha=digest(p)
    check('Cached full model matches pinned upstream LFS SHA '+model['repo'],model_sha,sibling['lfs']['sha256'])
    parameters.append({'model':model,'parameters_from_safetensors_shapes':count,'safetensors_header_bytes':n,'header_sha256':sha(header),'bytes':p.stat().st_size,'full_sha256':model_sha})
    for sibling in api['siblings']:
        if sibling['rfilename'] in {'.gitattributes','model.safetensors'}:
            continue
        b=(p.parent/sibling['rfilename']).read_bytes()
        oid=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
        check('Pinned public supporting-file bytes '+model['repo']+'/'+sibling['rfilename'],[len(b),oid],[sibling['size'],sibling['blobId']])
download_files=[{'repo':json.loads((OUT/'research'/f'{k}.raw').read_text())['id'],'path':s['rfilename'],'bytes':s['size']} for k in ['qwen-api','whisper-api'] for s in json.loads((OUT/'research'/f'{k}.raw').read_text())['siblings'] if s['rfilename']!='.gitattributes']
check('Pinned download 23 files / total bytes',[len(download_files),sum(x['bytes'] for x in download_files)],[23,5889111977])
check('Actual LM device, precision and image budget',[result[k] for k in ['gpu_name','dtype','min_pixels','max_pixels','max_tokens']],['NVIDIA L4','bfloat16',65536,524288,2048])
check('Actual maximum generation suffix length',max(g['generated_tokens'] for g in generations),384)
check('All raw prompts plus reserved generation fit total budget',all(g['input_tokens']+384<=2048 for g in generations),True)
for name,license_literal in [('qwen-readme','license: apache-2.0'),('whisper-readme','license: mit'),('nvidia-readme','license: cc-by-4.0'),('oasst2-readme','license: apache-2.0'),('fleurs-readme','license:\n- cc-by-4.0')]:
    check('Actual freshly retrieved source license '+name,license_literal in (OUT/'research'/f'{name}.raw').read_text(),True)
source=(ROOT/'scripts/modal_natural.py').read_text();tree=ast.parse(source)
names={'validate_release','safe_name','safe_relative','checkpoint_names'}
module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
ns={'re':re,'PurePosixPath':PurePosixPath,'MAX_PRIVATE_BACKUP_BYTES':128*1024*1024}
exec(compile(module,'modal_natural.py:only-local-contract-functions','exec'),ns)
try:
    ns['validate_release'](c)
except ValueError:
    rejected=True
else:
    rejected=False
check('Actual validator rejects unapproved candidate',rejected,True)
approved=dict(c,approved=True,reviewed=True)
check('Actual local validator accepts exact BASE artifact after truthful review flags',ns['validate_release'](approved)==approved,True)
for name in ['scripts/modal_natural.py','scripts/fetch_natural_release.py','tiny_perceptron/natural_assistant.py','tiny_perceptron/natural_ui.py','docs/natural-assistant/v4/DATA.md','docs/natural-assistant/v4/STUDENT.md','LICENSE']:
    p=ROOT/name;receipts.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
report={'schema_version':1,'auditor':'/root/v4_public_release_auditor','recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only release artifact audit, including fixed final-test arithmetic and evidence; no new inference, no semantic regrading, no model selection, no training, no Git command or external publication','candidate_sha256':digest(candidate_path),'model_card_sha256':sha(card),'checks_passed':len(checks),'all_checks_passed':all(x['passed'] for x in checks),'pretest_git_commit':commit,'pretest_committed_inputs':committed,'counts':{'lm_generations':len(generations),'audio_recordings':len(transcripts),'manual_grades':len(grades['grades']),'fixed_assets_verified':asset_count},'correct_counts':expected_correct,'denominators':expected_den,'asr_recomputed':dict(asr_group),'training':train,'model_parameters':parameters,'download_files':download_files,'commons_saved_official_source_receipts':commons_receipts,'source_receipts':receipts,'checks':checks}
(OUT/'local-evidence-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['candidate_sha256','model_card_sha256','all_checks_passed','checks_passed','counts']},ensure_ascii=False,indent=2))
