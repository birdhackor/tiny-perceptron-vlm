"""Offline, bounded independent checks of DATA claims; no training or downloads."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, ast, copy, gzip, hashlib, importlib.util, io, json, platform, sys, tarfile, tempfile
import torch, PIL, numpy, soundfile

ROOT=Path.cwd()
OUT=ROOT/'docs/technical-reviews/artifacts/natural-v4-supplemental/data'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
fetch=module('independent_fetch',ROOT/'scripts/fetch_natural_data.py')
build=module('independent_build',ROOT/'scripts/build_natural_v4_assets.py')
ocr=module('independent_ocr',ROOT/'scripts/prepare_natural_v4_ocr.py')
voice=module('independent_voice',ROOT/'scripts/replay_natural_v4_voice.py')
manifest_path=ROOT/'docs/natural-assistant/v4/manifest.json'
m=fetch.load_manifest(manifest_path,'0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60')
splits=('train','validation','test')
def category(r):
    if r['task']=='scene': return 'DOCCI '+r['references']['qa_type']
    if r['task']=='chat': return 'course chat' if r['family'].startswith('chat-v4-original') else 'OpenAssistant chat'
    if r['task']=='text_presence': return 'Chinese presence'
    return ('NVIDIA ' if 'nvidia' in r['image'] else 'Commons ')+r['task']
counts={s:dict(Counter(category(r) for r in m['rows'] if r['split']==s)) for s in splits}
print('Category counts',json.dumps(counts,ensure_ascii=False,sort_keys=True))
print('Chat family examples',sorted({r['family'] for r in m['rows'] if r['task']=='chat'})[:4])
family_splits=defaultdict(set); physical_splits=defaultdict(set)
file_map={f['path']:f for f in m['files']}
for r in m['rows']+m['audio_rows']:
    family_splits[r['family']].add(r['split'])
    for k in ['image','audio']:
        if r.get(k): physical_splits[file_map[r[k]]['sha256']].add(r['split'])
family_overlap={k:sorted(v) for k,v in family_splits.items() if len(v)>1}
physical_overlap={k:sorted(v) for k,v in physical_splits.items() if len(v)>1}
assert not family_overlap and not physical_overlap
material={}
for s in splits:
    docci=[r for r in m['rows'] if r['split']==s and r['task']=='scene']
    chat=[r for r in m['rows'] if r['split']==s and r['task']=='chat']
    material[s]={'docci_images':len({r['image'] for r in docci}),'docci_families':len({r['family'] for r in docci}),
                 'chat_families':len({r['family'] for r in chat})}
sources=json.loads((ROOT/'docs/natural-assistant/v4/data/ocr-sources.json').read_bytes())
used={r.get('image') for r in m['rows'] if r.get('image')}
retained=[a for a in sources['artifacts'] if a['path'] in used]
orig={s['source_id']:s for s in sources['sources']}
for s in splits:
    by=[a for a in retained if orig[a['source_id']]['split']==s]
    commons=[a for a in by if orig[a['source_id']]['dataset'].startswith('Wikimedia')]
    nvidia=[a for a in by if orig[a['source_id']]['dataset'].startswith('nvidia/')]
    material[s].update({'commons_source_images':len({a['source_id'] for a in commons}),
                       'commons_families':len({orig[a['source_id']]['family'] for a in commons}),
                       'commons_original_files':sum(a['kind']=='source_image' for a in commons),
                       'commons_crops':sum(a['kind']=='crop' for a in commons),
                       'nvidia_files':len(nvidia),'nvidia_original_pages':len({a['source_id'] for a in nvidia})})
print('Materials',json.dumps(material,ensure_ascii=False,sort_keys=True))
sizes=[]
for a in m['archives']:
    unpacked=sum(f['bytes'] for f in a['files'])
    sizes.append({'path':a['path'],'compressed_bytes':a['bytes'],'unpacked_bytes':unpacked,'files':len(a['files']),
                  'compressed_MB':format(a['bytes']/1e6,'.2f'),'unpacked_MB':format(unpacked/1e6,'.2f')})
print('Sizes',json.dumps(sizes,sort_keys=True))
print('Totals',sum(a['bytes'] for a in m['archives']),sum(f['bytes'] for f in m['files']),len(m['files']))
assert [len([r for r in m['rows'] if r['split']==s]) for s in splits]==[2077,124,170]
assert [len([r for r in m['audio_rows'] if r['split']==s]) for s in splits]==[0,16,22]
assert [material[s]['docci_images'] for s in splits]==[439,28,42]
assert [material[s]['docci_families'] for s in splits]==[126,11,12]
assembled=build.assemble()
assert assembled==(m['rows'],m['audio_rows'])
print('Current source assembly exactly matches complete manifest rows/audio_rows')
actual_files=[]
for f in m['files']:
    p=ROOT/'outputs/natural-v4/data'/f['path']
    actual_files.append({'path':f['path'],'exists':p.is_file(),'size_match':p.is_file() and p.stat().st_size==f['bytes'],
                         'sha_match':p.is_file() and digest(p)==f['sha256']})
print('Current selected local files',Counter((f['exists'],f['size_match'],f['sha_match']) for f in actual_files))
audio_meta=[]
for r in m['audio_rows']:
    p=ROOT/'outputs/natural-v4/data'/r['audio']; info=soundfile.info(p)
    assert (info.frames,info.samplerate,info.channels)==(r['num_samples'],r['sample_rate'],r['channels'])
    audio_meta.append({'id':r['id'],'frames':info.frames,'sample_rate':info.samplerate,'channels':info.channels,'duration':info.duration})
print('Selected WAV metadata verified',len(audio_meta))

checks=[]
def expect_error(label,fn):
    try: fn()
    except (ValueError,OSError) as e: checks.append({'label':label,'observed':'rejected','message':str(e)}); return
    raise AssertionError(label+' did not reject')
with tempfile.TemporaryDirectory(prefix='data-review-offline-',dir=OUT) as td:
    directory=Path(td)/'fixture'; directory.mkdir(); (directory/'vision').mkdir(); p=directory/'vision/x.txt'; p.write_bytes(b'cat')
    minimal={'files':[{'path':'vision/x.txt','bytes':3,'sha256':digest(p)}]}
    fetch.verify_directory(minimal,directory); checks.append({'label':'existing identical files','observed':'accepted'})
    p.write_bytes(b'dog'); before=p.read_bytes(); expect_error('different existing bytes',lambda:fetch.verify_directory(minimal,directory)); assert p.read_bytes()==before
    p.write_bytes(b'cat'); (directory/'extra.txt').write_bytes(b'keep'); expect_error('extra local file',lambda:fetch.verify_directory(minimal,directory)); assert (directory/'extra.txt').read_bytes()==b'keep'
    expect_error('path traversal',lambda:fetch.safe_relative('../escape'))
    class Response(io.BytesIO):
        status=200
        def __init__(self,raw,status=200,headers=None): super().__init__(raw); self.status=status; self.headers=headers or {}; self.requests=[]
        def geturl(self): return 'https://example.invalid/frozen'
    archive=Path(td)/'fixture.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('vision/x.txt'); info.size=3; tar.addfile(info,io.BytesIO(b'cat'))
    spec={'bytes':archive.stat().st_size,'sha256':digest(archive),'files':minimal['files']}
    dest=Path(td)/'extracted';dest.mkdir();fetch.extract_checked(archive,spec,dest);fetch.verify_directory(minimal,dest)
    checks.append({'label':'small pinned archive + per-file extraction','observed':'accepted'})
    archive.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 999\n')
    expect_error('LFS pointer instead of archive',lambda:fetch.extract_checked(archive,spec,Path(td)/'other'))
    with unittest_patch(ocr) if False else io.StringIO(): pass
    import unittest.mock
    with unittest.mock.patch.object(ocr.urllib.request,'urlopen',lambda *a,**kw:Response(b'abcd',200,{'Content-Range':'bytes 0-3/4'})):
        expect_error('server ignores byte Range',lambda:ocr.RangeFile('https://example.invalid/x',4,4).read(1))
    with unittest.mock.patch.object(ocr.urllib.request,'urlopen',lambda *a,**kw:Response(b'abcd',206,{'Content-Range':'bytes 0-3/4'})):
        rf=ocr.RangeFile('https://example.invalid/x',4,4);assert rf.read(2)==b'ab';assert rf.read(2)==b'cd';assert rf.transferred==4
        checks.append({'label':'exact partial range + cache accounting','observed':'accepted','transferred':rf.transferred})
        expect_error('range read exceeds budget',lambda:ocr.RangeFile('https://example.invalid/x',4,3).read(1))
    br=voice.BoundedReader(io.BytesIO(b'abcde'),3);assert br.read(2)==b'ab';assert br.read(2)==b'c';expect_error('voice stream exceeds cap',lambda:br.read(1))

from tiny_perceptron.natural_concepts import text_error_report
cer=[]
for ref,pred in [('今天去臺北','今天去台北'),('請推薦不辣的晚餐。','請推薦辣的晚餐。'),('','多字')]:
    r=text_error_report(ref,pred);cer.append({'reference':ref,'prediction':pred,'result':r})
assert cer[0]['result']['edits']==1 and cer[0]['result']['cer']==0.2
assert cer[1]['result']['reference_characters']==9 and cer[1]['result']['edits']==1
assert cer[2]['result']['cer'] is None
print('Bounded behavior probes',json.dumps(checks,ensure_ascii=False))
print('CER probes',json.dumps(cer,ensure_ascii=False))
print('Environment',json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','Pillow':PIL.__version__,'numpy':numpy.__version__,'soundfile':soundfile.__version__,'h5py':'not installed'}))
receipt={'manifest_sha256':digest(manifest_path),'counts':counts,'materials':material,'sizes':sizes,'family_overlap':family_overlap,
         'physical_hash_overlap':physical_overlap,'all_selected_file_checks':actual_files,'audio_metadata':audio_meta,'behavior_checks':checks,'cer_checks':cer}
(OUT/'cpu-audit-result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
