from pathlib import Path
import json,hashlib,shutil,difflib,platform
from PIL import Image
import soundfile as sf
ROOT=Path(__file__).resolve().parents[3];ART=ROOT/'docs/technical-reviews/artifacts'
hashof=lambda b:hashlib.sha256(b).hexdigest()
raw=(ROOT/'course/chapters/20.md').read_bytes(); text=raw.decode();start=text.index('## 20.1 ');end=text.index('## 20.2 ');section=text[start:end].encode(); intro=text[:start].encode()
old=(ROOT/'outputs/natural-extension/lesson-fragments/20.1-before-ui-label-fix.md').read_bytes()
assert '「辨識錄音」與「送出問題」' in old.decode()
assert '「辨識語音」與「送出問題」' in section.decode()
assert old.decode().replace('「辨識錄音」與「送出問題」','「辨識語音」與「送出問題」').encode()==section
(ART/'natural-20.1-initial-section.md').write_bytes(old)
(ART/'natural-20.1-current-section.md').write_bytes(section)
(ART/'natural-20.1-intro.md').write_bytes(intro)
(ART/'natural-20.1-label-diff.txt').write_text(''.join(difflib.unified_diff(old.decode().splitlines(keepends=True),section.decode().splitlines(keepends=True),fromfile='first-read-section',tofile='reviewed-current-section')))
version_record={'reviewer_task':'/root/natural_factual_20_1','initial_source_sha256':hashof(old),'current_source_sha256':hashof(section),'intro_sha256':hashof(intro),'initial_capture_provenance':'Root saved the original section bytes after this reviewer independently read and reported the UI label mismatch, immediately before root changed the single label. No initial JSON report had been written at that time. Reviewer subsequently read this snapshot and compared it with the first-read phrase and the current section; only the reported label changed.','initial_finding':{'claim':'對應的兩個入口是「辨識錄音」與「送出問題」','status':'contradicted','source':'tiny_perceptron/natural_ui.py:385','observed':'HTML button id=transcribe reads 辨識語音'},'recheck':{'action':'This reviewer re-read the complete current section and actual UI HTML. CPU execution additionally asserted both actual UI labels.','status':'resolved','revised_claim':'對應的兩個入口是「辨識語音」與「送出問題」'}}
(ART/'natural-20.1-version-history.json').write_text(json.dumps(version_record,ensure_ascii=False,indent=2)+'\n')
p=ROOT/'docs/natural-assistant/evidence/usage/cpu-base-input-routes.json';result=json.loads(p.read_text());shutil.copyfile(p,ART/'natural-20.1-recorded-cpu-routes.json')
cachepath=ROOT/'docs/natural-assistant/evidence/usage/cpu-base-cache.json';cache=json.loads(cachepath.read_text());shutil.copyfile(cachepath,ART/'natural-20.1-recorded-cpu-cache.json')
shutil.copyfile(ROOT/'docs/natural-assistant/evidence/usage/cpu-base-input-routes.py.txt',ART/'natural-20.1-recorded-route-driver.py.txt')
manifestpath=ROOT/'docs/natural-assistant/manifest.json';manifest=json.loads(manifestpath.read_text());assert hashof(manifestpath.read_bytes())==result['provenance']['manifest_sha256']
ids={r['id'] for r in result['image_generations']}|{result['speech_response']['id']}
selected=[r for r in manifest['rows']+manifest['audio_rows'] if r['id'] in ids]
assert len(selected)==3 and all(r['split']=='train' for r in selected)
for key,official in [('core','qwen'),('asr','whisper')]:
 files={f['name']:f for f in cache['snapshots'][key]['files']}
 for name,suffix in [('README.md','card.md'),('config.json','config.json')]:
  source=ART/f'natural-20.1-{official}-{suffix}'
  assert hashof(source.read_bytes())==files[name]['sha256']
 assert cache['snapshots'][key]['revision']==result['provenance']['model_revision' if key=='core' else 'asr_revision']
images=[]
for r in selected:
 asset=r.get('image') or r.get('audio');local=ROOT/'outputs/natural-extension/data'/asset
 assert local.is_file()
 expected=result['provenance']['asset_sha256'][asset];assert hashof(local.read_bytes())==expected
 if r.get('image'):
  kind='training-photo.jpg' if r['task']=='scene' else 'training-text-card.png'
  shutil.copyfile(local,ART/('natural-20.1-'+kind))
  with Image.open(local) as img:images.append({'path':asset,'sha256':expected,'dimensions':list(img.size),'mode':img.mode})
 else:
  samples,sr=sf.read(local);assert sr==16000 and len(samples)==124800 and r['synthetic'] is False
speech=next(r for r in selected if r.get('audio'))
assert result['audio']['transcript']==result['speech_response']['user']
ocr=next(r for r in result['image_generations'] if r['task']=='ocr');assert ocr['reference_answer']=='臺灣學生一起讀書。' and ocr['prediction']=='台灣學生一起讀書。'
assert '并未' not in result['speech_response']['prediction'] and '並未受到負面指責' in result['speech_response']['prediction']
audit={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-20.1-record-audit.py','environment':{'python':platform.python_version(),'device':'cpu','mode':'metadata/artifact audit only; no model generation'},'result':'All recorded route identifiers and asset hashes match the train manifest. Official fixed-revision cards/configs match the recorded cache hashes. The raw ASR transcript is exactly the speech-response user input. Recorded OCR and chat failure descriptions match the prose.','recorded_environment':result['provenance']['versions'],'recorded_device':result['provenance']['device'],'recorded_scope':result['scope'],'selected_train_rows':selected,'selected_images':images,'denominators':{'photo_examples':1,'ocr_examples':1,'human_mandarin_audio_examples':1,'audio_seconds':7.8,'adapter_loaded':False,'training_examples_only':True},'observed_transcript':result['audio']['transcript'],'original_speech_reference':speech['user'],'observed_speech_chat_reply':result['speech_response']['prediction'],'observed_ocr_prediction':ocr['prediction'],'limits':'This is independent inspection and hash validation of an existing raw CPU model run, not re-running the models or measuring heldout ability. No validation/test generations or scores were read.'}
(ART/'natural-20.1-record-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(version_record,ensure_ascii=False,indent=2));print(audit['result'])
