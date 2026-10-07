import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tiny_perceptron.natural_assistant import score_output
base=Path('docs/technical-reviews/artifacts/p7_technical_f');raw=Path('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw');m=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());cfg=json.loads((raw/'result.json').read_text());g={}
for i in range(1,8):g.update(json.loads((base/f'photo-grades-0{i}.json').read_text())['photos'])
c=json.loads((base/'photo-grades-correction-01.json').read_text());g[c['photo']][c['variant']][0]=c['current'];d=json.loads((base/'20-8-own-dialogue-judgments.json').read_text());rows={r['id']:r for r in m['rows']};gen={v:json.loads((raw/f'generations-{v}.json').read_text()) for v in d['text_chat']};tx={r['id']:r['transcript'] for r in json.loads((raw/'transcripts.json').read_text())}
for v,gs in gen.items():
 actual=[r for r in gs if r['task']=='speech_chat'];assert len(actual)==4 and all(r['user']==tx[r['id']] for r in actual)
 counts={'photo_summary':sum(r[v][0] for r in g.values()),'photo_fact':sum(sum(r[v][1:]) for r in g.values()),'text_chat':sum(d['text_chat'][v]),'voice_typed_reference_chat':sum(d['typed_voice'][v]),'voice_actual_asr_chat':sum(d['actual_asr_voice'][v])}
 for task,group in [('text_presence','text_presence'),('ocr','single_ocr'),('ocr_order','ordered_ocr')]:counts[group]=sum(bool(score_output(rows[r['id']],r['prediction'])['passed']) for r in gs if r['task']==task)
 weights={'photo_summary':270,'photo_fact':135,'text_presence':840,'single_ocr':1512,'ordered_ocr':5040,'text_chat':560,'voice_typed_reference_chat':1260,'voice_actual_asr_chat':1260};print(v,'own counts',counts,'own primary',sum(counts[k]*w for k,w in weights.items()),'/75600','same actual ASR transcript 4/4')
print('shared original cfg',{k:cfg[k] for k in ['model','model_revision','asr_model','asr_revision','max_pixels','min_pixels','max_tokens','seed']});print('scope: human judgment difference, not new blinded experiment/model inference; published historical aggregate is separate. Gates still reject both adapters independent of small subjective primary-score boundary.')
