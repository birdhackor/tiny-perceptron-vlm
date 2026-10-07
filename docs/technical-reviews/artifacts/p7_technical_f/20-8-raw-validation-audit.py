import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tiny_perceptron.natural_assistant import score_output
b=Path('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review');m=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());p=json.loads(Path('docs/natural-assistant/v4/validation-protocol-lower-lr.json').read_text());aliases=json.loads((b/'private-map.json').read_text())['aliases'];rawgrades=json.loads((b/'combined/grades.json').read_text())['grades'];grades={(aliases[g['candidate']],g['case_id']):g['passed'] for g in rawgrades}
rows={r['id']:r for r in m['rows']};rows.update({r['id']:r for r in m['audio_rows']});weights={'photo_summary':270,'photo_fact':135,'text_presence':840,'single_ocr':1512,'ordered_ocr':5040,'text_chat':560,'voice_typed_reference_chat':1260,'voice_actual_asr_chat':1260};all_counts={}
def complete(r):
 tok=r['generated_token_ids'];return bool(tok and len(tok)<=384 and len(tok)==r['generated_tokens'] and tok[-1] in r['eos_token_ids'] and r['ended_with_eos'] and r['stop_reason']=='eos' and not r['truncated'] and not r['completion_unknown'])
for variant in ['base','adapter-step-001039','adapter-step-002077']:
 generation=json.loads((b/f'raw/generations-{variant}.json').read_text());counts={k:0 for k in weights};denom={k:0 for k in weights};incomplete=[]
 for g in generation:
  row=rows[g['id']];task=g['task'];group= ('photo_summary' if g['id'].endswith('/scene') else 'photo_fact') if task=='scene' else {'chat':'text_chat','text_presence':'text_presence','ocr':'single_ocr','ocr_order':'ordered_ocr','typed_chat':'voice_typed_reference_chat','speech_chat':'voice_actual_asr_chat'}[task]
  denom[group]+=1;done=complete(g)
  if not done:incomplete.append({'id':g['id'],'task':task,'generated_tokens':g['generated_tokens'],'stop_reason':g['stop_reason'],'last_raw_token':g['generated_token_ids'][-1]})
  key=(variant,task+':'+g['id']);correct=grades[key] if group in ['photo_summary','photo_fact','text_chat','voice_typed_reference_chat','voice_actual_asr_chat'] else score_output(row,g['prediction'])['passed'];counts[group]+=int(done and correct)
 assert len(generation)==132 and denom=={k:p['validation_denominators'][k] for k in weights}
 numerator=sum(weights[k]*counts[k] for k in weights);all_counts[variant]=counts
 print(variant,json.dumps({'recorded_historical_semantic_counts_plus_own_exact_ocr':counts,'denominators':denom,'primary_numerator':numerator,'primary_denominator':75600,'incomplete':incomplete},ensure_ascii=False))
print('scope: aggregate original per-case human grades as historical measurement only; no prior verdict/summary used and no new mature model generation. Own semantic grades retained separately.')
# compare own scene/fact judgments, preserving both
own={}
folder=Path('docs/technical-reviews/artifacts/p7_technical_f')
for i in range(1,8):own.update(json.loads((folder/f'photo-grades-0{i}.json').read_text())['photos'])
c=json.loads((folder/'photo-grades-correction-01.json').read_text());own[c['photo']][c['variant']][0]=c['current']
for variant in all_counts:
 dif=[]
 for photo,r in own.items():
  for index,part in enumerate(['scene','fact1','fact2']):
   key=(variant,'scene:vision-v4:docci/'+photo+'/'+part)
   if r[variant][index]!=grades[key]:dif.append({'photo':photo,'part':part,'own':r[variant][index],'historical':grades[key],'own_reason':r['notes']})
 print('own-vs-historical differences',variant,json.dumps(dif,ensure_ascii=False))
