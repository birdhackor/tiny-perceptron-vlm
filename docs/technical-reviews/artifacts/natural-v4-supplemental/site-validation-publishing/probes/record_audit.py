from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import json,hashlib,unicodedata
root=Path.cwd();out=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-validation-publishing'
def read(p):return json.loads((root/p).read_text())
def sha(p):return hashlib.sha256((root/p).read_bytes()).hexdigest()
manifest=read('docs/natural-assistant/v4/manifest.json');protocol=read('docs/natural-assistant/v4/validation-protocol-lower-lr.json');directory='docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/'
private=read(directory+'private-map.json');grades=read(directory+'combined/grades.json');grade={(g['case_id'],private['aliases'][g['candidate']]):g['passed'] for g in grades['grades']}
rows={r['id']:r for r in manifest['rows']+manifest['audio_rows']}
def group(r):
 if r['task']=='scene':return 'photo_summary' if r['id'].endswith('/scene') else 'photo_fact'
 return {'chat':'text_chat','text_presence':'text_presence','ocr':'single_ocr','ocr_order':'ordered_ocr','typed_chat':'voice_typed_reference_chat','speech_chat':'voice_actual_asr_chat'}[r['task']]
def complete(r):
 tokens=r['generated_token_ids'];return bool(tokens and tokens[-1] in r['eos_token_ids'] and len(tokens)==r['generated_tokens'] and len(tokens)<=384 and r['ended_with_eos'] and r['stop_reason']=='eos' and not r['truncated'] and not r['completion_unknown'])
def norm(s,strip):
 s=unicodedata.normalize('NFKC',s).strip();return ''.join(s.split()) if strip else s
weights={'photo_summary':270,'photo_fact':135,'text_presence':840,'single_ocr':1512,'ordered_ocr':5040,'text_chat':560,'voice_typed_reference_chat':1260,'voice_actual_asr_chat':1260};denoms={k:protocol['validation_denominators'][k] for k in weights};assert sum(denoms[k]*weights[k] for k in weights)==75600
computed={};inputs={}
for variant in protocol['candidate_order']:
 records=read(directory+'raw/generations-'+variant+'.json');counts=Counter();den=Counter();incomplete=[];usedkeys=set()
 for r in records:
  assert r['split']=='validation';key=r['task']+':'+r['id'];assert key not in usedkeys;usedkeys.add(key);g=group(r);den[g]+=1
  if not complete(r):incomplete.append({'case':key,'tokens':len(r['generated_token_ids']),'last_token':r['generated_token_ids'][-1],'eos_ids':r['eos_token_ids']})
  if g in ['text_presence','single_ocr','ordered_ocr']:
   row=rows[r['id']];rubric=row['references'];target=rubric.get('text',row['answer']);passed=norm(target,rubric.get('strip_whitespace',False))==norm(r['prediction'],rubric.get('strip_whitespace',False))
  else:passed=grade[(key,variant)]
  counts[g]+=int(passed and complete(r))
 assert dict(den)==denoms
 numerator=sum(counts[g]*weights[g] for g in weights); photo=(Fraction(counts['photo_summary'],28)+Fraction(counts['photo_fact'],56))/2; dialogue=(Fraction(counts['text_chat'],9)+Fraction(counts['voice_typed_reference_chat'],4)+Fraction(counts['voice_actual_asr_chat'],4))/3
 macro=(photo+Fraction(counts['text_presence'],18)+Fraction(counts['single_ocr'],10)+Fraction(counts['ordered_ocr'],3)+dialogue)/5;assert macro==Fraction(numerator,75600)
 computed[variant]={'denominators':dict(den),'correct_counts':dict(counts),'primary_numerator':numerator,'primary_denominator':75600,'primary':float(macro),'incomplete':incomplete,'all_eos':not incomplete}
 inputs[variant]=records
base=computed['base'];selected='base'
for variant in protocol['candidate_order'][1:]:
 c=computed[variant];gates={g:c['correct_counts'][g]>=max(0,base['correct_counts'][g]-int(protocol['adapter_nonregression_gates_vs_base'][g+'_correct_count_minimum']=='base minus 1')) for g in weights};c['gates']=gates;c['eligible']=c['all_eos'] and all(gates.values())
 if c['eligible'] and c['primary_numerator']>computed[selected]['primary_numerator']:selected=variant
original=read(directory+'scored/scores.json')
for v,c in computed.items():assert c['correct_counts']==original['variants'][v]['correct_counts'];assert c['primary_numerator']==original['variants'][v]['primary_numerator']
assert selected==read('docs/natural-assistant/v4/selection.json')['selected_variant']=='base'
# Identical questions/history/images and actual ASR vs reference text enter same candidate.
paired=[]
for variant,records in inputs.items():
 by={(r['id'],r['task']):r for r in records};typed=[r for r in records if r['task']=='typed_chat'];assert len(typed)==4
 for t in typed:
  s=by[(t['id'],'speech_chat')];r=rows[t['id']];assert t['user']==r['user'];assert s['user']==s['transcript'];assert t.get('history')==s.get('history') and t.get('image')==s.get('image');paired.append({'variant':variant,'id':t['id'],'typed_text':t['user'],'actual_asr':s['user'],'same_transcript':t['user']==s['user']})
# Count split family metadata only: no media/data/release audit.
families={s:{r['family'] for r in manifest['rows']+manifest['audio_rows'] if r['split']==s} for s in ['train','validation','test']};overlap={a+'-'+b:len(families[a]&families[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};assert all(n==0 for n in overlap.values())
final='docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/';test=read(final+'generations-base.json');audio=read(final+'transcripts.json');bygroup=Counter(group(r) for r in test);photos={r['image'] for r in test if r['task']=='scene'};voice={r['id'] for r in test if r['task']=='speech_chat'};assert len(test)==178 and len(photos)==42 and len(voice)==4 and len(audio)==22
voice_pairs=[]
for r in test:
 if r['task']=='speech_chat':
  typed=next(t for t in test if t['id']==r['id'] and t['task']=='typed_chat');voice_pairs.append({'id':r['id'],'typed_reference_equal_manifest':typed['user']==rows[r['id']]['user'],'speech_user_equal_actual_transcript':r['user']==r['transcript']})
guard=read('docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/source-selection-guard-before-test.json');assert sha('docs/natural-assistant/v4/selection.json')==guard['committed_pretest_selection_sha256'];assert sha('docs/natural-assistant/v4/manifest.json')==guard['files']['docs/natural-assistant/v4/manifest.json']['sha256']
result={'variants':computed,'selection_recomputed':selected,'paired_validation_routes':paired,'split_family_intersection_counts':overlap,'final_test_structure':{'answers':len(test),'groups':dict(bygroup),'unique_photo_images':len(photos),'unique_voice_questions':len(voice),'audio_recordings':len(audio),'voice_pairs':voice_pairs},'pretest_frozen_selection_sha256_matches_current':True,'original_bound_input_hashes':private['artifact_binding'],'scope':'CPU reaggregation of original recorded semantic grades with raw decoder token/EOS conditions and independent exact-match/numeric calculations. This is not fresh semantic image regrading or GPU generation; family audit is manifest metadata only.'}
(out/'record-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
