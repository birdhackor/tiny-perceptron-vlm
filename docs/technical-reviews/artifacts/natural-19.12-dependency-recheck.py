"""Original owner's chapter20 dependency recheck, with no model inference or Git actions."""
from collections import Counter
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/technical-reviews/artifacts'
PRIOR='e05125f21927bff51807c9f7657d493ad2d2959ec6b3867afe7ec2609e1b7688'
HISTORY=OUT/'natural-19.12-history'/PRIOR
OLD_PATH=HISTORY/'report.json'
OLD=json.loads(OLD_PATH.read_text())
assert hashlib.sha256(OLD_PATH.read_bytes()).hexdigest()==PRIOR
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def dist(a,b):
    last=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(last[j]+1,row[-1]+1,last[j-1]+(x!=y)))
        last=row
    return last[-1]
def norm(x):return ''.join(unicodedata.normalize('NFKC',x).split())
def completion(row):
    ids=row['generated_token_ids'];assert len(ids)==row.get('generated_tokens',row.get('generated_token_count'))
    eos=bool(ids) and ids[-1] in row['eos_token_ids']
    assert eos==row['ended_with_eos'] and (row['stop_reason']=='eos')==eos
    assert row['completion_unknown'] is False
    assert not(row['truncated'] and eos)
    return eos

body20=(ROOT/'course/chapters/20.md').read_bytes();sha20=hashlib.sha256(body20).hexdigest()
chapter=body20.decode();heads=list(re.finditer(r'^## (20\.\d+) .*$',chapter,re.M))
assert [h[1] for h in heads]==[f'20.{n}' for n in range(1,9)]
section_hashes={h[1]:hashlib.sha256(chapter[h.start():heads[i+1].start() if i+1<len(heads) else len(chapter)].encode()).hexdigest() for i,h in enumerate(heads)}
snapshot=OUT/f'natural-19.12-dependency-current20-{sha20[:12]}.md'
assert snapshot.is_file() and snapshot.read_bytes()==body20,'Read/snapshot the current complete chapter before this audit.'
raw19=(ROOT/'course/chapters/19.md').read_text();body19=re.search(r'^## 19\.12 .*?(?=^## |\Z)',raw19,re.M|re.S)[0]
assert hashlib.sha256(body19.encode()).hexdigest()==OLD['source_sha256']
unaffected=[]
for item in OLD['artifacts']+[s for s in OLD['sources'] if s['kind']=='repository_code' and s['id']!='s_20']:
    assert sha(item['path'])==item['sha256'];unaffected.append(item['id'])
for path,digest in OLD['figure_sha256'].items():assert sha(path)==digest

train=read('docs/natural-assistant/evidence/train/result.json')
selection=read('docs/natural-assistant/selection.json')
release=read('docs/natural-assistant/public-release.json')
publication=read('docs/natural-assistant/evidence/release/result.json')
trial=read('docs/natural-assistant/evidence/student-trial/student-trial-summary.json')
initial,final=train['initial_adapter_tensors'],train['final_adapter_tensors']
assert len(initial)==len(final)==112 and set(initial)==set(final)
assert set(train['optimizer_parameter_names'])==set(initial)==set(train['trainable_parameter_names'])
assert all('lora_' in name and any(p in name for p in ['q_proj','v_proj']) for name in initial)
updated=sum(initial[k]['sha256_values']!=final[k]['sha256_values'] for k in initial)
parameters=sum(math.prod(v['shape']) for v in initial.values())
assert parameters==1_605_632 and updated==112 and train['trainable_parameters']==parameters
assert train['optimizer_only_lora'] and train['frozen_parameter_samples_unchanged']
assert train['completed_steps']==train['requested_steps']==180 and train['status']=='completed'
base_parameters=train['total_parameters']-parameters
assert base_parameters==2_127_532_032 and 'Dense pretrained Qwen3-VL' in train['architecture']
assert train['model']=='Qwen/Qwen3-VL-2B-Instruct' and train['asr_model']=='openai/whisper-small'
assert train['model_revision']==release['base_model']['revision']==selection['base_model']['revision']
assert train['asr_revision']==release['asr_model']['revision']==selection['asr_model']['revision']
assert selection['selected_variant']=='adapter' and selection['adapter_run_id']==train['execution']['run_id']
adapter_hash=next(f['sha256'] for f in release['files'] if f['output']=='adapter_model.safetensors')
assert adapter_hash==selection['adapter_sha256']
assert publication['status']=='completed' and publication['release']['revision']==release['revision']==trial['public_revision']=='8dab26b439499188149b7ac5b861dab4f6c0755a'
assert publication['release']['anonymous_download_verified'] and release['anonymous_download_verified']
assert sum(f['bytes'] for f in release['files'])==6_459_450
for f in release['files']:
    matched=next(x for x in trial['actual_downloads']['adapter_files'] if x['path']==f['path'])
    assert matched['sha256']==f['sha256'] and matched['bytes']==f['bytes'] and matched['revision']==release['revision'] and matched['token'] is False

generations=read('docs/natural-assistant/evidence/final/generations-adapter.json')
assert len(generations)==90
gen_by_key={(r['id'],r['task']):r for r in generations}
assert len(gen_by_key)==90
for r in generations:completion(r)
annotations=read('docs/natural-assistant/evidence/semantic-final/final-semantic-review.json')['all_records']
annotated=[r for r in annotations if r['variant']=='adapter'];assert len(annotated)==90
counts={}
for r in annotated:
    raw=r['raw_generation_record_preserved'];assert raw==gen_by_key[(r['id'],r['task'])]
    expected=bool(r['semantic_pass'] and r['instruction_pass'] and completion(raw))
    assert expected==r['completed_task_success']
    d=counts.setdefault(r['task'],{'count':0,'completed_success_under_recorded_labels':0})
    d['count']+=1;d['completed_success_under_recorded_labels']+=int(expected)
assert counts['scene']=={'count':36,'completed_success_under_recorded_labels':17}
assert counts['speech_chat']=={'count':12,'completed_success_under_recorded_labels':1}
assert counts['typed_chat']=={'count':12,'completed_success_under_recorded_labels':4}
open_rows=[r for r in annotated if r['task']=='scene' and r['id'].endswith('/scene')]
assert len(open_rows)==12 and not any(r['completed_task_success'] for r in open_rows)
assert sum(r['raw_generation_record_preserved']['truncated'] for r in open_rows)==5
ocr=[r for r in generations if r['task']=='ocr'];assert len(ocr)==18
assert all(r['prediction']==r['reference_answer'] and completion(r) for r in ocr)
external=[r for r in read('docs/natural-assistant/evidence/external-ocr/generations.json') if r['variant']=='adapter']
assert len(external)==10
external_errors=0;external_reference=0;external_exact=0
for r in external:
    completion(r);external_errors+=dist(r['reference_answer'],r['prediction']);external_reference+=len(r['reference_answer'])
    external_exact+=int(r['reference_answer']==r['prediction'] and completion(r))
assert (external_exact,external_errors,external_reference)==(1,729,484)
transcripts=read('docs/natural-assistant/evidence/final/transcripts.json');assert len(transcripts)==12
raw_errors=raw_chars=norm_errors=norm_chars=0
speech=[r for r in generations if r['task']=='speech_chat'];typed=[r for r in generations if r['task']=='typed_chat']
for t in transcripts:
    completion(t);a=t['reference_transcript'];b=t['transcript']
    raw_errors+=dist(a,b);raw_chars+=len(a);norm_errors+=dist(norm(a),norm(b));norm_chars+=len(norm(a))
    spoken=next(r for r in speech if r['id']==t['id']);control=next(r for r in typed if r['id']==t['id'])
    assert spoken['user']==spoken['transcript']==b and spoken['reference_user']==a
    assert control['user']==a and control['image']==spoken['image']==None
assert (raw_errors,raw_chars,norm_errors,norm_chars)==(150,481,129,464)

events=[json.loads(s) for s in (ROOT/'docs/natural-assistant/evidence/student-trial/events.jsonl').read_text().splitlines()]
event_by_index={r['index']:r for r in events}
core=[r for r in events if r['event']=='actual_core_generation']
asr=[r for r in events if r['event']=='actual_asr_transcription']
assert len(core)==8 and len(asr)==2
assert trial['status']=='actual_selected_public_cpu_student_trial_complete'
assert trial['actual_environment']['device']=='cpu' and trial['actual_environment']['dtype']=='float32'
assert trial['workflow']['true_ui_chat_count']==8 and trial['workflow']['true_asr_count']==2
original_ui=trial['original_ui_core_outputs']
for r in original_ui:
    event=event_by_index[r['core_event_index']];assert event['event']=='actual_core_generation'
    assert r['response']['prediction']==event['result']['prediction'];completion(event['result'])
    assert event['original_function_called_once'] and event['returned_result_unchanged']
ocr_trial=next(r for r in original_ui if r['case']=='traditional-chinese-ocr')
original_speech=next(r for r in original_ui if r['case']=='speech-original-chat')
edited_speech=next(r for r in original_ui if r['case']=='speech-edited-chat')
assert ocr_trial['response']['prediction']=='這是一張中文的圖片。'
assert original_speech['submitted_editor_text']==original_speech['original_hypothesis']
assert original_speech['response']['prediction']=='這句話是錯誤的。'
assert edited_speech['submitted_editor_text'].startswith(original_speech['original_hypothesis'])
assert trial['workflow']['reset_empty_history_and_no_old_image']

proof={
 'reviewer_task':'/root/natural_factual_19_12','reviewer_stage':'original-owner dependency recheck',
 'prior_report_sha256':PRIOR,'prior_report_archive':str(OLD_PATH.relative_to(ROOT)),
 'prior_dependency_sha256':next(s['sha256'] for s in OLD['sources'] if s['id']=='s_20'),
 'history_limits':{'old_full_20_snapshot_retained':False,'old_complete_source_available':False,
   'original_read_scope':['20.1','20.2'],'full_old_to_new_text_diff_available':False,
   'reason':'First review retained the dependency full-file hash but only read20.1/20.2 and did not preserve the complete chapter20 source. Parent found no matchingb821... snapshot; the old Git public source URL returned404. No old whole-source text or full-reading history is invented. Exact archived report claim/source/checks are available for real change comparison.'},
 'current_source':{'chapter20_path':'course/chapters/20.md','chapter20_full_sha256':sha20,'read_scope':'Intro and all20.1–20.8, including latest20.8 bfloat16/float32/float16 explanations','snapshot':str(snapshot.relative_to(ROOT)),'section_sha256':section_hashes,'19.12_raw_sha256':OLD['source_sha256'],'19.12_unchanged':True},
 'unaffected_registered_evidence_hashes_checked':unaffected,
 'route_and_training':{'base_repo':train['model'],'base_revision':train['model_revision'],'dense_base_parameters':base_parameters,'only_course_updated_parameters':parameters,'q_v_adapter_tensors':112,'changed_adapter_tensors':updated,'optimizer_only_lora':True,'frozen_base_check_scope':train['frozen_check_scope'],'asr_repo':train['asr_model'],'asr_parameters':241734912,'asr_course_updated':False,'inference_text_core_receives':'actual ASR transcript; waveform stays in separateWhisper','old_chapter19_moe_parameters':328128,'old_capstone_weights_used_as_parent':False},
 'capability_boundaries':{'counts_under_saved_frozen_semantic_labels':counts,'semantic_scope':'Count/conjunction/identity audit of original experiment annotations and generation records; not a substitute review of all photograph semantics or other chapter technical/reader reviews. No model forward executed.','scene_unique_photos':12,'scene_total':[17,36],'open_scene_total':[0,12],'open_scene_truncated':5,'synthetic_ocr_exact':[18,18],'external_ocr_raw_exact':[1,10],'external_ocr_raw_errors_reference':[729,484],'raw_asr_cer_errors_reference':[150,481],'normalized_asr_cer_errors_reference':[129,464],'speech_chat':[1,12],'typed_speech_control':[4,12],'limitations':'Small fixed photo/letter-card/reading sets and external annotation ambiguities; source-disjoint course split does not prove upstream pretraining disjointness. EOS does not certify semantic success. Chapter19 synthetic small-world scores are not transferred to chapter20.'},
 'actual_release_cpu_trial':{'public_revision':release['revision'],'public_adapter_sha256':adapter_hash,'release_status':publication['status'],'actual_trial_status':trial['status'],'matching_adapter_file_receipts':4,'adapter_bytes':6459450,'actual_core_events':len(core),'actual_asr_events':len(asr),'raw_trial_predictions':{'new_ocr_card':ocr_trial['response']['prediction'],'unmodified_asr_chat':original_speech['response']['prediction'],'user_edited_asr_chat':edited_speech['response']['prediction']},'scope':'Recorded actual public CPU workflow verified from linked raw events; this recheck does not repeat inference or download weights. Successful publication/UI does not certify natural-task reliability.'},
 'original_failures':{key:gen_by_key[(key,'scene')]['prediction'] for key in ['vision:docci/test_00729/scene','vision:docci/test_01103/fact-1','vision:docci/test_01103/fact-2']},
 'environment':{'python':sys.version.split()[0],'device':'cpu','libraries':'Python standard library JSON, Unicode, hashlib; no GPU or model load'},
 'result':'All dependency identity, arithmetic, raw completion/route, original failure preservation and release/trial event assertions passed.'
}
path=OUT/'natural-19.12-dependency-recheck.json';path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'current20_sha256':sha20,'old_full_snapshot_available':False,'19.12_unchanged':True,'updated_parameters':parameters,'scene':[17,36],'open_scene':[0,12],'synthetic_ocr':[18,18],'external_ocr':[1,10],'asr_chat':[1,12],'actual_public_cpu_trial_verified':True,'result':'passed'},indent=2))
