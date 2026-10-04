from pathlib import Path
import sys,json,hashlib,importlib.util
from collections import Counter
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from tiny_perceptron import natural_assistant as core
spec=importlib.util.spec_from_file_location('scorer',ROOT/'scripts/score_natural_v4_validation.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=load(ROOT/'docs/natural-assistant/v4/manifest.json');p=load(ROOT/'docs/natural-assistant/v4/validation-protocol-lower-lr.json');cases,audio=s.expected_cases(m,p)
directory=ROOT/'outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review';blind=ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review'
grades=load(blind/'combined/grades.json');mapping=load(blind/'private-map.json')['aliases'];ledger={(g['case_id'],g['candidate']):g['passed'] for g in grades['grades']};reverse={v:k for k,v in mapping.items()};stored=load(blind/'scored/scores.json')
transcripts=load(directory/'transcripts.json');ts={r['id']:r for r in transcripts};den=dict(Counter(c['group'] for c in cases.values()));scores={};hashes={}
assert len(ledger)==303 and len(ts)==16
for variant in p['candidate_order']:
    file=directory/f'generations-{variant}.json';hashes[file.relative_to(ROOT).as_posix()]=sha(file);data=load(file)
    idx={s.case_id(r['id'],r['task']):r for r in data};assert len(data)==len(idx)==132 and idx.keys()==cases.keys()
    counts={g:0 for g in s.WEIGHTS};incomplete=[]
    for key,case in cases.items():
        record=idx[key];row=case['row'];group=case['group']
        assert record['user']==(ts[row['id']]['transcript'] if group=='voice_actual_asr_chat' else row['user'])
        complete=s.eos_complete(record,384)
        if not complete:incomplete.append(key)
        passed=ledger[(key,reverse[variant])] if group in s.MANUAL else core.score_output(row,record['prediction'])['passed']
        counts[group]+=int(passed and complete)
    photo=(F(counts['photo_summary'],den['photo_summary'])+F(counts['photo_fact'],den['photo_fact']))/2
    chat=sum(F(counts[g],den[g]) for g in ['text_chat','voice_typed_reference_chat','voice_actual_asr_chat'])/3
    primary=(photo+F(counts['text_presence'],den['text_presence'])+F(counts['single_ocr'],den['single_ocr'])+F(counts['ordered_ocr'],den['ordered_ocr'])+chat)/5
    numerator=sum(counts[g]*s.WEIGHTS[g] for g in counts)
    assert primary==F(numerator,75600) and counts==stored['variants'][variant]['correct_counts']
    assert sorted(incomplete)==sorted(stored['variants'][variant]['incomplete_case_ids'])
    scores[variant]={'correct_counts':counts,'primary_fraction':str(primary),'primary_integer_numerator':numerator,'primary_integer_denominator':75600,'incomplete_count':len(incomplete),'all_EOS':not incomplete}
base=scores['base'];selected='base'
for variant in p['candidate_order'][1:]:
    d=scores[variant];failed=[]
    for group in counts:
        allowance=int(p['adapter_nonregression_gates_vs_base'][group+'_correct_count_minimum']=='base minus 1')
        if d['correct_counts'][group]<base['correct_counts'][group]-allowance:failed.append(group)
    d['failed_count_gates']=failed;d['eligible']=d['all_EOS'] and not failed
    if d['eligible'] and d['primary_integer_numerator']>scores[selected]['primary_integer_numerator']:selected=variant
assert selected==stored['selected_variant']==load(ROOT/'docs/natural-assistant/v4/selection.json')['selected_variant']=='base'
runtime=load(directory/'result.json');training=load(ROOT/'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review/result.json')
assert runtime['status']=='completed' and runtime['split']=='validation'
sources=[directory/'result.json',directory/'transcripts.json',blind/'combined/grades.json',blind/'private-map.json',blind/'scored/scores.json',ROOT/'docs/natural-assistant/v4/validation-protocol-lower-lr.json',ROOT/'docs/natural-assistant/v4/selection.json']
hashes.update({f.relative_to(ROOT).as_posix():sha(f) for f in sources})
print(json.dumps({'scope':'Re-audit of ORIGINAL benchmark generation and semantic grade ledger. No new blinded photo/chat grading and no personal GPU or ASR replication.','validation_run':runtime['execution']['run_id'],'denominators':den,'generations_per_candidate':132,'ASR_recordings':16,'original_semantic_grade_entries':len(ledger),'all_actual_ASR_inputs_match_single_transcript_ledger':True,'variants':scores,'independently_recomputed_selected_variant':selected,'training_runner_subprocess_seconds':training['execution']['seconds'],'training_runner_subprocess_minutes':training['execution']['seconds']/60,'source_sha256':hashes},ensure_ascii=False,indent=2))
