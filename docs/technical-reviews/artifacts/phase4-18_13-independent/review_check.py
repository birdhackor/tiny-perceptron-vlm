"""Bounded CPU verification of 18.13. No training or checkpoint inference."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.multimodal import expand_modalities, log_mel, tone

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):
    return json.loads((ART / 'inputs/docs/course-experiments/results' / name).read_bytes())
def report(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))

environment = {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
               'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'threads': '1',
               'cuda_build': str(torch.version.cuda), 'check_scope': 'raw measurements and short CPU tensor checks only'}
(ART/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
report('environment', environment)
meta = json.loads((ART/'extracted/extraction.json').read_bytes())
assert sha(ART/'extracted/section.md') == meta['source_sha256']
assert sha(ART/'inputs/course/chapters/18.md') == meta['source_file_sha256']
report('frozen_input', {'section_sha256': meta['source_sha256'], 'whole_chapter_sha256': meta['source_file_sha256'],
                        'whole_chapter_scope': 'frozen bytes only; reviewer read 18.13, not whole chapter'})

# Execute the untouched source fence, followed by the requested prefix variation.
fence=(ART/'extracted/fence-1.py').read_bytes()
assert hashlib.sha256(fence).hexdigest()==meta['python_fences'][0]['sha256']
for name, code in [('original',fence),('student_prefix_6',fence.replace(b'teacher_prefix, student_prefix, answer_tokens = 16, 4, 3',
                                                                      b'teacher_prefix, student_prefix, answer_tokens = 16, 6, 3'))]:
    namespace={}; buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer): exec(compile(code,name,'exec'),namespace)
    value=distillation_kl(namespace['s_answer'],namespace['t_answer'],namespace['labels'],temperature=1).item()
    assert namespace['t_rows']==[15,16,17]
    assert namespace['s_rows']==([3,4,5] if name=='original' else [5,6,7])
    assert tuple(namespace['s_answer'].shape)==(1,3,2)
    assert abs(value-(0.8*math.log(1.6)+0.2*math.log(0.4)))<1e-7
    report('fence_'+name,{'stdout':buffer.getvalue(),'kl':value})
try: distillation_kl(torch.zeros(1,7,2),torch.zeros(1,19,2),torch.zeros(1,7,dtype=torch.long))
except ValueError as exc: report('unaligned_shape_rejected',str(exc))
else: raise AssertionError('expected shape rejection')
report('analytic_kl_nats',0.8*math.log(1.6)+0.2*math.log(0.4))

m=load('multimodal_distillation.json');tok=ByteTokenizer()
for filename in ['multimodal_distillation.json','vqa.json','joint.json','encoders.json']:
    d=load(filename)
    report('raw_file_'+filename,{'sha256':sha(ART/'inputs/docs/course-experiments/results'/filename),
                                'top_level_key_types':{k:type(v).__name__ for k,v in d.items()}})
for code_name in ['tiny_perceptron/alignment.py','tiny_perceptron/multimodal.py','tiny_perceptron/data.py',
                  'scripts/course_experiments/modalities.py']:
    assert sha(ROOT/code_name)==m['code_sha256'][code_name]
assert sha(ART/'sources/compression-measured.py')==m['code_sha256']['scripts/course_experiments/compression.py']
report('measurement_provenance',{k:m[k] for k in ['revision','seed','device','torch_version','python_version','step_scale','evidence_status','unfinished_schedules']})

# Extract only necessary method AST, bypassing unrelated result interpretation strings.
tree=ast.parse((ART/'sources/modalities-joint-measured.py').read_bytes())
seq=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sequence')
ns={'torch':torch,'ByteTokenizer':ByteTokenizer}
exec(compile(ast.Module(body=[seq],type_ignores=[]),'original:_sequence','exec'),ns)

for task in ['vqa','joint']:
    parts={split:v['records'] for split,v in load(task+'.json')['results']['data']['splits'].items()}
    serialized=(json.dumps(parts,ensure_ascii=False,indent=2)+'\n').encode()
    assert hashlib.sha256(serialized).hexdigest()==m['results']['tasks'][task]['data']['sha256']
    dataset_file=ART/'sources'/(task+'-dataset-reconstructed-from-raw-records.json')
    assert dataset_file.read_bytes()==serialized
    families={k:set(r['family'] for r in records) for k,records in parts.items()}
    assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert {k:len(v) for k,v in parts.items()}==m['results']['tasks'][task]['data']['counts']
    report(task+'_data',{'dataset_sha256':hashlib.sha256(serialized).hexdigest(),
                       'counts':{k:len(v) for k,v in parts.items()},'families':{k:len(v) for k,v in families.items()},
                       'cross_split_intersections':0,'test_frequency_hz':sorted({r['frequency'] for r in parts['test']}) if task=='joint' else []})
    alignment=m['results']['tasks'][task]['runs']['ce_kl']['training']['alignment']
    row=next(r for r in parts['train'] if r['family']==alignment['family'] and tok.encode(r['answer'])+[tok.eos_id]==alignment['answer_ids'])
    ids,labels,count=ns['_sequence'](row,SimpleNamespace(device='cpu'))
    frames=None if task=='vqa' else log_mel(tone(row['frequency'])).shape[-1]
    derived={}
    embedding=torch.nn.Embedding(tok.vocab_size,8)
    for role,nvisual in [('teacher',16),('student',4)]:
        features={tok.image_id:torch.zeros(nvisual,8)}
        if frames is not None:features[tok.audio_id]=torch.zeros(frames,8)
        _,y=expand_modalities(ids,labels,embedding,features,{tok.image_id,tok.audio_id})
        valid=y[0]!=IGNORE
        positions=valid.nonzero().flatten().tolist();answers=y[0,valid].tolist()
        assert positions==alignment[role+'_prediction_rows'] and answers==alignment['answer_ids']
        derived[role]={'prediction_rows':positions,'answer_ids':answers,'input_answer_start':positions[0]+1}
    report(task+'_real_alignment',{'row':row,'audio_frames':frames,'effective_answer_tokens':count,'derived':derived})
    ce=m['results']['tasks'][task]['runs']['ce']['training'];kl=m['results']['tasks'][task]['runs']['ce_kl']['training']
    controls=['steps','optimizer_updates','batch_size','learning_rate','training_examples','effective_answer_tokens','initialization_sha256','batch_plan_sha256']
    assert all(ce[k]==kl[k] for k in controls)
    plan_rng=random.Random(m['seed']);plan=[[plan_rng.randrange(len(parts['train'])) for _ in range(4)] for _ in range(kl['steps'])]
    assert hashlib.sha256(json.dumps(plan).encode()).hexdigest()==kl['batch_plan_sha256']
    tokens=sum(len(tok.encode(parts['train'][i]['answer']))+1 for indices in plan for i in indices)
    assert tokens==kl['effective_answer_tokens']
    report(task+'_controls',{k:kl[k] for k in controls})
    results=[]
    for test in ['test','blank_image_test']:
        q=m['results']['tasks'][task]['runs']['ce_kl'][test]
        assert len(q['samples'])==q['examples']==len(parts['test'])==12
        effective=sum(len(tok.encode(r['answer']))+1 for r in parts['test']);assert effective==q['effective_tokens']
        recomputed_correct=0;shapes=[];pitch_correct=0
        for index,(sample,rawrow) in enumerate(zip(q['samples'],parts['test'],strict=True)):
            assert sample['row']==index and sample['family']==rawrow['family']
            assert sample['target']==rawrow['answer'] and sample['question']==rawrow['question']
            generated=sample['generated_ids'];before_eos=generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
            assert tok.decode(before_eos)==sample['generated']
            exact=before_eos==tok.encode(rawrow['answer']);assert exact==sample['exact_match'];recomputed_correct+=exact
            assert (tok.eos_id in generated)==sample['eos']
            if task=='joint':
                shape,pitch=sample['generated'].split(',');shapes.append(shape);pitch_correct+=pitch==rawrow['answer'].split(',')[1]
        assert recomputed_correct==q['correct'] and recomputed_correct/12==q['exact_match']
        if task=='joint':
            assert pitch_correct==q['pitch_correct']==12
            assert q['shape_correct']==6 and q['shape_accuracy']==0.5 and q['pitch_accuracy']==1
            assert set(shapes)=={'circle'}
        results.append({'condition':test,'examples':12,'correct':recomputed_correct,'effective_answer_tokens':effective,
                        'pitch_correct':pitch_correct if task=='joint' else None,'shapes':dict(Counter(shapes)),
                        'generation_errors':q['generation_errors'],'eos_rate':q['eos_rate']})
    a=m['results']['tasks'][task]['runs']['ce_kl']['test']['samples'];b=m['results']['tasks'][task]['runs']['ce_kl']['blank_image_test']['samples']
    equal=sum(x['generated_ids']==y['generated_ids'] for x,y in zip(a,b,strict=True))
    if task=='joint':assert equal==12
    report(task+'_raw_score_recomputation',{'results':results,'equal_normal_blank_id_sequences':equal,'scoring':'exact UTF-8 byte ids before EOS; component scores require two comma-separated fields'})

encoder=load('encoders.json')['results']['audio'];joint=load('joint.json')['results']['data']['splits']
train=encoder['data']['splits']['train']['records'];enc_frequencies={r['frequency'] for r in train}
test_frequencies={r['frequency'] for r in joint['test']['records']};joint_train_frequencies={r['frequency'] for r in joint['train']['records']}
assert test_frequencies=={260,340} and test_frequencies<=enc_frequencies and not test_frequencies&joint_train_frequencies
counts=Counter(r['frequency'] for step in range(encoder['training']['steps']) for r in random.Random(m['seed']+step).choices(train,k=8))
assert all(counts[f]>0 for f in test_frequencies)
report('frequency_exposure',{'joint_train_hz':sorted(joint_train_frequencies),'joint_test_hz':sorted(test_frequencies),
                            'encoder_train_hz':sorted(enc_frequencies),'encoder_training_sampling_counts':dict(counts),
                            'encoder_steps':encoder['training']['steps'],'encoder_effective_targets':encoder['training']['effective_targets'],
                            'scope':'joint teacher inherits trained 16-wide audio encoder; student audio encoder is initialized afresh with width 8 in measured run'})
print('ALL BOUNDED CPU AND RAW-MEASUREMENT ASSERTIONS PASSED')
