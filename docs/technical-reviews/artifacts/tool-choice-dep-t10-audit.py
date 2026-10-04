"""Bounded CPU audit of original T.10 receipts; no checkpoint inference/training."""
import copy
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.quantization import replace_linear_layers, pack_int4, unpack_int4
from scripts.course_experiments.common import records_sha256, split_records, text_examples, Context
from scripts.course_experiments.compression import _prompt, _steps
from scripts.course_experiments.modalities import modal_inputs, _steps as modal_steps
from scripts.course_experiments.run import experiment_spec

torch.set_num_threads(2)
tok = ByteTokenizer()
checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):
    return json.loads((ROOT / f'docs/course-experiments/results/{name}.json').read_text())
def raw(ids):
    return ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
def archive_records(asset, basename):
    path = ROOT / f'assets/training/{asset}-v1.tar.gz'
    with tarfile.open(path) as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(basename)]
        check(f'{asset}: unique source', len(members) == 1)
        content = archive.extractfile(members[0]).read()
    return [json.loads(line) for line in content.splitlines() if line.strip()], hashlib.sha256(content).hexdigest()

D, M = read('distillation'), read('multimodal_distillation')
dt, mt = D['results']['tasks'], M['results']['tasks']
out = {'scope': 'Recount persisted raw report IDs, source fingerprints, denominators and deterministic CPU mechanisms. No trained checkpoint was loaded, no inference reproduced, no optimizer update, no GPU/Modal call.', 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'threads': str(torch.get_num_threads())}, 'reports': {}, 'text': {}, 'modal': {}}
for name, report in [('distillation', D), ('multimodal_distillation', M)]:
    check(name+': complete L4 receipt', report['gpu']=='NVIDIA L4' and report['device']=='cuda' and report['seed']==42 and report['step_scale']==1 and report['evidence_status']=='complete_run' and report['status']=='completed' and report['unfinished_schedules']==[])
    historical, changed = {}, {}
    for path, digest in report['code_sha256'].items():
        original = subprocess.check_output(['git','show',f"{report['revision']}:{path}"],cwd=ROOT)
        actual = hashlib.sha256(original).hexdigest()
        check(name+': historical '+path, actual==digest)
        historical[path] = actual
        current = sha(ROOT/path)
        if current != digest:
            changed[path] = {'reported': digest, 'current': current}
    out['reports'][name] = {'sha256': sha(ROOT/f'docs/course-experiments/results/{name}.json'), 'revision': report['revision'], 'elapsed_seconds': report['elapsed_seconds'], 'timing_scope':report['timing_scope'], 'historical_fingerprints_verified':historical, 'current_source_changes':changed}

def text_eval(label, evaluation):
    samples = evaluation['generated_samples']
    hits = eos = invalid = 0
    for s in samples:
        exact = raw(s['generated_ids']) == tok.encode(s['expected'])
        ended = tok.eos_id in s['generated_ids']
        check(label+': raw flags '+str((s['family'],s['question'])), exact==s['exact'] and ended==s['ended_with_eos'])
        check(label+': visible answer', tok.decode(raw(s['generated_ids']))==s['generated'])
        hits += exact; eos += ended; invalid += sum(i<8 for i in raw(s['generated_ids']))
    check(label+': aggregates', hits==evaluation['correct'] and len(samples)==evaluation['examples'] and eos==evaluation['eos_count'])
    check(label+': NLL denominator', math.isclose(evaluation['answer_nll'], evaluation['nll_sum']/evaluation['supervised_tokens'],abs_tol=1e-12))
    return {'correct':hits,'examples':len(samples),'eos':eos,'invalid_control_ids':invalid,'targets':evaluation['supervised_tokens'],'chunks':evaluation['nll_sequence_chunks'],'answer_bytes':evaluation['answer_bytes'],'nll':evaluation['answer_nll']}

all_training=[]
upstream_map={'attributes':'sft','style_transfer':'style','moe_to_dense':'moe'}
for task,t in dt.items():
    upstream=read(upstream_map[task]); artifacts={a['path']:a for a in upstream['artifacts']}
    check(task+': original teacher checkpoint fingerprint',artifacts['model.pt']['sha256']==t['teacher_provenance']['sha256'])
    check(task+': original dataset fingerprint',artifacts['dataset.json']['sha256']==t['data']['sha256'])
    check(task+': frozen teacher',t['teacher_frozen_and_unchanged'])
    summary={'data':t['data'],'teacher':text_eval(task+': teacher',t['teacher_test']),'runs':{},'cache':t['teacher_cache']}
    for run,r in t['runs'].items():
        summary['runs'][run]=text_eval(task+': '+run,r['test'])
        summary['runs'][run]['storage']={k:r['storage'][k] for k in ['parameter_count','tensor_bytes','file_bytes','forward']}
        if 'training' in r:
            tr=r['training'];all_training.append(tr)
            check(task+': updates '+run,tr['steps']==tr['optimizer_updates'] and tr['weights_changed'] and tr['batch_size']==16 and tr['learning_rate']==0.003)
            summary['runs'][run]['training']={k:tr[k] for k in ['steps','effective_supervised_tokens','initialization_sha256','batch_plan_sha256','final_sha256','seconds']}
        if task!='moe_to_dense':
            check(task+': truth token denominator',r['test']['supervised_tokens']==sum(len(tok.encode(s['expected']))+1 for s in r['test']['generated_samples']))
    for width in ([16,32] if task=='attributes' else [32]):
        trains=[r['training'] for n,r in t['runs'].items() if n.startswith(f'w{width}_') and 'training' in r]
        for key in ['initialization_sha256','batch_plan_sha256','effective_supervised_tokens']:
            check(task+': matched '+str(width)+' '+key,len({r[key] for r in trains})==1)
    if t.get('hard_target_generation'):
        hard=t['hard_target_generation']; audit=hard['audit'];wrong=[]
        for a in audit:
            correct=raw(a['teacher_ids'])==tok.encode(a['gold_answer'])
            check(task+': hard raw labels',correct==a['teacher_correct'] and a['valid_target_tokens']==len(a['teacher_ids']) and a['eos']==(tok.eos_id in a['teacher_ids']))
            if not correct:wrong.append(a)
        check(task+': hard count',len(wrong)==hard['wrong'] and len(audit)-len(wrong)==hard['correct'])
        summary['hard']={'records':len(audit),'wrong':wrong,'seconds':hard['seconds'],'sha256':hard['sha256']}
    out['text'][task]=summary
check('11 students 3900 updates',len(all_training)==11 and sum(t['steps'] for t in all_training)==3900)
check('90.14 seconds rounded',round(D['elapsed_seconds'],2)==90.14)
a=dt['attributes'];check('attribute table',[round(a['teacher_test']['answer_nll'],4),a['teacher_test']['correct']]==[.5058,5])
for width,params,byte,nll in [(16,13744,54976,1.1134),(32,33632,134528,.4393)]:
    ce,hard=a['runs'][f'w{width}_ce'],a['runs'][f'w{width}_teacher_hard']
    check(f'width {width} CE/hard weights',ce['training']['final_sha256']==hard['training']['final_sha256'])
    check(f'width {width} table',ce['storage']['parameter_count']==params and ce['storage']['tensor_bytes']==byte and round(ce['test']['answer_nll'],4)==nll)
    check(f'width {width} targets',ce['training']['effective_supervised_tokens']==44985)
table=[(n,r['test']['correct'],round(r['test']['answer_nll'],4),r['storage']['tensor_bytes']) for n,r in a['runs'].items()]
check('KL and packed table',[(a['runs'][n]['test']['correct'],round(a['runs'][n]['test']['answer_nll'],4),a['runs'][n]['storage']['tensor_bytes']) for n in ['w16_ce_kl','w32_ce_kl','w32_ce_kl_packed4']]==[(1,1.2407,54976),(4,.5037,134528),(3,.5960,64160)])
ce,kl,packed=[a['runs'][n]['test']['generated_samples'] for n in ['w32_ce','w32_ce_kl','w32_ce_kl_packed4']]
rows=[]
for c,k,p,t in zip(ce,kl,packed,a['teacher_test']['generated_samples'],strict=True):
    check('paired family/question',(c['family'],c['question'],c['expected'])==(k['family'],k['question'],k['expected'])==(p['family'],p['question'],p['expected']))
    rows.append({'family':c['family'],'question':c['question'],'truth':c['expected'],'ce':c['generated'],'kl':k['generated'],'packed':p['generated'],'ce_exact':c['exact'],'kl_exact':k['exact'],'packed_exact':p['exact'],'ce_teacher_same_ids':c['generated_ids']==t['generated_ids'],'kl_teacher_same_ids':k['generated_ids']==t['generated_ids']})
check('teacher agreement 3 and 4',sum(r['ce_teacher_same_ids'] for r in rows)==3 and sum(r['kl_teacher_same_ids'] for r in rows)==4)
check('new correct and wrong',sum(not r['ce_exact'] and r['kl_exact'] for r in rows)==1 and sum(r['ce_exact'] and not r['kl_exact'] for r in rows)==1)
check('blue -> ble',any(r['kl']=='blue' and r['packed']=='ble' and r['kl_exact'] and not r['packed_exact'] for r in rows))
out['attribute_pairs']=rows
s=dt['style_transfer'];style_summary={}
for n,r in [('teacher',{'style':s['teacher_style'],'test':s['teacher_test']}),*[(n,r) for n,r in s['runs'].items() if 'training' in r]]:
    scored=[]
    for sample in r['test']['generated_samples']:
        ids=raw(sample['generated_ids']);text=tok.decode(ids);valid=all(i>=8 for i in ids);q=sample['question']
        if q.startswith('task=date;'):
            hit=ids==tok.encode(sample['expected']);kind='date';jvalid=False
        else:
            arithmetic=re.search(r'(\d+)\+(\d+)=\?',q);truth=sum(map(int,arithmetic.groups()));kind='arithmetic';jvalid=False
            if q.startswith('style=json;'):
                parsed=json.loads(text);jvalid=isinstance(parsed,dict) and set(parsed)=={'answer'} and type(parsed['answer']) is int and valid;hit=jvalid and parsed['answer']==truth
            elif q.startswith('style=vivid;'):hit=text.split('，',1)[0].strip()==str(truth) and valid
            else:hit=text.strip()==str(truth) and valid
        scored.append((kind,hit,jvalid,q,text))
    v={'content':sum(x[1] for x in scored),'arithmetic_correct':sum(x[1] for x in scored if x[0]=='arithmetic'),'arithmetic_total':sum(x[0]=='arithmetic' for x in scored),'date_correct':sum(x[1] for x in scored if x[0]=='date'),'date_total':sum(x[0]=='date' for x in scored),'json_valid':sum(x[2] for x in scored),'json_examples':sum(x[3].startswith('style=json;') for x in scored),'two_plus_two_json':[x[4] for x in scored if x[3]=='style=json; 2+2=?']}
    check('style '+n,list(v.values())[:7]==[3,0,21,3,6,7,7]);style_summary[n]=v
check('five wrong date hard answers',len(out['text']['style_transfer']['hard']['wrong'])==5 and all(x['question'].startswith('task=date;') for x in out['text']['style_transfer']['hard']['wrong']))
out['style_recount']=style_summary

stories,story_digest=archive_records('tinystories','tinystories-train-512.jsonl')
for record in stories:record['family']=record.get('text_sha256',records_sha256([{'text':record['text']}]))
parts=split_records(stories,42);moe=dt['moe_to_dense'];story_counts={}
for split in ['train','validation','test']:
    original=read('moe')['results']['dataset'][split];check('MoE original split sha '+split,records_sha256(parts[split])==original['sha256'])
    examples=text_examples(parts[split],max_length=128);bytes_=sum(len(r['text'].encode()) for r in parts[split]);targets=sum(int((y!=-100).sum()) for x,y in examples)
    check('story EOS one per document '+split,targets==bytes_+len(parts[split]))
    story_counts[split]={'documents':len(parts[split]),'chunks':len(examples),'bytes':bytes_,'targets':targets}
check('MoE test denominator',story_counts['test']=={'documents':52,'chunks':352,'bytes':41862,'targets':41914})
check('MoE validation denominator',story_counts['validation']=={'documents':51,'chunks':334,'bytes':39205,'targets':39256})
examples=text_examples(parts['train'][:32],max_length=128);rng=random.Random(42);plan=[[rng.randrange(len(examples)) for _ in range(16)] for _ in range(300)];effective=sum(int((examples[i][1]!=-100).sum()) for batch in plan for i in batch)
check('MoE effective targets',effective==559651)
out['story_source']={'archive_sha256':sha(ROOT/'assets/training/tinystories-v1.tar.gz'),'source_sha256':story_digest,'splits':story_counts,'student_documents':32,'student_chunks':len(examples),'student_effective_targets':effective}
gsm,gsm_digest=archive_records('gsm8k','gsm8k-train-first200.jsonl');eligible=[]
for index,row in enumerate(gsm):
    answer=row['answer'].split('####')[-1].strip().replace(',','');record={'question':row['question'],'answer':answer}
    if len(_prompt(record))+24<=128 and len(tok.encode(answer))+1<=24:eligible.append({'source_row':index,'question':row['question'],'answer':answer,'prefix_tokens':len(_prompt(record))})
check('GSM full source hash',gsm_digest==a['out_of_domain_gsm8k']['selection']['source_file_sha256'])
check('GSM eligible rows',[x['source_row'] for x in eligible]==[14,94]);check('GSM arithmetic answers',[x['answer'] for x in eligible]==['5','60'])
gsm_scores={'teacher':text_eval('gsm teacher',a['out_of_domain_gsm8k']['teacher'])}
for n,r in a['runs'].items():gsm_scores[n]=text_eval('gsm '+n,r['out_of_domain_gsm8k'])
check('GSM eight students and teacher',len(gsm_scores)==9 and all(x['correct']==0 and x['examples']==2 and x['eos']==2 for x in gsm_scores.values()))
out['gsm']={'source_rows':len(gsm),'eligible':eligible,'excluded':len(gsm)-len(eligible),'scores':gsm_scores}

def modal_eval(label,e):
    hits=eos=invalid=0
    for s in e['samples']:
        hit=raw(s['generated_ids'])==tok.encode(s['target']);ended=tok.eos_id in s['generated_ids'];bad=sum(i<8 for i in raw(s['generated_ids']))
        check(label+': raw sample',hit==s['exact_match'] and ended==s['eos'] and bad==s['invalid_special_tokens'])
        hits+=hit;eos+=ended;invalid+=bad
    check(label+': raw aggregate',hits==e['correct'] and len(e['samples'])==e['examples'] and eos/len(e['samples'])==e['eos_rate'] and invalid==e['invalid_special_tokens'])
    check(label+': complete',e['skipped']==[] and e['generation_errors']==0)
    check(label+': true target denominator',sum(len(tok.encode(s['target']))+1 for s in e['samples'])==e['effective_tokens'])
    return {'correct':hits,'examples':len(e['samples']),'eos':eos,'invalid':invalid,'targets':e['effective_tokens']}
for task,t in mt.items():
    upstream=read(task);artifacts={a['path']:a for a in upstream['artifacts']};check(task+': teacher source fingerprint',artifacts['model.pt']['sha256']==t['teacher_provenance']['sha256']);check(task+': dataset source fingerprint',artifacts['dataset.json']['sha256']==t['data']['sha256'])
    summary={'teacher':modal_eval(task+': teacher',t['teacher_test']),'data':t['data'],'runs':{}}
    original={s:v['records'] for s,v in upstream['results']['data']['splits'].items()}
    families=[{r['family'] for r in original[s]} for s in ['train','validation','test']];check(task+': disjoint families',all(not families[i]&families[j] for i in range(3) for j in range(i+1,3)))
    rng=random.Random(42);plan=[[rng.randrange(len(original['train'])) for _ in range(4)] for _ in range(350)];effective=sum(len(tok.encode(original['train'][i]['answer']))+1 for batch in plan for i in batch)
    for n,r in t['runs'].items():
        tr=r['training'];check(task+': full CPU-independent schedule',tr['steps']==350 and tr['optimizer_updates']==350 and tr['batch_size']==4 and tr['learning_rate']==.003 and tr['effective_answer_tokens']==effective)
        check(task+': storage',r['storage']['parameter_count']==36096 and r['storage']['tensor_bytes']==144384)
        summary['runs'][n]={'test':modal_eval(task+': '+n,r['test']),'blank_image':modal_eval(task+': '+n+' blank image',r['blank_image_test']),'training':tr}
        if task=='joint':summary['runs'][n]['blank_audio']=modal_eval(task+': '+n+' blank audio',r['blank_audio_test'])
    check(task+': same init and plan',all(t['runs']['ce']['training'][k]==t['runs']['ce_kl']['training'][k] for k in ['initialization_sha256','batch_plan_sha256','effective_answer_tokens']))
    check(task+': teacher storage',t['teacher_storage']['parameter_count']==145664 and t['teacher_storage']['tensor_bytes']==582656)
    out['modal'][task]=summary
check('modal table',(mt['vqa']['runs']['ce']['test']['correct'],mt['vqa']['runs']['ce_kl']['test']['correct'],mt['joint']['runs']['ce']['test']['correct'],mt['joint']['runs']['ce_kl']['test']['correct'])==(9,8,8,6))
j=mt['joint']['runs']['ce_kl'];pairs=[{'family':s['family'],'truth':s['target'],'normal':s['generated'],'blank_image':b['generated'],'identical_ids':s['generated_ids']==b['generated_ids']} for s,b in zip(j['test']['samples'],j['blank_image_test']['samples'],strict=True)]
check('joint KL all12 identical image ablation',all(x['identical_ids'] for x in pairs));check('joint square guessed circle',all(x['normal'].split(',')[0]=='circle' for x in pairs));check('joint blank audio falls3',j['blank_audio_test']['correct']==3)
check('joint teacher validation8',mt['joint']['teacher_validation']['correct']==8)
out['modal_ablation_pairs']=pairs

counts={}; models={}
for width,layers,expected in [(16,1,13744),(32,1,33632),(64,2,141568)]:
    model=TinyLM(ModelConfig(width=width,layers=layers,heads=2 if width!=64 else 1));count=sum(p.numel() for p in model.parameters());check('CPU parameter count '+str(width),count==expected);counts[str(width)]=count;models[width]=model
packed=replace_linear_layers(copy.deepcopy(models[32]),4);byte=sum(t.numel()*t.element_size() for t in list(packed.parameters())+list(packed.buffers()));check('CPU packed storage',byte==64160)
check('CPU packing preserves layer count',len(packed.blocks)==len(models[32].blocks)==1)
with torch.no_grad():
    check('CPU packed floating forward',packed(torch.tensor([[1,2]]))['logits'].dtype==torch.float32)
q=torch.tensor([-8,-1,0,1,7,2,3],dtype=torch.int8);p=pack_int4(q);check('CPU pack roundtrip',p.tolist()==[112,152,175,139] and torch.equal(unpack_int4(p,q.shape),q))
teacher=MultiModalLM(models[64]);student=MultiModalLM(models[32],vision_width=8,audio_width=8);student.vision=VisionEncoder(8,image_size=16,patch_size=8)
check('CPU modal parameter counts',sum(p.numel() for p in teacher.parameters())==145664 and sum(p.numel() for p in student.parameters())==36096)
alignment={}
for task in ['vqa','joint']:
    record=read(task)['results']['data']['splits']['train']['records'][0];ids,labels,image,waveform,_=modal_inputs(record,SimpleNamespace(device='cpu'));selected=[]
    with torch.no_grad():
        for model in [teacher,student]:
            o=model(ids,labels,image=image,waveform=waveform);valid=o['labels'][0]!=-100;selected.append({'rows':valid.nonzero().flatten().tolist(),'answer_ids':o['labels'][0,valid].tolist()})
    check('CPU real modal answer alignment '+task,selected[0]['answer_ids']==selected[1]['answer_ids'] and selected[0]['rows'][0]-selected[1]['rows'][0]==12);alignment[task]=selected
torch.manual_seed(42);sl=torch.randn(1,3,4);tl=torch.randn(1,3,4);labels=torch.tensor([[0,-100,1]]);T=2.;p=(tl/T).softmax(-1);lq=(sl/T).log_softmax(-1);expected=(p*(p.log()-lq)).sum(-1)[labels!=-100].mean()*T*T;observed=distillation_kl(sl,tl,labels,T);check('CPU KL once T squared',torch.allclose(expected,observed,atol=1e-6,rtol=1e-6))
specs={n:experiment_spec(n) for n in ['distillation','multimodal_distillation','encoders','projector','vqa','joint']};check('lookup current plan dependencies',specs['distillation']['dependencies']==['sft','style','moe'] and specs['multimodal_distillation']['dependencies']==['vqa','joint']);check('CPU compression full versus modal smoke',_steps(SimpleNamespace(step_scale=1),350)==350 and modal_steps(SimpleNamespace(device='cpu',step_scale=1),160,8)==8)
old_plan=json.loads(subprocess.check_output(['git','show',D['revision']+':docs/course-experiments/plan.json'],cwd=ROOT,text=True))
plan_differences={}
for name,spec in specs.items():
    old=next(s for s in old_plan['sequence'] if s['id']==name)
    delta={k:[old.get(k),spec.get(k)] for k in set(old)|set(spec) if old.get(k)!=spec.get(k)}
    check('current plan preserves historical dispatch '+name,set(delta)<={'status','evidence'})
    plan_differences[name]=delta
try:Context('cpu',ROOT/'outputs',ROOT/'outputs/tool-choice-dep-t10-nonexistent',ROOT/'assets/training').dependency('sft')
except FileNotFoundError:checks.append('missing checkpoint stops')
else:raise AssertionError('missing checkpoint did not stop')
out['cpu_mechanisms']={'parameter_counts':counts,'packed_tensor_bytes':byte,'packed_example':[112,152,175,139],'alignment':alignment,'kl_expected':float(expected),'kl_observed':float(observed),'current_specs':specs,'plan_sha256':sha(ROOT/'docs/course-experiments/plan.json'),'historical_plan_differences':plan_differences}
from scripts.course_release import inference_payload
fixture={'format_version':1,'config':{'vocab_size':264},'model':{'example':torch.ones(1)},'optimizer':{'example':1},'torch_rng':torch.ones(1),'python_rng':42,'metadata':{'report':{'example':1},'scope':'fixture'}}
clean=inference_payload(fixture,{'review_fixture':'T.10'})
check('public inference strips training state',not any(k in clean for k in ['optimizer','torch_rng','python_rng']) and 'report' not in clean['metadata'])
out['cpu_mechanisms']['public_payload_keys']=list(clean)
out['assertion_count']=len(checks);out['assertion_result']={'passed':len(checks),'failed':0}
Path(__file__).with_name('tool-choice-dep-t10-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'assertions':len(checks),'text_student_updates':sum(t['steps'] for t in all_training),'gsm_eligible_rows':[x['source_row'] for x in eligible],'modal_image_ablation_identical':sum(x['identical_ids'] for x in pairs),'environment':out['environment']},ensure_ascii=False))
