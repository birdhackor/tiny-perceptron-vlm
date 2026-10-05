"""Own arithmetic/input audit of fixed original course run records, not GPU replication."""
import copy
import hashlib
import json
import math
import platform
import random
from pathlib import Path
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.tokenization import ByteLevelBPE

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
BASE = ROOT/'outputs/text-behavior-interface-check'
REPORTS = {n:json.loads((ROOT/'docs/course-experiments/results'/(n+'.json')).read_text()) for n in ['text_foundation','real_text','tokenizer','sft','sft_ablation']}
TOK = ByteTokenizer()
receipt = {'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'}, 'original_reports':{}, 'inputs':{}, 'evaluations':{}, 'training':{}, 'source_assets':{}, 'calculations':{}}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def canonical(rs): return sha(json.dumps(rs,ensure_ascii=False,sort_keys=True).encode())
def load_parts(exp,directory):
    d = REPORTS[exp]; artifacts = {a['path']:a for a in d['artifacts']}
    parts = {}; groups = []
    for split in ['train','validation','test']:
        rel = directory+'/'+split+'.jsonl'; path = BASE/exp/rel
        raw = path.read_bytes(); rs=rows(path); digest=sha(raw)
        assert digest == artifacts[rel]['sha256'] and len(raw) == artifacts[rel]['bytes']
        parts[split]=rs; fam={r['family'] for r in rs};groups.append(fam)
        receipt['inputs'][exp+'/'+rel]={'path':str(path.relative_to(ROOT)),'sha256':digest,'bytes':len(raw),'records':len(rs),'families':len(fam),'record_canonical_sha256':canonical(rs),'matches_original_run_artifact':True}
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    return parts

def lengths(rs,mode,tok=TOK,max_length=128):
    result=[]
    for row in rs:
        if mode=='sft':
            x,y=render_chat(row['messages'],tok)
            expected=sum(len(tok.encode(m['content']))+1 for m in row['messages'] if m['role']=='assistant')
            assert int((y!=-100).sum())==expected and len(x)<=max_length
            result.append(expected)
        else:
            count=len(tok.encode(row['text']))+1
            covered=[]
            for start in range(0,count,max_length):
                stop=min(start+max_length,count);covered.extend(range(start,stop));result.append(stop-start)
            assert covered==list(range(count))
    return result

def training(key,record,rs,mode,max_length=128,batch=16):
    ls=lengths(rs,mode,max_length=max_length)
    rng=random.Random(42);recomputed=sum(sum(rng.choices(ls,k=batch)) for _ in range(record['steps']))
    assert recomputed==record['effective_tokens'] and record['records']==len(rs) and record['records_sha256']==canonical(rs)
    assert record['history'][-1]['step']==record['steps']
    receipt['training'][key]={'steps':record['steps'],'batch':batch,'records':len(rs),'training_windows_or_sft_examples':len(ls),'complete_training_target_count':sum(ls),'sampled_effective_targets_recomputed':recomputed,'initial_nll':record['initial_loss'],'final_nll':record['final_loss'],'parameters':record['parameters'],'checkpoint':record['checkpoint'],'schedule':'constant 0.003 in inspected original/current fit_lm source; no GPU execution by reviewer'}

def evaluation(key,report,rs,mode,tok=TOK,max_length=128):
    expected=sum(lengths(rs,mode,tok,max_length))
    assert report['effective_tokens']==expected and report['records']==len(rs)
    mean=report.get('nll',report.get('mean_token_nll'))
    total=report.get('nll_sum',mean*expected)
    assert math.isclose(total/expected,mean,rel_tol=0,abs_tol=1e-12)
    out={'records':len(rs),'effective_targets':expected,'nll_sum':total,'mean_nll_recomputed':total/expected,'samples':len(report['samples'])}
    if mode=='sft':
        assert len(report['samples'])==len(rs)
        matches=0;ended=0
        for sample,row in zip(report['samples'],rs,strict=True):
            target=row['messages'][-1]['content'];ids=sample['generated_ids']; content=ids[:-1] if ids and ids[-1]==2 else ids
            assert ids.count(2)<=1 and (2 not in ids or ids[-1]==2)
            assert sample['expected']==target and sample['messages']==row['messages'][:-1]
            exact=content==tok.encode(target)
            assert bool(sample['exact'])==exact and bool(sample['eos'])==(2 in ids)
            matches+=int(exact);ended+=int(2 in ids)
        assert report['matches']==matches and report['exact_match']==matches/len(rs) and report['eos_rate']==ended/len(rs)
        out.update(matches_recomputed=matches,match_denominator=len(rs),eos_recomputed=ended,eos_denominator=len(rs))
    else:
        byte_count=sum(len(r['text'].encode()) for r in rs)
        declared=report.get('raw_utf8_bytes')
        if declared is not None: assert declared==byte_count
        bpb=total/(byte_count*math.log(2))
        if 'bpb_including_eos_boundary_targets' in report: assert math.isclose(bpb,report['bpb_including_eos_boundary_targets'],rel_tol=0,abs_tol=1e-12)
        out.update(raw_utf8_bytes=byte_count,bpb_recomputed=bpb)
        for sample in report['samples']:
            if 'generated_ids' in sample: assert TOK.decode(sample['generated_ids'])==sample['generated']
    receipt['evaluations'][key]=out

for name,d in REPORTS.items():
    path=ROOT/'docs/course-experiments/results'/(name+'.json')
    assert d['seed']==42 and d['device']=='cuda' and d['gpu']=='NVIDIA L4' and d['step_scale']==1
    receipt['original_reports'][name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes()),'revision':d['revision'],'python':d['python_version'],'torch':d['torch_version'],'device':d['device'],'gpu':d['gpu'],'seed':d['seed'],'hf':{k:d['hf'][k] for k in ['repo','revision','prefix']},'timing_scope':d['timing_scope'],'elapsed_seconds':d['elapsed_seconds'],'source_code_sha256':{p:d['code_sha256'][p] for p in ['scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/model.py','tiny_perceptron/data.py']}}
    for asset in d['assets']:
        for f in asset['files']:
            if not f['path'].endswith('.jsonl') or 'manifest' in f['path']: continue
            path=ROOT/'data/training'/f['path']
            assert path.exists() and sha(path.read_bytes())==f['sha256']
            receipt['source_assets'][str(path.relative_to(ROOT))]={'sha256':f['sha256'],'bytes':len(path.read_bytes()),'rows':len(rows(path)),'original_asset_id':asset['id'],'license':asset['license']}

text=load_parts('text_foundation','data');tr=REPORTS['text_foundation']['results']
training('text_foundation',tr['training'],text['train'],'text')
for stage in ['before','after']:
    for split in ['validation','test']: evaluation('text_foundation/'+stage+'/'+split,tr[stage][split],text[split],'text')
assert tr['resume']['total_steps']==80 and tr['resume']['saved_step']==40 and tr['resume']['max_weight_difference']==0
assert tr['after']['validation']['samples'][0]['prompt']=='color=blue;shape=circle;'
assert tr['after']['validation']['samples'][0]['generated']=='side=right.' and text['validation'][0]['text'].endswith('side=left.')
receipt['calculations']['text_foundation']={'train_targets':sum(lengths(text['train'],'text')),'validation_generation':tr['after']['validation']['samples'][0],'validation_truth':text['validation'][0]['text'],'resume_record':tr['resume'],'rounded_table':[f"{tr['training'][k]:.5f}" for k in ['initial_loss','final_loss']]+[f"{tr[stage][split]['nll']:.5f}" for split in ['validation','test'] for stage in ['before','after']]}
for name in ['tinystories','chinese-poetry']:
    parts=load_parts('real_text',name+'-data');r=REPORTS['real_text']['results']['runs'][name]
    training('real_text/'+name,r['training'],parts['train'],'text')
    for stage in ['before','after']:
        for split in ['validation','test']: evaluation('real_text/'+name+'/'+stage+'/'+split,r[stage][split],parts[split],'text')
    raw_name='tinystories-train-512.jsonl' if name=='tinystories' else 'chinese-classical-train-365.jsonl'
    source=rows(ROOT/'data/training/text-initial'/raw_name)
    normalized={' '.join(x['text'].split()) for x in source};assert len(normalized)==r['deduplicated_records']
    assert {row['family'] for side in parts.values() for row in side}=={sha(t.encode()) for t in normalized}
    receipt['calculations']['real_text/'+name]={'source_records':len(source),'normalized_unique_complete_contents':len(normalized),'split_record_counts':[len(parts[s]) for s in ['train','validation','test']],'train_nll_rounded':[f"{r['training'][k]:.5f}" for k in ['initial_loss','final_loss']],'heldout_nll_rounded':{s:[f"{r[stage][s]['nll']:.5f}" for stage in ['before','after']] for s in ['validation','test']},'validation_bpb_rounded':f"{receipt['evaluations']['real_text/'+name+'/after/validation']['bpb_recomputed']:.5f}",'generation_example':r['after']['validation']['samples'][0]}

parts=load_parts('tokenizer','data');r=REPORTS['tokenizer']['results']
bp=BASE/'tokenizer/tokenizer-bpe512.json';by=BASE/'tokenizer/tokenizer-byte.json';art={a['path']:a for a in REPORTS['tokenizer']['artifacts']}
assert sha(bp.read_bytes())==art['tokenizer-bpe512.json']['sha256'] and sha(by.read_bytes())==art['tokenizer-byte.json']['sha256']
bpe=ByteLevelBPE(bp);assert bpe.vocab_size==512
rng=random.Random(42);schedule=[rng.choices(range(len(parts['train'])),k=8) for _ in range(400)]
schedule_sha=sha(json.dumps(schedule,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
assert schedule_sha==r['raw_document_schedule_sha256']
exposure=sum(len(parts['train'][i]['text'].encode()) for batch in schedule for i in batch)
for name,tok in [('byte256',TOK),('bpe512',bpe)]:
    run=r['runs'][name];assert run['training']['steps']==400 and run['training_raw_utf8_bytes_exposed']==exposure
    for stage in ['before','after']:
        for split in ['validation','test']: evaluation('tokenizer/'+name+'/'+stage+'/'+split,run[stage][split],parts[split],'text',tok,max_length=384)
for roundtrip in r['roundtrip']:
    encoded=bpe.encode(roundtrip['text']);assert encoded==roundtrip['ids'] and bpe.decode(encoded)==roundtrip['text'] and not (set(encoded)&set(range(8)))
wrong=json.loads(bp.read_text());wrong['model']['merges']=[]
wrong_path=ROOT/'outputs/natural-v4/factual-research/T.4/wrong-bpe.json';wrong_path.write_text(json.dumps(wrong))
from tiny_perceptron.tokenization import load_tokenizer
try:load_tokenizer(wrong_path,512,{'metadata':{'tokenizer':'bpe512','tokenizer_sha256':sha(bp.read_bytes())}})
except ValueError as error:receipt['calculations']['wrong_tokenizer_expected_error']=str(error)
else:raise AssertionError('mismatched tokenizer should fail')
receipt['calculations']['tokenizer']={'same_raw_exposure':exposure,'schedule_sha256':schedule_sha,'roundtrips_checked':len(r['roundtrip']),'vocab_including_specials':512,'training_steps_per_model':400,'batch':8,'width':32,'layers':1,'context':384}

parts=load_parts('sft','data');r=REPORTS['sft']['results'];training('sft/direct',r['training'],parts['train'],'sft')
for stage in ['before','after']:
    for split in ['validation','test']:evaluation('sft/direct/'+stage+'/'+split,r[stage][split],parts[split],'sft')
branch=r['pretrain_then_sft']; pre_rows=[{'text':row['messages'][0]['content']+row['messages'][1]['content'],'family':row['family']} for row in parts['train']]
training('sft/pretrain',branch['pretraining'],pre_rows,'text');training('sft/pretrain-sft',branch['sft'],parts['train'],'sft')
for stage in ['before_sft','after_sft']:
    for split in ['validation','test']:evaluation('sft/continuation/'+stage+'/'+split,branch[stage][split],parts[split],'sft')
pilot=load_parts('sft','ultrachat-excerpts');q=r['ultrachat_pilot'];training('sft/ultrachat-pilot',q['training'],pilot['train'],'sft',max_length=256,batch=4)
source=rows(ROOT/'data/training/behavior-initial/ultrachat-sft/train-first-100.jsonl')
for side in pilot.values():
    for row in side:
        orig=source[row['source_row']]
        digest=sha(json.dumps(orig,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()); assert row['source_record_sha256']==digest
        for role in ['user','assistant']:
            text=next(m['content'] for m in orig['messages'] if m['role']==role)
            excerpt=next(m['content'] for m in row['messages'] if m['role']==role)
            assert excerpt==text.encode()[:120].decode(errors='ignore') and len(excerpt.encode())<=120
for split in ['validation','test']:evaluation('sft/ultrachat/'+split,q['evaluation'][split],pilot[split],'sft',max_length=256)
receipt['calculations']['sft']={'direct_updates':900,'pretrain_updates':250,'continuation_updates':900,'generation_limit':32,'direct_test_matches':r['after']['test']['matches'],'pretrain_test_matches':branch['before_sft']['test']['matches'],'continuation_test_matches':branch['after_sft']['test']['matches'],'test_denominator':10,'ultrachat_updates':80,'ultrachat_splits':[len(pilot[s]) for s in ['train','validation','test']],'ultrachat_excerpt_bytes_each_side_max':120}

r=REPORTS['sft_ablation']['results'];attributes=load_parts('sft_ablation','attributes');arith=load_parts('sft_ablation','arithmetic')
assert sum(map(len,arith.values()))==64 and [len(arith[s]) for s in ['train','validation','test']]==[49,8,7]
for split,rs in arith.items():
    for row in rs:
        a,b=row['a'],row['b'];assert row['family']==f'{min(a,b)}+{max(a,b)}' and row['messages'][-1]['content']==str(a+b)
clean=attributes['train'];noisy=copy.deepcopy(clean)
for corruption in r['corruptions']:
    row=noisy[corruption['row']];assert row['family']==corruption['family'] and row['messages'][-1]['content']==corruption['correct']
    assert {corruption['correct'],corruption['wrong']}=={'circle','square'}
    row['messages'][-1]['content']=corruption['wrong']
assert len(r['corruptions'])==4 and len({c['family'] for c in r['corruptions']})==2
for name,rs in [('b-only',arith['train']),('replay',attributes['train']+arith['train']),('clean',clean),('noisy',noisy)]:
    q=r['runs'][name];training('sft_ablation/'+name,q['training'],rs,'sft')
    for task,parts in [('A_attributes',attributes),('B_arithmetic',arith)] if name in ['b-only','replay'] else [('attributes',attributes)]:
        for split in ['validation','test']:evaluation('sft_ablation/'+name+'/'+task+'/'+split,q[task][split],parts[split],'sft')
assert [x['messages'][0] for x in clean]==[x['messages'][0] for x in noisy]
assert lengths(clean,'sft')==lengths(noisy,'sft')
assert r['before']['A_attributes']==REPORTS['sft']['results']['after']
receipt['calculations']['sft_ablation']={'steps':{n:q['training']['steps'] for n,q in r['runs'].items()},'arithmetic_records':[49,8,7],'arithmetic_families':[len({x['family'] for x in arith[s]}) for s in ['train','validation','test']],'corruptions':r['corruptions'],'changed_fraction':4/45,'sampled_targets':{n:q['training']['effective_tokens'] for n,q in r['runs'].items()},'shared_baseline_A_matches_direct_sft_result':True}
receipt['scope']='Personally parsed exact relevant original fixed records, SHA-matched raw JSONL and tokenizer bytes, independently recomputed targets, split disjointness, sampled training exposures, NLL/BPB arithmetic, and all cited SFT raw-ID match/EOS counts. Did not run original GPU training, reload large checkpoints, or validate weights/upload status. Text NLL probabilities remain original-run observations, not newly replicated measurements.'
(OUT/'original-record-audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt['calculations'],ensure_ascii=False,indent=2))
