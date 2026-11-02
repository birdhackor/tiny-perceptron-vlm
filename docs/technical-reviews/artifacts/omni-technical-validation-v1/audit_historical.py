"""重算保存的歷史輸出；不載入大模型、不做新的模型測試或訓練。"""
import ast
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CACHE = Path('/workspace/work/reference/omni-technical-validation-v1')
OUT = Path(__file__).with_name('historical-recalculation.json')
torch.set_num_threads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def pinned_file(artifact_id, name):
    return next((CACHE/'github'/str(artifact_id)).rglob(name))

def stats(values):
    values = list(values)
    return dict(correct=sum(values), count=len(values))

def v2():
    result = {}
    for arch, aid, tid, fid, rev in [
        ('moe',11405157675,11401626276,11403753141,'bf43be260d3e819f4ff3df3b1155daac98d01071'),
        ('dense',11409120732,11399969210,11405458957,'9482b50212cca4c6b68cf7ad2d54abe27c2b308b')]:
        raw_source = subprocess.check_output(['git','show',f'{rev}:scripts/selftrained/evaluate.py'],cwd=ROOT).decode()
        source_file = CACHE/'originals'/f'{arch}-historical-evaluate.py'
        source_file.write_text(raw_source)
        names={'normalize','positive_visual_answer','numeric_reply_value','edit_distance','format_pass','score_reply'}
        tree = ast.parse(raw_source)
        selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        namespace={'re':re,'json':json}
        exec(compile(selected,str(source_file),'exec'),namespace)
        journal=pinned_file(aid,'outputs.jsonl')
        rows=[json.loads(x) for x in journal.read_text().splitlines()]
        differing=Counter(); counts=defaultdict(list); groups=defaultdict(set)
        for x in rows:
            recomputed=namespace['score_reply'](x['record'],x['trace']['final_output'],x['trace'])
            for k,v in recomputed.items():
                if x['score'].get(k)!=v: differing[k]+=1
            task=x['record']['task'];counts[task].append(recomputed['semantic']);groups[task].add(x['record']['group_id'])
        eval_receipt=load(pinned_file(aid,'evaluation-receipt.json'))
        final_metrics=load(pinned_file(aid,'metrics.json'))
        execution=load(pinned_file(aid,'execution.json'))
        freeze=load(pinned_file(fid,'frozen.json'))
        training=load(pinned_file(tid,'train-receipt.json'))
        train_log=pinned_file(tid,'metrics.jsonl')
        events=[json.loads(x) for x in train_log.read_text().splitlines()]
        train_events=[x for x in events if x['event']=='train']
        validations={x['step']:x['validation_loss'] for x in events if x['event']=='validation'}
        inf=load(pinned_file(tid,'inference-manifest.json'))
        result[arch]={
          'artifact_ids':dict(test=aid,training=tid,freeze=fid),
          'source':dict(revision=rev,path='scripts/selftrained/evaluate.py',sha256=sha(source_file),functions=sorted(names)),
          'journal':dict(path=str(journal),sha256=sha(journal),bytes=journal.stat().st_size,rows=len(rows),unique_ids=len({x['record']['id'] for x in rows}),hash_matches_receipt=sha(journal)==eval_receipt['outputs_sha256']),
          'recomputed_semantic':{k:{**stats(v),'source_groups':len(groups[k])} for k,v in sorted(counts.items())},
          'scorer_differences':dict(differing),
          'training':dict(steps=training['steps'],train_records=training['train_records'],validation_records=training['validation_records'],test_used_for_selection=training['test_used_for_selection'],origin=training['origin']['kind'],unique_update_steps=len({x['step'] for x in train_events}),max_logged_step=max(x['step'] for x in train_events),validation_losses=validations,argmin_validation_step=min(validations,key=validations.get),selected_step=inf['selected_step'],selected_checkpoint_sha256=inf['selected_checkpoint_sha256'],training_receipt_sha256=sha(pinned_file(tid,'train-receipt.json')),training_log_sha256=sha(train_log)),
          'freeze_binding':dict(created_at=freeze['created_at'],test_started_at=execution['started_at'],created_before_test=freeze['created_at']<execution['started_at'],selection=freeze['selection'],test_once=freeze['test_once'],safe_weights_equal=freeze['safe_weights_sha256']==final_metrics['safe_weights_sha256'],selected_checkpoint_equal=freeze['selected_checkpoint_sha256']==final_metrics['selected_checkpoint_sha256']==inf['selected_checkpoint_sha256'],generation_max_new_tokens=freeze['max_new_tokens'],teacher_forcing_used_for_generation=final_metrics['teacher_forcing_used_for_generation']),
          'fail_all_original_criteria_witness':dict(tool_roundtrip=final_metrics['per_task_final_reply']['tool_call']['tool_roundtrip'],threshold=final_metrics['thresholds']['tool_roundtrip']),
        }
    return result

def bc():
    from tiny_perceptron.lesson_products.data import build_rows,audit_rows
    from tiny_perceptron.lesson_products.model import Config,LoopedSolver,CandidateScorer
    freeze=load(ROOT/'docs/lesson-products/results/settings-freeze-v1.json')
    source_pins={n:dict(expected=h,actual=sha(ROOT/'tiny_perceptron/lesson_products'/n)) for n,h in freeze['source_sha256s'].items()}
    if any(x['expected']!=x['actual'] for x in source_pins.values()):
        raise RuntimeError('Current source differs from historical B/C freeze; do not regenerate with changed code')
    b,c,_=build_rows()
    data_audit=audit_rows(b,c)
    by_id={x['id']:x for x in b}
    raw=Path('/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz')
    manifest=load(CACHE/'originals/hf-raw-manifest.json')
    acc=defaultdict(list); ces=defaultdict(list); raw_kinds=Counter(); semantic_id_disagreement=0
    uncompressed_hash=hashlib.sha256()
    with gzip.open(raw,'rb') as stream:
        for line in stream:
            uncompressed_hash.update(line);x=json.loads(line);product=x['product'];split=x['split'];raw_kinds[f'{product}/{split}']+=1
            if product=='b' and split=='test':
                scores=x['scores_by_depth']['5'];prediction=max(range(len(scores)),key=scores.__getitem__)
                row=by_id[x['id']]
                if row['gold']!=x['gold']:raise ValueError('raw/generator gold mismatch')
                correct=prediction==x['gold'];acc['b5/'+x['task']].append(correct)
                if row.get('strata',{}).get('repeated_swap'):acc['b5/repeated_swap'].append(correct)
            elif product=='b' and split=='entity_shift':
                scores=x['raw_scores'];acc['b5/entity_shift'].append(max(range(len(scores)),key=scores.__getitem__)==x['gold'])
            elif product=='c':
                selected=min(zip(x['candidates'],x['raw_scores']),key=lambda x:(-x[1],x[0]))[0]
                acc[f'c/{split}/{x["question"]}'].append(selected==x['gold'])
                if split=='test' and x['question']=='choice':
                    semantic_id_disagreement+=selected!=x['selected']
                    acc['c/reverse_consistent'].append(selected==x['reverse_selected'])
                if split=='wording_shift':
                    index=x['candidates'].index(x['gold'])
                    for t in [1.0,freeze['calibration']['temperatures'][x['question']]]:
                        values=[s/t for s in x['raw_scores']];m=max(values)
                        ce=m+math.log(sum(math.exp(s-m) for s in values))-values[index]
                        ces[f'{x["question"]}/T={t}'].append(ce)
    package=load(ROOT/'docs/lesson-products/results/publication-package-v1.json')
    publication_matches={}; params={}; receipts={};downloads=[]
    base='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/'
    for product in ['b','c']:
        for name,expected in freeze['products'][product].items():
            key=f'{product}/{name}';p=CACHE/'publication'/key;p.parent.mkdir(parents=True,exist_ok=True)
            raw_download=urllib.request.urlopen(base+key,timeout=45).read();p.write_bytes(raw_download)
            downloads.append(dict(url=base+key,path=str(p),bytes=len(raw_download),sha256=sha(p)))
            publication_matches[key]=sha(p)==expected==package['files'][key]['sha256']
        cfg=load(CACHE/'publication'/product/'config.json');model=(LoopedSolver if product=='b' else CandidateScorer)(Config(**cfg))
        params[product]=sum(p.numel() for p in model.parameters())
    for key,name in [('b','hf-b-receipt.json'),('c','hf-c-receipt.json'),('fixed3','hf-fixed3-receipt.json'),('odd','hf-odd-receipt.json')]:
        d=load(CACHE/'originals'/name)
        receipts[key]={k:d[k] for k in ['parameters','last_step','selected_step','environment','seed','initial_weights_sha256','resume_verification']}
    freeze_path=ROOT/'docs/lesson-products/results/settings-freeze-v1.json';freeze_sha=sha(freeze_path)
    start=load(ROOT/'docs/lesson-products/results/final-report-v1.started.json')
    gitlog=subprocess.check_output(['git','log','--reverse','--format=%H %cI','--','docs/lesson-products/results/settings-freeze-v1.json'],cwd=ROOT,text=True).splitlines()[0]
    return dict(source_pins=source_pins,data_audit=data_audit,raw_input=dict(path=str(raw),sha256=sha(raw),uncompressed_sha256=uncompressed_hash.hexdigest(),published_manifest=manifest,compressed_hash_matches=sha(raw)==package['files']['raw/final-report-v1.predictions.jsonl.gz']['sha256'],uncompressed_hash_matches=uncompressed_hash.hexdigest()==load(ROOT/'docs/lesson-products/results/final-report-v1.json')['predictions']['sha256']),raw_rows=dict(raw_kinds),recomputed_accuracy={k:stats(v) for k,v in acc.items()},choice_recorded_selected_disagreements=semantic_id_disagreement,wording_ce={k:sum(v)/len(v) for k,v in ces.items()},parameter_counts=params,training_receipts=receipts,publication_actual_bytes_match=publication_matches,publication_downloads=downloads,freeze=dict(sha256=freeze_sha,started_sha_matches=start['settings_freeze_sha256']==freeze_sha,reported_sha_matches=load(ROOT/'docs/lesson-products/results/final-report-v1.json')['settings_freeze_sha256']==freeze_sha,git_first_commit=gitlog,test_started_at=start['recorded_at'],protocol=freeze['protocol']))

def other_records():
    cpu=load(ROOT/'docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json')
    inv=cpu['invocations'];chat=[x for x in inv if x.get('chat')]
    local_base=ROOT/'docs/selftrained/infrastructure/v2-local-stage/independent-review'
    local=[]
    for p in (local_base/'final-runs').glob('*/receipt.json'):
        d=load(p);local.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),status=d['status'],returncode=d['returncode'],device=d['job'].get('device'),argv_device=d['command'][d['command'].index('--device')+1]))
    rejects=[]
    for p in sorted(local_base.glob('reject-*.final.raw.log')):
        text=p.read_text();rejects.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),last_line=text.strip().splitlines()[-1]))
    old=load(ROOT/'docs/gpu-smoke-result.json')
    current=load(ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json')
    accepted=load(ROOT/'docs/course-review-20261027-tokenizer/assembly-all-01/manifest.json')
    old_ref=accepted['source_freeze'];old_path=old_ref['path'] if isinstance(old_ref,dict) else old_ref
    old_pages=load(ROOT/old_path)
    added=sorted({x['page_id'] for x in current['pages']}-{x['page_id'] for x in old_pages['pages']})
    return dict(cpu_smoke=dict(invocations=len(inv),chat_count=len(chat),append_count=sum(not x.get('chat',False) for x in inv),returncodes=[x['returncode'] for x in inv],anonymous_sources=[x['public_source'] for x in chat[:1]],neural_generation_rows=sum(bool(x.get('actual_generations')) for x in chat),scope='Read original argv/return codes/generated answer records only; no reliance on review status or pass conclusion.'),local_historical_receipts=local,local_rejections=rejects,new_pages=added,new_page_count=len(added),gpu_smoke=old)

if __name__=='__main__':
    result=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),environment=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),device='CPU',threads='1'),scope='Re-score historical journals and inspect historical instrumentation. No model inference, no optimizer update, no new final test, no paid action.',v2=v2(),bc=bc(),other=other_records())
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(v2={a:dict(count=x['journal']['rows'],semantic=x['recomputed_semantic'],selection=x['training']['argmin_validation_step'],scorer_differences=x['scorer_differences']) for a,x in result['v2'].items()},bc=result['bc']['recomputed_accuracy'],wording_ce=result['bc']['wording_ce'],publication_all_match=all(result['bc']['publication_actual_bytes_match'].values()),new_page_count=result['other']['new_page_count']),ensure_ascii=False))
