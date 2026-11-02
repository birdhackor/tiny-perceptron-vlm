"""Read saved historical scores and regenerate finite data; never run model inference."""
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.lesson_products.data import build_rows, audit_rows


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rec(p):
    p=Path(p)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def main():
    torch.set_num_threads(2)
    report=read(ROOT/'docs/lesson-products/results/final-report-v1.json')
    package=read(ROOT/'docs/lesson-products/results/publication-package-v1.json')
    frozen=read(ROOT/'docs/lesson-products/results/settings-freeze-v1.json')
    p=Path('/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz')
    raw=gzip.decompress(p.read_bytes())
    assert hashlib.sha256(raw).hexdigest()==report['predictions']['sha256']
    assert len(raw)==report['predictions']['bytes']
    assert sha(p)==package['files']['raw/final-report-v1.predictions.jsonl.gz']['sha256']
    rows=[json.loads(x) for x in raw.splitlines()]
    b,c,config=build_rows()
    audit=audit_rows(b,c)
    expected=read(ROOT/'docs/lesson-products/data-v2-manifest.json')
    # JSON object keys are strings; the in-memory arithmetic-label keys are integers.
    assert json.loads(json.dumps(audit))==expected['audit'] and config==expected['config']
    b_by_id={r['id']:r for r in b if r['split']=='test'}
    br=[r for r in rows if r['product']=='b' and r['split']=='test']
    assert {r['id'] for r in br}==set(b_by_id)
    b_counts={}
    for label,predicate in [('boxes',lambda r:r['task']=='boxes'),('sum',lambda r:r['task']=='sum'),('repeated_swap',lambda r:r.get('strata',{}).get('repeated_swap',False))]:
        subset=[r for r in br if predicate(b_by_id[r['id']])]
        b_counts[label]={'correct':sum(max(range(len(r['scores_by_depth']['5'])),key=r['scores_by_depth']['5'].__getitem__)==r['gold'] for r in subset),'count':len(subset),'families':len({r['family_id'] for r in subset})}
    er=[r for r in rows if r['product']=='b' and r['split']=='entity_shift' and r['task']=='boxes']
    b_counts['entity_shift']={'correct':sum(max(range(len(r['raw_scores'])),key=r['raw_scores'].__getitem__)==r['gold'] for r in er),'count':len(er)}
    cr=[r for r in rows if r['product']=='c' and r['question']=='choice' and r['split']=='test']
    shifted=[r for r in rows if r['product']=='c' and r['question']=='choice' and r['split']=='wording_shift']
    def c_summary(values):
        losses={}
        for t in [1.,.25]:
            all_terms=[F.cross_entropy(torch.tensor(r['raw_scores']).unsqueeze(0)/t,torch.tensor([r['candidates'].index(r['gold'])]),reduction='sum') for r in values]
            losses[str(t)]=float(torch.stack(all_terms).mean())
        return {'count':len(values),'correct':sum(r['candidates'][max(range(len(r['raw_scores'])),key=r['raw_scores'].__getitem__)]==r['gold'] for r in values),'mean_CE_by_temperature':losses}
    shares=[]
    for name,entries in frozen['products'].items():
        for file,digest in entries.items():
            shares.append({'path':name+'/'+file,'match':digest==package['files'][name+'/'+file]['sha256']})
    assert all(r['match'] for r in shares)
    training=[]
    for rel in ['initial-b-execution.json','initial-c-execution.json','mechanism-fixed3-runner-result.json','mechanism-odd135-runner-result.json']:
        p2=ROOT/'docs/lesson-products/results'/rel;d=read(p2);e=d.get('execution',d);t=e['training']
        training.append({'source':rec(p2),'gpu':e['gpu_name'],**{k:t[k] for k in ['product','mode','steps_this_attempt','last_step','selected_step','parameters','initial_weights_sha256','weights_changed','resume_verification']}})
    v2={}
    for arch,training_dir in [('moe','moe-native'),('dense','dense-weighted')]:
        base=ROOT/'docs/selftrained/results/public-raw'/arch
        m=read(base/'test/metrics.json');f=read(base/'freeze/frozen.json');r=read(base/'test/receipt.json')
        train=read(ROOT/'docs/selftrained/results/training-raw'/training_dir/'raw/train-receipt.json')
        selected=read(ROOT/'docs/selftrained/results/v2-final-public-results.json')['architectures'][arch]['selected_source']
        v2[arch]={'metrics':rec(base/'test/metrics.json'),'freeze':rec(base/'freeze/frozen.json'),'count':m['count'],
            'evaluation_complete':m['evaluation_complete'],'teacher_forcing_used_for_generation':m['teacher_forcing_used_for_generation'],
            'freeze_created_at':f['created_at'],'test_started_at':r['started_at'],'safe_weight_binding_equal':m['safe_weights_sha256']==f['safe_weights_sha256'],
            'test_once':f['test_once'],'selection':f['selection'],'train_steps':train['steps'],'train_records':train['train_records'],'validation_records':train['validation_records'],
            'test_used_for_selection':train['test_used_for_selection'],'origin_kind':train['origin']['kind'],'selected_source':selected,
            'ocr':m['per_task_final_reply']['ocr']['exact'], 'vision_clothing':m['per_task_final_reply']['vision_clothing']['exact'],
            'voice_qa':m['per_task_final_reply']['voice_qa']['semantic'], 'voice_topic_continuation':m['per_task_final_reply']['voice_topic_continuation']['semantic'],
            'tool_call':m['per_task_final_reply']['tool_call'],'thresholds':m['thresholds']}
    commit=subprocess.check_output(['git','log','--reverse','--format=%H %cI','--','docs/lesson-products/results/settings-freeze-v1.json'],cwd=ROOT,text=True).strip()
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU'},
            'no_neural_forward_or_training':True,'raw_predictions':rec(p),'uncompressed_bytes':len(raw),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
            'raw_product_counts':dict(Counter(r['product'] for r in rows)),'B':b_counts,'C':{'test':c_summary(cr),'reverse_consistent':sum(r['selected']==r['reverse_selected'] for r in cr),'wording_shift':c_summary(shifted)},
            'regenerated_data_audit_matches_manifest':True,'data_audit':audit,
            'frozen_public_product_file_bindings':shares,'freeze_commit':commit,'test_started':read(ROOT/'docs/lesson-products/results/final-report-v1.started.json'),
            'training':training,'selftrained_v2':v2,
            'limit':'Historical v2 metrics and frozen receipts read directly; autoregressive v2 neural outputs were not regenerated. Historical independent-review assertions are not used as evidence.'}
    out=Path(__file__).with_name('legacy-recomputation-v1.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out.relative_to(ROOT)),'bytes':out.stat().st_size,'sha256':sha(out),'B':b_counts,'C':result['C'],'v2_counts':{k:v['count'] for k,v in v2.items()},'C_shared':audit['c']['cross_split_shared_observations'],'C_rows_shared':audit['c']['rows_in_shared_observations']},ensure_ascii=False))


if __name__=='__main__':
    main()
