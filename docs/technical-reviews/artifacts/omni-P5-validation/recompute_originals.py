"""Reviewer-owned read-only recomputation; never constructs or calls a neural model."""

import copy
import hashlib
import json
import math
import platform
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import core_evaluation as measured
from scripts.omni_revision import core_report

FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
OUT = Path(__file__).with_name('original-recomputation-v1.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def receipt(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def differences(a, b, pointer=''):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), pointer
        return [x for k in a for x in differences(a[k], b[k], pointer + '/' + str(k))]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), pointer
        return [x for i, (v, w) in enumerate(zip(a, b)) for x in differences(v, w, pointer + '/' + str(i))]
    if a == b:
        return []
    assert isinstance(a, float) and isinstance(b, float), (pointer, a, b)
    return [{'pointer': pointer, 'original': a, 'recomputed': b, 'absolute_difference': abs(a-b)}]


def main():
    torch.set_num_threads(2)
    results = {part: read(FINAL / f'measurements/{part}/results.json') for part in ('av','sound','choice','integration')}
    selection = read(FINAL / 'frozen/model-selection.json')
    metrics = read(FINAL / 'frozen/metrics.json')
    data = read(FINAL / 'frozen/data-definition.json')
    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer_task': '/root/omni_p5_technical_validation',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'CPU', 'torch_threads': '2'},
              'scope': 'Original preserved rows/logits and package bytes only; zero neural forward, zero weight loading, zero training or selection changes.',
              'inputs': [receipt(FINAL / f'measurements/{p}/results.json') for p in results]}
    archive = FINAL.parent/'final-results.zip'
    manifest = read(FINAL/'package-file-manifest.json')
    mismatches=[]
    with zipfile.ZipFile(archive) as z:
        for rel, item in manifest.items():
            blob=(FINAL/rel).read_bytes()
            expected=item['sha256'] if isinstance(item,dict) else item
            if hashlib.sha256(blob).hexdigest()!=expected or z.read(rel)!=blob:
                mismatches.append(rel)
    output['package_integrity']={'archive':receipt(archive),'manifest':receipt(FINAL/'package-file-manifest.json'), 'checked_inner_files':len(manifest),'mismatches':mismatches}
    assert not mismatches
    output['opening']={k:v for k,v in read(FINAL/'frozen/final-opening.json').items() if k in ['created_at','final_opened','model_selection_sha256','runtime']}
    output['selection']={k:selection[k] for k in ['selected','temperature','threshold']}
    av=results['av']['models'][selection['selected']['av']]
    rows=av['records']
    output['AV']={'families':len(rows),'unique_families':len({r['family'] for r in rows}),
        'paired_correct':sum(r['answers']==r['gold'] for r in rows),
        'panels':{str(p):{'families':sum(r['panel']==p for r in rows),'paired_correct':sum(r['answers']==r['gold'] for r in rows if r['panel']==p)} for p in range(4)},
        'remove_image_correct':sum(r['remove_image_answer']==2 for r in rows),
        'remove_audio_correct':sum(r['remove_audio_answer']==2 for r in rows),
        'waveform_swap_paired_correct':sum(r['waveform_swap_answers']==r['waveform_swap_gold'] for r in rows)}
    sound=results['sound']['codec_and_gold_program']; chain=results['sound']['actual_A_chain']
    snrs=[r['active_reference_snr_db'] for r in sound['rows']]
    output['sound']={'families':len(snrs),'finite_snr_rows':sum(math.isfinite(x) for x in snrs),'mean_preserved_active_snr_db':sum(snrs)/len(snrs),
        'gold_correct':sum(r['generation']['program_success'] for r in sound['rows']),
        'legal':sum(r['generation']['legal'] for r in sound['rows']),
        'chain_families':len(chain['rows']),'chain_correct':sum(r['chain_correct'] for r in chain['rows']),
        'distinct_semantic_sound_programs':chain['distinct_semantic_sound_programs'],
        'sound_programs_seen_in_Talker_train':chain['sound_programs_seen_in_Talker_train'],
        'SNR_limit':'Mean recomputed from preserved per-family SNR values. Codec reconstruction samples are not saved; codec NN encode/decode was not repeated.'}
    replay=copy.deepcopy(results)
    diffs=[]
    for name,t,threshold in [('primary',selection['temperature'],selection['threshold']),('data_control_raw_only',1.0,None)]:
        replay['choice'][name]=measured.evaluate_c(results['choice'][name]['rows'],t,threshold)
        diffs.extend(differences(results['choice'][name],replay['choice'][name],'/choice/'+name))
    c=replay['choice']['primary']
    output['C']={'families':len(c['rows']),'unique_families':len({r['family'] for r in c['rows']}),
        'counts':{kind:c['types'][kind]['primary']['correct'] for kind in ['choice','info','rubric']},
        'accepted':c['policy']['accepted'],'errors':c['policy']['errors'],'risk':c['policy']['empirical_risk'],
        'three_question_types_share_same_families':True}
    auto=results['integration']['auto']['rows']
    output['Auto']={'families':len(auto),'unique_families':len({r['family'] for r in auto}),
        'route_correct':sum(r['actual_route']==r['gold_route'] for r in auto),
        'gold_route_end_to_end_correct':sum(r['gold_route_end_to_end_correct'] for r in auto),
        'real_route_end_to_end_correct':sum(r['real_route_end_to_end_correct'] for r in auto),
        'route_panels':dict(Counter(r['gold_route'] for r in auto)),
        'limit':'B correctness flags and partial C answers are saved; missing intermediate outputs were not reconstructed.'}
    try:
        core_report.validate_complete(results,selection,metrics,data)
        strict='accepted'
    except ValueError as e:
        strict=str(e)
    core_report.validate_complete(replay,selection,metrics,data)
    gates=core_report.verdict(replay,selection,metrics,data)
    output['float_replay']={'strict_original_comparator':strict,'different_floats':len(diffs),'maximum_absolute_difference':max((x['absolute_difference'] for x in diffs),default=0),
        'all_within_1e-11':all(x['absolute_difference']<=1e-11 for x in diffs), 'differences':diffs,
        'complete_validator_after_in_memory_statistics_replacement':'pass; all original integer/row/definition contracts checked',
        'capability_gates_from_recomputed_statistics':gates}
    jobs=[]
    for p in sorted((CORE/'core').glob('*/training/training-result.json')):
        d=read(p); job=read(p.parent.parent/'job-result.json')
        jobs.append({'job':p.parent.parent.name,'training_result':receipt(p),'result_keys':list(d),
                     'facts':{k:v for k,v in d.items() if k in ['updates','completed_updates','parameters','device','gpu','fresh_random_initialization','initialization','seed','status','actual_updates']},
                     'job_status':job.get('status')})
    output['training_jobs']=jobs
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)), 'sha256':sha(OUT),'bytes':OUT.stat().st_size,'AV':output['AV'],'sound':output['sound'],'C':output['C'],'Auto':output['Auto'],'float_differences':len(diffs),'max_difference':output['float_replay']['maximum_absolute_difference'],'training_jobs':len(jobs)},ensure_ascii=False))


if __name__=='__main__':
    main()
