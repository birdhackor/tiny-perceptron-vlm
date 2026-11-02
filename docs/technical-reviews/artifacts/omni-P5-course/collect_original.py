"""Independent whole-page course proof: read-only source checks and saved-row arithmetic.

No training, model loading, neural inference, calibration selection or publication.
"""

import ast
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def reference(path):
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    now = datetime.now(timezone.utc).isoformat()
    source = ROOT / 'course/README.md'
    body = source.read_text()
    links = []
    for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', body):
        label, url = match.groups()
        if '://' in url:
            continue
        filename, _, fragment = url.partition('#')
        target = (source.parent / filename).resolve() if filename else source
        row = {'label': label, 'url': url, 'source_line': body[:match.start()].count('\n') + 1,
               'target': str(target.relative_to(ROOT)), 'exists': target.is_file()}
        if target.is_file():
            text = target.read_text()
            row['target_sha256'] = sha(target)
            if fragment:
                heading = re.search(r'^## ' + re.escape(fragment) + r' .+$', text, re.M)
                row['anchor_exists'] = bool(heading)
                if heading:
                    next_heading = re.search(r'^## ', text[heading.end():], re.M)
                    end = heading.end() + next_heading.start() if next_heading else len(text)
                    section = text[heading.start():end]
                    row['target_heading'] = heading[0]
                    row['section_sha256'] = hashlib.sha256(section.encode()).hexdigest()
                    row['opening_excerpt'] = section[:1100]
            else:
                row['target_heading'] = text.splitlines()[0]
                row['opening_excerpt'] = text[:700]
        links.append(row)
    write('route-inspection-original.json', {'created_at': now, 'source': reference(source),
          'links': links, 'local_links': len(links),
          'missing_files': [r['url'] for r in links if not r['exists']],
          'missing_numbered_anchors': [r['url'] for r in links if r.get('anchor_exists') is False],
          'scope': 'Every guide link resolved against actual source; excerpts used for route semantics, not a review of the target lessons.'})

    notebook_checks = []
    bundle = ROOT / 'outputs/omni-P5-current-kernel-bundle-v7'
    for executed in sorted(bundle.rglob('*.ipynb')):
        original = ROOT / 'notebooks' / executed.relative_to(bundle)
        a, b = read(original), read(executed)
        ac, bc = a['cells'], b['cells']
        equal = len(ac) == len(bc) and all(x['cell_type'] == y['cell_type'] and x['source'] == y['source'] for x, y in zip(ac, bc))
        codes = [c for c in bc if c['cell_type'] == 'code']
        errors = [o for c in codes for o in c.get('outputs', []) if o['output_type'] == 'error']
        notebook_checks.append({'lesson': executed.stem, 'source_sha256': sha(original),
             'executed_sha256': sha(executed), 'same_cell_sources': equal,
             'code_cells': len(codes), 'all_execution_counts_present': all(c.get('execution_count') is not None for c in codes),
             'error_outputs': len(errors), 'outputs': sum(len(c.get('outputs', [])) for c in codes)})
    first = read(bundle / '01/1.1.ipynb')
    first_outputs = [o.get('text', o.get('data', {}).get('text/plain', ''))
                     for c in first['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', [])]
    html = ROOT / 'outputs/omni-P5-preview-v6/1.1.html'
    write('notebook-inspection-original.json', {'created_at': now,
          'original_input_location': str(bundle), 'total_notebooks': len(notebook_checks),
          'all_same_sources': all(r['same_cell_sources'] for r in notebook_checks),
          'all_executed_without_error': all(r['all_execution_counts_present'] and not r['error_outputs'] for r in notebook_checks),
          'total_code_cells': sum(r['code_cells'] for r in notebook_checks),
          'first_lesson_outputs': first_outputs,
          'first_lesson_preview': reference(html),
          'first_lesson_preview_has_CPU_output_label': '實際執行結果 · CPU' in html.read_text(),
          'inspection_receipt': reference(ROOT / 'outputs/omni-P5-current-kernel-bundle-v7-receipt.json'),
          'rows': notebook_checks,
          'scope': 'Independent inspection of saved author executions only. No fresh kernels, no Colab or live-site test; raw outputs remain outside artifacts.'})

    text = '貓看狗，狗看貓。'
    vocab = sorted(set(text))
    encoding = {c:i for i,c in enumerate(vocab)}
    ids = [encoding[c] for c in text]
    recovered = ''.join(vocab[i] for i in ids)
    assert recovered == text and ids[0] == ids[6]
    table = {'看': {'貓': .5, '狗': .5}}
    assert table['貓看'[-1]] == table['狗看'[-1]]
    write('toy-calculation-original.json', {'created_at': now, 'kind': 'artificial transparent CPU arithmetic',
          'text': text, 'vocab': vocab, 'ids': ids, 'recovered': recovered,
          'two_cats_same_id': ids[0] == ids[6],
          'one_character_context': {'contexts': ['貓看','狗看'], 'last_character': '看',
              'distribution_for_either': table['看'], 'reason': 'Lookup depends only on final character; preceding characters are absent from the key.'},
          'limitations': 'ID numbers are arbitrary labels, not semantic distances. This is not training or evidence of learned ability.'})

    recipes = read(CORE / 'frozen/formal-recipes.json')
    jobs = []
    inputs = [CORE / 'frozen/formal-recipes.json', FINAL / 'frozen/metrics.json', FINAL / 'frozen/model-selection.json']
    for job in recipes['job_order']:
        result_path = CORE / 'core' / job / 'training/training-result.json'
        job_path = CORE / 'core' / job / 'job-result.json'
        origin_path = CORE / 'core' / job / 'job-origin.json'
        result, job_result = read(result_path), read(job_path)
        inputs.extend([result_path, job_path, origin_path])
        jobs.append({'job': job, 'complete': result['complete'], 'updates': result['updates'],
            'prescribed_updates': recipes['jobs'][job]['training']['updates'],
            'optimizer_updates_observed': result['optimizer_updates_observed'],
            'initial_parameter_sha256': result['initial_parameter_sha256'],
            'latest_parameter_sha256': result['latest_parameter_sha256'],
            'parameters_changed': result['initial_parameter_sha256'] != result['latest_parameter_sha256'],
            'source_unchanged': result['source_unchanged_during_training'],
            'runtime': result['runtime'], 'job_result': job_result})
    metrics = read(FINAL / 'frozen/metrics.json')
    selection = read(FINAL / 'frozen/model-selection.json')
    results = {}
    for kind in ['av','choice','sound','integration']:
        p = FINAL / 'measurements' / kind / 'results.json'
        inputs.append(p)
        results[kind] = read(p)
    av = {}
    for job, data in results['av']['models'].items():
        rows = data['records']
        av[job] = {'families': len(rows), 'paired_correct': sum(r['answers'] == r['gold'] for r in rows),
            'panels': {str(i): {'correct': sum(r['answers'] == r['gold'] for r in rows if r['panel']==i),
                              'n': sum(r['panel']==i for r in rows)} for i in range(4)},
            'remove_image_correct': sum(r['remove_image_answer']==2 for r in rows),
            'remove_audio_correct': sum(r['remove_audio_answer']==2 for r in rows),
            'waveform_swap_paired_correct': sum(r['waveform_swap_answers']==r['waveform_swap_gold'] for r in rows)}
    selected = av[selection['selected']['av']]
    av_gate = selected['paired_correct'] >= metrics['AV']['paired_minimum_correct'] and all(
        row['correct']/row['n'] >= metrics['AV']['panel_minimum_accuracy'] for row in selected['panels'].values()) and all(
        selected[key]/selected['families'] >= metrics['AV'][threshold]
        for key, threshold in [('remove_image_correct','remove_image_minimum_accuracy'),('remove_audio_correct','remove_audio_minimum_accuracy')])
    sound = results['sound']['codec_and_gold_program']
    rows = sound['rows']
    snrs = [r['active_reference_snr_db'] for r in rows]
    sound_recomputed = {'families': len(rows), 'mean_active_reference_snr_db': sum(snrs)/len(rows),
        'all_snr_finite': all(math.isfinite(v) for v in snrs),
        'gold_program_correct': sum(r.get('generation',{}).get('program_success',False) for r in rows),
        'legal': sum(r.get('generation',{}).get('legal',False) for r in rows),
        'chain_families': len(results['sound']['actual_A_chain']['rows']),
        'chain_correct': sum(r['chain_correct'] for r in results['sound']['actual_A_chain']['rows']),
        'chunk_controls': sound['chunk_controls']}
    sound_gate = sound_recomputed['all_snr_finite'] and sound_recomputed['mean_active_reference_snr_db'] >= metrics['sound']['active_reference_snr_minimum_db'] and sound_recomputed['gold_program_correct']/len(rows)>=metrics['sound']['gold_program_minimum_accuracy'] and sound_recomputed['legal']/len(rows)>=metrics['sound']['legal_output_minimum_accuracy'] and sound_recomputed['chain_correct']/sound_recomputed['chain_families']>=metrics['sound']['actual_AV_chain_minimum_accuracy'] and all(r['chunks_bit_exact'] and r['future_code_prefix_bit_exact'] for r in sound['chunk_controls'])
    choice = results['choice']['primary']
    c_rows = choice['rows']
    type_counts = {kind: sum(r['tasks'][kind]['answer']==r['tasks'][kind]['gold'] for r in c_rows) for kind in ['choice','info','rubric']}
    policy_rows = []
    for r in c_rows:
        task, info = r['tasks']['choice'], r['choice_info']
        logits = task['logits']; peak = max(v / selection['temperature'] for v in logits)
        exps = [math.exp(v / selection['temperature']-peak) for v in logits]
        p = [v/sum(exps) for v in exps]
        index = min(range(len(p)), key=lambda i:(-p[i], task['ids'][i]))
        info_index = min(range(len(info['logits'])), key=lambda i:(-info['logits'][i], info['ids'][i]))
        answer = task['ids'][index]
        accepted = selection['threshold'] is not None and max(p)>=selection['threshold'] and answer!='none' and info['ids'][info_index]=='yes'
        error = answer!=task['gold'] or not r['world_supported'] or not r['choice_information_sufficient']
        policy_rows.append({'family':r['family'],'accepted':accepted,'error_if_accepted':bool(accepted and error)})
    accepted = sum(r['accepted'] for r in policy_rows)
    errors = sum(r['error_if_accepted'] for r in policy_rows)
    invariant_rows = results['choice']['development_invariants']['rows']
    numeric = [v for row in invariant_rows for key,v in row.items() if key.endswith('max_abs')]
    choice_recomputed = {'families':len(c_rows), 'type_correct':type_counts, 'frozen_temperature':selection['temperature'],
        'frozen_threshold':selection['threshold'], 'accepted':accepted,'errors':errors,
        'empirical_accepted_risk':errors/accepted if accepted else None,
        'policy_rows':policy_rows, 'development_invariant_families':len(invariant_rows),
        'maximum_development_equivalence_error':max(numeric),
        'development_permutation_answers_same':all(r['permutation_answer_same'] for r in invariant_rows),
        'development_invariants_pass':all(v<=metrics['C']['equivalence_atol'] for v in numeric) and all(r['permutation_answer_same'] for r in invariant_rows)}
    choice_gate = all(type_counts[k]>=metrics['C'][k+'_minimum_correct'] for k in type_counts) and accepted>=metrics['C']['minimum_accepted'] and errors/accepted<=metrics['C']['maximum_empirical_accepted_risk']
    b_rows = results['integration']['B_retention']['rows']
    b_recomputed = {'families':len(b_rows), 'old_correct':sum(r['old']['answer']==r['gold'] for r in b_rows),
        'wrapper_correct':sum(r['wrapper']['answer']==r['gold'] for r in b_rows),
        'same_answer_all':all(r['old']['answer']==r['wrapper']['answer'] for r in b_rows),
        'depths':sorted(set(r['old']['loops'] for r in b_rows))}
    old_a_rows = results['integration']['old_A_retention']['rows']
    old_a_recomputed = {task:{'n':sum(r['task']==task for r in old_a_rows),
        'old_correct':sum(r['old_correct'] for r in old_a_rows if r['task']==task),
        'wrapper_correct':sum(r['wrapper_correct'] for r in old_a_rows if r['task']==task)}
        for task in sorted(set(r['task'] for r in old_a_rows))}
    write('measurement-row-projections-original.json', {
        'created_at':now, 'scope':'Small sufficient original-row projections for portable arithmetic; no model files or large media.',
        'selected_AV_job':selection['selected']['av'],
        'selected_AV_rows':results['av']['models'][selection['selected']['av']]['records'],
        'sound_rows':[{'family_index':r['family_index'],'snr':r['active_reference_snr_db'],
                       'program_success':r.get('generation',{}).get('program_success',False),
                       'legal':r.get('generation',{}).get('legal',False)} for r in sound['rows']],
        'sound_chain_rows':[{'family':r['family'],'chain_correct':r['chain_correct']} for r in results['sound']['actual_A_chain']['rows']],
        'choice_rows':[{**{k:r[k] for k in ['family','world_supported','choice_information_sufficient']},
                        'tasks':{k:{a:v[a] for a in ['answer','ids','logits','gold']} for k,v in r['tasks'].items()},
                        'choice_info':{k:r['choice_info'][k] for k in ['ids','logits']}} for r in c_rows],
        'choice_development_invariants':invariant_rows,
        'old_A_rows':[{k:r[k] for k in ['task','family','old_correct','wrapper_correct']} for r in old_a_rows],
        'B_rows':[{'parent':r['parent'],'gold':r['gold'],'old_answer':r['old']['answer'],
                   'wrapper_answer':r['wrapper']['answer'],'loops':r['old']['loops']} for r in b_rows],
        'frozen_T':selection['temperature'],'frozen_tau':selection['threshold']})
    write('measurement-recomputation-original.json', {'created_at':now, 'action':'saved-row arithmetic only; no model loads or neural calls',
          'raw_input_receipts':[reference(p) for p in inputs], 'core_training_jobs':jobs,
          'ten_complete_jobs':len(jobs)==10 and all(r['complete'] and r['parameters_changed'] and r['optimizer_updates_observed'] and r['updates']==r['prescribed_updates'] for r in jobs),
          'selected_AV':selection['selected']['av'],'AV':av,'AV_primary_gate':av_gate,
          'sound':sound_recomputed,'sound_primary_gate':sound_gate,
          'C':choice_recomputed,'C_complete_primary_gate':choice_gate,'B_retention':b_recomputed,
          'old_A_retention':{'observations':len(old_a_rows),'parent_families':len(set(r['family'] for r in old_a_rows)),
              'tasks':old_a_recomputed,'all_task_counts_unchanged':all(r['old_correct']==r['wrapper_correct'] for r in old_a_recomputed.values()),
              'scope':'258 task observations share180 parents; not258 independent new inputs.'},
          'thresholds':{'AV':metrics['AV'],'sound':metrics['sound'],'C':{k:v for k,v in metrics['C'].items() if any(t in k for t in ['minimum','maximum','atol'])}},
          'scope':'Recomputes original measured labels and logged waveform analyses, not a repeat of final inference. Invariants are 8 development families, not proof of every future input. SNR is an average of original per-family SNR values, not fresh waveform inference.'})
    write('environment-original.json', {'created_at':now,'python':sys.version,'platform':platform.platform(),
        'device':'CPU arithmetic and file inspection only','neural_model_loaded':False,'trained':False,
        'environment_command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-course/collect_original.py'})
    print(json.dumps({'links':len(links), 'bad_links':sum(not r['exists'] or r.get('anchor_exists') is False for r in links),
        'notebooks':len(notebook_checks),'source_match':all(r['same_cell_sources'] for r in notebook_checks),
        'ten_jobs':len(jobs),'AV_primary_gate':av_gate,'sound_primary_gate':sound_gate,
        'C_type_counts':type_counts,'accepted_errors':[accepted,errors],'C_gate':choice_gate,
        'selected_AV':selected,'sound':{k:v for k,v in sound_recomputed.items() if k!='chunk_controls'},'B':b_recomputed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
