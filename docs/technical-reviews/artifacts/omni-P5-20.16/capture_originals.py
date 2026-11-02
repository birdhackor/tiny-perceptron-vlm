"""Read preserved provider packages; project only rows needed for lesson 20.16.

No neural libraries, weights, network, inference, training, or policy changes.
The separately saved minimal-inputs.json is sufficient for recompute.py.
"""

import array
import hashlib
import io
import json
import math
import platform
import re
import sys
import wave
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, __version__ as pillow_version


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
KEV = Path('/workspace/work/omni-kev-recovery-completed-41011677439')
QWEN = Path('/workspace/work/omni-qwen25-original-attempt-41015365689')
FINAL = Path('/workspace/work/omni-final-completed-41004229262')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def binding(path, url=None):
    value = Path(path).read_bytes()
    return {'path_at_inspection': str(path), 'bytes': len(value), 'sha256': digest(value), 'original_url': url}


def write(name, value):
    path = HERE / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    provenance = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_p5_technical_20_16',
                  'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=. uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/omni-P5-20.16/capture_originals.py',
                  'environment': {'python': platform.python_version(), 'pillow': pillow_version, 'device': 'CPU; original package inspection only'},
                  'raw_inputs': [], 'checks': {}}
    for tag, base, run in [('kev', KEV, 41011677439), ('qwen', QWEN, 41015365689)]:
        for name in ['original-provider-artifact.zip', 'artifact/input.zip'] + (['artifact/result.zip'] if tag == 'kev' else []):
            provenance['raw_inputs'].append(binding(base / name, f'https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/{run}'))
        with zipfile.ZipFile(base / 'original-provider-artifact.zip') as z:
            matches = []
            for n in z.namelist():
                local = base / 'artifact' / n
                if local.is_file():
                    assert local.read_bytes() == z.read(n), n
                    matches.append(n)
            provenance['checks'][tag + '_provider_archive_matches_extracted'] = matches
        receipt = json.loads((base / 'artifact/research-receipt.json').read_bytes())
        assert digest((base / 'artifact/input.zip').read_bytes()) == receipt['input_payload_sha256']
    with zipfile.ZipFile(KEV / 'artifact/result.zip') as z:
        result_entries = {}
        for n in z.namelist():
            b = z.read(n)
            assert b == (KEV / 'verified' / n).read_bytes(), n
            result_entries[n] = {'bytes': len(b), 'sha256': digest(b)}
        provenance['checks']['kev_all_result_entries_equal_verified_bytes'] = result_entries
    raw = (KEV / 'verified/kev-raw-predictions.jsonl').read_bytes()
    rows = []
    for line in raw.splitlines():
        r = json.loads(line)
        p = r['prediction']
        rows.append({'split': r['split'], 'language': r['language'], 'family': p['family'],
                     'tasks': p['tasks'], 'choice_info': p['choice_info'],
                     'world_supported': p['world_supported'],
                     'choice_information_sufficient': p['choice_information_sufficient']})
    data = json.loads((KEV / 'verified/research-result.json').read_bytes())
    with zipfile.ZipFile(KEV / 'artifact/input.zip') as z:
        definition = json.loads(z.read('data-definition.json'))
        all_def = {split: {r['family']: r for r in values} for split, values in definition['C']['panels'].items()}
        for r in rows:
            original = all_def[r['split']][r['family']]
            for k, task in r['tasks'].items():
                q = original['questions'][k]
                assert task['gold'] == q['gold']
                assert task['ids'] == [v['id'] for v in q['candidates']] + (['none'] if k == 'choice' else [])
            assert r['world_supported'] == original['world_supported']
            assert r['choice_information_sufficient'] == original['choice_information_sufficient']
        first = definition['C']['panels']['final'][0]
        panels = {s: [r['family'] for r in v] for s, v in definition['C']['panels'].items()}
        code = {}
        for n in ['scripts/omni_revision/kev_research.py', 'scripts/omni_revision/qwen_research.py',
                  'scripts/omni_revision/qwen_inputs.py', 'tiny_perceptron/omni_products/reliability.py',
                  'tiny_perceptron/omni_products/questions.py']:
            code[n] = {'bytes': len(z.read(n)), 'sha256': digest(z.read(n)),
                       'current_repo_bytes_equal_executed': z.read(n) == (REPO / n).read_bytes()}
        provenance['checks']['executed_code_bindings'] = code
    provenance['checks']['kev_raw_rows'] = {'bytes': len(raw), 'sha256': digest(raw), 'rows': len(rows),
        'counts': {str(k): v for k, v in Counter((r['split'], r['language']) for r in rows).items()},
        'all_saved_gold_and_candidate_orders_match_original_input': True}
    cases_projection = []
    audio_projection = []
    with zipfile.ZipFile(QWEN / 'artifact/input.zip') as z:
        cases = json.loads(z.read('cases/cases.json'))
        for case in cases['audio_text_pairs']:
            b = z.read('cases/' + case['audio'])
            assert digest(b) == case['audio_sha256']
            assert z.read('cases/' + case['audio'].replace('.wav', '.txt')).decode().strip() == case['transcript']
            with wave.open(io.BytesIO(b), 'rb') as w:
                audio_projection.append({'id': case['id'], 'gold': case['gold'], 'transcript': case['transcript'],
                    'wav_sha256': digest(b), 'sample_rate': w.getframerate(), 'channels': w.getnchannels(),
                    'sample_width': w.getsampwidth(), 'samples': w.getnframes(), 'duration_seconds': w.getnframes() / w.getframerate()})
        for case in cases['av_shift_pairs']:
            frames = []
            for frame in case['frames']:
                b = z.read('cases/' + frame['path'])
                assert digest(b) == frame['sha256']
                im = Image.open(io.BytesIO(b)); im.load()
                colors = im.getcolors()
                assert colors and len(colors) == 1 and colors[0][0] == 112 * 112
                frames.append({'sha256': digest(b), 'timestamp_ms': frame['timestamp_ms'], 'mode': im.mode,
                               'size': list(im.size), 'uniform_rgb': list(colors[0][1])})
            sounds = []
            for sound in case['pair']:
                b = z.read('cases/' + sound['audio'])
                assert digest(b) == sound['sha256']
                with wave.open(io.BytesIO(b), 'rb') as w:
                    assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 64000)
                    values = array.array('h', w.readframes(w.getnframes()))
                nonzero = [i for i, v in enumerate(values) if v]
                begin = round((sound['beep_center_seconds'] - 0.04) * 16000)
                expected = [0] * 64000
                for i in range(1280):
                    expected[begin + i] = round(32767 * 0.6 * math.sin(2 * math.pi * 440 * i / 16000))
                assert list(values) == expected
                sounds.append({'sha256': digest(b), 'center_seconds': sound['beep_center_seconds'],
                    'gold': sound['gold'], 'first_nonzero_sample': min(nonzero), 'last_nonzero_sample': max(nonzero),
                    'exact_shifted_440hz_pcm_match': True})
            cases_projection.append({'id': case['id'], 'pattern': case['abstract_pattern'], 'question': case['question'],
                                     'fps': case['fps'], 'duration_seconds': case['duration_seconds'], 'frames': frames, 'pair': sounds})
        provenance['checks']['qwen_cases_json'] = {'bytes': len(z.read('cases/cases.json')), 'sha256': digest(z.read('cases/cases.json')),
            'ffmpeg_version': cases['ffmpeg_version'], 'model_outputs_used': cases['model_outputs_used'],
            'human_listening_test': cases['human_listening_test'], 'source_sha256': cases['source_sha256']}
    qreceipt = json.loads((QWEN / 'artifact/runtime-receipt.json').read_bytes())
    research_receipt = json.loads((QWEN / 'artifact/research-receipt.json').read_bytes())
    apps = json.loads((QWEN / 'artifact/apps-after.json').read_bytes())
    jobs = json.loads((QWEN / 'native-jobs.json').read_bytes())
    provider = json.loads((QWEN / 'native-artifacts.json').read_bytes())
    assert provider['artifacts'][0]['digest'] == 'sha256:' + digest((QWEN / 'original-provider-artifact.zip').read_bytes())
    assert provider['artifacts'][0]['workflow_run']['head_sha'] == research_receipt['source_revision']
    assert apps[0]['app_id'] == qreceipt['app_id'] and apps[0]['state'] == 'stopped' and apps[0]['tasks'] == '0'
    native = (QWEN / 'native-job.log').read_text()
    excerpts = [x for x in native.splitlines() if 'Process completed with exit code 1' in x or
                'SHA256 digest of uploaded artifact zip' in x or 'Artifact ID is 12959736567' in x]
    provenance['checks']['qwen_native_provider'] = {'artifact': provider['artifacts'][0],
        'job_id': jobs['jobs'][0]['id'], 'conclusion': jobs['jobs'][0]['conclusion'],
        'native_log_sha256': digest((QWEN / 'native-job.log').read_bytes()), 'native_log_excerpts': excerpts}
    projection = {'raw_kev_sha256': digest(raw), 'kev_rows': rows, 'C_family_panels': panels,
                  'first_final_input': first, 'cache': data['cache'],
                  'preserved_selection': {'T': data['selection']['temperature']['temperature'], 'tau': data['selection']['policy']['threshold']},
                  'kev_versions': {k: data[k] for k in ['adapter', 'base', 'base_revision', 'dtype', 'option_isolation']},
                  'qwen_audio_inputs': audio_projection, 'qwen_AV_inputs': cases_projection,
                  'qwen_speech_prompts': cases['speech_output_prompts'], 'qwen_runtime_receipt': qreceipt,
                  'qwen_app_after': apps[0], 'qwen_final_stop': {'compute_stopped': research_receipt['compute_stopped'],
                    'resources_verified': research_receipt['resources_verified'], 'phase': research_receipt['phase'],
                    'result_asset_names': [a['name'] for a in research_receipt['assets']]}}
    av_path = FINAL / 'verified/measurements/av/results.json'
    av = json.loads(av_path.read_bytes())
    projection['own_A_remove_controls'] = {k: {'families': len(v['records']),
        'remove_image_answers': [r['remove_image_answer'] for r in v['records']],
        'remove_audio_answers': [r['remove_audio_answer'] for r in v['records']]} for k, v in av['models'].items()}
    provenance['raw_inputs'].append(binding(av_path, 'https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/41004229262'))
    provenance['result'] = 'All assertions passed; raw data projected, not neural inference.'
    write('minimal-inputs.json', projection)
    write('acquisition-v1.json', provenance)
    source = REPO / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/20.16.md'
    block = re.search(r'```python\n(.*?)\n```', source.read_text(), re.S).group(1) + '\n'
    p = HERE / 'inline-example.py'
    if p.exists():
        raise FileExistsError(p)
    p.write_text(block)
    print(json.dumps({'status': 'passed', 'rows': len(rows), 'AV_pairs_inspected': len(cases_projection),
        'audio_inputs_inspected': len(audio_projection), 'files': ['minimal-inputs.json', 'acquisition-v1.json', 'inline-example.py']}))


if __name__ == '__main__':
    main()
