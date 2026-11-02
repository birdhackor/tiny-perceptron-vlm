"""Small receipts/excerpts after independently reading originals, with safe helper checks."""

import ast
import hashlib
import html
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINALS = Path('/workspace/work/reference/p5-technical-20.16-originals')
OFFICIAL = ORIGINALS / 'official'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path, url, version, locators, note, excerpts):
    return {'url': url, 'version': version, 'accessed_on': '2026-10-30',
            'bytes': path.stat().st_size, 'sha256': sha(path), 'locators_read': locators,
            'inspection_note': note, 'short_supporting_excerpts': excerpts,
            'full_original_stays_outside_git': True}


def lines(path, first, last):
    return '\n'.join(f'{i}: {line}' for i, line in enumerate(path.read_text().splitlines(), 1) if first <= i <= last)


def main():
    projection = json.loads((HERE / 'minimal-inputs.json').read_text())
    source = (ORIGINALS / 'kev/scripts/omni_revision/kev_research.py').read_text()
    module = ast.parse(source)
    safe = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in {'state_view', 'public_record'}], type_ignores=[])
    namespace = {'re': re}
    exec(compile(safe, 'original-kev-public-record-functions-only', 'exec'), namespace)
    first = projection['first_final_input']
    en, ids_en = namespace['public_record'](first, 'english')
    zh, ids_zh = namespace['public_record'](first, 'chinese')
    assert ids_en == ids_zh and en['questions'] == zh['questions']
    assert en['state'] == first['questions']['choice']['state']
    assert zh['state'] == '裝置A：任務save；許可否；就緒否。 裝置B：任務play；許可是；就緒否。 裝置C：任務record；許可否；就緒否。'
    assert {k: first['questions'][k]['gold'] for k in ['choice', 'info', 'rubric', 'choice_info']} == {
        'choice': 'stop', 'info': 'yes', 'rubric': '0', 'choice_info': 'yes'}
    receipts = {}
    commit = '2074cc44256200bb37540103fa4572901faa10f3'
    base = f'https://raw.githubusercontent.com/jaredpalmer/kev/{commit}/kev/'
    for name, intervals, note in [
        ('kev-metrics.py', [(103, 111), (127, 141), (297, 314)],
         'Read coverage_at_error, risk_coverage_curve, cluster_resamples and paired_bootstrap. Post-hoc maximum coverage reads correctness; group resampling keeps sibling questions/variants together.'),
        ('kev-model.py', [(210, 218), (417, 438)],
         'Read DecisionHead and the full prefix/probs_and_prefix/probs_with_prefix/_branch_rows_from_prefix path. T divides logits once at eval; hybrid cache contains KV, convolution and recurrent state and branches copy state.'),
        ('kev-checkpoint.py', [(233, 242), (293, 305)],
         'Read checkpoint layout/Meta/LoadOptions and torch load path. Tokenizer is loaded from pinned base revision, head from checkpoint; explicit float32 and temperature=1 override defaults.'),
        ('kev-shared_prefix.py', [(6, 18)],
         'Read header, Prefix and functional state gathering. Distinguishes attention KV and DeltaNet convolution/recurrent state; this training helper is not a replacement for the research serving cache test.')]:
        p = OFFICIAL / name
        receipts[name] = receipt(p, base + name.removeprefix('kev-'), commit,
            [f'lines {a}-{b}' for a, b in intervals], note, [lines(p, a, b) for a, b in intervals])
    kev_rev = '6cfce5c2fa4b4bd64026336ab649c5ca78857d52'
    card = (OFFICIAL / 'kev-card.md').read_text()
    receipts['kev-card'] = receipt(OFFICIAL / 'kev-card.md', f'https://huggingface.co/jaredpalmer/kev-4b/raw/{kev_rev}/README.md', kev_rev,
        ['Model details', 'Out-of-scope uses', 'How to use', 'Limitations and trade-offs', 'Provenance and reproducibility'],
        'Read intended task, English scope, raw probability mode, FP32 evaluation, order sensitivity and pretrained/base provenance; card numbers are not substituted for this experiment.',
        [line for line in card.splitlines() if line.startswith(('| Backbone |', '| Languages |', '| Calibration |', '| License |', '- **Option order**'))])
    for name, filename in [('kev-adapter.json', 'adapter_config.json'), ('kev-training.json', 'training_config.json')]:
        p = OFFICIAL / name
        j = json.loads(p.read_text())
        excerpt = ({k: j[k] for k in ['base_model_name_or_path', 'r', 'lora_alpha', 'task_type']} if 'adapter' in name else
                   {'base_revision': j['base_revision'], 'args': {k: j['args'][k] for k in ['base', 'dtype', 'weights_dtype', 'lora', 'option_isolation', 'head_dim']}})
        receipts[name] = receipt(p, f'https://huggingface.co/jaredpalmer/kev-4b/raw/{kev_rev}/{filename}', kev_rev,
            ['full JSON config inspected'], 'Read original config, distinguish training autocast from stored weights and this experiment direct FP32 load.', [excerpt])
    q_rev = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
    p = OFFICIAL / 'qwen25-config.json'; j = json.loads(p.read_text())
    receipts['qwen25-config'] = receipt(p, f'https://huggingface.co/Qwen/Qwen2.5-Omni-3B/raw/{q_rev}/config.json', q_rev,
        ['root model_type/enable_audio_output/enable_talker', 'thinker_config.audio_config/vision_config', 'talker_config', 'token2wav_config'],
        'Read complete released 3B config. Audio input, vision input, Talker and Token2Wav are configured; presence is not proof the aborted worker loaded any component.',
        [{k: j[k] for k in ['model_type', 'enable_audio_output', 'enable_talker']},
         {'vision_temporal_patch_size': j['thinker_config']['vision_config']['temporal_patch_size'],
          'vision_spatial_patch_size': j['thinker_config']['vision_config']['patch_size'],
          'vision_spatial_merge_size': j['thinker_config']['vision_config']['spatial_merge_size']}])
    p = OFFICIAL / 'qwen25-LICENSE'; t = p.read_text()
    receipts['qwen25-license'] = receipt(p, f'https://huggingface.co/Qwen/Qwen2.5-Omni-3B/raw/{q_rev}/LICENSE', q_rev,
        ['§1(i)', '§2(a-b)', '§3', '§4(b)', '§6'],
        'Read full research license. Non-Commercial is research/evaluation, commercial use requires a separate license. Redistribution conditions and model-improvement attribution are separate; no blanket prohibition on outputs is inferred.',
        [line.strip() for line in t.splitlines() if 'research or evaluation purposes only' in line or 'FOR NON-COMMERCIAL PURPOSES ONLY' in line or 'Built with Qwen' in line])
    for name, filename, intervals, note in [
        ('qwen25-processing.py', 'processing_qwen2_5_omni.py', [(183, 188), (254, 263)],
         'Read processor __call__ and multimodal placeholder expansion. videos_kwargs fps and use_audio_in_video affect time coordinates and interleaving; metadata alone is insufficient to certify actual worker processor tensors.'),
        ('qwen25-modeling.py', 'modeling_qwen2_5_omni.py', [(3818, 3825), (3995, 4006)],
         'Read Qwen2_5OmniForConditionalGeneration.generate in full. Requested speech is decoded after Talker.generate and returned as one tensor tuple; it is not a generator yielding playable chunks. No worker output is claimed.')]:
        p = OFFICIAL / name
        receipts[name] = receipt(p, f'https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen2_5_omni/{filename}', 'transformers v4.57.6',
            [f'lines {a}-{b}' for a, b in intervals], note, [lines(p, a, b) for a, b in intervals])
    papers = [
        ('qwen25-omni-report-v1.html', 'https://arxiv.org/html/2503.20215v1', '2503.20215v1',
         ['§2.1–2.4', '§5.2 and Tables 9–10'],
         'Read original architecture, input sampling/temporal coordinates, streaming block context, and speech evaluation. Content consistency (WER), speaker similarity and subjective naturalness are separately assessed. These are paper methods, not this 3B attempt results.',
         ['we resample it to a frequency of 16kHz', 'we dynamically adjust the temporal IDs between frames based on the actual time corresponding to each frame', 'evaluated the subjective naturalness (NMOS) of the generated speech']),
        ('qwen3-omni-report-v1.html', 'https://arxiv.org/html/2509.17765v1', '2509.17765v1',
         ['§2.1, §2.4, §2.5; Tables 1–2'],
         'Read streaming generation and concurrency section, including theoretical latency table and vLLM/compile/CUDA Graph conditions. 234ms is single-concurrency audio first-packet latency for 30B-A3B; 547ms is video. Neither is measured by this 3B work.',
         ['Theoretical First-Packet Latency', 'Experiments are conducted on the vLLM framework', '234/547ms'])]
    for filename, url, version, locators, note, needles in papers:
        p = Path('/workspace/work/reference') / filename
        plain = re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]*>', ' ', p.read_text())))
        excerpts = []
        for needle in needles:
            index = plain.index(needle)
            excerpts.append(plain[max(0, index - 90):index + len(needle) + 200])
        receipts[filename] = receipt(p, url, version, locators, note, excerpts)
    p = OFFICIAL / 'guo17a.pdf'
    receipts['temperature-paper'] = receipt(p, 'https://proceedings.mlr.press/v70/guo17a/guo17a.pdf',
        'PMLR 70, 2017, pp. 1321–1330', ['§4.2, Eq. (9), printed p. 1326'],
        'Read original multiclass calibration section and Eq. 9; positive scalar T fit to validation NLL changes confidence, not argmax.',
        ['T is optimized with respect to NLL on the validation set.',
         'Because the parameter T does not change the maximum of the softmax function, the class prediction remains unchanged.'])
    provider_base = Path('/workspace/work/omni-kev-actual-independent-verification-v1')
    provider = json.loads((provider_base / 'provider-artifacts.json').read_text())['artifacts'][0]
    run = json.loads((provider_base / 'provider-run.json').read_text())
    release = json.loads((provider_base / 'provider-release.json').read_text())
    original_zip = Path('/workspace/work/omni-kev-recovery-completed-41011677439/original-provider-artifact.zip')
    assert provider['digest'] == 'sha256:' + sha(original_zip)
    assert run['id'] == 41011677439 and run['head_sha'] == provider['workflow_run']['head_sha'] == release['target_commitish']
    for asset in release['assets']:
        p = original_zip.parent / 'artifact' / asset['name']
        assert 'sha256:' + sha(p) == asset['digest'] and p.stat().st_size == asset['size']
    output = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_p5_technical_20_16',
        'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=. uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/omni-P5-20.16/inspect_sources.py',
        'environment': {'python': platform.python_version(), 'device': 'CPU; AST helper functions and static source inspection only'},
        'result': 'Original source receipts and short excerpts saved; safe public state functions and provider digest assertions passed.',
        'source_receipts': receipts, 'first_final_public_record_execution': {'english': en, 'chinese': zh, 'candidate_ids': ids_en,
            'questions_equal_between_languages': True, 'gold': {k: v['gold'] for k, v in first['questions'].items()}},
        'kev_provider_digest_checks': {'artifact_url': provider['archive_download_url'], 'artifact_id': provider['id'],
            'artifact_bytes': provider['size_in_bytes'], 'artifact_digest': provider['digest'],
            'run_url': run['html_url'], 'head_sha': run['head_sha'], 'conclusion': run['conclusion'],
            'release_api_url': release['url'], 'release_id': release['id'],
            'assets': [{k: a[k] for k in ['id', 'url', 'name', 'size', 'digest']} for a in release['assets']],
            'metadata_input_sha256s': {n: sha(provider_base / n) for n in ['provider-artifacts.json', 'provider-run.json', 'provider-release.json']},
            'all_local_original_archives_match_native_provider_digest': True},
        'bound_controller_source': {n: {'sha256': sha(ORIGINALS / 'qwen' / n),
            'current_bytes_equal_original': (ORIGINALS / 'qwen' / n).read_bytes() == (REPO / n).read_bytes()}
            for n in ['scripts/omni_revision/staged_execution.py', 'scripts/omni_revision/experiment_execution.py', 'scripts/omni_revision/research_cases.py']},
        'inspection_limits': ['No teacher/final neural inference or live Modal RPCs.',
            'Cache tensors were not preserved here; only original per-family diagnostic outputs and source were checked.',
            'No live website/Colab/MPS/CUDA re-execution, listening or naturalness measurement.',
            'Full papers, external code, config and license originals remain outside Git; report requires only the small saved receipts/projection.']}
    destination = HERE / 'source-inspection-v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'external_originals_read': len(receipts), 'provider_digest_checks': True,
                      'state_only_translation_verified': True}))


if __name__ == '__main__':
    main()
