"""Small receipts for the chapter introduction's provenance and scope claims."""

import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-chapter-20-v1')
OUT = Path(__file__).with_name('evidence-results.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': sha(raw), 'bytes': len(raw)}


def section_pin(path, section):
    raw = path.read_text()
    match = re.search(r'^## ' + re.escape(section) + r' .+$', raw, re.M)
    end = raw.find('\n## ', match.end())
    piece = raw[match.start():end + 1 if end >= 0 else None]
    # These lines, including the mechanism description, were actually inspected.
    first = ''.join(piece.splitlines(keepends=True)[:5])
    start = raw[:match.start()].count('\n') + 1
    return {**pin(path), 'section': section, 'inspected_start_line': start,
            'inspected_end_line': start + len(first.splitlines()) - 1,
            'inspected_slice_sha256': sha(first.encode()), 'short_quote': first.splitlines()[0]}


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite an existing evidence receipt')
    freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
    page = next(p for p in load(freeze_path)['pages'] if p['page_id'] == 'chapter-20')
    chapter = ROOT / page['source']
    text = chapter.read_text()
    intro = text.split('\n## ', 1)[0] + '\n'
    assert sha(intro.encode()) == page['source_sha256'] == sha((ROOT / page['snapshot']).read_bytes())
    assert page['figures_sha256'] == {} and '![' not in intro and '```' not in intro
    links = []
    for label, target in re.findall(r'\[([^]]+)\]\(([^)]+)\)', intro):
        file, _, anchor = target.partition('#')
        path = (chapter.parent / file).resolve()
        assert path.is_file(), target
        if anchor:
            assert re.search(r'^## ' + re.escape(anchor) + r' ', path.read_text(), re.M), target
        links.append({'label': label, 'target': target, **pin(path), 'anchor_found': bool(anchor)})

    # Original MoE receipts, not the technical/reader review conclusions.
    moe = []
    for stage in ['moe-pretrain', 'moe-sft', 'moe-vision', 'moe-ocr', 'moe-audio', 'moe-joint', 'moe-weighted', 'moe-native']:
        path = ROOT / 'docs/selftrained/results/training-raw' / stage / 'raw/train-receipt.json'
        value = load(path)
        assert value['origin']['kind'] == 'all-neural-weights-random'
        moe.append({**pin(path), 'stage': value['stage'], 'steps': value['steps'],
                    'origin_kind': value['origin']['kind'], 'seed': value['origin']['seed']})
    bc = {}
    for product in ['b', 'c']:
        path = CACHE / (product + '-training-receipt.json')
        value = load(path)
        manifest = load(ROOT / 'docs/lesson-products/results' / (product + '-artifact-manifest.json'))
        assert sha(path.read_bytes()) == manifest['files']['training-receipt.json']['sha256']
        assert value['random_initialization'] is True and value['resume_source'] is None and value['weights_changed']
        bc[product] = {**pin(path), **{key: value[key] for key in ['random_initialization', 'resume_source', 'last_step', 'selected_step', 'seed', 'weights_changed', 'initial_weights_sha256', 'selected_weights_sha256']}}

    train_index_path = ROOT / 'docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json'
    index = load(train_index_path)
    original_root = ROOT / index['local_original_artifact_directory']
    expected = {row['path']: row for row in index['files']}
    originals = []
    for name in ['review/training.json', 'review/checkpoints/step-001039/training.json', 'review/checkpoints/step-002077/training.json']:
        path = original_root / name
        assert pin(path)['sha256'] == expected[name]['sha256']
        value = load(path)
        assert value['optimizer_only_lora'] and set(value['optimizer_parameter_names']) == set(value['trainable_parameter_names'])
        assert all('lora_' in name for name in value['optimizer_parameter_names'])
        originals.append({**pin(path), **{key: value[key] for key in ['status', 'completed_steps', 'trained_rows', 'trainable_parameters', 'optimizer_only_lora', 'model_revision', 'lora_rank', 'lora_targets']}, 'optimizer_name_count': len(value['optimizer_parameter_names'])})
    train = load(original_root / 'review/training.json')
    changed = sum(train['initial_adapter_tensors'][name]['sha256_values'] != end['sha256_values'] for name, end in train['final_adapter_tensors'].items())
    assert changed == train['changed_adapter_tensor_count'] == 112
    assert train['frozen_parameter_samples_initial'] == train['frozen_parameter_samples_final']

    scores_path = ROOT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json'
    scores = load(scores_path)
    selection_path = ROOT / 'docs/natural-assistant/v4/selection.json'
    selection = load(selection_path)
    protocol_path = ROOT / 'docs/natural-assistant/v4/validation-protocol-lower-lr.json'
    protocol = load(protocol_path)
    release_path = ROOT / 'docs/natural-assistant/v4/public-release.json'
    release = load(release_path)
    assert pin(scores_path)['sha256'] == selection['validation_scoring_sha256']
    assert pin(protocol_path)['sha256'] == selection['validation_protocol_sha256']
    val_index_path = ROOT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/actual-artifact-index.json'
    val_index = load(val_index_path)
    val_path = ROOT / val_index['local_original_artifact_directory'] / 'review/result.json'
    assert pin(val_path)['sha256'] == selection['validation_result_sha256']
    assert load(val_path)['status'] == 'completed'
    variants = {}
    base = scores['variants']['base']
    for name, value in scores['variants'].items():
        numerator = sum(count * scores['integer_weights'][key] for key, count in value['correct_counts'].items())
        assert numerator == value['primary_numerator']
        record = {'primary_numerator': numerator, 'primary_denominator': value['primary_denominator'],
                  'all_generations_complete': value['all_generations_complete'],
                  'incomplete_count': len(value['incomplete_case_ids']),
                  'strict_improvement': numerator > base['primary_numerator']}
        if name != 'base':
            failed = [key for key, gate in value['gates'].items() if not gate['passed']]
            assert failed == ['voice_typed_reference_chat', 'voice_actual_asr_chat']
            assert value['eligible'] is False and not record['strict_improvement'] and not record['all_generations_complete']
            record.update(eligible=value['eligible'], failed_nonregression_groups=failed)
        variants[name] = record
    assert selection['selected_variant'] == scores['selected_variant'] == release['selected_variant'] == 'base'
    assert release['adapter_parameters'] == 0
    assert pin(ROOT / 'tiny_perceptron/natural_assistant.py')['sha256'] == release['code_files']['tiny_perceptron/natural_assistant.py']

    contexts = [section_pin(ROOT / 'course/chapters' / filename, section) for filename, section in
                [('09.md', '9.17'), ('16.md', '16.15'), ('14.md', '14.19'), ('17.md', '17.17'), ('12.md', '12.17'), ('07.md', '7.11')]]
    external = []
    locators = {
        'qwen3vl-card.md': ('pinned model card lines 53, 81–130; image/text messages and generate/decode', 'This is the weight repository for Qwen3-VL-2B-Instruct.'),
        'whisper-turbo-card.md': ('pinned model card lines 117–128 and 208–219', 'By default, Whisper performs the task of *speech transcription*'),
        'lora-v2.pdf': ('arXiv:2106.09685v2 §4.1, equation (3); pdftotext -layout lines 205–224', 'During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable parameters.'),
        'kev-card.md': ('pinned card Model summary and Model details, lines 61–74', 'It is a LoRA adapter and a pointer head on Qwen3.5-4B-Base'),
        'qwen25-omni-v1.html': ('arXiv:2503.20215v1 Abstract; text conversion lines 99–105', 'an end-to-end multimodal model designed to perceive diverse modalities'),
        'qwen3-omni-v1.html': ('arXiv:2509.17765v1 Abstract; text conversion lines 102–106', 'Qwen3-Omni adopts a Thinker–Talker Mixture-of-Experts (MoE) architecture'),
        'b-training-receipt.json': ('JSON /random_initialization, /resume_source, /weights_changed', 'random_initialization: true; resume_source: null'),
        'c-training-receipt.json': ('JSON /random_initialization, /resume_source, /weights_changed', 'random_initialization: true; resume_source: null'),
    }
    for row in load(CACHE / 'fetch-receipts.json'):
        assert 'error' not in row and sha(Path(row['path']).read_bytes()) == row['sha256']
        locator, quote = locators[row['name']]
        external.append({key: row[key] for key in ['name', 'url', 'accessed_at', 'http_status', 'path', 'bytes', 'sha256']} | {'inspected_locator': locator, 'short_excerpt': quote})

    result = {
        'created_at': datetime.now(UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-20-v1/audit_inputs.py',
        'environment': {'python': platform.python_version(), 'torch': importlib.metadata.version('torch'), 'device': 'CPU; JSON/hash inspection only'},
        'canonical_source': page, 'source_freeze': pin(freeze_path), 'canonical_source_exact_match': True,
        'canonical_figure_check': 'No image references or image objects in introduction; freeze map exactly {}. No figure rendered or claimed viewed.',
        'resolved_introduction_links': links,
        'chapter19_historical_initialization': {'moe_original_receipts': moe, 'public_b_c_original_receipts': bc,
            'scope': 'Historical three products only; the new 19.21–19.24 extensions are not certified trained or capability-tested by this review.'},
        'lora_original_training': {'index': pin(train_index_path), 'originals': originals, 'changed_adapter_tensors': changed,
            'frozen_samples_unchanged': True, 'frozen_scope': train['frozen_check_scope'],
            'scope': 'Read and hashed old original training JSON. No new training, tensor loading or GPU execution.'},
        'validation_selection': {'scores': pin(scores_path), 'protocol': pin(protocol_path), 'selection': pin(selection_path),
            'public_release': pin(release_path), 'original_validation_result': pin(val_path),
            'denominators': scores['denominators'], 'variants': variants, 'selected_variant': selection['selected_variant'],
            'scope': 'Recomputed selection from preserved counts; did not regrade historical open-ended generations or infer population quality.'},
        'inspected_prerequisite_slices': contexts,
        'original_external_sources': external,
        'result': 'All asserted provenance, link, frozen-source, LoRA-update and selection consistency checks passed.',
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'canonical_source_exact_match': True, 'links': len(links), 'moe_original_receipts': len(moe),
                      'historical_lora_archives': [r['completed_steps'] for r in originals[1:]],
                      'selected_variant': selection['selected_variant'], 'external_originals': len(external)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
