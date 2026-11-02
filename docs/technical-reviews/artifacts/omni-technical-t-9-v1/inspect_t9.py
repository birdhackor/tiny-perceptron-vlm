"""Inspect original-source identity and historical metadata without training."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-t-9-v1')
ARTIFACTS = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    historical_path = ROOT / 'docs/course-experiments/results/quantization.json'
    historical = json.loads(historical_path.read_text())
    result = historical['results']
    originals = json.loads((CACHE / 'local-source-receipts.json').read_text())
    originals += json.loads((CACHE / 'originals/fetch-receipts.json').read_text()) if (CACHE / 'originals/fetch-receipts.json').exists() else json.loads((CACHE / 'fetch-receipts.json').read_text())
    historical_matches = {}
    for relative in ['scripts/course_experiments/compression.py', 'scripts/course_experiments/run.py', 'tiny_perceptron/quantization.py']:
        raw = subprocess.check_output(['git', 'show', historical['revision'] + ':' + relative], cwd=ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == historical['code_sha256'][relative]
        historical_matches[relative] = {'sha256': digest, 'matches_recorded_code': True,
                                        'current_sha256': sha(ROOT / relative)}
    execution = json.loads((ARTIFACTS / 'execution-summary-v1.json').read_text())
    infer = []
    for command in execution['commands']:
        if 'scripts/infer.py' in command['command']:
            response = json.loads(Path(command['raw_path']).read_text())
            infer.append({'command': command['command'], 'generated_token_count': len(response['generated_ids']),
                          'generated_ids': response['generated_ids'], 'generation_status': response['generation_status'],
                          'eos': response['eos']})
    assert all(entry['generated_token_count'] <= 24 for entry in infer)
    snapshot = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.9.md'
    original_quotes = [
        {'file': 'torch-_torch_docs.py', 'locator': 'torch.round lines9897-9910',
         'quote': "The return type of output is same as that of input's dtype."},
        {'file': 'torch-_tensor_docs.py', 'locator': 'Tensor.element_size lines1729-1741',
         'quote': 'Returns the size in bytes of an individual element.'},
        {'file': 'torch-_torch_docs.py', 'locator': 'torch.numel lines8815-8820',
         'quote': 'Returns the total number of elements in the :attr:`input` tensor.'},
        {'file': 'torch-serialization.py', 'locator': 'torch.load lines1376-1379',
         'quote': 'loading only tensors, primitive types, dictionaries'},
        {'file': 'jacob-1712.05877v1.txt', 'locator': 'Section3 text lines271-278; PDF p5',
         'quote': 'We propose an approach that simulates quantization ef-'},
        {'file': 'torch-nn-modules-linear.py', 'locator': 'Linear lines54-57',
         'quote': 'y = xA^T + b'},
        {'file': 'torch-nn-modules-sparse.py', 'locator': 'Embedding lines15-19',
         'quote': 'A simple lookup table that stores embeddings of a fixed dictionary and size.'},
        {'file': 'torch-nn-modules-normalization.py', 'locator': 'LayerNorm lines106-123',
         'quote': 'learnable affine transform parameters'},
    ]
    for item in original_quotes:
        path = CACHE / 'originals' / item['file']
        assert item['quote'] in path.read_text()
        item['original_cache_path'] = str(path)
        item['original_sha256'] = sha(path)
    summary = {
        'created_at': datetime.now(UTC).isoformat(),
        'reviewer_task': '/root/omni_technical_coordinator/technical_t_9',
        'source_snapshot': {'path': str(snapshot.relative_to(ROOT)), 'sha256': sha(snapshot)},
        'original_source_receipts': originals, 'read_excerpt_receipts': original_quotes,
        'pytorch_note': 'Inspected actual installed official 2.14.1+cpu Python source with torch.version.git_version pin. Fetching serialization.rst at that git pin returned HTTP404; that unavailable document is not used as evidence.',
        'sklearn_note': 'Read official version1.7 Cross-validation: evaluating estimator performance introduction, especially the paragraph about hyperparameter tuning on test and the final held-out test. Full HTML/text stay outside Git.',
        'jacob_note': 'Read original arXiv1712.05877v1 (15Dec2017): intro weight-only storage vs measured efficiency; Section2.1 Eq1 r=S(q-Z); Section3 simulated forward quantization and floating master weights; Algorithm1 separate low-bit inference graph; Section4 gemmlowp/Eigen hardware measurements. No claim that its 8-bit CNN results reproduce this 4-bit TinyLM.',
        'historical_record': {
            'path': str(historical_path.relative_to(ROOT)), 'sha256': sha(historical_path),
            **{k: historical[k] for k in ['revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu',
                                         'elapsed_seconds', 'timing_scope', 'step_scale', 'evidence_status', 'modal']},
            'rounded_elapsed_one_decimal': round(historical['elapsed_seconds'], 1),
            'teacher_config': result['teacher_provenance']['config'],
            'teacher_steps': result['teacher_provenance']['steps'],
            'training': {k: result['training'][k] for k in ['steps', 'optimizer_updates', 'batch_size', 'learning_rate',
                                                             'effective_supervised_tokens', 'training_examples']},
            'data': result['data'], 'same_fp32_source': result['same_fp32_source'],
            'reloaded_before_evaluation': result['reloaded_packed_checkpoints_before_evaluation'],
            'variant_summary': {name: {'tensor_bytes': run['storage']['tensor_bytes'],
                                      'validation_examples': run['validation']['examples'],
                                      'validation_supervised_tokens': run['validation']['supervised_tokens'],
                                      'test_examples': run['test']['examples'],
                                      'test_supervised_tokens': run['test']['supervised_tokens'],
                                      'max_new_tokens': run['test']['max_new_tokens']}
                                for name, run in result['runs'].items()},
            'code_identity': historical_matches,
            'scope': 'Historical author record and recorded code identity inspected; L4 timing not independently rerun.'},
        'infer_outputs': infer,
        'probe_membership': execution['probe_membership'],
        'probe_note': 'Current prepare_data seed42 gives test row7; historical fixed-recipe JSON has another row order. Neither is evidence that the learner actually used test outputs to select bits.',
    }
    output = ARTIFACTS / 'inspection-summary-v1.json'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(output), 'sha256': sha(output),
                      'historical_elapsed_one_decimal': summary['historical_record']['rounded_elapsed_one_decimal'],
                      'code_identity_matched': True, 'probe': summary['probe_membership']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
