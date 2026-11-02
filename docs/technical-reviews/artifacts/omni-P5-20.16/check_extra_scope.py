"""Focused post-original coverage check; no model computation."""

import hashlib
import json
import platform
import zipfile
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def main():
    data = json.loads((HERE / 'minimal-inputs.json').read_bytes())
    checked = 0
    for row in data['kev_rows']:
        for task in [*row['tasks'].values(), row['choice_info']]:
            pred = min(range(len(task['ids'])), key=lambda i: (-task['logits'][i], task['ids'][i]))
            assert task['answer'] == task['ids'][pred]
            checked += 1
    excerpts = {}
    with zipfile.ZipFile('/workspace/work/omni-kev-recovery-completed-41011677439/artifact/input.zip') as z:
        definition = json.loads(z.read('data-definition.json'))
        parents = [tuple(r['parent']) for values in definition['C']['panels'].values() for r in values]
        assert len(parents) == len(set(parents)) == 264
        for name, ranges in [('tiny_perceptron/omni_products/data.py', [(63, 67), (98, 112)]),
                             ('tiny_perceptron/omni_products/av.py', [(75, 82)])]:
            raw = z.read(name)
            assert raw == (REPO / name).read_bytes()
            text = raw.decode().splitlines()
            excerpts[name] = {'original_input_entry': name, 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'current_bytes_equal_original': True,
                'short_excerpt': '\n'.join(f'{i}: {line}' for i, line in enumerate(text, 1)
                    if any(a <= i <= b for a, b in ranges))}
    results = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_p5_technical_20_16',
        'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=. uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/omni-P5-20.16/check_extra_scope.py',
        'environment': {'python': platform.python_version(), 'device': 'CPU; standard library; no neural imports'},
        'result': 'All assertions passed; original grayscale shape source directly inspected, all saved answers equal logit argmax, underlying parent IDs unique.',
        'timing': 'Focused additional coverage after saving technical-original-v1; original bytes remain immutable.',
        'own_A_source_excerpts': excerpts,
        'own_A_interpretation': 'render_shape returns one-channel float masks: circle x²+y²<=64 or square |x|,|y|<=7. collate_av image shape has one channel. This is grayscale shape data, unlike uniform RGB red/blue Qwen input.',
        'raw_argmax_checks': {'task_rows': checked, 'mismatches': 0},
        'C_underlying_parent_checks': {'families_all_four_panels': len(parents), 'unique_parent_triples': len(set(parents))},
        'scope': 'No changes to data/code/selection and no teacher or final NN inference.'}
    output = HERE / 'scope-supplement-v1.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'argmax_task_rows': checked, 'parent_triples': len(parents)}))


if __name__ == '__main__':
    main()
