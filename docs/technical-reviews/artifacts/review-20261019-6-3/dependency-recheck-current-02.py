"""核對教材引文、既有證據與 tokenizer 小節程式的現行版本。"""

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import tokenizers
import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recipe(raw):
    return raw.decode().split('<details id="fixed-sft-recipe"', 1)[1].split('</details>', 1)[0]


def main():
    previous_path = 'docs/course-review-20261019/reviews/rv1019_technical_6-3/technical-6-3-current-01.json'
    previous_raw = (ROOT / previous_path).read_bytes()
    assert sha(previous_raw) == '4b06c573ee6677c1930acd6fa536c89a64b379de9408fda00b9788e1e37b4440'
    previous = json.loads(previous_raw)
    original_raw = (ROOT / previous['original_report']['path']).read_bytes()
    assert sha(original_raw) == previous['original_report']['sha256']
    receipt = {
        'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-3/dependency-recheck-current-02.py',
        'environment': {'python': sys.version, 'tokenizers': tokenizers.__version__,
                        'torch': str(torch.__version__), 'device': 'cpu', 'platform': platform.platform()},
        'previous_report': {'path': previous_path, 'sha256': sha(previous_raw)},
        'original_report': previous['original_report'],
        'artifact_integrity': [], 'repository_sources': [],
    }
    for item in previous['artifacts']:
        actual = sha((ROOT / item['path']).read_bytes())
        assert actual == item['sha256']
        receipt['artifact_integrity'].append({'path': item['path'], 'sha256': actual, 'unchanged': True})
    for item in previous['sources']:
        if item['kind'] == 'repository_code':
            actual = sha((ROOT / item['path']).read_bytes())
            unchanged = actual == item['sha256']
            if item['path'] != 'course/training.md':
                assert unchanged
            receipt['repository_sources'].append({'path': item['path'], 'sha256': actual,
                                                  'previous_sha256': item['sha256'], 'unchanged': unchanged})
    prior = json.loads((ARTIFACT / 'verification-output-current-02.json').read_bytes())
    for name in ['original_experiment_record', 'training_asset_manifest']:
        item = prior[name]
        actual = sha((ROOT / item['path']).read_bytes())
        assert actual == item['sha256']
        receipt[name] = {'path': item['path'], 'sha256': actual, 'unchanged': True}
    frozen_path = ROOT / 'docs/course-review-20261019/freeze-05/sources/6.3.md'
    frozen = frozen_path.read_bytes()
    assert sha(frozen) == previous['source_sha256']
    spec = importlib.util.spec_from_file_location('section_facts', ROOT / 'docs/review-tools/section_facts.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    current, _, _ = helper.original_section(ROOT / 'course/chapters/06.md', '6.3')
    assert current == frozen
    run = ROOT / 'outputs/reviewer-tools/runs/20261020T152350784273Z'
    extraction = json.loads((run / 'extraction.json').read_bytes())
    assert extraction['source_sha256'] == sha(frozen)
    assert (run / 'section.md').read_bytes() == frozen
    assert extraction['figure_sha256'] == {}
    copied = {'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/06.md#6.3" --execute',
              'result': 'exit_code=0；兩個還原斷言通過。', 'files': {}}
    for name in ['extraction.json', 'execution.json', 'environment.json', 'stdout.txt', 'stderr.txt', 'fence-1.py', 'section.md']:
        raw = (run / name).read_bytes()
        copied['files'][name] = {'sha256': sha(raw), 'content': json.loads(raw) if name.endswith('.json') else raw.decode()}
    assert copied['files']['execution.json']['content']['exit_code'] == 0
    with (ARTIFACT / 'section-execution-current-02.json').open('x', encoding='utf-8') as output:
        output.write(json.dumps(copied, ensure_ascii=False, indent=2) + '\n')
    receipt['source'] = {'path': str(frozen_path.relative_to(ROOT)), 'sha256': sha(frozen),
                         'unchanged_from_previous_review': True, 'current_section_matches_freeze': True,
                         'figure_sha256': {}, 'stdout': copied['files']['stdout.txt']['content']}
    before = subprocess.run(['git', 'show', 'ecf6dec71280:course/training.md'], cwd=ROOT, capture_output=True, check=True).stdout
    after = subprocess.run(['git', 'show', 'c6b9339d2595:course/training.md'], cwd=ROOT, capture_output=True, check=True).stdout
    current_training = (ROOT / 'course/training.md').read_bytes()
    assert sha(current_training) == '1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b'
    assert current_training == after
    expected_previous = next(item['sha256'] for item in previous['sources'] if item['id'] == 's-training')
    assert sha(before) == expected_previous
    assert recipe(before) == recipe(after) == recipe(current_training)
    command = '.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda'
    assert command in recipe(current_training)
    diff = subprocess.run(['git', 'diff', 'ecf6dec71280', 'c6b9339d2595', '--', 'course/training.md'],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout
    receipt['training_reference'] = {'path': 'course/training.md', 'sha256': sha(current_training),
                                     'previous_sha256': sha(before), 'after_commit_sha256': sha(after),
                                     'current_matches_after_commit': True, 'cited_recipe_unchanged': True,
                                     'cited_recipe_sha256': sha(recipe(current_training).encode()),
                                     'tokenizer_command': command, 'diff': diff}
    receipt['cli_help'] = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'],
                                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert '--experiment' in receipt['cli_help'] and '--device' in receipt['cli_help']
    receipt['byte_tokenizer_usage'] = [
        {'path': str(p.relative_to(ROOT)), 'sha256': sha(p.read_bytes()), 'occurrences': p.read_text().count('ByteTokenizer')}
        for p in sorted((ROOT / 'course/chapters').glob('*.md')) if 'ByteTokenizer' in p.read_text()
    ]
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
