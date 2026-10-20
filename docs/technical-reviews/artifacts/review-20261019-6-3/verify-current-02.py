"""核對 ByteLevel 空格分段、新版原文與重做入口，保存可重查的執行收據。"""

import ast
import difflib
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_bytes())


def recipe(text):
    return text.split('<details id="fixed-sft-recipe"', 1)[1].split('</details>', 1)[0]


def main():
    original_path = 'docs/course-review-20261019/reviews/rv1019_technical_6-3/technical-6-3-original-01.json'
    original_raw = (ROOT / original_path).read_bytes()
    assert sha(original_raw) == '5a5bb57d78331583d5517df9a784d75342a05b68fdcfe23fb998400cc94ea88f'
    original = json.loads(original_raw)
    old_path = ROOT / 'docs/course-review-20261019/freeze-01/sources/6.3.md'
    new_path = ROOT / 'docs/course-review-20261019/freeze-05/sources/6.3.md'
    old, new = old_path.read_bytes(), new_path.read_bytes()
    assert sha(new) == '717cb033cff96f9458b18273cf404f04b372994370e2348555985e3d75235ea3'
    spec = importlib.util.spec_from_file_location('section_facts', ROOT / 'docs/review-tools/section_facts.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    current, chapter, _ = helper.original_section(ROOT / 'course/chapters/06.md', '6.3')
    assert current == new
    old_code = old.decode().split('```python\n', 1)[1].split('```', 1)[0]
    new_code = new.decode().split('```python\n', 1)[1].split('```', 1)[0]
    assert old_code == new_code
    result = {
        'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-3/verify-current-02.py',
        'environment': {'python': sys.version, 'tokenizers': tokenizers.__version__,
                        'torch': str(torch.__version__), 'device': 'cpu', 'platform': platform.platform()},
        'source_comparison': {'old_sha256': sha(old), 'new_sha256': sha(new),
                              'current_chapter_section_matches_freeze': current == new,
                              'python_fence_unchanged': old_code == new_code,
                              'python_fence_sha256': sha(new_code.encode()),
                              'diff': ''.join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                                                fromfile=str(old_path.relative_to(ROOT)),
                                                                tofile=str(new_path.relative_to(ROOT))))},
        'original_report': {'path': original_path, 'sha256': sha(original_raw)},
        'original_artifact_integrity': [], 'repository_source_integrity': [],
    }
    for item in original['artifacts']:
        actual = sha((ROOT / item['path']).read_bytes())
        assert actual == item['sha256']
        result['original_artifact_integrity'].append({'path': item['path'], 'sha256': actual, 'unchanged': True})
    for item in original['sources']:
        if item['kind'] == 'repository_code':
            actual = sha((ROOT / item['path']).read_bytes())
            result['repository_source_integrity'].append({
                'path': item['path'], 'original_sha256': item['sha256'], 'current_sha256': actual,
                'unchanged': actual == item['sha256'],
            })
    old_output = read_json('docs/technical-reviews/artifacts/review-20261019-6-3/verification-output.json')
    record_path = old_output['recorded_experiment']['path']
    record_sha = sha((ROOT / record_path).read_bytes())
    assert record_sha == old_output['recorded_experiment']['sha256']
    result['original_experiment_record'] = {'path': record_path, 'sha256': record_sha, 'unchanged': True}
    manifest_path = 'assets/training/manifest.json'
    result['training_asset_manifest'] = {'path': manifest_path, 'sha256': sha((ROOT / manifest_path).read_bytes())}
    pretokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    decoder = decoders.ByteLevel()
    expected = [
        ('a cat', ['a', ' cat']), ('貓 狗', ['貓', ' 狗']), ('a  cat', ['a', ' ', ' cat']),
        ('a   cat', ['a', '  ', ' cat']), ('a\ncat', ['a', '\n', 'cat']),
        ('a\r\ncat', ['a', '\r', '\n', 'cat']), ('a\tcat', ['a', '\t', 'cat']),
        ('a cat  ', ['a', ' cat', '  ']),
    ]
    result['whitespace_segments'] = []
    for text, wanted in expected:
        visible = pretokenizer.pre_tokenize_str(text)
        decoded = [decoder.decode([chunk]) for chunk, _ in visible]
        assert decoded == wanted
        result['whitespace_segments'].append({'text': text, 'visible_pieces': visible,
                                               'decoded_pieces': decoded, 'expected': wanted, 'matches': True})
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pretokenizer
    tok.decoder = decoder
    tok.train_from_iterator(['貓看狗。', 'a cat sees a dog.'], trainers.BpeTrainer(
        vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    state = json.loads(tok.to_str())
    result['current_short_example'] = {'vocab_size': tok.get_vocab_size(),
                                       'merge_count': len(state['model']['merges']), 'samples': []}
    for text, wanted_bytes, wanted_tokens in [('貓看狗。', 12, 2), ('小鳥🙂new', 13, 13), ('未見字🦊', 13, 13)]:
        encoded = tok.encode(text)
        assert len(text.encode()) == wanted_bytes and len(encoded.ids) == wanted_tokens
        assert tok.decode(encoded.ids) == text
        result['current_short_example']['samples'].append({'text': text, 'utf8_bytes': len(text.encode()),
                                                           'token_count': len(encoded.ids), 'same': True})
    original_sources = read_json('docs/technical-reviews/artifacts/review-20261019-6-3/source-excerpts.json')
    source = original_sources['sources'][0]
    response = subprocess.run(['curl', '-fsSL', source['url']], capture_output=True, check=True).stdout
    assert sha(response) == source['full_response_sha256']
    original_regex = next(item for item in source['excerpts'] if item['locator'] == 'regular expression')
    assert original_regex['text'] in response.decode()
    result['official_bytelevel_source'] = {
        'url': source['url'], 'version': 'v0.23.2', 'accessed_on': '2026-10-20',
        'curl_command': ['curl', '-fsSL', source['url']],
        'response_sha256': sha(response), 'unchanged_from_original_read': True,
        'reused_excerpt_path': 'docs/technical-reviews/artifacts/review-20261019-6-3/source-excerpts.json',
        'inspected_identifiers': ['RE', 'PreTokenizer::pre_tokenize', 'bytes_char', 'Decoder::decode_chain'],
    }
    training = (ROOT / 'course/training.md').read_bytes()
    before = subprocess.run(['git', 'show', '5ef56bc756f3:course/training.md'], cwd=ROOT,
                            check=True, capture_output=True).stdout
    after = subprocess.run(['git', 'show', '6a7aa1de5227:course/training.md'], cwd=ROOT,
                           check=True, capture_output=True).stdout
    before_recipe, after_recipe, current_recipe = [recipe(raw.decode()) for raw in (before, after, training)]
    assert before_recipe == after_recipe == current_recipe
    command = '.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda'
    assert command in current_recipe
    diff = subprocess.run(['git', 'diff', '5ef56bc756f3', '6a7aa1de5227', '--', 'course/training.md'],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout
    result['training_reference'] = {
        'path': 'course/training.md', 'current_sha256': sha(training),
        'before_commit_sha256': sha(before), 'after_commit_sha256': sha(after),
        'current_matches_after_commit': training == after,
        'cited_recipe_unchanged': True, 'cited_recipe_sha256': sha(current_recipe.encode()),
        'tokenizer_command': command, 'diff': diff,
    }
    result['cli_help'] = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'],
                                        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    run_tree = ast.parse((ROOT / 'scripts/course_experiments/run.py').read_text())
    device_call = next(node for node in ast.walk(run_tree)
                       if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                       and node.args[0].value == '--device')
    result['cli_device_argument'] = ast.unparse(device_call)
    assert any(kw.arg == 'default' and isinstance(kw.value, ast.Constant) and kw.value.value == 'cpu'
               for kw in device_call.keywords)
    section, _, _ = helper.original_section(ROOT / 'course/chapters/06.md', '6.2')
    result['preceding_example'] = {'current_section_sha256': sha(section),
                                    'original_section_sha256': old_output['preceding_handwritten_example']['section_sha256'],
                                    'python_fence_unchanged': section.decode().split('```python\n', 1)[1].split('```', 1)[0]
                                    == old_output['preceding_handwritten_example']['code']}
    result['byte_tokenizer_usage'] = [
        {'path': str(p.relative_to(ROOT)), 'sha256': sha(p.read_bytes()), 'occurrences': p.read_text().count('ByteTokenizer')}
        for p in sorted((ROOT / 'course/chapters').glob('*.md')) if 'ByteTokenizer' in p.read_text()
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
