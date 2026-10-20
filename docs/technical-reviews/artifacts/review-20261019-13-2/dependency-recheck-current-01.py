"""核對相依教材改動、舊證據完整性及目前的對話對齊；不執行訓練。"""

import contextlib
import hashlib
import inspect
import io
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, capture_output=True, check=True, **kwargs)


def section(raw, title):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    chosen = next(i for i, heading in enumerate(headers) if heading[0].decode().endswith(title))
    end = headers[chosen + 1].start() if chosen + 1 < len(headers) else len(raw)
    return raw[headers[chosen].start():end]


original_path = Path('docs/course-review-20261019/reviews/rv1019_technical_13-2/technical-13-2-original-01.json')
original_raw = (ROOT / original_path).read_bytes()
assert sha(original_raw) == '6f24e5f06bf4ccd9d1680a17348df06522efe1654702fec92144852c4f3c52e2'
original = json.loads(original_raw)
assert original['issues'] == []
old_artifacts = []
for item in original['artifacts']:
    actual = sha((ROOT / item['path']).read_bytes())
    assert actual == item['sha256']
    old_artifacts.append({'path': item['path'], 'sha256': actual, 'unchanged': True})
old_self_check = ROOT / 'docs/technical-reviews/artifacts/review-20261019-13-2/self-check.txt'
assert sha(old_self_check.read_bytes()) == 'b56b9cb1b735e4e4138ab6467796fb19b16ec0d8cb6fc3e09e8a43a8a86ba564'
repo_sources = []
for source in original['sources']:
    if source['kind'] != 'repository_code':
        continue
    actual = sha((ROOT / source['path']).read_bytes())
    changed = actual != source['sha256']
    assert changed == (source['path'] == 'course/chapters/07.md')
    repo_sources.append({'source_id': source['id'], 'path': source['path'], 'old_sha256': source['sha256'], 'current_sha256': actual, 'changed': changed})

before = command(['git', 'show', '5ef56bc756f3:course/chapters/07.md']).stdout
at_diff_end = command(['git', 'show', '3af7f13ef959:course/chapters/07.md']).stdout
current = (ROOT / 'course/chapters/07.md').read_bytes()
assert sha(before) == next(item['sha256'] for item in original['sources'] if item['id'] == 'r_chat')
assert current == at_diff_end
assert sha(current) == '4b02242d4c281a8a0554a3a87bdc898c040bb7afa3250dc8c055184407952174'
before_example = section(before, '對話怎麼表示？')
current_example = section(current, '對話怎麼表示？')
assert before_example == current_example
chapter_diff_command = ['git', 'diff', '5ef56bc756f3', '3af7f13ef959', '--', 'course/chapters/07.md']
chapter_diff = command(chapter_diff_command).stdout.decode('utf-8')
example_code = re.search(r'```python\n(.*?)\n```', current_example.decode('utf-8'), re.S)[1]
namespace = {'torch': torch}
example_stdout = io.StringIO()
with contextlib.redirect_stdout(example_stdout):
    exec(compile(example_code, 'course/chapters/07.md:render_chat example', 'exec'), namespace)
expected_x = [1, 3, 57, 51, 57, 69, 71, 2, 4, 58]
expected_y = [-100] * 8 + [58, 2]
assert namespace['x'].tolist() == expected_x
assert namespace['y'].tolist() == expected_y

chapter_raw = (ROOT / 'course/chapters/13.md').read_bytes()
page_raw = section(chapter_raw, '同一提問的兩份回答，如何對齊成資料？')
frozen_raw = (ROOT / 'docs/course-review-20261019/freeze-01/sources/13.2.md').read_bytes()
assert page_raw == frozen_raw
assert sha(page_raw) == original['source_sha256']
freeze_path = 'docs/course-review-20261019/freeze-05.json'
assert sha((ROOT / freeze_path).read_bytes()) == 'e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba'
section_command = ['.venv/bin/python', 'docs/review-tools/section_facts.py', 'course/chapters/13.md#13.2', '--execute']
section_result = command(section_command, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
summary = json.loads(section_result.stdout)
assert summary['source_sha256'] == original['source_sha256']
assert summary['execution_exit_code'] == 0
assert summary['python_fences'] == 1 and summary['svg_references'] == 0
run_directory = Path(summary['artifact_directory'])
assert run_directory.is_relative_to(ROOT / 'outputs/reviewer-tools/runs')
section_files = []
for name in ['section.md', 'extraction.json', 'execution.json', 'environment.json', 'stdout.txt', 'stderr.txt', 'fence-1.py', 'bootstrap.py']:
    raw = (run_directory / name).read_bytes()
    section_files.append({'name': name, 'sha256': sha(raw), 'content': raw.decode('utf-8')})
stdout = (run_directory / 'stdout.txt').read_bytes()
assert stdout == (ROOT / 'docs/technical-reviews/artifacts/review-20261019-13-2/section-stdout.txt').read_bytes()
assert (run_directory / 'stderr.txt').read_bytes() == b''
assert (run_directory / 'fence-1.py').read_bytes() == (ROOT / 'docs/technical-reviews/artifacts/review-20261019-13-2/section-fence-1.py').read_bytes()
assert (run_directory / 'bootstrap.py').read_bytes() == (ROOT / 'docs/technical-reviews/artifacts/review-20261019-13-2/section-bootstrap.py').read_bytes()
environment = json.loads((run_directory / 'environment.json').read_bytes())
assert environment['guard_events'] == []

# 同位置的 logits 與 labels 直接計分，不再移位。
torch.manual_seed(42)
torch.set_num_threads(1)
with torch.no_grad():
    x, y = render_chat([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}])
    model = TinyLM(ModelConfig(width=8)).eval()
    logits = model(x[None])['logits']
    assert logits.shape == (1, len(x), 264)
    effective = y != -100
    direct = torch.nn.functional.cross_entropy(logits[0, effective], y[effective])
    actual = masked_loss(logits, y[None])
    assert torch.equal(direct, actual)
    first = int(effective.nonzero()[0])
    assert int(x[first]) == 4 and int(y[first]) == 73
    alignment_check = {'input_shape': list(x[None].shape), 'logits_shape': list(logits.shape), 'first_answer_position': first, 'input_at_first_answer': int(x[first]), 'target_at_first_answer': int(y[first]), 'masked_loss': float(actual), 'direct_same_position_loss': float(direct)}

    # 注意力遮罩不會略過 PAD 的位置向量查表。
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(vocab_size=10, width=8)).eval()
    base = model(torch.tensor([[1, 2, 3]]))['logits']
    ids = torch.tensor([[0, 0, 1, 2, 3]])
    valid = ids != 0
    comparisons = []
    for pad_positions in [(0, 0), (17, 97), (127, 3)]:
        positions = torch.tensor([[*pad_positions, 0, 1, 2]])
        logits = model(ids, valid=valid, positions=positions)['logits'][:, 2:]
        delta = float((base - logits).abs().max())
        assert torch.allclose(base, logits, atol=1e-6, rtol=0)
        comparisons.append({'pad_positions': list(pad_positions), 'max_abs_difference': delta})
    try:
        model(ids, valid=valid, positions=torch.tensor([[-1, 0, 0, 1, 2]]))
    except IndexError as error:
        rejected_position = {'exception_type': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError('位置表不應接受範圍外的負索引')

ppo_url = 'https://spinningup.openai.com/en/latest/algorithms/ppo.html'
ppo_command = ['curl', '-fL', '--max-time', '45', '-sS', ppo_url]
ppo_result = command(ppo_command)
ppo_text = re.sub(r'\s+', ' ', unescape(re.sub('<[^>]+>', ' ', ppo_result.stdout.decode('utf-8'))))
excerpt = 'it is still possible to end up with a new policy which is too far from the old policy'
assert excerpt in ppo_text
assert 'remove incentives for the new policy to get far from the old policy' in ppo_text

result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-2/dependency-recheck-current-01.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
    'head_at_check': command(['git', 'rev-parse', 'HEAD']).stdout.decode().strip(),
    'original_report': {'path': str(original_path), 'sha256': sha(original_raw)},
    'original_issues': original['issues'],
    'previous_artifacts': old_artifacts,
    'previous_self_check_unchanged': True,
    'repository_sources': repo_sources,
    'chapter_diff': {'command': chapter_diff_command, 'stdout': chapter_diff, 'exit_code': 0},
    'linked_example': {'title': '對話怎麼表示？', 'old_sha256': sha(before_example), 'current_sha256': sha(current_example), 'unchanged': True, 'executed_code': example_code, 'stdout': example_stdout.getvalue(), 'x': namespace['x'].tolist(), 'y': namespace['y'].tolist()},
    'section_facts': {'command': section_command, 'exit_code': section_result.returncode, 'stdout': section_result.stdout.decode(), 'stderr': section_result.stderr.decode(), 'summary': summary, 'copied_files': section_files, 'stdout_matches_previous': True, 'fence_matches_previous': True, 'bootstrap_matches_previous': True},
    'dependency_checks': {
        'same_position_alignment': alignment_check,
        'valid_pad_positions': comparisons,
        'negative_pad_position': rejected_position,
        'repository_code_sha256': {path: sha((ROOT / path).read_bytes()) for path in ['tiny_perceptron/model.py', 'tiny_perceptron/attention.py', 'tiny_perceptron/data.py']},
        'inspected_source': {'TinyLM.forward': inspect.getsource(TinyLM.forward), 'loss_sum': inspect.getsource(loss_sum), 'masked_loss': inspect.getsource(masked_loss), 'attention_mask': inspect.getsource(attention_mask)},
        'ppo_official_source': {'url': ppo_url, 'version': 'Spinning Up latest documentation as retrieved; exact HTML identified by retrieved_sha256', 'accessed_on': datetime.now().astimezone().date().isoformat(), 'command': ppo_command, 'exit_code': ppo_result.returncode, 'retrieved_sha256': sha(ppo_result.stdout), 'retrieved_bytes': len(ppo_result.stdout), 'locator': 'Background; Key Equations; You Should Know', 'excerpt': excerpt, 'inspection_note': 'PPO-Clip 用裁切目標減少大幅變動的誘因，但仍可能產生離舊策略過遠的新策略；「設法避免」描述目的，沒有承諾每次機率改變都受硬限制。', 'whole_third_party_source_saved': False}
    },
    'result': '全部斷言通過；本節原文、引用算例及舊證據相符；未執行訓練。'
}
print(json.dumps(result, ensure_ascii=False, indent=2))
