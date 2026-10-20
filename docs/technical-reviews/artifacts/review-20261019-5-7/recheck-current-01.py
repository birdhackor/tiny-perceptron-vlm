import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.training import choose_device, load_checkpoint


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return {'argv': argv, 'exit_code': result.returncode, 'stdout': result.stdout.decode(), 'stderr': result.stderr.decode()}


def main():
    original_path = ROOT / 'docs/course-review-20261019/reviews/rv1019_technical_5-7/technical-5-7-original-01.json'
    original_raw = original_path.read_bytes()
    assert sha(original_raw) == '4c55f6d65e10d2ce1f76f3071c7ec0930ff394288166e97fb167d7679a2eef8b'
    report = json.loads(original_raw)
    checks = []
    for entry in report['sources'] + report['artifacts']:
        if 'path' not in entry:
            continue
        current = sha((ROOT / entry['path']).read_bytes())
        changed = current != entry['sha256']
        assert not changed or entry['id'] == 's_recipe'
        checks.append({'id': entry['id'], 'path': entry['path'], 'previous_sha256': entry['sha256'], 'current_sha256': current, 'unchanged': not changed})
    assert next(x for x in checks if x['id'] == 's_recipe')['current_sha256'] == 'd59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94'
    frozen = (ROOT / 'docs/course-review-20261019/freeze-01/sources/5.7.md').read_bytes()
    assert sha(frozen) == report['source_sha256']
    chapter = (ROOT / 'course/chapters/05.md').read_bytes()
    current_section = re.search(rb'^## 5\.7 [^\n]*\n.*?(?=^## |\Z)', chapter, re.M | re.S).group()
    assert current_section == frozen
    freeze_raw = (ROOT / 'docs/course-review-20261019/freeze-06.json').read_bytes()
    assert sha(freeze_raw) == '9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9'
    page = next(x for x in json.loads(freeze_raw)['pages'] if x['page_id'] == report['lesson_id'])
    assert page['source_sha256'] == sha(frozen) and page['figures_sha256'] == {}
    diff = command(['git', 'diff', '5ef56bc756f3', '6a7aa1de5227', '--', 'course/training.md'])
    before = command(['git', 'show', '5ef56bc756f3:course/training.md'])['stdout'].encode()
    after = command(['git', 'show', '6a7aa1de5227:course/training.md'])['stdout'].encode()
    assert sha(before) == next(x for x in checks if x['id'] == 's_recipe')['previous_sha256']
    assert after == (ROOT / 'course/training.md').read_bytes()
    recipe = '.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda'
    assert recipe in before.decode() and recipe in after.decode()
    def recipe_block(raw):
        text = raw.decode()
        start = text.index('<details id="fixed-sft-recipe"')
        return text[start:text.index('</details>', start) + len('</details>')]
    assert recipe_block(before) == recipe_block(after)
    run = ROOT / 'outputs/reviewer-tools/runs/20261020T102216262947Z'
    files = {name: {'sha256': sha((run / name).read_bytes()), 'content': (run / name).read_text()} for name in ['extraction.json', 'execution.json', 'environment.json', 'stdout.txt', 'stderr.txt', 'fence-1.py', 'bootstrap.py']}
    execution = json.loads(files['execution.json']['content'])
    assert execution['exit_code'] == 0 and execution['source_sha256'] == sha(frozen)
    assert json.loads(files['extraction.json']['content'])['svg_references'] == []
    cli_paths = ['scripts/train.py', 'scripts/evaluate.py', 'scripts/infer.py', 'scripts/quantize.py', 'scripts/prepare_data.py', 'scripts/fetch_training_assets.py']
    help_results = [command([sys.executable, path, '--help']) for path in cli_paths]
    help_results.append(command([sys.executable, '-m', 'scripts.course_experiments.run', '--help']))
    for result, path in zip(help_results[:6], cli_paths):
        assert ('--device' in result['stdout']) == (path in cli_paths[:3])
    selection = []
    for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
        with patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
            observed = str(choose_device())
        assert observed == expected
        selection.append({'mock_cuda_available': cuda, 'mock_mps_available': mps, 'expected': expected, 'observed': observed, 'scope': '只模擬選擇邏輯，未執行該硬體運算'})
    assert inspect.signature(load_checkpoint).parameters['device'].default == 'cpu'
    relevant = {'tiny_perceptron/training.py': ['def choose_device', 'def load_checkpoint'], 'scripts/train.py': ['p.add_argument("--device"'], 'scripts/evaluate.py': ['p.add_argument("--device"'], 'scripts/infer.py': ['parser.add_argument("--device"'], 'scripts/course_experiments/run.py': ['parser.add_argument("--device"'], 'scripts/quantize.py': ['model, source = load_checkpoint(args.checkpoint)', 'replace_linear_layers(model, args.bits)']}
    code_evidence = []
    for path, markers in relevant.items():
        raw = (ROOT / path).read_bytes()
        lines = raw.decode().splitlines()
        excerpts = []
        for marker in markers:
            start = next(i for i, line in enumerate(lines) if marker in line)
            excerpts.append({'identifier': marker, 'first_line_computed': start + 1, 'text': '\n'.join(lines[start:start + (10 if marker == 'def choose_device' else 3)])})
        code_evidence.append({'path': path, 'sha256': sha(raw), 'excerpts': excerpts})
    old_execution = json.loads((BASE / 'verify_checkpoint-output.json').read_bytes())
    gpu = old_execution['gpu_record_audit']
    assert sha((ROOT / gpu['path']).read_bytes()) == gpu['sha256']
    result = {'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-7/recheck-current-01.py', 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'mps_available': str(torch.backends.mps.is_available())}, 'original_report': {'path': str(original_path.relative_to(ROOT)), 'sha256': sha(original_raw)}, 'previous_issues': report['issues'], 'resolved_issue_ids': [], 'hash_checks': checks, 'section_sha256': sha(frozen), 'section_unchanged': True, 'source_freeze_sha256': sha(freeze_raw), 'requested_git_diff': diff, 'training_source_history': {'before_sha256': sha(before), 'after_sha256': sha(after), 'after_matches_worktree': True, 'cited_recipe': recipe, 'entire_fixed_sft_recipe_unchanged': True, 'recipe_excerpt': recipe_block(after)}, 'section_execution': {'invoked_command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.7" --execute', 'files': files}, 'help_checks': help_results, 'device_selection_checks': selection, 'quantize_load_device_default': 'cpu', 'code_evidence': code_evidence, 'gpu_result_file_unchanged': True, 'claims_reviewed': [{'claim_id': c['id'], 'status': 'verified', 'basis': '原文與相關實作、已引用證據的 bytes 均未變；本回合重跑原例。' if c['id'] != 'separate_experiments_same_command' else '重讀現行固定配方；引用的完整補充區與原版相同，實驗入口及原紀錄雜湊未變。'} for c in report['claims']], 'auxiliary_change_reviews': [{'location': 'course/training.md「Python 怎麼寫」', 'status': 'verified', 'details': 'train/evaluate/infer 的 auto 預設與 choose_device 分支吻合；experiment runner 預設 cpu；quantize 呼叫 load_checkpoint 的 cpu 預設，且與 prepare_data、fetch_training_assets 均無 --device。以原碼、--help 與選擇邏輯模擬核對，不把 mock 當硬體實測。'}, {'location': 'course/training.md 的 circle/high/square,low 期望答案說明', 'status': 'verified', 'details': '新增「不同電腦可能得到不同答案」是非確定性限制，不是新增測得差異；沿用已封存 PyTorch Reproducibility 原文，只支持沒有跨平台保證，不支持任何新的差異率。'}, {'location': 'course/training.md 的 blue 量化比較說明', 'status': 'verified', 'details': '量化流程只替換 Linear 的權重表示，沒有訓練或將答案強制設成 blue；新增句要求比較相同題目前後答案，沒有保證量化必然保留或修正答案。'}], 'result': '全部核對通過'}
    output = BASE / 'recheck-current-01-output.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'path': str(output.relative_to(ROOT)), 'sha256': sha(output.read_bytes()), 'result': result['result']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
