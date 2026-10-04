from pathlib import Path
import subprocess, json, os
root = Path('/workspace/tiny-perceptron-vlm')
out = root / 'docs/reader-reviews/artifacts/natural-v4-cleanup/9.8'
python = root / '.venv/bin/python'
records = []
for name in ['example', 'exercise']:
    command = [str(python), '-I', '-B', str(out / (name + '.py'))]
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=10)
    (out / (name + '-stdout.txt')).write_text(result.stdout)
    (out / (name + '-stderr.txt')).write_text(result.stderr)
    lines = result.stdout.splitlines()
    expected_test_lines = 1 if name == 'example' else 2
    checks = {
        'exit_zero': result.returncode == 0,
        'train_lines': sum(line.startswith('train ') for line in lines) == 1,
        'test_lines': sum(line.startswith('test ') for line in lines) == expected_test_lines,
        'expected_target_on_each_prompt': all(line.endswith('→ 資訊不足，請提供數量或可看清的圖片。') for line in lines[:-1]),
        'final_false': bool(lines) and lines[-1] == '訓練問句與測試問句相同 False',
        'total_lines': len(lines) == expected_test_lines + 2,
    }
    records.append({'name': name, 'command': command, 'cwd': str(root), 'timeout_seconds': 10, 'cpu_only': True, 'returncode': result.returncode, 'stdout_artifact': name + '-stdout.txt', 'stderr_artifact': name + '-stderr.txt', 'prediction_checks': checks, 'scope': 'Manual target-printing demonstration only. No model weights, training, dataset download or benchmark evaluation.'})
    print(name + ':')
    print(result.stdout, end='')
    print('Prediction checks: ' + json.dumps(checks, ensure_ascii=False))
(out / 'execution-evidence.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
if not all(all(record['prediction_checks'].values()) for record in records):
    raise SystemExit('A saved prediction failed')
