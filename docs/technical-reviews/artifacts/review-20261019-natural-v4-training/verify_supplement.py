"""核對頁面的保存程式與 Bash 語法，並記錄可讀證據的範圍。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PIN = 'a008cec6fd447ca977503dbfd1130cd59a7c7a49'
comparison = {}
for path in ('tiny_perceptron/natural_assistant.py', 'scripts/modal_natural.py'):
    command = ['git', 'show', f'{PIN}:{path}']
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    comparison[path] = {'command': command, 'returncode': process.returncode, 'stderr': process.stderr}

page = (ROOT / 'docs/course-review-20261019/freeze-01/sources/natural-v4-training.md').read_text()
body = re.search(r"\.venv-natural/bin/python - <<'PY'\n(.*?)\nPY\n", page, re.S).group(1)
raw = json.loads((ROOT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw/result.json').read_bytes())
with tempfile.TemporaryDirectory(prefix='snapshot-', dir=ARTIFACTS) as temporary:
    fixture = Path(temporary)
    manifest = fixture / 'docs/natural-assistant/v4/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}\n')
    manifest.with_name('validation-protocol-lower-lr.json').write_text('{}\n')
    fixture_record = dict(raw, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    validation = fixture / 'outputs/natural-my-v4/validation/result.json'
    validation.parent.mkdir(parents=True)
    validation.write_text(json.dumps(fixture_record))
    run = subprocess.run([sys.executable, '-'], input=body, cwd=fixture, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    snapshot = fixture / 'outputs/natural-my-v4/chosen-base-before-test'
    config = json.loads((snapshot / 'configuration.json').read_bytes())
    assert config['adapter'] is None and config['selected_variant'] == 'base'
    assert config['max_new_tokens'] == 384 and config['do_sample'] is False
    for line in (snapshot / 'files.sha256').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
    repeat = subprocess.run([sys.executable, '-'], input=body, cwd=fixture, capture_output=True, text=True)
    assert repeat.returncode and 'FileExistsError' in repeat.stderr
    snapshot_result = {
        'stdout': run.stdout, 'returncode': run.returncode, 'configuration': config,
        'saved_filenames': sorted(file.name for file in snapshot.iterdir()),
        'all_recorded_hashes_match': True, 'repeat_returncode': repeat.returncode,
        'repeat_stderr': repeat.stderr,
        'scope': '原封執行頁面 Python 區塊；fixture 清單以小檔替代，使用真實驗證結果欄位，未執行 GPU 生成。',
    }
    syntax = []
    for block in re.findall(r'```bash\n(.*?)```', page, re.S):
        process = subprocess.run(['bash', '-n'], input=block, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        syntax.append({'returncode': process.returncode, 'stderr': process.stderr})

command = [sys.executable, 'scripts/fetch_natural_data.py', '--manifest',
           'docs/natural-assistant/v4/manifest.json', '--revision',
           '3a282925bb46bb68c87fed4157ae4df5f6de30ee', '--manifest-sha256',
           '0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60',
           '--output', str(ARTIFACTS / 'not-created-data'), '--list']
listing = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
assert listing.returncode == 0 and not (ARTIFACTS / 'not-created-data').exists(), listing.stderr
audit_path = 'docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json'
index_path = 'docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json'
audit_record = json.loads((ROOT / audit_path).read_bytes())
result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-natural-v4-training/verify_supplement.py',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU',
                    'cuda_available': str(torch.cuda.is_available()), 'bf16_supported': str(torch.cuda.is_bf16_supported())},
    'historical_implementation_access': comparison,
    'snapshot': snapshot_result, 'bash_syntax_checks': syntax,
    'data_list': {'command': command, 'returncode': listing.returncode, 'stdout': listing.stdout, 'stderr': listing.stderr},
    'validation_runtime_versions': raw['versions'],
    'training_record_read': [{'path': path, 'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                             for path in (audit_path, index_path)],
    'training_provenance_limit': '未讀原始訓練 provenance、result 或權重；訓練摘要與索引可讀，但不能重新執行原先稽核器。',
    'probe_setup_corrections': [
        '初次在 PEFT lora/model.py 尋找凍結方法得到 StopIteration；實際方法位於 tuners/tuners_utils.py，改正後完成來源擷取。',
        '嘗試 git show 原始訓練提交得到 exit 128；本檔保留當次可重現的命令與 stderr，不假定這個提交仍在目前 Git 物件庫。',
    ],
}
recorded_hashes = audit_record['paired_training_control']['runtime_source_sha256_unchanged_between_heads']
result['training_source_hash_comparison'] = {
    path: {'recorded': expected, 'current': hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
           'matches': expected == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
    for path, expected in recorded_hashes.items()
}
with (ARTIFACTS / 'supplemental-execution.json').open('x') as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
    file.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
