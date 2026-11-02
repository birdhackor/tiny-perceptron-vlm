"""Fresh anonymous ZIP installation and six CPU guide examples, never final data."""
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave

root = Path('/workspace/work/omni-book-integrated-v1')
out = root / 'outputs/omni-P5-public-anonymous-guide-v1'
out.mkdir(exist_ok=False)
models = out / 'fresh-model-root'
assert not models.exists()
env = os.environ.copy()
for name in ('GH_TOKEN', 'GITHUB_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
             'HUGGINGFACE_HUB_TOKEN', 'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'):
    env.pop(name, None)
env.update(UV_PROJECT_ENVIRONMENT='/workspace/tiny-perceptron-vlm/.venv', PYTHONPATH=str(root))
prefix = ['uv', 'run', '--extra', 'cpu', '--no-sync', 'python', '-m']
checks = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(name, module, arguments):
    cmd = prefix + [module] + arguments
    stdout = out / (name + '.stdout.json')
    stderr = out / (name + '.stderr.log')
    started = datetime.now(UTC).isoformat()
    with stdout.open('xb') as stream, stderr.open('xb') as error:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=stream, stderr=error, timeout=300)
    row = {'case': name, 'started_at': started, 'finished_at': datetime.now(UTC).isoformat(),
           'command': cmd, 'exit_code': result.returncode,
           'stdout': {'path': str(stdout.relative_to(root)), 'sha256': sha(stdout)},
           'stderr': {'path': str(stderr.relative_to(root)), 'sha256': sha(stderr)}}
    checks.append(row)
    assert result.returncode == 0, name
    return json.loads(stdout.read_bytes())

download = run('anonymous-download', 'scripts.omni_revision.public_download', ['--output', str(models)])
assert download['anonymous'] is True
receipt = json.loads((models / 'download-receipt.json').read_bytes())
assert set(receipt['assets']) == {'av', 'choice', 'codec', 'talker'}
assert receipt['anonymous'] is True and receipt['full_continuation_requested'] is False
for name, mode, extra in [
    ('choice', 'choice', []), ('av-early', 'av', ['--which', '0']),
    ('av-late', 'av', ['--which', '1']),
    ('av-missing-image', 'av', ['--missing', 'image']),
    ('av-missing-audio', 'av', ['--missing', 'audio']), ('sound', 'sound', []),
]:
    args = [mode, '--model-root', str(models)] + extra
    target = out / (name + '.wav')
    if name in ('av-early', 'av-late', 'sound'):
        args += ['--output-wav', str(target)]
    value = run(name, 'scripts.omni_revision.public_demo', args)
    if name == 'choice':
        assert value['prediction']['answer'] == 'record' and value['information']['answer'] == 'yes'
        assert value['accepted'] is True and value['decision'] == 'accept'
    elif name.startswith('av-'):
        expected = {'av-early': 'circle', 'av-late': 'square'}.get(name, 'unknown')
        assert value['answer'] == expected
        if expected == 'unknown':
            assert 'sound' not in value and 'waveform' not in value
        else:
            assert value['sound']['samples'] == 4000
    else:
        assert value['samples'] == 4000 and value['sample_rate'] == 8000
    if target.exists():
        with wave.open(str(target), 'rb') as waveform:
            assert (waveform.getnchannels(), waveform.getsampwidth(), waveform.getframerate(), waveform.getnframes()) == (1, 2, 8000, 4000)
        checks[-1]['WAV'] = {'path': str(target.relative_to(root)), 'bytes': target.stat().st_size, 'sha256': sha(target)}
    checks[-1]['key_output'] = {k: value[k] for k in ('answer', 'accepted', 'decision', 'samples') if k in value}
result = {
    'created_at': datetime.now(UTC).isoformat(), 'observer_task': '/root',
    'kind': 'author actual fresh anonymous public download and CPU guide checks, not independent textbook review',
    'existing_CPU_environment': 'Python and CPU dependencies already installed; --no-sync preserves that environment',
    'fresh_download_destination': str(models), 'credential_variables_removed_from_children': True,
    'public_download_code_uses_no_Authorization_header': True,
    'download_receipt': {'path': str((models / 'download-receipt.json').relative_to(root)), 'sha256': sha(models / 'download-receipt.json')},
    'checks': checks, 'passed': 7, 'guide_inference_cases': 6,
    'scope': 'First training-family AV example and fixed public Choice/sound examples only. No training, no formal final evaluation, no new Modal resources.',
}
(out / 'receipt-v1.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'download': 'anonymous4originalZIPs', 'CPU_guide_cases': 6, 'all_passed': True}))
