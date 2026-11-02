"""19.19 獨立技術核對；僅下載固定小模型與 CPU 推論，不訓練。"""

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-19-19-v1')
SNAP = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/19.19.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, env=None):
    started = datetime.now(UTC).isoformat()
    value = subprocess.run(
        ['bash', '-c', 'source .venv/bin/activate\n' + command],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
    )
    result = {'command': command, 'started_at': started, 'exit_code': value.returncode,
              'stdout': value.stdout, 'stderr': value.stderr}
    if value.returncode == 0:
        try:
            result['json'] = json.loads(value.stdout)
        except ValueError:
            pass
    return result


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    commands = re.findall(r'```bash\n(.*?)```', SNAP.read_text(), re.S)[0].strip().splitlines()
    release = json.loads((ROOT / 'tiny_perceptron/lesson_products/release.json').read_bytes())
    embedded_release = json.loads(re.findall(r'```json\n(.*?)```', SNAP.read_text(), re.S)[0])
    assert release == embedded_release
    report = {'started_at': datetime.now(UTC).isoformat(),
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                              'torch_git': torch.version.git_version, 'device': 'cpu',
                              'platform': platform.platform()},
              'source_sha256': sha(SNAP), 'embedded_release_matches': True,
              'scope': '固定既有 B/C 權重的下載、離線 CPU 推論及本機 HTTP；沒有新訓練或新版產品驗收。',
              'preexisting_default_packages': {p: (ROOT / 'outputs/lesson-products' / p).exists() for p in ('b', 'c')}}
    assert not any(report['preexisting_default_packages'].values()), 'fresh download requires empty default locations'
    report['embedded_commands'] = [run(command) for command in commands]
    assert all(x['exit_code'] == 0 for x in report['embedded_commands'])
    report['packages'] = {}
    for product in ('b', 'c'):
        directory = ROOT / 'outputs/lesson-products' / product
        manifest_path = directory / 'artifact-manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        assert sha(manifest_path) == release['products'][product]['manifest_sha256']
        files = {name: {'bytes': (directory/name).stat().st_size, 'sha256': sha(directory/name)} for name in manifest['files']}
        assert files == manifest['files']
        state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        assert all(isinstance(v, torch.Tensor) for v in state.values())
        receipt = json.loads((directory / 'training-receipt.json').read_bytes())
        report['packages'][product] = {
            'directory': str(directory), 'manifest_sha256': sha(manifest_path), 'files': files,
            'parameter_keys': len(state), 'all_values_tensors': True,
            'has_optimizer_or_rng_keys': any(k in state for k in ('optimizer', 'rng', 'sampler_rng', 'torch_rng')),
            'training_receipt_selected_fields': {k: receipt.get(k) for k in ('product', 'random_initialization', 'seed', 'selected_step', 'last_step')},
            'config': json.loads((directory/'config.json').read_bytes()),
        }
    guard = CACHE / 'offline-guard'
    guard.mkdir(exist_ok=True)
    guard_source = '''import sys
def deny_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise RuntimeError("REVIEW_OFFLINE_NETWORK_BLOCKED:" + event)
sys.addaudithook(deny_network)
'''
    (guard / 'sitecustomize.py').write_text(guard_source)
    offline_env = dict(os.environ)
    offline_env['PYTHONPATH'] = str(guard) + os.pathsep + str(ROOT)
    report['offline_guard'] = {
        'path': str(guard / 'sitecustomize.py'), 'sha256': sha(guard/'sitecustomize.py'),
        'mechanism': 'Python audit hook denies socket.connect, socket.getaddrinfo and socket.sendto in each child interpreter; no operating-system network namespace isolation claimed.',
        'probe': run("python -c 'import socket; socket.create_connection((\"huggingface.co\",443))'", offline_env),
    }
    assert report['offline_guard']['probe']['exit_code'] != 0
    assert 'REVIEW_OFFLINE_NETWORK_BLOCKED' in report['offline_guard']['probe']['stderr']
    report['network_blocked_inference'] = [run(command, offline_env) for command in commands[2:]]
    assert all(x['exit_code'] == 0 for x in report['network_blocked_inference'])
    prefix = 'python -m tiny_perceptron.lesson_products '
    box = "'兩盒追蹤;A=2;B=6;事件=交換;查詢=B'"
    report['extra_cli'] = [
        run(prefix + 'auto --state ' + box + ' --effort medium --max-loops 2', offline_env),
        run(prefix + 'judge --state ' + box + ' --question choice', offline_env),
        run(prefix + 'judge --state ' + box + ' --question rubric', offline_env),
        run(prefix + "solve --state '請幫我寫一首詩' --loops 3", offline_env),
    ]
    assert all(x['exit_code'] == 0 for x in report['extra_cli'][:3])
    assert report['extra_cli'][0]['json']['status'] == 'budget_limited'
    assert report['extra_cli'][0]['json']['requested_loops'] == 3
    assert report['extra_cli'][0]['json']['b_loops'] == 2
    assert report['extra_cli'][3]['exit_code'] != 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server_command = prefix + 'serve --port 8765'
    server_log = CACHE / 'http-server.log'
    http = {'command': server_command, 'requests': []}
    with server_log.open('w') as log:
        server = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-m', 'tiny_perceptron.lesson_products', 'serve', '--port', '8765'], cwd=ROOT, stdout=log, stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError('server exited: ' + server_log.read_text())
                try:
                    with opener.open('http://127.0.0.1:8765/health', timeout=0.2) as response:
                        health = {'endpoint': '/health', 'http_status': response.status, 'json': json.load(response)}
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError('HTTP health timeout')
            http['requests'].append(health)
            port_hex = f'{8765:04X}'
            http['listening_sockets'] = [line.strip() for line in Path(f'/proc/{server.pid}/net/tcp').read_text().splitlines()[1:] if line.split()[1].endswith(':'+port_hex) and line.split()[3]=='0A']
            assert all('0100007F:' + port_hex in line for line in http['listening_sockets'])
            assert http['listening_sockets']
            for endpoint, payload in [
                ('solve', {'state': '兩盒追蹤;A=2;B=6;事件=交換;查詢=B', 'loops': 3}),
                ('judge', {'state': '卡片進度;申請編號=?', 'question': 'info'}),
                ('auto', {'state': '兩盒追蹤;A=2;B=6;事件=交換;查詢=B', 'effort': 'auto'}),
                ('solve', {'state': '任意聊天文字', 'loops': 3}),
            ]:
                request = urllib.request.Request('http://127.0.0.1:8765/'+endpoint,
                    data=json.dumps(payload,ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
                try:
                    response = opener.open(request, timeout=5)
                except urllib.error.HTTPError as e:
                    response = e
                with response:
                    http['requests'].append({'endpoint':'/'+endpoint,'input':payload,'http_status':response.status,'json':json.load(response)})
            assert [x['http_status'] for x in http['requests']] == [200,200,200,200,400]
        finally:
            server.terminate()
            server.wait(timeout=10)
    http['server_log_sha256'] = sha(server_log)
    http['server_log_path'] = str(server_log)
    report['http'] = http
    report['completed_at'] = datetime.now(UTC).isoformat()
    target = ART / 'execution-v1.json'
    with target.open('x') as file:
        json.dump(report,file,ensure_ascii=False,indent=2)
        file.write('\n')
    print(json.dumps({'path':str(target),'sha256':sha(target),'five_embedded_commands_exit_codes':[x['exit_code'] for x in report['embedded_commands']], 'offline_inference_exit_codes':[x['exit_code'] for x in report['network_blocked_inference']], 'http_statuses':[x['http_status'] for x in http['requests']]},ensure_ascii=False))


if __name__ == '__main__':
    main()
