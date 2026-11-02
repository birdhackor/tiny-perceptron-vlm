"""Verify W.1 state/restart and its altered-text exercise in a real CPU kernel."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[4]
PREFIX = Path('/workspace/.review-artifacts/omni_wave02_technical_first_steps/kernel-prefix')
spec_manager = KernelSpecManager(kernel_dirs=[str(PREFIX / 'share/jupyter/kernels')])
manager = KernelManager(kernel_name='tiny-perceptron', kernel_spec_manager=spec_manager)
manager.start_kernel(cwd=str(ROOT / 'notebooks/01'))
client = manager.blocking_client()
client.start_channels()


def execute(code):
    output = []

    def receive(msg):
        if msg['header']['msg_type'] in ('stream', 'error', 'execute_result'):
            output.append({'type': msg['header']['msg_type'], 'content': msg['content']})

    reply = client.execute_interactive(code, timeout=30, output_hook=receive)
    assert reply['content']['status'] == 'ok', reply
    return output


try:
    client.wait_for_ready(timeout=30)
    print(json.dumps({'check':'environment','python':platform.python_version(),
                      'jupyter_client':importlib.metadata.version('jupyter_client'),
                      'ipykernel':importlib.metadata.version('ipykernel'),
                      'created_at':datetime.now(timezone.utc).isoformat()}, ensure_ascii=False))
    print(json.dumps({'check':'first_cell','output':execute('label = "舊資料"\nprint(label)')},ensure_ascii=False))
    print(json.dumps({'check':'state_persists','output':execute('print(label)')},ensure_ascii=False))
    manager.restart_kernel(now=True)
    client.wait_for_ready(timeout=30)
    clean = execute('print("label" in globals())')
    assert clean[0]['content']['text'] == 'False\n'
    print(json.dumps({'check':'restart_clears_state','output':clean},ensure_ascii=False))
    nb = json.loads((ROOT / 'notebooks/01/1.1.ipynb').read_text())
    setup = next(''.join(c['source']) for c in nb['cells'] if c.get('metadata',{}).get('course_setup'))
    text_code = next(''.join(c['source']) for c in nb['cells'] if 'text = "貓看狗，狗看貓。"' in ''.join(c['source']))
    output = execute(setup + '\n' + text_code.replace('貓看狗，狗看貓。','鳥看狗，狗看鳥。'))
    combined = ''.join(o['content'].get('text','') for o in output)
    assert combined.splitlines()[-1] == '鳥看狗，狗看鳥。'
    print(json.dumps({'check':'W1_fresh_kernel_exercise','output':output,
                      'notebook_sha256':hashlib.sha256((ROOT/'notebooks/01/1.1.ipynb').read_bytes()).hexdigest()},ensure_ascii=False))
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)
print(json.dumps({'check':'finished','passed':True},ensure_ascii=False))
