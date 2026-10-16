"""19.19 reviewer checks: CPU only, no parameter updates or training."""

import contextlib
import hashlib
import io
import json
import platform
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.selftrained.chat import main as chat_main
from scripts.selftrained.train import restore_rng, rng_state, seed_all

WORK = Path('/workspace/work/curriculum-20261016/19.19')
torch.set_num_threads(1)
seed_all(1919)
dropout = torch.nn.Dropout(0.5)
before_eval = torch.get_rng_state().clone()
dropout.eval()
after_eval = torch.get_rng_state().clone()
x = torch.ones(16)
first, second = dropout(x), dropout(x)
samples = [torch.multinomial(torch.tensor([0.5, 0.5]), 16, replacement=True).tolist() for _ in range(2)]
assert torch.equal(before_eval, after_eval)
assert torch.equal(first, x) and torch.equal(second, x)
assert samples[0] != samples[1]

state = rng_state()
draw1 = [random.random(), float(np.random.rand()), torch.rand(2).tolist()]
restore_rng(state)
draw2 = [random.random(), float(np.random.rand()), torch.rand(2).tolist()]
assert draw1 == draw2

linear = torch.nn.Linear(1, 1, bias=False)
outputs = []
for weight in (1.0, -1.0):
    linear.load_state_dict({'weight': torch.tensor([[weight]])})
    outputs.append(linear(torch.tensor([[2.0]])).item())
assert outputs == [2.0, -2.0]

def no_network(*args, **kwargs):
    raise AssertionError('Network call attempted during offline verification')

socket.create_connection = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
offline_argv = [
    '--model-dir', str(WORK / 'public-moe-joint'),
    '--asset-dir', str(WORK),
    '--manifest-sha256', 'f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e',
    '--messages', str(ROOT / 'docs/selftrained/examples/v2/text.messages.json'),
    '--task', 'text', '--device', 'cpu', '--max-new-tokens', '128', '--threads', '2',
]
with contextlib.redirect_stdout(io.StringIO()):
    offline = chat_main(offline_argv)
online = json.loads((WORK / 'public-online.json').read_text())
assert offline['answer'] == online['answer']
assert offline['answer'] == '1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。'
payloads = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (WORK / 'public-moe-joint').iterdir()}

result = {
    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': 'CPU; CUDA unavailable; MPS unavailable'},
    'eval': {'rng_state_unchanged': torch.equal(before_eval, after_eval),
             'dropout_eval_is_identity': torch.equal(first, x),
             'random_samples_after_eval': samples,
             'sampling_still_uses_rng': samples[0] != samples[1]},
    'rng_restore': {'first': draw1, 'restored': draw2, 'equal': draw1 == draw2},
    'same_architecture_changed_weights': {'input': [[2.0]], 'weights': [[[1.0]], [[-1.0]]],
                                         'outputs': outputs,
                                         'scope': 'possibility demonstration; no training, Auto quality, or latency measured'},
    'public_a': {'online_source': online['public_source'],
                 'offline_argv': offline_argv,
                 'offline_network': 'socket connect/create_connection/connect_ex replaced with raising guards',
                 'online_answer': online['answer'], 'offline_answer': offline['answer'],
                 'online_offline_generated_ids_equal': online['generations'][0]['generated_ids'] == offline['generations'][0]['generated_ids'],
                 'manifest_hash_pinned': offline['model']['manifest_hash_pinned'],
                 'selected_step': offline['model']['selected_step'], 'payload_sha256': payloads,
                 'cases': 1, 'generation_calls_online': 1, 'generation_calls_offline': 1,
                 'scope': 'known text smoke case only; no heldout quality estimate; no training'},
}
destination = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/19.19-execution.json'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
