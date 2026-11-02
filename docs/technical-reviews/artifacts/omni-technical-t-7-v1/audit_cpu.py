"""T.7 independent review: CLI checks, a two-update CPU fixture, and tiny card demo.

The random fixture is not the T.5 trained checkpoint. No 200-step language-model
training, external-data download, CUDA, Modal, or publication is performed.
"""

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _preference_parts
from scripts.course_experiments.common import Context, split_records
from scripts.course_experiments.text import _save_splits, arithmetic_records
from scripts.train import parser, prepare_examples
from tiny_perceptron.data import ByteTokenizer, load_jsonl, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint

EXTERNAL = Path('/workspace/work/reference/omni-technical-t-7-v1/execution')
EXTERNAL.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).with_name('execution.json')
if OUT.exists():
    raise SystemExit('Refusing to overwrite original execution receipt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    command = [sys.executable, *map(str, args)]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=50)
    row = {'command': shlex.join(command), 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    if p.returncode:
        raise RuntimeError(json.dumps(row, ensure_ascii=False))
    return row


receipt = {
    'created_at': datetime.now(UTC).isoformat(),
    'reviewer_task': '/root/omni_technical_coordinator/technical_t_7',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git': torch.version.git_version, 'device': 'cpu',
                    'executable': sys.executable, 'platform': platform.platform()},
    'limits': 'Random model fixture only; two LM updates at the 200-step schedule, not completed T.5/T.7 training. '
              'Full posttraining is only the tiny finite-card CPU teaching example. Historical GPU run is inspected, not rerun.',
    'cli_help': [],
}
snapshot = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.7.md'
receipt['snapshot_sha256'] = sha(snapshot)
receipt['literal_train_args'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(parser().parse_args([
    '--task', 'dpo', '--checkpoint', 'checkpoints/style.pt', '--data',
    'data/generated/preference/train.jsonl', '--train', '--steps', '200',
    '--output', 'checkpoints/preferred.pt'])).items()}
for args in [('scripts/prepare_data.py', '--help'), ('scripts/train.py', '--help'),
             ('scripts/infer.py', '--help'), ('scripts/fetch_training_assets.py', '--help'),
             ('-m', 'scripts.course_experiments.run', '--help'),
             ('-m', 'scripts.course_experiments.run', '--list-assets', 'dpo'),
             ('-m', 'scripts.course_experiments.run', '--list-assets', 'posttraining')]:
    receipt['cli_help'].append(run(args))

receipt['prepare_data'] = run(['scripts/prepare_data.py', '--kind', 'preference', '--output', EXTERNAL/'generated'])
parts = {s: load_jsonl(EXTERNAL/'generated'/'preference'/f'{s}.jsonl') for s in ('train', 'validation', 'test')}
receipt['generated'] = {s: {'records': len(rows), 'sha256': sha(EXTERNAL/'generated'/'preference'/f'{s}.jsonl'),
                            'example_0_plus_5': [r for r in rows if r['prompt'] == '0+5=?']} for s, rows in parts.items()}

torch.manual_seed(42)
torch.set_num_threads(2)
fixture = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=256))
initial = state_sha(fixture.state_dict())
checkpoint = EXTERNAL/'untrained-style-fixture.pt'
save_checkpoint(checkpoint, fixture, metadata={'scope': 'random shape-compatible fixture, NOT T.5 trained style'})
receipt['fixture'] = {'config': vars(fixture.config), 'sha256': sha(checkpoint), 'initial_state_sha256': initial}
dataset = EXTERNAL/'generated'/'preference'/'train.jsonl'
args = parser().parse_args(['--task', 'dpo', '--data', str(dataset), '--max-length', '256'])
pairs = prepare_examples(args, ByteTokenizer())
receipt['dpo_input'] = {'pairs': len(pairs), 'longest_input': max(len(x) for pair in pairs for x,y in pair),
                        'first_raw_row': parts['train'][0], 'sides': []}
for pair in pairs[:1]:
    for x, y in pair:
        bx, by, mask = pad_batch([(x, y)], max_length=256)
        receipt['dpo_input']['sides'].append({'input_ids': x.tolist(), 'labels': y.tolist(),
            'shape': list(bx.shape), 'dtype': str(bx.dtype), 'valid_mask': mask.tolist(),
            'effective_answer_tokens_including_eos': int((by != -100).sum())})

common = ['scripts/train.py', '--task', 'dpo', '--checkpoint', checkpoint,
          '--data', dataset, '--steps', '200', '--device', 'cpu']
receipt['dry_run'] = run([*common, '--output', EXTERNAL/'not-created.pt'])
receipt['dry_run_no_checkpoint_saved'] = not (EXTERNAL/'not-created.pt').exists()
preferred = EXTERNAL/'two-updates-fixture.pt'
receipt['two_update_training'] = run([*common, '--train', '--stop-after', '2', '--output', preferred])
payload = torch.load(preferred, map_location='cpu', weights_only=True)
reference = payload['training_state']['dpo_reference']['model']
receipt['two_update_result'] = {'step': payload['step'], 'policy_state_sha256': state_sha(payload['model']),
    'reference_state_sha256': state_sha(reference), 'reference_equal_initial': state_sha(reference) == initial,
    'policy_changed': state_sha(payload['model']) != initial,
    'data_sha256': payload['metadata']['data_sha256'], 'schedule_steps': payload['metadata']['schedule_steps']}
receipt['inference'] = []
for source in (checkpoint, preferred):
    infer_args = ['scripts/infer.py', source, '--chat', '--prompt', '0+5=?', '--tokens', '32',
                  '--temperature', '0', '--json', '--device', 'cpu']
    a, b = run(infer_args), run(infer_args)
    receipt['inference'].append({'first': a, 'repeat_exactly_equal': a['stdout'] == b['stdout']})

receipt['posttraining_cli'] = run(['-m', 'scripts.course_experiments.run', '--experiment', 'posttraining',
                                   '--device', 'cpu', '--output', EXTERNAL/'cards'])
cards_path = EXTERNAL/'cards'/'posttraining'/'result.json'
cards = json.loads(cards_path.read_text())
historical_cards = json.loads((ROOT/'docs/course-experiments/results/posttraining.json').read_text())
receipt['posttraining_result'] = {'raw_result_path': str(cards_path), 'raw_result_sha256': sha(cards_path),
    'elapsed_seconds': cards['elapsed_seconds'], 'evidence_status': cards['evidence_status'],
    'config': cards['results']['config'], 'splits': cards['results']['splits'],
    'test_policies': cards['results']['evaluations']['test']['policies'],
    'state_checks': {r['file']: r['state_sha256'] for r in cards['results']['checkpoints']},
    'historical_state_checks': {r['file']: r['state_sha256'] for r in historical_cards['results']['checkpoints']}}

historical = json.loads((ROOT/'docs/course-experiments/results/dpo.json').read_text())
history_source = subprocess.check_output(['git', 'show', historical['revision']+':scripts/course_experiments/behavior.py'], cwd=ROOT)
current_source = (ROOT/'scripts/course_experiments/behavior.py').read_bytes()
ctx = Context('cpu', EXTERNAL/'recreated-historical-splits', EXTERNAL, ROOT/'assets/training', seed=42)
recreated = _save_splits(ctx, _preference_parts(split_records(arithmetic_records(), seed=42)))
receipt['historical_dpo'] = {'report_sha256': sha(ROOT/'docs/course-experiments/results/dpo.json'),
    'revision': historical['revision'], 'code_sha256_recorded': historical['code_sha256']['scripts/course_experiments/behavior.py'],
    'git_show_code_sha256': hashlib.sha256(history_source).hexdigest(),
    'dpo_functions_unchanged_since_historical': history_source[history_source.index(b'def _preference_parts'):]
        == current_source[current_source.index(b'def _preference_parts'):],
    'historical_data': historical['results']['data'], 'recreated_data': recreated,
    'recreated_split_hashes_match': all(historical['results']['data'][s]['sha256']==recreated[s]['sha256']
                                      for s in ('train','validation','test')),
    'elapsed_seconds': historical['elapsed_seconds'], 'timing_scope': historical['timing_scope'],
    'device': historical['device'], 'gpu': historical['gpu'], 'seed': historical['seed'],
    'torch_version': historical['torch_version'], 'modal_run_id': historical['modal']['run_id'],
    'sample_4_plus_2_preference': next(r for r in historical['results']['runs']['model']['preference']['test']['samples'] if r['prompt']=='4+2=?'),
    'sample_4_plus_2_generation': next(r for r in historical['results']['runs']['model']['arithmetic']['test']['samples'] if r['messages'][0]['content']=='4+2=?')}

receipt['completed_at'] = datetime.now(UTC).isoformat()
OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'execution_receipt': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
    'two_update_result': receipt['two_update_result'],
    'generated_counts': {s: len(rows) for s, rows in parts.items()},
    'posttraining_elapsed_seconds': cards['elapsed_seconds'],
    'posttraining_states_match_historical': receipt['posttraining_result']['state_checks'] == receipt['posttraining_result']['historical_state_checks'],
    'historical_dpo_split_hashes_match': receipt['historical_dpo']['recreated_split_hashes_match']}, ensure_ascii=False))
