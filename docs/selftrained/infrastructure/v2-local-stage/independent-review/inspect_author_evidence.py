"""Read-only independent verification of author raw artifacts, without trainer runs."""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path('/tmp/p5-local-wrapper-independent')
AUTHOR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/p5-local-stage-candidate')
REPO = Path('/workspace/selftrained-v2')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

out = {'scope': 'Read-only raw artifact verification; author subprocesses are distinct from independent reruns.',
       'author_wrapper_sha256': sha(AUTHOR / 'scripts/selftrained/train_local_stage.py'),
       'author_patch_sha256': sha(AUTHOR / 'candidate.patch'),
       'author_proof_final_sha256': sha(AUTHOR / ('proof-final.json' if (AUTHOR / 'proof-final.json').exists() else 'proof.json')), 'attempts': []}
for name in ('baseline-joint', 'weighted44-joint', 'native414-joint', 'invalid-zero-step',
             'weighted44-interrupted', 'weighted44-exact-resume'):
    root = AUTHOR / 'actual-cpu-subprocesses' / name
    execution = json.loads((root / 'execution.json').read_text())
    receipt = json.loads((root / 'receipt.json').read_text())
    assert all(receipt.get(key) == value for key, value in execution.items()), name
    assert len({item['path'] for item in receipt['files']}) == len(receipt['files']), name
    for item in receipt['files']:
        path = root / item['path']
        assert path.is_file() and not path.is_symlink(), str(path)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
    assert execution['cwd'] == str(REPO)
    assert execution['command'][:3] == ['/workspace/tiny-perceptron-vlm/.venv/bin/python', '-u', str(REPO / 'scripts/selftrained/train.py')]
    assert all(sha(REPO / relative) == digest for relative, digest in execution['code_sha256'].items())
    training_path = root / 'train-receipt.json'
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    metrics_path = root / 'metrics.jsonl'
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    new_steps = len([row for row in rows if row.get('event') == 'train'])
    out['attempts'].append({'name': name, 'execution_sha256': sha(root / 'execution.json'),
                            'receipt_sha256': sha(root / 'receipt.json'), 'status': execution['status'],
                            'returncode': execution['returncode'], 'interrupted': training.get('interrupted'),
                            'saved_step': training.get('steps'), 'new_train_metrics_rows': new_steps,
                            'effective_steps_target': execution['job']['steps'],
                            'effective_batch': execution['job']['batch_size'],
                            'effective_context': execution['job']['context'],
                            'all_output_files_sha256_verified': True, 'tracked_current_core_identity_verified': True})
out['actual_new_optimizer_steps_total'] = sum(item['new_train_metrics_rows'] for item in out['attempts'])
assert out['actual_new_optimizer_steps_total'] == 11
native = json.loads((AUTHOR / 'actual-cpu-subprocesses/native414-joint/train-receipt.json').read_text())
parent = AUTHOR / 'actual-cpu-subprocesses/weighted44-joint'
binding = native['origin']['new_joint_initialization']['source_integrity']
assert binding['source_checkpoint_sha256'] == sha(parent / 'best.pt')
assert binding['source_execution_sha256'] == sha(parent / 'execution.json')
assert binding['source_receipt_sha256'] == sha(parent / 'receipt.json')
assert binding['source_completed_steps'] == 1
out['actual_native_core_selected_parent_and_execution_receipt_binding_verified'] = True
target = BASE / (sys.argv[2] if len(sys.argv) > 2 else 'author-raw-evidence-inspection.json')
target.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'attempts': len(out['attempts']), 'actual_new_steps': out['actual_new_optimizer_steps_total'],
                  'inspection_sha256': sha(target)}))
