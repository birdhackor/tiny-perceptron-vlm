"""Freeze review closure and archive only small inputs, logs, and metadata."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE = Path('/tmp/p5-local-wrapper-independent')
AUTHOR = Path('/tmp/p5-local-stage-candidate-final')
ROOT_SMOKE = Path('/tmp/p5-root-applied-local-pretrain-smoke')
REPO = Path('/workspace/selftrained-v2')
ARCHIVE = REPO / 'docs/selftrained/infrastructure/v2-local-stage/independent-review'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

negative = json.loads((AUTHOR / 'negative-subprocess-proof.json').read_text())
for item in negative['cases']:
    assert sha(AUTHOR / (item['name'] + '-subprocess.log')) == item['log_sha256']
    output = Path(item['command'][item['command'].index('--output-dir') + 1])
    assert not output.exists() and item['no_attempt_output'] is True

root_execution = json.loads((ROOT_SMOKE / 'execution.json').read_text())
root_receipt = json.loads((ROOT_SMOKE / 'receipt.json').read_text())
root_training = json.loads((ROOT_SMOKE / 'train-receipt.json').read_text())
assert all(root_receipt.get(key) == value for key, value in root_execution.items())
for item in root_receipt['files']:
    path = ROOT_SMOKE / item['path']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
assert root_execution['revision'].startswith('0e79975a')
assert root_execution['wrapper_sha256'] == '675e9f63f264365d59a5104811d6899354b412f46b876b8128ed63e40dff2561'
assert root_execution['code_sha256']['scripts/selftrained/train_local_stage.py'] == root_execution['wrapper_sha256']
assert root_execution['status'] == 'completed' and root_execution['returncode'] == 0
assert root_training['steps'] == 1 and root_training['completed_requested_steps'] is True
assert root_execution['command'][2] == str(REPO / 'scripts/selftrained/train.py')
assert sha(REPO / 'scripts/selftrained/train_local_stage.py') == root_execution['wrapper_sha256']
assert sha(REPO / 'docs/selftrained/TRAINING.md') == sha(AUTHOR / 'docs/selftrained/TRAINING.md')

closure = {
    'conclusion': 'Accept final wrapper after three independently reproduced initial defects were fixed; no remaining blocker in reviewed scope.',
    'scope': 'Synthetic CPU contract review; no production training, original test records, production private checkpoints, paid dispatch, GPU, Modal, HF writes, or reviewer commit.',
    'initial_wrapper_sha256': sha(BASE / 'frozen/train_local_stage.py'),
    'initial_patch_sha256': sha(BASE / 'frozen/candidate.patch'),
    'final_wrapper_sha256': sha(BASE / 'frozen-final/train_local_stage.py'),
    'final_author_patch_sha256': sha(AUTHOR / 'candidate.patch'),
    'final_author_proof_sha256': sha(AUTHOR / 'proof.json'),
    'final_author_negative_proof_sha256': sha(AUTHOR / 'negative-subprocess-proof.json'),
    'final_author_raw_inspection_sha256': sha(BASE / 'final-author-raw-evidence-inspection.json'),
    'initial_author_raw_inspection_sha256': sha(BASE / 'author-raw-evidence-inspection.json'),
    'final_independent_actual_results_sha256': sha(BASE / 'final-independent-actual-results.json'),
    'initial_independent_actual_results_sha256': sha(BASE / 'independent-actual-results.json'),
    'final_independent_cli_cases': 9,
    'final_independent_new_optimizer_steps': 3,
    'final_author_genuine_cpu_check_count': 37,
    'final_author_new_optimizer_steps': 11,
    'final_author_additional_negative_cli_cases_verified': len(negative['cases']),
    'final_native_actual_source_best_execution_receipt_sha_binding_verified': True,
    'operation_doc': {'sha256': sha(AUTHOR / 'docs/selftrained/TRAINING.md'),
                      'review': 'Read-only consistency review: frozen manifest package/digests/counts, stage job solver values and selected index values match referenced repository metadata; explicit full args, own best/init, latest-only resume, fresh attempt, public-safe inference boundary and CPU scope match the reviewed wrapper. Installation/download commands were not rerun.'},
    'root_applied_file_mode_smoke': {'original_path': str(ROOT_SMOKE), 'revision': root_execution['revision'],
                                     'wrapper_sha256': root_execution['wrapper_sha256'], 'status': root_execution['status'],
                                     'actual_child_returncode': root_execution['returncode'], 'pretrain_steps': root_training['steps'],
                                     'new_committed_wrapper_present_in_source_hashes': True, 'all_raw_output_hashes_verified': True,
                                     'no_repo_root_override': True, 'independent_rerun': False},
    'archive_policy': 'Preserve exact original SHA for small synthetic inputs, scripts, patches, raw logs and JSON metadata. Checkpoint .pt and safetensors weights remain in /tmp and are excluded from ordinary Git archive. Receipts retain their verified original weight digests; archive is not a complete training checkpoint directory.',
}
write(BASE / 'final-author-closure.json', closure)
shutil.copyfile(AUTHOR / 'candidate.patch', BASE / 'frozen-final/candidate.patch')
review = BASE / 'review-final.md'
review.write_text(review.read_text().replace(
    '最终作者 patch/proof 的冻结哈希及最终六个 joint/raw attempts 检查另附于 `final-author-closure.json`（待作者冻结文件到齐后生成）。',
    '最终作者 patch SHA256 `e7b6e66ebdb948340beac9a378db63d7f1fdfc3b19e51619083bb1943d09a00d`、proof SHA256 `d48f8caa05c6b611d2170acda0070ac6f54e6a7c27776aa417f1902ad9c71c82`，以及最终六个 joint/raw attempts 的文件哈希和 native 来源绑定已核验，见 `final-author-closure.json` 与 `final-author-raw-evidence-inspection.json`。作者另七个真实 CLI 拒绝日志的 SHA 和无 output 状态均核验通过。操作文档 SHA `cc27b1a3d12fbc840b47e65876192396716f4fb6877c82cd166e63a442a6ab92` 的 manifest/job/selected index 数值与本机入口合同一致；安装与下载指令没有重跑。Root 后续已提交并从真实 repo 文件路径、无 --repo-root 参数执行自己合成 pretrain1，原始 execution 确认 revision `0e79975a`、script SHA `675e9f63...` 且新的 committed wrapper 被 source hashes 记录，所有输出哈希核验通过；本 reviewer 没有重跑该 smoke 或提交 commit。'))

ARCHIVE.mkdir(parents=True, exist_ok=False)
copies = []
def copy(source, relative):
    target = ARCHIVE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    original_sha = sha(source)
    shutil.copyfile(source, target)
    assert sha(target) == original_sha
    copies.append({'path': relative, 'bytes': target.stat().st_size, 'original_path': str(source),
                   'original_sha256': original_sha, 'archived_sha256': sha(target)})

for source in BASE.iterdir():
    if source.is_file() and source.suffix in ('.py', '.md', '.json', '.log'):
        copy(source, source.name)
for directory in ('data', 'frozen', 'frozen-final'):
    for source in sorted((BASE / directory).rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            copy(source, source.relative_to(BASE).as_posix())
metadata_names = ('execution.json', 'receipt.json', 'train-receipt.json', 'runner.log', 'metrics.jsonl',
                  'inference-manifest.json', 'model-config.json', 'input-model-config.json', 'tokenizer.json')
for directory in ('runs', 'final-runs'):
    for run in sorted((BASE / directory).iterdir()):
        for name in metadata_names:
            source = run / name
            if source.is_file():
                copy(source, source.relative_to(BASE).as_posix())
for label, author in (('author-initial', Path('/tmp/p5-local-stage-candidate')), ('author-final', AUTHOR)):
    for name in ('proof.json', 'proof-final.json', 'negative-subprocess-proof.json', 'verify_actual_subprocess.py', 'verify_negative_subprocess.py'):
        source = author / name
        if source.is_file():
            copy(source, label + '/' + name)
    for source in sorted(author.glob('*-subprocess.log')):
        copy(source, label + '/' + source.name)
    for run in sorted((author / 'actual-cpu-subprocesses').iterdir()):
        for name in metadata_names:
            source = run / name
            if source.is_file():
                copy(source, label + '/actual-cpu-subprocesses/' + run.name + '/' + name)
for name in metadata_names:
    source = ROOT_SMOKE / name
    if source.is_file():
        copy(source, 'root-applied-file-mode-smoke/' + name)
write(ARCHIVE / 'archive-index.json', {'scope': closure['scope'], 'policy': closure['archive_policy'], 'files': copies,
                                      'file_count': len(copies), 'total_bytes': sum(item['bytes'] for item in copies)})
print(json.dumps({'archive': str(ARCHIVE), 'file_count': len(copies), 'total_bytes': sum(item['bytes'] for item in copies),
                  'archive_index_sha256': sha(ARCHIVE / 'archive-index.json'),
                  'final_author_closure_sha256': sha(BASE / 'final-author-closure.json'),
                  'final_review_sha256': sha(BASE / 'review-final.md')}))
