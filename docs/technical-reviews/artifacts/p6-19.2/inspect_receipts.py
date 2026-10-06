import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
rows = []
keys = []
for folder in sorted((ROOT / 'docs/selftrained/results/training-raw').iterdir()):
    if not folder.is_dir():
        continue
    documents = {}
    for relative in ['raw/execution.json', 'raw/train-receipt.json', 'receipt.json']:
        p = folder / relative
        raw = p.read_bytes(); document = json.loads(raw)
        keys.append({'path': str(p.relative_to(ROOT)), 'sha256': digest(raw), 'top_level_keys': {k: type(v).__name__ for k,v in document.items()}})
        documents[relative] = document
        if folder.name in ['moe-native', 'dense-weighted', 'moe-pretrain', 'dense-pretrain']:
            dst = ART/'raw-snapshots'/folder.name/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
            assert digest(dst.read_bytes()) == digest(raw)
    ex = documents['raw/execution.json']; tr = documents['raw/train-receipt.json']; outer = documents['receipt.json']
    assert ex['returncode'] == outer['returncode'] == 0
    assert ex['run_id'] == outer['run_id']
    assert ex['manifest_sha256'] == outer['manifest_sha256'] == '3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b'
    row = {'stage_key': folder.name, 'execution': {k: ex[k] for k in ['run_id','stage','revision','status','returncode','command','manifest_sha256']},
           'train': {k: tr[k] for k in ['stage','architecture','steps','tokens','target_tokens','train_records','validation_records','test_used_for_selection','seed','config','tokenizer_sha256','selection','total_parameters','trainable_parameters']},
           'raw_integrity': {relative: {'sha256': digest((folder/relative).read_bytes()), 'bytes': (folder/relative).stat().st_size} for relative in documents},
           'outer_file_table': outer['files']}
    for k in ['freeze_perception_backbones','sampling_mode','tool_loss_weight','numeric_run_loss_weight','native_voice_loss_weight','interrupted','completed_requested_steps','selected_checkpoint_available','inference_exported']:
        if k in tr: row['train'][k] = tr[k]
    row['train']['stage_history'] = [{k: h[k] for k in ['stage','step','tokens','checkpoint','checkpoint_sha256','selection'] if k in h} for h in tr['stage_history']]
    if 'init_checkpoint' in ex['job']:
        row['execution']['init_checkpoint'] = ex['job']['init_checkpoint']
    if 'origin' in tr and 'new_joint_initialization' in tr['origin']:
        nji = tr['origin']['new_joint_initialization']
        row['train']['new_joint_initialization'] = {k: nji[k] for k in ['source_stage','loaded_step','source_checkpoint_sha256','source_checkpoint_kind','source_completed_steps','reset_state'] if k in nji}
    expected_params = 5447107 if tr['architecture'] == 'moe' else 2288067
    assert tr['total_parameters'] == expected_params
    assert tr['config']['vocab_size'] == 550 and tr['config']['tied']
    rows.append(row)
assert len(rows) == 15
for relative in ['docs/selftrained/v2-manifest.json', 'docs/selftrained/results/v2-final-public-results.json', 'docs/selftrained/results/public-cpu-raw/source-core31-before.json']:
    raw = (ROOT/relative).read_bytes(); (ART/Path(relative).name).write_bytes(raw)
public = json.loads((ROOT/'docs/selftrained/results/v2-final-public-results.json').read_bytes())
public_checked = {'manifest_sha256': public['manifest_sha256'], 'hf_repo': public['hf_repo'], 'hf_revision': public['hf_revision'], 'architectures': {}}
for arch, stage in [('moe','moe-native'), ('dense','dense-weighted')]:
    r = public['architectures'][arch]
    public_checked['architectures'][arch] = {'capacity': {k: r['capacity'][k] for k in ['config','parameters','language_parameters','perception_parameters','expert_parameters','router_parameters','active_language_parameters_per_token']}, 'selected_source': r['selected_source']}
    rr = next(x for x in rows if x['stage_key'] == stage)
    assert r['selected_source']['run_id'] == rr['execution']['run_id']
    best = [f for f in rr['outer_file_table'] if f['path'] == 'best.pt']
    assert len(best) == 1 and best[0]['sha256'] == r['selected_source']['sha256']
before = json.loads((ROOT/'docs/selftrained/results/public-cpu-raw/source-core31-before.json').read_bytes())
source_checks = {}
for relative in ['tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/selftrained/dataset.py','tiny_perceptron/modern.py','tiny_perceptron/model.py']:
    raw = (ROOT/relative).read_bytes()
    assert before[relative] == digest(raw)
    dst = ART/'code'/relative; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(raw)
    source_checks[relative] = {'sha256': digest(raw), 'public_cpu_source_matches': True}
result = {'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), 'actual_read_pointers': {'execution': ['/run_id','/stage','/revision','/status','/returncode','/command','/manifest_sha256','/job/init_checkpoint'], 'train_receipt': ['/stage','/architecture','/steps','/tokens','/target_tokens','/train_records','/validation_records','/test_used_for_selection','/seed','/config','/tokenizer_sha256','/selection','/total_parameters','/trainable_parameters','/freeze_perception_backbones','/sampling_mode','/tool_loss_weight','/numeric_run_loss_weight','/native_voice_loss_weight','/interrupted','/completed_requested_steps','/selected_checkpoint_available','/inference_exported','/stage_history/*/{stage,step,tokens,checkpoint,checkpoint_sha256,selection}','/origin/new_joint_initialization/{source_stage,loaded_step,source_checkpoint_sha256,source_checkpoint_kind,source_completed_steps,reset_state}'], 'outer_receipt': ['/returncode','/run_id','/manifest_sha256','/files'], 'public': ['/manifest_sha256','/hf_repo','/hf_revision','/architectures/{moe,dense}/capacity/{config,parameters,language_parameters,perception_parameters,expert_parameters,router_parameters,active_language_parameters_per_token}','/architectures/{moe,dense}/selected_source']}, 'top_level_inspection': keys, 'stages': rows, 'public_configuration_and_selection': public_checked, 'source_code_identity': source_checks}
(ART/'receipts-checked.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print('GIT_HEAD',result['git_head'])
print('STAGES_VERIFIED',len(rows))
for r in rows:
    t=r['train']; print(json.dumps({'stage_key': r['stage_key'], 'steps': t['steps'], 'history_selected_steps': [h['step'] for h in t['stage_history']], 'tool':t.get('tool_loss_weight',1), 'numeric':t.get('numeric_run_loss_weight',1), 'native_voice':t.get('native_voice_loss_weight',1), 'parameters':t['total_parameters'], 'returncode':r['execution']['returncode']},ensure_ascii=False))
print('PUBLIC_BEST_SOURCE_SHA_MATCH', True)
print('PUBLIC_CPU_CODE_SHA_MATCH',json.dumps(source_checks,ensure_ascii=False))
