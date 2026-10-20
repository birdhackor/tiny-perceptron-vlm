"""核對保存的訓練鏈，並執行小型確定性目標與推論檢查；不啟動訓練。"""

import ast
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.selftrained import train
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    inputs = {}

    def read(relative):
        path = ROOT / relative
        inputs[relative] = sha(path)
        return json.loads(path.read_text())

    index = read('docs/selftrained/v2-training-stage-index.json')
    baseline = [row for row in index['stages'] if row['stage_key'].startswith('moe-')]
    weighted = read('docs/selftrained/results/v2-moe-weighted-10000-training-stage-index.json')
    native = read('docs/selftrained/results/v2-moe-native-balanced-4000-training-stage-index.json')
    selection = read('docs/selftrained/results/v2-final-training-source-selection.json')['moe_joint']
    public = read('docs/selftrained/results/v2-final-public-results.json')['architectures']['moe']
    names = [row['stage_key'] for row in baseline] + ['moe-weighted', 'moe-native']
    rows = baseline + [weighted, native]
    chain = []
    previous_best = None
    history_expected = []
    for name, row in zip(names, rows):
        prefix = 'docs/selftrained/results/training-raw/' + name + '/'
        outer = read(prefix + 'receipt.json')
        receipt = read(prefix + 'raw/train-receipt.json')
        execution = read(prefix + 'raw/execution.json')
        table = {item['path']: item for item in outer['files']}
        for basename in ['train-receipt.json', 'execution.json']:
            assert table[basename]['sha256'] == inputs[prefix + 'raw/' + basename], '原始收據指紋不相符'
        assert execution['returncode'] == 0 and execution['status'] == 'completed', '訓練未完成'
        assert receipt['steps'] == row['completed_steps'] and receipt['completed_requested_steps'], '完成步數不相符'
        assert receipt['stage_history'] == history_expected, '訓練歷史沒有接續前段'
        command = execution['command']
        parent = command[command.index('--init-checkpoint') + 1] if '--init-checkpoint' in command else None
        if previous_best:
            assert parent is not None, '缺少父檢查點'
            assert receipt['stage_history'][-1]['checkpoint_sha256'] == previous_best, '父檢查點指紋不相符'
        else:
            assert parent is None, '起點應為隨機初始化'
        best = table['best.pt']['sha256']
        assert row['selected_private_best_receipt']['sha256'] == best, '選定檢查點指紋不相符'
        chain.append({'stage': name, 'completed': receipt['steps'], 'selected': row['selected_step'],
                      'parent': parent, 'best_sha256': best, 'train_records': receipt['train_records'],
                      'validation_records': receipt['validation_records'],
                      'weights': {k: receipt.get(k, 1.0) for k in
                                  ['tool_loss_weight', 'numeric_run_loss_weight', 'native_voice_loss_weight']}})
        if name != 'moe-native':
            next_name = names[names.index(name) + 1]
            next_receipt = read('docs/selftrained/results/training-raw/' + next_name + '/raw/train-receipt.json')
            history_expected = next_receipt['stage_history']
            assert len(history_expected) == len(receipt['stage_history']) + 1, '歷史長度未增加一段'
            assert history_expected[:-1] == receipt['stage_history'], '舊歷史遭改寫'
            assert history_expected[-1]['step'] == row['selected_step'], '承接步數不相符'
        previous_best = best
    assert native['selected_step'] == selection['selected_step'] == 1000, '公開選定步數不同'
    assert previous_best == selection['sha256'] == public['selected_source']['sha256'], '公開來源不相符'
    assert native['attempt'] == selection['source_training_run'] == public['selected_source']['run_id']
    native_receipt = read('docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json')
    config = SelftrainedConfig(**native_receipt['config'])
    model = LimitedAssistant(config)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == native_receipt['total_parameters'] == public['capacity']['parameters'] == 5447107
    parameter_result = {'count': count, 'millions': count / 1e6, 'rounded_two_decimals': round(count / 1e6, 2),
                        'mtp_named_modules': [name for name, _ in model.named_modules() if 'mtp' in name.lower()]}
    del model

    tokenizer = CharacterTokenizer.build(['App 問題請用兩點回答先重開再更新。12x'])
    record = {'id': 'example', 'task': 'text_qa', 'split': 'train',
              'messages': [{'role': 'user', 'content': 'App 問題請用兩點回答'},
                           {'role': 'assistant', 'content': '先重開再更新。'}]}
    encoder = RecordEncoder(tokenizer, ROOT, context=128)
    pretrain = encoder.encode(record, pretrain=True)
    sft = encoder.encode(record)
    for position, label in enumerate(pretrain['labels'][:-1]):
        assert label == pretrain['input_ids'][position + 1], '預訓練目標未對齊下一 token'
    kept = [label for label in sft['labels'] if label != -100]
    assert kept == tokenizer.encode('先重開再更新。') + [tokenizer.eos_id], 'SFT 有效目標不符'

    labels = torch.tensor([tokenizer.encode('12x') + [-100]] * 3)
    records = [{'task': 'tool_call', 'split': 'train'}, {'task': 'text_qa', 'split': 'train'},
               {'task': 'voice_qa', 'split': 'train'}]
    logits = torch.randn(3, 4, tokenizer.vocab_size, requires_grad=True)
    ce = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1),
                         ignore_index=-100, reduction='none').reshape_as(labels)
    cases = []
    for weights in [(4, 4, 1), (4, 1, 4)]:
        actual_weights = train.language_loss_weights(labels, records, tokenizer, *weights)
        expected_rows = [16, 4, 4] if weights == (4, 4, 1) else [4, 1, 4]
        expected_weights = torch.tensor([[value, value, value, 0] for value in expected_rows], dtype=torch.float32)
        assert torch.equal(actual_weights, expected_weights), '有效目標權重不符'
        expected = (ce * expected_weights).sum() / expected_weights.sum()
        actual = train.weighted_language_loss({'logits': logits}, labels, records, tokenizer, *weights)
        assert torch.equal(actual, expected), '加權交叉熵公式不符'
        cases.append({'coefficients': list(weights), 'token_weights': actual_weights.tolist(),
                      'supervised_tokens': int((labels != -100).sum()), 'weight_mass': float(actual_weights.sum()),
                      'expected_loss': float(expected.detach()), 'actual_loss': float(actual.detach())})

    small_config = SelftrainedConfig(vocab_size=tokenizer.vocab_size, width=16, layers=2,
                                    heads=2, kv_heads=1, ffn_hidden=32, experts=2, top_k=1,
                                    max_length=64, backend='manual')
    small = LimitedAssistant(small_config).eval()
    state = {name: value.clone() for name, value in small.state_dict().items()}
    child = LimitedAssistant(small_config).eval()
    child.load_state_dict(state)
    assert all(torch.equal(value, child.state_dict()[name]) for name, value in state.items()), '載入後權重不同'
    ids = torch.tensor([tokenizer.encode('App 問題')])
    with torch.no_grad():
        full = small(ids)['logits']
        cache = None
        partial = []
        for position in range(ids.shape[1]):
            result = small(ids[:, position:position + 1], cache=cache)
            cache = result['cache']
            partial.append(result['logits'])
        cached = torch.cat(partial, dim=1)
        for block in child.lm.blocks:
            block.attention.backend = 'sdpa'
        fused = child(ids)['logits']
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(full, fused, atol=1e-5, rtol=1e-5)
    backend_result = {'configuration': vars(small_config), 'input_ids': ids.tolist(),
                      'dtype': str(full.dtype), 'logits_shape': list(full.shape),
                      'cache_max_abs_difference': float((full - cached).abs().max()),
                      'sdpa_max_abs_difference': float((full - fused).abs().max()),
                      'cache_argmax_equal': torch.equal(full.argmax(-1), cached.argmax(-1)),
                      'sdpa_argmax_equal': torch.equal(full.argmax(-1), fused.argmax(-1)),
                      'tolerance': 'atol=1e-5, rtol=1e-5', 'speed_measured': False}
    assert backend_result['cache_argmax_equal'] and backend_result['sdpa_argmax_equal']

    code_paths = ['scripts/selftrained/train.py', 'tiny_perceptron/selftrained/model.py',
                  'tiny_perceptron/selftrained/dataset.py', 'tiny_perceptron/selftrained/tokenizer.py',
                  'tiny_perceptron/attention.py', 'tiny_perceptron/model.py']
    for relative in code_paths:
        inputs[relative] = sha(ROOT / relative)
    objective_tree = ast.parse(inspect.getsource(train.objective))
    calls = sorted({ast.unparse(node.func) for node in ast.walk(objective_tree) if isinstance(node, ast.Call)})
    assert not any('ppo' in name.lower() or 'dpo' in name.lower() for name in calls)
    output = {'command': 'PYTHONDONTWRITEBYTECODE=1 uv run --frozen --no-sync python docs/technical-reviews/artifacts/review-20261019-19-8/verify_claims.py',
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                              'platform': platform.platform(), 'device': 'cpu'},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'input_sha256': inputs, 'chain': chain,
              'public_selection': {'step': selection['selected_step'], 'completed': native['completed_steps'],
                                   'best_sha256': previous_best, 'teacher_forced_selection': native_receipt['selection']},
              'parameters': parameter_result,
              'label_alignment': {'pretrain_valid_targets': sum(x != -100 for x in pretrain['labels']),
                                  'sft_valid_targets': len(kept), 'sft_target_ids': kept,
                                  'pretrain_shift': 1, 'sft_shift': 1},
              'weighted_cross_entropy': cases, 'objective_calls': calls,
              'loaded_state_identical': True, 'inference_equivalence': backend_result,
              'public_final_test': {'count': public['final_test']['count'],
                                    'evaluation_complete': public['final_test']['evaluation_complete'],
                                    'capability_acceptance_passed': public['capability_acceptance_passed']},
              'result': '所有斷言通過；只支持收據一致性與當前 CPU 小模型行為，未重訓公開模型或測量加速。'}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
