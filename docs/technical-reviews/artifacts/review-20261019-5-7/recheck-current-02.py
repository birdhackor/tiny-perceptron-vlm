import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
from scripts import train
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import save_checkpoint


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv, expected=0):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
    assert result.returncode == expected, result.stderr.decode()
    return {'argv': argv, 'exit_code': result.returncode, 'stdout': result.stdout.decode(), 'stderr': result.stderr.decode()}


def main():
    torch.set_num_threads(1)
    previous_path = 'docs/course-review-20261019/reviews/rv1019_technical_5-7/technical-5-7-current-01.json'
    previous_raw = (ROOT / previous_path).read_bytes()
    assert sha(previous_raw) == '3f9294408648a49387aad8a651016e7adda781f2d58894273827b87e7ee26569'
    previous = json.loads(previous_raw)
    hash_checks = []
    for entry in previous['sources'] + previous['artifacts']:
        if 'path' not in entry:
            continue
        current = sha((ROOT / entry['path']).read_bytes())
        unchanged = current == entry['sha256']
        assert unchanged or entry['id'] == 's_recipe'
        hash_checks.append({'id': entry['id'], 'path': entry['path'], 'previous_sha256': entry['sha256'], 'current_sha256': current, 'unchanged': unchanged})
    frozen = (ROOT / 'docs/course-review-20261019/freeze-01/sources/5.7.md').read_bytes()
    chapter = (ROOT / 'course/chapters/05.md').read_bytes()
    actual_section = re.search(rb'^## 5\.7 [^\n]*\n.*?(?=^## |\Z)', chapter, re.M | re.S).group()
    assert actual_section == frozen and sha(frozen) == previous['source_sha256']
    freeze = (ROOT / 'docs/course-review-20261019/freeze-07.json').read_bytes()
    assert sha(freeze) == '67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753'
    diff = command(['git', 'diff', 'ecf6dec71280', 'c6b9339d2595', '--', 'course/training.md'])
    before = command(['git', 'show', 'ecf6dec71280:course/training.md'])['stdout'].encode()
    after = command(['git', 'show', 'c6b9339d2595:course/training.md'])['stdout'].encode()
    assert sha(before) == 'd59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94'
    assert sha(after) == '1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b'
    assert after == (ROOT / 'course/training.md').read_bytes()
    def recipe(raw):
        text = raw.decode()
        start = text.index('<details id="fixed-sft-recipe"')
        return text[start:text.index('</details>', start) + len('</details>')]
    assert recipe(before) == recipe(after)
    run = ROOT / 'outputs/reviewer-tools/runs/20261020T152147866817Z'
    files = {name: {'sha256': sha((run / name).read_bytes()), 'content': (run / name).read_text()} for name in ['extraction.json', 'execution.json', 'environment.json', 'stdout.txt', 'stderr.txt', 'fence-1.py', 'bootstrap.py']}
    execution = json.loads(files['execution.json']['content'])
    assert execution['exit_code'] == 0 and execution['source_sha256'] == sha(frozen)
    assert json.loads(files['extraction.json']['content'])['svg_references'] == []
    sampling = []
    class CaptureModel:
        def __call__(self, ids, labels, **inputs):
            return {'logits': torch.zeros(1, len(labels), 264), 'labels': labels[None], 'auxiliary': torch.tensor(0.0)}
    for task in ['vision', 'audio', 'joint']:
        with patch('scripts.train.random.choice', wraps=train.random.choice) as choice:
            loss = train.modal_loss(CaptureModel(), task, ByteTokenizer(), 'cpu')
            assert choice.call_count == 3
            sampling.append({'task': task, 'random_choice_calls': choice.call_count, 'candidate_sets': [list(x.args[0]) for x in choice.call_args_list], 'loss_finite': bool(torch.isfinite(loss)), 'scope': '只驗證 modal_loss 每次動態抽題；CaptureModel 不訓練或量測能力'})
    with tempfile.TemporaryDirectory(dir=BASE) as directory:
        work = Path(directory)
        checkpoint = work / 'multimodal.pt'
        save_checkpoint(checkpoint, MultiModalLM(TinyLM(ModelConfig(width=8))))
        rejection = command([sys.executable, 'scripts/evaluate.py', str(checkpoint), '--device', 'cpu', '--output', str(work / 'evaluation.json')], expected=1)
        assert '此入口只評估 TinyLM 文字 checkpoint' in rejection['stderr']
    vqa_path = 'docs/course-experiments/results/vqa.json'
    vqa_raw = (ROOT / vqa_path).read_bytes()
    vqa = json.loads(vqa_raw)
    modal_path = 'scripts/course_experiments/modalities.py'
    modal_raw = (ROOT / modal_path).read_bytes()
    assert sha(modal_raw) == vqa['code_sha256'][modal_path]
    paired = []
    for name, variant in vqa['results']['variants'].items():
        if 'text_before' not in variant:
            continue
        before_samples = variant['text_before']['samples']
        after_samples = variant['text_after']['samples']
        questions_before = [{'messages': x['messages'], 'expected': x['expected']} for x in before_samples]
        questions_after = [{'messages': x['messages'], 'expected': x['expected']} for x in after_samples]
        assert questions_before == questions_after
        assert len(questions_before) == vqa['results']['text_data']['test']['count']
        paired.append({'variant': name, 'before_examples': len(before_samples), 'after_examples': len(after_samples), 'same_questions_and_targets': True, 'text_probe_is_training_diagnostic': variant['text_probe_is_training_diagnostic']})
    assert len(paired) == 4
    old_gpu = json.loads((BASE / 'verify_checkpoint-output.json').read_bytes())['gpu_record_audit']
    assert sha((ROOT / old_gpu['path']).read_bytes()) == old_gpu['sha256']
    text = modal_raw.decode()
    probe_source = text[text.index('def run_vqa(ctx):'):text.index('\ndef run_vision_ablation', text.index('def run_vqa(ctx):'))]
    changes = [{'statement': '這三條未指定 --data 的模態路線每步動態產生訓練例子，沒有固定驗證集；三題只算抽查。', 'status': 'verified', 'evidence': 'train.modal_loss 每次呼叫 random.choice；main 每步每例呼叫 modal_loss，報告與存檔流程沒有固定驗證迴圈。本次三種 task 的函數探針各捕捉三次隨機選取。', 'scope': '僅指頁面列出的 vision/audio/joint 命令；不外推到指定資料檔或固定課程實驗。'}, {'statement': 'evaluate.py 直接讀入完整多模態 checkpoint 會拒絕；頁面三個 infer_modal 命令沒有做文字能力前後評估。', 'status': 'verified', 'evidence': 'evaluate.main 的 TinyLM 型別檢查，本次實際製作小型多模態檢查點並執行入口，得到預期 ValueError；頁面命令只有模態推論。'}, {'statement': 'vqa 固定實驗用同一批文字測試題比較圖文訓練前後的文字核心。', 'status': 'verified', 'evidence': 'run_vqa 對同一 text_holdout 先後 evaluate_lm(model.language)；實驗紀錄四個對照分支各有前後10題，逐題 messages 與 expected 完全相同，當時 modalities.py 全檔SHA與現行相同。', 'scope': '核對已有實驗設計與紀錄，不重跑 GPU，也不把固定 vqa 結果套到本機 CLI 配方。'}]
    result = {'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-7/recheck-current-02.py', 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'mps_available': str(torch.backends.mps.is_available())}, 'previous_report': {'path': previous_path, 'sha256': sha(previous_raw)}, 'previous_issues': previous['issues'], 'resolved_issue_ids': [], 'hash_checks': hash_checks, 'section_sha256': sha(frozen), 'section_unchanged': True, 'freeze_sha256': sha(freeze), 'requested_git_diff': diff, 'recipe_unchanged': True, 'recipe_excerpt': recipe(after), 'section_execution': {'invoked_command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.7" --execute', 'files': files}, 'sampling_probes': sampling, 'multimodal_evaluation_rejection': rejection, 'vqa_record_check': {'path': vqa_path, 'sha256': sha(vqa_raw), 'code_path': modal_path, 'code_sha256': sha(modal_raw), 'code_matches_record': True, 'paired_comparisons': paired, 'text_data': vqa['results']['text_data'], 'source_excerpt': probe_source}, 'source_excerpts': {'scripts/train.py': {'sha256': sha((ROOT / 'scripts/train.py').read_bytes()), 'modal_loss': inspect.getsource(train.modal_loss), 'loop': inspect.getsource(train.main).split('for step in range(start, start + steps):', 1)[1].split('else:\n            selected', 1)[0]}, 'scripts/evaluate.py': {'sha256': sha((ROOT / 'scripts/evaluate.py').read_bytes()), 'type_guard': (ROOT / 'scripts/evaluate.py').read_text().split('if not isinstance(model, TinyLM):', 1)[1].split('tok = load_tokenizer', 1)[0]}}, 'auxiliary_change_reviews': changes, 'claims_reviewed': [{'claim_id': x['id'], 'status': 'verified', 'basis': '原文與實作未變，舊證據雜湊核對一致；本次重跑原始範例。' if x['id'] != 'separate_experiments_same_command' else '被引用的固定配方完整區塊不變，入口實作與既有GPU結果檔雜湊未變。'} for x in previous['claims']], 'gpu_result_file_unchanged': True, 'result': '全部核對通過'}
    path = BASE / 'recheck-current-02-output.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes()), 'result': result['result']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
