"""補核不同字表的文字介面及既有 MoE 教師；不訓練。"""
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.compression import _parameter_hash
from tiny_perceptron.data import ByteTokenizer, CharTokenizer
from tiny_perceptron.model import generate
from tiny_perceptron.training import load_checkpoint

DEST = Path(__file__).with_name('additional-execution-v1.json')
assert not DEST.exists()
torch.set_num_threads(1)
text = '貓'
teacher_tokenizer = ByteTokenizer()
student_tokenizer = CharTokenizer('貓狗鳥')
teacher_ids = teacher_tokenizer.encode(text)
student_ids = student_tokenizer.encode(text)
assert teacher_tokenizer.decode(teacher_ids) == text == student_tokenizer.decode(student_ids)
assert len(teacher_ids) == 3 and len(student_ids) == 1
assert teacher_tokenizer.vocab_size != student_tokenizer.vocab_size

path = Path('/workspace/work/reference/omni-technical-technical_18_1-v1/moe-teacher.pt')
model, _ = load_checkpoint(path, device='cpu')
model.eval().requires_grad_(False)
assert model.config.experts == 4 and model.config.top_k == 2
original = json.loads((ROOT / 'docs/course-experiments/results/distillation.json').read_text())
sample = original['results']['tasks']['moe_to_dense']['teacher_test']['generated_samples'][0]
prefix = [teacher_tokenizer.bos_id] + teacher_tokenizer.encode(sample['question'])
before = _parameter_hash(model)
with torch.no_grad():
    ids = generate(model, torch.tensor([prefix]), max_new_tokens=24)[0, len(prefix):].tolist()
assert ids == sample['generated_ids']
assert before == _parameter_hash(model)
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_18_1-v1/additional_probe.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'torch_threads': '1'},
    'retokenization': {'text': text, 'teacher_ids': teacher_ids, 'student_ids': student_ids, 'teacher_vocab_size': teacher_tokenizer.vocab_size, 'student_vocab_size': student_tokenizer.vocab_size, 'same_decoded_text': True, 'scope': '文字可重新切詞；沒有對不同字表逐欄比較分布。'},
    'moe_teacher': {'checkpoint_path_outside_git': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'question': sample['question'], 'generated_ids': ids, 'generated_text': teacher_tokenizer.decode(ids), 'original_audit_ids_match': True, 'experts': model.config.experts, 'top_k': model.config.top_k, 'parameter_hash_unchanged': before == _parameter_hash(model), 'scope': '既有故事續寫任務的一筆 CPU greedy 前向；不評為故事品質成功。'},
}
DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(DEST.relative_to(ROOT)), 'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(), 'all_assertions_passed': True, 'retokenization': result['retokenization'], 'moe_teacher': result['moe_teacher']}, ensure_ascii=False))
