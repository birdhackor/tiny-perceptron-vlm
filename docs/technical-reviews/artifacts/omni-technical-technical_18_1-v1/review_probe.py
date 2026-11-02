"""18.1 的有界 CPU 核驗；不訓練、不更新任何權重。"""

import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.compression import _cache_text, _parameter_hash, _prompt
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import generate
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import load_checkpoint

CACHE = Path('/workspace/work/reference/omni-technical-technical_18_1-v1')
DEST = Path(__file__).with_name('execution-v1.json')
if DEST.exists():
    raise SystemExit('既有核驗原件不可覆寫')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(code):
    scope, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, '18.1-exact-embedded-block', 'exec'), scope)
    return stream.getvalue(), scope


torch.set_num_threads(1)
packet = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v10/technical_18_1.json').read_text())
snapshot = ROOT / packet['page']['snapshot']
body = snapshot.read_text()
current = re.search(r'^## 18\.1 .*?(?=^## |\Z)', (ROOT / 'course/chapters/18.md').read_text(), re.M | re.S).group()
assert current == body
assert sha(snapshot) == packet['page']['source_sha256']
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 1
stdout, scope = capture(blocks[0])
answers = [scope['candidates'][p.argmax().item()] for p in scope['probabilities']]
assert answers == ['4', '4']
sums = [p.sum().item() for p in scope['probabilities']]
assert all(abs(s - 1) < 1e-6 for s in sums)
exercise = blocks[0].replace('candidates = ["4", "3", "100"]', 'candidates = ["貓", "狗", "鳥"]').replace('[torch.tensor([0.7, 0.2, 0.1]), torch.tensor([0.4, 0.35, 0.25])]', '[torch.tensor([0.8, 0.15, 0.05]), torch.tensor([0.8, 0.05, 0.15])]')
exercise_stdout, exercise_scope = capture(exercise)
exercise_answers = [exercise_scope['candidates'][p.argmax().item()] for p in exercise_scope['probabilities']]
exercise_second = [exercise_scope['candidates'][p.argsort(descending=True)[1].item()] for p in exercise_scope['probabilities']]
assert exercise_answers == ['貓', '貓'] and exercise_second == ['狗', '鳥']

hard_gradients, soft_gradients = [], []
for target in [torch.tensor([0]), torch.tensor([0])]:
    logits = torch.zeros(1, 3, requires_grad=True)
    F.cross_entropy(logits, target).backward()
    hard_gradients.append(logits.grad.tolist())
for p in scope['probabilities']:
    logits = torch.zeros(1, 3, requires_grad=True)
    F.cross_entropy(logits, p[None]).backward()
    soft_gradients.append(logits.grad.tolist())
assert hard_gradients[0] == hard_gradients[1]
assert soft_gradients[0] != soft_gradients[1]

student = torch.zeros(1, 3, 3, requires_grad=True)
teacher = torch.tensor([[[0.7, 0.2, 0.1]] * 3]).log().requires_grad_()
labels = torch.tensor([[-100, 0, 1]])
kl = distillation_kl(student, teacher, labels, temperature=1)
manual = (teacher.detach().softmax(-1) * (teacher.detach().log_softmax(-1) - student.detach().log_softmax(-1))).sum(-1)[labels != IGNORE].mean()
kl.backward()
assert torch.allclose(kl.detach(), manual, atol=1e-7, rtol=0)
assert teacher.grad is None and torch.equal(student.grad[:, 0], torch.zeros(1, 3))

records = [
    ('sft-teacher.pt', 'attributes', 'color=green;shape=square;pitch=low;color?', 'green', 24),
    ('style-teacher.pt', 'style_transfer', 'style=concise; 1+2=?', '3', 96),
    ('style-teacher.pt', 'style_transfer', 'style=json; 1+2=?', '{"answer": 3}', 96),
]
original = json.loads((ROOT / 'docs/course-experiments/results/distillation.json').read_text())
release = json.loads((ROOT / 'docs/course-experiments/public-releases/distillation.json').read_text())['release']
files = {f['output']: f for f in release['public_manifest']['files']}
teacher_results, models = [], {}
for name in ['sft-teacher.pt', 'style-teacher.pt', 'moe-teacher.pt']:
    path = CACHE / name
    assert sha(path) == files[name]['sha256']
    model, payload = load_checkpoint(path, device='cpu')
    model.eval().requires_grad_(False)
    models[name] = model
    teacher_results.append({
        'name': name, 'path_outside_git': str(path), 'sha256': sha(path),
        'download_url': f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{files[name]['path']}",
        'class': type(model).__name__, 'format_version': payload['format_version'],
        'step': payload.get('step'), 'metadata': payload.get('metadata'),
        'config': asdict(model.config),
        'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
    })

generation_results = []
for name, task, question, expected, budget in records:
    model = models[name]
    record = {'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': expected}]}
    prefix = _prompt(record, model.config.max_length)
    before = _parameter_hash(model)
    with torch.no_grad():
        ids = generate(model, prefix[None], max_new_tokens=budget)[0, len(prefix):].tolist()
    result = generation_report(ByteTokenizer(), ids)
    audit = next(x for x in original['results']['tasks'][task]['hard_target_generation']['audit'] if x['question'] == question)
    assert result['answer'] == expected and ids == audit['teacher_ids']
    assert before == _parameter_hash(model)
    cache, _ = _cache_text(model, [record], 'cpu')
    _, y = render_chat(record['messages'])
    assert tuple(cache[0].shape) == (int((y != IGNORE).sum()), 264)
    generation_results.append({
        'checkpoint': name, 'question': question, 'expected': expected,
        'observed': result['answer'], 'generated_ids': ids, 'max_new_tokens': budget,
        'greedy': True, 'temperature': 0.0, 'ended_with_eos': ByteTokenizer().eos_id in ids,
        'matches_original_audit_ids': True, 'effective_targets': int((y != IGNORE).sum()),
        'cache_shape': list(cache[0].shape), 'cache_dtype': str(cache[0].dtype),
        'parameter_hash_unchanged': before == _parameter_hash(model),
    })

tok = ByteTokenizer()
assert len(tok.encode('100')) == 3
api_file = Path(inspect.getsourcefile(F.softmax))
docs_file = Path(torch.__file__).parent / '_torch_docs.py'
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'reviewer_task': '/root/omni_technical_coordinator/technical_18_1',
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_18_1-v1/review_probe.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'torch_threads': str(torch.get_num_threads())},
    'source_snapshot': {'path': packet['page']['snapshot'], 'sha256': sha(snapshot), 'current_section_equal': True},
    'embedded': {'code_sha256': hashlib.sha256(blocks[0].encode()).hexdigest(), 'stdout': stdout, 'shape_each': [list(p.shape) for p in scope['probabilities']], 'dtype': str(scope['probabilities'][0].dtype), 'answers': answers, 'probability_sums': sums},
    'exercise': {'stdout': exercise_stdout, 'answers': exercise_answers, 'second_choices': exercise_second},
    'gradient_control': {'initial_logits': [0, 0, 0], 'hard_targets': [0, 0], 'same_full_target_gradients': hard_gradients, 'soft_target_gradients': soft_gradients, 'not_a_training_quality_test': True},
    'kl_control': {'labels': labels.tolist(), 'input_shape': list(student.shape), 'kl_teacher_student': kl.item(), 'manual_forward_kl': manual.item(), 'absolute_error': abs(kl.item() - manual.item()), 'teacher_grad': None, 'student_grad': student.grad.tolist(), 'temperature': 1},
    'tokenizer': {'class': 'ByteTokenizer', 'vocab_size': tok.vocab_size, 'encode_100': tok.encode('100'), 'role': '三個答案候選不是本專案字表的三個token'},
    'downloaded_teachers': teacher_results,
    'generation_checks': generation_results,
    'official_pytorch_inspection': {'functional_source': str(api_file), 'functional_source_sha256': sha(api_file), 'docs_source': str(docs_file), 'docs_source_sha256': sha(docs_file), 'locators': {fn.__name__: inspect.getsourcelines(fn)[1] for fn in [F.softmax, F.cross_entropy, F.kl_div]}, 'argmax_short_quote': torch.argmax.__doc__.split('Args:')[0].strip(), 'softmax_short_quote': F.softmax.__doc__.split('See :class:')[0].strip(), 'cross_entropy_short_quote': F.cross_entropy.__doc__.split('reduce (bool')[0].strip()},
    'limits': '只執行教材人工例、三筆已發布教師的greedy CPU前向與小型KL控制；未執行新訓練、GPU或品質/速度比較。公開推論匯出檔SHA與原訓練檔SHA不同，不能混同。',
}
DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(DEST.relative_to(ROOT)), 'sha256': sha(DEST), 'generation_checks': generation_results, 'embedded_stdout': stdout, 'exercise_stdout': exercise_stdout, 'all_assertions_passed': True}, ensure_ascii=False))
