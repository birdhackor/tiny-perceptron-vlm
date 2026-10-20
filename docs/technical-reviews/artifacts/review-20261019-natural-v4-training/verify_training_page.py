"""離線核對資料、配方、數值與公開原始來源；不載入預訓練權重。"""

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as core


def read(path):
    return json.loads((ROOT / path).read_bytes())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch(url, version, selectors=None, json_keys=None):
    command = ['curl', '-fsSL', '--max-time', '45', url]
    process = subprocess.run(command, capture_output=True, check=True)
    raw = process.stdout
    text = raw.decode()
    lines = text.splitlines()
    excerpts = []
    for selector, following in selectors or []:
        start = next(index for index, line in enumerate(lines) if selector in line)
        excerpts.append({'locator': selector, 'text': '\n'.join(lines[start:start + following])})
    result = {
        'url': url, 'version': version, 'command': command,
        'accessed_at': datetime.now(UTC).isoformat(),
        'original_sha256': hashlib.sha256(raw).hexdigest(),
        'original_bytes': len(raw), 'excerpts': excerpts,
    }
    if json_keys:
        data = json.loads(raw)
        result['selected_fields'] = {key: data[key] for key in json_keys}
    return result


sources = [
    fetch('https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/cuda/memory.py', 'PyTorch v2.8.0',
          [('def max_memory_allocated(', 9), ('def memory_allocated(', 3), ('def reset_peak_memory_stats(', 7)]),
    fetch('https://raw.githubusercontent.com/huggingface/peft/v0.18.1/src/peft/tuners/lora/layer.py', 'PEFT v0.18.1',
          [('self.lora_A[adapter_name] = nn.Linear', 9)]),
    fetch('https://raw.githubusercontent.com/huggingface/peft/v0.18.1/src/peft/tuners/tuners_utils.py', 'PEFT v0.18.1',
          [('def _mark_only_adapters_as_trainable', 7)]),
    fetch('https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst', 'scikit-learn 1.7.2',
          [('When evaluating different settings', 12)]),
    fetch('https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json',
          '89644892e4d85e24eaac8bacfd4f463576704203', json_keys=['text_config', 'vision_config', 'tie_word_embeddings']),
    fetch('https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/generation_config.json',
          '89644892e4d85e24eaac8bacfd4f463576704203', json_keys=['do_sample', 'eos_token_id', 'repetition_penalty']),
    fetch('https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/generation/utils.py', 'Transformers v4.57.6',
          [('next_tokens = torch.argmax(next_token_scores', 2)]),
    fetch('https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py', 'Transformers v4.57.6',
          [('class Qwen3VLVisionPatchEmbed', 21), ('class Qwen3VLVisionPatchMerger', 19),
           ('class Qwen3VLVisionMLP', 10), ('class Qwen3VLVisionAttention', 14),
           ('class Qwen3VLVisionBlock', 12), ('class Qwen3VLTextAttention', 36),
           ('class Qwen3VLTextMLP', 14), ('class Qwen3VLTextDecoderLayer', 12),
           ('self.pos_embed = nn.Embedding', 2), ('self.deepstack_merger_list = nn.ModuleList', 4),
           ('self.embed_tokens = nn.Embedding', 5), ('_tied_weights_keys', 2)]),
]
with (ARTIFACTS / 'original-source-excerpts.json').open('x') as file:
    json.dump({'scope': '只保存與主張相關的原始來源摘錄，未保存第三方全文。', 'sources': sources}, file, ensure_ascii=False, indent=2)
    file.write('\n')

manifest_path = 'docs/natural-assistant/v4/manifest.json'
manifest = read(manifest_path)
train = [row for row in manifest['rows'] if row['split'] == 'train']
assert digest(manifest_path) == '0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60'
assert len(train) == 2077
visits = [core.training_row_at(train, index, 42)['id'] for index in range(4154)]
assert len(set(visits)) == 2077 and set(Counter(visits).values()) == {2}
scoring = load_module('review_scoring', 'scripts/score_natural_v4_validation.py')
protocol = read('docs/natural-assistant/v4/validation-protocol-lower-lr.json')
cases, audio = scoring.expected_cases(manifest, protocol)
groups = Counter(case['group'] for case in cases.values())
base_example = sum(map(Fraction, ['1/2', '1', '4/5', '1/3', '1'])) / 5
chat_example = (Fraction(1) + Fraction(3, 4) + Fraction(1)) / 3
adapter_example = (Fraction(3, 4) + 1 + Fraction(4, 5) + Fraction(1, 3) + chat_example) / 5
assert base_example == Fraction(109, 150) and adapter_example == Fraction(19, 25)

official_config = sources[4]['selected_fields']
text_config = official_config['text_config']
layers, hidden, head = (text_config[key] for key in ('num_hidden_layers', 'hidden_size', 'head_dim'))
q = text_config['num_attention_heads'] * head
v = text_config['num_key_value_heads'] * head
lora_count = layers * 8 * (hidden + q + hidden + v)
assert lora_count == 1605632 and layers * 2 * 2 == 112
text_per_layer = hidden * (q + 2*v + q) + 2*head + 3*hidden*text_config['intermediate_size'] + 2*hidden
text_parameters = text_config['vocab_size'] * hidden + layers * text_per_layer + hidden
vision = official_config['vision_config']
dim, intermediate = vision['hidden_size'], vision['intermediate_size']
patch = dim * vision['in_channels'] * vision['temporal_patch_size'] * vision['patch_size']**2 + dim
vision_block = 4*dim + (dim*3*dim+3*dim) + (dim*dim+dim) + (dim*intermediate+intermediate) + (intermediate*dim+dim)
merged = dim * vision['spatial_merge_size']**2
merger_linear = merged*merged+merged + merged*vision['out_hidden_size']+vision['out_hidden_size']
vision_parameters = patch + vision['num_position_embeddings']*dim + vision['depth']*vision_block + 2*dim+merger_linear + len(vision['deepstack_visual_indexes'])*(2*merged+merger_linear)
assert text_parameters + vision_parameters + lora_count == 2129137664

raw_dir = 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw'
scores = read('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json')
transcripts = read(raw_dir + '/transcripts.json')
transcript_by_id = {row['id']: row['transcript'] for row in transcripts}
actual_validation = {}
for variant in protocol['candidate_order']:
    path = raw_dir + '/generations-' + variant + '.json'
    records = read(path)
    complete = sum(scoring.eos_complete(row, 384) for row in records)
    assert len(records) == 132
    for row in records:
        if row['task'] == 'speech_chat':
            assert row['user'] == row['transcript'] == transcript_by_id[row['id']]
    counts = scores['variants'][variant]['correct_counts']
    primary = sum(counts[group] * weight for group, weight in scoring.WEIGHTS.items())
    assert primary == scores['variants'][variant]['primary_numerator']
    actual_validation[variant] = {'generations': len(records), 'eos_complete': complete,
                                  'noncomplete': len(records)-complete, 'file_sha256': digest(path),
                                  'recorded_grades_primary': str(Fraction(primary, scoring.DENOMINATOR)),
                                  'recorded_correct_counts': counts}

source = (ROOT / 'docs/course-review-20261019/freeze-01/sources/natural-v4-training.md').read_text()
commands = []
for block in __import__('re').findall(r'```bash\n(.*?)```', source, flags=__import__('re').S):
    if 'scripts/natural_assistant.py' not in block:
        continue
    import shlex
    for command in block.replace('\\\n', ' ').splitlines():
        if 'scripts/natural_assistant.py ' in command:
            tokens = shlex.split(command)
            cli = load_module('review_cli', 'scripts/natural_assistant.py')
            parsed = cli.parser().parse_args(tokens[tokens.index('scripts/natural_assistant.py')+1:])
            commands.append({'stage': parsed.stage, 'output': str(parsed.output), 'steps': parsed.steps,
                             'adapter': str(parsed.adapter), 'learning_rate': parsed.learning_rate,
                             'checkpoint_steps': parsed.checkpoint_steps, 'split': parsed.split,
                             'selected_only': parsed.selected_only, 'max_tokens': parsed.max_tokens,
                             'max_new_tokens': parsed.max_new_tokens})
pending = {str(n): ','.join(str(step) for step in (1039, 2077) if step > n) for n in (0, 1000, 1039, 1100, 2076)}
assert pending['1000'] == '1039,2077' and pending['1100'] == '2077'

run_command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/test_natural_assistant.py', 'tests/test_natural_selected_only.py',
               'tests/test_natural_v4_scoring.py', 'tests/test_natural_v4_data_download.py']
with tempfile.TemporaryDirectory(prefix='verification-', dir=ARTIFACTS) as temporary:
    process = subprocess.run(run_command + ['--basetemp', temporary], cwd=ROOT,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)

inspection_paths = [
    'tiny_perceptron/natural_assistant.py', 'scripts/natural_assistant.py', 'scripts/modal_natural.py',
    'scripts/fetch_natural_data.py', 'scripts/score_natural_v4_validation.py', 'tiny_perceptron/natural_ui.py',
    'requirements-natural.txt', '.github/workflows/natural-assistant.yml',
    'docs/natural-assistant/v4/STUDENT.md', 'docs/natural-assistant/v4/DATA.md',
    'docs/natural-assistant/v4/validation-protocol-lower-lr.json', 'docs/natural-assistant/v4/selection.json',
    'docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json',
    'docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json',
]
inspection = []
for path in inspection_paths:
    entry = {'path': path, 'sha256': digest(path)}
    if path.endswith('.py'):
        tree = ast.parse((ROOT / path).read_text())
        entry['functions'] = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    inspection.append(entry)
audit = read('docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json')
result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-natural-v4-training/verify_training_page.py',
    'environment': {'python': sys.version, 'platform': platform.platform(),
                    'torch': importlib.metadata.version('torch'), 'device': 'CPU；未執行 GPU 訓練',
                    'transformers': '未安裝；核對官方固定版本原始碼', 'peft': '未安裝；核對官方固定版本原始碼'},
    'source_sha256': digest('docs/course-review-20261019/freeze-01/sources/natural-v4-training.md'),
    'freeze_sha256': digest('docs/course-review-20261019/freeze-04.json'),
    'manifest_sha256': digest(manifest_path),
    'data': {'train_rows': len(train), 'row_visits': len(visits), 'visits_per_row': 2,
             'validation_groups': dict(groups), 'lm_generations': len(cases), 'audio_recordings': len(audio),
             'train_audio_rows': sum(row['split'] == 'train' for row in manifest['audio_rows'])},
    'arithmetic': {'base_example': str(base_example), 'base_percent': float(100*base_example),
                   'dialogue_example': str(chat_example), 'candidate_example': str(adapter_example),
                   'candidate_percent': float(100*adapter_example),
                   'core_minutes_from_recorded_summary': audit['resources']['training_elapsed_seconds']/60,
                   'outer_minutes_from_recorded_summary': audit['resources']['runner_subprocess_seconds']/60,
                   'allocated_gib_from_recorded_summary': audit['resources']['peak_cuda_memory_allocated_bytes']/2**30,
                   'adapter_mb': 6438952/10**6,
                   'lora_parameters': lora_count, 'lora_tensors': layers*4,
                   'text_parameters': text_parameters, 'vision_parameters': vision_parameters,
                   'total_with_lora': text_parameters+vision_parameters+lora_count,
                   'formula_scope': '依固定 config 與 Transformers 原始碼逐層計數；embedding 與 lm_head 共用，不重複計數。這不是實際更新證據。'},
    'validation_raw_recount': actual_validation, 'parsed_page_commands': commands,
    'resume_pending': pending,
    'pytest': {'command': run_command + ['--basetemp', '<artifact 目錄內，結束後移除的暫存子目錄>'],
               'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr,
               'scope': '既有離線測試，使用小型 fixture／mock；未下載權重，未證明 GPU 效能或完整續訓的數值等價。'},
    'inspected_repository_files': inspection,
    'training_raw_limit': '原始訓練 result、provenance、adapter 與 training_state 位於本單禁止讀取的 outputs。只讀允許的摘要、索引與驗證器原始碼，未把摘要冒充原始訓練結果。',
}
with (ARTIFACTS / 'execution.json').open('x') as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
    file.write('\n')
print(json.dumps({'pytest': result['pytest'], 'arithmetic': result['arithmetic'],
                  'data': result['data'], 'validation_raw_recount': actual_validation}, ensure_ascii=False, indent=2))
raise SystemExit(process.returncode)
