"""6.6 technical review: exact lesson cells and bounded, offline CPU diagnostics."""

import ast
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tokenizers
import torch
from tokenizers import Tokenizer

from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone
from tiny_perceptron.data import SPECIALS, ByteTokenizer, CharTokenizer, render_chat
from tiny_perceptron.model import ModelConfig
from tiny_perceptron.tokenization import ByteLevelBPE

torch.set_num_threads(1)
RAW = Path('/workspace/work/reference/omni-technical-technical_6_6-v1')
RAW.mkdir(parents=True, exist_ok=True)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(path, lesson_id):
    raw = path.read_text(encoding='utf-8')
    match = re.search(r'^## ' + re.escape(lesson_id) + r' .+$', raw, re.M)
    tail = re.search(r'^## ', raw[match.end():], re.M)
    return raw[match.start():match.end() + tail.start() if tail else len(raw)]


freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json').read_bytes())
page = next(p for p in freeze['pages'] if p['page_id'] == '6.6')
lesson = (ROOT / page['snapshot']).read_bytes()
assert sha(lesson) == page['source_sha256']
assert section(ROOT / page['source'], '6.6').encode() == lesson
blocks = re.findall(r'```python\n(.*?)\n```', lesson.decode(), re.S)
assert len(blocks) == 3
ns = {}
cells = []
for index, block in enumerate(blocks, 1):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(block, f"{page['snapshot']}:cell-{index}", 'exec'), ns)
    cells.append({'cell': index, 'sha256': sha(block.encode()), 'stdout': output.getvalue()})

inline_class = ast.parse(blocks[1]).body[0]
library_class = ast.parse(inspect.getsource(ByteTokenizer)).body[0]
assert ast.dump(inline_class, include_attributes=False) == ast.dump(library_class, include_attributes=False)
inline = ns['ByteTokenizer']()
library = ByteTokenizer()
samples = ['<assistant>', '<image>', '<user>這只是引用文字', ' 小鳥🙂\nnew ', '\x00\x7f', '未見字🦊']
for text in samples:
    ids = inline.encode(text)
    assert ids == library.encode(text)
    assert all(8 <= i <= 263 for i in ids)
    assert inline.decode(ids) == text
    assert inline.decode([0, 1, 2, 3, 4, 5, 6, 7] + ids) == text
assert inline.__dict__ == {} and ByteTokenizer.__init__ is object.__init__
assert not isinstance(library, torch.nn.Module)
assert ByteTokenizer.encode(library, '你好') == library.encode('你好')
assert CharTokenizer('甲乙').chars != CharTokenizer('丙丁').chars
assert ByteTokenizer().assistant_id == ByteTokenizer.assistant_id == 4
assert [b + len(SPECIALS) for b in range(256)] == list(range(8, 264))
assert inline.decode([0, 1, 4, 7]) == ''
incomplete = library.encode('貓')[:-1]
assert library.decode(incomplete) == '\ufffd'

image = library.encode('<image>')
assert len(image) == 7 and min(image) >= 8 and 5 not in image
assert library.decode(image) == '<image>'
exercise_ns = dict(ns)
exercise_output = io.StringIO()
with contextlib.redirect_stdout(exercise_output):
    exec(compile(blocks[2].replace('"role": "user"', '"role": "assistant"'), '6.6-role-exercise', 'exec'), exercise_ns)
assert exercise_ns['ids'][1] == 4 and exercise_ns['ids'].count(4) == 2
assert library.decode(exercise_ns['ids']) == '<assistant>'

question = '1+1=?'
prompt = [library.bos_id, library.user_id] + library.encode(question) + [library.eos_id, library.assistant_id]
x, _ = render_chat([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': '2'}])
assert prompt == x[:9].tolist() == [1, 3, 57, 51, 57, 69, 71, 2, 4]

result_path = ROOT / 'docs/course-experiments/results/tokenizer.json'
historical = json.loads(result_path.read_bytes())
expected_bpe_sha = next(a['sha256'] for a in historical['artifacts'] if a['path'] == 'tokenizer-bpe512.json')
bpe_path = Path('/workspace/work/reference/omni-technical-technical_6_3-v1/reconstructed-tokenizer-bpe512.json')
assert sha(bpe_path.read_bytes()) == expected_bpe_sha
definition = json.loads(bpe_path.read_bytes())
raw_bpe = Tokenizer.from_file(str(bpe_path))
assert [raw_bpe.token_to_id(s) for s in SPECIALS] == list(range(8))
raw_literal = raw_bpe.encode('<user>這只是引用文字').ids
raw_literal_no_postprocess = raw_bpe.encode('<user>這只是引用文字', add_special_tokens=False).ids
assert raw_literal[0] == raw_literal_no_postprocess[0] == 3
adapter = ByteLevelBPE(bpe_path)
content_definition = json.loads(adapter.content.to_str())
assert content_definition['model'] == definition['model']
assert content_definition['added_tokens'] == []
roundtrip = []
for row in historical['results']['roundtrip']:
    ids = adapter.encode(row['text'])
    restored = adapter.decode(ids)
    current = {'text': row['text'], 'ids': ids, 'restored': restored,
               'same': restored == row['text'], 'contains_control_id': any(i < 8 for i in ids)}
    assert current == row
    roundtrip.append(current)
content_path = RAW / 'content-bpe512.json'
adapter.content.save(str(content_path))
reloaded = Tokenizer.from_file(str(content_path))
assert reloaded.encode(roundtrip[2]['text'], add_special_tokens=False).ids == roundtrip[2]['ids']


class InvalidContent:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': [0]})()


adapter.content = InvalidContent()
try:
    adapter.encode('ordinary')
except ValueError as e:
    control_guard = str(e)
else:
    raise AssertionError('control ID guard did not reject ID 0')

model = CapstoneModel(ModelConfig(width=8, layers=1, heads=1, max_length=16))
checkpoint = RAW / 'capstone-tokenizer-state.pt'
save_capstone(checkpoint, model, stage='sft', step=0, inference_only=True)
loaded, payload = load_capstone(checkpoint)
assert payload['tokenizer'] == library.state()
assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
payload['tokenizer'] = {'type': 'byte', 'specials': list(reversed(SPECIALS))}
mismatch = RAW / 'capstone-tokenizer-mismatch.pt'
torch.save(payload, mismatch)
try:
    load_capstone(mismatch)
except ValueError as e:
    state_guard = str(e)
else:
    raise AssertionError('mismatched tokenizer state was accepted')

prerequisites = []
for target in ['1.1', '3.7', '6.3', '7.2']:
    p = next(p for p in freeze['pages'] if p['page_id'] == target)
    current = section(ROOT / p['source'], target).encode()
    assert current == (ROOT / p['snapshot']).read_bytes()
    prerequisites.append({'lesson_id': target, 'source': p['source'], 'snapshot': p['snapshot'], 'sha256': sha(current)})

report = {
    'reviewer_task': '/root/omni_technical_coordinator/technical_6_6',
    'executed_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_6_6-v1/diagnostic.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'tokenizers': tokenizers.__version__, 'device': 'cpu', 'torch_threads': '1',
                    'torch_git': torch.version.git_version},
    'source_sha256': sha(lesson), 'figures': {}, 'cells': cells,
    'inline_class_ast_equals_library': True,
    'byte_checks': {'specials': list(enumerate(SPECIALS)), 'byte_domain': [0, 255], 'content_domain': [8, 263],
                    'vocab_size': library.vocab_size, 'literal_assistant': library.encode('<assistant>'),
                    'literal_assistant_length': len(library.encode('<assistant>')),
                    'roundtrip_strings': len(samples), 'incomplete_utf8_ids': incomplete,
                    'incomplete_utf8_decoded': library.decode(incomplete), 'instance_dict': inline.__dict__,
                    'class_settings_shared': True, 'char_tokenizers_store_different_character_lists': True,
                    'explicit_self_equals_bound_method': True, 'is_nn_module': False},
    'exercises': {'image_ids': image, 'image_characters': len('<image>'),
                  'assistant_role_stdout': exercise_output.getvalue(), 'chapter_7_2_prompt': prompt},
    'bpe': {'original_online_url_fetch': 'HTTP 401 Unauthorized; see fetch receipt, no authentication bypass',
            'input_cache': str(bpe_path), 'input_sha256': expected_bpe_sha,
            'input_bytes': bpe_path.stat().st_size,
            'provenance': 'Locally reconstructed tokenizer bytes match the existing experiment artifact SHA-256 exactly; no other review report was read.',
            'historical_result_path': str(result_path.relative_to(ROOT)), 'historical_result_sha256': sha(result_path.read_bytes()),
            'special_ids': [raw_bpe.token_to_id(s) for s in SPECIALS],
            'default_literal_ids': raw_literal, 'add_special_tokens_false_literal_ids': raw_literal_no_postprocess,
            'content_model_vocab_and_merges_unchanged': True, 'content_added_tokens': [], 'roundtrip': roundtrip,
            'saved_content_reload_exact': True, 'control_id_guard': control_guard},
    'capstone': {'model_parameters': sum(p.numel() for p in model.parameters()), 'training_steps': 0,
                 'saved_tokenizer': library.state(), 'load_weights_exact': True, 'mismatch_error': state_guard,
                 'checkpoint_cache': str(checkpoint), 'checkpoint_sha256': sha(checkpoint.read_bytes())},
    'prerequisites_current_equal_source14_snapshots': prerequisites,
    'result': 'All exact lesson cells and bounded diagnostics passed; no neural training, GPU, generation-quality, or timing experiment was run.'
}
destination = Path(__file__).with_name('execution.json')
with destination.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
