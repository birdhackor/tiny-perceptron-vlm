"""Bounded CPU arithmetic and audit of saved historical tokenizer evidence. No optimizer/training."""
import ast
import hashlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-6_4-independent'
HIST = OUT / 'historical'
sys.path.insert(0, str(HIST))
import torch
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tiny_perceptron.data import ByteTokenizer, SPECIALS
from tiny_perceptron.model import ModelConfig, TinyLM

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device('cpu')
environment = {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'tokenizers': tokenizers.__version__, 'device': 'cpu', 'cuda_build': str(torch.version.cuda), 'validation_scope': 'No training, optimizer, checkpoints, model download, dataset download or GPU calls. Historical results inspected and short CPU recomputation only.'}
(OUT / 'bounded-environment.json').write_text(json.dumps(environment, indent=2) + '\n')

# Freeze the installed public API implementation to document the newer local version.
for name, obj in [('installed-Embedding.py', torch.nn.Embedding), ('installed-Linear.py', torch.nn.Linear)]:
    (OUT / name).write_text(inspect.getsource(obj))

result = json.loads((OUT / 'inputs/tokenizer-result.json').read_text())
report = {'environment': environment, 'historical_run': {k: result[k] for k in ['revision', 'device', 'seed', 'torch_version', 'python_version', 'step_scale']}}
table = []
for width in [32, 64]:
    for vocab in [300, 1000, 3000]:
        embedding = torch.nn.Embedding(vocab, width, dtype=torch.float32)
        output = torch.nn.Linear(width, vocab, bias=False, dtype=torch.float32)
        observed = {'vocab': vocab, 'width': width, 'input_shape': list(embedding.weight.shape), 'output_shape': list(output.weight.shape), 'embedding': embedding.weight.numel(), 'input_FP32_bytes': embedding.weight.numel() * embedding.weight.element_size(), 'untied_tables': embedding.weight.numel() + output.weight.numel(), 'output_bias': output.bias is not None}
        assert observed['embedding'] == vocab * width
        assert observed['input_FP32_bytes'] == 4 * vocab * width
        assert observed['untied_tables'] == 2 * vocab * width
        table.append(observed)
for i in range(3):
    assert table[i + 3]['embedding'] == table[i]['embedding'] * 2
    assert table[i + 3]['input_FP32_bytes'] == table[i]['input_FP32_bytes'] * 2
    assert table[i + 3]['untied_tables'] == table[i]['untied_tables'] * 2
report['original_and_width_variation'] = table
report['vocab_600_variation'] = {'embedding': 600 * 32, 'input_FP32_bytes': 600 * 32 * 4, 'untied_tables': 600 * 32 * 2}
assert all(report['vocab_600_variation'][k] == 2 * table[0][k] for k in report['vocab_600_variation'])

# Execute only personally inspected pure helper bodies from their recorded original bytes.
namespace = {'hashlib': hashlib, 'json': json, 'random': random, 'SPECIALS': SPECIALS}
for rel, names in [('scripts/course_experiments/common.py', {'split_records'}), ('scripts/course_experiments/text.py', {'_deduplicate_text', '_utf8_prefix', '_BPE', '_digest', '_json_bytes'})]:
    path = HIST / rel
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


raw = read_rows(OUT / 'inputs/assets/tinystories-train-512.jsonl')[:96]
raw += read_rows(OUT / 'inputs/assets/chinese-classical-train-365.jsonl')[:96]
complete = namespace['split_records'](namespace['_deduplicate_text'](raw), seed=42)
parts = {split: [{**row, 'text': namespace['_utf8_prefix'](row['text'], 256), 'complete_text_sha256': hashlib.sha256(row['text'].encode()).hexdigest()} for row in rows] for split, rows in complete.items()}
split_checks = {}
for split, rows in parts.items():
    computed = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    stored = (OUT / ('inputs/experiment/data/' + split + '.jsonl')).read_bytes()
    assert computed == stored
    assert hashlib.sha256(computed).hexdigest() == result['results']['data'][split]['sha256']
    split_checks[split] = {'records': len(rows), 'sha256': hashlib.sha256(computed).hexdigest(), 'matches_original_bytes': True, 'max_utf8_bytes': max(len(row['text'].encode()) for row in rows)}
for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
    assert {r['family'] for r in parts[left]}.isdisjoint({r['family'] for r in parts[right]})
report['split_reconstruction'] = split_checks

definition = json.loads((OUT / 'inputs/experiment/tokenizer-bpe512.json').read_text())
assert definition['normalizer'] is None
assert definition['pre_tokenizer']['type'] == 'ByteLevel' and not definition['pre_tokenizer']['add_prefix_space']
assert definition['decoder']['type'] == 'ByteLevel'
byte = ByteTokenizer()
bpe = namespace['_BPE'](Tokenizer.from_file(str(OUT / 'inputs/experiment/tokenizer-bpe512.json')))
assert byte.state() == json.loads((OUT / 'inputs/experiment/tokenizer-byte.json').read_text())
assert [bpe.tokenizer.token_to_id(token) for token in SPECIALS] == list(range(8))
assert len(pre_tokenizers.ByteLevel.alphabet()) == 256
assert all(bpe.tokenizer.token_to_id(symbol) is not None for symbol in pre_tokenizers.ByteLevel.alphabet())
runs = {}
for name, tok in [('byte256', byte), ('bpe512', bpe)]:
    rows = parts['validation']
    lengths = [len(tok.encode(row['text'])) for row in rows]
    assert lengths == result['results']['runs'][name]['validation_text_token_lengths']
    assert all(tok.decode(tok.encode(row['text'])) == row['text'] for side in parts.values() for row in side)
    assert all(not any(value in range(8) for value in tok.encode(row['text'])) for side in parts.values() for row in side)
    # Count parameters of a fresh CPU model made from the historical ModelConfig, no weights loaded.
    model = TinyLM(ModelConfig(width=32, layers=1, max_length=384, vocab_size=tok.vocab_size))
    model_count = sum(p.numel() for p in model.parameters())
    recorded = result['results']['runs'][name]
    assert model_count == recorded['parameters']
    assert model.embedding.weight.numel() == recorded['input_embedding_parameters']
    assert model.output.bias is None and model.output.weight is not model.embedding.weight
    breakdown = {n: p.numel() for n, p in model.named_parameters()}
    runs[name] = {'vocab_size_including_8_specials': tok.vocab_size, 'ordinary_validation_tokens': sum(lengths), 'per_record_lengths': lengths, 'mean_tokens_per_validation_record': sum(lengths) / len(rows), 'raw_utf8_bytes': sum(len(row['text'].encode()) for row in rows), 'effective_targets_with_EOS': sum(lengths) + len(rows), 'parameters': model_count, 'parameter_breakdown': breakdown, 'exact_roundtrips_across_all_saved_records': sum(len(side) for side in parts.values())}
    for when in ['before', 'after']:
        assert recorded[when]['validation']['effective_tokens'] == sum(lengths) + len(rows)
        assert recorded[when]['validation']['raw_utf8_bytes'] == runs[name]['raw_utf8_bytes']
    assert recorded['training']['steps'] == 400
    del model
assert runs['byte256']['ordinary_validation_tokens'] == 3894
assert runs['bpe512']['ordinary_validation_tokens'] == 2112
assert runs['bpe512']['parameters'] - runs['byte256']['parameters'] == 2 * (512 - 264) * 32 == 15872
report['historical_saved_tokenizer_recomputation'] = runs
sampler = random.Random(42)
schedule = [sampler.choices(range(len(parts['train'])), k=8) for _ in range(400)]
schedule_hash = namespace['_digest'](schedule)
assert schedule_hash == result['results']['raw_document_schedule_sha256']
exposed = sum(len(parts['train'][index]['text'].encode()) for batch in schedule for index in batch)
assert all(result['results']['runs'][name]['training_raw_utf8_bytes_exposed'] == exposed for name in ['byte256', 'bpe512'])
report['matched_schedule'] = {'sha256': schedule_hash, 'steps': 400, 'documents_per_batch': 8, 'raw_utf8_bytes_exposed_each': exposed, 'comparison_scope': 'same raw documents/exposure; parameters/effective tokens/FLOPs not matched, no universal optimal vocabulary inference'}

# Tiny self-supplied counterexample: English-only merges leave this Chinese text at byte granularity.
tiny_training = ['walking walker walked happily slowly wonderful token tokenization encoding encoder encoded', 'the red cat sees the blue dog and the green bird']
sample = '罕見中文小鳥🙂'
tiny_results = []
for budget in [264, 300]:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(tiny_training, trainers.BpeTrainer(vocab_size=budget, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    encoded = tok.encode(sample)
    assert tok.decode(encoded.ids) == sample
    tiny_results.append({'budget': budget, 'actual_vocabulary': tok.get_vocab_size(), 'ordinary_tokens': len(encoded.ids), 'roundtrip': tok.decode(encoded.ids), 'same': True})
assert tiny_results[0]['actual_vocabulary'] < tiny_results[1]['actual_vocabulary']
assert tiny_results[0]['ordinary_tokens'] == tiny_results[1]['ordinary_tokens']
report['bounded_noncoverage_counterexample'] = {'training_strings': tiny_training, 'input': sample, 'runs': tiny_results, 'scope': 'Existence of no length improvement; not a measurement of typical Chinese tokenizers or model quality.'}

# Fixed V counterexample: the spelling length of pieces itself does not change table dimensions.
fixed_v_sample = 'abcde'
fixed_v_results = []
base = {symbol: index for index, symbol in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
for name, merges in [('short_pieces', [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]), ('long_piece', [('a', 'b'), ('ab', 'c'), ('abc', 'd'), ('abcd', 'e')])]:
    vocab = dict(base)
    for left, right in merges:
        vocab[left + right] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    encoded = tok.encode(fixed_v_sample)
    assert tok.decode(encoded.ids) == fixed_v_sample
    fixed_v_results.append({'name': name, 'merges': merges, 'V': tok.get_vocab_size(), 'tokens': encoded.tokens, 'sequence_length': len(encoded.ids), 'max_added_piece_bytes': max(len(left + right) for left, right in merges), 'input_parameters_width32': tok.get_vocab_size() * 32, 'untied_weight_tables_width32': 2 * tok.get_vocab_size() * 32, 'same_roundtrip': True})
assert fixed_v_results[0]['V'] == fixed_v_results[1]['V'] == 260
assert fixed_v_results[0]['sequence_length'] == 3 and fixed_v_results[1]['sequence_length'] == 1
assert fixed_v_results[0]['input_parameters_width32'] == fixed_v_results[1]['input_parameters_width32'] == 8320
report['fixed_v_piece_length_counterexample'] = {'input': fixed_v_sample, 'runs': fixed_v_results, 'scope': 'Existing V budget allocated to different merge definitions; larger token spellings are not sufficient to imply larger V or weight tables. Does not imply that adding merges to a fixed existing vocabulary leaves V unchanged.'}

# Input rows receive only observed-token positive examples; dense output gradients can still be nonzero.
torch.manual_seed(42)
embedding = torch.nn.Embedding(4, 2)
output = torch.nn.Linear(2, 4, bias=False)
loss = torch.nn.functional.cross_entropy(output(embedding(torch.tensor([0, 0, 1]))), torch.tensor([1, 1, 0]))
loss.backward()
assert torch.count_nonzero(embedding.weight.grad[3]) == 0
assert torch.count_nonzero(output.weight.grad[3]) > 0
report['rare_row_scope'] = {'absent_input_row_gradient': embedding.weight.grad[3].tolist(), 'absent_output_row_gradient': output.weight.grad[3].tolist(), 'scope': 'Rare large-token claim interpreted as few direct observed examples; it does not claim absent output logits receive no softmax gradient.'}
(OUT / 'bounded-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'pass', 'original_fence_and_bounded_arithmetic': 'exact', 'ordinary_validation_tokens': [runs[n]['ordinary_validation_tokens'] for n in ['byte256', 'bpe512']], 'parameters': [runs[n]['parameters'] for n in ['byte256', 'bpe512']], 'means': [runs[n]['mean_tokens_per_validation_record'] for n in ['byte256', 'bpe512']], 'matched_raw_bytes_exposed': exposed, 'tiny_noncoverage': tiny_results, 'history_not_retrained': True}, ensure_ascii=False, indent=2))
