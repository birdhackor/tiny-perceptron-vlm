"""Independent CPU recount of historical evidence, never train/load a model."""
import ast
import collections
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None
hist = OUT / "inputs/historical"


def selected(path, names, namespace):
    """Execute only selected original AST definitions, preserving their code semantics."""
    tree = ast.parse((hist / path).read_bytes())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(hist / path), "exec"), namespace)


ns = {"torch": torch, "json": json, "hashlib": hashlib, "random": random}
selected("tiny_perceptron/data.py",
         {"IGNORE", "SPECIALS", "ByteTokenizer", "render_chat", "pad_batch", "shifted"}, ns)
selected("scripts/prepare_data.py", {"conversation", "generate_records"}, ns)
selected("scripts/course_experiments/common.py",
         {"split_records", "records_sha256", "text_examples"}, ns)
selected("scripts/course_experiments/text.py", {"arithmetic_records"}, ns)

evidence = json.loads((OUT / "inputs/current/docs/course-experiments/results/sft_ablation.json").read_text())
r = evidence["results"]
parts = {
    "A": ns["split_records"](ns["generate_records"]("attributes-sft"), seed=42),
    "B": ns["split_records"](ns["arithmetic_records"](), seed=42),
}
out = {"environment": {"python": sys.version, "torch": str(torch.__version__),
       "torch_git": torch.version.git_version, "device": "cpu", "platform": platform.platform(),
       "historical_training_runtime": f"Python {evidence['python_version']}, torch {evidence['torch_version']}, {evidence['gpu']}",
       "scope": "historical sampling/label/score recount only; no model weights, training, inference, download"},
       "splits": {}, "training": {}, "evaluations": {}, "toy_variants": {}}

for task, splits in parts.items():
    out["splits"][task] = {}
    fam = {s: {row['family'] for row in rows} for s, rows in splits.items()}
    assert not (fam['train'] & fam['validation'] or fam['train'] & fam['test'] or fam['validation'] & fam['test'])
    for split, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        h = hashlib.sha256(raw).hexdigest()
        manifest = r['data']['attributes' if task == 'A' else 'arithmetic'][split]
        assert h == manifest['sha256']
        assert len(rows) == manifest['records'] and len(fam[split]) == manifest['families']
        q = OUT / 'reconstructed-inputs' / task / f'{split}.jsonl'
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(raw)
        examples = ns['text_examples'](rows, mode='sft', max_length=128)
        lengths = [int((y != -100).sum()) for x,y in examples]
        out['splits'][task][split] = {'records': len(rows), 'families': len(fam[split]),
            'sha256': h, 'effective_targets': sum(lengths), 'target_length_distribution': dict(collections.Counter(lengths)),
            'input_length_distribution': dict(collections.Counter(len(x) for x,y in examples))}

for branch, rows, sources in [
    ('b-only', parts['B']['train'], ['B'] * len(parts['B']['train'])),
    ('replay', parts['A']['train'] + parts['B']['train'], ['A'] * len(parts['A']['train']) + ['B'] * len(parts['B']['train']))
]:
    examples = ns['text_examples'](rows, mode='sft', max_length=128)
    train = r['runs'][branch]['training']
    assert ns['records_sha256'](rows) == train['records_sha256']
    rng = random.Random(42)
    drawn = [i for step in range(500) for i in rng.choices(range(len(examples)), k=16)]
    counts = collections.Counter(sources[i] for i in drawn)
    target_counts = {s: sum(int((examples[i][1] != -100).sum()) for i in drawn if sources[i] == s)
                     for s in ['A','B']}
    total = sum(target_counts.values())
    assert total == train['effective_tokens'] and train['steps'] == 500
    assert train['records'] == len(rows)
    input_total = sum(len(examples[i][0]) for i in drawn)
    padded_input_total = sum(16 * max(len(examples[i][0]) for i in drawn[j:j+16]) for j in range(0,len(drawn),16))
    first_x, first_y, first_valid = ns['pad_batch']([examples[i] for i in drawn[:16]])
    assert int(first_valid.sum()) == sum(len(examples[i][0]) for i in drawn[:16])
    assert int((first_y != -100).sum()) == sum(int((examples[i][1] != -100).sum()) for i in drawn[:16])
    out['training'][branch] = {'steps': 500, 'batch_size': 16, 'draws': len(drawn),
         'sampling': 'uniform random.Random(42).choices with replacement from all training rows; not a fixed A:B quota per batch',
         'source_counts': dict(counts), 'unique_sampled_records': len(set(drawn)),
         'effective_targets_by_source': target_counts, 'effective_targets': total,
         'input_nonpadding_positions': input_total, 'input_padded_positions': padded_input_total,
         'first_batch_axes': {'x': list(first_x.shape), 'labels': list(first_y.shape), 'valid': list(first_valid.shape)},
         'records_sha256': train['records_sha256']}

tok = ns['ByteTokenizer']()
for branch, report in [('before',r['before']), ('b-only',r['runs']['b-only']), ('replay',r['runs']['replay'])]:
    out['evaluations'][branch] = {}
    for task, taskkey in [('A','A_attributes'),('B','B_arithmetic')]:
        out['evaluations'][branch][task] = {}
        for split in ['validation','test']:
            ev=report[taskkey][split]
            assert len(ev['samples']) == len(parts[task][split]) == ev['records'] == ev['examples']
            targets = out['splits'][task][split]['effective_targets']
            assert targets == ev['effective_tokens']
            matches=ended=0
            flags=[]
            for row,sample in zip(parts[task][split],ev['samples']):
                assert sample['messages'] == row['messages'][:-1]
                assert sample['expected'] == row['messages'][-1]['content']
                ids=sample['generated_ids']; assert len(ids) <= 32
                eos=tok.eos_id in ids
                raw=ids[:ids.index(tok.eos_id)] if eos else ids
                exact=raw == tok.encode(sample['expected'])
                assert tok.decode(raw) == sample['generated']
                assert exact == sample['exact'] and eos == sample['eos']
                matches+=exact; ended+=eos; flags.append(exact)
            assert matches == ev['matches'] and matches/ev['records'] == ev['exact_match']
            assert ended/ev['records'] == ev['eos_rate']
            division_error=abs(ev['nll_sum']/targets - ev['nll'])
            assert division_error <= 1e-12
            out['evaluations'][branch][task][split] = {'matches': matches, 'records': ev['records'],
                'eos': ended, 'effective_targets': targets, 'exact_match': matches/ev['records'],
                'nll_division_absolute_error': division_error, 'raw_token_exact_flags': flags,
                'scope': 'Recompute from recorded generated IDs; no new inference or NLL forward pass'}

base_flags=out['evaluations']['before']['A']['test']['raw_token_exact_flags']
rep_flags=out['evaluations']['replay']['A']['test']['raw_token_exact_flags']
out['test_retention']={'base_correct_still_correct':sum(a and b for a,b in zip(base_flags,rep_flags)),
    'base_correct_now_wrong':sum(a and not b for a,b in zip(base_flags,rep_flags)),
    'base_wrong_now_correct':sum(not a and b for a,b in zip(base_flags,rep_flags))}
out['cost_ratios']={'replay_to_b_only_effective_targets':36384/18453,
    'expected_replay_source_row_share_A':45/94,
    'actual_replay_source_row_share_A':out['training']['replay']['source_counts']['A']/8000}

ns_toy={}
exec(compile((OUT/'original-execution/fence-1.py').read_bytes(),'original-fence','exec'),ns_toy)
a,b=ns_toy['a'],ns_toy['b']
def recipe_stats(rows):
    by_src=collections.Counter(row['source'] for row in rows)
    targets={s:sum(row['answer_tokens'] for row in rows if row['source']==s) for s in ['A','B']}
    return {'sources':[row['source'] for row in rows], 'records':len(rows), 'source_counts':dict(by_src),
            'targets':targets,'total_targets':sum(targets.values()),
            'token_share_A': targets['A']/sum(targets.values()),
            'unique_object_count':len({id(x) for x in rows})}
out['toy_variants']['original']=recipe_stats(a+b)
out['toy_variants']['B_only']=recipe_stats([b[0],b[1],b[0],b[1]])
variant=[a[0],b[0],b[1],b[0]]
out['toy_variants']['25_75_records']=recipe_stats(variant)
assert out['toy_variants']['25_75_records']['total_targets']==8
assert out['toy_variants']['25_75_records']['unique_object_count']==3
for row in b: row['answer_tokens']=4
out['toy_variants']['longer_B']=recipe_stats(variant)
assert out['toy_variants']['longer_B']['targets']=={'A':2,'B':12}
assert out['toy_variants']['longer_B']['total_targets']==14
assert out['toy_variants']['longer_B']['token_share_A'] == 1/7
(OUT/'recomputed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
print('PASS: exact integers/hashes/ID match; NLL denominator division <= 1e-12; no training or new inference.')
