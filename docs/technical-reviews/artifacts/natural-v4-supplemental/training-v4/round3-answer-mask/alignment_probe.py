"""Bounded answer-mask correction check; reuse byte-identical original guide audit."""
from pathlib import Path
import contextlib, hashlib, io, json, re, subprocess, sys
OUT = Path(__file__).resolve().parent
EVIDENCE = OUT.parent
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
prior = json.loads((OUT / 'report.prior.original.json').read_text())
assert sha(OUT / 'report.prior.original.json') == 'aafd27515cb0415bb187c001795df9130e7284deabcc256a83f04597f02a3ce1'
assert (ROOT / prior['source']).read_bytes() == (EVIDENCE / 'TRAINING.round2.raw.md').read_bytes()
previous_checks = json.loads((EVIDENCE / 'round2/recheck.stdout.json').read_text())['unchanged_input_and_evidence']
baseline = {}
changed = {}
for p, record in previous_checks.items():
    actual = sha(ROOT / p)
    if p in {'course/chapters/20.md', 'course/figures/natural-v4-answer-mask.svg'}:
        changed[p] = {'prior_sha256': record['sha256'], 'current_sha256': actual}
    else:
        assert actual == record['sha256'], p
        baseline[p] = record
for a in prior['artifacts']:
    assert sha(ROOT / a['path']) == a['sha256'], a['path']
for s in prior['sources']:
    if s['kind'] == 'repository_code': assert sha(ROOT / s['path']) == s['sha256'], s['path']
for p, h in prior['figure_sha256'].items():
    if p != 'course/figures/natural-v4-answer-mask.svg': assert sha(ROOT / p) == h
assert sha(ROOT / 'course/figures/natural-v4-answer-mask.svg') == 'ffc11a178aef0ba81ed1416c10ddeb90f8b233927cc5bdee4a7d5b620a04707b'

chapter = (ROOT / 'course/chapters/20.md').read_text()
sections = {}
for key in ['20.6', '20.7', '20.8']:
    body = re.search(r'^## ' + re.escape(key) + r' .*?(?=^## |\Z)', chapter, re.M | re.S).group()
    sections[key] = {'sha256': hashlib.sha256(body.encode()).hexdigest(), 'lines': len(body.splitlines())}
    (OUT / (key + '.current.raw.md')).write_text(body)
body = re.search(r'^## 20\.7 .*?(?=^## |\Z)', chapter, re.M | re.S).group()
code = re.search(r'```python\n(.*?)\n```', body, re.S).group(1)
capture = io.StringIO()
with contextlib.redirect_stdout(capture): exec(compile(code, '<current 20.7 exact code>', 'exec'), {})
observed = capture.getvalue()
expected = '輸入位置數 6\n計分目標數 2\n第一個計分目標的位置 4\n計分目標ID [73, 2]\n'
assert observed == expected
cases = {}
tok = ByteTokenizer()
for answer in ['A', 'AB']:
    x, y = render_chat([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': answer}])
    positions = (y != -100).nonzero().flatten().tolist()
    assert positions[0] == 4 and len(positions) == len(answer) + 1
    assert x[4].item() == tok.assistant_id and y[4].item() == tok.encode(answer)[0]
    assert y[positions[-1]].item() == tok.eos_id
    assert bool((y[:4] == -100).all()) and tok.encode('Q')[0] in x.tolist()
    cases[answer] = {'input_ids': x.tolist(), 'already_aligned_target_ids': y.tolist(), 'active_positions': positions, 'active_targets': y[y != -100].tolist()}

nb_path = ROOT / 'outputs/notebooks/20/20.7.ipynb'
notebook = json.loads(nb_path.read_text())
cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']) == code)
nb_stdout = ''.join(''.join(o['text']) for o in cell['outputs'] if o['output_type'] == 'stream' and o['name'] == 'stdout')
assert cell['execution_count'] == 3 and nb_stdout == observed
svg_cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code' and 'display(SVG' in ''.join(c['source']))
svg_output = next(o['data']['image/svg+xml'] for o in svg_cell['outputs'] if 'image/svg+xml' in o.get('data', {}))
svg_bytes = ''.join(svg_output).encode()
# IPython removes the final newline when embedding SVG; compare that exact known normalisation.
assert svg_bytes == (ROOT / 'course/figures/natural-v4-answer-mask.svg').read_bytes().rstrip(b'\n')

authority = ROOT / 'outputs/natural-v4/factual-research/training-v4/transformers-loss-v4.57.6.py'
official = next(s for s in prior['sources'] if s['id'] == 's_transformers_loss_v4')
assert sha(authority) == official['retrieval_sha256']
source = authority.read_text()
assert 'shift_labels = labels[..., 1:].contiguous()' in source and 'ignore_index: int = -100' in source
# A faithful labels[1:] illustration, without importing absent Transformers or loading weights.
original_targets = [-100, -100, -100, -100, -100, 73, 2]
assert original_targets[1:] == cases['A']['already_aligned_target_ids']
print(json.dumps({'scope': 'New complete necessary20.6–20.8 source read and new figure alignment check. Guide full327-line audit reused after byte equality; no fictitious new whole-guide read/release rerun.', 'prior_report_sha256': sha(OUT / 'report.prior.original.json'), 'guide_sha256': sha(ROOT / prior['source']), 'guide_byte_equal_to_prior_fullread_snapshot': True, 'unchanged_baseline': baseline, 'changed_registered_prerequisites': changed, 'current_section_snapshots': sections, 'new_svg_sha256': sha(ROOT / 'course/figures/natural-v4-answer-mask.svg'), 'source_locators': {'tiny_perceptron/data.py': {'sha256': sha(ROOT / 'tiny_perceptron/data.py'), 'locator': 'ByteTokenizer14–30;render_chat54–68'}, 'tiny_perceptron/natural_assistant.py': {'sha256': sha(ROOT / 'tiny_perceptron/natural_assistant.py'), 'locator': 'encode_training_row157–173;train labels[:,1:] denominator612–617'}, 'original_transformers_loss': {'url': official['url'], 'version': official['version'], 'retrieval_sha256': sha(authority), 'locator': 'ForCausalLMLoss46–68;original labels pad/labels[...,1:] lines59–60,ignore_index=-100'}} , 'exact_current_20_7_code': {'expected_stdout': expected, 'observed_stdout': observed, 'cases': cases}, 'existing_notebook_audit': {'path': nb_path.relative_to(ROOT).as_posix(), 'sha256': sha(nb_path), 'execution_count': cell['execution_count'], 'actual_saved_stdout': nb_stdout, 'matches_own_fresh_cpu_probe': True, 'embedded_svg_matches_current_after_IPython_final_newline_removal': True, 'scope': 'Existing executed CPU notebook inspected; not claimed as own whole-notebook or book rerun.'}, 'conclusion': 'The new next-target label accurately shows already aligned toy targets: assistant position4 predictsA, Aposition5 predictsEOS. Inputs retainQ; onlyA/EOS are scored. Production Transformers shifts original labels internally once. The diagram does not imply an extra shift or six-token real Qwen input.', 'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu'}}, ensure_ascii=False, indent=2))
