import sys,json,torch
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'CPU'}))

print('fence 0')
truth = "1010"
predicted = "1000"
assert len(truth) == len(predicted)  # 本實驗只比較等長且對齊的字串
matched = sum(a == b for a, b in zip(truth, predicted))
print("等長逐位正確率", matched / len(truth))
print("整串正確", truth == predicted)
print("本例一次替換的CER", 1 / len(truth))
print("獨立錯誤假設下五位全對", round(0.9**5, 4))

# Owner supplied proportionate control
import ast
from pathlib import Path
helper = Path('docs/technical-reviews/artifacts/p7_technical_c/originals/torchmetrics-edit-helper-v1.8.2.py')
tree = ast.parse(helper.read_text())
func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_edit_distance')
ns = {}; exec(compile(ast.Module(body=[func], type_ignores=[]), str(helper), 'exec'), ns)
def own_distance(pred, ref):
    row = list(range(len(ref) + 1))
    for i, pc in enumerate(pred, 1):
        new = [i]
        for j, rc in enumerate(ref, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (pc != rc)))
        row = new
    return row[-1]
for predicted_example in ['1000', '10100', '010', '1010', '101']:
    distance = own_distance(predicted_example, '1010')
    assert distance == ns['_edit_distance'](list(predicted_example), list('1010'))
    print('edit example', predicted_example, 'reference 1010', 'edits', distance, 'CER', distance / 4)
    if len(predicted_example) != 4:
        try:
            assert len('1010') == len(predicted_example)
        except AssertionError:
            print('unequal length rejected', predicted_example)
raw_path = Path('docs/technical-reviews/artifacts/p7_technical_c/originals/ocr-raw.json')
raw = json.loads(raw_path.read_text())
samples = raw['results']['test']['samples']
edits = total_ref = eos = exact = 0
for item in samples:
    pred, ref = item['generated'], item['target']
    distance = own_distance(pred, ref)
    assert distance == ns['_edit_distance'](list(pred), list(ref))
    print('historical row', item['row'], 'target', ref, 'generated', pred, 'edits', distance, 'reference', len(ref), 'EOS', item['eos'])
    edits += distance; total_ref += len(ref); eos += int(item['eos']); exact += int(pred == ref)
print('history', 'samples', len(samples), 'edits', edits, 'reference', total_ref, 'CER', edits / total_ref, 'EOS', eos, 'exact', exact)
assert (len(samples), edits, total_ref, eos, exact) == (30, 40, 57, 30, 2)
