from pathlib import Path
import ast, copy, json, io, contextlib, sys

base = Path(__file__).resolve().parents[1]
raw = (base / 'code/fence-1.py').read_text()
tree = ast.parse(raw)
record_node = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'records')
original = ast.literal_eval(record_node.value)
compute = ast.Module(body=tree.body[1:], type_ignores=[])

def check(name, records, expected):
    before = copy.deepcopy(records)
    namespace = {'records': records}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(compute, 'original-fence-1.py:computation', 'exec'), namespace)
    overlap = sorted(namespace['families']['train'] & namespace['families']['test'])
    assert overlap == expected, (name, overlap, expected)
    assert records == before, 'The demonstration unexpectedly mutates the records'
    print(json.dumps({'case':name,'overlap':overlap,'rows':len(records),'stdout':output.getvalue(),'records_unchanged':True},ensure_ascii=False))

check('original', copy.deepcopy(original), [])
cross = copy.deepcopy(original); cross[1]['split'] = 'test'
check('move_second_row_to_test', cross, ['貓照片A'])
rename = copy.deepcopy(cross); rename[1]['family'] = '貓照片A-renamed'
check('renaming_hides_identity_from_this_checker', rename, [])
print(json.dumps({'python':sys.version,'device':'cpu','scope':'bounded set operations, no training or evaluation'},ensure_ascii=False))
