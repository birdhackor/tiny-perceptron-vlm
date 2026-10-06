"""Bounded filesystem/path callback; never invokes evaluation.main or a model."""
from pathlib import Path,PureWindowsPath,PurePosixPath
from types import SimpleNamespace
import ast,hashlib,json,os,platform,re,sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tiny_perceptron.selftrained.dataset import file_sha256
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(label,value):print(label,json.dumps(value,ensure_ascii=False,sort_keys=True))
report=json.loads((OUT/'before-report.json').read_bytes())
prior=(OUT/'previous-evaluate.py').read_bytes()
current=(ROOT/'scripts/selftrained/evaluate.py').read_bytes()
assert current==(OUT/'current-evaluate.py').read_bytes()
old_source=next(s for s in report['sources'] if s['id']=='s_eval')
assert hashlib.sha256(prior).hexdigest()==old_source['sha256']
expected=prior.replace(b'str(path.relative_to(root))',b'path.relative_to(root).as_posix()').replace(b'journal.receipt_path.open("rb")',b'journal.receipt_path.open("r+b")')
assert expected==current
emit('exact_source_delta',{'prior_sha256':hashlib.sha256(prior).hexdigest(),'current_sha256':sha(ROOT/'scripts/selftrained/evaluate.py'),'only_changes':['code_fingerprints: str(relative_path) -> relative_path.as_posix()','main pre-marker fsync: receipt open rb -> r+b'],'all_other_source_bytes_identical':True})
ta,tb=ast.parse(prior),ast.parse(current)
na={n.name:n for n in ta.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
nb={n.name:n for n in tb.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
unchanged=['EvaluationJournal','protocol_contents','validate_protocol','Generator','summarize','score_reply']
for name in unchanged:assert ast.dump(na[name],include_attributes=False)==ast.dump(nb[name],include_attributes=False)
get_branch=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="record.get('evaluation', {}).get('mode') == 'voice_topic_continuation'")
assert ast.dump(get_branch(ta),include_attributes=False)==ast.dump(get_branch(tb),include_attributes=False)
emit('reused_original_contract_evidence',{'unchanged_ast_definitions':unchanged,'voice_continuation_gold_exclusion_branch_unchanged':True,'prior_cpu_stub_and_input_exclusion_evidence_reusable':True,'original_heldout_results_remain_prior_version_measurements':True})
def fingerprint(node):
 ns={'Path':Path,'__file__':str(ROOT/'scripts/selftrained/evaluate.py'),'file_sha256':file_sha256}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'evaluate.py:code_fingerprints','exec'),ns)
 return ns['code_fingerprints']()
old_map,new_map=fingerprint(na['code_fingerprints']),fingerprint(nb['code_fingerprints'])
assert old_map==new_map and new_map['scripts/selftrained/evaluate.py']==sha(ROOT/'scripts/selftrained/evaluate.py')
assert all('\\' not in key for key in new_map)
windows=PureWindowsPath('tiny_perceptron/selftrained/dataset.py')
assert '\\' in str(windows) and windows.as_posix()=='tiny_perceptron/selftrained/dataset.py'
assert PurePosixPath('tiny_perceptron/selftrained/dataset.py').as_posix()==windows.as_posix()
emit('path_checks',{'original_and_current_function_maps_equal_on_current_posix_tree':True,'fingerprint_entries':len(new_map),'all_keys_posix':True,'current_evaluate_fingerprint_tracks_new_bytes':True,'pure_windows_path':str(windows),'windows_and_posix_as_posix_same':True,'actual_windows_execution':False})
receipt=OUT/'fixture-receipt.json';receipt.write_bytes(b'{"purpose":"filesystem-portability-helper-only"}\n')
original_receipt=receipt.read_bytes();original_sha=sha(receipt)
fsync_branch=next(n for n in ast.walk(tb) if isinstance(n,ast.With) and len(n.items)==1 and ast.unparse(n.items[0].context_expr)=="journal.receipt_path.open('r+b')")
ns={'journal':SimpleNamespace(receipt_path=receipt),'os':os}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fsync_branch],type_ignores=[])),'evaluate.py:receipt-fsync','exec'),ns)
assert receipt.read_bytes()==original_receipt and sha(receipt)==original_sha
emit('receipt_fsync',{'executed_original_current_ast_lines':[fsync_branch.lineno,fsync_branch.end_lineno],'mode':'r+b','fsync_completed_on_platform':platform.system(),'receipt_bytes_sha256_unchanged':original_sha,'scope':'A benign helper fixture only; no evaluation journal or protocol was created/resumed.'})
changed=[]
for s in report['sources']:
 if s['kind']=='repository_code' and s['id']!='s_eval' and sha(ROOT/s['path'])!=s['sha256']:changed.append(s['path'])
assert not changed,changed
raw=(ROOT/'course/chapters/19.md').read_bytes();text=raw.decode('utf-8');hs=list(re.finditer(r'^## .+$',text,re.M));i,h=next((i,h) for i,h in enumerate(hs) if h.group().startswith('## 19.3 '));end=hs[i+1].start() if i+1<len(hs) else len(text);body=text[h.start():end].encode()
assert hashlib.sha256(body).hexdigest()==report['source_sha256']
for path,digest in report['figure_sha256'].items():assert sha(ROOT/path)==digest
emit('other_review_sources',{'other_repository_code_unchanged':True,'section_sha256_unchanged':report['source_sha256'],'figures_unchanged':True})
emit('scope_and_result',{'verdict':'pass','cpu_helper_only':True,'training_test_or_model_generation_executed':False,'windows_runtime_verified':False,'frozen_protocol_code_sha_gate_preserved':True,'old_frozen_test_protocol_not_regenerated_or_reexecuted':True})
emit('environment',{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu'})
