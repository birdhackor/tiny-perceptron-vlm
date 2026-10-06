import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
original=OUT.parent/'inputs/current/docs/course-experiments/results/tokenizer.json'
raw=original.read_bytes()
result=json.loads(raw)
top_keys={k:type(v).__name__ for k,v in result.items()}
results_types={k:type(v).__name__ for k,v in result['results'].items()}
def pointer(path):
 value=result
 for part in path.split('/')[1:]:value=value[int(part)] if isinstance(value,list) else value[part]
 return value
selected={}
names=('byte256','bpe512')
for name in names:
 for field in ('records','effective_tokens','raw_utf8_bytes','mean_token_nll','bpb_including_eos_boundary_targets'):
  key=f'/results/runs/{name}/after/test/{field}'
  selected[key]=pointer(key)
 for field in ('training_raw_utf8_bytes_exposed',):
  key=f'/results/runs/{name}/{field}'
  selected[key]=pointer(key)
 for field in ('steps',):
  key=f'/results/runs/{name}/training/{field}'
  selected[key]=pointer(key)
for field in ('records','families','sha256'):
 for split in ('train','validation','test'):
  key=f'/results/data/{split}/{field}'
  selected[key]=pointer(key)
for name in ('scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/data.py','tiny_perceptron/model.py'):
 expected=result['code_sha256'][name]
 actual=hashlib.sha256((OUT.parent/'inputs/historical'/name).read_bytes()).hexdigest()
 assert expected==actual
 selected[f'/code_sha256/{name}']={'recorded':expected,'actual_snapshot':actual,'match':True}
for field in ('text','ids','restored','same','contains_control_id'):
 key=f'/results/roundtrip/2/{field}'
 selected[key]=pointer(key)
assert selected['/results/roundtrip/2/text']=='<user>這只是引用文字'
assert selected['/results/roundtrip/2/restored']==selected['/results/roundtrip/2/text']
assert selected['/results/roundtrip/2/contains_control_id'] is False
observations={}
for name in names:
 prefix=f'/results/runs/{name}/after/test/'
 records=selected[prefix+'records']
 count=selected[prefix+'effective_tokens']
 raw_bytes=selected[prefix+'raw_utf8_bytes']
 mean=selected[prefix+'mean_token_nll']
 bpb=selected[prefix+'bpb_including_eos_boundary_targets']
 computed=mean*count/(raw_bytes*math.log(2))
 assert records==20 and raw_bytes==4193
 assert count=={'byte256':4213,'bpe512':2503}[name]
 assert math.isclose(computed,bpb,rel_tol=0,abs_tol=1e-12)
 assert f'{mean:.5f}'=={'byte256':'2.77947','bpe512':'3.98741'}[name]
 assert f'{bpb:.5f}'=={'byte256':'4.02906','bpe512':'3.43401'}[name]
 assert selected[f'/results/runs/{name}/training_raw_utf8_bytes_exposed']==664759
 assert selected[f'/results/runs/{name}/training/steps']==400
 observations[name]={'original_records':records,'original_effective_tokens_including_EOS':count,'raw_UTF8_bytes_without_EOS':raw_bytes,'mean_token_nll':mean,'reported_BPB':bpb,'recomputed_BPB_from_original_aggregates':computed,'BPB_absolute_difference':abs(computed-bpb),'five_decimal_mean_label':f'{mean:.5f}','five_decimal_BPB_label':f'{bpb:.5f}'}
# Only arithmetic/inspection of original saved aggregates; no model score or original CPU demonstration rerun.
small={'raw_bytes':len('貓🙂'.encode('utf-8')),'A':5/(7*math.log(2)),'B':4/(7*math.log(2)),'smile_unicode':'U+%04X'%ord('🙂')}
assert small['raw_bytes']==7 and small['smile_unicode']=='U+1F642'
assert f'{small["A"]:.4f}'=='1.0305' and f'{small["B"]:.4f}'=='0.8244'
receipt={'original_result_path':original.relative_to(ROOT).as_posix(),'original_result_sha256':hashlib.sha256(raw).hexdigest(),'top_level_key_types':top_keys,'results_key_types':results_types,'actual_inspected_pointers':selected,'original_test_aggregate_checks':observations,'current_shared_denominator_figure_arithmetic':small,'tolerance':'exact integer/ID/label equality; five-decimal andfour-decimal rounded figures checked; recomputed historicalaggregate BPB abs<=1e-12','scope':'Narrow necessary context aggregate/figure audit only; original results/code/inputs retained intact, no notes/review/*_scope_correction author evaluation read, no training, weights, scoring, data preparation, or rerun of original CPU demonstrations.'}
(OUT/'measurement-pointer-check-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'observations':observations,'figure_arithmetic':small,'inspected_pointer_count':len(selected),'original_result_sha256':receipt['original_result_sha256']},ensure_ascii=False,indent=2))
