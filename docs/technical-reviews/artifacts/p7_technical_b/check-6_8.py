import json,platform
from tiny_perceptron.data import shifted
ids=list(range(10));rows=[]
for start in [0,3,6]:
 segment=ids[start:start+4];x,y=shifted(segment);assert len(x)==len(y)==3
 rows.append({'start':start,'segment':segment,'x':x.tolist(),'y':y.tolist()})
assert [i for r in rows for i in r['y']]==list(range(1,10))
failed=[]
for start in [0,3,6]:
 x,y=shifted(ids[start:start+5]);failed.append({'start':start,'x':x.tolist(),'y':y.tolist(),'length':len(x),'length_assert_passes':len(x)==len(y)==4})
assert [r['length_assert_passes'] for r in failed]==[True,True,False]
fixed=[]
for start in [0,4]:
 segment=ids[start:start+5];x,y=shifted(segment);assert len(x)==len(y)==4
 fixed.append({'start':start,'segment':segment,'x':x.tolist(),'y':y.tolist()})
print(json.dumps({'python':platform.python_version(),'device':'cpu','original_context3':rows,'context4_original_starts':failed,'context4_fixed_starts':fixed,'byte256_excerpt_max_shifted_input_including_bos_eos':256+1,'model_max_length':384,'no_tokenizer_change':True},ensure_ascii=False,indent=2))

