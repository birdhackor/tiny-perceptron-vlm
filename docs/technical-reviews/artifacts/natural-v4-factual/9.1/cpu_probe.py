from pathlib import Path
import re, sys, io, contextlib, hashlib, json, itertools
import torch
root=Path("docs/technical-reviews/artifacts/natural-v4-factual/9.1")
section=(root/"section.raw.md").read_text(encoding="utf-8")
code=re.findall(r"```python\n(.*?)```",section,re.S)[0]
pred=json.loads((root/"predictions-before-execution.json").read_text())
print("environment",json.dumps({"python":sys.version,"torch":torch.__version__,"device":"cpu","cuda_available":torch.cuda.is_available()},ensure_ascii=False))
print("input_sha256",hashlib.sha256(code.encode()).hexdigest())
for name,text in [("original",code),("exercise",code.replace('"回答0安全": False','"回答0安全": True'))]:
    ns={};out=io.StringIO()
    with contextlib.redirect_stdout(out):exec(compile(text,"lesson-9.1-"+name,"exec"),ns)
    result=out.getvalue()
    print(name+"_stdout",repr(result))
    assert result==pred[name+"_stdout"]
    idx=ns["pair"]["相對較安全"];key=f"回答{idx}安全"
    print(name+"_selected",json.dumps({"index":idx,"key":key,"value":ns["pair"][key],"pair_has_helpfulness":"有用" in ns["pair"],"pair_has_honesty":"誠實" in ns["pair"]},ensure_ascii=False))
    assert idx==0 and key=="回答0安全"
    assert ("有用" not in ns["pair"]) and ("誠實" not in ns["pair"])
    assert ns["record"]=={"提問":"盒中有幾顆球？","回答":"目前沒有數量資訊，請提供球數。","有用":True,"誠實":True,"安全":True}
# Verify literal values are supplied by humans, not inferred from response text.
changed=code.replace('"回答": "目前沒有數量資訊，請提供球數。"','"回答": "UNRELATED PROBE TEXT"')
ns={}
with contextlib.redirect_stdout(io.StringIO()):exec(changed,ns)
print("changed_response_labels",[ns["record"][k] for k in ["有用","誠實","安全"]])
assert all(ns["record"][k] for k in ["有用","誠實","安全"])
# Same-quote reuse is supported by the actual Python runtime since 3.12.
print("same_quote_f_string",eval('f"回答{dict(相對較安全=0)["相對較安全"]}安全"'))
all_fail=[list(v) for v in itertools.product([False,True],repeat=3) if not all(v)]
print("overall_false_criterion_vectors",json.dumps(all_fail))
print("probe_scope",json.dumps({"original_inputs":1,"exercise_inputs":1,"response_only_inputs":1,"seeds":"not applicable: deterministic","data_split":"none: manually assigned dictionaries","training_updates":0,"model_inference":False,"external_dataset_loaded":False,"gpu_used":False,"timing_scope":"no performance benchmark"}))
print("assertions_passed")
