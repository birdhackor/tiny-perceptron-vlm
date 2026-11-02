from pathlib import Path
from collections import Counter
import contextlib, io, json, sys, platform, hashlib, re
snapshot=Path("docs/course-review-20261019/freeze-01/sources/20.11.md")
raw=snapshot.read_bytes()
code=re.search(r"```python\n(.*?)```",raw.decode(),re.S).group(1)
namespace={}
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code,str(snapshot)+":python_block_1","exec"),namespace)
reference=namespace["reference"]
predicted=namespace["predicted"]
exercise_code=code.replace('predicted = "麵包\\n牛奶"','predicted = "牛奶麵包"')
assert exercise_code != code
exercise={}
exercise_stdout=io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code,str(snapshot)+":exercise","exec"),exercise)
normalize=lambda s: "".join(s.split())
result={
 "environment":{"python":sys.version,"implementation":platform.python_implementation(),"device":"CPU; Python strings only"},
 "source_sha256":hashlib.sha256(raw).hexdigest(),
 "embedded_code_sha256":hashlib.sha256(code.encode()).hexdigest(),
 "original":{"stdout":stdout.getvalue(),"reference_repr":repr(reference),"predicted_repr":repr(predicted),"reference_len":len(reference),"predicted_len":len(predicted),"reference_codepoints":[f"U+{ord(c):04X}" for c in reference],"predicted_codepoints":[f"U+{ord(c):04X}" for c in predicted],"reference_counter":dict(Counter(reference)),"predicted_counter":dict(Counter(predicted))},
 "exercise":{"stdout":exercise_stdout.getvalue(),"predicted_repr":repr(exercise["predicted"]),"predicted_len":len(exercise["predicted"]),"predicted_counter":dict(Counter(exercise["predicted"])),"predicted_lines":exercise["predicted"].splitlines()},
 "whitespace_removal":{"method":"''.join(s.split())","reference":normalize(reference),"exercise":normalize(exercise["predicted"]),"equal_after_removing_whitespace":normalize(reference)==normalize(exercise["predicted"]),"swapped_lines_still_unequal":normalize(reference)!=normalize(predicted)},
 "permutation_loss":{"sort_reference_lines":sorted(reference.splitlines()),"sort_predicted_lines":sorted(predicted.splitlines()),"equal_after_sorting":sorted(reference.splitlines())==sorted(predicted.splitlines())}
}
assert stdout.getvalue()=="字元種類與數量相同 True\n完整字串相同 False\n正確答案的行序 ['牛奶', '麵包']\n預測的行序 ['麵包', '牛奶']\n"
assert exercise_stdout.getvalue()=="字元種類與數量相同 False\n完整字串相同 False\n正確答案的行序 ['牛奶', '麵包']\n預測的行序 ['牛奶麵包']\n"
assert result["whitespace_removal"]["equal_after_removing_whitespace"] is True
assert result["whitespace_removal"]["swapped_lines_still_unequal"] is True
assert result["permutation_loss"]["equal_after_sorting"] is True
print(json.dumps(result,ensure_ascii=False,indent=2))
