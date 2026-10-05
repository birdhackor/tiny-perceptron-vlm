"""Independent bounded probes using the exact original fence's merge function."""
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
import io
import itertools
import json
import platform
import sys

OUT=Path(__file__).resolve().parent
fence=OUT/"original/fence-1.py"
namespace={}
original_stdout=io.StringIO()
with redirect_stdout(original_stdout): exec(compile(fence.read_bytes(),str(fence),"exec"),namespace)
merge=namespace["merge"]
def count(corpus):
 return Counter(pair for row in corpus for pair in zip(row,row[1:]))
def pairs_json(pairs):
 return [{"pair":list(pair),"count":n} for pair,n in pairs.items()]
def example(corpus):
 pairs=count(corpus); pair=pairs.most_common(1)[0][0]
 outputs=[merge(row,pair) for row in corpus]
 assert all("".join(row)=="".join(out) for row,out in zip(corpus,outputs))
 return {"input":corpus,"pairs":pairs_json(pairs),"selected":list(pair),"output":outputs,
         "input_token_counts":[len(row) for row in corpus],"output_token_counts":[len(row) for row in outputs],
         "letters":[len("".join(row)) for row in corpus],"reconstruction_equal":True}
base=example([list("abab"),list("abac")])
assert base["selected"]==["a","b"] and base["output"]==[["ab","ab"],["ab","a","c"]]
exercise=example([list("abab"),list("abac"),list("acacacac")])
assert exercise["selected"]==["a","c"] and exercise["output"]==[list("abab"),["a","b","ac"],["ac"]*4]
ties=[example([list("aba")]),example([list("bab")])]
assert [item["selected"] for item in ties]==[["a","b"],["b","a"]]
overlap=example([list("aaaa")])
assert overlap["pairs"]==[{"pair":["a","a"],"count":3}] and overlap["output"]==[["aa","aa"]]
boundaries=[[],["a"],["b"],[]]
assert not count(boundaries)
assert [merge(row,("a","b")) for row in boundaries]==boundaries
edge_cases={"empty":merge([],("a","b")),"single":merge(["a"],("a","b")),
 "nonmatching":merge(list("acac"),("a","b")),"pair_at_end":merge(list("cab"),("a","b")),
 "already_merged":merge(["ab","ab"],("ab","ab"))}
assert edge_cases=={"empty":[],"single":["a"],"nonmatching":list("acac"),"pair_at_end":["c","ab"],"already_merged":["abab"]}
exhaustive=0
for length in range(9):
 for row in itertools.product("ab",repeat=length):
  for pair in itertools.product("ab",repeat=2):
   output=merge(list(row),pair)
   # Oracle finds all matches independently and removes overlaps in positional order.
   candidate_positions=[i for i in range(max(0,len(row)-1)) if tuple(row[i:i+2])==pair]
   starts=[]
   for i in candidate_positions:
    if not starts or i>starts[-1]+1: starts.append(i)
   assert "".join(output)=="".join(row)
   assert len(output)==len(row)-len(starts)
   exhaustive+=1
# This optional reviewer-only loop is not presented as the lesson's implementation.
rows=[list("abab"),list("abac")]
vocab=set(itertools.chain.from_iterable(rows)); merges=[]; loop=[]
while count(rows):
 pair=count(rows).most_common(1)[0][0]
 before=sum(map(len,rows)); rows=[merge(row,pair) for row in rows]
 vocab.add("".join(pair)); merges.append(list(pair)); after=sum(map(len,rows))
 assert after<before
 loop.append({"pair":list(pair),"output":rows,"total_positions_before":before,"total_positions_after":after,"vocabulary_size":len(vocab)})
assert merges==[["a","b"],["ab","ab"],["ab","a"],["aba","c"]]
result={"source_fence_sha256":sha256(fence.read_bytes()).hexdigest(),"original_stdout":original_stdout.getvalue(),
 "base":base,"exercise":exercise,"ties":ties,"overlap":overlap,"boundaries":{"input":boundaries,"pairs":pairs_json(count(boundaries)),"outputs":boundaries},
 "edge_cases":edge_cases,"exhaustive_checks":exhaustive,"finite_loop_illustration":loop,
 "scope":"Only short CPU checks; exhaustive rows length 0..8 over a,b and four character pairs. Independent illustration is not original full BPE training, not model training and not a benchmark.",
 "environment":{"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"device":"CPU; standard library only"}}
(OUT/"probe-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
