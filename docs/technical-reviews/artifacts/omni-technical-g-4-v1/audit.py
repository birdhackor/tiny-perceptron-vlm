"""G.4 reviewer-owned deterministic link/formula audit; never trains a model."""
import ast
import contextlib
import io
import hashlib
import json
import math
import platform
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"
spec = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "G.4")
body = (ROOT / spec["snapshot"]).read_text()
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(body.encode()) == spec["source_sha256"]
canonical = (ROOT / spec["source"]).read_text()
match = re.search(r"^## G\.4\b.*?(?=^## |\Z)", canonical, re.M | re.S)
assert match and match.group() == body
assert not re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
assert spec["figures_sha256"] == {}

links = []
for m in re.finditer(r"(?<!!)\[([^]]+)\]\(([^)]+)\)", body):
    label, href = m.groups()
    rel, _, fragment = href.partition("#")
    target = ROOT / "course" / rel
    text = target.read_text()
    if fragment:
        sec = re.search(rf"^## {re.escape(fragment)}(?:\s|$).*?(?=^## |\Z)", text, re.M | re.S)
        assert sec, (label, href)
        excerpt = sec.group()
        start = text[:sec.start()].count("\n") + 1
    else:
        excerpt = text[:text.find("\n## ")] if "\n## " in text else text
        start = 1
    links.append({"label": label, "href": href, "source_line": body[:m.start()].count("\n")+1,
                  "target": str(target.relative_to(ROOT)), "fragment": fragment,
                  "target_start_line": start, "target_end_line": start+excerpt.count("\n"),
                  "target_heading": excerpt.splitlines()[0], "slice_sha256": sha(excerpt.encode())})

index = body.split("### 英文與縮寫索引", 1)[1]
normalize = lambda s: "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if c.isalnum())
index_entries = []
for line in index.splitlines():
    if not line.startswith("**"):
        continue
    letter, content = re.match(r"\*\*([A-Z])\*\*\s*(.*)", line).groups()
    for item in content.split(" · "):
        term = "decode" if item.startswith("decode（") else re.match(r"\[([^]]+)\]", item).group(1)
        index_entries.append({"letter": letter, "term": term, "normalized": normalize(term),
                              "hrefs": re.findall(r"\]\(([^)]+)\)", item)})
names = [item["normalized"] for item in index_entries]
ordering = [{"before": a["term"], "after": b["term"]} for a, b in zip(index_entries, index_entries[1:])
            if a["normalized"] > b["normalized"]]
duplicates = [term for term in sorted(set(names)) if names.count(term)>1]

def softmax(xs, t):
    peak = max(xs)
    vals = [math.exp((x-peak)/t) for x in xs]
    return [x/sum(vals) for x in vals]

p, q = [0.8, 0.2], [0.5, 0.5]
kl = sum(a*(math.log(a)-math.log(b)) for a,b in zip(p,q))
entropy = -sum(a*math.log(a) for a in p)
same_kl = sum(a*(math.log(a)-math.log(a)) for a in p)
concentration = (0.6-1/3)/(1-1/3)
temperature = {str(t): {name: max(softmax(v,t)) for name,v in {"A":[1,0,0],"B":[0.5,0,-10]}.items()}
               for t in (1,0.1)}
assert same_kl == 0 and abs(concentration-0.4)<1e-15
assert temperature["1"]["A"] < temperature["1"]["B"]
assert temperature["0.1"]["A"] > temperature["0.1"]["B"]

# Read and execute the *textbook's* C.6 scoring function, not the external trainer.
c6 = re.search(r"^## C\.6\b.*?(?=^## |\Z)", (ROOT/"course/chapters/0C.md").read_text(), re.M|re.S).group()
c6_blocks = re.findall(r"```python\n(.*?)\n```", c6, re.S)
scoring = c6_blocks[0]
scoring_result = {"textbook_block":scoring,"textbook_block_sha256":sha(scoring.encode())}
for variant,code in [("strict",scoring),("loose_exercise",scoring.replace('answers = ["3", "4", "5", "答案是4"]',
    'answers = ["3", "4", "5", "答案是4", "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"]')
    .replace('reward = float(parsed == truth)','reward = float(str(truth) in text.split())'))]:
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(compile(code,"course/chapters/0C.md#C.6","exec"),{})
    scoring_result[variant]=stream.getvalue()

# Each index term's actual target definition is read separately by the reviewer.
bindings=[]
aliases={"accepted risk":"接受風險","hybrid":"DeltaNet","codec":"編碼（encode）",
 "optimizer state":"最佳化器狀態","padding":"PAD","recurrent state":"循環層",
 "shared memory":"state-only encoder","source holdout":"新來源","support interval":"支持區間",
 "timestamp":"時間戳","tool call":"工具請求"}
for item in index_entries:
    for href in item["hrefs"]:
        rel,_,fragment=href.partition("#")
        target=ROOT/"course"/rel
        text=target.read_text()
        sec=re.search(rf"^## {re.escape(fragment)}(?:\s|$).*?(?=^## |\Z)",text,re.M|re.S).group() if fragment else text[:text.find("\n## ")]
        term=item["term"]
        needle=re.search(re.escape(term),sec,re.I)
        lookup=term if needle else aliases.get(term,term)
        paras=[p for p in sec.split('\n\n') if re.search(re.escape(lookup),p,re.I)]
        def rank(p):
            if p.startswith(('```','![','#')):return -5
            return (2 if '**' in p else 0)+(1 if re.search('叫|稱|指|是',p) else 0)
        context=max(paras,key=rank) if paras else sec[:400]
        bindings.append({"term":term,"href":href,"literal_term_found":bool(needle),"heading":sec.splitlines()[0],
                         "matched_term_or_chinese_alias":lookup,"definition_context":context[:900],"slice_sha256":sha(sec.encode())})
(OUT/"index-target-bindings.json").write_text(json.dumps(bindings,ensure_ascii=False,indent=2)+"\n")

# Direct bounded software checks of the textbook's generation/tool functions.
from types import SimpleNamespace
from tiny_perceptron.model import generate
from tiny_perceptron.retrieval import call_tool
class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(max_length=30)
    def forward(self,ids,cache=None):
        scores=torch.zeros(*ids.shape,4)
        scores[...,3]=1
        return {"logits":scores,"cache":None}
prompt=torch.tensor([[1,3,1]])
generated=generate(ConstantModel(),prompt,max_new_tokens=2,eos_id=2)
assert generated.tolist()==[[1,3,1,3,3]]
tool_results=[]
for request in ['{"name":"add","arguments":{"a":2,"b":3}}',
                '{"name":"delete_all","arguments":{"a":2,"b":3}}',
                '{"name":"add","arguments":{"a":true,"b":3}}']:
    try:tool_results.append({"input":request,"output":call_tool(request)})
    except ValueError as e:tool_results.append({"input":request,"error":str(e)})

result = {"created_at":datetime.now(UTC).isoformat(),"command":"uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-g-4-v1/audit.py",
          "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","platform":platform.platform()},
          "source":{"path":spec["snapshot"],"sha256":sha(body.encode()),"matches_canonical_slice":True,"figures":{}},
          "link_count":len(links),"links":links,"index_count":len(index_entries),"index_entries":index_entries,
          "index_order_violations":ordering,"index_duplicates":duplicates,
          "numeric":{"inputs":{"p":p,"q":q,"choice_pmax":0.6,"n":3},"kl_pq":kl,"entropy_p":entropy,
                     "kl_pp":same_kl,"choice_concentration":concentration,
                     "coverage_example":{"all":10,"accepted":4,"wrong_accepted":1,"coverage":4/10,"risk":1/4,"zero_accepted_risk":"undefined"},
                     "calibration_0.8":{"reported":0.8,"illustrative_correct":8,"illustrative_count":10,"frequency":8/10},
                     "temperature_rank_counterexample":temperature},"c6_textbook_execution":scoring_result,
          "software":{"generation":{"input":prompt.tolist(),"max_new_tokens":2,"output":generated.tolist()},"tool_calls":tool_results}}
(OUT/"audit-output.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("created_at","environment","source","link_count","index_count","index_order_violations","index_duplicates","numeric","c6_textbook_execution")},ensure_ascii=False,indent=2))
