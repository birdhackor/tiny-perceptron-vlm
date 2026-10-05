"""Bounded CPU review: exact fence variants and original data/result audit."""
import ast
import contextlib
import hashlib
import io
import itertools
import json
import math
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from scripts.course_experiments.common import split_records, text_examples, records_sha256
from scripts.course_experiments.text import _deduplicate_text
from tiny_perceptron.data import IGNORE, load_jsonl

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")

def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def input_path(rel):return OUT/"inputs"/rel

environment={"python":platform.python_version(),"torch":str(torch.__version__),"torch_git_version":str(torch.version.git_version),"device":"cpu","cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),"threads":str(torch.get_num_threads()),"scope":"No model creation, no backward/optimizer, no training, no network; data processing and prior-result audit only."}
save(OUT/"probe-environment.json",environment)

code=(OUT/"original/fence-1.py").read_text()
variants=[]
for name,record,expected in (
    ("original",None,0.01),
    ("exercise-low",{"lr":0.1,"validation":0.6},0.01),
    ("exercise-high",{"lr":0.1,"validation":0.9},0.1),
    ("tie-preserves-first-best",{"lr":0.1,"validation":0.8},0.01),
):
    source=code if record is None else code.replace(
        '{"lr": 0.01, "validation": 0.8},',
        '{"lr": 0.01, "validation": 0.8},\n    '+repr(record)+','
    )
    target=OUT/f"variant-{name}.py";target.write_text(source)
    stream=io.StringIO();ns={}
    with contextlib.redirect_stdout(stream):exec(compile(source,str(target),"exec"),ns)
    assert ns["selected"]["lr"]==expected
    assert stream.getvalue().splitlines()[-1]=="test仍封存，尚未計算最終成績"
    assert "test" not in ns
    tree=ast.parse(source)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree))
    item={"name":name,"sha256":sha(source.encode()),"selected":ns["selected"],"stdout":stream.getvalue(),"no_test_value_computed":True}
    variants.append(item);print(json.dumps(item,ensure_ascii=False))

# Exact enumeration illustrates selection optimism under explicit independent
# +/-0.1 score noise around the same true success rate 0.5. It is not a model run.
selection=[]
for count in (1,2,4,8):
    cases=list(itertools.product((0.4,0.6),repeat=count))
    maximums=[max(scores) for scores in cases]
    expected_max=sum(maximums)/len(maximums)
    probability_lucky=sum(m>0.5 for m in maximums)/len(maximums)
    assert math.isclose(probability_lucky,1-2**(-count),abs_tol=1e-12)
    assert math.isclose(expected_max,0.6-0.2*(2**(-count)),abs_tol=1e-12)
    selection.append({"candidates":count,"equally_likely_cases":len(cases),"true_success_each":0.5,"expected_selected_validation":expected_max,"probability_at_least_one_lucky":probability_lucky})

result=json.loads(input_path("docs/course-experiments/results/real_text.json").read_text())
assert result["revision"]=="5581462ef01959636425eb7795ae3153142dbfeb"
assert result["seed"]==42 and result["step_scale"]==1.0 and result["status"]=="completed"
matched_code={}
for rel in ("scripts/course_experiments/text.py","scripts/course_experiments/common.py","tiny_perceptron/data.py"):
    observed=sha(input_path(rel).read_bytes());expected=result["code_sha256"][rel]
    assert observed==expected
    matched_code[rel]=observed

runs={}
for identifier,basename,expected_counts in (
    ("tinystories","tinystories-train-512.jsonl",{"train":409,"validation":51,"test":52}),
    ("chinese-poetry","chinese-classical-train-365.jsonl",{"train":292,"validation":36,"test":37}),
):
    raw=load_jsonl(input_path("data/training/text-initial/"+basename))
    complete=_deduplicate_text(raw)
    parts=split_records(complete,seed=42)
    report=result["results"]["runs"][identifier]
    assert report["source_records"]==len(raw)
    assert report["deduplicated_records"]==len(complete)
    sets={key:{r["family"] for r in rows} for key,rows in parts.items()}
    for a,b in itertools.combinations(sets,2):assert not sets[a]&sets[b]
    split_audits={}
    for split,rows in parts.items():
        raw_split="".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
        saved=OUT/"reconstructed-splits"/(identifier+"-data")/(split+".jsonl")
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(raw_split)
        expected=report["data"][split]
        assert sha(raw_split)==expected["sha256"]
        assert len(rows)==expected["records"]==expected_counts[split]
        assert len(sets[split])==expected["families"]
        examples=text_examples(rows,mode="text",max_length=128)
        effective=sum(int((y!=IGNORE).sum()) for _,y in examples)
        raw_utf8=sum(len(r["text"].encode()) for r in rows)
        assert effective==raw_utf8+len(rows)
        if split in ("validation","test"):
            for phase in ("before","after"):
                m=report[phase][split]
                assert m["records"]==len(rows)
                assert m["examples"]==len(examples)
                assert m["effective_tokens"]==effective
                assert m["raw_utf8_bytes"]==raw_utf8
                assert math.isclose(m["nll"],m["nll_sum"]/effective,rel_tol=1e-12)
                assert math.isclose(m["bpb_including_eos_boundary_targets"],m["nll_sum"]/(raw_utf8*math.log(2)),rel_tol=1e-12)
                assert len(m["samples"])==min(8,len(rows))
        else:
            assert report["training"]["records"]==len(rows)
            assert report["training"]["records_sha256"]==records_sha256(rows)
            assert report["training"]["steps"]==800
        split_audits[split]={"records":len(rows),"families":len(sets[split]),"sha256":sha(raw_split),"windows":len(examples),"effective_tokens_per_full_pass":effective,"raw_utf8_bytes":raw_utf8}
    assert {r["family"] for r in complete}==set.union(*sets.values())
    runs[identifier]={"source_records":len(raw),"deduplicated_records":len(complete),"disjoint_full_document_families":True,"split_audits":split_audits,"optimizer_input_train_sha256":report["training"]["records_sha256"],"source_revision":report["source_revision"],"warning":report["split_warning"],"test_is_local_holdout_not_official":True}

# Independently compare retained complete stories with original bounded raw prefix.
prefix=input_path("data/training/text-initial/tinystories-train-prefix-complete.txt").read_text()
stories=[x.strip() for x in prefix.split("<|endoftext|>") if x.strip()]
story_rows=load_jsonl(input_path("data/training/text-initial/tinystories-train-512.jsonl"))
assert [r["text"] for r in story_rows]==stories
assert len(stories)==512
assert all(r["source_split"]=="train" for r in story_rows)
assert all(r["source_revision"]=="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64" for r in story_rows)
assert sha((OUT/"sources/tinystories-f54c09f-README.md").read_bytes())==sha(input_path("data/training/text-initial/tinystories-source-README.md").read_bytes())

# The upstream poetry anthology consists of complete poem objects, not split data.
anthology=json.loads(input_path("data/training/text-initial/tang300-source.json").read_text())
assert all(not ({"split","train","validation","test"}&set(r)) for r in anthology)
rendered=[r["title"]+"\n"+r["author"]+"\n"+"\n".join(r["paragraphs"]) for r in anthology]
poems=load_jsonl(input_path("data/training/text-initial/chinese-classical-train-365.jsonl"))
assert [r["text"] for r in poems]==list(dict.fromkeys(rendered))
assert len(anthology)==366 and len(poems)==365

# Concrete bounded group variation: descendants stay in one split.
toy=[{"family":f"story-{i}","window":j,"text":f"story {i}, window {j}"} for i in range(6) for j in range(3)]
grouped=split_records(toy,seed=42)
membership={}
for split,rows in grouped.items():
    for row in rows:membership.setdefault(row["family"],set()).add(split)
assert all(len(v)==1 for v in membership.values())

audit={"scope":"Existing historical CUDA results audited with exact input identities and matching original code; no retraining and no model inference. Historical evaluations contain before-test and after-test; no claim of first/only untouched test evaluation is made.","result_revision":result["revision"],"result_sha256":sha(input_path("docs/course-experiments/results/real_text.json").read_bytes()),"matched_original_code_sha256":matched_code,"original_fence_variants":variants,"selection_noise_enumeration":selection,"runs":runs,"complete_story_prefix_matches":True,"poetry_complete_objects":len(anthology),"poetry_unique_complete_objects":len(poems),"group_descendants_disjoint":True,"historical_training_device":result["device"],"historical_training_torch":result["torch_version"],"historical_training_python":result["python_version"],"denominator_policy":"One full text pass predicts every UTF-8 byte plus one EOS per complete record. Window counts are separate from record counts. Aggregate NLL is token weighted, samples stores only first eight text generations."}
save(OUT/"probe-results.json",audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
