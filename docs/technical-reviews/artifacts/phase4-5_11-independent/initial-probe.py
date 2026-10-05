"""Bounded exact-fence variations and original JSON/data contract audit; no training."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from contextlib import redirect_stdout

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.data import fingerprint
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
environment={"python":platform.python_version(),"python_executable":sys.executable,
    "torch":str(torch.__version__),"cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),
    "device":"CPU","training":"No model creation, inference, backward or optimizer steps",
    "offline":"CUDA_VISIBLE_DEVICES empty; HF_HUB_OFFLINE/HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE=1"}
write("probe-environment.json",environment)
fence=(OUT/"original/fence-1.py").read_text()
variants=[("exact_original",fence,False),
    ("bird_old_assert",fence.replace('"貓 看狗"','"鳥 看狗"'),True),
    ("bird_corrected_assert",fence.replace('"貓 看狗"','"鳥 看狗"').replace('hashes[0] == hashes[1]','hashes[0] != hashes[1]'),False),
    ("extra_whitespace",fence.replace('"貓 看狗"',repr("貓 \t\n看狗")),False)]
variations=[]
for name,code,expect_assert in variants:
    p=OUT/"variations"/(name+".py"); p.parent.mkdir(exist_ok=True); p.write_text(code)
    namespace={}; stream=io.StringIO(); raised=False
    try:
        with redirect_stdout(stream): exec(compile(code,str(p),"exec"),namespace)
    except AssertionError: raised=True
    assert raised==expect_assert,(name,raised)
    variations.append({"name":name,"code_path":str(p.relative_to(ROOT)),"code_sha256":sha(p.read_bytes()),
        "assertion_error":raised,"expected_assertion_error":expect_assert,"stdout":stream.getvalue(),
        "normalized":namespace["normalized"],"unique_hashes":len(set(namespace["hashes"])),
        "hashes":namespace["hashes"]})
assert [v["unique_hashes"] for v in variations]==[2,3,3,2]
edges={text:{"normalized":" ".join(text.split()),"hash":fingerprint(text)} for text in
    ("  貓\t看狗\n", "貓\u00a0看狗", "", " \t\n", "貓看狗", "貓 看狗", "貓 注視狗")}
assert edges["  貓\t看狗\n"]["hash"]==fingerprint("貓 看狗")
assert edges["貓\u00a0看狗"]["hash"]==fingerprint("貓 看狗")
assert fingerprint("")==fingerprint(" \t\n")
assert fingerprint("貓看狗")!=fingerprint("貓 看狗")
assert fingerprint("貓 注視狗")!=fingerprint("貓 看狗")
prefix_groups={}
for index in range(17):
    text=f"document {index}"; digest=fingerprint(text)
    prefix_groups.setdefault(digest[:1],[]).append({"text":text,"full_hash":digest})
pair=next(v[:2] for v in prefix_groups.values() if len(v)>1)
assert pair[0]["full_hash"]!=pair[1]["full_hash"]
bits_changed=(int(fingerprint("貓 看狗"),16)^int(fingerprint("鳥 看狗"),16)).bit_count()
python_text="if True:\n    value = 1\n"
compile(python_text,"original_python","exec")
normalized_python=" ".join(python_text.split())
try: compile(normalized_python,"normalized_python","exec")
except SyntaxError: compile_difference=True
else: compile_difference=False
assert compile_difference
duplicates=Counter(" ".join(t.split()) for t in ["貓  看狗","貓 看狗","狗 看貓"])
sampling={k:{"with_duplicates":str(Fraction(v,3)),"after_dedup":str(Fraction(1,2))} for k,v in duplicates.items()}
write("variation-results.json",{"environment":environment,"variations":variations,"edge_cases":edges,
    "changed_bits_one_character_example":bits_changed,"short_prefix_collision":pair,
    "whitespace_can_be_significant":{"python_original":python_text,"normalized":normalized_python,
        "normalizing_changes_parse":compile_difference,"string_value_original":"A  B","string_value_normalized":" ".join("A  B".split())},
    "uniform_record_sampling_example":sampling,"sampling_scope":"Exact probability over a uniformly sampled record, not a model score or universal sampler policy"})

record=json.loads((OUT/"inputs/docs/course-experiments/results/real_text.json").read_text())
manifest=json.loads((OUT/"inputs/assets/training/manifest.json").read_text())
revision=record["revision"]
historical={}
for relative in ("scripts/course_experiments/text.py","scripts/course_experiments/common.py"):
    command=["git","show",f"{revision}:{relative}"]
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=10,check=True)
    path=OUT/"historical"/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(completed.stdout)
    expected=record["code_sha256"][relative]
    assert sha(completed.stdout)==expected
    historical[relative]={"path":str(path.relative_to(ROOT)),"sha256":expected,"command_argv":command,
        "exit_code":completed.returncode,"stderr":completed.stderr.decode(),
        "matches_current_snapshot":sha((OUT/"inputs"/relative).read_bytes())==expected}
def original_function(relative,name):
    raw=(OUT/"historical"/relative).read_text(); tree=ast.parse(raw)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    source="".join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    (OUT/"historical"/(name+".py")).write_text(source)
    namespace={"hashlib":hashlib,"json":json,"random":__import__("random")}
    exec(compile(source,f"{revision}:{relative}:{node.lineno}","exec"),namespace)
    return namespace[name]
deduplicate=original_function("scripts/course_experiments/text.py","_deduplicate_text")
split=original_function("scripts/course_experiments/common.py","split_records")
audits={}
for identifier,basename in (("tinystories","tinystories-train-512.jsonl"),("chinese-poetry","chinese-classical-train-365.jsonl")):
    path=OUT/"inputs/data/training/text-initial"/basename; raw_bytes=path.read_bytes()
    asset=next(a for a in manifest["assets"] if a["id"]==identifier)
    file_spec=next(f for f in asset["files"] if f["path"].endswith("/"+basename))
    assert sha(raw_bytes)==file_spec["sha256"] and len(raw_bytes)==file_spec["bytes"]
    rows=[json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    clean=deduplicate(rows); parts=split(clean,seed=record["seed"])
    original=record["results"]["runs"][identifier]
    assert len(rows)==original["source_records"]==asset["training_records"]
    assert len(clean)==original["deduplicated_records"]
    assert [r["text"] for r in clean]==[r["text"] for r in rows]
    assert len(set(" ".join(r["text"].split()) for r in rows))==len(rows)
    assert all(r["family"]==fingerprint(r["text"]) for r in clean)
    split_audit={}
    for name,part in parts.items():
        rendered="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in part).encode()
        expected=original["data"][name]
        actual={"records":len(part),"families":len(set(r["family"] for r in part)),"sha256":sha(rendered)}
        assert all(actual[k]==expected[k] for k in actual),(identifier,name,actual,expected)
        split_audit[name]=actual
    families={name:set(r["family"] for r in part) for name,part in parts.items()}
    assert not families["train"]&families["validation"]
    assert not families["train"]&families["test"]
    assert not families["validation"]&families["test"]
    audits[identifier]={"input_sha256":sha(raw_bytes),"source_records":len(rows),"deduplicated_records":len(clean),
        "removed":len(rows)-len(clean),"split_audit":split_audit,"cross_split_family_overlap":0,
        "records_with_newlines":sum("\n" in r["text"] for r in rows),"text_values_exactly_preserved":True,
        "near_duplicate_clustering_performed":False,"source_revision":original["source_revision"]}
write("data-audit-results.json",{"original_report_sha256":sha((OUT/"inputs/docs/course-experiments/results/real_text.json").read_bytes()),
    "original_experiment_revision":revision,"original_device":record["device"],"original_torch":record["torch_version"],
    "historical_code":historical,"audit_environment":environment,"data":audits,
    "scope":"Recomputed cleaning/splitting identities and counts against the original saved report. No model scores, inference, training or near-duplicate detection."})
print(json.dumps({"variations":[{"name":v["name"],"unique_hashes":v["unique_hashes"],"assertion_error":v["assertion_error"]} for v in variations],
    "data":audits,"historical_code":historical,"all_assertions_passed":True},ensure_ascii=False,indent=2))
