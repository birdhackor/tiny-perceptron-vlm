"""Write this reviewer's original technical assessment from preserved evidence."""
import hashlib
import json
import shlex
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
REL = BASE.relative_to(ROOT).as_posix()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

env0 = json.loads((BASE / "original-environment.json").read_text())
env = {k: str(env0[k]) for k in ("python", "torch", "torch_git_version", "cuda_build", "cuda_available", "device_requested")}
checks = json.loads((BASE / "cpu-checks.json").read_text())
original = json.loads((BASE / "original-execution.json").read_text())
acquisition = json.loads((BASE / "source-acquisition.json").read_text())
extract = json.loads((BASE / "original-extraction.json").read_text())

inspection = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_7_2",
    "scope": "Independent correctness review of current 7.2 only; no old technical or reader reports were opened, no delegated agent was spawned, and no textbook or figure was edited.",
    "actually_read": ["course/chapters/07.md lines 1–220 (includes complete 7.1–7.6 and opening 7.7); 7.2 reread from exact frozen bytes, lines 40–71",
        "course/chapters/06.md 6.6 full and incidental lines 310–end of 6.9",
        "full factual-reviewer-instructions.md, section_facts.py, check_technical_reviews.py, clear-tutorial review-protocol.md",
        "full tiny_perceptron/data.py and model.py; scripts/evaluate.py lines 1–125; scripts/infer.py full; scripts/train.py prepare_examples/modal_loss; scripts/build_course.py BOOTSTRAP; pyproject.toml"],
    "chapter_intro": {"status": "not_applicable", "reason": "7.2 is not the chapter's first section; introductory lines were incidentally visible in the contextual read."},
    "figures": {"status": "not_applicable", "references": [], "rendered": False,
        "reason": "No image/SVG/HTML image is referenced in the exact section. The nine-item ID list directly exposes ordering; no undisclosed spatial claim needs a figure."},
    "historical_metrics": {"status": "not_applicable", "reason": "No model score, training result, dataset denominator, checkpoint identity or historical empirical value is claimed in 7.2."},
    "shell_contract": {"status": "not_applicable", "reason": "No shell fence exists. The training recipe is a navigation link, not required by this tokenizer-only demonstration; no training/data preparation/download recipe was run."},
    "official_sources_actually_inspected": [
        {"file": "hf-chat-v4.57.1.md", "locator": "lines 24–28 overview; Using apply_chat_template; add_generation_prompt; continue_final_message; Model training", "note": "Read the complete official Markdown. It explains role/content dictionaries becoming token sequences, distinct formats, generation header examples with old history, exceptions where no generation prompt is used, and matching training formatting. The EOS+assistant convention is local, not a rule for every chat model; the chapter calls it 本字表."},
        {"file": "hf-generation-v4.57.1.py", "locator": "lines 2782–2838, especially 2800 and 2831", "note": "Read the implementation selecting outputs.logits[:, -1, :] and appending argmax/sample. This supports the time-axis claim, not an arithmetic-correctness claim."},
        {"file": "python-3.13-stdtypes.html", "locator": "#common-sequence-operations, table s+t and s[i], note (3); #str.encode; #bytes", "note": "Read the actual official HTML paragraphs: sequence + concatenates, indices are zero-based and negative i means len(s)+i; str.encode emits bytes; bytes range is 0<=x<256. Derived ASCII offset values independently and checked them by execution."},
        {"file": "pytorch-sparse-v2.9.0.py", "locator": "Embedding docstring lines 17–43; Embedding.forward lines 191–200", "note": "Read the official source shape/lookup/learnable-weight contract and F.embedding invocation. Fixed upstream version is 2.9.0; installed CPU version is 2.14.1. The actual deterministic indexing probe checks the shared API contract, without claiming these versions are identical."}],
    "capability_limit": "Original fence constructs a list and asserts a boundary only. Deterministic Embedding and synthetic-logit stub are API/control-flow checks, not trained-model forward evaluations. No checkpoint loaded, no optimizer/backward/update, no arithmetic accuracy score.",
    "claim_inventory": "Role/content/control distinction; exact byte offset/IDs/count/list operations; last-position next-token rule; local EOS versus assistant semantics and training match; old-history versus current target; representation versus answer ability; missing/restored boundary exercise.",
    "issues": []
}
save(BASE / "inspection.json", inspection)
derivation = {
    "question": "1+1=?", "utf8_integer_bytes": [49,43,49,61,63],
    "offset": 8, "ordinary_ids": [57,51,57,69,71],
    "special_mapping": {"BOS":1,"user":3,"EOS":2,"assistant":4},
    "prompt_count": "1 BOS + 1 user + 5 ASCII bytes + 1 EOS + 1 assistant = 9 tokens",
    "prompt_positions": "zero-based 0..8; -1 resolves to len(prompt)-1 = 8; x[8]=4 predicts y[8]=58",
    "answer_vs_eos": "Answer character '2' is byte 50 + 8 = ID58. ID2 already in prompt is EOS, not the answer character.",
    "training_alignment": "Training raw sequence is prompt + [58,2]; x drops final EOS, y shifts once, giving x length10 and y[8]=58,y[9]=2.",
    "multi_turn": "BOS1 + user segment7 + old assistant segment3 + user segment7 + new assistant1 =19; new first answer target at index18, old first answer target at index8.",
    "units_axes": "All counts are tokens/byte IDs, no averages, probabilities, rates or metric denominators. Probe uses B=1,T=9,V=264, arbitrary H=4; [1,9,4] embedding and [1,9,264] synthetic logits. Numeric ID4 and position index8 are different quantities.",
    "tolerance": "Exact equality for integers, IDs, lengths and deterministic table values; no floating accuracy measurement."
}
save(BASE / "derivation.json", derivation)

executions = {
    "original-execution.json": (original["command"], "exit0, original fence attempted once; stdout exact nine-token prompt and last boundary4"),
    "original-stdout.txt": (original["command"], "[1,3,57,51,57,69,71,2,4]; 最後邊界4; exit0"),
    "original-stderr.txt": (original["command"], "empty stderr; original exit0"),
    "original-launch.json": (shlex.join(json.loads((BASE / "original-launch.json").read_text())["command_argv"]), "extract+execute timeout45s; exit0; artifacts copied into permanent docs"),
    "original-launch.stdout.txt": (".venv/bin/python " + REL + "/acquire_and_run.py", "helper reports original source SHA and execution_exit_code0"),
    "original-launch.stderr.txt": (".venv/bin/python " + REL + "/acquire_and_run.py", "empty launch stderr; original exit0"),
    "source-acquisition.json": (".venv/bin/python " + REL + "/acquire_and_run.py", "4 actual HTTPS official source fetches, status200, URL/bytes/SHA saved; no model/data downloads"),
    "cpu-checks.json": (".venv/bin/python " + REL + "/run_cpu_checks.py", "missing boundary rc1 expected; restored rc0; independent contract probe rc0; all bounded20s"),
}
for name, record in checks.items():
    for suffix in ("stdout.txt", "stderr.txt"):
        executions[name + "." + suffix] = (shlex.join(record["command_argv"]),
            f"exit{record['exit_code']} (expected{record['expected_exit_code']}); preserved {suffix}")
artifacts, ids = [], {}
for path in sorted(BASE.rglob("*")):
    if not path.is_file() or path.name.startswith("checker") or path.name == "manifest.json":
        continue
    local = path.relative_to(BASE).as_posix()
    identifier = "a-" + local.replace("/", "-").replace(".", "-")
    ids[local] = identifier
    kind = "code" if path.suffix == ".py" else "source_snapshot"
    if local == "derivation.json":
        kind = "derivation"
    artifact = {"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "kind": kind,
                "description": "7.2 independent review preserved actual " + local}
    if local in executions:
        artifact.update(kind="execution", command=executions[local][0], result=executions[local][1], environment=env)
    artifacts.append(artifact)

sources = []
for identifier, title, file, version, reason, note in [
    ("hf-chat", "Hugging Face Transformers official Chat templates", "hf-chat-v4.57.1.md", "Transformers v4.57.1 tagged source", "Maintainer-authored official documentation in huggingface/transformers.", inspection["official_sources_actually_inspected"][0]["note"]),
    ("hf-generate", "Hugging Face Transformers GenerationMixin token selection", "hf-generation-v4.57.1.py", "Transformers v4.57.1 tagged source", "Official maintained generation implementation in huggingface/transformers.", inspection["official_sources_actually_inspected"][1]["note"]),
    ("python", "Python 3.13 Built-in Types reference", "python-3.13-stdtypes.html", "Python3.13 official documentation, snapshot accessed2026-10-05", "Python Software Foundation official language/library reference at docs.python.org.", inspection["official_sources_actually_inspected"][2]["note"]),
    ("pytorch-embedding", "PyTorch nn.Embedding official implementation", "pytorch-sparse-v2.9.0.py", "PyTorch v2.9.0 official tagged source; installed probe version2.14.1+cpu", "Official maintained pytorch/pytorch source repository.", inspection["official_sources_actually_inspected"][3]["note"]),
]:
    source = {"id":identifier,"kind":"official_docs" if identifier in ("hf-chat","python") else "official_source",
              "title":title,"verified":True,"checked_original":True,"url":acquisition[file]["url"],"version":version,
              "accessed_on":"2026-10-05","authority_reason":reason,"inspection_note":note,
              "snapshot_artifact_id":ids["sources/"+file],"snapshot_sha256":acquisition[file]["sha256"]}
    sources.append(source)
for identifier, name, note in [
    ("data", "tiny_perceptron/data.py", "Read full actual ByteTokenizer/render_chat contract. SPECIALS count8, role from metadata, plain encode has no special-token string parser, labels shift once."),
    ("model", "tiny_perceptron/model.py", "Read full TinyLM embedding/position/output contract and generate; scores=result['logits'][:, -1] selects the last position. Not executing a trained TinyLM."),
    ("evaluate", "scripts/evaluate.py", "Read chat_prompt and evaluation prefix construction: old messages preserved, terminal current assistant target withheld via messages[:-1]. No evaluator was run on weights."),
    ("infer", "scripts/infer.py", "Read full --chat path with exact BOS,user,question,EOS,assistant order; no checkpoint or inference CLI run."),
]:
    sources.append({"id":identifier,"kind":"repository_code","title":name,"verified":True,"path":name,"sha256":sha(ROOT/name),
        "version":"HEAD022dc9b2ffde92c14ca133406849c1c378bb8e8f plus actual file SHA256",
        "inspection_note":note,"snapshot_artifact_id":ids["inputs/"+name]})
sources.extend([
    {"id":"derivation","kind":"derivation","title":"Independent 7.2 ID/count/position derivation","verified":True,"details":"See exact byte/offset arithmetic, token units and zero-based indices in "+REL+"/derivation.json"},
    {"id":"original-run","kind":"execution","title":"Original 7.2 Python fence actual CPU execution","verified":True,"artifact_id":ids["original-execution.json"]},
    {"id":"probe-run","kind":"execution","title":"Bounded independent serialization and API/control-flow probe","verified":True,"artifact_id":ids["contract.stdout.txt"]},
    {"id":"exercise-run","kind":"execution","title":"Actual missing/restored assistant exercises","verified":True,"artifact_id":ids["cpu-checks.json"]},
])
def evidence(source, locator, supports):
    return {"source_id":source,"locator":locator,"supports":supports}
def verification(expected, observed, details, numeric=False):
    value={"method":"executed","expected":expected,"observed":observed,"details":details}
    if numeric: value["tolerance"]="Exact integer/list equality; no numerical rounding tolerance required."
    return value
claims = [
    {"id":"prompt-construction","kind":"software","statement":"本節在本課 ByteTokenizer 約定下，用 BOS,user,問題內容,EOS,assistant 建立生成前文，當前未知答案不先輸入。",
     "location":"07.md:42–57","scope":"Single-user prompt in this repository; not a universal chat template or a ban on intentionally prefilling other tasks.","status":"verified",
     "evidence":[evidence("data","ByteTokenizer lines14–25; render_chat lines54–68","Special IDs and training serialization contract."),evidence("infer","main lines40–43 --chat branch","Identical generation-prefix order."),evidence("hf-chat","Using apply_chat_template and add_generation_prompt","An assistant-start header can end a generation prompt before any new answer content.")],
     "artifact_ids":[ids["original-execution.json"],ids["original-fence-1.py"],ids["contract.stdout.txt"]],
     "verification":verification("Prompt consists of nine IDs with no answer-content ID58 and ends in4.","Original fence rc0; independent exact prompt assertion passed.","This is list construction/assertion only; the EOS ID2 is not answer character2 (which is ID58).")},
    {"id":"ids-list-count","kind":"numeric","statement":"BOS1,user3,EOS2,assistant4；ASCII問題 bytes 加8後是57,51,57,69,71；相接產生九項[1,3,57,51,57,69,71,2,4]，[-1]是第8位置的末項4。",
     "location":"07.md:44,51–57","scope":"Exact byte tokenizer and question '1+1=?'; numeric token IDs differ from zero-based position indices.","status":"verified",
     "evidence":[evidence("python","#str.encode; #bytes; #common-sequence-operations s+t,s[i],note3","UTF-8 bytes are integers; list + concatenates; negative indices refer to len(s)+i."),evidence("data","SPECIALS lines11; ByteTokenizer lines17–21","Eight special entries and +8 ordinary-byte offset."),evidence("derivation","derivation.json question bytes, prompt_count, prompt_positions","49,43,49,61,63 plus8; 1+1+5+1+1=9; last index8.")],
     "artifact_ids":[ids["original-execution.json"],ids["contract.stdout.txt"],ids["derivation.json"]],
     "verification":verification("Exact IDs and count9, final position8 and ID4.","Original stdout and independent bytes/list/length assertions match exactly.","No tensor addition is used by list +. There is no rate or metric denominator here.",True)},
    {"id":"role-id-learning","kind":"concept","statement":"角色標記是序列中的專用ID；普通內容的標記拼寫不會自行取得角色，模型透過輸入表示與相符的對話訓練使用此結構。",
     "location":"07.md:44,59–61; needed prerequisite6.6 and7.1","scope":"ByteTokenizer/render_chat metadata-to-ID contract. Learned role behavior assumes compatible chat training; this tokenizer-only fence does not train the model.","status":"verified",
     "evidence":[evidence("hf-chat","overview lines24–28; Using apply_chat_template","role/content dictionaries become a control-token sequence and chat models learn from formatted messages."),evidence("pytorch-embedding","Embedding docstring lines17–43 and forward191–200","Embedding retrieves learnable vectors by integer indices; it does not parse their English spelling."),evidence("data","encode20–25; render_chat58–64","Content is always UTF-8 byte IDs>=8, while role metadata explicitly inserts3 or4."),evidence("model","TinyLM.__init__ line60 and forward71","Actual input IDs index nn.Embedding before sequence computation.")],
     "artifact_ids":[ids["contract.stdout.txt"],ids["contract_probe.py"]]},
    {"id":"tail-next-token","kind":"concept","statement":"在此下一token語言模型裡，prompt 尾端assistant位置的 logits 用於生成首回答token；該位置預測下一項。",
     "location":"07.md:42,57","scope":"No-padding single prompt. Checks generation's time-axis selection and training-target alignment, not actual answer correctness.","status":"verified",
     "evidence":[evidence("hf-generate","GenerationMixin token selection lines2800,2831–2838","Official implementation selects last time-position logits, then appends the selected token."),evidence("model","generate lines118–125; TinyLM.forward85","Project helper selects result['logits'][:, -1] and appends one next ID."),evidence("data","render_chat lines63–68","One target shift yields y[8]=58 while x[8]=assistant4."),evidence("probe-run","contract.stdout.txt and synthetic LastPositionStub in contract_probe.py","Actual helper chooses the final row in synthetic [B,T,V]=[1,9,264]; prompt training alignment checked independently.")],
     "artifact_ids":[ids["contract.stdout.txt"],ids["derivation.json"]]},
    {"id":"boundary-format","kind":"concept","statement":"本格式中EOS結束user消息、assistant開始下一角色；兩者功能不同，訓練與推論應維持相符的邊界順序，不能由標記字義保證替換格式後仍接對。",
     "location":"07.md:59–61","scope":"EOS2 followed by assistant4 is the local convention. Official documentation explicitly permits different model templates and models without assistant generation headers; no universal two-token requirement is asserted.","status":"verified",
     "evidence":[evidence("hf-chat","overview; Mistral/Zephyr comparison; add_generation_prompt including Llama exception; Model training","Models use different control formats; use training-compatible templates, and an assistant header marks a response start when that template requires one."),evidence("data","render_chat63–64","Project serialization ends each message with EOS and inserts next role before its content."),evidence("infer","main41–42","Inference mirrors user-content-EOS-assistant order.")],
     "artifact_ids":[ids["contract.stdout.txt"],ids["inspection.json"]]},
    {"id":"history-current-target","kind":"software","statement":"前文可以保留完整舊回答，但當前新回答尚未輸入；尾端只放新回答起點，仍沿用相同序列格式。",
     "location":"07.md:61","scope":"A normal new assistant turn with complete earlier messages, not a prefill continuation API.","status":"verified",
     "evidence":[evidence("hf-chat","add_generation_prompt before/after history examples","Old assistant content remains in history while a new assistant header is appended after the latest user message."),evidence("evaluate","chat_prompt17–22; evaluate SFT branch95–96","Serialize all prior messages and append assistant; exclude the final target message from prefix."),evidence("probe-run","contract.stdout.txt multi_turn_prompt and positions","19-token history has old answer1 at index9, new assistant at18, no current answer ID58; training first current target is y[18]=58.")],
     "artifact_ids":[ids["contract.stdout.txt"],ids["cpu-checks.json"]],
     "verification":verification("Old answer is retained, current target absent, current assistant index18 matches next-answer label58.","All exact multi-turn assertions passed, contract rc0.","Used small literal messages only; imported chat_prompt, not a checkpoint evaluator or learned model.")},
    {"id":"format-vs-capability","kind":"software","statement":"本結構指出誰應接續說話，但建立prompt和assert本身不提供或驗證算術正解；內容能力仍須以新題另查。",
     "location":"07.md:61,68","scope":"Limitation of this demonstration: no inference/training/evaluation of an actual learned model is claimed.","status":"verified",
     "evidence":[evidence("original-run","original-fence-1.py and original-execution.json","Original program instantiates tokenizer, concatenates IDs, prints and asserts boundary only; no model/gradient/optimizer."),evidence("hf-chat","overview, Model training","Chat behavior comes from fine-tuning on formatted data, rather than a list automatically creating task competence.")],
     "artifact_ids":[ids["original-execution.json"],ids["original-fence-1.py"],ids["inspection.json"]],
     "verification":verification("Only tokenizer and list/assert demonstration executes; no trained answer output or model metric.","Original attempted fence1 once, imports only tokenizer repository module; stdout is IDs/boundary.","Synthetic helper probe explicitly supplies logits and never counts its chosen58 as an arithmetic success. No optimizer/backward/checkpoint used.")},
    {"id":"exercise-missing-restored","kind":"software","statement":"僅移除尾端assistant後末項為EOS2，保留原assert會失敗；恢復assistant4後原檢查通過。",
     "location":"07.md:63","scope":"Expected contract failure and recovery; it does not prove the missing-boundary model would always answer incorrectly.","status":"verified",
     "evidence":[evidence("exercise-run","cpu-checks.json; exercise-missing/restored-fence.py and stdout/stderr","Actual unchanged assertion fails at EOS2 (rc1 AssertionError); restored original bytes end4 and pass(rc0)."),evidence("data","ByteTokenizer18","EOS and assistant are unequal IDs2 and4.")],
     "artifact_ids":[ids["cpu-checks.json"],ids["exercise-missing.stdout.txt"],ids["exercise-missing.stderr.txt"],ids["exercise-restored.stdout.txt"]],
     "verification":verification("Missing: end2, rc1 AssertionError; restored: end4, rc0.","Observed exactly those outcomes. Restored fence bytes equal original fence SHA.","Changed only the last list literal by removing assistant ID; retained assert unchanged; separate fresh CPU processes, timeout20s each.")},
]
report = {
    "schema_version":1,"review_stage":"technical","lesson_id":"7.2","source":"course/chapters/07.md#7.2",
    "source_sha256":extract["source_sha256"],"figure_sha256":{},"verdict":"pass",
    "reviewer_task":"/root/phase4_factual_coordinator/factual_7_2","reviewer_context":"fresh","author_tasks":[],
    "reviewed_on":"2026-10-05","review_scope":inspection,
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"Eight substantive claim groups independently checked against official originals and actual repository contracts. Local protocol specificity, role metadata versus text spelling, learned representation versus arithmetic ability, and new-turn history were explicitly considered.","claim_ids":[c["id"] for c in claims]},
        "numeric_verification":{"status":"pass","details":"Independently computed bytes+8, nine-token total, last position8 versus ID4, current answer ID58 rather than EOS2, single/multi-turn alignment. Exact CPU assertions passed. No historical metric or average/denominator exists in this section.","claim_ids":["ids-list-count"]},
        "figure_consistency":{"status":"not_applicable","details":"Exact raw section has no referenced SVG/image; no figure to render. Ordered ID list and position arithmetic are fully explicit in text. No visual inspection is claimed.","claim_ids":[]},
        "source_verification":{"status":"pass","details":"Personally read official fixed-tag Hugging Face chat and generation sources, fixed-tag PyTorch Embedding source, and Python3.13 official HTML. HTTPS/version/authority/locator/access date and immutable SHA snapshots are preserved. Installed torch2.14.1+cpu is distinguished from upstream2.9.0 source inspected.","claim_ids":[c["id"] for c in claims]},
        "limitations":{"status":"pass","details":"Original is a tokenizer/format check. Extra probes use deterministic lookup and synthetic logits, not learned-model or checkpoint evaluation. No training, data preparation, downloads of data/models, GPU, paid computation or long shell recipe. No historical empirical claim to audit; non-first-section chapter intro and absent figures specifically NA.","claim_ids":["boundary-format","format-vs-capability","exercise-missing-restored"]}
    }
}
save(ROOT / "docs/technical-reviews/7.2.json", report)
save(BASE / "manifest.json", {"lesson_id":"7.2","source_sha256":extract["source_sha256"],
    "report_sha256":sha(ROOT/"docs/technical-reviews/7.2.json"),"artifacts":[{"path":a["path"],"sha256":a["sha256"]} for a in artifacts],
    "all_required_artifacts_in_docs":True,"no_pt_files_or_ignored_inputs_required":True})
print(json.dumps({"verdict":report["verdict"],"claims":len(claims),"report_sha256":sha(ROOT/"docs/technical-reviews/7.2.json"),"artifacts":len(artifacts)}))
