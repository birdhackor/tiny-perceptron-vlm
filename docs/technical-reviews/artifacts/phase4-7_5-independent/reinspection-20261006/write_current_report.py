"""Independently authored current report and actual reinspection receipt."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OLD=HERE.parent
ROOT=HERE.parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
run=json.loads((HERE/"current-controls.stdout.json").read_text())
scope=json.loads((HERE/"scope-and-reuse.json").read_text())
previous_raw=json.loads((OLD/"command-receipts.json").read_text())["controls"]
assert sha(OLD/"bounded_controls.py")==previous_raw["code_sha256"]
assert sha(OLD/"bounded-controls.stdout.json")==previous_raw["stdout_sha256"]
assert sha(OLD/"bounded-controls.stderr.txt")==previous_raw["stderr_sha256"]
current_execution=json.loads((HERE/"current/execution.json").read_text())
assert current_execution["exit_code"]==0
receipt={"schema_version":1,"kind":"same_owner_technical_reinspection","reviewer_task":scope["reviewer_task"],"reviewed_on":"2026-10-06",
 "prior_history_path":scope["prior_history_path"],"prior_history_sha256":scope["prior_history_sha256"],
 "prior_source_sha256":scope["prior_source_sha256"],"current_source_sha256":scope["current_source_sha256"],
 "old_fence_sha256":sha(OLD/"original/fence-1.py"),"current_fence_sha256":sha(HERE/"current/fence-1.py"),
 "current_fence_changed":True,"changed_code":"logits.grad[0,0] -> logits.grad[0,2]; print labels and corresponding explanation/exercise now consistently name the question position",
 "actual_scope":scope["read_scope"],"current_execution":current_execution,
 "outer_command":".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.5 --execute --output /tmp/phase4-7_5-reinspection-20261006 --timeout 45",
 "controls_command":"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_5-independent/reinspection-20261006/current_controls.py",
 "controls_observed_exit_code":0,"controls_code_sha256":sha(HERE/"current_controls.py"),"controls_stdout_sha256":sha(HERE/"current-controls.stdout.json"),"controls_environment":run["environment"],
 "verified_current_checks":run["checks"],"reused_authorities":scope["unchanged_authorities"],"reused_contracts":scope["unchanged_contracts"],
 "reused_previous_raw_controls":previous_raw,"reused_figure":scope["figure_reuse"],
 "preview":{"url":"http://127.0.0.1:8765/7.5.html","current_fence_DOM_exact_match":True,"screenshot_command":".venv/bin/python docs/technical-reviews/artifacts/phase4-7_5-independent/reinspection-20261006/capture_page.py","observed_exit_code":0,"actual_visual_scope":"desktop and mobile current-code crops viewed; full-page screenshots captured but not visually inspected","browser":"system Chromium 151.0.7922.173","initial_failure":"managed Playwright executable absent; retained initial stderr; resolved using existing /usr/bin/chromium, no installation"},
 "verdict":"pass","unresolved_issues":[],"limits":run["limits"],"canonical_report_path":"docs/technical-reviews/7.5.json","canonical_receipt_artifact_id":"reinspection-receipt"}
dump(HERE/"reinspection-receipt.json",receipt)

artifacts=[]; fmap={}; env={k:str(v) for k,v in run["environment"].items()}
paths=sorted(p for p in HERE.rglob("*") if p.is_file() and not p.name.startswith("checker"))
reuse=[OLD/"bounded_controls.py",OLD/"bounded-controls.stdout.json",OLD/"bounded-controls.stderr.txt",OLD/"command-receipts.json",OLD/"context-figure/alignment.svg",OLD/"context-figure/alignment.png"]
for record in scope["unchanged_authorities"]:
 p=ROOT/record["path"];reuse.append(p)
 if p.suffix in (".html",".pdf") and p.with_suffix('.txt').is_file():reuse.append(p.with_suffix('.txt'))
for record in scope["unchanged_contracts"]:reuse.append(ROOT/record["own_frozen_snapshot"])
paths += sorted(set(reuse))
for index,p in enumerate(paths):
 identifier="reinspection-receipt" if p.name=="reinspection-receipt.json" else f"r{index:03d}"
 fmap[p]=identifier
 kind="code" if p.suffix==".py" else "figure_render" if p.suffix==".png" else "derivation" if p.name=="inspection.md" else "source_snapshot"
 item={"id":identifier,"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p),"kind":kind,"description":"本人的20261006真複查永久證據："+p.name+"；沿用範圍、實際執行與版本見reinspection-receipt/inspection。"}
 if p in [HERE/"current/stdout.txt",HERE/"current-controls.stdout.json",OLD/"bounded-controls.stdout.json",HERE/"preview-capture.stdout.txt"]:
  item.update(kind="execution",environment=env.copy())
  if p==HERE/"current/stdout.txt":item.update(command=receipt["outer_command"],result="20261006 actual modified fence exit0；問題索引2 logits grad0、Q embedding norm0.00571822514757514。")
  elif p==HERE/"current-controls.stdout.json":item.update(command=receipt["controls_command"],result="20261006 exit0；actual-current Q/Z fence、target-axis/denominator、analytic CE gradients、Q blocking、detach、causal axes and no update all asserted。")
  elif p==OLD/"bounded-controls.stdout.json":item.update(command=previous_raw["command"],result="20261005 prior personal raw CPU controls reused only for unchanged norm/nonleaf retention semantics; exact code/stdout hashes verified, not called a current-fence rerun。")
  else:item.update(command=receipt["preview"]["screenshot_command"],result="exit0 using system Chromium; 2 preview viewports captured and current-code crops viewed。")
 artifacts.append(item)
def aids(*ps):return [fmap[p] for p in ps]
current_a=aids(HERE/"current/stdout.txt",HERE/"current/fence-1.py",HERE/"current/execution.json",HERE/"current/environment.json")
control_a=aids(HERE/"current-controls.stdout.json",HERE/"current_controls.py",HERE/"variant-Z.py",HERE/"reinspection-receipt.json",HERE/"inspection.md")
prior_a=aids(OLD/"bounded-controls.stdout.json",OLD/"bounded_controls.py")

authority_specs={
 "ce":("cross-entropy.html","PyTorch CrossEntropyLoss","class-index equations, ignore_index, Shape; source text123–168,223–226,252–266"),
 "autograd":("autograd.html","PyTorch Autograd mechanics","How autograd encodes the history132–148; Setting requires_grad325–359"),
 "retain":("retain-grad.html","PyTorch Tensor.retain_grad","API123–126"),"leaf":("is-leaf.html","PyTorch Tensor.is_leaf","API123–135"),
 "detach":("detach.html","PyTorch Tensor.detach","API123–140"),"embedding":("embedding.html","PyTorch nn.Embedding","lookup, weight Shape, padding_idx123–176"),
 "norm":("torch-norm.html","PyTorch torch.norm","one-dimensional default fro=p2; source text123–220"),"tnorm":("norm.html","PyTorch Tensor.norm","signature default p=fro123–126"),
 "backward":("backward.html","PyTorch Tensor.backward","chain rule, leaf accumulation123–180"),"seed":("manual-seed.html","PyTorch manual_seed","API123–135"),"item":("item.html","PyTorch Tensor.item","single-element scalar API123–130"),
 "installed-functional":("installed-functional.py","PyTorch original nn/functional.py","cross_entropy3478–3571; SDPA mask6366–6409,6480–6508"),
 "installed-tensor":("installed-tensor.py","PyTorch original _tensor.py","backward566–630; norm888–902"),
 "paper":("attention-paper-v7.pdf","Attention Is All You Need original paper","PDFpp3–4 §3.1 Decoder, §3.2.1 Eq.(1)")}
provenance={r["file"]:r for r in json.loads((OLD/"sources/download-provenance.json").read_text())}
sources=[]
for sid,(filename,title,locator) in authority_specs.items():
 rec=provenance[filename]; kind="paper" if sid=="paper" else "official_source" if filename.endswith('.py') else "official_docs"
 sources.append({"id":sid,"kind":kind,"title":title,"verified":True,"url":rec["requested_url"],
  "version":"arXiv1706.03762v7 2023-08-02" if kind=="paper" else "installed commit5c4886908584029761b579af026dcfb627c84070" if kind=="official_source" else "PyTorch official docs2.9; execution2.14.1+cpu distinguished",
  "accessed_on":"2026-10-05","rechecked_on":"2026-10-06","authority_reason":"Original authors' versioned arXiv paper" if kind=="paper" else "PyTorch official maintainer repository at exact installed commit" if kind=="official_source" else "PyTorch maintainer's versioned docs.pytorch.org original API documentation",
  "checked_original":True,"inspection_note":"Same original reviewer personally inspected original authority, now exact raw snapshot hash rechecked and unchanged scope retained; locator: "+locator+". Relevant CE, attention, autograd, retain/leaf and norm paragraphs were reread for the changed question-position claim; no source-library summary substituted."})
for sid,path,locator in [("data","tiny_perceptron/data.py","ByteTokenizer10–21 and render_chat54–68"),("model","tiny_perceptron/model.py","ModelConfig14–28; TinyLM53–86; loss_sum/masked_loss92–108"),("attention","tiny_perceptron/attention.py","attention_mask/manual_attention10–28; CausalAttention48–74"),("modern","tiny_perceptron/modern.py","positionwise DenseFFN42–61")]:
 sources.append({"id":sid,"kind":"repository_code","title":path,"verified":True,"path":path,"sha256":sha(ROOT/path),"version":"current exact file SHA identical to original personal frozen input","inspection_note":locator+"; exact bytes checked unchanged against my original snapshot, current necessary data/loss/mask routines reread."})
sources += [{"id":"current-run","kind":"execution","title":"Actual current modified fence CPU run","verified":True,"artifact_id":fmap[HERE/"current/stdout.txt"]},{"id":"controls","kind":"execution","title":"Current-fence and necessary bounded CPU controls","verified":True,"artifact_id":fmap[HERE/"current-controls.stdout.json"]},{"id":"prior-controls","kind":"execution","title":"Personal unchanged prior raw norm/retain controls","verified":True,"artifact_id":fmap[OLD/"bounded-controls.stdout.json"]}]
def ev(s,l,t):return {"source_id":s,"locator":l,"supports":t}
def ver(e,o,d,n=False):
 v={"method":"executed","expected":e,"observed":o,"details":d}
 if n:v['tolerance']='IDs/indexes/ignored gradients exact; scalar norm sqrt-sum-square atol1e-8; float64 CE/Jacobian atol1e-14.'
 return v
claims=[]
def claim(cid,kind,statement,location,scope,evidence,arts,verification=None):
 c={"id":cid,"kind":kind,"statement":statement,"location":location,"scope":scope,"status":"verified","evidence":evidence,"artifact_ids":arts}
 if verification:c['verification']=verification
 claims.append(c)
claim('c1','concept','問題Q不算直接答案代價；問題格候選分數梯度可以0，Q輸入特徵卻從後面答案收到梯度，兩者針對不同變量。','07.md:148,169','Finite class-index logits with default untied single-layer causal model; not a universal nonzero gradient guarantee.',[ev('ce','class-index/ignore_index','Per-target direct score derivative ignored.'),ev('paper','§3.1/§3.2.1 Eq1','Allowed earlier keys/values can affect later output.'),ev('autograd','history132–148','Upstream derivatives follow actual dependency graph.'),ev('controls','Q/block_Q_key/detach','Question-index2 direct gradient0, input gradient positive; distinct controls sever input path.')],current_a+control_a)
claim('c2','numeric','Q位於X索引2、ID89；embedding.weight第89列是Q字向量。','07.md:150','Byte81+8=89 under this course tokenizer only.',[ev('data','10–21,54–68','Role boundaries, byte encoding, one target shift.'),ev('embedding','Variables/Shape','Integer lookup indexes a weight row.'),ev('controls','checks.Q','Actual X[2]=89 and shape[1,6,264].')],control_a,ver('X[2]=89, Y[2]=-100.','Both exact in current run; X=[1,3,89,2,4,73].','Positions and token IDs checked separately.',True))
claim('c3','software','現稿程式查看logits.grad[0,2]與embedding.weight.grad[x[2]]，問題logits梯度應0，Q row通常非零。','07.md:153–167','Changed current fence actually executed; no substitution of prior first-position stdout.',[ev('current-run','current/stdout.txt/current/fence-1.py','Actual modified index2 output.'),ev('model','14–28,53–108','TinyLM,width8,loss/backward contracts.'),ev('seed','manual_seed API','Seed42 initialization.'),ev('retain','API','Intermediate gradient retained.'),ev('backward','API','Scalar loss backward.'),ev('tnorm','signature','One-dimensional norm.'),ev('item','API','Scalar extraction.')],current_a+control_a,ver('Question position2 score norm0, Q vector norm>0.','0.0 and0.00571822514757514; current fence SHAce8105e8...','Entire current fence and current variant ran, not just a replaced report SHA.'))
claim('c4','concept','logits是中間結果，PyTorch一般不保留它.grad，retain_grad明確要求保留，葉參數不同。','07.md:167','Nonleaf gradient retention semantics unchanged.',[ev('leaf','123–135','Leaf vs nonleaf grad population.'),ev('retain','123–126','Explicit intermediate retention.'),ev('prior-controls','without_retain_grad','Unchanged personal control shows logits.gradNone while input gradients remain; original bytes/stdout hashes verified.')],control_a+prior_a)
claim('c5','numeric','norm把各格平方加總開根；兩種梯度大小不當同一單位比較。','07.md:167,169','One-dimensional real 264-logit-coordinate vs8-embedding-coordinate derivatives; no relative effect ranking.',[ev('norm','p/dim123–220','Default fro equals Euclidean norm here.'),ev('installed-tensor','norm888–902','Installed-commit implementation dispatches to torch.norm.'),ev('prior-controls','baseline_Q norm_manual_sqrt_sum_squares_float64','Unchanged norm derivation personally executed, raw hash reused transparently.')],prior_a+control_a,ver('sqrt sum squares equals row norm within1e-8.','Prior personal unchanged control manual0.005718225021817441 vs norm0.00571822514757514, difference1.26e-10.','Operation is unchanged; that result is not described as a new fence execution. Current question-index score row has264 exact zeros.',True))
claim('c6','concept','整個embedding層.grad非None不能證明Q那列參與；問題位置logits梯度0不能推論所有user內容凍結。','07.md:169','Whole table and individual derivatives distinguished; no claim that every lookup row is nonzero.',[ev('embedding','Variables/Shape','Individual lookup rows of learned table.'),ev('autograd','Setting requires_grad','Freezing requires distinct parameter flag.'),ev('controls','block_Q_key/detach/Q','Same direct score0 with differing input gradient, blocked row0 vs table gradients.')],control_a)
claim('c7','concept','忽略user loss没有凍結共享輸入表；detach輸入或禁止回答讀Q會切掉另一條影響路線。','07.md:171','Default untied single-layer model; blocking Q as key at all cross-position paths. Detach cuts graph but preserves forward input.',[ev('detach','123–140','Returned tensor detached, not numerical deletion.'),ev('attention','10–28,48–74','Mask acts on keys, weighted values.'),ev('modern','DenseFFN42–61','Positionwise FFN introduces no other cross-position route.'),ev('controls','detach/block_Q_key','Current bounded controls preserve logits with detach and sever Q derivative with key block.')],control_a)
claim('c8','software','本段只求梯度，沒有step，尚未調整出回答能力。','07.md:171','No optimizer update or answer-accuracy evidence; raw randomly initialized model.',[ev('backward','123–138','Computes/accumulates derivatives.'),ev('installed-tensor','backward566–630','Actual installed source delegates autograd, no optimizer.'),ev('controls','Q/Z no_parameter_update;limits','Actual current fence values match separately seeded init; controls leave all values unchanged.')],current_a+control_a,ver('Parameter values unchanged and0 optimizer steps.','Exact parameter comparisons pass for current Q/Z and all controls.','No training, existing weight evaluation, download, GPU or chapter-wide execution.'))
claim('c9','software','把user Q換成Z，問題位置logits仍0，新的Z字表列通常有梯度，X索引2會查新列。','07.md:173','Only actual current user content byte changed, same answer/seed/config.',[ev('data','ByteTokenizer/render_chat','Z90+8=98 retains position2.'),ev('controls','checks.Z and saved variant-Z.py','Actual changed-fence Z variant executed with modified index2 print.')],control_a,ver('X[2]=98; question score norm0; Z embedding norm>0.','X[2]=98; score0; Z norm0.004994203336536884.','Variant SHA31bc9e9f...; not the old first-position exercise.'))
claim('c10','software','masked_loss以-100忽略目標格，與注意力讀取許可不同；此例候選軸及有效答案分母按實際helper契約計算。','07.md:158–162,167; referenced helper contract','Finite unweighted integer-class CE; logits[1,6,264], effective count2>0; mask[B,H,query,key].',[ev('ce','class-index formula/Shape','Ignore and class axes/effective denominator.'),ev('installed-functional','3478–3571,6366–6409,6480–6508','Exact installed API loss/mask semantics.'),ev('model','92–108','Flatten candidate axis and explicit nonzero count.'),ev('controls','math_and_target_axis/causal_axes','Hand CE/Jacobian; extra target at Q gives count3/direct gradient; all-ignored rejection; causal query/key control.')],control_a,ver('Count2; dL/dz=M/N*(softmax-onehot); query4 readsQ2 but not future5.','Loss diff8.88e-16; Jacobian maxdiff2.60e-18; question score0; adding target gives count3/norm0.33264448; future influence0.','Independent numeric substitution on actual current logits; altered target is a loss-only mechanism control, not a correct dialogue sample.'))
report={"schema_version":1,"review_stage":"technical","lesson_id":"7.5","source":"course/chapters/07.md#7.5","source_sha256":scope['current_source_sha256'],"figure_sha256":{},"reviewer_task":scope['reviewer_task'],"reviewer_context":"fresh","author_tasks":[],"verdict":"pass","reviewed_on":"2026-10-06","original_fresh_review_date":"2026-10-05","reinspection_mode":"same original technical owner; independently authored new current canonical after real modified-fence verification","read_scope":scope['read_scope'],"artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],"reinspection":{"artifact_id":"reinspection-receipt","receipt_path":(HERE/'reinspection-receipt.json').relative_to(ROOT).as_posix(),"receipt_sha256":sha(HERE/'reinspection-receipt.json'),"prior_history_path":scope['prior_history_path'],"prior_history_sha256":scope['prior_history_sha256'],"changed_code":receipt['changed_code'],"unresolved_issues":[]},"checks":{
 "factual_accuracy":{"status":"pass","claim_ids":[c['id'] for c in claims],"details":"Current section reread fully and compared to own original; altered index0->2 independently checked against actual X/Y and fresh CPU run.10 substantive claims checked within scope; no author repair answer or other review imported."},
 "numeric_verification":{"status":"pass","claim_ids":["c2","c3","c5","c9","c10"],"details":"Current Q/Z question gradients0; input norms0.005718225/0.004994203; count2; independent CE/Jacobian errors<1e-14.Changed-target control count3/direct gradient nonzero. Unchanged norm calculation reused after raw evidence fingerprint check."},
 "figure_consistency":{"status":"not_applicable","claim_ids":["c2","c3"],"details":"No referenced image in current7.5. Own previous7.4 SVG is unchanged, prior render/view explicitly reused. Current supplied local page DOM code matches current fence exactly; desktop/mobile current-code crops actually viewed via existing Chromium, labels readable; mobile long code overflows horizontally."},
 "source_verification":{"status":"pass","claim_ids":[c['id'] for c in claims],"details":"16 original authority raw snapshots match personal download provenance hashes; relevant original CE/attention/chain-rule/retain/leaf/norm passages reread. Unchanged docs2.9 vs actual torch2.14.1+cpu/installed commit distinguished. Eight executable/protocol contracts unchanged and current necessary routines reread. No new source search or generic paper fetch."},
 "limitations":{"status":"pass","claim_ids":["c1","c7","c8","c9","c10"],"details":"Actual modified fence and bounded controls only; no optimizer step, training, existing weight evaluation, models/data download, GPU, full chapter rerun or paid compute. Untied single-layer causal control not universal nonzero gradient guarantee; extra Q target is loss-only illustration."}},"required_artifacts":[a['path'] for a in artifacts]}
dump(ROOT/'docs/technical-reviews/7.5.json',report)
print(json.dumps({"verdict":report['verdict'],"report_sha256":sha(ROOT/'docs/technical-reviews/7.5.json'),"receipt_artifact_id":"reinspection-receipt","receipt_path":report['reinspection']['receipt_path'],"receipt_sha256":sha(HERE/'reinspection-receipt.json'),"prior_history_path":scope['prior_history_path'],"prior_history_sha256":scope['prior_history_sha256'],"claims":len(claims)},ensure_ascii=False,indent=2))
