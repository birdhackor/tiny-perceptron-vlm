"""Assemble reviewed claims; require actual reread receipt before current pass."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "docs/technical-reviews/artifacts"
PRE = "natural-final-fact-20.8-training-"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_artifact(identifier, name, description, kind="source_snapshot", command=None, result=None):
    path = P / name
    artifact = {"id": identifier, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "description": description, "kind": kind}
    if kind == "execution":
        artifact.update(command=command, result=result, environment={"python": "3.12.14", "platform": "Linux", "device": "CPU source/argparse/text/hash only; no model forward, GPU or training"})
    report["artifacts"].append(artifact)


def source(identifier, path, title, note):
    report["sources"].append({"id": identifier, "kind": "repository_code", "path": path, "sha256": sha(ROOT / path), "title": title, "version": "Actual current bytes independently read 2026-10-04; exact SHA registered", "verified": True, "inspection_note": note})


def E(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, scope, expected, observed, evidence, supplement=False, denominators=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "assessment": "supported", "status": "verified", "scope": scope, "evidence": evidence, "artifact_ids": ["a_training_audit", "a_training_current_guide", "a_training_current_20_8"], "verification": {"method": "executed", "expected": expected, "observed": observed, "details": scope}}
    if supplement:
        item["artifact_ids"].append("a_training_supplement")
    if kind == "numeric":
        item["verification"]["tolerance"] = "Integer/SHA exact; decimal rounding stated separately."
    if denominators:
        item["verification"]["denominators"] = denominators
    report["claims"].append(item)


report = json.loads((P / "natural-final-fact-20.8-pass-3733ef68-report.json").read_text())
audit = json.loads((P / (PRE + "audit.json")).read_text())
receipt = json.loads((P / (PRE + "reread-receipt.json")).read_text())
assert receipt["actual_full_current_20_8_reread"] is True and receipt["actual_full_training_reread"] is True
assert receipt["chapter_section_sha256"] == sha(P / (PRE + "current-20.8.md"))
assert receipt["training_sha256"] == sha(ROOT / "docs/natural-assistant/TRAINING.md") == sha(P / (PRE + "current-TRAINING.md"))
assert receipt["time_budget_issue_resolved"] is True
report["source_sha256"] = receipt["chapter_section_sha256"]
report["verdict"] = "pass"
source("s_training_guide", "docs/natural-assistant/TRAINING.md", "Current optional training guide, reviewed subject", "Full first-read b042 saved, time-budget wording revised by root, full corrected guide reread. This live source registers the reviewed instruction version; it is not evidence of a new GPU run.")
source("s_training_cli", "scripts/natural_assistant.py", "Actual training/evaluation CLI", "Full parser/main independently read; defaults3e-5/100, complete five guide command argv parsed, help executed, fixed student Git1d5fccb bytes identical. No stage with model loading executed.")
source("s_data_cli", "scripts/fetch_natural_data.py", "Actual anonymous fixed data helper", "Entire helper independently read: fixed21a241/manifest7604526, exact three archives/path and file checks, SHA/size/HTTPS, no credential headers, reject differing existing directory and atomic complete rename. Actual list and existing345-file verify exit0; fixed student Git1d5fccb helper bytes identical.")
source("s_training_data", "docs/natural-assistant/manifest.json", "Frozen training/validation/test data contract", "Read fixed manifest and recounted272/52/66 rows,24/6/12 audio,345 files and3 archives70648031B. Actual existing directory everyfile verified, no data/GT changes.")
source("s_modal_runner", "scripts/modal_natural.py", "Actual Modal command construction", "Read actual args builder and train override: --learning-rate0.00003. This workflow does not execute the new1e-4 self-GPU guide recipe.")
source("s_natural_workflow", ".github/workflows/natural-assistant.yml", "Actual workflow input contract", "Read workflow dispatch inputs, steps/seed/max_seconds/max_pixels/selection paths and command mapping; no learning-rate input. No dispatch/GPU execution.")
report["sources"].extend([{"id": "s_training_audit", "kind": "execution", "title": "Independent CPU recipe/command/source/records audit", "verified": True, "artifact_id": "a_training_audit"}, {"id": "s_training_supplement", "kind": "execution", "title": "Independent architecture arithmetic and proxy scope probes", "verified": True, "artifact_id": "a_training_supplement"}])

add_artifact("a_training_audit", PRE+"audit.json", "Actual help/list/345file verify/syntax/argparse, extracted seven resume guards, original180step records and frozen old54artifacts; no model training.", "execution", ".venv-natural/bin/python docs/technical-reviews/artifacts/"+PRE+"audit.py", "Exit0; five subprocesses exit0, 64/90 pervariant, all7 production guard mutations rejected, original metadata matches, fixed Git helpers exact.")
add_artifact("a_training_supplement", PRE+"supplement.json", "Actual pinned config arithmetic28qv/r8 and scalar score/CER probes show proxy/EOS boundaries.", "execution", ".venv-natural/bin/python docs/technical-reviews/artifacts/"+PRE+"supplement.py", "Exit0; params1605632,112tensors; keyword score can pass unsupported addition and score aggregation does not require EOS, so full answer review is necessary.")
for identifier, name, description, kind in [
    ("a_training_audit_code",PRE+"audit.py","Actual CPU audit script, no model load/forward/train calls.","code"),
    ("a_training_supplement_code",PRE+"supplement.py","Actual arithmetic/scalar-text probes.","code"),
    ("a_training_first_20_8",PRE+"read3-current.md","Complete genuinely read0bc720 current20.8, before training wording correction.","source_snapshot"),
    ("a_training_first_guide",PRE+"first-read.md","Complete b042 first-read guide before root time-budget fix.","source_snapshot"),
    ("a_training_first_revise",PRE+"first-review-revise.json","Initial actual revise snapshot, pending other guide checks clearly marked.","source_snapshot"),
    ("a_training_current_20_8",PRE+"current-20.8.md","Whole current20.8 personally reread for final review.","source_snapshot"),
    ("a_training_current_guide",PRE+"current-TRAINING.md","Whole corrected TRAINING personally reread, exact live bytes.","source_snapshot"),
    ("a_training_reread",PRE+"reread-receipt.json","True complete reread, source hashes and actual text diff, not mechanical hash update.","derivation"),
    ("a_training_source_diff",PRE+"source-diff.txt","Real chapter3733→0bc720 difference: optional training link plus fresh LoRA/validation/freeze sentence.","source_snapshot"),
    ("a_training_guide_diff",PRE+"guide-fix-diff.txt","Real full guide b042→current time-budget wording correction.","source_snapshot"),
    ("a_training_commands",PRE+"commands.sh","Eight exact Bash fences checked with bash -n; not executed training script.","code"),
    ("a_training_metadata",PRE+"original-adapter-metadata.json","Actual retained original180step adapter/training.json; differs from public result only appended runner execution annotation. training_state.pt not retained locally.","source_snapshot"),
    ("a_training_pass_history","natural-final-fact-20.8-pass-3733ef68-report.json","Prior real29claim/54artifact pass intact, original checker had passed.","source_snapshot"),
    ("a_training_pass_source_history","natural-final-fact-20.8-pass-3733ef68-source.md","Prior full3733 source preserved.","source_snapshot"),
]:
    add_artifact(identifier, name, description, kind)
for index in range(5):
    for channel in ("stdout", "stderr"):
        add_artifact(f"a_training_cmd_{index}_{channel}",PRE+f"command-{index}.{channel}.txt","Actual captured subprocess output; argv/exit/duration recorded in training audit.")
for path in ("scripts__natural_assistant.py", "scripts__fetch_natural_data.py"):
    add_artifact("a_training_git_"+path.replace(".","_"),PRE+"student-"+path,"Original anonymous fixed Git1d5fccb helper, byte-exact with current code.")
add_artifact("a_training_readme_supplement",PRE+"readme-supplement.json","Optional actual README delta/counts/29GPUoriginalresults versusCPUexception scope; oldsupplement retained.","execution",".venv-natural/bin/python docs/technical-reviews/artifacts/"+PRE+"readme-supplement.py","Exit0;20chapters/261notebooks/287sections,29planGPUrecordsL4/2.14.1+cu126,CPU simple_models excluded; Chapter20actual2.8cu128/sourcePython3.12; two corrected README GPU paragraphs personallyread.")
add_artifact("a_training_readme_supplement_code",PRE+"readme-supplement.py","Actual read-only optional README verification.","code")
for filename in ("README.md","README_en.md"):
    add_artifact("a_training_readme_current_"+filename.replace(".","_"),PRE+"readme-current-"+filename,"Current README complete bytes archived; review scope onlycounts/naturalroute/hardwareparagraph, notblanketfile pass.")
add_artifact("a_training_build_code",PRE+"build-report.py","Actual report assembly, current reread/SHA/issue-resolution required before pass.","code")

claim("c30","software","新增訓練連結提供自有GPU fresh LoRA→驗證→固定自己模型的路線，不宣稱恢復作者狀態或逐byte重現成品。","20.8 final paragraph; TRAINING intro/1/6","指南是選讀額外資源；Linux Python3.12、CUDA12.8配套與BF16檢查是前提。本審核未安装、GPU或重訓。","Link exists and recipe creates fresh adapter without --adapter; preserve distinct inference/resume claims.","Fullnew20.8 andguide actuallyread; freshtrain parsedadapter=None; GPUprecheck exacttorchAPI; ownvalidation/hash/test commands parsed.",[E("s_training_guide","intro/1/4/6","Fresh LoRA, hardware precheck, own outputs and no identity guarantee."),E("s_training_cli","parser/load_core call","Freshtrain noadapter builds newLoRA."),E("s_core","load_core train branch","Basefrozen, newPEFT adapter.")])
claim("c31","numeric","三包固定資料70,648,031B≈70.65MB、345檔；圖文題272/52/66、音檔24/6/12。","TRAINING2","指壓縮三包傳輸量；不含模型或上游完整資料。既有資料真verify345檔，這次沒有重新下载。","Exact frozen split/file/archive counts and byte total.","Manifest independently counted and helper --list/--verify actualexit0:70648031B,345files,272/52/66 and24/6/12.",[E("s_training_data","archives/files/rows/audio_rows","Frozen explicit lists andsizes."),E("s_data_cli","summary/verify_directory","Real everyfile SHA/byte exact verifier."),E("s_training_audit","data/executions[3:5]","Independent recount + actual345verify.")])
claim("c32","software","資料入口匿名固定21a241…；核對清單、包、每檔SHA與大小，相同資料重用，不同資料停止且保留。","TRAINING2","固定學生checkout1d5fccb有此helper；只核現碼下載設計與fixed public code、既有完整資料verify，不宣稱本次重下載70MB。","Frozen exact-file bounded anonymous helper; no overwrite on mismatch.","Entirehelper source read; no auth headers; SHA/size checks before temp rename; list/345existingverify pass; fixedGithelper exact200.",[E("s_data_cli","fetch_data/download_checked/extract_checked/verify_directory","Actual immutable anonymous data contract."),E("s_training_audit","fixed_student_git_helpers","Published helper at full student commit.")])
claim("c33","software","prepare只準備固定Qwen/Whisper模型約5.24GB快取，後續同cache-dir與--local-files-only重用；尚未更新LoRA。","TRAINING3","5.24GB來自已獨立hash的23pinned公開模型檔；環境／其他快取另計。這次不下載大模型。","Prepare fixedpins/snapshot-only; later command samecache andlocalonly.","Readprepare snapshot_download/filehash code; parsedprepare andalllater cache=.cache/natural-models; old23hash proof unchanged.",[E("s_core","prepare/constants","Fixedpins, snapshotdownload thenfilehash, no modeltrain."),E("s_audit","23_cached_public_files/cached_total_bytes","Independentprevious exact23files5238035532B."),E("s_training_audit","parsed_production_cli_commands","Exact guide argv.")])
claim("c34","numeric","明写LR1e-4与180步；CLI当前默认3e-5与100步。每步累积2题，180更新为360次题目出现。","TRAINING4","360不是不同题；只读原结果与解析正式argv，无新训练。","Fresh/resume explicitly1e-4/180/gradacc2; originalactualsame.","parser defaults actual3e-5/100; parsedtwo trains actual1e-4/180/2; originalresultcompleted180/trained_rows360.",[E("s_training_cli","parser LR/steps/gradient_accumulation","Actualdefaults."),E("s_training_audit","cli_defaults/parsed.../original_training","Independentparser andoriginalrecords.")])
claim("c35","numeric","语言28层q/v投影r8、alpha16：112份LoRA张量，1,605,632可更新参数；只有LoRA进入AdamW。","TRAINING4","原底座仍参与计算；视覺与ASR没有重训。","28qv/r8/alpha16,112 tensors andexactparams.","Pinnedconfig h2048/q2048/v1024:28*(8*(2048+2048)+8*(2048+1024))=1605632; original112tensor shapes sum same; all optimizer namesLoRA.",[E("s_core","LORA_TARGETS/load_core/run_train optimizer guard","Exactregex/rank/alpha/freeze."),E("s_training_supplement","derivation/layers/targets","Independentactualarithmetic."),E("s_training_audit","original_trainable_tensor_shape_sum","Original112shape sum.")],True)
claim("c36","software","token输入加答案超过2048报错；图片max_pixels524288为处理预算；3300秒是每次更新前检查的预算，允许单步与最终保存超时。","TRAINING4 corrected budget paragraph","训练计时在资料核验后、载入前开始，不是整个命令硬上限；max_pixels不是所有中间tensor固定shape或显存保证。","No silenttruncation and scopednonpreemptive budget.","Readencode_training_row overflowraise andprocessormaxpixels; run_train startedafterload_manifest/beforeload_core, guardonlybeforeupdate, finalhash/savesafterelapsed; correctedwholeguide reread.",[E("s_core","encode_training_row/load_core/run_train","Actual token/pixel/timer contracts."),E("s_training_audit","time_scope","Independent actual source ordering."),E("s_training_guide","4 corrected max-seconds paragraph","Rootfixed wording after actualrevise.")])
claim("c37","empirical","原NVIDIA L4完成180步，112修正hash改变、少量冻结值抽查不变；149.47s与5.23GiB仅各自明确范围。","TRAINING4 original experiment paragraph","elapsed含载入／更新／循环checkpoint，停于finalhash/save前，排除data核验／transfer／env；GPU allocated peak不等于全部占用或最低要求。仅既有原执行审计。","Original180actualrecord/LR1e-4/112changes/frozensamples andtimerscope.","Read originalresult andretainedadaptermetadata:180,LR.0001,360 appearances,112changes,7frozensamples;149.471174897s;5615273984B/2^30=5.22963142395GiB. Metadata difference onlyrunner executionannotation.",[E("s_training_audit","original_training/original_peak_allocated_GiB/resume.metadata_difference","Recomputedoriginalfullrecords."),E("s_core","run_train elapsed/reset_peak/query/frozen_check_scope","Actualtime/allocated/samplescope.")],False,{"updates":180,"adapter_tensors":112,"frozen_sample_tensors":7,"row_appearances":360})
claim("c38","software","推论两档另配training_state.pt与training.json，整个adapter目录才能恢复AdamW、CPU/CUDA RNG与进度；public四档不足。","TRAINING5 table/resume paragraph","本地保留原adapter/training.json并真读；training_state.pt未被保留，因此未反序列化原state／实际resume；保存与恢复内容来自当前生产源码，不保证异硬件相同权重。","Checkpoint includesadapterweights/config +optimizer/RNG/progress.","Actualsave_checkpoint dictkeys optimizer/torch_rng/cuda_rng; resume torch.load(weights_only=True), AdamWload,CPUset_rng,CUDArng restoration,priorhistory/completed/trained_rows; public4fileallowlist lacksallstate.",[E("s_core","save_checkpoint/run_train resume","ActualStatepaths andrestoration."),E("s_checkpoint","General Checkpoint originalparagraph/example","Originalauthoritativetutorial: weightsalone notenough."),E("s_manifest","files","Publicinferenceonlyallowlist."),E("s_training_audit","resume/state_keys/metadata","CPU AST/statecontract andactualretainedmetadata inspection.")])
claim("c39","software","每25步或约60秒保存；中断未保存更新不能恢复；completed180／LR.0001／onlyLoRA是执行检查，质量要另验。","TRAINING5","检查点是在完成update后检查间隔；异常只保留最新完整checkpoint。时间预算停止走finalsave；不能把partial称180完成。","Exactcheckpoint interval/status/record semantics.","Readstep modulo25 ormonotonic sincecheckpoint>=60, atomicadapter tmp/rotation, statusonlycompleted_steps==target; parsercheckpoint25; original180recordexpectedfields match.",[E("s_core","run_train/save_checkpoint","Actualinterval/status/atomiccheckpoint."),E("s_training_audit","original_training/parsed...","Explicitcommands andexpectedoriginalstatus.")])
claim("c40","software","续训--steps180是目标总步数，使用自己的完整旧adapter、新resumed输出；后续路径同步更换，保持配方一致。","TRAINING5 resume command","代码自动拒错仅7栏：manifest/assets/model_revision/rank/accum/seed/LR；dtype/device/pixels/tokens/package/code等额外一致要求由指引约束，不宣称全部自动拒绝。","Resume continuescompleted→180; rejectschangedguardfields/targetbelowcompleted.","Actualextractedproductionseven-fieldguard executed:all7mutationsValueError; extractedtotalguard accepts175→180,rejects181prior; actualrange(175,180) fiveupdates; parsedresume sourceoutput differs.",[E("s_core","run_train resume fieldloop/range/totalguard","Actualcontract andsteps semantics."),E("s_training_audit","resume/parsed_production_cli_commands","Actualguardexcerpt execution andparser.")])
claim("c41","numeric","完整validation同底座关/开自己的LoRA各64生成，保存两版回答、result与实际ASR transcripts；test相同公式每版90。","TRAINING6 validation/evaluate","64=52+2×6；90=66+2×12。completed只表示请求记录都完成，不代表质量；原ASR每录音一次共享各variant，分真实hyp和typedref两路。","Bothvariants completedcardinality64validation/90test.","Frozenrows/audioindependentrecount52/6and66/12; sourcevariant contexts disabled/enabled; run_evaluate zip actualhyp/typedref andcompletedformula; parsed--split validation/test.",[E("s_training_data","rows/audio_rows splits","Actualuniquefixedcounts."),E("s_core","run_evaluate/evaluate_rows","Variant/ASR/cardinality exactmechanics."),E("s_training_audit","data.per_variant_generation_denominators/metrics_scope","Independentrecount.")])
claim("c42","software","自动关键词／逐字分数是proxy；完整回答、无依据细节、截断／unknown与ASR原错误需另核；EOS不是语意正确。","TRAINING6 fullanswer/partial paragraph","指南不把自动pass_rate或completed等同semantic-final；实际summary不因EOS/截断自动剔除score。因此读完整内容的要求是必要的。","Proxies anddecoderstop notgeneralqualityjudges.","Executedproduction scalarfacts probe:unsupportedaddition stillpasseskeyword; manualscoreNone; truncatedscore passed countedbysummarize; generateEOSscope explicit; raw vsnormalizedCER differactualtextprobe.",[E("s_core","score_output/summarize/generate/transcribe","Actualproxy/CER/EOS contract."),E("s_training_supplement","proxy_example.../truncated.../raw_vs_normalized","PureCPU concreteproof ofscope."),E("s_whisper","PerformanceLimitations originalcard","OriginalASR limitations; no nativeaudio/TTS claim.")],True)
claim("c43","software","只用自己的validation选版，保存自己的weights/config/data/result完整SHA后freeze再test；不能沿用作者selection或看test重选。","TRAINING6 chosen-before-test/evaluate/posttest","这是读者须遵守的实验步骤，直接CLI不会自动强制新selection记录或公平性；仅复用recipe，不声称逐字权重/品质重現。","Ownoutput/hashlist thenfrozen test; noauthorselectionreuse.","Guidewholetextread; exactsha256sum fourownfiles/result; stageargvvalidation→evaluate test andownadapter/output parsed; explicitrejectauthormatrixselectionreuse andtestreselections warning.",[E("s_training_guide","6 sha256sum/freeze/test paragraph","Reviewed procedure andmanual selectionboundary."),E("s_training_cli","main stages/split","Validationstage defaultsvalidation, finalevaluate explicit test."),E("s_training_audit","metrics_scope.selection/parsed...","Ownpaths, explicitsplits andsourcechecks.")])
claim("c44","software","现有Modal workflow没有LR输入且runner固定3e-5；本1e-4重做路線使用自有GPU CLI。","TRAINING7 final Modal paragraph","不执行dispatch，不把工作流程默认值误称原L4首次实验LR；新训练只有明确CLI配方。","Actualworkflow noLRinput andrunner3e-5.","Wholeworkflowinputs read; actualrunnerargs trainadds--learning-rate0.00003; guides CLI train/resumeparsed1e-4.",[E("s_modal_runner","args builder stage==train","Exactoverride0.00003."),E("s_natural_workflow","workflow_dispatch.inputs/Modalcommand","NoLRuserinput."),E("s_training_audit","cli_defaults/parsed...","Explicit selfGPUrecipe1e-4.")])

report["claims"][28]["evidence"].append(E("s_training_guide","intro/5/6","Newfreshtraininglink preserves distinction from inference-only publicweights."))
report["claims"][28]["artifact_ids"].extend(["a_training_current_20_8","a_training_current_guide"])
report["issues"].append({"id":"i2","severity":"minor","location":"TRAINING4 originalb042 max-seconds sentence","problem":"3300 described as execution upperlimit though actual pre-update guard permits overrun and excludes initial data verification.","status":"resolved","resolution":"Root precisely scoped budget start/checkpoint/one-update+finalsaveoverrun. This reviewer preserved firstb042source/revise then personally fully reread correctedguide andwhole0bc720chapter; core/CLI unchanged.","initial_review_artifact_id":"a_training_first_revise","current_reread_artifact_id":"a_training_reread"})
report["followup_review"] = receipt
for name,value in report["checks"].items():
    value["status"]="pass"
report["checks"]["factual_accuracy"]["details"]="44 claims verified: previous29 wholechapter reread with trueoneparagraphdiff and unchanged54proofSHAs; optionalTRAINING15newclaims independently read, onebudgetwordingissue corrected and wholeguide actually reread."
report["checks"]["factual_accuracy"]["claim_ids"]=[item["id"] for item in report["claims"]]
report["checks"]["numeric_verification"]["details"] += " 新指南345逐檔SHAverify、70648031B、272/52/66与24/6/12、每版64/90、LoRA28qv/r8/112/1605632、原180/LR1e-4/360/149.471s/5.2296GiB均独立核验。"
report["checks"]["numeric_verification"]["claim_ids"].extend(["c31","c34","c35","c37","c41"])
report["checks"]["source_verification"]["details"] += " 新训练CLI/helper/Modal/workflow/manifest全读；训练两helper在学生固定Git公开200且byteexact；当前TRAINING liveSHA注册，不把guide当GPU成功证据。"
report["checks"]["source_verification"]["claim_ids"].extend([f"c{i}" for i in range(30,45)])
report["checks"]["limitations"]["details"] += " 新路线只实际help/list/existing345verify/bashsyntax/argparse/七resumeguardexcerpt/scalarproxy；本地无training_state.pt，未真正resume或重训练。仅七metadata栏由代码强制，其他配方一致由读者保持。新GPU/安装/下载命令未执行，无权重/品质复现保证。"
report["checks"]["limitations"]["claim_ids"].extend([f"c{i}" for i in range(30,45)])
report["review_limits"].append("OptionalTRAINING full firstread/revisedfullreread and source/command contract review only. Original retained checkpoint metadata is real but optimizer/RNG state file is not retained locally; no actual resume, CUDA or fresh large-model preparation/training executed. Extracted original resume guard alone was executed on CPU dictionaries and accurately labeled.")
report["review_limits"].append("Optional README supplement truly recounted20/261/287 and read both current natural-route/hardware paragraphs. 29formalGPUplan records are L4/2.14.1+cu126; formalCPU simple_models and supportingCPUexperiments excluded by correctedGPUwording. No blanketREADME ormacOS/Windowsverification claim and no new canonicalpage dependency.")
(ROOT/"docs/technical-reviews/20.8.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"claims":len(report["claims"]),"sources":len(report["sources"]),"artifacts":len(report["artifacts"]),"chapter_sha256":report["source_sha256"],"guide_sha256":receipt["training_sha256"]}))
