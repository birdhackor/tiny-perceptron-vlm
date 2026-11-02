"""Bounded post-current historical inspection; no kernels, inference, training or finals."""
from pathlib import Path
from datetime import datetime,UTC
from collections import Counter
import json,hashlib,copy,platform,sys,importlib.util,re
ROOT=Path("/tmp/tpb").resolve()
ROUND="docs/course-revision-20261101-c-v3"
ART=ROOT/"docs/technical-reviews/artifacts/c-v3-v21-canonical-validation"
ACTOR="/root/c_v3_technical_coord/canonical_validation"
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:(ROOT/p).read_bytes()
obj=lambda p:json.loads(read(p))
ref=lambda p:{"path":p,"sha256":sha(read(p))}
cache={};authenticated={}
def readref(r,parse=True):
 raw=read(r["path"]);assert sha(raw)==r["sha256"],r["path"]
 authenticated[r["path"]]={"path":r["path"],"sha256":sha(raw),"bytes":len(raw)}
 if not parse:return None
 if r["path"] not in cache:cache[r["path"]]=json.loads(raw)
 return cache[r["path"]]
def pointer(o,p):
 for token in p.split("/")[1:]:
  token=token.replace("~1","/").replace("~0","~")
  o=o[int(token)] if isinstance(o,list) else o[token]
 return o
started=datetime.now(UTC).isoformat()
sourcefreeze={"path":ROUND+"/source-freeze-v22.json","sha256":"0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59"}
freeze=readref(sourcefreeze)
page=next(p for p in freeze["pages"] if p["page_id"]=="validation")
body=read(page["snapshot"]);assert body==read(page["source"]) and sha(body)==page["source_sha256"]=="0c3302d961be0600ce7f3f40ba7ac0db0b9a14c8de599f7d196a34131f6adb05"
original=ref(ROUND+"/technical-v21/canonical_validation/original-v1.json")
current=ref(ROUND+"/technical-v21/canonical_validation/current-v1.json")
assert original["sha256"]=="921003bde79f37ec87dd678e83ea418300bd39ba8a85c3416ba90b0673560fe3"
assert current["sha256"]=="35900f37f9f4651c35fde62e5c664326353dc16a005fe088219679451511dffe"
full=obj(current["path"])
for source in full["sources"]:
 if source["kind"]=="repository_code":assert sha(read(source["path"]))==source["sha256"],source["path"]
for artifact in full["artifacts"]:assert sha(read(artifact["path"]))==artifact["sha256"],artifact["path"]
packets={}
expected={6:"383b871ee366a7466b6c52294d1fa59fc901c68c4289f1f33e2d37a462d1195e",7:"66b9cea702a3ceb0d4c27eff8c2c9aae718cb51478626c0c221f362e4c095977"}
identities={}
for group in (6,7):
 packetref={"path":ROUND+f"/technical-coordination-v21/history-prompts/child-00-group-{group:02}-historical-identities.json","sha256":expected[group]}
 packet=readref(packetref);packets[group]=packet
 assert packet["source_group_index"]==group and packet["page_id"]=="validation"
 readref(packet["source_chain"],False);readref(packet["prior_active_pointer"],False)
 unavailable=pointer(readref(packet["unavailability"]["report"]),packet["unavailability"]["json_pointer"])
 rows=[]
 for row in packet["historical_identities_only"]:
  issue=pointer(readref(row["original_report"]),row["json_pointer"])
  disposition=pointer(readref(row["disposition_ref"]),row["disposition_ref"]["json_pointer"])
  assert disposition["issue_id"]==row["issue_id"]
  assert disposition["final_necessity"]==row["original_necessity"]
  fix=pointer(readref(row["historical_fix_verification"]["report"]),row["historical_fix_verification"]["json_pointer"])
  readref(row["prior_appendix_row"]["report"],False)
  rows.append({"identity":copy.deepcopy(row),"original_issue":issue,"accepted_necessity":disposition["final_necessity"],"accepted_reason":disposition.get("reason"),"original_fix_scope":fix.get("reader_summary",fix.get("review_scope"))})
 identities[group]=rows
 assert len(rows)==(1 if group==6 else 24)
techfix=readref(packets[7]["historical_identities_only"][0]["historical_fix_verification"]["report"])
fixarts={a["id"]:a for a in techfix["artifacts"]}
for identifier in ("b1","b2","d3"):readref(fixarts[identifier],False)
b1=obj(fixarts["b1"]["path"]);b2=obj(fixarts["b2"]["path"]);d3=obj(fixarts["d3"]["path"])
# Read raw old fix evidence, not old pass/still_fixed fields.
nbcode=read("scripts/check_notebooks.py").decode()
assert "NotebookClient(" in nbcode and '"failed": len(failures)' in nbcode and "raise SystemExit(1)" in nbcode
assert "validation-{args.mode}.json" in nbcode
text=body.decode()
for token in ("exit code（結束碼）","0表示成功","不是0就是沒有通過","echo $?","$LASTEXITCODE","failed","failures","outputs/notebooks/validation-kernel.json","pytest -ra","通過、失敗與跳過"):
 assert token in text,token
probe=b2["isolated_probe"];summary=probe["summary"]
assert b2["real_notebook_command"]["exit_code"]==0 and b2["real_notebook_summary"]["failed"]==0
assert probe["exit_code"]==1 and summary["passed"]==2 and summary["failed"]==1 and summary["failures"]
observed=[]
for name in ("one","two"):
 nb=probe["successful_notebooks"][name]
 stream="".join(o.get("text","") for c in nb["cells"] for o in c.get("outputs",[]) if o.get("output_type")=="stream")
 pid,marker,device=stream.strip().split();assert marker=="1" and device=="cpu";observed.append({"name":name,"pid":int(pid),"marker":marker,"device":device})
assert observed[0]["pid"]!=observed[1]["pid"]
readref(b2["real_notebook"],False)
# Read original stage/trainer/execution receipts and only relevant fields; never weights.
manifest_path="docs/selftrained/v2-manifest.json"
manifest=obj(manifest_path)
assert sha(read(manifest_path))==b1["input_files"][manifest_path]["sha256"]
record_hashes={Path(r["path"]).name:r["sha256"] for r in manifest["records"]}
stage_rows={};run_best={}
for name,oldstage in b1["training_stages"].items():
 base=f"docs/selftrained/results/training-raw/{name}/"
 outerpath=base+"receipt.json";trainpath=base+"raw/train-receipt.json";execpath=base+"raw/execution.json"
 for path in (outerpath,trainpath):
  e=b1["input_files"][path];assert sha(read(path))==e["sha256"] and len(read(path))==e["bytes"]
 outer=obj(outerpath);train=obj(trainpath);execution=obj(execpath)
 files={Path(f["path"]).name:f for f in outer["files"]}
 assert sha(read(trainpath))==files["train-receipt.json"]["sha256"]
 assert sha(read(execpath))==files["execution.json"]["sha256"]
 assert train["origin"]["kind"]=="all-neural-weights-random"
 assert train["steps"]==oldstage["training"]["steps"]
 assert train["data_sha256"]==record_hashes and train["test_used_for_selection"] is False
 assert train["completed_requested_steps"] is True and train["interrupted"] is False
 assert outer["run_id"]==execution["run_id"]==oldstage["receipt"]["run_id"]
 parent=oldstage.get("parent")
 run_best[outer["run_id"]]=files["best.pt"]["sha256"]
 stage_rows[name]={"receipt":ref(outerpath),"trainer":ref(trainpath),"execution":ref(execpath),"run_id":outer["run_id"],"steps":train["steps"],"train_records":train["train_records"],"validation_records":train["validation_records"],"origin":train["origin"]["kind"],"stage_history":[{"stage":s["stage"],"step":s["step"],"checkpoint_sha256":s["checkpoint_sha256"]} for s in train["stage_history"]],"selected_best_sha256":files["best.pt"]["sha256"],"safe_weights_sha256":files["model.safetensors"]["sha256"],"parent":parent,"completed_requested_steps":train["completed_requested_steps"],"test_used_for_selection":train["test_used_for_selection"]}
assert len(stage_rows)==15
for name,row in stage_rows.items():
 if row["parent"] and "run_id" in row["parent"]:
  assert row["parent"]["sha256"]==row["parent"]["parent_best_sha256"]==run_best[row["parent"]["run_id"]]
  assert row["stage_history"][-1]["checkpoint_sha256"]==row["parent"]["sha256"]
 else:
  assert row["parent"]=={"none":True,"pretrain_root":True} and not row["stage_history"]
for name in ("moe-joint","moe-native","dense-joint","dense-weighted"):
 assert (stage_rows[name]["train_records"],stage_rows[name]["validation_records"])==(28876,2435)
selection_path="docs/selftrained/results/v2-final-training-source-selection.json"
selection=obj(selection_path)
assert sha(read(selection_path))==b1["input_files"][selection_path]["sha256"]
final_rows={}
for arch,stage in (("moe","moe-native"),("dense","dense-weighted")):
 path=f"docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json"
 data=obj(path);assert sha(read(path))==b1["input_files"][path]["sha256"]
 checkpoint=data["checkpoint"]
 assert data["completed_count"]==data["expected_count"]==3734
 assert sum(v["count"] for v in data["per_task_final_reply"].values())==3734
 assert checkpoint["selected_step"]==1000
 assert checkpoint["sha256"]==stage_rows[stage]["selected_best_sha256"]
 assert checkpoint["safe_weights_sha256"]==stage_rows[stage]["safe_weights_sha256"]
 assert checkpoint["source_training_run"]==stage_rows[stage]["run_id"]
 assert checkpoint["completed_requested_steps"]==stage_rows[stage]["steps"]
 final_rows[arch]={"index":ref(path),"run":data["run"],"count":data["completed_count"],"checkpoint":checkpoint}
assert stage_rows["moe-native"]["steps"]==4000 and stage_rows["dense-weighted"]["steps"]==10000
assert selection["moe_joint"]["selected_step"]==selection["dense_joint"]["selected_step"]==1000
assert selection["moe_joint"]["sha256"]==stage_rows["moe-native"]["selected_best_sha256"]
assert selection["dense_joint"]["best_sha256"]==stage_rows["dense-weighted"]["selected_best_sha256"]
evalcode=read("scripts/selftrained/evaluate.py").decode()
assert "class EvaluationJournal" in evalcode and 'with test_marker.open("x"' in evalcode
assert "唯一 prefix" in evalcode and "既有 test marker 協定不同" in evalcode
# Fix c45: bounded actual freeze/final/publication identity, not unknowable full post-final history.
freeze_path="docs/lesson-products/results/settings-freeze-v1.json"
final_path="docs/lesson-products/results/final-report-v1.json"
package_path="docs/lesson-products/results/publication-package-v1.json"
settings=obj(freeze_path);final=obj(final_path);package=obj(package_path)
assert sha(read(freeze_path))==final["settings_freeze_sha256"]
package_settings=next(v for k,v in package["files"].items() if Path(k).name=="settings-freeze.json")
package_report=next(v for k,v in package["files"].items() if Path(k).name=="report.json")
assert package_settings["sha256"]==sha(read(freeze_path)) and package_report["sha256"]==sha(read(final_path))
for product,files in settings["products"].items():
 for name,digest in files.items():assert package["files"][product+"/"+name]["sha256"]==digest
assert "評分後沒有重訓或調參" not in text
# Fix c46: read the four named original execution records; do not rerun their model/statistics.
audit_records={}
for kind in ("b-mechanisms","c"):
 old=b1["two_saved_recalculation_records"][kind]
 originalpath=old["original_path"];executionpath=old["execution_path"]
 for p in (originalpath,executionpath):assert sha(read(p))==b1["input_files"][p]["sha256"]
 orig=obj(originalpath);run=obj(executionpath)
 assert orig["reviewer_task"]==run["reviewer_task"]
 assert run["command"] and run["environment"]
 audit_records[kind]={"original":ref(originalpath),"execution":ref(executionpath),"reviewer_task":run["reviewer_task"],"command":run["command"],"environment":run["environment"],"scope":run.get("scope"),"boundary":run.get("boundary"),"verification":orig.get("verification"),"raw_reference":run.get("raw_reference")}
assert audit_records["b-mechanisms"]["reviewer_task"]!=audit_records["c"]["reviewer_task"]
rawsha="8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896"
assert audit_records["b-mechanisms"]["verification"]["raw_sha256"]==rawsha
assert audit_records["c"]["raw_reference"]["sha256"]==rawsha
assert audit_records["b-mechanisms"]["verification"]["group_tables_recomputed"]==206
assert audit_records["b-mechanisms"]["verification"]["metric_cells_compared"]==1236
# One genuine bounded inspection is reused across the four original/current identities per fix.
fixes={
 "reader":{"location":"docs/validation.md:L100-104","claim":"命令完成、失敗與跳過的判尺直接留在正文。","observed":"L102定義exit code及0/非0、查看時機、兩平台查法、failed/failures/validation-kernel.json、pytest通過/失敗/跳過與原因；L100分開Notebook工作產物與模型分數。","reason":"原reader缺口是只給命令而沒有可觀察完成判尺。現行L102逐一指出結束碼、JSON欄位/保存路徑與pytest摘要，且原修正reader紀錄明示只判閱讀、不稱實跑。本人讀現文並核check_notebooks.py與原成功/刻意失敗probe的實際JSON/exit契約；不要求貼全log或新增訓練。","sources":["h_text","h_nb","h_inspection"]},
 "c03":{"location":"docs/validation.md:L11-12、L95-102；scripts/check_notebooks.py:run_kernel/main","claim":"kernel隔離、CPU、結果保存及failed/failures/exit規則有實際原始執行支持。","observed":"原成功課程Notebook1/1、exit0；兩隔離probe PID不同、marker=1、device=cpu；刻意失敗2passed/1failed、exit1與failure細節保存。現行run_kernel每次新NotebookClient，main保存validation-kernel.json並在failures時exit1。","reason":"原缺口是當時本機缺notebook依賴而未能實核，不是要求重跑全書。已親讀原fix b2真command/environment/成功副本/隔離probe與故障輸出，並讀現行run_kernel/main；程式指紋仍相同。312是原當輪全書摘要；source21的328+4保留鏈亦保留其版本限制，不冒稱本輪重跑NN或source22全部新執行。","sources":["h_text","h_nb","h_inspection"]},
 "c09":{"location":"docs/validation.md:L24、L64-66；scripts/selftrained/evaluate.py:EvaluationJournal/test marker","claim":"v2自家隨機初始化至公開權重與固定final的保存譜系有原收據閉合。","observed":"15段原outer/trainer/execution bytes/SHA與原fix一致；random origin、parent best與各原stage_history相接；兩固定final index各3734且best/safe weights接所選父訓練。現行journal限定同條件唯一prefix與test marker排他開啟。","reason":"原疑問是隨機初始化程式/作者摘要不足以證明保存的歷史譜系。本人核15段具名原始receipt/train-receipt/execution與父best鏈、公開來源選定及兩final index，另讀現行journal/marker實作；只支持已保存管線與同協定guard，不推未記錄私下行為不可能發生，也未重訓/重測。","sources":["h_text","h_eval","h_inspection"]},
 "c45":{"location":"docs/validation.md:L34","claim":"B/C段只主張評分、freeze與公開套件的具體內容指紋同一。","observed":"settings完整SHA1a221870…與final.settings_freeze_sha256及公開package相等；final-report完整SHA亦與package相等，10個B/C凍結產品檔指紋一致。無「評分後沒有重訓或調參」無界句。","reason":"原缺口針對不可由freeze+單份final自證的無界歷史斷言。現文L34已限縮為指定freeze/final/package指紋；本人對完整JSON算SHA並逐個比對B/C產品檔身份，不以這些檔案相同推論整個開發史或能力。","sources":["h_text","h_inspection"]},
 "c46":{"location":"docs/validation.md:L34","claim":"兩份歷史重算有不同記錄者、實際command/environment及相同保存raw指紋，範圍不是新模型測試。","observed":"原及execution分別同owner /root/tech_20261016_19_15、/root/tech_20261016_19_16；兩owner不同、命令/版本具體，rawSHA皆8b593421…，B206表/1236cells；原scope保留只讀保存raw且未重跑正式模型。","reason":"原缺口是只有逐題檔指紋而未閉合兩人的實際離線執行。本人讀四個具名original/execution原件並核原fix記錄的bytes/SHA、actor配對、command/environment與raw reference；只對本句描述的保存重算紀錄成立，不把舊pass當新教材審閱、也不新呼叫模型或重做其統計。","sources":["h_text","h_inspection"]},
 "c52":{"location":"docs/validation.md:L66","claim":"28876/2435訓練與validation列數以及每版3734 final來自實際原trainer/index。","observed":"moe-joint/moe-native/dense-joint/dense-weighted原trainer各train_records28876、validation_records2435；12個data_sha256與固定manifest相等；兩final count及per-task總和各3734。","reason":"原缺口是當時只有LFS pointer、未讀實際分母。本人讀四份原joint trainer字段和manifest-bound12個資料SHA，再對兩版原final index逐task count加總；沿原固定trainer讀取後分母，沒有下載LFS或解包重數，題目列數不冒稱獨立照片/錄音/家族。","sources":["h_text","h_inspection"]},
 "c53":{"location":"docs/validation.md:L66","claim":"完成步數與選定存檔步數分開由原trainer及selection/final checkpoint綁定。","observed":"原moe-native trainer steps4000、dense-weighted steps10000，completed_requested_steps=true且interrupted=false；來源selection/final checkpoint各selected_step1000，best與safe weights SHA一致。","reason":"原缺口要求以原始實際trainer/checkpoint metadata分清完成與選定。本人讀4000/10000完成字段及未中斷/完成紀錄，另核source-selection與兩final checkpoint的1000-step best SHA；不是拿job要求步數或best.pt檔名代替完成，也未載入權重/訓練。","sources":["h_text","h_inspection"]}
}
inspection={"schema":"validation-bounded-historical-inspection-v1","reviewer_task":ACTOR,"started_at":started,"completed_at":datetime.now(UTC).isoformat(),"physical_root":str(ROOT),"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"device":"cpu; byte/JSON/source inspection only"},"authorization_and_pause":{"formal_dispatch_window_as_supplied":{"not_before":"2026-11-01T23:04:19+00:00","not_after":"2026-11-01T23:04:41+00:00"},"resume_observed_at":started,"note":"Initial bounded source reads occurred before a coordinator-reported interruption. User explicitly resumed with 那你繼續. No historical appendix had been persisted at resume. This genuine final inspection/persistence uses current actual times; no time backfill or claim of uninterrupted work."},"source_freeze":sourcefreeze,"current_source_sha256":page["source_sha256"],"current_figure_sha256":page["figures_sha256"],"original":original,"current":current,"packet_refs":[{"path":ROUND+f"/technical-coordination-v21/history-prompts/child-00-group-{g:02}-historical-identities.json","sha256":expected[g],"source_group_index":g} for g in (6,7)],"authenticated_original_refs":list(authenticated.values()),"authenticated_issue_rows":{str(g):identities[g] for g in (6,7)},"original_fix_artifacts":[{k:a[k] for k in ("id","path","sha256","kind")} for a in (fixarts["b1"],fixarts["b2"],fixarts["d3"])],"underlying_fixes":fixes,"kernel_original_probe":{"successful":observed,"real_command":b2["real_notebook_command"]["argv"],"real_exit":b2["real_notebook_command"]["exit_code"],"failure_exit":probe["exit_code"],"failure_summary":{"total":summary["total"],"passed":summary["passed"],"failed":summary["failed"]},"scope":b2["scope"]},"training_original_receipts":stage_rows,"final_original_indices":final_rows,"freeze_publication_identity":{"settings":ref(freeze_path),"final_report":ref(final_path),"publication_package":ref(package_path),"product_files_compared":sum(len(v) for v in settings["products"].values())},"two_original_recalculations":audit_records,"scope":"Only1 reader obligation and6 underlying technical fixes represented by24 original/current identities. Original issue/fix/authentication refs and accepted necessity preserved. No new unavailable event, old still_fixed verdict copied, model loading, kernels, NN inference, training, final opening, full study recomputation, publication or protected-file edits."}
inspection["inspection_script_revision_note"]="v1 stopped before output because its generic parent assertion did not distinguish two pretrain-root marker objects. v2 then stopped before output on assumed archive member aliases. Exact v1/v2 code remain. v3 validates random-root markers separately and uses the actual settings-freeze.json/report.json package aliases; no source/evidence edits."
inspection["command"]="PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-validation/historical-inspection-v3.py"
out=ART/"historical-inspection-v1.json"
with out.open("x",encoding="utf-8") as f:f.write(json.dumps(inspection,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"path":out.relative_to(ROOT).as_posix(),"sha256":sha(out.read_bytes()),"started_at":started,"completed_at":inspection["completed_at"],"groups":[{"source_group_index":g,"rows":len(identities[g])} for g in (6,7)],"raw_training_stages":len(stage_rows),"underlying_technical_fixes":6,"reader_obligations":1},ensure_ascii=False))
