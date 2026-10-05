"""Produce the complete revised report from this reviewer's own initial report only."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = OUT.parent
ROOT = ART.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = json.loads((OUT / "own-initial-revise.json").read_bytes())
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_12_9"
assert sha(OUT / "own-initial-revise.json") == "61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38"
receipt = json.loads((OUT / "resolution-receipt.json").read_bytes())
assert receipt["source_sha256"] == "0930ed04d140c96d076a3ca8af5d3334f20d684293d2f6eaff8f04830d991149"
new_artifacts = []
ids = {}
for p in sorted(OUT.iterdir()):
    if not p.is_file() or p.name.startswith(("checker.", "report-generation.", "final-manifest")):
        continue
    local = p.name
    identifier = "a-recheck-" + local.replace(".", "-")
    ids[local] = identifier
    item = {"id": identifier, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p),
        "kind": "code" if p.suffix == ".py" else "figure_render" if p.suffix == ".png" else "derivation" if local == "inspection.md" else "source_snapshot",
        "description": "12.9 本人新版複查正式證據：" + local + "；原问题与 initial revise 保留，新版原 bytes、实际 raw pointers、画面和解决依据分别记录。"}
    if local == "resolution-receipt.json":
        item.update(kind="execution", command=".venv/bin/python docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/recheck.py > docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/recheck.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/recheck.stderr.txt",
            result="exit 0；重新核原 raw JSON 得11/14、row4/6/7=290/300/300Hz，各frequency的振幅/时长完全成对；原fence/code/JSON/前置图SHA不变。",
            environment={"python":"3.13.5","device":"CPU, JSON-only"})
    if local == "render-receipt.json":
        item.update(kind="execution", command=".venv/bin/python docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/render_current.py > docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/render.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_9-independent/recheck/render.stderr.txt",
            result="exit 0；新版current source artifact按原bytes渲染，1280x800/390x844已实际view，改正文句与原fence匹配；不声称production parity。",
            environment={"python":"3.13.5","device":"CPU","browser":"Chromium 151.0.7922.173 /usr/bin/chromium --no-sandbox"})
    new_artifacts.append(item)
report["artifacts"] += new_artifacts
report["sources"].append({"id":"exec-recheck","kind":"execution","title":"本人新版原资料与修正文句複查","verified":True,"artifact_id":ids["resolution-receipt.json"]})
claim = next(c for c in report["claims"] if c["id"] == "duration-scope")
claim.update(status="verified", statement="三個錯例的頻率都是290或300Hz；資料將振幅與時長一起改動，不能分辨兩者各自影響。",
    location="course/chapters/12.md:295，新版三错频率与成对设计限制句",
    scope="只陳述本次原始三個測試錯例與資料設計的識別限制，沒有聲稱獨立時長或振幅效果。")
claim["evidence"] = [
    {"source_id":"exec-recheck","locator":"audio.json /results/test/samples/4,/6,/7; /results/data/splits/*/records; recheck/resolution-receipt.json /failed_rows,/paired_design", "supports":"本人由全部原generated_ids重算11/14，定位三错为290/300/300Hz；每个split的每个frequency仅有(0.25,0.1),(0.5,0.12)两配对，无法分别识别两因素。"},
    {"source_id":"repo-recipe","locator":"_audio_records195-215；本轮current SHA与初轮亲读原码相同", "supports":"按每个频率成对构造振幅与秒数，原构造契约与原records一致；不是只改时长的控制。"},
]
claim["artifact_ids"] += [ids["resolution-receipt.json"], ids["section.md"], ids["inspection.md"]]
claim["verification"].update(expected="应有三个错例，均在290或300Hz；每个frequency仅有振幅/时长成对条件。", observed="本人原JSON重算11/14、错误rows4/6/7的频率为290/300/300Hz；每split每frequency的pair精确为(0.25,0.1),(0.5,0.12)。", details="本轮JSON-only重算与原records核对；修订现在只记观察及完全共变设计限制，未再归因独立时长；没有模型重评。")
report["source_sha256"] = receipt["source_sha256"]
report["verdict"] = "pass"
report["read_scope"] = "本人初轮完整查证后，本轮重新亲读新版12.9全文，并逐一核自己的七项claim支持范围；重新检查必要原JSON pointers及不变原码/fence/前置图SHA。完整独立记录在recheck/inspection.md。"
report["current_input"] = {"path":(OUT/"section.md").relative_to(ROOT).as_posix(),"sha256":sha(OUT/"section.md"),"meaning":"本轮重新亲读的当前小节原UTF-8 bytes；初轮完整Markdown frozen_input字段保留其历史意义。"}
issue = report["issues"][0]
issue.update(status="resolved", resolution="本人亲读新版原段落，并重新从原generated_ids核11/14与三错290/300/300Hz；每个split每frequency的振幅/时长完全成对。新版只记这些观察与无法分辨两因素的限制，已消除原时长归因超出证据问题；详recheck/resolution-receipt.json。", initial_status="unresolved", initial_verdict="revise", original_source_sha256="409619baa10be2b64f80d549d541280646be65ae3ffa8ff7f97fba539d48d378", resolved_source_sha256=receipt["source_sha256"], resolution_artifact_id=ids["resolution-receipt.json"])
for name, text in {
    "factual_accuracy":"七项实质claims全部本人核实；原duration-confound保留为resolved，新版只给三错频率与成对设计限制。",
    "limitations":"新版准确限制观察为人工单音历史run及振幅/时长共变；原示范仍只forward/backward。没有模型重评或训练，原问题与初稿revise历史保留。",
    "source_verification":"初轮本人亲读权威原文与实作；本轮核必要原码、rawJSON、原fence及前置图SHA相同，真正重新查看具名JSON pointers，不沿用协调者结论。",
}.items():
    report["checks"][name].update(status="pass", details=text)
report["checks"]["figure_consistency"]["details"] = "本节仍无图；初轮现页和前置图render/view证据保留，本轮另外render/view新版current source artifact的桌面/手机全页截图。该artifact明标新SHA，不宣称8765或production parity。"
report["checks"]["numeric_verification"]["details"] += "本轮重新核原JSON三错rows4/6/7与290/300/300Hz、成对条件；未改的计算/fence按相同原码与原bytes保留初轮真实执行支持范围。"
report["checks"]["numeric_verification"]["claim_ids"] += ["duration-scope"]
report["recheck_history"] = [{"initial_report_path":"docs/technical-reviews/history/phase4-12_9-own-initial-revise-61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38.json", "initial_report_sha256":"61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38", "own_initial_copy":(OUT/"own-initial-revise.json").relative_to(ROOT).as_posix(), "source_before":"409619baa10be2b64f80d549d541280646be65ae3ffa8ff7f97fba539d48d378", "source_after":receipt["source_sha256"], "verdict_before":"revise", "verdict_after":"pass", "resolution_artifact_id":ids["resolution-receipt.json"], "reviewer_task":report["reviewer_task"]}]
report["limitations"].append("本轮画面是明确标示的current source artifact；未验证production/live page parity。只copy明确小型文件，未copytree运行workspace或跟随symlinks。")
path = ROOT / "docs/technical-reviews/12.9.json"
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert json.loads(path.read_bytes())["reviewer_task"] == "/root/phase4_factual_coordinator/factual_12_9"
print(json.dumps({"report":path.relative_to(ROOT).as_posix(),"verdict":"pass","report_sha256":sha(path),"source_sha256":report["source_sha256"],"callback_artifact_id":ids["resolution-receipt.json"],"callback_artifact_sha256":sha(OUT/"resolution-receipt.json")},ensure_ascii=False))
