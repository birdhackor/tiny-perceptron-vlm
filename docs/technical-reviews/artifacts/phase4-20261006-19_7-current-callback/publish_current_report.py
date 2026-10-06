"""Carry immutable own evidence, add independently performed current inspection."""
import hashlib
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
TASK = "/root/phase4_factual_coordinator/factual_19_7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


# This same original reviewer mechanically carries its unchanged claim/evidence
# registry. No prior verdict is used: the current decision was reached separately
# after the full current section, true diff, exact fingerprints and raw leaves.
opaque = ART / "opaque-history/19.7-prior-canonical.json"
assert sha(opaque) == "8ca7d27615e0b671ea5018bc7704d5435706e2da64406eb08c348b166aa1404a"
prior = json.loads(opaque.read_bytes())
assert prior["reviewer_task"] == TASK
assert prior["source_sha256"] == "713f6d3ddd667feedd5c4053db4641926484e3e94c943df0f9482670e39ca35c"
inspection_file = ART / "current-inspection.json"
inspection = json.loads(inspection_file.read_bytes())
assert inspection["reviewer_task"] == TASK and inspection["verdict"] == "pass"
assert inspection["unresolved_issues"] == []
assert inspection["current_section_sha256"] == "c8ab7022b13b34697ddecb5b36eabeb1cca228dce415d538d60d21e3e9396c72"
assert inspection["website"]["response_equals_own_file"] is True
artifacts = prior["artifacts"]


def add(identifier, filename, kind, description, **extra):
    p = ART / filename
    artifacts.append({"id": identifier, "path": rel(p), "sha256": sha(p), "kind": kind, "description": description, **extra})


add("current-inspection-19_7", "current-inspection.json", "derivation", "原技術審閱者本人讀完整current19.7與真frozen diff，核改字詞的支持scope及必要原始EOS/content leaves、own網站段落。")
add("current-section-19_7", "inputs/section-current.md", "source_snapshot", "這次實際完整閱讀的19.7 UTF-8原始bytes。")
add("current-frozen-chapter-19_7", "inputs/19-current-frozen.md", "source_snapshot", "本次初讀全章bytes的frozen input；只讀19.7，不能代表目前整章或章導言已讀。")
add("current-input-manifest-19_7", "current-input-manifest.json", "source_snapshot", "current與prior section、真正閱讀scope、必要ctx=[]及intro未讀的原始SHA記錄。")
add("current-true-diff-19_7", "section-frozen-diff.patch", "derivation", "本人從原凍結section bytes與current bytes產生並閱讀的真diff；僅隻差→只差。")
add("current-reuse-fingerprints-19_7", "fingerprint-reuse-receipt.json", "derivation", "14項live method/source/figure hashes與54份本人原證據精確bytes核對；明示沿用scope，沒有重跑原CPU或重抓paper。")
add("prior-opaque-canonical-19_7", "opaque-history/19.7-prior-canonical.json", "source_snapshot", "本人的完整prior canonical opaque原始檔；保留確切SHA，未用prior判定代替本次查證。")
add("prior-opaque-preservation-19_7", "opaque-preservation-receipt.json", "source_snapshot", "本人prior canonical及54份priorproof的完整opaque副本精確FILE/SHA。")
add("current-inspection-code-19_7", "inspect_current.py", "code", "本輪scalar hash/JSON/EOS/token bytes核對；未開模型。")
add("current-website-code-19_7", "inspect_website.py", "code", "只取得當前own19.7路由及檢查改字詞段落；無root/site全域parity或repair。")
add("current-publisher-code-19_7", "publish_current_report.py", "code", "本原審閱者完整current callback authoring程序；prior verdict未作current證據。")
command = json.loads((ART / "current-command-receipt.json").read_bytes())
add("current-execution-19_7", "current-inspection-stdout.txt", "execution", "本輪實際scalar CPU current inspection stdout。", command=shlex.join(command["argv"]), result="exit0；真diff僅隻差→只差；兩筆生成有EOS且内容為0/111、真工具回1；14 live sources及54 priorproof精確指紋不變。", environment=command["environment"])
add("current-command-19_7", "current-command-receipt.json", "execution", "本輪實際argv/cwd/exit/time/code/stdout/stderr/env/hash。", command=shlex.join(command["argv"]), result="exit_code=0", environment=command["environment"])
add("current-stderr-19_7", "current-inspection-stderr.txt", "source_snapshot", "保留本輪實際空stderr。")
website_command = json.loads((ART / "website-command-receipt.json").read_bytes())
add("current-website-execution-19_7", "own-website-stdout.txt", "execution", "本人實際own page HTTP200、response SHA與必要paragraph核對stdout。", command=shlex.join(website_command["argv"]), result="exit0；outputs/site/19.7.html與本人取得response exact bytes一致；改後只差段落與原SVG引用一致；未rerender。", environment=website_command["environment"])
add("current-website-command-19_7", "website-command-receipt.json", "execution", "own網站取得/解析實際命令與code/stdout/stderr/env/hash。", command=shlex.join(website_command["argv"]), result="exit_code=0", environment=website_command["environment"])
add("current-website-receipt-19_7", "own-website-receipt.json", "derivation", "本人own網站檢查scope、local/HTTP fingerprints及實際改後paragraph；無全頁/新visual判定。")
add("current-website-snapshot-19_7", "inputs/19.7-current-own-page.html", "source_snapshot", "本輪實際own網站原始HTML永久副本；只語義讀改後paragraph及own SVG img。")
add("current-http-snapshot-19_7", "inputs/19.7-current-http-response.html", "source_snapshot", "本人當次HTTP取得的own19.7完整response raw bytes。")
add("current-website-stderr-19_7", "own-website-stderr.txt", "source_snapshot", "保留own網站檢查實際空stderr。")

sources = prior["sources"]
sources.append({"id": "current-eos-scalar-run-19_7", "kind": "execution", "title": "本人current字詞修正的原始EOS/content scalar核對", "verified": True, "artifact_id": "current-execution-19_7"})
sources.append({"id": "current-website-run-19_7", "kind": "execution", "title": "本人own19.7當前網站paragraph檢查", "verified": True, "artifact_id": "current-website-execution-19_7"})
claims = prior["claims"]
claims.append({
    "id": "current-eos-content-19_7", "kind": "empirical", "status": "verified",
    "statement": "當前修正後『兩次最後生成都有EOS，所以不是只差一個結束符號』仍由兩筆既有原始trace支持；此次隻→只校正不改變實質claim。",
    "location": "19.7最後測試段落，『所以不是只差一個結束符號』；own outputs/site/19.7.html相同段落。",
    "scope": "只回查原始test-joint records18/20必要EOS/content/runtime測量leaves與當前own paragraph；12/12及10/12等未變主張由本人原始精確版本證據沿用。沒有新增模型測量、全頁視覺或他節依賴。",
    "evidence": [
        {"source_id":"test-records","locator":"/records/18及/records/20：final_trace/eos,stop_reason,generated_ids,raw; runtime/status,result; end_to_end_correct","supports":"兩筆最終EOS=True、stop=eos、最後token2；byte decode為DIRECT:0/DIRECT:111；工具真回1且end_to_end=False。live raw與原本人保存bytes一致。"},
        {"source_id":"current-eos-scalar-run-19_7","locator":"current-inspection.json/current_original_measurement_pointers_read及current stdout selected_EOS_measurements","supports":"本人此次實際重新只解碼必要14 named pointers，未用舊判定作證。"},
        {"source_id":"current-website-run-19_7","locator":"own-website-receipt.json/changed_paragraph","supports":"本人當前own page HTTP200 paragraph有只差與兩次EOS敘述；是必要文字實際呈現核對，不是root全域parity。"}
    ],
    "artifact_ids": ["current-inspection-19_7","current-true-diff-19_7","current-reuse-fingerprints-19_7","current-execution-19_7","current-website-execution-19_7","current-website-snapshot-19_7"],
    "verification": {"method":"executed","expected":"唯一改字為隻差→只差；兩筆生成都EOS且仍內容答錯。","observed":"完整真diff僅一字；兩筆EOS=True/stop=eos/token2且raw精確DIRECT:0/DIRECT:111，runtime1。own response200也呈現只差。","details":"Python hash/JSON/byte decode/own HTTP only；未重复原模型CPU、原fence或figure渲染。原11項不變claims的authority/code/CPU證據逐檔SHA核未變而沿用。","denominators":{"changed_failure_records_reinspected":2,"named_original_measurement_pointers":14,"unchanged_complete_calculator_test_denominator":12}}
})

report = {
    "schema_version":1, "review_stage":"technical", "lesson_id":"19.7", "source":"course/chapters/19.md#19.7",
    "source_sha256":inspection["current_section_sha256"], "reviewer_task":TASK,
    "reviewer_context":"fresh", "reviewer_callback":"same original independent technical reviewer; current inspection after localized spelling edit",
    "verdict":"pass", "figure_sha256":prior["figure_sha256"],
    "current_inspection":{"artifact_id":"current-inspection-19_7","path":rel(inspection_file),"sha256":sha(inspection_file),"source_sha256":inspection["current_section_sha256"],"figure_sha256":inspection["current_figure_sha256"],"intro":None,"necessary_context":[],"current_own_website":inspection["website"]["own_page_path"],"current_own_website_sha256":inspection["website"]["own_page_sha256"]},
    "prior_review_preservation":{"canonical_path":rel(opaque),"canonical_sha256":sha(opaque),"proof_manifest_path":rel(ART/'opaque-preservation-receipt.json'),"proof_manifest_sha256":sha(ART/'opaque-preservation-receipt.json'),"scope":"complete prior canonical and54 priorproof files preserved as exact opaque bytes; original initial frozen fingerprints unchanged"},
    "read_scope":{"current":"本人完整current19.7與真frozen19.7 diff；own current網站僅改字詞paragraph與SVG引用。","intro":None,"necessary_context":[],"current_frozen_full_chapter":{"path":rel(ART/'inputs/19-current-frozen.md'),"sha256":sha(ART/'inputs/19-current-frozen.md'),"meaning":"callback initial frozen input only, not latest complete-chapter/read-intro assertion"},"initial_original_read_scope_preserved":"Opaque complete prior report retains initial read scope and frozen whole-chapter meaning; no historical snapshot hash replaced."},
    "review_limits":"本次只scalar CPU hashes/JSON/byte decode及own HTTP/HTML paragraph。原先真核的authority/source/fence/CPU/figure證據exact hashes未變明示沿用。未讀peer verdict、rootparity、repair答案、作者額外結果修正解讀；未開模型/權重、GPU、訓練/下載/fullrecipe或source edit。",
    "sources":sources, "artifacts":artifacts, "claims":claims, "issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"本人完整讀current19.7/真diff，僅隻差→只差；必要rawEOS/content再次核，原11項不變claim原始證據SHA一致而沿用。"},
        "numeric_verification":{"status":"pass","claim_ids":["c6","c7","c8","c11","current-eos-content-19_7"],"details":"本次兩筆EOS與0/111內容scalar decode一致。原始算術/各分母/10/12等不變且原真CPU/supportscope以exact指紋沿用，未重評模型。"},
        "figure_consistency":{"status":"pass","claim_ids":["c1","c11"],"details":"SVG SHA a9c3d469eff7a1377f5e2c53c39583ddab6664c83657e82876e1a9275a82621a與本人前次render/view證據exact一致，current沒有新visual claim；故沿用本人真圖查看，不重畫。"},
        "source_verification":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"14 live依賴source/code/圖及54 priorproof核exact hashes；本次current inspection與own網站bytes正式永久保存。原authority version/定位保留本人真來源證據。"},
        "limitations":{"status":"pass","claim_ids":["c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","current-eos-content-19_7"],"details":"原固定模板/僅歷史checkpoint/回放診斷及成品計畫限制全不變；本次只字詞修正，不泛化新能力或成品驗收，不把舊pass/rootparity當判定。"},
    }
}
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_19_7"
assert report["current_inspection"]["artifact_id"] in {a["id"] for a in report["artifacts"]}
assert report["verdict"] == inspection["verdict"] == "pass"
(ROOT / "docs/technical-reviews/19.7.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({"reviewer_task_asserted":TASK,"source_sha256":report["source_sha256"],"current_inspection":report["current_inspection"],"prior_opaque_file":rel(opaque),"prior_opaque_sha256":sha(opaque),"independent_current_verdict":"pass"},ensure_ascii=False,indent=2))
