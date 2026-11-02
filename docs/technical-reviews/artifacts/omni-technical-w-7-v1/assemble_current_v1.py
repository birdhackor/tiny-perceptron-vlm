"""Save the personally rechecked W.7 current without replacing original/raw."""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FOLDER = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_7"
ORIGINAL = FOLDER / "technical-original-v1.json"
CURRENT = FOLDER / "technical-current-v1.json"
ORIGINAL_SHA = "bd5e74c9bf7cb3f815d8194e06fb97966299075df09ba870f0ad0544486ccf9a"
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-7-v1/audit_w7_current_v1.py"
EXPORT_COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-7-v1/audit_export_context_current_v1.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def main():
    assert not CURRENT.exists()
    assert sha(ORIGINAL) == ORIGINAL_SHA
    original = json.loads(ORIGINAL.read_text())
    assert original["issues"] == []
    current = copy.deepcopy(original)
    routes = json.loads((HERE / "route-inspection-current-v1.json").read_text())
    cpu = json.loads((HERE / "cpu-verification-current-v1.json").read_text())
    callback = json.loads((HERE / "bounded-run-current-v1.json").read_text())
    exporter = json.loads((HERE / "exporter-context-current-v1.json").read_text())
    for name, digest in callback["original_raw_sha256_unchanged"].items():
        assert sha(HERE / name) == digest
    numbered_path = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v11.json"
    numbered = json.loads(numbered_path.read_text())
    page = next(p for p in numbered["pages"] if p["page_id"] == "W.7")
    body = (ROOT / page["snapshot"]).read_text()
    raw = (ROOT / page["source"]).read_text()
    headings = list(re.finditer(r"^## (\S+) .+$", raw, re.M))
    start = next(h for h in headings if h[1] == "W.7")
    end = next((h.start() for h in headings if h.start() > start.start()), len(raw))
    base = raw[:start.start()].count("\n") + 1
    assert raw[start.start():end] == body
    assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"] == original["source_sha256"]
    assert page["figures_sha256"] == {}
    now = datetime.datetime.now(datetime.UTC).isoformat()
    current["created_at"] = now
    current["source_sha256"] = page["source_sha256"]
    current["figure_sha256"] = page["figures_sha256"]
    current["verdict"] = "pass"
    current["issues"] = []
    current["original_issue_resolutions"] = []

    source_changes = []
    for source in current["sources"]:
        if source["kind"] != "repository_code":
            continue
        path = ROOT / source["path"]
        observed = sha(path)
        if source["sha256"] != observed:
            source_changes.append({"path": source["path"], "original_sha256": source["sha256"],
                                   "current_sha256": observed})
        source["sha256"] = observed
        source["version"] = "2026-10-29 本人 bounded current 實讀／重跑時工作樹；HEAD " + cpu["environment"]["head"] + "；本筆 SHA 綁定目前 bytes"
        if source["path"] == "scripts/export_course.py":
            source["inspection_note"] = (
                "本人完整讀目前使用的 exporter 介面與必要控制流程，不以原 792f snapshot 作 current 豁免。"
                "目前 SHA=662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf。"
                "main L486–514 仍預設 ROOT / outputs/site；build L317–341 拒絕 outputs 外目的地，正式分支先呼叫 verify_colab_notebooks。"
                "verify_colab_notebooks L192–232 以真 git cat-file --batch 核 40hex commit 及每份 Notebook bytes；"
                "section_footer L234–287 的 preview 分支明示未發布、下載後到 Colab 首頁 Upload notebook 或 File → Upload notebook，並明示未測雲端。"
                "build L394–402 接正確旗標；metadata L463–476 保存 upload/publication_ready=false/colab_git_objects_verified=false。"
                "本人實呼叫 footer／正式版本拒絕閘／outputs 範圍拒絕閘，且 HTTP 逐331檔核 source13、工作樹、網站、下載 bytes 一致；"
                "真 current HEAD 尚不滿足全部331同bytes，故被拒絕，未發布。詳 a_export。")
        else:
            source["inspection_note"] += " 本次 current 再核目前必要定義／用途與指紋；65 個目標的必要實作路由文字已重新逐項讀取，58 次原樣 AST 對照另真重跑。沒有把連結全章的理論、圖或實測數字冒稱重審。"
    assert source_changes == [{"path": "scripts/export_course.py",
                               "original_sha256": "792f6b269d0c1afd34619d2ea10e1db52a6c19a67ce3365539a612f6ef625874",
                               "current_sha256": "662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf"}]
    current["sources"].append({"id": "run_export_current", "kind": "execution", "title": "本人實核目前 exporter／預覽頁尾／331 Notebook bytes 與正式拒絕閘",
                               "verified": True, "artifact_id": "a_export"})

    # Same IDs now point at genuinely new executions; immutable original report
    # continues to point at the untouched original files.
    current["artifacts"] = [a for a in current["artifacts"] if a["id"] != "a_assembly"]
    for artifact in current["artifacts"]:
        if artifact["id"] in {"a_routes", "a_cpu"}:
            name = "route-inspection-current-v1.json" if artifact["id"] == "a_routes" else "cpu-verification-current-v1.json"
            artifact.update(ref(HERE / name))
            artifact["command"] = COMMAND
            artifact["environment"] = cpu["environment"]
            artifact["description"] = "本次 current 新執行：" + artifact["description"]
        assert sha(ROOT / artifact["path"]) == artifact["sha256"]

    for identifier, path, kind, description in [
        ("a_current_code", HERE / "audit_w7_current_v1.py", "code", "本人 current 重跑入口：沿用不可變原算法，但明確選 v11 freeze／snapshot，所有輸出 exclusive-create 新檔並核原件未變。"),
        ("a_current_run", HERE / "bounded-run-current-v1.json", "execution", "本次 v11 的實際重跑方法、freeze 與原件保留指紋收據。"),
        ("a_export_code", HERE / "audit_export_context_current_v1.py", "code", "本人對目前 exporter、本地 HTTP 與331份 Notebook bytes 所跑的唯讀查核。"),
        ("a_export", HERE / "exporter-context-current-v1.json", "execution", "目前 exporter 必要原文、preview 頁尾與 metadata、331個真 HTTP下載 SHA／bytes及正式版本拒絕／單檔接受結果。"),
        ("a_current_assembly", Path(__file__).resolve(), "code", "本次 current 組裝程式；真95 claims 再核後獨立保存，保留原件並更新真正已改變的當前依賴證據。"),
    ]:
        artifact = {"id": identifier, "kind": kind, **ref(path), "description": description}
        if kind == "execution":
            artifact.update(command=EXPORT_COMMAND if identifier == "a_export" else COMMAND,
                            environment=exporter["environment"] if identifier == "a_export" else cpu["environment"],
                            result="331份 source13／工作樹／網站／HTTP bytes 全同；preview明示未發布，release拒絕未對齊版本；沒有雲端執行或發布。" if identifier == "a_export" else "65條路由、58次AST及原樣程式重跑通過；原始報告與6份raw全部未變。")
        current["artifacts"].append(artifact)

    claims = {c["id"]: c for c in current["claims"]}
    for claim in current["claims"]:
        quote = claim["source_quote"]
        assert quote in body
        line = body[:body.index(quote)].count("\n") + 1
        claim["location"] = f"W.7 v11 凍結切片 L{line}；course/first-steps.md L{base + line - 1}"
        if claim["kind"] == "software":
            claim["verification"]["details"] = claim["verification"]["details"].replace("本人實跑 audit_w7.py", "本人本次實跑 audit_w7_current_v1.py（明確用 v11，重新讀當前目的小節與主要程式）")
    for claim in current["claims"]:
        if claim["id"].startswith("route") and claim["id"] != "route_policy":
            claim["verification"]["details"] += " 本次重讀精確具名解釋／節錄範圍，並再執行相應 AST 對照；非只變更 hash。"
    outputs = claims["dir7"]
    outputs["evidence"] = [
        {"source_id": "code27", "locator": "scripts/export_course.py main L486–514；build L317–341、L394–402、L463–476",
         "supports": "目前主要 exporter 仍將預設網站產物放在 outputs/site，並拒絕 outputs 外目的地；新增preview／release狀態有真正依賴重核。"},
        {"source_id": "code26", "locator": "scripts/evaluate.py main L174", "supports": "JSON 報告預設位置仍為 outputs/evaluation.json。"},
        {"source_id": "run_export_current", "locator": "build_info / bundle / example_footer_text / release_guard_actual_results",
         "supports": "實際 outputs/omni-review-site-technical-fix13 及 localhost:8774 回應保存網站與331份 Notebook；明示 preview upload 未發布，正式版本需實存commit及全檔相同bytes。"},
    ]
    # Guard against accidentally retaining a positional source ID from a
    # different assembly: bind the actual evaluate source by path.
    evaluate_id = next(s["id"] for s in current["sources"] if s.get("path") == "scripts/evaluate.py")
    outputs["evidence"][1]["source_id"] = evaluate_id
    outputs["artifact_ids"] += ["a_export", "a_export_code", "a_current_run"]
    outputs["verification"]["observed"] = "目前 exporter SHA 662355e2…；預設輸出 ROOT/outputs/site，實際fix13產物在outputs/子目錄；331份 Notebook 的凍結、工作樹、網站與HTTP下載位元組一致；preview upload 且 publication_ready=false。"
    outputs["verification"]["details"] = "本人重讀目前 exporter 的輸出位置、頁尾、build接線與metadata；真呼叫preview footer及正式git版本閘，HTTP解析目前1.1與first-steps，逐331份下載比較。main／不存在40hex／當前HEAD不齊331bytes均如實拒絕；一份真commit相同Notebook子集接受。未把這個單檔接受或下載成功冒稱331已可發布／Colab實測。"
    outputs["scope"] += " 網站目前為明示未發布的本地預覽；release需要真40hex commit及全331相同bytes。本次未登入Colab、上傳、重建或部署。"
    claims["module_remedy"]["evidence"].append({"source_id": "run_export_current", "locator": "example_notebook_metadata / example_setup_source", "supports": "HTTP下載的1.1 Notebook metadata 真有 Tiny Perceptron / tiny-perceptron，setup 沿cwd往上尋找pyproject.toml及共用程式。未開啟GUI kernel。"})
    claims["module_remedy"]["artifact_ids"].append("a_export")
    claims["notebook_cwd"]["artifact_ids"].append("a_export")
    claims["notebook_cwd"]["evidence"].append({"source_id": "run_export_current", "locator": "example_setup_source", "supports": "目前真下載的Notebook setup從Path.cwd往上找根目錄，與W.7所述cwd可能不同的必要上下文相符。"})
    claims["index_example"]["verification"]["details"] = "本次再手算 n=2，合法非負整數索引滿足0≤i<2，故只有0與1；2是第3項並越界。另有本次真CPU IndexError輸出，手算仍明確標為hand_calculation。"

    current["checks"]["factual_accuracy"]["details"] = "本次完整重讀未變的W.7，逐65條必要實作／具名解釋路由重讀目前目的小節與主要程式，再跑58次原樣AST、路徑／環境／錯誤／dry-run。變更的exporter本人核目前功能／輸出／頁尾／metadata及真331份本地下載；95 claims均仍成立，沒有新issue。"
    current["checks"]["source_verification"]["details"] = "目前30份repository_code逐筆對當前SHA與必要定義；只有exporter變更，已重讀並真執行新功能相關邊界，而非引用舊snapshot豁免或只換hash。官方CPython/uv證據沿本人未變原始核對及其指紋；current執行用v11切片、source13真331bundle。"
    current["checks"]["numeric_verification"]["details"] = "本次再手算2項list的合法索引0、1及2越界，並以新CPU輸出補核；軟體數量與布林結果依目前實際執行保存。沒有把331份bytes一致解讀為331份程式或Colab皆已實測。"
    current["checks"]["figure_consistency"]["details"] = "v11的W.7仍無任何圖，figure_sha256恰為{}。65個導覽目的節的圖不屬於本callback的圖審範圍；只核其實作路由文字／程式，不宣稱新看過那些圖。"
    current["checks"]["limitations"]["details"] = "Linux CPU／本地HTTP的有界重核；Windows/macOS路徑仍依官方契約而未實跑。未重審連結全章理論／測得品質、未重新reader整頁first-steps；其fullreader仍wave02 scope。preview明示未發布、雲端未實測；無GPU／正式訓練／付費／commit／build／deployment。正式331同bytes commit尚未成立，正確被gate拒絕。"
    current["review_scope"].update({"source_freeze": numbered_path.relative_to(ROOT).as_posix(),
                                    "snapshot": page["snapshot"], "canonical_source_freeze": ref(ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"),
                                    "reader_task_exclusions_only": ["/root/rv1019_reader_entry"],
                                    "reader_completion_claim": False,
                                    "first_steps_full_reader_scope": "wave02 retained; this technical callback neither reruns nor expands that reader scope",
                                    "current_review_method": "本人讀當前W7全文；逐65路由的真正必要文字／節錄與主要定義；真58AST與pathlib/exception/dryrun重跑；exporter/頁尾/metadata及331真本地下載bytes重核。",
                                    "independence": "同一原reviewer本人第一次bounded current；只讀本人original/raw與教材／程式／本地實物，未讀作者PLAN／OUTLINE或他人review／reader／sidecar答案。"})
    current["bounded_current_recheck"] = {
        "original_report": ref(ORIGINAL), "current_stage": "first actual bounded current after original-as-current",
        "source_dependency_changes": source_changes,
        "reviewed_current_source": "scripts/export_course.py current 662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf; not an original_source_snapshot exemption",
        "route_result": routes["summary"], "changed_route_target_count": 0,
        "9.12": {"artifact_id": "a_export", "json_pointer": "/section_9_12_routing_assessment",
                  "conclusion": exporter["section_9_12_routing_assessment"]["conclusion"]},
        "actual_methods": [COMMAND, EXPORT_COMMAND],
        "publication_scope": "本地fix13 preview；upload、publication_ready=false；沒有Colab雲端或部署測試。331下載同bytes只是格式／版本一致的證據。",
        "reader_scope": "first-steps fullreader remains wave02; --reader-task used only for role exclusion",
        "original_issue_count": 0, "original_issue_resolutions_count": 0,
    }
    assert len(current["claims"]) == 95
    assert sha(ORIGINAL) == ORIGINAL_SHA
    with CURRENT.open("x", encoding="utf-8") as stream:
        json.dump(current, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": ref(CURRENT), "created_at": now, "verdict": current["verdict"],
                      "claims": len(current["claims"]), "issues": current["issues"],
                      "original_issue_resolutions": current["original_issue_resolutions"],
                      "original_unchanged": sha(ORIGINAL) == ORIGINAL_SHA,
                      "dependency_changes": source_changes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
