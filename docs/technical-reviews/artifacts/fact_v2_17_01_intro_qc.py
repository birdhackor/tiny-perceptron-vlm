"""Persist the reviewer's actually read chapter introduction and own judgement."""

import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXPECTED_INTRO = """# 第 17 章：量化，用更少位元表示模型

模型權重是許多數字；減少每個數字的儲存位元，可以讓同一套模型規則占較少空間，也會失去部分細節。本章先從[權重bytes](17.md#17.1)與刻度還原開始，逐步比較位元數、分組與真正的四位元打包。接著檢查中間特徵範圍、極端值與訓練時模擬量化，最後回到答案和行為。參考運算先還原浮點數，方便追蹤誤差；儲存收益與硬體速度分別量測。

"""
EXPECTED_BODY_SHA = "bbbd6a41fae53426959e02f29db048d7cdd3e9bb0735b220abb803a5d7c94784"
SUMMARY = (
    "本人完整親讀17.md第一個##之前的章標題與導讀段落。導讀將低位元權重表示、較少儲存與細節損失作為起點，"
    "再以權重bytes、刻度、位元数／分組、真4-bit打包、中間特徵範圍、極端值、訓練時模擬量化及答案行為安排章節路線；"
    "本次已逐一核對後續小節標題的對應，但沒有把標題檢查冒充各後續小節的完整技術審閱。"
    "17.1先用形狀及dtype算數值bytes，再拆正式模型的低位元係數、保留FP32數字與scale，正好接上導讀起點及先前已親讀的W.5矩陣前置；"
    "W.6梯度與16.3快取前置支持17.1末段把參數檔和執行記憶體分開的限制。"
    "導讀的細節損失描述降低表示精度與可能的量化誤差；17.1明示全零可精確表示，故不是每個數值必然失真的主張。"
    "導讀還明說參考運算先還原浮點數、儲存與硬體速度分別量測；已核的QuantizedLinear確實反量化再做普通Linear，"
    "與17.1不能由權重bytes推出檔案位元比或執行峰值同比縮小的限制一致。"
    "正文hash與本人上一輪審閱相同，此次沿用已真正讀過與核實的正文和必要前置證據，未重新執行既有模型或GPU實驗。"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "course/chapters/17.md"
    raw = source.read_bytes().decode("utf-8")
    first = re.search(r"^## .+$", raw, re.M)
    intro = raw[: first.start()]
    # Bind the update to the exact text displayed and read before this script was authored.
    assert intro == EXPECTED_INTRO
    assert first[0] == "## 17.1 權重到底佔多少空間？"
    body = dict(sections(source))["17.1"]
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert body_sha == EXPECTED_BODY_SHA
    report_path = ROOT / "docs/technical-reviews/17.1.json"
    old_bytes = report_path.read_bytes()
    report = json.loads(old_bytes)
    assert report["reviewer_task"] == "/root/integration_technical_coordinator/fact_v2_17_01"
    assert report["source_sha256"] == body_sha
    assert "intro_sha256" not in report
    history = OUT / "fact_v2_17_01_pre_intro_report.json"
    if history.exists():
        assert history.read_bytes() == old_bytes
    else:
        history.write_bytes(old_bytes)
    snapshot = OUT / "fact_v2_17_01_intro_snapshot.txt"
    snapshot.write_bytes(intro.encode("utf-8"))
    intro_sha = sha(snapshot)
    receipt = OUT / "fact_v2_17_01_intro_read_receipt.json"
    command = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_17_01_intro_qc.py"
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}
    record = {
        "command": command,
        "result": "Reviewer personally read the full pre-first-## introduction, found it consistent, and preserved the prior report. No model experiment rerun.",
        "environment": environment,
        "reviewer_task": report["reviewer_task"],
        "read_at_utc": datetime.now(UTC).isoformat(),
        "source": "course/chapters/17.md before first ##",
        "first_numbered_section": "17.1",
        "intro_snapshot": snapshot.relative_to(ROOT).as_posix(),
        "intro_sha256": intro_sha,
        "intro_text": intro,
        "intro_summary": SUMMARY,
        "reviewer_decision": "pass",
        "source_sha256": body_sha,
        "body_matches_previous_review": True,
        "prior_report_snapshot": history.relative_to(ROOT).as_posix(),
        "prior_report_sha256": sha(history),
        "previously_personally_read_prerequisites": ["course/first-steps.md#W.5", "course/first-steps.md#W.6", "course/chapters/16.md#16.3"],
        "chapter_headings_read_for_navigation_only": re.findall(r"^## .+$", raw, re.M),
        "limitations": ["Later-section headings are navigation evidence, not full technical reviews.", "Existing model evidence is reused only because the audited body SHA is unchanged.", "No new GPU, paid job, training, or curriculum/environment edits."],
    }
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["intro_sha256"] = intro_sha
    report["intro_summary"] = SUMMARY
    report["intro_status"] = "pass"
    for identifier, kind, path, description in (
        ("a_intro_snapshot", "source_snapshot", snapshot, "本人實際讀取的17.md第一個##之前完整原文；含章標題、導讀及原換行。"),
        ("a_intro_read", "execution", receipt, "本人讀取時間、原文snapshot/SHA、獨立摘要、與既有17.1/前置接續及限制的判斷。"),
        ("a_intro_qc_code", "code", Path(__file__), "限定本人已讀原文與原先正文SHA的intro QC腳本；保留舊報告後補自身判斷。"),
        ("a_pre_intro_report", "source_snapshot", history, "增加intro欄位前本人完整報告的逐byte歷史副本；保留原結論、證據與SHA。"),
    ):
        artifact = {"id": identifier, "kind": kind, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "description": description}
        if kind == "execution":
            artifact.update(command=command, result=record["result"], environment=environment)
        report["artifacts"].append(artifact)
    report["claims"].extend([
        {
            "id": "c17_intro_representation",
            "kind": "concept",
            "statement": "章導讀以較少位元保存權重能降低數值儲存、降低表示精度並可能引入量化誤差作起點。",
            "location": "17.md首個##前導讀第1句",
            "status": "verified",
            "evidence": [{"source_id": "s_paper", "locator": "§1 weight-only storage discussion; §2.1 Eq(1) r=S(q−Z)", "supports": "低位元離散表示及原始數值的scale映射；不保證每個數值都產生誤差或直接加速。"}],
            "artifact_ids": ["a_intro_snapshot", "a_intro_read", "a_jacob_paper", "a_paper_text"],
            "scope": "導讀的一般表示精度／可能誤差敘述，17.1全零可準確表示的限制仍成立；不聲稱固定品質或速度收益。",
        },
        {
            "id": "c18_intro_reference_forward",
            "kind": "software",
            "statement": "章導讀說本專案參考運算先還原浮點數，並分開量測儲存收益與硬體速度。",
            "location": "17.md首個##前導讀最後1句",
            "status": "verified",
            "evidence": [
                {"source_id": "s_quant", "locator": "QuantizedLinear.forward and storage_bytes", "supports": "q乘scale再轉x.dtype並F.linear，storage_bytes另按buffer計；正式FP32來源路徑還原FP32。"},
                {"source_id": "s_original_compression", "locator": "run_quantization L449–497; _storage L149–180; _timing L288–334", "supports": "storage及timing是不同report欄位與測量函式，限制明示不能據此推論低位元kernel/RAM收益。"},
            ],
            "artifact_ids": ["a_intro_snapshot", "a_intro_read", "a_cpu", "a_original_code", "a_report"],
            "verification": {
                "method": "executed",
                "expected": "FP32反量化參考運算；storage與timing分欄。",
                "observed": "先前本人真CPU載入/逐題執行QuantizedLinear與storage比對通過；正式report以storage/timing分欄且forward記FP32 dequantized reference。",
                "details": "沿用本人已核的a_cpu與正式run原碼/原始report，正文SHA未變；本次只親讀intro和章標題導航，不重跑模型或GPU。",
            },
            "scope": "本專案FP32來源的參考推論契約，非低位元硬體kernel；精度/速度/峰值無外推。",
        },
    ])
    for check in ("factual_accuracy", "source_verification", "limitations"):
        report["checks"][check]["claim_ids"].extend(["c17_intro_representation", "c18_intro_reference_forward"])
        report["checks"][check]["details"] += " 此次本人完整親讀並核首個##前導讀；新增intro原文snapshot/SHA、獨立摘要與兩個intro主張。"
    report["review_notes"]["intro_qc"] = "2026-10-04本人親讀整段導讀、核導航標題與先前已讀正文/前置接续。正文SHA未變，不冒稱本次重讀全部正文或重跑既有模型；舊報告逐byte保留。"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"intro_sha256": intro_sha, "source_sha256": body_sha, "intro_status": report["intro_status"], "report_sha256": sha(report_path), "prior_report_sha256": sha(history)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
