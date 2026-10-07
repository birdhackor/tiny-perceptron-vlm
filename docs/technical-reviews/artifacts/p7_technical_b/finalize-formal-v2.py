"""Preserve first metadata draft; add the missing chapter-07 navigation applicability."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
first = Path("docs/technical-reviews/phase7-freeze03-technical-b-initial-complete.json")
final = Path("docs/technical-reviews/phase7-freeze03-technical-b-initial-complete-v2.json")
base = Path("docs/technical-reviews/artifacts/p7_technical_b")
def sha(path):
    return hashlib.sha256((root / path).read_bytes()).hexdigest()
report = json.loads((root / first).read_text(encoding="utf-8"))
assert sha(first) == "31e43c616fd25783d22f410b00804b14b508303f6215b665c912ab1f95acb961"
page = next(item for item in report["pages"] if item["page_id"] == "chapter-07")
assert not page.get("applicability") and not page["claims"]
page["applicability"] = {
    "substantive_claims": False,
    "reason": "本人原读且此次通过past-unit-only event50回看：仅一段第7章教学路线，用1+1对话预告角色、首答案位置、读前文/计答案、padding/EOS/截断和后续教法；没有此页算法、可执行程序、测量或独立能力结论。",
}
page["checks"] = {
    "factual_accuracy": {"status": "not_applicable", "details": "本入口把角色与答案预测位置列为本章待追踪问题，未在此给出机制推导或独立技术结论。", "claim_ids": []},
    "numeric_verification": {"status": "not_applicable", "details": "1+1→2是章路线的示例名称，下一/再下一token是后文问题预告；此段不报告训练步数、容量值或实测成绩。", "claim_ids": []},
    "figure_consistency": {"status": "not_applicable", "details": "章入口没有图、表、公式或示意箭头。", "claim_ids": []},
    "source_verification": {"status": "not_applicable", "details": "单段导航无论文本页结果或API契约；具体SFT、loss等来源在本人已读正文页分别核对。", "claim_ids": []},
    "limitations": {"status": "not_applicable", "details": "此入口只预告留出检查与教法比较，没有宣称预训练/后训练或多步目标必然改善能力。", "claim_ids": []},
}
report["provenance"]["retained_formal_draft"] = {"path": str(first), "sha256": sha(first), "status": "retained failed metadata draft, not the submitted final report"}
report["provenance"]["first_metadata_preflight"] = {"receipt": {"path": str(base / "group-preflight-failed-receipt.json"), "sha256": sha(base / "group-preflight-failed-receipt.json")}, "output": {"path": str(base / "group-preflight-output.txt"), "sha256": sha(base / "group-preflight-output.txt")}, "failure": "chapter-07 missing top-level applicability; explicit navigation applicability now added based on own original event50 unit, with no change to original trace/provisional/verdict"}
report["provenance"]["format_corrections"].append("chapter-07 original record omitted top-level navigation applicability; add my explicit reason and separate check scopes after actual own past-unit-only event50 recall. Preserve failed draft/preflight and all original page records.")
report["provenance"]["finalization_script"] = {"path": str(base / "finalize-formal-v2.py"), "sha256": sha(base / "finalize-formal-v2.py")}
assert len(report["pages"]) == 61 and report["verdict"] == "pass"
assert all(sha(ref["path"]) == ref["sha256"] for ref in report["provisional_records"])
with (root / final).open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"report": str(final), "sha256": sha(final), "verdict": report["verdict"], "pages": len(report["pages"])}, ensure_ascii=False))
