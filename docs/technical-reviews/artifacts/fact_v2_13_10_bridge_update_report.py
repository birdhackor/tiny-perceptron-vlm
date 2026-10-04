"""Update only this reviewer's report after the genuine bridge recheck receipt."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(identifier, kind, filename, description, receipt=None):
    path = BASE / (PREFIX + filename)
    item = {"id": identifier, "kind": kind, "path": str(path.relative_to(ROOT)),
            "sha256": digest(path), "description": description}
    if receipt is not None:
        item.update(command=receipt["command"], result=receipt["result"], environment=receipt["environment"])
    return item


def evidence(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}


def numeric(identifier, statement, expected, observed, tolerance, source):
    return {"id": identifier, "kind": "numeric", "statement": statement,
            "location": "当前13.10新增第3段：分數差換算到機率的桥樑", "status": "verified",
            "evidence": [evidence("s_bridge_derivation", source, "逐步代入自然指数、sigmoid及四舍五入。")],
            "artifact_ids": ["a_bridge_derivation", "a_bridge_recheck"],
            "verification": {"method": "executed", "expected": expected, "observed": observed,
                             "tolerance": tolerance, "details": "本人以math.exp/math.e及torch.float64 sigmoid/preference_loss执行短CPU验证；原代码float32 stdout亦完整执行且literal匹配，未重训。"},
            "scope": "本节给出的透明算例；近似中间值不冒充精确常数，比较概率不是答对概率。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-checker", action="store_true")
    args = parser.parse_args()
    prior = BASE / f"{PREFIX}_before_bridge_review.json"
    receipt = json.loads((BASE / f"{PREFIX}_bridge_receipt.json").read_text())
    report = copy.deepcopy(json.loads(prior.read_text()))
    body = dict(sections(ROOT / "course/chapters/13.md"))["13.10"]
    current_hash = hashlib.sha256(body.encode()).hexdigest()
    assert digest(prior) == receipt["prior_report_sha256"]
    assert report["source_sha256"] == receipt["prior_source_sha256"]
    assert current_hash == receipt["source_sha256"] == digest(BASE / f"{PREFIX}_bridge_lesson.md")
    assert digest(BASE / f"{PREFIX}_bridge_recheck.py") == receipt["recheck_code_sha256"]
    assert receipt["only_change_is_bridge_paragraph"] is True and receipt["training_executed"] is False
    for old in report["artifacts"]:
        assert digest(ROOT / old["path"]) == old["sha256"], old["id"]
    for source in report["sources"]:
        if source["kind"] == "repository_code":
            assert digest(ROOT / source["path"]) == source["sha256"]
    for figure, expected in report["figure_sha256"].items():
        assert digest(ROOT / figure) == expected
    report["source_sha256"] = current_hash
    for old in report["artifacts"]:
        if old["id"] == "a_lesson":
            old.update(path=str((BASE / f"{PREFIX}_bridge_lesson.md").relative_to(ROOT)), sha256=current_hash,
                       description="本人完整读过并短CPU重执行的当前13.10原文，含新增exp/sigmoid换算段；旧原snapshot另保存。")
        if old["id"] == "a_builder_code":
            old["description"] += " 当前新增桥樑由a_bridge_update_code依据真实新receipt更新；此旧builder及旧证据保持原hash，不能用其旧guard直接刷当前SHA。"
    report["artifacts"].extend([
        artifact("a_before_bridge_lesson", "source_snapshot", "_lesson.md", "上一版亲讀/原执行的完整13.10 snapshot，sourceSHA ac9ec8…未改动，保留历史。"),
        artifact("a_before_bridge_review", "source_snapshot", "_before_bridge_review.json", "本次新增桥樑前本reviewer完整report原bytes，含旧判定/主张/hash与已resolved issues；未覆盖。"),
        artifact("a_bridge_dpo_p3", "source_snapshot", "_bridge_dpo_page3.txt", "本人重读的DPOv3原PDF p3 Eq1-2 fresh pdftotext摘录；偏好事件及BT假设位置完整。"),
        artifact("a_bridge_derivation", "derivation", "_bridge_derivation.md", "新段公式代数、exp/e定义、d0/2及舍入、负差阈值和概率语义的逐步核查推导。"),
        artifact("a_bridge_recheck_code", "code", "_bridge_recheck.py", "真实短CPU重验新公式/数值，完整当前代码/练习，保存当前snapshot、前置hash和亲读summary，不训练。"),
        artifact("a_bridge_recheck", "execution", "_bridge_receipt.json", "实际短CPU执行和本人全文/原DPO公式重新核对之receipt，保留新旧SHA、版本、summary及所有数值。", receipt),
        artifact("a_bridge_update_code", "code", "_bridge_update_report.py", "只按当前真实recheck与旧report完整hash证据更新本人报告，补实质claims及限制；拒绝直接只刷SHA。"),
        artifact("a_bridge_ruff", "execution", "_bridge_ruff.txt", "两个新Python实际Ruff imports及项目规则检查输出。",
                 {"command": ".venv/bin/ruff check --no-force-exclude docs/technical-reviews/artifacts/fact_v2_13_10_bridge_recheck.py docs/technical-reviews/artifacts/fact_v2_13_10_bridge_update_report.py", "result": "Exit0; All checks passed!", "environment": receipt["environment"]}),
    ])
    if args.include_checker:
        report["artifacts"].append(artifact("a_bridge_checker", "execution", "_bridge_checker_snapshot.txt", "这次新正文真实--lesson13.10 checker的稳定stdout副本；当前final live检查另保存。",
                                   {"command": ".venv/bin/python scripts/check_technical_reviews.py --lesson 13.10", "result": "Exit0;1/1节来源与证据版本一致。", "environment": receipt["environment"]}))
    derivation_text = (BASE / f"{PREFIX}_bridge_derivation.md").read_text()
    report["sources"].append({"id": "s_bridge_derivation", "kind": "derivation", "title": "New score-gap bridge independent derivation", "verified": True, "details": derivation_text})
    report["sources"].append({"id": "s_bridge_execution", "kind": "execution", "title": "Current bridge original math/torch short CPU recheck", "verified": True, "artifact_id": "a_bridge_recheck"})
    for source in report["sources"]:
        if source["id"] == "s_dpo":
            source["inspection_note"] += " 本次新增桥樑本人再读实际v3原PDF§3 p3 Eq1-2，fresh extract/hash及读后summary见a_bridge_recheck：除以exp(c)得1/(1+exp(-(c-r)))；事件仍是同x的偏好，未变成factual correctness。"
    claims = {item["id"]: item for item in report["claims"]}
    for identifier in ["c3", "c5", "c6", "c9", "c10", "c11"]:
        claims[identifier]["artifact_ids"].append("a_bridge_recheck")
    claims["c3"]["location"] = "第2段与新增第3段固定换算规则，以及程式win_probability"
    claims["c3"]["scope"] += " d是chosen-rejected；有限实数sigmoid在(0,1)，浮点极端值可舍入为端点，正文说明的是数学换算及此小算例。"
    claims["c5"]["location"] = "第2段、新增第3段d=0逐步换算及原程式第一行"
    claims["c6"]["location"] = "第2段、新增第3段d=2/exp(-2)换算及原程式第二行"
    claims["c9"]["location"] = "新增第3段末句与程式后第2段：偏好机率不是答对机率、score不是校准正确率"
    claims["c9"]["evidence"].append(evidence("s_dpo", "v3 §3 Eq1: p*(y1≻y2|x), not P(correct)", "新增末句的事件解释符合原始偏好分布定义，并非事实正确率校准。"))
    report["claims"].extend([
        {"id": "c19", "kind": "concept", "statement": "若d是同题chosen分数减rejected分数，Bradley–Terry固定换算为sigmoid(d)=1/(1+exp(-d))。",
         "location": "新增第3段前两句：d定义和固定换算公式", "status": "verified",
         "evidence": [evidence("s_dpo", "v3 §3 p3 Eq1 and Eq2", "exp(c)/(exp(c)+exp(r))除以exp(c)，得到logistic difference；Eq2用其负log。"), evidence("s_bridge_derivation", "first paragraph algebra", "明确d=c-r；分母exp(r-c)=exp(-d)。")],
         "artifact_ids": ["a_bridge_dpo_p3", "a_bridge_derivation", "a_bridge_recheck"],
         "scope": "这是BT偏好模型的假设/固定数学换算，不是从比较推出普遍真实概率或事实正确率；这里chosen是标签名，当前估计可偏好rejected。"},
        numeric("c20", "exp(x)以e为底，e=exp(1)约2.718。", "e=exp(1)=2.718281828459045;三位小数2.718",
                f"math.e={receipt['math_e']};math.exp(1)={receipt['exp_one']};各toy x的math.exp(x)与math.e**x相符", "三位小数舍入容忍0.0005；toy exp与幂值相对误差≤1e-15。", "exp(x)=e^x and e=exp(1)"),
        numeric("c21", "差距2的exp(-2)约0.1353；用该近似值代入1/(1+0.1353)，概率四位小数仍0.8808。", "exp(-2)=0.135335…→0.1353;exact p=0.880797…→0.8808;rounded-intermediate p→0.8808",
                f"exp(-2)={receipt['exp_minus_two']}; exact p={receipt['numeric_rows'][1]['math_probability']};1/(1+0.1353)={receipt['probability_using_rounded_0_1353']}", "四位小数literal舍入吻合；exact math/torch/both forms误差≤1e-15；近似分母不是精确e^-2。", "d=2 exact and rounded intermediate substitution"),
        {"id": "c22", "kind": "concept", "statement": "有限实数d为负时，sigmoid(d)<0.5；分数差可以为任何实数，而sigmoid数学值在0与1之间。",
         "location": "新增第3段：负差低于0.5与换到0和1之间", "status": "verified",
         "evidence": [evidence("s_dpo", "v3 §3 Eq1 logistic comparison", "原式分子和分母指数都正；相同分数概率1/2，chosen较低则更小。"), evidence("s_bridge_derivation", "finite-real inequality proof", "d<0→-d>0→exp(-d)>1→分母>2→0<p<1/2；任意finite real分母>1→0<p<1。")],
         "artifact_ids": ["a_bridge_derivation", "a_bridge_recheck"], "scope": "是有限实数数学式的范围；float32/64极端输入可舍入为0/1。新增段没有证明模型偏好可靠，也没有声称回答答对的机率。"},
    ])
    assert all(item["status"] == "verified" for item in report["claims"])
    assert all(item["status"] == "resolved" for item in report["issues"])
    report["verdict"] = "pass"
    history = report.setdefault("recheck_history", [])
    history.append({"reason": "新增score-gap到sigmoid数值的解释桥樑，原代码/图/算法/实验不变。",
                    "reviewer_task": report["reviewer_task"], "prior_source_sha256": receipt["prior_source_sha256"],
                    "current_source_sha256": current_hash, "receipt_artifact_id": "a_bridge_recheck",
                    "read_summary": receipt["read_summary"], "prerequisite_sha256": receipt["prerequisite_sha256"],
                    "new_claim_ids": ["c19", "c20", "c21", "c22"], "training_executed": False})
    for name in ["factual_accuracy", "source_verification", "limitations"]:
        report["checks"][name]["claim_ids"].extend(["c19", "c22"])
    report["checks"]["numeric_verification"]["claim_ids"].extend(["c20", "c21"])
    report["checks"]["factual_accuracy"]["details"] += " 本人完整重读新段/必要前置，并重新核DPOv3 Eq1-2；新增d/exp/e定义、数值桥樑与负差范围符合原式。"
    report["checks"]["numeric_verification"]["details"] += " 新增math/torch短CPU核exp(0)、e=exp(1)、exp(-2)、exact及rounded denominator概率；完整原例/rejected1/共同+10再次执行。"
    report["checks"]["source_verification"]["details"] += " 新桥樑重新亲读DPOv3原PDF§3 Eq1-2并保存fresh excerpt/新snapshot/summary和真实receipt。"
    report["checks"]["limitations"]["details"] += " 新段的偏好事件定义正确；有限实数数学sigmoid范围与float端点舍入分开记录，0.1353是明确近似。"
    target = ROOT / "docs/technical-reviews/13.10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "source_sha256": current_hash,
                      "claims": len(report["claims"]), "artifacts": len(report["artifacts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
