"""Personally recheck the new score-gap bridge without retraining or losing history."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from tiny_perceptron.posttraining import preference_loss

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_10"
REVIEWER = "/root/integration_technical_coordinator/fact_v2_13_10"
EXPECTED_SOURCE = "1e2fa8e96195db27a878a7401e12dce9f28ea2b5bd38c53eec2002bc9e5fe0aa"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_code(code):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "current13.10", "exec"), {})
    return output.getvalue()


def main():
    torch.set_num_threads(2)
    body = dict(sections(ROOT / "course/chapters/13.md"))["13.10"]
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    assert body_sha == EXPECTED_SOURCE
    previous = (BASE / f"{PREFIX}_lesson.md").read_text(encoding="utf-8")
    previous_report = json.loads((BASE / f"{PREFIX}_before_bridge_review.json").read_text())
    assert hashlib.sha256(previous.encode()).hexdigest() == previous_report["source_sha256"]
    code = re.findall(r"```python\n(.*?)\n```", body, re.S)[0]
    old_code = re.findall(r"```python\n(.*?)\n```", previous, re.S)[0]
    assert code == old_code
    paragraphs = body.split("\n\n")
    bridge = next(part for part in paragraphs if part.startswith("怎麼從分數差算到機率？"))
    assert body.replace(bridge + "\n\n", "", 1) == previous
    snapshot = BASE / f"{PREFIX}_bridge_lesson.md"
    snapshot.write_text(body, encoding="utf-8")
    original = run_code(code)
    changed = code.replace("rejected = torch.tensor([0.0])", "rejected = torch.tensor([1.0])")
    modified = run_code(changed)
    shifted = run_code(changed.replace("chosen = torch.tensor([preferred_score])", "chosen = torch.tensor([preferred_score + 10.0])")
                       .replace("rejected = torch.tensor([1.0])", "rejected = torch.tensor([11.0])"))
    assert original == "差距 0.0 勝出機率 0.5 代價 0.6931\n差距 2.0 勝出機率 0.8808 代價 0.1269\n"
    assert modified == "差距 -1.0 勝出機率 0.2689 代價 1.3133\n差距 1.0 勝出機率 0.7311 代價 0.3133\n"
    assert modified == shifted
    numeric_rows = []
    for chosen, rejected in [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0), (-2.0, 0.0)]:
        gap = chosen - rejected
        exponential = math.exp(-gap)
        probability = 1 / (1 + exponential)
        paper_probability = math.exp(chosen) / (math.exp(chosen) + math.exp(rejected))
        torch_probability = torch.sigmoid(torch.tensor(gap, dtype=torch.float64)).item()
        expected_loss = -math.log(probability)
        observed_loss = preference_loss(torch.tensor([chosen], dtype=torch.float64), torch.tensor([rejected], dtype=torch.float64)).item()
        assert math.isclose(exponential, math.e ** (-gap), rel_tol=1e-15)
        assert abs(probability - paper_probability) <= 1e-15
        assert abs(probability - torch_probability) <= 1e-15
        assert abs(expected_loss - observed_loss) <= 1e-15
        assert 0 < probability < 1
        if gap < 0:
            assert probability < 0.5
        numeric_rows.append({"chosen": chosen, "rejected": rejected, "gap": gap, "exp_minus_gap": exponential,
                             "math_probability": probability, "paper_eq1_probability": paper_probability,
                             "torch_float64_probability": torch_probability, "negative_log_probability": expected_loss,
                             "executed_preference_loss": observed_loss})
    exp_one = math.exp(1)
    exp_zero = math.exp(0)
    exp_minus_two = math.exp(-2)
    approximate_denominator_probability = 1 / (1 + 0.1353)
    assert round(math.e, 3) == round(exp_one, 3) == 2.718
    assert exp_zero == 1.0
    assert round(exp_minus_two, 4) == 0.1353
    assert round(approximate_denominator_probability, 4) == 0.8808
    prerequisite_hashes = {}
    for path, ids in [("course/chapters/13.md", {"13.1", "13.2", "13.15"}), ("course/chapters/01.md", {"1.8"})]:
        for lesson, text in sections(ROOT / path):
            if lesson in ids:
                prerequisite_hashes[lesson] = hashlib.sha256(text.encode()).hexdigest()
    derivation = """# Score-gap bridge recheck\n\nRead the actual DPO arXiv:2305.18290v3 PDF, section3 p3 equations(1)-(2) again. Equation(1) models conditional comparison p(y1 preferred to y2 | x), not p(answer factually correct). For scores c and r under the same x, divide numerator and denominator of exp(c)/(exp(c)+exp(r)) by exp(c): p=1/(1+exp(r-c)). With d=c-r this is 1/(1+exp(-d)), the logistic sigmoid. Equation(2) uses the chosen winner's -ln p.\n\nexp(x)=e^x; e=exp(1)=2.718281828459045, rounded to three decimals2.718. d=0 gives exp(0)=1 and p=1/(1+1)=1/2 exactly. d=2 gives exp(-2)=0.1353352832366127 and p=0.8807970779778823; their four-decimal displays are0.1353 and0.8808. Substituting the explicitly rounded intermediate0.1353 instead gives0.8808244516867788, still0.8808 to four decimals; the displayed approximation is not an exact equality.\n\nFor every finite real d<0, -d>0, exp(-d)>1, so 1+exp(-d)>2 and 0<p<1/2. For finite real d generally exp(-d)>0, hence0<p<1. Float32/64 may round to endpoints for extreme inputs; the mathematical mapping and this small example are the scope. No absolute correctness target or truth calibration follows from the comparison model. A labeled chosen winner can receive model probability below0.5 if its estimated reward is currently lower.\n\nCurrent code block matches the previous personally executed code exactly. The short CPU recheck executes that whole code plus rejected=1 and common+10, and checks math.exp/log against both equation(1) and torch.float64 sigmoid/preference_loss. Tolerance for finite toy analytic values1e-15; literal four-decimal stdout comparison; no training. Existing165-context/825-pair and portable checkpoint evidence is retained because its source code, result and tensor payloads remain unchanged.\n"""
    derivation_path = BASE / f"{PREFIX}_bridge_derivation.md"
    derivation_path.write_text(derivation, encoding="utf-8")
    receipt = {"schema_version": 1, "reviewer_task": REVIEWER, "reviewer_context": "fresh",
               "accessed_at": datetime.now(UTC).isoformat(),
               "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_10_bridge_recheck.py",
               "result": "Exit0; personally reread new paragraph and required prerequisites, original DPO Eq1-2; all math/torch and exact current-code/exercise checks passed.",
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                               "dtype": "float32 lesson; float64 bridge checks", "cpu_threads": "2", "platform": platform.platform()},
               "source_sha256": body_sha, "snapshot_path": str(snapshot.relative_to(ROOT)), "snapshot_sha256": digest(snapshot),
               "prior_report_path": str((BASE / f"{PREFIX}_before_bridge_review.json").relative_to(ROOT)),
               "prior_report_sha256": digest(BASE / f"{PREFIX}_before_bridge_review.json"),
               "prior_source_sha256": previous_report["source_sha256"], "prerequisite_sha256": prerequisite_hashes,
               "read_summary": "完整亲讀当前13.10、13.1、13.2、13.15及1.8。新增段以同题score difference定义d，写exp/e、d0/2的逐步换算、负差阈值和偏好非答对概率；与原DPOv3 §3 BT假设/Eq1-2一致。13.15仍是structured four-card机制、明确未读中文/算术；现报告URL已固定commit，含义与保留证据一致。",
               "original_source": {"url": "https://arxiv.org/pdf/2305.18290v3", "version": "arXiv2305.18290v3,29Jul2024",
                                   "checked_original": True, "locator": "section3 p3 equations1-2 and BT model assumption / same prompt x",
                                   "pdf_path": "outputs/posttrain-design/ppo-reference/dpo-original.pdf",
                                   "pdf_sha256": digest(ROOT / "outputs/posttrain-design/ppo-reference/dpo-original.pdf"),
                                   "fresh_page3_path": str((BASE / f"{PREFIX}_bridge_dpo_page3.txt").relative_to(ROOT)),
                                   "fresh_page3_sha256": digest(BASE / f"{PREFIX}_bridge_dpo_page3.txt"),
                                   "inspection_note": "本人重新亲讀原论文原PDF提取的§3，不使用先前review结论替代原式。式1的事件是conditional preference，式2是其负log似然。"},
               "code_unchanged_sha256": hashlib.sha256(code.encode()).hexdigest(), "only_change_is_bridge_paragraph": True,
               "numeric_rows": numeric_rows, "math_e": math.e, "exp_one": exp_one, "exp_zero": exp_zero,
               "exp_minus_two": exp_minus_two, "probability_using_rounded_0_1353": approximate_denominator_probability,
               "outputs": {"original": original, "rejected_1": modified, "both_shift_10": shifted},
               "derivation_path": str(derivation_path.relative_to(ROOT)), "derivation_sha256": digest(derivation_path),
               "recheck_code_sha256": digest(Path(__file__)), "training_executed": False, "python_executable": sys.executable}
    target = BASE / f"{PREFIX}_bridge_receipt.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_sha256": body_sha, "e": math.e, "exp_minus_two": exp_minus_two,
                      "probability_d2": numeric_rows[1]["math_probability"], "result": receipt["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
