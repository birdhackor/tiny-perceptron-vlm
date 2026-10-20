"""核對集中度算例、候選排列及公開判斷介面，保存原始來源的短摘錄。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.lesson_products.data import CANDIDATES, judgment
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import Config, Tokenizer, validate_candidates


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    argv = ["curl", "--fail", "--silent", "--show-error", "--location", "--include", "--max-time", "45", url]
    result = subprocess.run(argv, capture_output=True, check=True)
    raw = result.stdout
    headers = {}
    status = ""
    while raw.startswith(b"HTTP/"):
        head, raw = raw.split(b"\r\n\r\n", 1)
        lines = head.decode().splitlines()
        status = lines[0]
        headers = {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:]) if k}
    return raw.decode(), {
        "command_argv": argv,
        "exit_code": result.returncode,
        "url": url,
        "accessed_utc": datetime.now(UTC).isoformat(),
        "http_status": status,
        "http_date": headers.get("date"),
        "last_modified": headers.get("last-modified"),
        "etag": headers.get("etag"),
        "body_sha256": digest(raw),
        "body_bytes": len(raw),
        "full_body_saved": False,
    }


def concentration(probabilities):
    n = len(probabilities)
    return (max(probabilities) - 1 / n) / (1 - 1 / n)


def ordered_score(probabilities):
    n = len(probabilities)
    peak = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform)


def main():
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-15/verify_claims.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "accessed_on": datetime.now(UTC).date().isoformat(),
    }
    doc, receipt = fetch("https://docs.typesafe.ai/confidence.md")
    choice_section = doc.split("### Choice\n", 1)[1].split("### Score\n", 1)[0]
    noul_section = doc.split("### Noul\n", 1)[1].split("### Choice\n", 1)[0]
    score_section = doc.split("### Score\n", 1)[1]
    choice_equation = re.search(r"\$\$(.*?)\$\$", choice_section, re.S)[1].strip()
    score_equation = re.search(r"\$\$(.*?)\$\$", score_section, re.S)[1].strip()
    assert r"\frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}" in choice_equation
    assert "TypeSafe returns no separate `confidence` for it." in noul_section
    assert "levels are ordered" in score_section
    receipt.update(
        inspected_sections=["Confidence is derived from the probabilities", "How confidence is calculated", "Noul", "Choice", "Score"],
        excerpts={
            "choice_equation": choice_equation,
            "noul": "TypeSafe returns no separate `confidence` for it.",
            "choice_scope": "Only the top probability counts",
            "score_equation": score_equation,
        },
        interpretation="Choice只用最高候選機率；Noul直接報yes機率；Score另用有序等級的距離。",
        temporal_scope="讀取當日的動態官方頁面；沒有取得指定歷史日期的不可變文件版本，HTTP Last-Modified不能證明該歷史日期的內容。",
    )
    result["typesafe_original"] = receipt

    landing, paper = fetch("https://proceedings.mlr.press/v70/guo17a.html")
    assert "poorly calibrated" in landing
    assert "true correctness likelihood" in landing
    paper.update(
        inspected_sections=["Abstract", "Bibliographic Information"],
        version="ICML 2017, Proceedings of Machine Learning Research 70:1321–1330；出版方原始摘要頁",
        excerpt="predicting probability estimates representative of the true correctness likelihood",
        interpretation="摘要區分模型報告的機率與真實答對機率，並報告現代神經網路可能校準不佳；沒有把摘要當成特定Jev品質的證據。",
    )
    result["calibration_paper_landing"] = paper

    commit_body, commit_receipt = fetch("https://api.github.com/repos/gpleiss/temperature_scaling/commits/master")
    commit = json.loads(commit_body)
    revision = commit["sha"]
    code, author = fetch(f"https://raw.githubusercontent.com/gpleiss/temperature_scaling/{revision}/temperature_scaling.py")
    assert "confidences, predictions = torch.max(softmaxes, 1)" in code
    assert "accuracies = predictions.eq(labels)" in code
    author.update(
        commit=revision,
        commit_date=commit["commit"]["committer"]["date"],
        commit_lookup=commit_receipt,
        inspected_symbol="_ECELoss.forward",
        excerpts=["confidences, predictions = torch.max(softmaxes, 1)", "accuracies = predictions.eq(labels)"],
        interpretation="論文作者的程式取最大softmax機率為confidence，並另外與labels比較取得正確標記。沒有執行這份外部舊版PyTorch程式。",
    )
    result["calibration_author_source"] = author

    freeze = ROOT / "docs/course-review-20261019/freeze-01/sources/9.15.md"
    source = freeze.read_bytes()
    block = re.search(rb"```python\n(.*?)```", source, re.S)[1]
    outputs = []
    for p in ([0.6, 0.2, 0.2], [0.2, 0.6, 0.2]):
        namespace = {}
        changed = block.replace(b"p = [0.6, 0.2, 0.2]", f"p = {p}".encode())
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(changed, "frozen-concentration-example", "exec"), namespace)
            print(namespace["p"].index(max(namespace["p"])))
        outputs.append({"p": p, "stdout": stream.getvalue(), "concentration": concentration(p)})
    assert outputs[0]["stdout"] == "最大機率 0.6 集中度 0.4\n0\n"
    assert outputs[1]["stdout"] == "最大機率 0.6 集中度 0.4\n1\n"
    exact = (Fraction(3, 5) - Fraction(1, 3)) / (1 - Fraction(1, 3))
    assert exact == Fraction(2, 5)
    binary = concentration([0.6, 0.4])
    assert abs(binary - 0.2) < 1e-12
    wrong_high = concentration([0.99, 0.005, 0.005])
    assert abs(wrong_high - 0.985) < 1e-12
    for n in range(2, 10):
        assert abs(concentration([1 / n] * n)) < 1e-12
        assert concentration([1] + [0] * (n - 1)) == 1
    try:
        concentration([1.0])
    except ZeroDivisionError:
        singleton = "ZeroDivisionError"
    else:
        raise AssertionError("單一候選的公式應無定義")
    result["numeric_checks"] = {
        "source_sha256": digest(source),
        "original_and_permuted": outputs,
        "exact_fraction": str(exact),
        "three_candidate_subtraction": str(Fraction(3, 5) - Fraction(1, 3)),
        "three_candidate_denominator": str(1 - Fraction(1, 3)),
        "binary_same_max": binary,
        "wrong_high_probability_hypothesis": {"p": [0.99, 0.005, 0.005], "gold_index": 1, "chosen_index": 0, "concentration": wrong_high},
        "uniform_and_point_mass_checked_n": list(range(2, 10)),
        "single_candidate_exception": singleton,
        "tolerance": "浮點比較絕對差小於1e-12；Fraction有理數比較精確相等。",
        "boundedness_derivation": "合法n維機率的總和為1且各項非負，故1/n≤p_max≤1；n>1時分母為正，線性縮放後0≤c≤1。",
        "cross_question_limit": "固定p_max時，c=(n*p_max−1)/(n−1)會隨n改變；式中沒有正確標籤，不能單憑c大小比較不同問題的真實可靠度。",
    }

    class FixedScores(torch.nn.Module):
        def __init__(self, probabilities):
            super().__init__()
            self.register_buffer("scores", torch.tensor(probabilities).log())

        def forward(self, tokens):
            assert tokens.device.type == "cpu"
            assert tokens.shape[0] == self.scores.numel()
            return self.scores

    product = object.__new__(Product)
    product.product = "c"
    product.device = torch.device("cpu")
    product.tokenizer = Tokenizer()
    product.config = Config(len(product.tokenizer), width=16, heads=2, ff=32, c_layers=1)
    product.calibration = {"temperatures": {q: 1.0 for q in CANDIDATES}}
    predictions = {}
    inputs = {"choice": [0.6] + [0.08] * 5, "info": [0.005, 0.995], "rubric": [0.6, 0.2, 0.2]}
    for question, probabilities in inputs.items():
        product.model = FixedScores(probabilities)
        answer = product.judge("卡片進度;申請編號=?", question)
        assert set(answer["probabilities"]) == set(CANDIDATES[question])
        assert "concentration" in answer and "confidence" not in answer
        observed = list(answer["probabilities"].values())
        assert abs(sum(observed) - 1) < 2e-7
        assert abs(answer["concentration"] - concentration(observed)) < 2e-7
        predictions[question] = answer
    assert judgment({"task": "card", "application": None}, (), "info", ["yes", "no"]) == "no"
    assert predictions["info"]["selected"] == "no"
    assert predictions["info"]["concentration"] > 0.98
    try:
        validate_candidates("info", ["yes"])
    except ValueError as error:
        singleton_rejection = str(error)
    else:
        raise AssertionError("公開介面應拒絕單一候選")
    rubric_p = list(predictions["rubric"]["probabilities"].values())
    official_score = ordered_score(rubric_p)
    assert abs(official_score - predictions["rubric"]["concentration"]) > 0.29
    result["product_contract_checks"] = {
        "predictions": predictions,
        "singleton_rejection": singleton_rejection,
        "official_ordered_score_on_rubric_probabilities": official_score,
        "missing_application_gold": "no",
        "scope": "以固定logits替換scorer，實際呼叫Product.judge；驗證三種問題的輸出契約及公式，不載入正式權重、不訓練，也不聲稱測得品質或校準效果。",
    }
    paths = [
        "tiny_perceptron/lesson_products/inference.py",
        "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/data.py",
        "course/chapters/09.md",
        "docs/review-tools/section_facts.py",
        "docs/course-review-20261019/freeze-04.json",
    ]
    result["repository_hashes"] = {p: digest((ROOT / p).read_bytes()) for p in paths}
    result["result"] = "所有數值與本地介面斷言通過；官方定義核對通過；指定歷史日期的官方文件版本仍無證據。"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
