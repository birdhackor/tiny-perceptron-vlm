"""保存原始執行紀錄，並核對草稿接受、因果位置與抽樣校正。"""

import ast
import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/16.14.md"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T065406995927Z"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    path = DEST / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def preserve_execution():
    names = ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]
    records = {}
    for name in names:
        raw = (RUN / name).read_bytes()
        records[name] = {"sha256": digest(raw), "content": raw.decode()}
    write("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/16.md#16.14" --execute',
        "origin": RUN.relative_to(ROOT).as_posix(),
        "records": records,
    })
    print(records["stdout.txt"]["content"], end="")


def check_examples():
    raw = SOURCE.read_bytes()
    body = re.search(rb"(?m)^## 16\.14 .*?(?=^## |\Z)", (ROOT / "course/chapters/16.md").read_bytes(), re.S)[0]
    assert body == raw
    code = re.search(r"```python\n(.*?)```", raw.decode(), re.S)[1]
    tree = ast.parse(code)
    original_stdout = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(original_stdout):
        exec(compile(tree, str(SOURCE), "exec"), namespace)
    assert namespace["accepted"] == ["今天", "晴天"]

    def run_draft(draft):
        changed = ast.parse(code)
        assignment = next(node for node in changed.body if isinstance(node, ast.Assign) and node.targets[0].id == "draft")
        assignment.value = ast.List(elts=[ast.Constant(value=word) for word in draft], ctx=ast.Load())
        ast.fix_missing_locations(changed)
        current = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(changed, str(SOURCE), "exec"), current)
        return current["accepted"], stdout.getvalue()

    original, original_text = run_draft(["今天", "下雨", "嗎"])
    exercise, exercise_text = run_draft(["今天", "晴天", "嗎"])
    assert original == ["今天", "晴天"]
    assert exercise == ["今天", "晴天", "〈E〉"]
    target_sequence = ["今天", "晴天", "〈E〉"]
    vocabulary = ["今天", "晴天", "下雨", "嗎", "〈E〉"]
    tested = 0
    for draft in itertools.product(vocabulary, repeat=3):
        result, _ = run_draft(draft)
        assert result == target_sequence[:len(result)]
        tested += 1

    hashability = {}
    try:
        {[]: "值"}
    except TypeError as error:
        hashability["list_key"] = str(error)
    else:
        raise AssertionError("清單鍵應被拒絕")
    assert {tuple(["今天"]): "晴天"}[("今天",)] == "晴天"
    hashability["tuple_lookup"] = "晴天"

    rules = namespace["rules"]
    trace = []
    accepted = []
    for candidate in ["今天", "下雨", "嗎"]:
        context = accepted.copy()
        target = rules[tuple(accepted)]
        trace.append({"context": context, "candidate": candidate, "target": target, "matches": candidate == target})
        accepted.append(target)
        if candidate != target or candidate == "〈E〉":
            break

    p = [Fraction(6, 10), Fraction(3, 10), Fraction(1, 10)]
    q = [Fraction(2, 10), Fraction(2, 10), Fraction(6, 10)]
    acceptance = [min(Fraction(1), a / b) for a, b in zip(p, q)]
    accepted_mass = [b * a for a, b in zip(acceptance, q)]
    rejection = 1 - sum(accepted_mass)
    positive_difference = [max(Fraction(0), a - b) for a, b in zip(p, q)]
    residual = [a / sum(positive_difference) for a in positive_difference]
    final = [a + rejection * b for a, b in zip(accepted_mass, residual)]
    assert final == p

    prefix = ["天氣如何？"]
    draft = ["今天", "下雨", "嗎"]
    positions = [{"candidate": candidate, "visible_context": prefix + draft[:index]} for index, candidate in enumerate(draft)]
    # 嚴格下三角描述「預測候選」可見的 token；不把候選自身當成前文。
    mask = [[int(key < query) for key in range(4)] for query in range(1, 4)]
    assert mask == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]

    freeze_path = ROOT / "docs/course-review-20261019/freeze-04.json"
    freeze = json.loads(freeze_path.read_bytes())
    page = next(page for page in freeze["pages"] if page["page_id"] == "16.14")
    figure_path = "course/figures/p7-16.14-draft-verify.svg"
    figure_digest = digest((ROOT / figure_path).read_bytes())
    render_path = "outputs/course-review-20261019/figure-renders-04/" + figure_digest + ".png"
    receipt_path = ROOT / "docs/course-review-20261019/figure-renders-04.json"
    receipts = json.loads(receipt_path.read_bytes())
    assert figure_digest in json.dumps(receipts)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-14/verify.py examples",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；純 Python 算例"},
        "source_sha256": digest(raw),
        "current_source_equals_freeze": body == raw,
        "freeze_sha256": digest(freeze_path.read_bytes()),
        "page_freeze": page,
        "figure_sha256": {figure_path: figure_digest},
        "render_inspected": render_path,
        "render_sha256": digest((ROOT / render_path).read_bytes()),
        "render_receipt_sha256": digest(receipt_path.read_bytes()),
        "original_stdout": original_text,
        "exercise_stdout": exercise_text,
        "original_result": original,
        "exercise_result": exercise,
        "target_sequence": target_sequence,
        "original_lookup_trace": trace,
        "exhaustive_drafts": {"vocabulary": vocabulary, "draft_length": 3, "tested": tested, "all_outputs_are_target_prefixes": True},
        "hashability": hashability,
        "causal_alignment": {"positions": positions, "mask": mask, "column_tokens": prefix + draft},
        "sampling_check": {key: [str(value) for value in values] for key, values in {"p": p, "q": q, "acceptance": acceptance, "accepted_mass": accepted_mass, "residual": residual, "final": final}.items()},
        "rejection_mass": str(rejection),
        "sampling_check_exact": True,
        "limits": "人工規則和有理數推導；未呼叫神經模型、未計時、未測 GPU、未測 MTP 模型品質。枚舉只涵蓋指定字表與長度。",
    }
    write("verification.json", result)
    print(json.dumps({"原稿結果": original, "練習結果": exercise, "枚舉草稿數": tested, "抽樣校正精確相等": final == p}, ensure_ascii=False))


SOURCES = [
    ("leviathan_html", "https://arxiv.org/html/2211.17192v2", "arXiv:2211.17192v2；PMLR 發表版的對照全文", "If", "the increased concurrency", "已讀 Speculative Decoding、Standardized Sampling、Speculative Sampling、Algorithm 1、Walltime Improvement、Number of Arithmetic Operations；與 PMLR PDF 相關段落交叉核對。"),
    ("leviathan_pdf", "https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf", "PMLR 202:19274–19286, ICML 2023", None, None, "以 web PDF 原文抽取讀取 Overview 至 Choosing gamma 與 Algorithm 1；curl 取得同一發表版 PDF 的原檔雜湊。"),
    ("gloeckle_html", "https://arxiv.org/html/2404.19737v1", "arXiv:2404.19737v1, 2024-04-30", "During inference time", "greedy self-speculative decoding", "已讀 Method 的架構、Inference，及 Faster inference；額外分支可選用，未把其特定硬體速度外推為本課速度。"),
    ("gloeckle_pdf", "https://arxiv.org/pdf/2404.19737v1", "arXiv:2404.19737v1, 2024-04-30", None, None, "凍結頁面所連結的原始 PDF；全文內容讀取同版本 arXiv HTML。"),
    ("deepseek_html", "https://arxiv.org/html/2412.19437v2", "arXiv:2412.19437v2, 2025-02-18", "Our MTP strategy mainly aims", "directly discard the MTP modules", "已讀 Multi-Token Prediction 全段，特別核對 MTP in Inference；主模型可以獨立推論，MTP 模組也可用於推測解碼。"),
    ("deepseek_pdf", "https://arxiv.org/pdf/2412.19437v2", "arXiv:2412.19437v2, 2025-02-18", None, None, "凍結頁面所連結的原始 PDF；全文內容讀取同版本 arXiv HTML。"),
    ("transformer_html", "https://arxiv.org/html/1706.03762v7", "arXiv:1706.03762v7, 2023-08-02", "We also modify the self-attention", "prevent positions from attending to subsequent positions", "已讀 Encoder and Decoder Stacks 的 Decoder、Scaled Dot-Product Attention 與 Applications of Attention；遮罩與位移阻擋未來資訊，矩陣形式同時計算位置。"),
    ("distillation_html", "https://arxiv.org/html/1503.02531v1", "arXiv:1503.02531v1, 2015-03-09", "In the simplest form of distillation", "using a soft target distribution", "已讀 Distillation 與 Matching logits is a special case of distillation；教師提供目標並產生學生的訓練梯度，與生成時草稿驗證不同。"),
    ("python_docs", "https://docs.python.org/3.13/tutorial/datastructures.html", "Python 3.13.16 documentation，讀取日版本", "Tuples can be used as keys", "You can’t use lists as keys", "已讀 Dictionaries；字串 tuple 可當鍵，list 不可當鍵。未聲稱任意 tuple 都可雜湊。"),
]


def retrieve(item):
    identifier, url, version, start, excerpt, note = item
    command = ["curl", "-fLsS", "--max-time", "45", url]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    receipt = {"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(), "command_argv": command, "exit_code": 0, "bytes": len(raw), "original_sha256": digest(raw), "inspection_note": note}
    if start is not None:
        text = unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw.decode())))
        assert start in text
        assert excerpt in text
        receipt["excerpt"] = excerpt
        receipt["start_locator"] = start
    return receipt


def collect_sources():
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(retrieve, SOURCES))
    write("source-receipts.json", {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-14/verify.py sources", "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；curl HTTPS 讀取原始來源"}, "receipts": receipts, "storage_policy": "僅保存原始檔雜湊、定位、小段摘錄與核對摘要；未保存第三方全文。"})
    print(json.dumps({"取得原始來源數": len(receipts), "所有來源讀取成功": True}, ensure_ascii=False))


if __name__ == "__main__":
    {"preserve": preserve_execution, "examples": check_examples, "sources": collect_sources}[sys.argv[1]]()
