"""核對凍結導言、原文取得收據與實驗紀錄的共同評分目標。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from tiny_perceptron.data import ByteTokenizer


ROOT = Path.cwd()
ARTIFACT = Path(__file__).relative_to(ROOT).as_posix()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.parts = None
        self.identifier = None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.parts = []
            self.identifier = dict(attrs).get("id")

    def handle_endtag(self, tag):
        if tag == "p" and self.parts is not None:
            self.items.append((self.identifier, " ".join("".join(self.parts).split())))
            self.parts = None

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)


def fetch(url):
    command = ["curl", "-fLsS", "--max-time", "50", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, {"command": command, "exit_code": result.returncode,
                           "bytes": len(result.stdout), "sha256": digest(result.stdout)}


def inspect_external(spec):
    pdf, pdf_receipt = fetch(spec["url"])
    require(pdf.startswith(b"%PDF-"), "原始來源回應必須是 PDF")
    html, html_receipt = fetch(spec["transcription_url"])
    parser = Paragraphs()
    parser.feed(html.decode("utf-8"))
    excerpts = []
    for phrase in spec["phrases"]:
        matches = [identifier for identifier, text in parser.items if phrase in text]
        require(bool(matches), "原文轉錄找不到指定摘錄")
        excerpts.append({"text": phrase, "paragraph_ids": matches})
    return {**{key: value for key, value in spec.items() if key != "phrases"},
            "pdf_receipt": pdf_receipt, "transcription_receipt": html_receipt,
            "excerpts": excerpts,
            "reading_method": "人工讀取原始 PDF 的指定段落；curl 取得原檔與全文轉錄，僅保存短摘錄與雜湊。轉錄未提供明確修訂號，版本以原始 PDF 為準。"}


specs = [
    {"title": "Distilling the Knowledge in a Neural Network",
     "url": "https://arxiv.org/pdf/1503.02531v1",
     "version": "arXiv:1503.02531v1，2015-03-09",
     "transcription_url": "https://ar5iv.labs.arxiv.org/html/1503.02531v1",
     "locator": "Introduction；Distillation；Matching logits is a special case of distillation",
     "phrases": ["use the class probabilities produced by the cumbersome model as “soft targets”"]},
    {"title": "Sequence-Level Knowledge Distillation",
     "url": "https://aclanthology.org/D16-1139.pdf",
     "version": "EMNLP 2016 正式出版版，DOI 10.18653/v1/D16-1139",
     "transcription_url": "https://ar5iv.labs.arxiv.org/html/1606.07947",
     "locator": "Knowledge Distillation；Word-Level Knowledge Distillation；Sequence-Level Knowledge Distillation；Experimental Setup；Decoding Speed",
     "phrases": ["the student network is trained on a newly generated dataset"]},
    {"title": "On Calibration of Modern Neural Networks",
     "url": "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf",
     "version": "ICML 2017 正式出版版，PMLR 70:1321–1330",
     "transcription_url": "https://ar5iv.labs.arxiv.org/html/1706.04599",
     "locator": "Definitions，perfect calibration；Temperature scaling",
     "phrases": ["we define perfect calibration as", "temperature scaling does not affect the model’s accuracy."]},
]

with ThreadPoolExecutor(max_workers=3) as executor:
    external = list(executor.map(inspect_external, specs))

snapshot_path = "docs/course-review-20261019/freeze-01/sources/chapter-18.md"
freeze_path = "docs/course-review-20261019/freeze-04.json"
snapshot = (ROOT / snapshot_path).read_bytes()
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-18")
current = (ROOT / page["source"]).read_bytes()
boundary = re.search(rb"^## ", current, re.MULTILINE).start()
require(current[:boundary] == snapshot, "現行導言必須與凍結內容相同")
require(digest(snapshot) == page["source_sha256"], "凍結原文雜湊必須相符")
require(page["figures_sha256"] == {}, "凍結導言沒有圖")
require(b"```" not in snapshot, "凍結導言沒有程式區塊")

prior = (ROOT / "course/chapters/09.md").read_text()
prior_section = re.search(r"^## 9\.14 .*?(?=^## )", prior, re.MULTILINE | re.DOTALL)[0]
linked_statement = "若 q 來自教師模型的猜測，模型學到的首先是貼近教師，而不是貼近事實。"
require(linked_statement in prior_section, "連結前置必須包含所述提醒")

tokenizer = ByteTokenizer()
evaluation = []
result_files = ["docs/course-experiments/results/distillation.json",
                "docs/course-experiments/results/multimodal_distillation.json"]
for relative in result_files:
    document = json.loads((ROOT / relative).read_bytes())
    modal = "multimodal" in relative
    sample_key, target_key, exact_key = (("samples", "target", "exact_match") if modal
                                         else ("generated_samples", "expected", "exact"))
    for task, task_data in document["results"]["tasks"].items():
        for split in ("validation", "test"):
            teacher = task_data["teacher_" + split]
            references = [(row["family"], row["question"], row[target_key])
                          for row in teacher[sample_key]]
            entries = [("teacher", teacher)] + [(name, run[split])
                                                for name, run in task_data["runs"].items()]
            for name, entry in entries:
                samples = entry[sample_key]
                require([(row["family"], row["question"], row[target_key]) for row in samples] == references,
                        "教師與學生必須用同一組題目和標準答案評分")
                hits = []
                for row in samples:
                    generated = row["generated_ids"]
                    raw = generated[:generated.index(tokenizer.eos_id)] if tokenizer.eos_id in generated else generated
                    hit = raw == tokenizer.encode(row[target_key])
                    require(hit == row[exact_key], "逐題判定必須符合原始 token 與標準答案的比較")
                    hits.append(hit)
                require(len(samples) == entry["examples"], "評分分母必須等於題數")
                require(sum(hits) == entry["correct"], "正確題數必須符合逐題重新計算")
                require(math.isclose(sum(hits) / len(hits), entry["exact_match"], abs_tol=1e-12),
                        "正確率必須使用同一組題目為分母")
            evaluation.append({"file": relative, "task": task, "split": split,
                               "examples_per_model": len(references), "models": len(entries),
                               "shared_targets": True, "raw_token_scores_recomputed": True})

teacher = [0.9, 0.1]
student = teacher.copy()
gold = 1
entropy = -sum(p * math.log(p) for p in teacher)
cross_entropy = -sum(p * math.log(q) for p, q in zip(teacher, student, strict=True))
kl = sum(p * math.log(p / q) for p, q in zip(teacher, student, strict=True))
require(math.isclose(entropy, cross_entropy, abs_tol=1e-12) and kl == 0,
        "完全模仿教師時交叉熵等於教師熵且 KL 為零")
require(max(range(len(student)), key=student.__getitem__) != gold, "教師與學生仍可能答錯")

paths = [snapshot_path, freeze_path, "course/chapters/18.md", "course/chapters/09.md",
         "scripts/course_experiments/compression.py", "scripts/course_experiments/modalities.py",
         "tiny_perceptron/data.py", *result_files]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python " + ARTIFACT,
    "result": "凍結版本、來源取得、共同標準答案與逐題評分核對全部通過。",
    "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                    "device": "CPU；僅重新計算紀錄，不載入權重或訓練", "platform": platform.platform(),
                    "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
    "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
    "local_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
    "freeze": {"source_matches_current_introduction": True, "page": page,
               "contains_code_block": False, "svg_reference_count": 0},
    "linked_prerequisite": {"path": "course/chapters/09.md", "title": "有機率輸出，還要怎樣訓練與校準？",
                            "checked_statement": linked_statement},
    "external_sources": external,
    "evaluation_record_checks": evaluation,
    "counterexample": {"teacher": teacher, "student": student, "true_class": gold,
                       "teacher_student_kl": kl, "teacher_entropy": entropy,
                       "teacher_target_cross_entropy": cross_entropy, "teacher_agreement": True,
                       "correct": False, "confidence": max(student), "accuracy": 0.0,
                       "calibration_gap": max(student),
                       "derivation": "CE(p,q)=H(p)+KL(p||q)。取 q=p 可完全貼近教師；若真實類別與其 argmax 不同，學生仍錯。所有輸入均輸出同一分布時，信心與真實正確率也可不相符。"},
    "limitations": ["未重跑訓練、載入教師權重或重現論文成績。",
                    "紀錄重算支持評分方法與共同標準答案，不獨立證明保存的生成確實由標示模型產生。",
                    "本次只有章首導言；後續算例、圖與實測數字不屬於這份審閱。"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
