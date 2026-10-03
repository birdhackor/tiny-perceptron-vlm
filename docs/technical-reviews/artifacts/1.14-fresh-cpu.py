"""獨立 1.14 審查：原例、指定練習、前置算例與固定報告核對。"""
import contextlib
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import BigramLM

torch.set_num_threads(1)
torch.set_default_device("cpu")


def section(path, lesson):
    raw = (ROOT / path).read_bytes()
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    n = next(i for i, h in enumerate(headings) if h[0].startswith(f"## {lesson} ".encode()))
    return raw[headings[n].start():headings[n + 1].start() if n + 1 < len(headings) else len(raw)]


def code(path, lesson):
    return re.findall(rb"```python\n(.*?)```", section(path, lesson), re.S)[0].decode()


def run(program):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(program, "original-section-fence", "exec"), namespace)
    return namespace, stdout.getvalue()


program = code("course/chapters/01.md", "1.14")
records = []
for label, program_variant in [
    ("original_seed42", program),
    ("restart_seed42", program),
    ("exercise_seed7", program.replace("manual_seed(42)", "manual_seed(7)")),
    ("exercise_three_steps", program.replace("range(12)", "range(3)").replace("len(output) == 13", "len(output) == 4")),
]:
    ns, stdout = run(program_variant)
    p = ns["p"]
    assert p.device.type == "cpu" and p.shape == (4, 4)
    assert torch.allclose(p.sum(1), torch.ones(4))
    records.append({"case": label, "stdout": stdout.strip(), "chars": ns["chars"],
                    "p": p.tolist(), "row_sums": p.sum(1).tolist(), "length": len(ns["output"]),
                    "start": ns["output"][0], "current_type": type(ns["current"]).__name__,
                    "generator_seed": ns["g"].initial_seed(),
                    "generator_device": str(ns["g"].device)})
assert records[0] == {**records[1], "case": "original_seed42"}
for key in ("chars", "p", "row_sums", "length", "start"):
    assert records[0][key] == records[2][key]
assert records[0]["stdout"] != records[2]["stdout"]
assert records[3]["length"] == 4

# 多步重複抽樣各自採單一元素。句號後仍新增下一字；程式沒有標點停止條件。
ns, _ = run(program)
g = torch.Generator().manual_seed(42)
sample = torch.multinomial(ns["p"][0], 1, generator=g)
api = {"sample_shape": list(sample.shape), "sample_dtype": str(sample.dtype),
       "sample_item": sample.item(), "sample_item_type": type(sample.item()).__name__}
punctuation = ["。"]
current = 3
for _ in range(3):
    current = torch.multinomial(ns["p"][current], 1, generator=g).item()
    punctuation.append(ns["chars"][current])
assert len(punctuation) == 4

# 精確有理數核對每列及非零路徑；不把單次抽樣當頻率估計。
from fractions import Fraction
decimal_rows = [[".05", ".85", ".05", ".05"], [".40", ".05", ".50", ".05"],
                [".05", ".70", ".05", ".20"], [".50", ".05", ".40", ".05"]]
assert all(sum(map(Fraction, row)) == 1 for row in decimal_rows)
paths = {"貓看狗": str(Fraction(".85") * Fraction(".50")),
         "貓貓": str(Fraction(".05")), "。。": str(Fraction(".05"))}

prerequisite = {}
for lesson in ("1.1", "1.5"):
    ns, stdout = run(code("course/chapters/01.md", lesson))
    prerequisite[lesson] = {"stdout": stdout.strip()}
    if lesson == "1.1":
        prerequisite[lesson].update(chars=ns["chars"], ids=ns["ids"], restored=ns["restored"])
    else:
        prerequisite[lesson].update(chars=ns["chars"], cat_counts=ns["counts"][ns["row"]].tolist(),
                                    cat_probabilities=ns["probability"][ns["row"]].tolist())

documents = toy_documents()
split = split_documents(documents, 42)
denominators = {name: sum(len(s) + 1 for s in rows) for name, rows in split.items()}
report_path = ROOT / "docs/course-experiments/results/simple_models.json"
report = json.loads(report_path.read_text())
bigram = report["results"]["runs"]["bigram"]
assert len(documents) == 12
assert {k: len(v) for k, v in split.items()} == {"train": 9, "validation": 1, "test": 2}
assert bigram["steps"] == 200 and bigram["context"] == 1
assert bigram["samples"][0] == {"prompt": "顏色=", "generated": "三角。"}
assert bigram["after_nll_same_post_update_time"]["train"] < bigram["before_nll"]["train"]
formal = {"path": str(report_path.relative_to(ROOT)), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
          "revision": report["revision"], "device": report["device"], "seed": report["seed"],
          "torch_version": report["torch_version"], "python_version": report["python_version"],
          "mode": report["evidence_status"], "bigram": {k: bigram[k] for k in
              ("parameters", "context", "steps", "before_nll", "after_nll_same_post_update_time", "samples")},
          "source_hash_match": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == report["code_sha256"][p]
              for p in ("tiny_perceptron/data.py", "tiny_perceptron/simple.py", "scripts/course_experiments/text.py")}}
# 任意權重下相同最近字必給同一列；此檢查只驗證模型結構，沒有更新權重。
model = BigramLM(17)
equal_context = bool(torch.equal(model(torch.tensor([[3, 8, 4]])), model(torch.tensor([[7, 1, 4]]))))
assert equal_context

result = {"command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/1.14-fresh-cpu.py",
          "environment": {"python": sys.version, "torch": str(torch.__version__),
              "torch_git_version": str(torch.version.git_version), "device": "cpu",
              "cuda_build": str(torch.version.cuda), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
              "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
          "section_sha256": hashlib.sha256(section("course/chapters/01.md", "1.14")).hexdigest(),
          "cases": records, "api": api, "punctuation_not_eos": punctuation,
          "exact_row_sums": ["1"] * 4, "nonzero_path_probabilities": paths,
          "prerequisites": prerequisite, "documents": {"all_count": len(documents), "split": split,
              "target_denominators_including_eos": denominators}, "formal_report_inspection": formal,
          "bigram_equal_last_id_equal_logits": equal_context,
          "scope": "原例與練習之短 CPU 抽樣；固定正式報告的資料、設定、輸出核對；未重跑正式訓練。"}
print(json.dumps(result, ensure_ascii=False, indent=2))
