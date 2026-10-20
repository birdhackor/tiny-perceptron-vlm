"""核對示意計算、既有實驗紀錄、程式契約與原始來源；不進行訓練。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import generate


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def score(truth, prediction):
    return {
        "correct": sum(a == b for a, b in zip(truth, prediction, strict=True)),
        "accuracy": sum(a == b for a, b in zip(truth, prediction, strict=True)) / len(truth),
        "exact_match": truth == prediction,
    }


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-8/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "accessed_on": datetime.now(timezone.utc).date().isoformat(),
}
freeze = ROOT / "docs/course-review-20261019/freeze-01/sources/5.8.md"
result["source_sha256"] = digest(freeze.read_bytes())
assert result["source_sha256"] == "967fe1669653d4848baaa11b90496cc152f7e30a4a9512078a16491e6965eca3"
result["freeze_sha256"] = digest((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes())
assert result["freeze_sha256"] == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
truth = [1, 2, 3, 4]
result["toy_scores"] = {
    "teacher_forced": score(truth, [1, 2, 0, 4]),
    "generated": score(truth, [1, 2, 0, 0]),
    "exercise_teacher_forced": score(truth, [1, 2, 3, 4]),
    "exercise_generated": score(truth, [1, 2, 0, 0]),
}
assert [row["accuracy"] for row in result["toy_scores"].values()] == [0.75, 0.5, 1.0, 0.5]
result["strict_zip"] = []
for left, right in [([1, 2], [1]), ([1], [1, 2])]:
    try:
        list(zip(left, right, strict=True))
    except ValueError as error:
        result["strict_zip"].append({"left": left, "right": right, "error": str(error)})
assert len(result["strict_zip"]) == 2
result["python_comparisons"] = {
    "bool_as_int": [int(False), int(True)],
    "boolean_sum": sum([True, True, False, True]),
    "same_values": [1, 2] == [1, 2],
    "different_order": [1, 2] == [2, 1],
    "different_length": [1, 2] == [1, 2, 3],
    "different_value": [1, 2] == [1, 3],
}
result["nll_derivation"] = {
    "formula": "單一正確目標的交叉熵為 -ln(p_target)；平均為各有效目標之和除以有效目標數。",
    "probabilities": [0.6, 0.9],
    "negative_logs": [-math.log(0.6), -math.log(0.9)],
    "counterexample": {
        "before_target_probabilities": [0.6, 0.6],
        "after_target_probabilities": [0.99, 0.49],
        "before_mean_nll": -math.log(0.6),
        "after_mean_nll": -(math.log(0.99) + math.log(0.49)) / 2,
        "before_binary_accuracy": 1.0,
        "after_binary_accuracy": 0.5,
        "scope": "二元候選且每個位置的另一候選機率為一減正確候選機率；平均代價下降仍可失去整串匹配。",
    },
}
tok = ByteTokenizer()
sample = "as a big was a she was a ber the"
result["byte_tokenizer"] = {
    "sample": sample, "characters": len(sample), "utf8_bytes": len(sample.encode()),
    "encoded_ids": tok.encode(sample), "decoded": tok.decode(tok.encode(sample)), "eos_id": tok.eos_id,
}
assert len(sample) == len(sample.encode()) == len(tok.encode(sample))
rows = generate_records("toy-text")
parts = split_records(rows, seed=42)
validation = parts["validation"]
examples = text_examples(validation, mode="text", max_length=128)
result["toy_data"] = {
    "records": len(rows), "split_counts": {k: len(v) for k, v in parts.items()},
    "validation": validation,
    "possible_continuations": [r["text"] for r in rows if r["text"].startswith("color=blue;shape=circle;")],
    "effective_tokens": sum(int((labels != IGNORE).sum()) for _, labels in examples),
}
assert validation[0]["text"] == "color=blue;shape=circle;side=left."
assert result["toy_data"]["effective_tokens"] == 35


class GreedyProbe(torch.nn.Module):
    """以固定候選分數記下 generate 每次實際使用的前文。"""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"max_length": 128})()
        self.inputs = []

    def forward(self, ids, cache=None):
        self.inputs.append(ids.tolist())
        logits = torch.zeros((*ids.shape, 16), device=ids.device)
        logits[..., 9] = 3
        return {"logits": logits, "cache": None}


probe = GreedyProbe()
output = generate(probe, torch.tensor([[1, 8]]), max_new_tokens=3)
result["greedy_probe"] = {"inputs": probe.inputs, "output": output.tolist()}
assert output.tolist() == [[1, 8, 9, 9, 9]]
assert probe.inputs == [[[1, 8]], [[1, 8, 9]], [[1, 8, 9, 9]]]
result["experiments"] = {}
paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]
for name in ["text_foundation", "real_text"]:
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    record = json.loads(path.read_bytes())
    body = record["results"] if name == "text_foundation" else record["results"]["runs"]["tinystories"]
    entry = {"record_path": str(path.relative_to(ROOT)), "record_sha256": digest(path.read_bytes())}
    entry["configuration"] = {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]}
    entry["data"] = body["data"]
    entry["training"] = {k: v for k, v in body["training"].items() if k != "history"}
    entry["evaluation"] = {}
    for when in ["before", "after"]:
        for split in ["validation", "test"]:
            data = body[when][split]
            computed = data["nll_sum"] / data["effective_tokens"]
            assert math.isclose(computed, data["nll"], rel_tol=0, abs_tol=1e-12)
            assert data["effective_tokens"] == data["raw_utf8_bytes"] + data["records"]
            entry["evaluation"][when + "_" + split] = {
                "nll": data["nll"], "recomputed_nll": computed, "rounded_nll": f"{computed:.5f}",
                "nll_sum": data["nll_sum"], "effective_tokens": data["effective_tokens"],
                "records": data["records"], "examples": data["examples"], "raw_utf8_bytes": data["raw_utf8_bytes"],
                "samples": data["samples"],
            }
            for row in data["samples"]:
                assert tok.decode(row["generated_ids"]) == row["generated"]
                assert len(row["generated_ids"]) <= 32
    entry["code_binding"] = {
        p: {"recorded": record["code_sha256"][p], "current": digest((ROOT / p).read_bytes())} for p in paths
    }
    result["experiments"][name] = entry
real = result["experiments"]["real_text"]
assert all(v["recorded"] == v["current"] for v in real["code_binding"].values())
assert len(real["evaluation"]["after_validation"]["samples"]) == 8
assert len(real["evaluation"]["after_test"]["samples"]) == 8
assert real["evaluation"]["after_validation"]["samples"][1]["generated"] == sample
assert real["evaluation"]["after_validation"]["samples"][1]["prompt"] == "Once upon a time there w"
result["asset_availability"] = {}
manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
for asset in manifest["assets"]:
    if asset["id"] in ["tinystories", "chinese-poetry"]:
        path = ROOT / asset["archive"]
        result["asset_availability"][asset["id"]] = {
            "path": asset["archive"], "local_bytes": path.stat().st_size,
            "lfs_pointer": path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1\n"),
            "expected_archive_sha256": asset["archive_sha256"],
        }
for args in [["scripts/fetch_training_assets.py", "--list"], ["-m", "scripts.course_experiments.run", "--help"]]:
    command = [str(ROOT / ".venv/bin/python"), *args]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    result.setdefault("read_only_commands", []).append({"command": command, "stdout": process.stdout, "stderr": process.stderr})
result["repository_hashes"] = {
    p: digest((ROOT / p).read_bytes()) for p in paths + ["scripts/prepare_data.py", "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py", "course/training.md"]
}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


readings = [
    ("python_functions", "https://docs.python.org/3.13/library/functions.html", "zip()", "zip() is lazy", "讀 zip 與 sum：zip 依位置配對；strict 在迭代耗盡且長度不同時拋 ValueError，sum 完整消耗本節迭代器。"),
    ("python_types", "https://docs.python.org/3.13/library/stdtypes.html", "Common Sequence Operations; Boolean Type", "bool is a subclass of int", "讀同型序列逐元素比較與 bool 的整數語意；本節兩邊都是 int 清單。"),
    ("teacher_forcing", "https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html", "DecoderRNN.forward; Training", "Teacher forcing: Feed the target as the next input", "讀 DecoderRNN 與訓練說明：有 target_tensor 時餵正確目標，否則將自己 topk 預測接回。僅核對概念，不主張本專案使用該 RNN。"),
    ("scheduled_sampling", "https://arxiv.org/html/1506.03099v3", "Model; Training; Inference; Image Captioning", "Since we do not have access to the true previous token", "讀序列似然分解、最大化訓練似然、自回歸推論與累積錯誤；亦讀影像描述有多種合法答案與似然和任務指標可能不同的說明。"),
    ("answer_equivalence", "https://aclanthology.org/2022.emnlp-main.20/", "Abstract", "accepting answers that are equivalent to or improve over the reference", "讀作者摘要：有限參考答案的涵蓋限制、等價答案及評分對問題的依賴；只用來支持字串相同不等於所有任務的語意正確。"),
    ("utf8", "https://www.rfc-editor.org/rfc/rfc3629.txt", "UTF-8 definition", "US-ASCII characters are encoded in one octet", "讀 UTF-8 對 US-ASCII 使用一 octet 與其他字元變長編碼的規定；本例字母、空格皆在 ASCII。"),
]
result["source_readings"] = []
for identifier, url, locator, excerpt, note in readings:
    command = ["curl", "-L", "--fail", "--silent", "--show-error", url]
    raw = subprocess.check_output(command)
    parser = TextParser()
    parser.feed(raw.decode())
    text = " ".join(" ".join(parser.parts).split())
    assert excerpt in text
    title = re.search(r"<title>(.*?)</title>", raw.decode(), re.S)
    result["source_readings"].append({
        "id": identifier, "url": url, "command": command, "exit_code": 0,
        "retrieved_sha256": digest(raw), "retrieved_bytes": len(raw),
        "title": title.group(1) if title else "RFC 3629 (November 2003)",
        "locator": locator, "excerpt": excerpt, "inspection_note": note,
        "storage": "全文只在記憶體讀取；只保存定位、短摘錄、讀取收據及全文雜湊。",
    })
result["limitations"] = [
    "未重跑既有 CUDA 訓練或用原檢查點重新生成；實測值核對既有紀錄、完整分母、算術及程式。",
    "TinyStories 與詩文的資料包本機是 LFS pointer，未另行下載；故事全文的逐筆 token 重數未重新執行。",
    "屬性實驗的 common.py 與 text.py 全檔 hash 不同於現行版；data.py 與 model.py 相同，屬性切分與有效目標數已由現行產生器獨立重算。",
    "故事實驗四個相關程式檔全檔 hash 均與紀錄相同。",
    "故事續寫只代表已列的一筆 greedy 結果，不外推全部提示、其他 seed、硬體或自然語言能力。",
]
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
