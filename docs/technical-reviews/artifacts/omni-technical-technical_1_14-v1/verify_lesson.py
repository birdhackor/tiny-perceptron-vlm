"""1.14 獨立技術核對：直接執行凍結內嵌程式，只載入既有模型，不訓練。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.text import _simple_examples, _simple_sample
from scripts.infer_simple import generate_simple, load_simple_checkpoint
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import BigramLM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/1.14.md"
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
freeze = json.loads(freeze_path.read_text())
page = next(x for x in freeze["pages"] if x["page_id"] == "1.14")
assert sha(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
body = snapshot.read_text()
assert not re.search(r"!\[[^\]]*\]\(", body)
code, = re.findall(r"```python\n(.*?)```", body, re.S)


def embedded(seed=42, count=12):
    edited = code.replace("manual_seed(42)", f"manual_seed({seed})")
    edited = edited.replace("range(12)", f"range({count})")
    edited = edited.replace("len(output) == 13", f"len(output) == {count + 1}")
    ns, printed = {}, io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(edited, f"{snapshot}#embedded", "exec"), ns)
    chars, p = ns["chars"], ns["p"]
    ids = [chars.index(char) for char in ns["output"]]
    assert all(type(i) is int for i in ids)
    assert len(ids) == count + 1
    assert chars == ["貓", "看", "狗", "。"]
    assert p.dtype == torch.float32 and p.shape == (4, 4)
    assert torch.allclose(p.sum(1), torch.ones(4), rtol=0, atol=1e-7)
    assert torch.all(torch.isfinite(p)) and torch.all(p > 0)
    assert abs(float(p[0, 1]) - 0.85) < 1e-7
    assert type(ns["current"]) is int
    return {
        "seed": seed,
        "draws": count,
        "length": len(ids),
        "text": printed.getvalue().strip(),
        "ids": ids,
        "period_positions": [i for i, v in enumerate(ids) if v == 3],
        "trace": [
            {"row": a, "selected": b, "probability": float(p[a, b])}
            for a, b in zip(ids, ids[1:])
        ],
        "row_sums": p.sum(1).tolist(),
        "dtype": str(p.dtype),
        "input_shape_per_draw": [4],
        "sample_shape": [1],
        "item_type": type(ns["current"]).__name__,
    }


base = embedded()
repeat = embedded()
assert base == repeat
variants = [embedded(7), embedded(42, 3), embedded(7, 3)]

record_path = ROOT / "docs/course-experiments/results/simple_models.json"
record = json.loads(record_path.read_text())
run = record["results"]["runs"]["bigram"]
parts = split_documents(toy_documents(), seed=42)
assert len(toy_documents()) == len(set(toy_documents())) == 12
assert {k: len(v) for k, v in parts.items()} == {"train": 9, "validation": 1, "test": 2}
data_receipts = {}
for name, documents in parts.items():
    path = ROOT / f"outputs/course-experiments/course-v1/simple_models/data/{name}.jsonl"
    stored = [json.loads(line)["text"] for line in path.read_text().splitlines()]
    assert stored == documents
    assert sha(path) == record["results"]["data"][name]["sha256"]
    data_receipts[name] = {"documents": len(documents), "sha256": sha(path)}

directory = ROOT / "checkpoints/course/simple_models"
export_path, download_path = directory / "export-manifest.json", directory / "download-manifest.json"
export, download = json.loads(export_path.read_text()), json.loads(download_path.read_text())
checkpoint = directory / "bigram.pt"
export_item = next(x for x in export["files"] if x["output"] == "bigram.pt")
download_item = next(x for x in download["files"] if x["output"] == "bigram.pt")
source_item = next(x for x in record["artifacts"] if x["path"] == "bigram.pt")
assert sha(checkpoint) == export_item["sha256"] == download_item["sha256"]
assert export_item["source_sha256"] == source_item["sha256"]
assert export["provenance"]["revision"] == record["revision"]
model, saved, vocabulary = load_simple_checkpoint(checkpoint, "cpu")
assert saved["kind"] == "bigram" and saved["context"] == 1
assert len(vocabulary) == 15
torch.manual_seed(42)
before_model = BigramLM(len(vocabulary) + 2)
evaluation = {}
with torch.no_grad():
    for split, documents in parts.items():
        x, y = _simple_examples(documents, vocabulary, 1)
        before, after = float(F.cross_entropy(before_model(x), y)), float(F.cross_entropy(model(x), y))
        expected_before = run["before_nll"][split]
        expected_after = run["after_nll_same_post_update_time"][split]
        assert abs(before - expected_before) < 1e-6
        assert abs(after - expected_after) < 1e-6
        evaluation[split] = {
            "documents": len(documents),
            "targets_including_boundaries": len(y),
            "before_observed": before,
            "before_recorded": expected_before,
            "after_observed": after,
            "after_recorded": expected_after,
        }
    samples = [_simple_sample(model, vocabulary, 1, p, "cpu") for p in ("顏色=", "顏色=紅；形狀=")]
    assert samples == run["samples"]
    direct_sample = generate_simple(model, vocabulary, 1, "顏色=", 24)
    assert direct_sample["answer"] == "三角。" and direct_sample["eos"]
    a = torch.tensor([[vocabulary[c] for c in "顏色="]])
    b = torch.tensor([[vocabulary[c] for c in "形狀="]])
    assert torch.equal(model(a), model(b))
    logits_equal = torch.equal(model(a), model(b))

official = Path("/workspace/work/reference/omni-technical-technical_1_14-v1")
official_docs = official / "pytorch-_torch_docs.py"
installed_docs = Path(torch.__file__).parent / "_torch_docs.py"
assert sha(official_docs) == sha(installed_docs)
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu", "platform": platform.platform(), "default_dtype": str(torch.get_default_dtype()),
    },
    "source_sha256": sha(snapshot), "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "freeze_sha256": sha(freeze_path), "figures": [],
    "embedded": base, "identical_seed_repeat_equal": base == repeat, "variants": variants,
    "data": data_receipts,
    "provenance": {
        "experiment_record_sha256": sha(record_path),
        "experiment_revision": record["revision"],
        "published_checkpoint_sha256": sha(checkpoint),
        "source_checkpoint_sha256_in_export": export_item["source_sha256"],
        "export_manifest_sha256": sha(export_path),
        "download_manifest_sha256": sha(download_path),
        "public_revision": download["revision"],
        "old_local_checkpoint_sha256": sha(ROOT / "outputs/course-experiments/course-v1/simple_models/bigram.pt"),
        "note": "舊 outputs checkpoint 與紀錄雜湊不符；未用該檔驗證。使用通過 download/export 雜湊及 source_sha256 鏈的公開推論檔。",
    },
    "bigram": {"evaluation": evaluation, "samples": samples, "inference": direct_sample,
        "equal_suffix_logits_equal": logits_equal, "context": saved["context"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "recorded_steps_not_rerun": run["steps"], "training_performed": False,
        "vocabulary_characters": len(vocabulary), "prompt_unknown_characters": direct_sample["unknown_prompt_characters"]},
    "official_installed_torch_docs_equal": True,
    "checked_code_sha256": {p: sha(ROOT / p) for p in ["tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py", "scripts/infer_simple.py"]},
    "result": "所有斷言通過；只執行 CPU 抽樣、既有模型前向、資料及 provenance 核對，未更新參數。",
}
output = Path(sys.argv[1])
with output.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"path": str(output), "sha256": sha(output), "result": result["result"],
    "embedded_text": base["text"], "samples": samples, "evaluation": evaluation}, ensure_ascii=False))
