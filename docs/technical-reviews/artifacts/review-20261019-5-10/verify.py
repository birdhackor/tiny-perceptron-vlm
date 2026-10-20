"""核對選設定、完整文件切分與來源；第三方全文只在記憶體中處理。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import common, run, text


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pieces = []

    def handle_data(self, value):
        self.pieces.append(value)


receipts = []


def fetch(url, label, extra=()):
    command = ["curl", "-fsSL", "--max-time", "60", *extra, url]
    data = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    receipts.append(
        {
            "label": label,
            "url": url,
            "command_argv": command,
            "accessed_at": datetime.now(UTC).isoformat(),
            "bytes": len(data),
            "sha256": digest(data),
        }
    )
    return data


def read_html(url, label):
    raw = fetch(url, label)
    parser = Text()
    parser.feed(raw.decode())
    return " ".join(" ".join(parser.pieces).split())


sklearn = read_html("https://scikit-learn.org/1.9/modules/cross_validation.html", "scikit-learn")
for expected in (
    "knowledge about the test set can",
    "A test set should still be held out for final evaluation",
    "groups of dependent samples",
    "not represented at all in the paired training fold",
):
    assert expected in sklearn
receipts[-1].update(
    version=re.search(r"scikit-learn [\d.]+ documentation", sklearn)[0],
    inspected=["Cross-validation: evaluating estimator performance", "Cross-validation iterators for grouped data"],
    excerpts=[
        "A test set should still be held out for final evaluation",
        "the same group is not represented in both testing and training",
    ],
    inspection="已讀三份資料的用途、測試集調參洩漏與群組相依樣本段落；全文按故事或詩作分組是此處的領域套用。",
)
jmlr = read_html("https://www.jmlr.org/papers/v11/cawley10a.html", "Cawley and Talbot")
assert "a non-negligible variance introduces the potential for over-fitting in model selection" in jmlr
receipts[-1].update(
    version="JMLR 11(70):2079–2107 (2010), publisher HTML abstract",
    inspected=["Abstract"],
    excerpts=["a non-negligible variance introduces the potential for over-fitting in model selection"],
    inspection="讀出版方摘要，核對有限樣本模型選擇準則的變異與後續選擇偏差；沒有把摘要當作全文實驗的重現。",
)
card = read_html(
    "https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md",
    "TinyStories dataset card",
)
assert "tinystories-train.txt" in card.lower() and "tinystories-valid.txt" in card.lower()
receipts[-1].update(
    version="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    inspected=["Dataset Description / dataset files"],
    inspection="上游分別列出 train 與 valid 檔；自行從 train 留出的材料不會變成上游 valid。",
)

body = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.10.md").read_text()
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
variants = {}
for name, added, expected in (
    ("original", None, {"lr": 0.01, "validation": 0.8}),
    ("lower_candidate", {"lr": 0.1, "validation": 0.6}, {"lr": 0.01, "validation": 0.8}),
    ("higher_candidate", {"lr": 0.1, "validation": 0.9}, {"lr": 0.1, "validation": 0.9}),
):
    tree = ast.parse(code)
    if added:
        tree.body[0].value.elts.append(ast.parse(repr(added), mode="eval").body)
        ast.fix_missing_locations(tree)
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(tree, "<selection>", "exec"), namespace)
    assert namespace["selected"] == expected
    assert capture.getvalue().splitlines()[1] == "測試資料仍封存，尚未計算最終成績"
    variants[name] = {
        "selected": namespace["selected"],
        "compared_candidates": namespace["candidates"][1:],
        "stdout": capture.getvalue(),
    }

manifest = json.loads((ROOT / "assets/training/manifest.json").read_text())
recorded = json.loads((ROOT / "docs/course-experiments/results/real_text.json").read_text())
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
datasets = {}
records_by_asset = {}
for identifier, basename in (
    ("tinystories", "tinystories-train-512.jsonl"),
    ("chinese-poetry", "chinese-classical-train-365.jsonl"),
):
    entry = next(a for a in manifest["assets"] if a["id"] == identifier)
    archive = fetch(
        f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{revision}/{entry['archive']}",
        identifier + " fixed archive",
    )
    assert len(archive) == entry["archive_bytes"] and digest(archive) == entry["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive)) as package:
        member = entry["target"] + "/" + basename
        raw = package.extractfile(member).read()
        expected = next(f for f in entry["files"] if f["path"] == member)
        assert digest(raw) == expected["sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        if identifier == "tinystories":
            prefix = package.extractfile("text-initial/tinystories-train-prefix-complete.txt").read()
            stories = [b.decode().strip() for b in prefix.split(b"<|endoftext|>")[:-1]]
            assert len(stories) == len(rows) == 512
            assert stories == [r["text"] for r in rows]
            assert all(r["source_split"] == "train" for r in rows)
            original = fetch(
                "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt",
                "TinyStories upstream prefix",
                ("--range", "0-458751", "--max-filesize", "1048576"),
            )
            assert len(original) >= len(prefix)
            original_stories = [b.decode().strip() for b in original.split(b"<|endoftext|>")[:512]]
            assert stories == original_stories
        else:
            provenance = json.loads((ROOT / entry["source_metadata"]).read_text())
            original = fetch(provenance["source_url"], "Chinese poetry upstream anthology")
            source_poems = json.loads(original)
            rendered = [p["title"] + "\n" + p["author"] + "\n" + "\n".join(p["paragraphs"]) for p in source_poems]
            assert list(dict.fromkeys(rendered)) == [r["text"] for r in rows]
            assert len(source_poems) == 366 and len(rows) == 365
    parts = common.split_records(text._deduplicate_text(rows), seed=42)
    training_record_hash = common.records_sha256(parts["train"])
    assert training_record_hash == recorded["results"]["runs"][identifier]["training"]["records_sha256"]
    groups = {name: {r["family"] for r in selected} for name, selected in parts.items()}
    assert all(not groups[left] & groups[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")))
    split_summary = {}
    for name, selected in parts.items():
        saved = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
        expected = recorded["results"]["runs"][identifier]["data"][name]
        assert digest(saved) == expected["sha256"]
        windows = common.text_examples(selected, max_length=128)
        tok = common.ByteTokenizer()
        assert all(len(x) <= 128 and len(x) == len(y) for x, y in windows)
        actual_targets = torch.cat([y for _, y in windows]).tolist()
        expected_targets = [target for row in selected for target in tok.encode(row["text"]) + [tok.eos_id]]
        assert actual_targets == expected_targets
        split_summary[name] = {
            "documents": len(selected), "families": len(groups[name]), "saved_sha256": digest(saved),
            "matches_recorded_result": True, "windows": len(windows), "targets": len(actual_targets),
            "all_document_targets_preserved_in_order": True,
        }
    datasets[identifier] = {
        "archive_sha256_verified": True, "complete_upstream_documents_verified": True,
        "original_records": len(rows), "disjoint_groups": True, "splits": split_summary,
        "training_records_sha256_matches_recorded_result": training_record_hash,
        "record_keys": sorted(rows[0]),
    }
    records_by_asset[identifier] = rows

calls = []


def capture_splits(ctx, parts, name="data"):
    calls.append({"operation": "save_splits", "name": name, "rows": {k: len(v) for k, v in parts.items()}})
    return {k: {"records": len(v)} for k, v in parts.items()}


def capture_fit(model, rows, ctx, **kwargs):
    identifier = kwargs["name"]
    expected = common.split_records(text._deduplicate_text(records_by_asset[identifier]), seed=42)
    assert rows == expected["train"]
    assert not {r["family"] for r in rows} & {r["family"] for name in ("validation", "test") for r in expected[name]}
    calls.append({"operation": "fit_lm", "name": identifier, "rows": len(rows), "only_train_records": True})
    return {"probe_only": True}


with patch.object(text, "_asset_rows", side_effect=lambda ctx, identifier, basename: records_by_asset[identifier]), patch.object(
    text, "_save_splits", side_effect=capture_splits
), patch.object(text, "new_lm", return_value=object()), patch.object(text, "_evaluations", return_value={}), patch.object(
    text, "fit_lm", side_effect=capture_fit
):
    text.run_real_text(SimpleNamespace(seed=42, step_scale=1))

synthetic = [{"text": f"人工文件 {i}", "family": str(i)} for i in range(12)]
with tempfile.TemporaryDirectory(dir=ARTIFACT) as directory:
    context = SimpleNamespace(output=Path(directory))
    parts = common.split_records(synthetic, seed=42)
    saved_manifest = text._save_splits(context, parts)
    saved_files = {}
    for name, rows in parts.items():
        content = (Path(directory) / "data" / (name + ".jsonl")).read_text()
        assert [json.loads(line) for line in content.splitlines()] == rows
        saved_files[name] = {"content": content, "sha256": digest(content.encode())}
    manifest_bytes = (Path(directory) / "data/manifest.json").read_bytes()
    assert json.loads(manifest_bytes) == saved_manifest

spec = run.experiment_spec("real_text")
assert spec["module"] == "text" and spec["function"] == "run_real_text"
cli_stdout = io.StringIO()
with patch.object(sys, "argv", ["run", "--experiment", "real_text", "--device", "cpu"]), patch.object(
    run, "execute", return_value={"experiment_id": "real_text", "elapsed_seconds": 0, "evidence_status": "probe", "artifacts": []}
) as mocked_execute, contextlib.redirect_stdout(cli_stdout):
    run.main()
    cli_arguments = mocked_execute.call_args.args
    assert cli_arguments[0:2] == ("real_text", "cpu")

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-10/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "revision": revision},
    "result": "所有斷言通過；未執行模型訓練。",
    "source_sha256": digest(body.encode()), "selection_variants": variants, "datasets": datasets,
    "real_text_call_probe": {"calls": calls, "limitations": "資產讀取、模型建立、評估、更新與完整語料保存已以記錄器替代；核對原函式呼叫順序及訓練資料參數，沒有量測模型品質。"},
    "save_splits_probe": {"input": synthetic, "manifest": saved_manifest, "saved_files": saved_files},
    "cli_probe": {"spec": spec, "forwarded_arguments": [str(x) for x in cli_arguments], "stdout": cli_stdout.getvalue(), "scope": "只執行參數解析與配方選取；execute 以記錄器替代。"},
    "source_receipts": receipts,
    "repository_inputs": {
        p: digest((ROOT / p).read_bytes()) for p in (
            "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py",
            "assets/training/manifest.json", "assets/training/sources/tinystories.json", "assets/training/sources/chinese-poetry.json",
            "docs/course-experiments/results/real_text.json",
            "docs/course-experiments/plan.json",
        )
    },
}
(ARTIFACT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "selection_variants": variants, "datasets": datasets, "calls": calls}, ensure_ascii=False, indent=2))
