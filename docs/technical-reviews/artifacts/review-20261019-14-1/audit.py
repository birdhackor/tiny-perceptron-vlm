"""離線算例、模型接線、實驗紀錄與原始來源的獨立核對。"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.attention import CausalAttention
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import rope as library_rope


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "annotation"):
            self.skip += 1
        if tag == "math":
            self.parts.append(" " + attrs.get("alttext", "") + " ")
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "annotation", "math"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def plain(raw):
    parser = PlainText()
    parser.feed(raw)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, {"url": url, "command": command, "bytes": len(result.stdout), "sha256": sha(result.stdout)}


def source_receipts():
    raw, rope_receipt = fetch("https://arxiv.org/html/2104.09864v5")
    html = raw.decode()
    inspected = html[html.index('id="S3.SS1"'):html.index('id="S3.SS3"')]
    text = plain(inspected)
    assert "10000^{-2(i-1)/d}" in text
    assert "orthogonal matrix" in text
    rope_receipt.update(
        version="arXiv:2104.09864v5，2023-11-08；HTML 對應同一修訂版",
        inspected="Formulation、A 2D case、General form；公式 (11)–(16)",
        inspected_text_sha256=sha(text.encode()),
        short_excerpt="we divide the d-dimension space into d/2 sub-spaces",
        findings="逐對二維旋轉；偶數維度；轉速 10000**(-2*(i-1)/d)；R(m).T R(n)=R(n-m)，固定內容的點積只依位置差。",
    )
    _, pdf_receipt = fetch("https://arxiv.org/pdf/2104.09864v5")
    raw, attention_receipt = fetch("https://arxiv.org/html/1706.03762v7")
    html = raw.decode()
    text = plain(html[html.index('id="S3.SS2"'):html.index('id="S3.SS3"')])
    assert "softmax" in text and "different, learned linear projections" in text
    attention_receipt.update(
        version="arXiv:1706.03762v7，2023-08-02",
        inspected="Attention、Scaled Dot-Product Attention、Multi-Head Attention、Applications of Attention in our Model；公式 (1)",
        inspected_text_sha256=sha(text.encode()),
        short_excerpt="Multi-Head Attention consists of several attention layers running in parallel.",
        findings="Q/K 點積先縮放，再 softmax 成權重；V 是混合內容；每個頭有自己的 Q/K/V 投影。因果遮罩另行限制可見 Key。",
    )
    raw, math_receipt = fetch("https://docs.python.org/3.13/library/math.html")
    html = raw.decode()
    snippets = {}
    for name in ("math.cos", "math.sin", "math.pi"):
        start = html.index('id="' + name + '"')
        snippets[name] = plain(html[start:html.index("</dd>", start)])
    assert "Python 3.13.16" in html
    math_receipt.update(
        version="Python 3.13.16 官方文件",
        inspected="math.cos、math.sin、math.pi",
        short_excerpts=["Return the cosine of x radians.", "Return the sine of x radians."],
        findings="cos/sin 接受弧度；pi 是可用浮點精度下的圓周率。",
        inspected_text_sha256=sha(json.dumps(snippets, sort_keys=True).encode()),
    )
    return {"accessed_on": datetime.now(timezone.utc).date().isoformat(), "rope": rope_receipt,
            "rope_pdf": pdf_receipt, "attention": attention_receipt, "python_math": math_receipt,
            "storage": "完整第三方內容僅在記憶體讀取；只保存收據、短摘錄與核對結論。"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
                   "docs/technical-reviews/artifacts/review-20261019-14-1/audit.py "
                   "--section-run outputs/reviewer-tools/runs/20261020T054747974309Z",
        "environment": {"python": sys.version, "torch": str(torch.__version__),
                        "device": "cpu", "platform": platform.platform(), "seed": "20261020"},
    }
    run = (ROOT / args.section_run).resolve()
    copied = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
        raw = (run / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "text": raw.decode()}
    execution = json.loads(copied["execution.json"]["text"])
    assert execution["exit_code"] == 0
    assert execution["source_sha256"] == "fa53ca9cee66aed018448db1fd4ee784fc741efd15840c6f8ab9ae711fc1e094"
    result["section_execution_copy"] = {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/14.md#14.1" --execute',
        "copy_policy": "各檔 text 保存原始 UTF-8 全文；sha256 可從 text.encode('utf-8') 重算。",
        "files": copied,
    }
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/14.1.md"
    body = frozen.read_text()
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for block in blocks:
            exec(compile(block, str(frozen), "exec"), namespace)
    assert stdout.getvalue() == copied["stdout.txt"]["text"]
    rotate, score, rope = (namespace[name] for name in ("rotate", "score", "rope"))
    result["frozen_code_stdout"] = stdout.getvalue()
    result["trigonometry"] = {
        "unit_coordinates": {str(degree): [math.cos(math.radians(degree)), math.sin(math.radians(degree))]
                             for degree in (45, 90, 135)},
        "half_turn_radians": math.radians(180), "degrees_per_radian": math.degrees(1),
        "shift_rotation_degrees": 10 * 45, "position_gap_degrees": (5 - 2) * 45,
        "rotation_formula": "R(t)=[ [cos(t),-sin(t)], [sin(t),cos(t)] ]; R(t)[x,y]=[cos(t)x-sin(t)y,sin(t)x+cos(t)y]",
    }
    result["single_pair_scores"] = {str((m, n)): {"raw": score(m, n), "rounded": round(score(m, n), 4)}
                                     for m, n in ((2, 5), (12, 15), (3, 5), (22, 25), (22, 26))}
    assert [round(score(m, n), 4) for m, n in ((2, 5), (12, 15), (3, 5), (22, 25), (22, 26))] == [-0.7071, -0.7071, 0.0, -0.7071, -1.0]
    positions = torch.arange(16)
    x = torch.ones(1, 1, 16, 8)
    r = rope(x, positions)[0, 0]
    frequency = 10000 ** (-torch.arange(0, 8, 2) / 8)
    measured = [(r[2] @ r[5]).item(), (r[12] @ r[15]).item(), (r[3] @ r[5]).item()]
    expected = [2 * sum(math.cos(gap * 10000 ** (-2 * i / 8)) for i in range(4)) for gap in (3, 3, 2)]
    assert [round(v, 4) for v in measured] == [3.9298, 3.9298, 5.1274]
    assert max(abs(a - b) for a, b in zip(measured, expected)) < 1e-6
    last = rope(torch.ones(1, 1, 16, 2), positions)[0, 0]
    exercise = (last[2] @ last[5]).item()
    assert abs(exercise - 2 * math.cos(3)) < 1e-6
    pairs = torch.stack((torch.tensor([10, 11]), torch.tensor([20, 21])), -1)
    assert pairs.flatten(-2).tolist() == [10, 20, 11, 21]
    check = torch.randn(2, 4, 16, 8)
    assert torch.equal(rope(check, positions), library_rope(check, positions))
    per_batch = torch.stack((positions, positions + 3))
    batched = library_rope(check, per_batch)
    reference = torch.cat([library_rope(check[i:i + 1], per_batch[i]) for i in range(2)])
    assert torch.equal(batched, reference)
    odd_error = None
    try:
        rope(torch.ones(1, 1, 2, 3), torch.arange(2))
    except ValueError as error:
        odd_error = str(error)
    assert odd_error is not None
    raw = torch.randn(2, 4, 1, 8)
    before, after = library_rope(raw, torch.tensor([2])), library_rope(raw, torch.tensor([12]))
    other = torch.randn_like(raw)
    dot_before = (before * library_rope(other, torch.tensor([5]))).sum(-1)
    dot_after = (after * library_rope(other, torch.tensor([15]))).sum(-1)
    norm_error = (before.norm(dim=-1) - raw.norm(dim=-1)).abs().max().item()
    shift_error = (dot_after - dot_before).abs().max().item()
    assert norm_error < 2e-6 and shift_error < 1e-5
    result["tensor_rope"] = {
        "input_shape": list(x.shape), "selected_shape": list(r.shape), "frequency": frequency.tolist(),
        "angle_table_shape": [16, 4], "broadcast_angle_shape": [1, 1, 16, 4],
        "raw_scores": measured, "float64_scalar_formula": expected,
        "score_rounding": "四位小數相同；浮點原始值不要求逐位相等。",
        "two_feature_exercise": {"observed": exercise, "expected": 2 * math.cos(3)},
        "stack": pairs.tolist(), "flatten": pairs.flatten(-2).tolist(),
        "random_library_equal": True, "per_batch_positions_equal": True,
        "odd_feature_error": odd_error, "norm_max_absolute_error": norm_error,
        "shared_shift_max_absolute_error": shift_error,
        "dtype_outputs": {str(dtype): str(library_rope(check.to(dtype), positions).dtype)
                          for dtype in (torch.float32, torch.float64, torch.bfloat16)},
        "derivation": "每對 R(mθ)u · R(nθ)v = u.T R((n-m)θ)v。共同加 s 後差仍為 n-m；R.T R=I 保長。對 u=v=[1,1]，長度平方為 2，所以每對為 2cos((n-m)θ)，多對點積相加。",
        "all_ones_input_swap": "每對的 a=b=1，交換輸入 a/b 無法辨別；隨機不同座標可辨別此類錯誤。",
    }
    attention = CausalAttention(8, heads=2, rotary=True)
    inputs = torch.randn(2, 4, 8)
    calls = []

    def capture(tensor, pos):
        calls.append({"shape": list(tensor.shape), "positions": pos.tolist()})
        return library_rope(tensor, pos)

    with patch("tiny_perceptron.modern.rope", capture):
        output, cache = attention(inputs)
    expected_v = attention.v(inputs).view(2, 4, 2, 4).transpose(1, 2)
    assert len(calls) == 2 and torch.equal(cache[1], expected_v)
    baseline = TinyLM(ModelConfig(width=64, layers=2, heads=4))
    rotary = TinyLM(ModelConfig(width=64, layers=2, heads=4, rotary=True))
    counts = [sum(p.numel() for p in model.parameters()) for model in (baseline, rotary)]
    assert counts == [141568, 133376] and counts[0] - counts[1] == 128 * 64
    assert rotary.position is None and tuple(baseline.position.weight.shape) == (128, 64)
    assert rotary(torch.tensor([[1, 20, 30]]))["logits"].shape == (1, 3, 264)
    type_error = None
    try:
        rotary.position(torch.arange(3))
    except TypeError as error:
        type_error = str(error)
    assert type_error is not None
    result["integration"] = {"rope_calls": calls, "v_cache_unrotated": True,
                             "attention_output_shape": list(output.shape), "parameter_counts": counts,
                             "removed_position_parameters": 128 * 64, "position_none_call_error": type_error,
                             "rotary_forward_logits_shape": [1, 3, 264]}
    from scripts.course_experiments.run import experiment_spec

    help_command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
    help_result = subprocess.run(help_command, cwd=ROOT, capture_output=True, text=True, check=True)
    spec = experiment_spec("modern")
    assert spec["module"] == "architecture" and spec["function"] == "run_modern"
    result["command_route"] = {"command": help_command, "exit_code": help_result.returncode,
                               "stdout": help_result.stdout, "stderr": help_result.stderr, "experiment_spec": spec,
                               "sources_sha256": {path: sha((ROOT / path).read_bytes()) for path in
                                                  ("scripts/course_experiments/run.py", "docs/course-experiments/plan.json", "course/training.md")}}
    log_path = ROOT / "docs/course-experiments/results/modern.json"
    log = json.loads(log_path.read_bytes())
    variants = log["results"]["variants"]
    summary = {}
    for name, variant in variants.items():
        config = ModelConfig(**variant["model"]["config"])
        model = TinyLM(config)
        parameters = sum(p.numel() for p in model.parameters())
        assert parameters == variant["model"]["parameters"]
        training = variant["training"]
        assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 240
        assert training["skipped_updates"] == 0
        assert config.width == 64 and config.layers == 2 and config.heads == 4
        expected_changes = {key: value for key, value in asdict(config).items()
                            if value != variants["baseline"]["model"]["config"][key]}
        assert expected_changes == variant["changes"]
        heldout = {}
        for split, values in variant["heldout"].items():
            recomputed = values["nll_sum"] / values["effective_tokens"]
            assert recomputed == values["nll"]
            heldout[split] = {key: values[key] for key in ("nll", "nll_sum", "effective_tokens", "examples", "records")}
            heldout[split]["recomputed_nll"] = recomputed
        summary[name] = {"config": asdict(config), "changes": variant["changes"], "parameters": parameters,
                         "optimizer_updates": training["optimizer_updates"], "training_effective_tokens": training["effective_tokens"],
                         "training_history_steps": [row["step"] for row in training["history"]], "heldout": heldout}
    asset = log["assets"][0]
    archive = ROOT / asset["archive"]
    archive_raw = archive.read_bytes()
    archive_receipt = {"local_bytes": len(archive_raw), "local_sha256": sha(archive_raw)}
    if archive_raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        assert ("oid sha256:" + asset["archive_sha256"]).encode() in archive_raw
        archive_receipt["local_lfs_pointer"] = archive_raw.decode()
        archive_raw, archive_receipt["download"] = fetch(
            "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/" + asset["archive"]
        )
        archive_receipt["inspection_note"] = "本機是 LFS 指標，完整包只下載到記憶體；SHA-256 與實驗紀錄相同。"
    archive_sha = sha(archive_raw)
    assert archive_sha == asset["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as handle:
        member = next(item for item in handle.getmembers() if item.name.endswith("tinystories-train-512.jsonl"))
        rows = [json.loads(line) for line in handle.extractfile(member).read().splitlines() if line.strip()]
    for row in rows:
        row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
    split = split_records(rows, log["seed"])
    replay = {}
    for name, records in split.items():
        assert records_sha256(records) == log["results"]["dataset"][name]["sha256"]
        examples = text_examples(records)
        effective = sum(int((y != IGNORE).sum()) for _, y in examples)
        eos = sum(int((y == ByteTokenizer.eos_id).sum()) for _, y in examples)
        assert effective == sum(len(row["text"].encode("utf-8")) + 1 for row in records)
        assert eos == len(records)
        replay[name] = {"records": len(records), "examples": len(examples), "effective_targets": effective,
                        "eos_targets": eos, "maximum_chunk_length": max(len(x) for x, _ in examples),
                        "records_sha256": records_sha256(records)}
        if name != "train":
            assert effective == variants["baseline"]["heldout"][name]["effective_tokens"]
    sampler = random.Random(log["seed"])
    examples = text_examples(split["train"])
    sampled_targets = sum(sum(len(y) for _, y in sampler.choices(examples, k=16)) for _ in range(240))
    assert all(sampled_targets == variant["training"]["effective_tokens"] for variant in variants.values())
    first = variants["rope"]["heldout"]["test"]["samples"][0]
    decoded = ByteTokenizer().decode(first["generated_ids"])
    assert decoded == first["generated"] == "was a loked there was a bough a "
    assert first["prompt"] == "Once upon a time, there "
    tracked = ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py",
               "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py"]
    result["repository_sources"] = {name: {"sha256": sha((ROOT / name).read_bytes()),
                                          "matches_experiment_code": sha((ROOT / name).read_bytes()) == log["code_sha256"][name]}
                                    for name in tracked}
    historical, historical_receipt = fetch(
        "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + log["revision"]
        + "/scripts/course_experiments/architecture.py"
    )
    assert sha(historical) == log["code_sha256"]["scripts/course_experiments/architecture.py"]

    def functions(raw):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    old, current = functions(historical), functions((ROOT / "scripts/course_experiments/architecture.py").read_bytes())
    related = ["_text_dataset", "_train", "_heldout", "run_modern", "_clone_config", "_copy_matching",
               "_nll", "_forward", "_steps", "_sync", "_description", "_runtime", "_data_report", "_amp"]
    comparisons = {name: old[name] == current[name] for name in related}
    assert all(comparisons.values())
    result["historical_experiment_code"] = {"receipt": historical_receipt, "ast_equal_functions": comparisons,
                                             "inspection_note": "全檔版本不同，但與現代零件比較有關的函式 AST 全部相同。"}
    assert all(value["matches_experiment_code"] for name, value in result["repository_sources"].items()
               if name != "scripts/course_experiments/architecture.py")
    result["experiment_record_audit"] = {
        "path": log_path.relative_to(ROOT).as_posix(), "sha256": sha(log_path.read_bytes()),
        "recorded_environment": {key: log[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "status")},
        "variants": summary, "archive_sha256": archive_sha, "archive_receipt": archive_receipt, "dataset_replay": replay,
        "replayed_training_effective_targets": sampled_targets,
        "rope_first_test_sample": first, "sample_ids_decode": decoded,
        "tokenizer": {"vocab_size": ByteTokenizer.vocab_size, "special_count": len(SPECIALS), "specials": list(SPECIALS),
                      "english_bytes": len("A".encode("utf-8")), "chinese_bytes": len("貓".encode("utf-8"))},
        "nll_interpretation": "NLL=(Σ -ln p(目標|前文))/有效目標數；降低平均 NLL 等於提高這批目標的幾何平均預測機率，不保證每個位置都改善。",
        "scope": "本回合重新計數與核對保存紀錄；沒有重跑 GPU 訓練、重新載入檢查點或量測長文章外推。",
    }
    ns = {"s": "http://www.w3.org/2000/svg"}
    figures = {}
    for filename in ("rewrite-14-rotation-components.svg", "rewrite-14-shared-rotation.svg"):
        path = ROOT / "course/figures" / filename
        root = ET.fromstring(path.read_bytes())
        arrows = []
        for line in root.findall("s:line", ns):
            if line.get("marker-end") and line.get("stroke") in ("#245ba8", "#107b6e", "#ac5016"):
                x1, y1, x2, y2 = (float(line.get(key)) for key in ("x1", "y1", "x2", "y2"))
                dx, dy = x2 - x1, y1 - y2
                arrows.append({"stroke": line.get("stroke"), "start": [x1, y1], "end": [x2, y2],
                               "length": math.hypot(dx, dy), "math_angle_degrees": math.degrees(math.atan2(dy, dx)) % 360})
        figures[filename] = {"sha256": sha(path.read_bytes()), "arrows": arrows}
    components = figures["rewrite-14-rotation-components.svg"]["arrows"]
    assert all(abs(components[index]["length"] - 93) < 1e-10 for index in (0, 3))
    shared = figures["rewrite-14-shared-rotation.svg"]["arrows"]
    assert all(abs(item["length"] - 80) < 1e-10 for item in shared)
    assert [round(item["math_angle_degrees"]) for item in shared] == [90, 225, 180, 315]
    receipts = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
    render_records = [item for item in receipts["records"] if item.get("page_id") == "14.1"]
    for item in render_records:
        assert sha((ROOT / item["png"]["path"]).read_bytes()) == item["png"]["sha256"]
    result["figure_geometry"] = figures
    result["render_receipts"] = render_records
    result["original_sources"] = source_receipts()
    result["result"] = "所有斷言通過；保存紀錄的數字、算例與函式接線均相符。"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
