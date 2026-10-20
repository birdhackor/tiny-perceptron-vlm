"""重跑導言驗證，並核對章檔變動的路由統計說明。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _routing
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN, MoEFFN

BASE = Path("docs/technical-reviews/artifacts/review-20261019-chapter-15")
ORIGINAL = Path("docs/course-review-20261019/reviews/rv1019_technical_chapter-15/technical-chapter-15-original-01.json")
FREEZE = Path("docs/course-review-20261019/freeze-05.json")
CHAPTER = Path("course/chapters/15.md")
RECORD = Path("docs/course-experiments/results/moe.json")


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    assert digest(ORIGINAL) == "d7877dc187e87c81e111781cef3e1cfaa2f3035f5a686eb0b5d1e5d44f0ed683", "原始報告內容改變"
    report = json.loads((ROOT / ORIGINAL).read_bytes())
    assert report["issues"] == [], "原始報告存在需要逐項處理的問題"
    preserved = {}
    for artifact in report["artifacts"]:
        actual = digest(artifact["path"])
        assert actual == artifact["sha256"], "原始證據內容改變：" + artifact["path"]
        preserved[artifact["path"]] = actual
    preserved[str(ORIGINAL)] = digest(ORIGINAL)
    preserved[str(BASE / "self-check.txt")] = digest(BASE / "self-check.txt")

    freeze = json.loads((ROOT / FREEZE).read_bytes())
    page = next(page for page in freeze["pages"] if page["page_id"] == report["lesson_id"])
    snapshot = (ROOT / page["snapshot"]).read_bytes()
    introduction = (ROOT / CHAPTER).read_bytes().split(b"## ", 1)[0]
    assert snapshot == introduction, "現行導言與凍結原文不同"
    assert hashlib.sha256(snapshot).hexdigest() == report["source_sha256"] == page["source_sha256"], "凍結原文雜湊不符"
    assert page["figures_sha256"] == {}, "凍結頁面有未核對圖片"

    diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", str(CHAPTER)]
    diff = subprocess.check_output(diff_command, cwd=ROOT, text=True)
    before = subprocess.check_output(["git", "show", "5ef56bc756f3:" + str(CHAPTER)], cwd=ROOT)
    after = subprocess.check_output(["git", "show", "3af7f13ef959:" + str(CHAPTER)], cwd=ROOT)
    assert after == (ROOT / CHAPTER).read_bytes(), "目前章檔與指定差異的終點不同"
    assert before.split(b"## ", 1)[0] == introduction, "指定差異改動導言"
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert len(added) == 2, "指定差異的新增段落數不符"
    assert all("各個 MoE 組" in line and "null" in line for line in added), "新增說明不是預期的路由統計範圍"

    probe_path = ROOT / BASE / "probe.py"
    spec = importlib.util.spec_from_file_location("original_probe", probe_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        probe.main()
    probe_output = json.loads(captured.getvalue())
    advertised_recipe = probe_output.pop("command")
    assert probe_output["result"] == "全部檢查通過", "導言原始驗證失敗"

    record = json.loads((ROOT / RECORD).read_bytes())
    variants = []
    for name, variant in record["results"]["variants"].items():
        config = variant["model"]["config"]
        routing = variant["validation_routing"]
        if config["experts"] == 0:
            assert routing is None, "Dense 紀錄含非空的路由統計"
            variants.append({"name": name, "experts": 0, "validation_routing": routing})
        else:
            required = {"dispatch_counts", "dispatch_denominator", "mean_router_probability", "mean_batch_auxiliary"}
            assert len(routing["layers"]) == config["layers"], "路由統計層數不符"
            for layer in routing["layers"]:
                assert required <= layer.keys(), "MoE 紀錄缺少新增說明列出的欄位"
                assert len(layer["dispatch_counts"]) == config["experts"], "expert 次數欄數不符"
                assert sum(layer["dispatch_counts"]) == layer["dispatch_denominator"], "分派分母與次數不符"
                assert layer["dispatch_denominator"] == routing["effective_input_tokens"] * config["top_k"], "分派分母與有效輸入數不符"
            variants.append({"name": name, "experts": config["experts"], "top_k": config["top_k"], "validation_routing": routing})

    torch.manual_seed(732)
    context = SimpleNamespace(device="cpu")
    examples = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([4, 5]), torch.tensor([5, 6])),
    ]
    routing_runs = []
    for experts, top_k in ((0, 2), (4, 1), (4, 2)):
        config = ModelConfig(vocab_size=16, width=4, layers=2, heads=1, max_length=8, experts=experts, top_k=top_k)
        model = TinyLM(config)
        routing = _routing(model, examples, context)
        if not experts:
            assert all(isinstance(block.ffn, DenseFFN) for block in model.blocks), "Dense 模型含 MoE FFN"
            assert not any(isinstance(module, MoEFFN) for module in model.modules()), "Dense 模型含 router 模組"
            assert routing is None, "Dense 的統計函式沒有回傳 None"
        else:
            assert routing["effective_input_tokens"] == 5 and routing["padding_excluded"], "補齊位置未排除"
            assert len(routing["layers"]) == 2, "小模型統計層數不符"
            for layer in routing["layers"]:
                assert layer["dispatch_denominator"] == 5 * top_k, "小模型分派分母不符"
                assert sum(layer["dispatch_counts"]) == 5 * top_k, "小模型分派次數不符"
        routing_runs.append({"experts": experts, "top_k": top_k, "validation_routing": routing})

    inputs = [CHAPTER, FREEZE, Path(page["snapshot"]), RECORD, Path("tiny_perceptron/modern.py"), Path("tiny_perceptron/model.py"), Path("scripts/course_experiments/architecture.py")]
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python " + str(BASE / "recheck-current-01.py"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads()), "seed_original_probe": "731", "seed_routing_probe": "732"},
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "preserved_files_sha256": preserved,
        "input_sha256": {str(path): digest(path) for path in inputs},
        "source_snapshot_matches_current_introduction": True,
        "original_issue_count": len(report["issues"]),
        "diff_command": diff_command,
        "diff_stdout": diff,
        "changed_dependency_paragraphs": added,
        "before_chapter_sha256": hashlib.sha256(before).hexdigest(),
        "after_chapter_sha256": hashlib.sha256(after).hexdigest(),
        "original_probe_recipe": advertised_recipe,
        "original_probe_invoked_function": str(BASE / "probe.py") + ":main",
        "original_probe_output": probe_output,
        "record_variants": variants,
        "routing_function_executions": routing_runs,
        "result": "全部複查通過",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
