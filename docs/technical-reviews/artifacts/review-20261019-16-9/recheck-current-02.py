"""核對現行依據、合成訓練入口與固定文字對照；不啟動訓練。"""

import ast
import contextlib
import hashlib
import importlib
import io
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-9/technical-16-9-current-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def select(raw, name):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h[0].startswith(("## " + name + " ").encode()))
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def run(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return p.stdout, {"command_argv": argv, "exit_code": p.returncode, "stdout": p.stdout.decode(), "stderr": p.stderr.decode()}


def main():
    previous = json.loads(PREVIOUS.read_bytes())
    assert digest(PREVIOUS.read_bytes()) == "a5f56752d48a60b190913d739f9942087db951f04d487d4cb6f51a43b1a900f8"
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-9/recheck-current-02.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
        "previous_report": {"path": str(PREVIOUS.relative_to(ROOT)), "sha256": digest(PREVIOUS.read_bytes())},
        "previous_issues": previous["issues"],
    }
    old_paths = [ROOT / a["path"] for a in previous["artifacts"]]
    old_paths += [PREVIOUS, ROOT / previous["original_report"]["path"], BASE / "self-check-01.json", BASE / "self-check-current-01.json"]
    result["preserved_files"] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in old_paths}
    for a in previous["artifacts"]:
        assert digest((ROOT / a["path"]).read_bytes()) == a["sha256"]
    result["repository_source_checks"] = []
    for s in previous["sources"]:
        if s["kind"] != "repository_code":
            continue
        current = digest((ROOT / s["path"]).read_bytes())
        if s["path"] == "course/training.md":
            assert current == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
        else:
            assert current == s["sha256"]
        result["repository_source_checks"].append({"path": s["path"], "previous_sha256": s["sha256"], "current_sha256": current, "unchanged": current == s["sha256"]})
    inherited = json.loads((BASE / "audit-results.json").read_bytes())
    result["inherited_local_input_checks"] = []
    for path, old_sha in inherited["local_inputs"].items():
        actual = digest((ROOT / path).read_bytes())
        if path != "course/training.md":
            assert actual == old_sha
        result["inherited_local_input_checks"].append({"path": path, "current_sha256": actual, "unchanged": actual == old_sha})
    before, _ = run(["git", "show", "ecf6dec71280:course/training.md"])
    after, _ = run(["git", "show", "c6b9339d2595:course/training.md"])
    current = (ROOT / "course/training.md").read_bytes()
    assert digest(before) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert after == current
    assert select(before, "T.8") == select(current, "T.8")
    _, diff = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
    result["training_change"] = {"previous_sha256": digest(before), "current_sha256": digest(current), "current_equals_diff_target": True, "T8_bytes_unchanged": True, "T8_sha256": digest(select(current, "T.8")), "diff_receipt": diff}
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/16.9.md").read_bytes()
    assert select((ROOT / "course/chapters/16.md").read_bytes(), "16.9") == frozen
    assert digest(frozen) == previous["source_sha256"]
    freeze_path = ROOT / "docs/course-review-20261019/freeze-07.json"
    assert digest(freeze_path.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
    page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == "16.9")
    assert page["source_sha256"] == digest(frozen)
    assert page["figures_sha256"] == previous["figure_sha256"]
    result["freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path.read_bytes()), "page": page}
    render_receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
    result["figure_checks"] = []
    for r in render_receipt["records"]:
        if r["page_id"] != "16.9":
            continue
        assert digest((ROOT / r["figure"]).read_bytes()) == previous["figure_sha256"][r["figure"]]
        assert digest((ROOT / r["png"]["path"]).read_bytes()) == r["png"]["sha256"]
        result["figure_checks"].append({"figure": r["figure"], "svg_sha256": previous["figure_sha256"][r["figure"]], "png": r["png"], "visual_inspection": "本回合由審閱者以view_image重看，箭頭、數字、相同起點與柱尺度均符合正文。"})
    sys.path.insert(0, str(ROOT))
    train = importlib.import_module("scripts.train")
    tok = train.ByteTokenizer()
    lines = [line for line in select(current, "T.6").decode().splitlines() if line.startswith(".venv/bin/python scripts/train.py --task")]
    assert len(lines) == 3
    result["synthetic_training_checks"] = []
    for line in lines:
        args = train.parser().parse_args(shlex.split(line)[2:])
        assert args.task in ("vision", "audio", "joint") and args.data is None
        assert train.prepare_examples(args, tok) is None
        samples = []

        class ProbeModel:
            def __call__(self, ids, labels, image=None, waveform=None):
                samples.append({"label_tail": labels[labels != -100].tolist(), "image_present": image is not None, "waveform_present": waveform is not None})
                return {"logits": torch.zeros(1), "labels": labels, "auxiliary": torch.zeros(())}

        draws = ["red", "circle", False, "blue", "square", True]
        with patch.object(train.random, "choice", side_effect=draws) as choice, patch.object(train, "scene", return_value=torch.ones(1)), patch.object(train, "tone", return_value=torch.ones(1)), patch.object(train, "masked_loss", return_value=torch.zeros(())):
            for _ in range(2):
                train.modal_loss(ProbeModel(), args.task, tok, torch.device("cpu"))
        expected = ["circle", "square"] if args.task == "vision" else ["low", "high"] if args.task == "audio" else ["circle,low", "square,high"]
        assert choice.call_count == 6
        assert [s["label_tail"] for s in samples] == [tok.encode(x) + [tok.eos_id] for x in expected]
        result["synthetic_training_checks"].append({"command_from_current_source": line, "task": args.task, "data_argument": None, "prepare_examples": None, "controlled_choice_calls": choice.call_count, "sample_count": len(samples), "expected_answer_tails": expected, "labels_matched": True, "scope": "只直接呼叫合成loss分支；model、scene/tone與loss被替換成探針，沒有訓練或真實推論。"})
    train_raw = (ROOT / "scripts/train.py").read_text()
    train_tree = ast.parse(train_raw)
    train_main = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = sorted({ast.unparse(n.func) for n in ast.walk(train_main) if isinstance(n, ast.Call)})
    assert not any(name in calls for name in ("evaluate", "evaluate_lm", "evaluate_modal", "split_records"))
    options = sorted(s for action in train.parser()._actions for s in action.option_strings)
    validation_options = [s for s in options if "validation" in s or "holdout" in s or "test" in s]
    assert validation_options == []
    loss_node = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "modal_loss")
    draws_node = next(n for n in ast.walk(loss_node) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and [x.id for x in n.targets[0].elts] == ["color", "shape", "high"])
    result["fixed_validation_inspection"] = {"main_call_identifiers": calls, "validation_cli_options": validation_options, "random_draw_code": ast.get_source_segment(train_raw, draws_node), "conclusion": "在本節三條未傳--data的路線，主迴圈每步每例呼叫modal_loss即時抽樣；完整main讀取中沒有另建固定validation切分、評估呼叫或驗證指標輸出。不是對所有train.py用法或所有模態工具的概括。"}
    evaluate = importlib.import_module("scripts.evaluate")
    model = train.MultiModalLM(train.TinyLM(train.ModelConfig(width=8, heads=1, layers=1)))
    error = None
    stdout = io.StringIO()
    with patch.object(sys, "argv", ["evaluate.py", "unused-modal-checkpoint.pt", "--device", "cpu"]), patch.object(evaluate, "load_checkpoint", return_value=(model, {})), patch.object(evaluate, "evaluate") as evaluation, contextlib.redirect_stdout(stdout):
        try:
            evaluate.main()
        except ValueError as exception:
            error = str(exception)
    assert error == "此入口只評估 TinyLM 文字 checkpoint；多模態模型請使用對應評估入口"
    assert not evaluation.called
    result["multimodal_evaluation_guard"] = {"model_type": type(model).__name__, "loader_replaced_with_mock": True, "error_type": "ValueError", "error": error, "evaluation_called": evaluation.called, "stdout": stdout.getvalue(), "scope": "實際執行main的型別守衛；未載入或寫入checkpoint，未生成答案。"}
    source_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer_modal.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "docs/course-experiments/results/vqa.json", "course/chapters/11.md", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py"]
    result["additional_source_hashes"] = {p: digest((ROOT / p).read_bytes()) for p in source_paths}
    vqa = json.loads((ROOT / "docs/course-experiments/results/vqa.json").read_bytes())
    for p in ("scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py"):
        assert result["additional_source_hashes"][p] == vqa["code_sha256"][p]
    variants = []
    for name in ("projector_only", "partial", "all", "all_replay"):
        v = vqa["results"]["variants"][name]
        pairs = [[{"messages": sample["messages"], "expected": sample["expected"]} for sample in v[stage]["samples"]] for stage in ("text_before", "text_after")]
        assert pairs[0] == pairs[1]
        for stage in ("text_before", "text_after"):
            assert v[stage]["records"] == v[stage]["examples"] == len(pairs[0]) == 10
            assert sum(s["exact"] for s in v[stage]["samples"]) == v[stage]["matches"]
        variants.append({"variant": name, "same_text_test_inputs": True, "test_input_pairs_sha256": digest(json.dumps(pairs[0], ensure_ascii=False, sort_keys=True).encode()), "test_examples": len(pairs[0]), "effective_tokens_before": v["text_before"]["effective_tokens"], "effective_tokens_after": v["text_after"]["effective_tokens"], "matches_before": v["text_before"]["matches"], "matches_after": v["text_after"]["matches"], "diagnostic_only": v["text_probe_is_training_diagnostic"]})
    modal_source = (ROOT / "scripts/course_experiments/modalities.py").read_text()
    assert 'text_holdout = text_splits["test"]' in modal_source
    assert modal_source.count('evaluate_lm(model.language, text_holdout, mode="sft")') == 2
    title = select((ROOT / "course/chapters/11.md").read_bytes(), "11.7").splitlines()[0].decode()
    result["vqa_comparison"] = {"record_path": "docs/course-experiments/results/vqa.json", "record_sha256": result["additional_source_hashes"]["docs/course-experiments/results/vqa.json"], "revision": vqa["revision"], "device": vqa["device"], "seed": vqa["seed"], "gpu": vqa["gpu"], "code_hashes_match_record": True, "dataset_test": vqa["results"]["text_data"]["test"], "variants": variants, "linked_section_title": title, "scope": "只核對保存紀錄及相符程式的前後比較流程，不重跑訓練；direct_vqa不是具有文字前後欄位的四版之一。"}
    _, listing = run([sys.executable, "-B", "-m", "scripts.course_experiments.run", "--list-assets", "flash_probe"])
    result["flash_readonly_entrypoint"] = listing
    result["result"] = "本節與T.8重跑主張仍成立；現行多模態改句由即時合成分支、型別守衛及固定實驗對照支持。"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
