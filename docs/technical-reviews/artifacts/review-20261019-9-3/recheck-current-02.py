"""核對依賴文件修訂、即時模態練習與固定文字能力對照；不訓練或載入權重。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
import scripts.evaluate as evaluate_cli
import scripts.train as train_cli
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def section(raw, identifier):
    return re.search(rb"(?m)^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)", raw, re.S)[0]


def main():
    previous_path = "docs/course-review-20261019/reviews/rv1019_technical_9-3/technical-9-3-current-01.json"
    previous_raw = (ROOT / previous_path).read_bytes()
    assert sha(previous_raw) == "4f23115a9618f3a2af3896630e87e027548ebf1ac64f27e9bc72ccefc6cdeb7d"
    previous = json.loads(previous_raw)
    assert previous["issues"] == []
    source_checks = []
    for source in previous["sources"]:
        if source["kind"] != "repository_code":
            continue
        current_sha = sha((ROOT / source["path"]).read_bytes())
        unchanged = current_sha == source["sha256"]
        assert unchanged or source["path"] == "course/training.md"
        source_checks.append({"source_id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current_sha, "unchanged": unchanged})
    artifact_checks = []
    for artifact in previous["artifacts"]:
        current_sha = sha((ROOT / artifact["path"]).read_bytes())
        assert current_sha == artifact["sha256"]
        artifact_checks.append({"path": artifact["path"], "sha256": current_sha, "unchanged": True})
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.3.md").read_bytes()
    assert section((ROOT / "course/chapters/09.md").read_bytes(), "9.3") == frozen
    freeze_raw = (ROOT / "docs/course-review-20261019/freeze-07.json").read_bytes()
    assert sha(freeze_raw) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
    frozen_page = next(p for p in json.loads(freeze_raw)["pages"] if p["page_id"] == "9.3")
    assert frozen_page["source_sha256"] == sha(frozen) and frozen_page["figures_sha256"] == {}
    old = run(["git", "show", "ecf6dec71280:course/training.md"])["stdout"].encode()
    target = run(["git", "show", "c6b9339d2595:course/training.md"])["stdout"].encode()
    current = (ROOT / "course/training.md").read_bytes()
    assert sha(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert target == current
    assert sha(current) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
    assert section(old, "T.5") == section(current, "T.5")
    diff = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
    t6 = section(current, "T.6").decode()
    immediate = t6.split("<details>", 1)[0]
    commands = [line for block in re.findall(r"```bash\n(.*?)```", immediate, re.S) for line in block.splitlines() if line.strip()]
    assert all("scripts/evaluate.py" not in line for line in commands)
    assert "--experiment vqa --device cuda" in t6
    assert "### 重跑第11章的固定實驗" in t6
    tok = ByteTokenizer()
    modal_checks = []
    for task in ("vision", "audio", "joint"):
        args = train_cli.parser().parse_args(["--task", task])
        assert args.data is None
        captured = {}
        def capture_model(ids, labels, **kwargs):
            captured.update({"answer": tok.decode(labels[labels >= 0].tolist()), "has_image": kwargs["image"] is not None, "has_waveform": kwargs["waveform"] is not None})
            return {"logits": torch.empty(0), "labels": labels, "auxiliary": torch.tensor(0.0)}
        with patch.object(train_cli.random, "choice", side_effect=["blue", "circle", True]) as choices, patch.object(train_cli, "masked_loss", return_value=torch.tensor(0.0)):
            loss = train_cli.modal_loss(capture_model, task, tok, "cpu")
        assert choices.call_count == 3
        expected = {"vision": "circle", "audio": "high", "joint": "circle,high"}[task]
        assert captured["answer"] == expected
        assert captured["has_image"] == (task != "audio")
        assert captured["has_waveform"] == (task != "vision")
        modal_checks.append({"task": task, "default_data": args.data, "random_choice_calls": choices.call_count, "controlled_choices": ["blue", "circle", True], "observed": captured, "loss": float(loss)})
    tree = ast.parse((ROOT / "scripts/train.py").read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
    assert not any("evaluat" in name for name in calls)
    main_text = ast.get_source_segment((ROOT / "scripts/train.py").read_text(), node)
    assert "modal_loss(" in main_text and "random.choice(modal_records) if modal_records else None" in main_text
    model = MultiModalLM(TinyLM(ModelConfig(width=8, heads=1, layers=1)))
    with patch.object(sys, "argv", ["evaluate.py", "unread-checkpoint", "--device", "cpu"]), patch.object(evaluate_cli, "load_checkpoint", return_value=(model, {})) as loader:
        try:
            evaluate_cli.main()
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError("多模態模型應在文字評估入口被拒絕")
    assert "只評估 TinyLM" in rejection and loader.call_count == 1
    vqa_path = "docs/course-experiments/results/vqa.json"
    vqa_raw = (ROOT / vqa_path).read_bytes()
    vqa = json.loads(vqa_raw)
    modal_path = "scripts/course_experiments/modalities.py"
    assert sha((ROOT / modal_path).read_bytes()) == vqa["code_sha256"][modal_path]
    variants = []
    common_identity = None
    for name in ("projector_only", "partial", "all", "all_replay"):
        variant = vqa["results"]["variants"][name]
        before, after = variant["text_before"], variant["text_after"]
        identities = lambda rows: [(sample["messages"], sample["expected"]) for sample in rows]
        identity_before = identities(before["samples"])
        assert identity_before == identities(after["samples"])
        assert len(identity_before) == before["records"] == after["records"] == vqa["results"]["text_data"]["test"]["count"]
        identity_sha = sha(json.dumps(identity_before, ensure_ascii=False, sort_keys=True).encode())
        if common_identity is None:
            common_identity = identity_sha
        assert identity_sha == common_identity
        variants.append({"name": name, "path": "results.variants." + name, "records": before["records"], "before_matches": before["matches"], "after_matches": after["matches"], "before_effective_tokens": before["effective_tokens"], "after_effective_tokens": after["effective_tokens"], "same_prompt_and_target_order": True, "identity_sha256": identity_sha})
    cli = run([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
    run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T152652852851Z"
    files = {}
    for name in ("execution.json", "extraction.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (run_dir / name).read_bytes()
        files[name] = {"sha256": sha(raw), "content": raw.decode()}
    assert json.loads(files["execution.json"]["content"])["exit_code"] == 0
    assert files["stdout.txt"]["content"].splitlines()[-1] == "可核對事實 4" and not files["stderr.txt"]["content"]
    additional_paths = [modal_path, vqa_path, "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py"]
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-3/recheck-current-02.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "previous_report": {"path": previous_path, "sha256": sha(previous_raw)},
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "original_issues": [],
        "source_sha256": sha(frozen),
        "training_diff": diff,
        "cited_section_unchanged": {"selector": "T.5", "sha256": sha(section(current, "T.5")), "unchanged": True},
        "immediate_modal_commands": commands,
        "modal_loss_checks": modal_checks,
        "training_main_calls": calls,
        "evaluation_rejection": {"method": "mock checkpoint loader returning an in-memory MultiModalLM; no checkpoint read or written", "error_type": "ValueError", "message": rejection},
        "vqa_record_check": {"path": vqa_path, "sha256": sha(vqa_raw), "modalities_source_matches_recorded_sha256": True, "recorded_environment": {k: vqa[k] for k in ("revision", "device", "seed", "python_version", "torch_version", "evidence_status")}, "text_data": vqa["results"]["text_data"], "variants": variants},
        "additional_repository_sources": [{"path": path, "sha256": sha((ROOT / path).read_bytes())} for path in additional_paths],
        "cli_help": cli,
        "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/09.md#9.3\" --execute", "original_run_directory": str(run_dir.relative_to(ROOT)), "files": files},
        "scope": "只實跑小節算例與受控函式分支；舊實驗核對原始JSON與程式雜湊，不重訓或重新生成。即時模態分支以受控亂數及假模型核對輸入／目標，文字評估入口以mock loader核對型別拒絕。",
        "assertions": "全部通過",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
