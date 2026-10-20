"""核對引用文件差異與模態評估說明；不訓練或載入權重。"""
import ast
import hashlib
import inspect
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts import train, evaluate
from scripts.course_experiments import modalities
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.multimodal import MultiModalLM

AD = ROOT / "docs/technical-reviews/artifacts/review-20261019-7-15"
PREVIOUS = "docs/course-review-20261019/reviews/rv1019_technical_7-15/technical-7-15-current-01.json"
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_7-15/technical-7-15-original-01.json"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def load(path):
    return json.loads(Path(path).read_bytes())

def git(*args):
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {"command_argv": ["git", *args], "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

previous = load(ROOT / PREVIOUS)
assert sha((ROOT / PREVIOUS).read_bytes()) == "8a9537e8bcc728230e22b6612372f49ced6d61358b8c297d4f0dde557a27d5a1"
assert sha((ROOT / ORIGINAL).read_bytes()) == "247a175afb9d57268bbbe7eb5b19b01e9f2cd32d6fe75d14161e9895e4efbd23"
assert previous["issues"] == []
source_inventory = []
for item in previous["sources"]:
    if item["kind"] != "repository_code":
        continue
    current = sha((ROOT / item["path"]).read_bytes())
    changed = current != item["sha256"]
    assert (item["path"] == "course/training.md") == changed, "引用來源的變動範圍須相符"
    source_inventory.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": current, "changed": changed})
preserved = []
for item in previous["artifacts"]:
    current = sha((ROOT / item["path"]).read_bytes())
    assert current == item["sha256"], "既有證據不得變動"
    preserved.append({"path": item["path"], "sha256": current})
old = subprocess.check_output(["git", "show", "ecf6dec71280:course/training.md"], cwd=ROOT)
new = (ROOT / "course/training.md").read_bytes()
pinned_new = subprocess.check_output(["git", "show", "c6b9339d2595:course/training.md"], cwd=ROOT)
assert sha(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert sha(new) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
assert new == pinned_new

def recipe(raw):
    start = raw.index(b'<details id="fixed-sft-recipe"')
    end = raw.index(b'</details>', start) + len(b'</details>')
    return raw[start:end]
assert recipe(old) == recipe(new), "引用的固定對話配方須逐位元比較"
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert sha((ROOT / freeze_path).read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
page = next(item for item in load(ROOT / freeze_path)["pages"] if item["page_id"] == "7.15")
frozen = (ROOT / page["snapshot"]).read_bytes()
raw = (ROOT / "course/chapters/07.md").read_bytes()
start = re.search(rb"(?m)^## 7\.15 ", raw).start()
match = re.search(rb"(?m)^## ", raw[start + 1:])
end = start + 1 + match.start() if match else len(raw)
assert raw[start:end] == frozen
assert sha(frozen) == previous["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == {}
old_audit = load(AD / "audit-output-current-01.json")
new_audit = load(AD / "audit-output-current-02.json")
fields = ["section_stdout", "exercise", "data_regeneration", "arithmetic_population", "evaluations", "baseline_evaluation_identical", "baseline_checkpoint", "training", "digit_with_eos_example", "record_environment", "cli_help", "cli_list_assets_stdout"]
assert new_audit["checks_passed"] is True
for field in fields:
    assert old_audit[field] == new_audit[field], "本回合重算結果須與先前一致"

class InputCaptured(Exception):
    pass

modal_inputs = []
class CaptureModel:
    def __call__(self, ids, labels, image=None, waveform=None):
        modal_inputs.append({"task": task, "input_tokens": len(ids), "supervised_tokens": int((labels != -100).sum()), "image_shape": list(image.shape) if image is not None else None, "waveform_shape": list(waveform.shape) if waveform is not None else None})
        raise InputCaptured

tok = ByteTokenizer()
modal_routes = []
for task in ("vision", "audio", "joint"):
    args = train.parser().parse_args(["--task", task, "--train"])
    assert args.data is None and train.prepare_examples(args, tok) is None
    rng = random.Random(42)
    with patch.object(train.random, "choice", side_effect=rng.choice) as choices:
        for _ in range(2):
            try:
                train.modal_loss(CaptureModel(), task, tok, "cpu")
            except InputCaptured:
                pass
        assert choices.call_count == 6
    modal_routes.append({"task": task, "data_argument": args.data, "prepare_examples": None, "modal_loss_calls": 2, "random_choice_calls": choices.call_count, "forward_or_training_executed": False})
main_tree = ast.parse(inspect.getsource(train.main))
call_names = sorted({node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(main_tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))})
assert not any(name.startswith("evaluate") or name.startswith("validate") for name in call_names)

fake_multimodal = MultiModalLM.__new__(MultiModalLM)
torch.nn.Module.__init__(fake_multimodal)
rejection = None
with patch.object(sys, "argv", ["evaluate.py", "unused.pt", "--device", "cpu"]), patch.object(evaluate, "load_checkpoint", return_value=(fake_multimodal, {})):
    try:
        evaluate.main()
    except ValueError as exc:
        rejection = str(exc)
assert rejection == "此入口只評估 TinyLM 文字 checkpoint；多模態模型請使用對應評估入口"

vqa_path = "docs/course-experiments/results/vqa.json"
vqa = load(ROOT / vqa_path)
vqa_comparisons = []
reference = None
for name in ("projector_only", "partial", "all", "all_replay"):
    variant = vqa["results"]["variants"][name]
    before = variant["text_before"]
    after = variant["text_after"]
    before_pairs = [(row["messages"], row["expected"]) for row in before["samples"]]
    after_pairs = [(row["messages"], row["expected"]) for row in after["samples"]]
    assert before_pairs == after_pairs
    assert len(before_pairs) == before["records"] == after["records"] == 10
    assert variant["text_probe_is_training_diagnostic"] is False
    if reference is None:
        reference = before_pairs
    assert reference == before_pairs
    vqa_comparisons.append({"variant": name, "same_questions_and_targets_before_after": True, "test_records": len(before_pairs), "question_target_sha256": sha(json.dumps(before_pairs, ensure_ascii=False, sort_keys=True).encode())})
run_vqa_source = inspect.getsource(modalities.run_vqa)
assert 'text_holdout = text_splits["test"]' in run_vqa_source
assert run_vqa_source.count('evaluate_lm(model.language, text_holdout, mode="sft")') == 2
new_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py", vqa_path]
extra_sources = [{"path": path, "sha256": sha((ROOT / path).read_bytes())} for path in new_paths]
source_fragments = {"modal_loss": inspect.getsource(train.modal_loss), "run_vqa_text_comparison": run_vqa_source[:run_vqa_source.index('    seed_everything(ctx.seed)\n    direct =')]}
changed_sentences = [
    {"statement": "三個模態 CLI 的例子逐步隨機產生，沒有另設固定驗證題，三題推論只是抽查。", "status": "verified", "evidence": "train.prepare_examples 的模態任務回傳 None；main 每步呼叫 modal_loss；無資料檔時每次抽 color、shape、音高，且 main 沒有 evaluate/validate 呼叫。實跑輸入攔截探針，未訓練模型。", "scope": "限定文件中的三條未指定 --data 的 train.py 命令；不宣稱帶入自訂資料時也隨機生成合成素材。"},
    {"statement": "evaluate.py 不接受多模態 checkpoint，所以所列 CLI 命令沒有測量文字能力是否退步。", "status": "verified", "evidence": "evaluate.main 的 TinyLM 型別檢查在讀資料前拒絕 MultiModalLM；本回合以不含權重的 MultiModalLM 物件攔截載入，實際收到預期 ValueError；文件三條 infer_modal 僅問模態題。", "scope": "型別檢查與命令覆蓋範圍；沒有使用原 checkpoint 重現生成。"},
    {"statement": "文字能力前後比較應使用同一批文字測試題；固定 vqa 實驗做了這個對照。", "status": "verified", "evidence": "run_vqa 在更新前後都以 model.language 與同一個 text_holdout 評估，text_holdout 取自 sft 的 test；四個續訓支線的保存 text_before/text_after 逐題問題與目標相同，每支十題。", "scope": "同題比較原則沿用原報告的方法推導；只證明有保存這個對照，沒有從中外推一般能力或重新測量品質。"},
]
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-15/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "checks_passed": True,
    "previous_report": {"path": PREVIOUS, "sha256": sha((ROOT / PREVIOUS).read_bytes())},
    "original_report": {"path": ORIGINAL, "sha256": sha((ROOT / ORIGINAL).read_bytes())},
    "previous_issues": [], "resolved_issue_ids": [],
    "source_inventory": source_inventory, "preserved_artifacts": preserved,
    "diff": git("diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"),
    "fixed_recipe": {"identical": True, "previous_sha256": sha(recipe(old)), "current_sha256": sha(recipe(new)), "locator": 'details id="fixed-sft-recipe"'},
    "current_page": {"matches_frozen": True, "source_sha256": sha(frozen), "figure_sha256": {}},
    "recomputed_result_fields_identical": fields,
    "all_claims_retained": [{"claim_id": item["id"], "kind": item["kind"], "status": "verified"} for item in previous["claims"]],
    "modal_routes": modal_routes, "modal_inputs": modal_inputs,
    "train_main_evaluation_calls": [],
    "multimodal_rejection_probe": {"checkpoint_loading_mocked": True, "weights_loaded": False, "exception_type": "ValueError", "message": rejection},
    "vqa_same_text_holdout_comparisons": vqa_comparisons,
    "extra_sources": extra_sources, "source_fragments": source_fragments,
    "changed_source_sentences": changed_sentences,
    "limitations": "未訓練或載入原權重，未重跑 GPU 生成、loss、計時或固定 VQA 實驗。模態訓練探針在 forward 前停止；評估型別探針替代 checkpoint 載入。VQA 只核對現行程式與保存的題目及目標。",
}, ensure_ascii=False, indent=2))
