"""核對凍結例子、練習、規則與逐 token 生成介面，保存可重查的執行結果。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.tool_choice import ACTIONS, _expected_action, build_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(code):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-example", "exec"), namespace)
    return namespace, output.getvalue()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def inspect_paper():
    url = "https://arxiv.org/html/1706.03762v7"
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--write-out", "\n%{http_code}\t%{url_effective}", url]
    response = subprocess.check_output(command)
    body, metadata = response.rsplit(b"\n", 1)
    status, effective = metadata.decode().split("\t", 1)
    parser = Text()
    parser.feed(body.decode())
    text = " ".join(" ".join(parser.parts).split())
    excerpts = [
        "consuming the previously generated symbols as additional input when generating the next.",
        "predicted next-token probabilities.",
    ]
    for excerpt in excerpts:
        assert excerpt in text, "原文應包含實際核對的短摘錄"
    assert "1706.03762v7" in text, "原文應標示指定的論文版本"
    return {
        "command_argv": command,
        "http_status": status,
        "effective_url": effective,
        "accessed_at": datetime.now(UTC).isoformat(),
        "version": "arXiv:1706.03762v7, 2023-08-02",
        "response_sha256": sha(body),
        "locator": ["Model Architecture", "Embeddings and Softmax"],
        "excerpts": excerpts,
        "inspection": "已讀生成架構與輸出投影段落：解碼逐步依賴已生成符號，輸出層提供下一 token 機率。這支持以文字標記作為生成目標的機制，不證明本專案模型的泛化或完成整個工具流程。第三方全文只在記憶體處理，未存入 repo。",
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source_path = "docs/course-review-20261019/freeze-01/sources/B.5.md"
    raw = (ROOT / source_path).read_bytes()
    assert sha(raw) == "6846dca1eecfae5cf30a31647bde7dc1bee13f94f1efd624c3e436ae16c3bbfa", "凍結原文應與指定版本相同"
    code = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    assert len(code) == 1, "凍結原文應有一段 Python 程式"
    code = code[0]
    original_run = ROOT / "outputs/reviewer-tools/runs/20261020T073405111275Z"
    copied = {}
    for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        data = (original_run / name).read_bytes()
        copied[name] = {"sha256": sha(data), "text": data.decode()}
    assert copied["fence-1.py"]["text"] == code, "原始工具應執行相同的凍結程式"
    copied_receipt = {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0B.md#B.5" --execute',
        "original_run": str(original_run.relative_to(ROOT)),
        "files_copied_verbatim": copied,
    }
    destination = ARTIFACTS / "section-execution.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(copied_receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    baseline, baseline_output = execute(code)
    assert baseline_output == copied["stdout.txt"]["text"], "獨立執行與原始工具的輸出應相同"
    assert baseline_output.count("符合規則") == 5, "原始五筆應全部符合規則"
    first_change = code.replace('(\"1加2等於多少？\", \"精確計算\", True, \"TOOL\")', '(\"1加2等於多少？\", \"精確計算\", False, \"TOOL\")', 1)
    second_change = first_change.replace('(\"1加2等於多少？\", \"精確計算\", False, \"TOOL\")', '(\"1加2等於多少？\", \"精確計算\", False, \"ASK\")', 1)
    changed, changed_output = execute(first_change)
    repaired, repaired_output = execute(second_change)
    assert changed["examples"][0][3] == "TOOL", "變更工具狀態不應自動改寫人工標註"
    assert changed_output.splitlines()[0].endswith("TOOL 與規則不符"), "停用後仍標註 TOOL 應顯示不符"
    assert changed_output.count("與規則不符") == 1, "只應有第一筆不符"
    assert repaired_output.count("符合規則") == 5, "修正人工標註後應全部符合"

    policy = []
    mapping = {"精確計算": "numerical_addition", "解釋": "explanation", "照抄": "copy", "缺資訊": "missing_quantity"}
    for kind, intent in mapping.items():
        for available in (True, False):
            observed = baseline["rule"](kind, available)
            implementation = _expected_action(intent, available)
            assert observed == implementation, "課文規則與實驗規則應一致"
            policy.append({"kind": kind, "calculator": available, "rule": observed, "experiment_rule": implementation})
    tree = ast.parse(code)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    calls = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert not calls, "選卡規則本身應只返回文字，不呼叫工具"
    records = build_records()
    assert {row["messages"][-1]["content"] for row in records} == set(ACTIONS), "實驗目標應只包含三種文字標記"
    reasons = sorted({row["ask_reason"] for row in records if row["expected_action"] == "ASK"})
    assert reasons == ["calculator_unavailable", "missing_quantity"], "兩種 ASK 應對應不同原因"

    tokenizer = ByteTokenizer()
    model = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=64))
    token_checks = []
    for label in ("TOOL", "DIRECT", "ASK"):
        ids = tokenizer.encode(label)
        assert tokenizer.decode(ids) == label, "固定標記應能以現有 tokenizer 還原"
        inputs, labels = render_chat([{"role": "user", "content": "選卡"}, {"role": "assistant", "content": label}], tokenizer)
        target_ids = labels[labels != -100].tolist()
        assert target_ids == ids + [tokenizer.eos_id], "標記應使用逐 token 目標與 EOS"
        model.zero_grad(set_to_none=True)
        loss = masked_loss(model(inputs.unsqueeze(0))["logits"], labels.unsqueeze(0))
        loss.backward()
        assert torch.isfinite(loss), "標記的下一 token 代價應為有限值"
        assert model.output.weight.grad.abs().sum().item() > 0, "標記應透過既有輸出層接受監督"
        token_checks.append({"label": label, "encoded_ids": ids, "target_ids": target_ids, "loss": loss.item(), "output_gradient_nonzero": True})
    lengths = []
    handle = model.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    prefix = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
    generated = generate(model, prefix, max_new_tokens=4, eos_id=-1)
    handle.remove()
    assert lengths == [1, 2, 3, 4] and generated.shape[1] == 5, "生成器應每次加入一個 token"

    files = ("tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/tool_choice.py", "tiny_perceptron/retrieval.py", "docs/review-tools/section_facts.py")
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-5/verify.py",
        "exit_result": "所有斷言通過",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads())},
        "source_path": source_path,
        "source_sha256": sha(raw),
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "repository_code_sha256": {name: sha((ROOT / name).read_bytes()) for name in files},
        "baseline": {"rows": baseline["examples"], "stdout": baseline_output, "unpacked_final_row": [baseline[name] for name in ("question", "kind", "calculator", "action")]},
        "first_exercise": {"first_row": changed["examples"][0], "rule_action": changed["rule"](*changed["examples"][0][1:3]), "stdout": changed_output},
        "second_exercise": {"first_row": repaired["examples"][0], "stdout": repaired_output},
        "policy_truth_table": policy,
        "rule_function_calls": calls,
        "ask_reasons": reasons,
        "assistant_target_labels": sorted(ACTIONS),
        "label_training_interface": token_checks,
        "generation": {"input_lengths": lengths, "output_ids": generated.tolist(), "model_class": type(model).__name__, "note": "僅核對生成介面與一次反向傳播；未更新參數，未衡量選卡準確率。"},
        "original_paper": inspect_paper(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
