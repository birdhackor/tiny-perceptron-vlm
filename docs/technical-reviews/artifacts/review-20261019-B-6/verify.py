"""核對對話資料、監督遮罩、選卡程式入口與原始來源。"""

import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import applications, run, tool_choice
from scripts.course_experiments.common import Context, new_lm, records_sha256
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import loss_sum, masked_loss


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip:
            self.parts.append(value)


def read_source(url, version, locators, excerpts, note):
    command = ["curl", "-L", "--fail", "--max-time", "30", "-sS", url]
    raw = subprocess.check_output(command)
    parser = Text()
    parser.feed(raw.decode())
    text = " ".join(" ".join(parser.parts).split())
    for excerpt in excerpts:
        assert excerpt in text, "原始來源缺少指定摘錄"
    return {
        "url": url,
        "version": version,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "locators": locators,
        "excerpts": excerpts,
        "inspection_note": note,
        "full_text_saved": False,
    }


def calls(function):
    tree = ast.parse(inspect.getsource(function))
    return sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)})


torch.set_num_threads(1)
tokenizer = ByteTokenizer()
cases = []
for state, answer, size, target_count in [
    ("計算器可用。", "TOOL", 51, 5),
    ("計算器停用。", "ASK", 50, 4),
]:
    messages = [
        {"role": "system", "content": state},
        {"role": "user", "content": "1 加 2 等於多少？"},
        {"role": "assistant", "content": answer},
    ]
    x, y = render_chat(messages, tokenizer)
    target = y[y != IGNORE]
    ids, unshifted_targets = [tokenizer.bos_id], [IGNORE]
    roles = {"system": tokenizer.system_id, "user": tokenizer.user_id, "assistant": tokenizer.assistant_id}
    for message in messages:
        content = tokenizer.encode(message["content"]) + [tokenizer.eos_id]
        ids += [roles[message["role"]]] + content
        unshifted_targets += [IGNORE] + (content if message["role"] == "assistant" else [IGNORE] * len(content))
    assert x.tolist() == ids[:-1], "輸入序列必須只移位一次"
    assert y.tolist() == unshifted_targets[1:], "答案目標必須只移位一次"
    assert len(x) == size and len(target) == target_count, "輸入或目標長度不符"
    assert target.tolist() == tokenizer.encode(answer) + [tokenizer.eos_id], "有效目標應只有答案與結束符號"
    assert tokenizer.decode(target.tolist()) == answer, "解碼應略過結束符號"
    byte_sizes = {m["role"]: len(m["content"].encode("utf-8")) for m in messages}
    calculated = 1 + 3 + 3 + sum(byte_sizes.values()) - 1
    assert calculated == size, "位元組與特殊符號的手算不符"
    logits = torch.zeros((1, len(y), tokenizer.vocab_size), requires_grad=True)
    total, count = loss_sum(logits, y.unsqueeze(0))
    total.backward()
    ignored_gradient_zero = bool((logits.grad[0][y == IGNORE] == 0).all())
    assert ignored_gradient_zero, "被忽略目標不得貢獻 logits 梯度"
    cases.append({
        "messages": messages,
        "input_count": len(x),
        "target_count": len(target),
        "input_dtype": str(x.dtype),
        "target_dtype": str(y.dtype),
        "target_ids": target.tolist(),
        "target_decoded": tokenizer.decode(target.tolist()),
        "last_target_is_eos": bool(target[-1] == tokenizer.eos_id),
        "content_utf8_byte_counts": byte_sizes,
        "length_derivation": f"1 + 3 + 3 + {sum(byte_sizes.values())} - 1 = {calculated}",
        "supervised_input_positions": (y != IGNORE).nonzero().flatten().tolist(),
        "ignored_target_count": int((y == IGNORE).sum()),
        "ignored_logits_gradient_zero": ignored_gradient_zero,
        "effective_loss_denominator": int(count),
    })

records = tool_choice.build_records()
splits = tool_choice.split_records(records)
groups = defaultdict(list)
for record in records:
    messages = record["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"], "三種訊息角色順序不符"
    assert messages[0]["content"] in {"計算器可用。", "計算器停用。"}, "system 應只提供工具狀態"
    assert messages[2]["content"] in tool_choice.ACTIONS, "答案應只包含卡片標籤"
    assert not messages[1]["content"].startswith(("CALC", "COPY")), "問句不應使用任務前綴"
    groups[messages[1]["content"]].append(record)
assert all({r["calculator_available"] for r in rows} == {True, False} for rows in groups.values()), "每題都需要可用與停用的對照"
record_path = ROOT / "docs/course-experiments/results/tool_choice.json"
saved = json.loads(record_path.read_bytes())
result = saved["results"]
for name, rows in splits.items():
    assert records_sha256(rows) == result["data"][name]["sha256"], "重建資料應與保存的資料雜湊相同"
script_hash = hashlib.sha256((ROOT / "scripts/course_experiments/tool_choice.py").read_bytes()).hexdigest()
assert script_hash == result["reproducibility"]["script_sha256"], "實驗程式版本不符保存的版本"
assert result["training"]["checkpoint"] == "tool-choice.pt", "訓練紀錄應記下參數檔名"
assert len(result["frozen_evaluation"]["checkpoint_sha256"]) == 64, "參數檔應有完整雜湊"
record_examples = [s for s in result["test"]["samples"] if s["record"]["messages"][1]["content"] == "1 加 2 等於多少？"]
assert len(record_examples) == 2, "原始生成應保留同題兩種工具狀態"

ctx = Context("cpu", Path(__file__).parent, Path(__file__).parent, ROOT / "assets/training", 42)
model = new_lm(ctx, width=64, layers=2, heads=2, max_length=160)
x, y = render_chat(cases[0]["messages"], tokenizer)
before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
model.train()
before = masked_loss(model(x.unsqueeze(0))["logits"], y.unsqueeze(0))
optimizer.zero_grad(set_to_none=True)
before.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
with torch.no_grad():
    after = masked_loss(model(x.unsqueeze(0))["logits"], y.unsqueeze(0))
changed = sum(not torch.equal(value, before_state[name]) for name, value in model.state_dict().items())
assert changed and after < before.detach(), "此示範的一次更新應改變參數並降低答案的負對數概似"

dispatch = {}


def fake_execute(*args, **kwargs):
    dispatch.update(args=[str(value) for value in args], kwargs=kwargs)
    return {"experiment_id": "tool_choice", "elapsed_seconds": 0, "evidence_status": "complete_run", "artifacts": []}


captured = io.StringIO()
with patch.object(sys, "argv", ["scripts/course_experiments/run.py", "--experiment", "tool_choice", "--device", "cpu"]):
    with patch.object(run, "execute", fake_execute), contextlib.redirect_stdout(captured):
        run.main()
assert dispatch["args"][0:2] == ["tool_choice", "cpu"], "命令列必須選到指定的 CPU 實驗"
spec = run.experiment_spec("tool_choice")
assert spec["module"] == "tool_choice" and spec["function"] == "run", "實驗配方應對應到選卡函式"
choice_calls = calls(tool_choice.run)
assert "new_lm" in choice_calls and not any("load" in name for name in choice_calls), "選卡實驗應重新建立模型且不載入既有參數"
assert "call_tool" not in calls(tool_choice.evaluate), "選卡評估不應執行計算器"

sources = [
    read_source(
        "https://arxiv.org/html/2302.04761v1",
        "arXiv:2302.04761v1，2023-02-09，原作者論文的 arXiv HTML",
        ["Abstract", "Approach / Sampling API Calls", "Approach / Executing API Calls", "Approach / Filtering API Calls", "Approach / Model Finetuning", "Approach / Inference"],
        ["We then execute these API calls", "using a standard language modeling objective."],
        "已讀 Approach 全節：用少量示範提示既有 LM 產生 API 候選，執行候選後以其結果對未來 token 的 loss 改善篩選，再把呼叫與結果插回語料訓練。推論時由外層中斷生成、執行 API、回填結果。這支持工具學習的研究方向，但其自監督資料建立與篩選不同於本專案的人工三卡標註；也不支持將選卡等同自我認知。",
    ),
    read_source(
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
        "PyTorch 2.14 文件；當次執行 torch 2.14.1",
        ["CrossEntropyLoss / class-index target loss formula", "Parameters / ignore_index"],
        ["Specifies a target value that is ignored and does not contribute to the input gradient."],
        "已讀類別索引目標的交叉熵公式與 ignore_index 條件：忽略的是目標位置的 loss 與其 logits 梯度；不會刪去模型輸入的提示文字。交叉熵是正確類別機率的負對數，故降低同一目標的平均 loss 對應提高目標 token 機率的幾何平均，不保證任意題目都改善。",
    ),
]
source_path = Path(__file__).with_name("source-excerpts.json")
with source_path.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")

code_paths = [
    "tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/tool_choice.py",
    "scripts/course_experiments/common.py", "scripts/course_experiments/applications.py",
    "scripts/course_experiments/run.py", "docs/course-experiments/plan.json",
    "docs/course-experiments/results/tool_choice.json",
]
out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-6/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cpu_threads": str(torch.get_num_threads())},
    "cases": cases,
    "data": {
        "records": len(records), "intents": dict(Counter(r["intent"] for r in records)),
        "question_groups": len(groups), "all_questions_have_both_tool_states": True,
        "system_only_tool_state": True, "user_no_task_prefix": True, "assistant_only_action": True,
        "split_hashes": {name: records_sha256(rows) for name, rows in splits.items()},
        "saved_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "saved_training": result["training"], "saved_reproducibility": result["reproducibility"],
        "saved_frozen_evaluation": result["frozen_evaluation"], "saved_example_generations": record_examples,
        "checkpoint_inspection_scope": "核對保存紀錄、參數檔雜湊與儲存程式；未重新開啟本機參數檔，未重跑完整訓練。",
    },
    "single_update": {"seed": 42, "steps": 1, "examples": 1, "supervised_tokens": 5, "loss_before": before.item(), "loss_after": after.item(), "changed_state_tensors": changed, "scope": "只驗證此 CPU 算例中的更新機制，不是品質或泛化測試。"},
    "dispatch": {"captured": dispatch, "output": captured.getvalue(), "spec": spec, "execution_scope": "只執行命令列解析與配方查找；execute 被替換為記錄函式，完整訓練未執行。"},
    "source_inspection": {
        "choice_run_calls": choice_calls, "choice_evaluate_calls": calls(tool_choice.evaluate),
        "json_model_run_calls": calls(applications.run_tools), "new_lm_source": inspect.getsource(new_lm),
        "saving_scope": "run 先保存 dataset.json；fit_lm 保存 tool-choice.pt；evaluate 保存原始生成至 results.json；這是程式與歷史執行紀錄共同支持的儲存行為。",
    },
    "repository_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in code_paths},
    "result": "所有當次斷言通過。",
}
print(json.dumps(out, ensure_ascii=False, indent=2))
