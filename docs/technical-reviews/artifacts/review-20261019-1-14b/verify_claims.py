"""核對凍結接字示例、練習變體、文件切分與小型接字表生成。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.text import _simple_examples, _simple_sample
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.model import generate
from tiny_perceptron.simple import BigramLM, ContextMLP


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (DESTINATION / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_fence(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "frozen-generation-example", "exec"), namespace)
    return namespace, stdout.getvalue()


class FixedAnswerModel(torch.nn.Module):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.config = type("Configuration", (), {"max_length": 32})()
        self.calls = 0

    def forward(self, ids, cache=None):
        self.calls += 1
        logits = torch.zeros((*ids.shape, 5))
        logits[..., self.answer] = 1
        return {"logits": logits, "cache": None}


def main():
    torch.set_num_threads(1)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
    }
    frozen_path = "docs/course-review-20261019/freeze-01/sources/1.14.md"
    frozen = (ROOT / frozen_path).read_bytes()
    assert digest(frozen) == "49084a2a9f6005497469c618fba1151b74fc8cc6852ad8e783aa4fdba0fb7a4d"
    code = re.search(r"```python\n(.*?)```", frozen.decode(), re.S).group(1)
    namespace, stdout = run_fence(code)
    repeated, repeated_stdout = run_fence(code)
    assert namespace["output"] == repeated["output"]
    assert stdout == repeated_stdout
    torch.manual_seed(999)
    torch.rand(100)
    state = torch.get_rng_state().clone()
    _, independent_stdout = run_fence(code)
    assert stdout == independent_stdout
    assert torch.equal(state, torch.get_rng_state())
    changed_seed, changed_stdout = run_fence(code.replace("manual_seed(42)", "manual_seed(7)"))
    shorter, shorter_stdout = run_fence(code.replace("range(12)", "range(3)").replace("len(output) == 13", "len(output) == 4"))
    forced, forced_stdout = run_fence(code.replace("g = torch.Generator()", "p = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4)\ng = torch.Generator()"))
    assert forced["output"] == ["貓"] + ["。"] * 12
    p = namespace["p"]
    assert namespace["chars"] == ["貓", "看", "狗", "。"]
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-7, rtol=0)
    assert abs(p[0, 1].item() - 0.85) < 1e-7
    assert abs(p[0, 0].item() - 0.05) < 1e-7
    assert abs(p[1, 1].item() - 0.05) < 1e-7
    g = torch.Generator().manual_seed(42)
    current = 0
    transitions = []
    for step in range(12):
        answer = torch.multinomial(p[current], 1, generator=g)
        next_id = answer.item()
        assert isinstance(next_id, int)
        transitions.append({"step": step + 1, "row": current, "next_id": next_id, "shape": list(answer.shape), "dtype": str(answer.dtype)})
        current = next_id
    assert ["貓"] + [namespace["chars"][row["next_id"]] for row in transitions] == namespace["output"]
    sampling = {
        "char_to_id": dict(zip(namespace["chars"], range(4), strict=True)),
        "prior_char_to_id": {char: index for index, char in enumerate(sorted(set("貓看狗，狗看貓。")))},
        "probability_rows": p.tolist(),
        "row_sums": p.sum(-1).tolist(),
        "transitions": transitions,
        "stdout_seed_42": stdout,
        "stdout_seed_7": changed_stdout,
        "stdout_three_iterations": shorter_stdout,
        "stdout_forced_period": forced_stdout,
        "lengths": {"seed_42": len(namespace["output"]), "seed_7": len(changed_seed["output"]), "three_iterations": len(shorter["output"])},
        "same_seed_exact_repeat": True,
        "dedicated_generator_ignores_global_rng": True,
        "global_rng_unchanged_by_example": True,
        "loop_variable_value": namespace["_"],
        "untrained_tensor_requires_grad": p.requires_grad,
    }
    documents = toy_documents()
    parts = split_documents(documents, seed=42)
    vocabulary = {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
    archived_path = "docs/course-experiments/results/simple_models.json"
    archived_raw = (ROOT / archived_path).read_bytes()
    archived = json.loads(archived_raw)
    data_hashes = {}
    for name, values in parts.items():
        rows = [{"text": text, "family": text} for text in values]
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        data_hashes[name] = digest(raw)
        assert data_hashes[name] == archived["results"]["data"][name]["sha256"]
    torch.manual_seed(42)
    model = BigramLM(len(vocabulary) + 2)
    batches = {name: _simple_examples(values, vocabulary, context=1) for name, values in parts.items()}
    with torch.no_grad():
        before = {name: float(F.cross_entropy(model(x), y)) for name, (x, y) in batches.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        x, y = batches["train"]
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        after = {name: float(F.cross_entropy(model(x), y)) for name, (x, y) in batches.items()}
        equal_contexts = torch.tensor([[vocabulary["色"], vocabulary["="]], [vocabulary["狀"], vocabulary["="]]])
        equal_logits = model(equal_contexts)
        assert torch.equal(equal_logits[0], equal_logits[1])
    samples = [_simple_sample(model, vocabulary, 1, prompt, "cpu") for prompt in ("顏色=", "顏色=紅；形狀=")]
    archived_bigram = archived["results"]["runs"]["bigram"]
    assert samples == archived_bigram["samples"]
    assert samples[0]["generated"] == "三角。"
    assert after["train"] < before["train"]
    for name in parts:
        assert abs(before[name] - archived_bigram["before_nll"][name]) < 1e-6
        assert abs(after[name] - archived_bigram["after_nll_same_post_update_time"][name]) < 1e-6
    mlp = ContextMLP(len(vocabulary) + 2, context=3, width=16)
    with torch.no_grad():
        mlp_output = mlp(torch.tensor([[vocabulary[c] for c in "顏色="], [vocabulary[c] for c in "形狀="]]))
    eos_model = FixedAnswerModel(2)
    eos_result = generate(eos_model, torch.tensor([[1]]), max_new_tokens=12, eos_id=2)
    assert eos_result.tolist() == [[1, 2]]
    assert eos_model.calls == 1
    assert eos_model.training
    ordinary_model = FixedAnswerModel(3)
    ordinary_result = generate(ordinary_model, torch.tensor([[1]]), max_new_tokens=3, eos_id=2)
    assert ordinary_result.tolist() == [[1, 3, 3, 3]]
    paths = ["tiny_perceptron/simple.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/text.py", "course/training.md", "course/chapters/01.md", "course/chapters/02.md", "course/chapters/06.md", archived_path]
    repository_sources = {path: digest((ROOT / path).read_bytes()) for path in paths}
    evidence = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-14b/verify_claims.py",
        "environment": environment,
        "frozen_source": {"path": frozen_path, "sha256": digest(frozen)},
        "sampling": sampling,
        "experiment": {
            "documents": documents,
            "split_documents": parts,
            "split_counts": {name: len(values) for name, values in parts.items()},
            "data_sha256": data_hashes,
            "vocabulary": vocabulary,
            "seed": 42,
            "steps": 200,
            "learning_rate": 0.01,
            "optimizer": "AdamW，其他設定使用 PyTorch 預設；clip_grad_norm_=1.0",
            "batch": "每步完整訓練集；以字元為單位，0 為邊界兼結束，1 為未知字；每篇補一個結束目標",
            "effective_targets": {name: int(y.numel()) for name, (_, y) in batches.items()},
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "before_nll": before,
            "after_nll": after,
            "samples": samples,
            "archived_bigram": archived_bigram,
            "same_equal_suffix_logits_exactly_equal": torch.equal(equal_logits[0], equal_logits[1]),
            "mlp_three_character_output_shape": list(mlp_output.shape),
            "scope": "一次 CPU／seed=42 的小型接字表重跑；不外推到一般生成品質、跨裝置或跨版本序列。未保存權重。",
        },
        "eos_probe": {"selected_eos": eos_result.tolist(), "calls": eos_model.calls, "ordinary_candidate": ordinary_result.tolist(), "scope": "batch=1、未碰長度上限時核對 generate 的 EOS 分支；固定分數模型只測停止控制流程"},
        "repository_source_sha256": repository_sources,
        "independence": {
            "reviewer_task": "/root/rv1019_technical_1-14b",
            "reviewer_context": "fresh",
            "accidental_exposure": "git status --short 意外列出其他審閱目錄與 artifact 檔名；未讀取其內容，未使用其判斷。",
            "judgment_inputs": "凍結原文、規約、指定工具、相關課文與專案原始碼、實驗結果 JSON、官方原文與本次執行；未讀其他審閱內容。",
        },
        "result": "全部斷言通過",
    }
    tool_directory = ROOT / "outputs/reviewer-tools/runs/20261020T113631708353Z"
    copied = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (tool_directory / name).read_bytes()
        copied[name] = {"sha256": digest(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
    save("section-execution.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/01.md#1.14\" --execute",
        "result": "execution_exit_code=0；抽取原文 SHA-256 與凍結版相同",
        "environment": environment,
        "copies": copied,
    })
    save("verification.json", evidence)
    print(json.dumps({"result": evidence["result"], "sampling": sampling["lengths"], "samples": samples, "before_nll": before["train"], "after_nll": after["train"], "effective_targets": evidence["experiment"]["effective_targets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
