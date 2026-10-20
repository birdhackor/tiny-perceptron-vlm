"""重算教學算例與保存紀錄，不訓練模型，也不把停止率當成答案正確率。"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import generate
from tiny_perceptron.tokenization import generation_report


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(name, value):
    with (DIRECTORY / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class ScriptedModel(torch.nn.Module):
    """用固定候選隔離 EOS 與輸出上限的控制流程。"""

    def __init__(self, candidates, maximum):
        super().__init__()
        self.candidates = candidates
        self.config = SimpleNamespace(max_length=maximum)
        self.calls = 0

    def forward(self, ids, cache=None):
        logits = torch.zeros((1, ids.shape[1], 264))
        logits[0, -1, self.candidates[self.calls]] = 1
        self.calls += 1
        return {"logits": logits, "cache": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/A.7.md"
    body = frozen.read_bytes()
    expected_sha = "7b529aabd96833337f46217d1162340d57e582ff69c4768667facea43607f6cb"
    assert sha(body) == expected_sha, "凍結原文指紋不符"
    whole = (ROOT / "course/chapters/0A.md").read_bytes()
    current = re.search(rb"(?ms)^## A\.7 .*?(?=^## |\Z)", whole)[0]
    assert current == body, "目前小節與凍結原文不同"
    manifest = ROOT / "docs/course-review-20261019/freeze-04.json"
    assert sha(manifest.read_bytes()) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單指紋不符"
    page = next(p for p in json.loads(manifest.read_bytes())["pages"] if p["page_id"] == "A.7")
    assert page["source_sha256"] == expected_sha and page["figures_sha256"] == {}, "頁面指紋或圖清單不符"

    files = {}
    for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
        raw = (args.section_run / name).read_bytes()
        files[name] = {"sha256": sha(raw), "content": raw.decode("utf-8")}
    extraction = json.loads(files["extraction.json"]["content"])
    execution = json.loads(files["execution.json"]["content"])
    assert extraction["source_sha256"] == expected_sha, "執行的原文版本不符"
    assert execution["exit_code"] == 0, "小節執行失敗"
    assert files["stderr.txt"]["content"] == "", "小節執行產生錯誤輸出"
    expected_stdout = "P 75 G 25 P+G 100 放得下 True\n歷史變長後 P 105 P+G 130 超出 30\n只看 P 也超出 5\n"
    assert files["stdout.txt"]["content"] == expected_stdout, "小節算例輸出不符"
    write_new("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0A.md#A.7" --execute',
        "source_sha256": expected_sha,
        "result": "exit_code=0；stdout 與凍結文字的整數算例完全相同",
        "copied_tool_outputs": files,
    })
    code = re.search(r"```python\n(.*?)```", body.decode(), re.S)[1]
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code + '\nparts["公告"] = 10\nprint(sum(parts.values()) + G)\n', str(frozen), "exec"), namespace)
    assert output.getvalue() == expected_stdout + "100\n", "練習算例輸出不符"
    assert namespace["parts"] == {"歷史": 50, "問題與格式": 15, "公告": 10}, "練習輸入狀態不符"

    rag_path = ROOT / "docs/course-experiments/results/rag.json"
    rag_raw = rag_path.read_bytes()
    rag = json.loads(rag_raw)
    results = rag["results"]
    maximum = results["model"]["config"]["max_length"]
    assert maximum == 160, "RAG 模型上下文設定不符"
    tokenizer = ByteTokenizer()
    roles = {"user": tokenizer.user_id, "assistant": tokenizer.assistant_id, "system": tokenizer.system_id}
    audited = []
    for group in ["samples", "icl_samples_same_model_weights"]:
        for row in results[group]:
            generation = row["generation"]
            messages = generation["messages"]
            prompt = [tokenizer.bos_id]
            for message in messages:
                prompt += [roles[message["role"]]] + tokenizer.encode(message["content"]) + [tokenizer.eos_id]
            prompt += [tokenizer.assistant_id]
            assert prompt == generation["input_ids"], "完整訊息重編碼與保存 ID 不同"
            assert len(prompt) == generation["input_tokens"], "保存的前文長度不符"
            assert len(messages) == 1 and messages[0]["role"] == "user", "紀錄包含未預期的歷史訊息"
            if group == "samples":
                context = "\n".join(f"[{doc['id']}] {doc['text']}" for doc in row["documents"]) or "EMPTY"
                question = f"Docs:\n{context}\nFind:{row['family']}\nReply:address[source] or UNKNOWN"
                assert messages[0]["content"] == question, "公告內容被截短或改寫"
            else:
                value = int(row["family"].split(":")[1])
                offset = {"zero_shot": None, "few_shot": 2, "changed_examples": 5, "reversed_examples": 2}[row["mode"]]
                pairs = [] if offset is None else [f"0->{offset}", f"1->{(1 + offset) % 10}"]
                if row["mode"] == "reversed_examples":
                    pairs.reverse()
                examples = ";".join(pairs) if pairs else "EMPTY"
                assert messages[0]["content"] == f"Examples:{examples}\nMap:{value}\nReply:digit or UNKNOWN", "示例輸入不符"
            assert generation["max_new_tokens"] == 16, "新 token 上限不符"
            assert len(prompt) + 16 <= maximum, "保留的答案空間超出上下文"
            assert len(generation["samples"]) == generation["candidate_count"] == 1, "生成候選數不符"
            sample = generation["samples"][0]
            ids = sample["generated_ids"]
            assert all(isinstance(token, int) and 0 <= token < tokenizer.vocab_size for token in ids), "生成 ID 不在詞表內"
            assert ids and ids[-1] == tokenizer.eos_id and tokenizer.eos_id not in ids[:-1], "EOS 原始 ID 不符"
            invalid = [token for token in ids[:-1] if token < 8]
            assert invalid == [] and sample["invalid_special_tokens"] == [], "生成混入非法特殊 token"
            assert sample["eos"] is True, "EOS 欄位與原始 ID 不符"
            decoded = tokenizer.decode(ids[:-1])
            assert decoded == sample["generated"], "回答文字與原始 ID 不符"
            assert len(ids) == sample["generated_tokens"] == generation["generated_tokens"] <= 16, "生成長度不符"
            audited.append({"group": group, "family": row["family"], "mode": row["mode"], "P": len(prompt), "G": 16, "P_plus_G": len(prompt) + 16, "generated_tokens": len(ids), "last_id": ids[-1], "invalid_special_ids": invalid, "generated": decoded, "expected": row["expected"], "matches_expected": decoded == row["expected"]})
    evaluation = [row for row in audited if row["group"] == "samples"]
    examples = [row for row in audited if row["group"] != "samples"]
    modes = dict(sorted(Counter(row["mode"] for row in evaluation).items()))
    assert len(evaluation) == 84 and len(modes) == 7 and set(modes.values()) == {12}, "公告評測母體不符"
    assert len(examples) == 4, "示例生成數不符"
    assert (min(row["P"] for row in evaluation), max(row["P"] for row in evaluation)) == (58, 94), "公告前文長度範圍不符"
    assert max(row["P_plus_G"] for row in evaluation) == 110, "公告總預算不符"
    assert sorted(set(row["P"] for row in examples)) == [47, 51], "示例前文長度不符"
    assert any(not row["matches_expected"] for row in audited), "未找到正文聲稱存在的回答錯誤"

    witness = [ord("A") + 8, tokenizer.user_id, ord("B") + 8, tokenizer.eos_id]
    visible = tokenizer.decode(witness)
    diagnostic = generation_report(tokenizer, witness)
    assert visible == "AB" and diagnostic["answer"] == "A<user>B" and diagnostic["invalid_special_tokens"], "特殊 ID 被 decoder 隱去的反例不符"
    candidates = [ord("A") + 8, tokenizer.eos_id, ord("B") + 8]
    eos_model = ScriptedModel(candidates, 10)
    eos_ids = generate(eos_model, torch.tensor([[1, 3]]), max_new_tokens=5)[0, 2:].tolist()
    limited_model = ScriptedModel(candidates, 10)
    limited_ids = generate(limited_model, torch.tensor([[1, 3]]), max_new_tokens=1)[0, 2:].tolist()
    assert eos_ids == candidates[:2] and limited_ids == candidates[:1], "EOS 或新 token 上限的控制流程不符"
    character_example = {"text": "店", "unicode_characters": len("店"), "byte_token_ids": tokenizer.encode("店")}
    assert len(character_example["byte_token_ids"]) == 3, "中文字數與 token 數不同的反例不符"

    path = "scripts/course_experiments/applications.py"
    original = subprocess.run(["git", "show", f"{rag['revision']}:{path}"], cwd=ROOT, capture_output=True, text=True)
    provenance = {"command_argv": ["git", "show", f"{rag['revision']}:{path}"], "exit_code": original.returncode, "stderr": original.stderr, "recorded_code_sha256": rag["code_sha256"][path], "current_code_sha256": sha((ROOT / path).read_bytes()), "scope": "原始訓練來源不能由這個本機 commit 路徑取得；本次重算保存結果並用完整訊息重建 ID，沒有重跑訓練或模型推論。"}
    if original.returncode == 0:
        provenance["original_sha256"] = sha(original.stdout.encode())
        ast.parse(original.stdout)
    summary = {"evaluation_generations": len(evaluation), "modes": modes, "unique_evaluation_families": len({row["family"] for row in evaluation}), "evaluation_P_range": [min(row["P"] for row in evaluation), max(row["P"] for row in evaluation)], "G": 16, "max_evaluation_P_plus_G": max(row["P_plus_G"] for row in evaluation), "icl_generations": len(examples), "icl_P_values": sorted(set(row["P"] for row in examples)), "all_generations": len(audited), "eos_terminal_count": len(audited), "invalid_special_token_generations": 0, "evaluation_text_mismatches": sum(not row["matches_expected"] for row in evaluation), "icl_text_mismatches": sum(not row["matches_expected"] for row in examples)}
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " --section-run " + str(args.section_run), "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()}, "source_sha256": expected_sha, "rag_record": {"path": str(rag_path.relative_to(ROOT)), "sha256": sha(rag_raw), "recorded_environment": {key: rag[key] for key in ["device", "seed", "python_version", "torch_version", "gpu"]}, "max_length": maximum, "training_steps": results["training"]["steps"]}, "arithmetic_and_exercise_stdout": output.getvalue(), "summary": summary, "raw_ID_audit": audited, "decoder_counterexample": {"ids": witness, "ordinary_decode": visible, "diagnostic": diagnostic}, "EOS_and_limit_control_flow": {"candidates": candidates, "G_5": eos_ids, "G_1": limited_ids}, "token_count_counterexample": character_example, "historical_source_provenance": provenance, "result": "全部重算與控制流程斷言通過；實測結論限於已保存的生成紀錄。"}
    write_new("fact-verification.json", result)
    print(json.dumps({"environment": result["environment"], "summary": summary, "result": result["result"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
