"""以既有訓練示範與固定 token stub 核對分項評分及停止原因記錄；不跑神經模型。"""
import datetime as dt
import hashlib
import importlib.util
import json
import platform
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

spec = importlib.util.spec_from_file_location("evaluation_19_5", ROOT / "scripts/selftrained/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
with tarfile.open(ROOT / "assets/training/selftrained-v2.tar.gz") as archive:
    rows = [json.loads(line) for line in archive.extractfile("text-tools-train.jsonl").read().decode().splitlines()]
examples = []
for question in ("App打不開，該怎麼辦？", "那個問題要怎麼處理？", "請預測明天天氣。"):
    row = next(r for r in rows if r["messages"][-2]["content"] == question and r["supervision"].get("format") == "one_sentence")
    correct = row["messages"][-1]["content"]
    correct_score = evaluate.score_reply(row, correct, {"tool_call": None, "status": "no_tool"})
    wrong_score = evaluate.score_reply(row, "不知道。", {"tool_call": None, "status": "no_tool"})
    assert correct_score["semantic"] and correct_score["format"]
    assert not wrong_score["semantic"] and wrong_score["format"]
    examples.append({"id": row["id"], "question": question, "correct_score": correct_score, "wrong_content_same_format_score": wrong_score})
messages = [{"role": "user", "content": "問"}]
tok = CharacterTokenizer.build(["問答"])
encoder = RecordEncoder(tok, ".")


class FixedTokenStub:
    def __init__(self, tokens):
        self.tokens = tokens

    def generate(self, *_args, **_kwargs):
        return torch.tensor([self.tokens], dtype=torch.long)


record = {"id": "termination-check", "task": "text", "messages": messages}
stops = []
for tokens, reason in ((tok.encode("答") + [tok.eos_id], "eos"), (tok.encode("答") * 2, "max_new_tokens")):
    generator = evaluate.Generator(FixedTokenStub(tokens), encoder, 2)
    generator.generate(record, messages)
    assert generator.calls[0]["stop_reason"] == reason
    stops.append({"tokens": tokens, "observed_stop_reason": generator.calls[0]["stop_reason"]})
result = {
    "created_at": dt.datetime.now(dt.UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_19_5",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_19_5-v1/verify_evaluation.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scoring_checks": examples,
    "stop_recording_checks": stops,
    "scope": "三個已存示範分別檢查內容與格式；兩個固定 token stub 只檢查停止原因記錄。沒有推論模型、權重、訓練或重估品質。",
    "result": "all assertions passed",
}
path = Path(__file__).with_name("evaluation-check-v1.json")
with path.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"result": result["result"], "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))
