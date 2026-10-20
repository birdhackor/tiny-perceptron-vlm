"""核對凍結示範、標籤、固定資料包與原始來源；不更新模型參數。"""

import copy
import hashlib
import inspect
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from scripts.selftrained.train import set_trainable, stage_records
from scripts.selftrained.evaluate import score_reply
from tiny_perceptron.model import loss_sum
from tiny_perceptron.selftrained.dataset import RecordEncoder, train_tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS, SPECIAL_NAMES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (BASE / name).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "55", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, {"url": url, "command_argv": command, "returncode": result.returncode,
                           "bytes": len(result.stdout), "sha256": sha(result.stdout),
                           "accessed_at": datetime.now(UTC).isoformat()}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


torch.set_num_threads(1)
torch.manual_seed(42)
environment = {"python": sys.version, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "device": "cpu",
               "platform": platform.platform()}
out = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-5/verify.py",
       "environment": environment}
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.5.md").read_text()
blocks = re.findall(r"```python\n(.*?)```", frozen, re.S)
namespace = {}
for block in blocks:
    exec(compile(block, "frozen_section", "exec"), namespace)
tok, row = namespace["tok"], namespace["row"]
encoded = RecordEncoder(tok, ".").encode(row)
out["demo"] = {"vocab_size": tok.vocab_size, "ordinary_characters": len(tok.characters),
               "characters": tok.characters, "required_ocr_count": len(set(OCR_CHARACTERS)),
               "special_ids": {name: getattr(tok, name + "_id") for name in SPECIAL_NAMES},
               "input_ids": encoded["input_ids"], "labels": encoded["labels"],
               "assistant_target": tok.decode([x for x in encoded["labels"] if x != -100], False),
               "pure_text_modalities": encoded["modalities"]}
assert tok.vocab_size == 57 and len(tok.characters) == 46
assert out["demo"]["assistant_target"] == row["messages"][1]["content"] + "<eos>"
assert encoded["input_ids"] == namespace["ours"]["input_ids"]
assert encoded["labels"] == namespace["ours"]["labels"]
both = copy.deepcopy(row)
both["messages"][0]["role"] = "assistant"
both_enc = RecordEncoder(tok, ".").encode(both)
out["role_exercise"] = tok.decode([x for x in both_enc["labels"] if x != -100], False)
assert out["role_exercise"] == "那個問題要怎麼處理？<eos>請說明是地址、App還是卡片的問題。<eos>"
enc = RecordEncoder(tok, ".")
pretrain = enc.encode(row, pretrain=True)
generation = enc.encode(row, generation=True)
explicit = enc.encode(row, generation=True, messages=row["messages"])
assert all(x != -100 for x in pretrain["labels"][:-1]) and pretrain["labels"][-1] == -100
assert generation["input_ids"][-1] == tok.assistant_id
assert generation["input_ids"] == enc.encode({**row, "messages": row["messages"][:-1]})["input_ids"] + [tok.assistant_id]
assert explicit["input_ids"] == encoded["input_ids"] + [tok.assistant_id]
try:
    RecordEncoder(tok, ".", context=2).encode(row)
except ValueError as error:
    out["context_error"] = str(error)
else:
    raise AssertionError("應拒絕超過 context 的輸入")


class TensorTokenizer:
    def __getattr__(self, name):
        return getattr(tok, name)

    def encode(self, text):
        return torch.tensor(tok.encode(text))


assert RecordEncoder(TensorTokenizer(), ".").encode(row)["input_ids"] == encoded["input_ids"]
image_row = {**row, "image": "public.png"}
fake_payload = {"kind": "image", "values": torch.ones(2, 1, 32, 32)}
with patch.object(enc, "_modality", return_value=fake_payload):
    image_encoded = enc.encode(image_row, modality_override="zero")
assert image_encoded["input_ids"].count(tok.image_id) == 2
assert not image_encoded["modalities"][0]["values"].any()
batch = enc.batch([row, both])
assert batch["input_ids"].dtype == torch.long and batch["input_ids"].device.type == "cpu"
out["encoder_options"] = {"pretrain_target_count": sum(x != -100 for x in pretrain["labels"]),
                          "generation_drops_final_record_message": True,
                          "explicit_messages_retains_provided_messages": True,
                          "tensor_conversion_matches": True, "zero_override_slots": 2,
                          "batch_shape": list(batch["input_ids"].shape),
                          "note": "模態測試用替身特徵驗證 slots 與歸零分支；未重做真實圖片或錄音前處理。"}

config = SelftrainedConfig(vocab_size=57, architecture="dense", width=16, layers=1,
                          heads=2, kv_heads=1, ffn_hidden=32, experts=2, max_length=256)
model = LimitedAssistant(config)
set_trainable(model, "sft")
names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
assert names and all(name.startswith("lm.") for name in names)
result = model(**enc.batch([row]))
labels = torch.tensor([encoded["labels"]])
manual = torch.nn.functional.cross_entropy(result["logits"][labels != -100], labels[labels != -100])
assert torch.allclose(result["language_loss"], manual)
logits = result["logits"].detach().clone().requires_grad_(True)
total, count = loss_sum(logits, labels)
total.backward()
assert torch.count_nonzero(logits.grad[labels == -100]) == 0
out["sft_objective"] = {"trainable_prefix": "lm.", "trainable_tensors": len(names),
                        "supervised_tokens": int(count), "loss": float(result["language_loss"].detach()),
                        "manual_loss": float(manual.detach()), "ignored_logit_gradient_nonzero": 0,
                        "optimizer_steps": 0, "parameter_updates": 0,
                        "installed_cross_entropy_signature": str(inspect.signature(torch.nn.functional.cross_entropy))}
seen = []


def next_token(input_ids, **kwargs):
    seen.append(input_ids.detach().tolist()[0])
    scores = torch.full((1, input_ids.shape[1], 57), -100.0)
    scores[0, -1, tok.encode("請")[0] if len(seen) == 1 else tok.eos_id] = 100
    return {"logits": scores, "cache": None}


with patch.object(model, "forward", side_effect=next_token):
    generated = model.generate(torch.tensor([generation["input_ids"]]), max_new_tokens=3, use_cache=False)
assert generated.tolist() == [[tok.encode("請")[0], tok.eos_id]]
assert seen[1] == seen[0] + [tok.encode("請")[0]]
out["autoregressive_path"] = {"generated_ids": generated.tolist(), "forward_calls": len(seen),
                              "second_input_is_first_plus_generated_character": True,
                              "note": "固定 logits 只核對自回歸輸入與 EOS 停止，不測訓練後品質。"}

manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_text())
package = manifest["package"]
archive_url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{package['revision']}/{package['path']}"
raw, receipt = fetch(archive_url)
assert len(raw) == package["bytes"] and sha(raw) == package["sha256"]
records, file_receipts = [], []
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
    member_names = archive.getnames()
    for expected in manifest["records"]:
        matches = [name for name in member_names if name == expected["path"] or name.endswith("/" + expected["path"])]
        assert len(matches) == 1
        content = archive.extractfile(matches[0]).read()
        assert sha(content) == expected["sha256"] and len(content) == expected["bytes"]
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        records.extend(rows)
        file_receipts.append({**expected, "rows": len(rows)})
del raw
train_text = [r for r in records if r["split"] == "train" and r["task"] in ["text", "tool_unavailable"]]
pairs = [
    ("App打不開，該怎麼辦？", "先檢查網路並重新啟動App，仍失敗再詢問官方客服。"),
    ("那個問題要怎麼處理？", "請說明是地址、App還是卡片的問題。"),
    ("算0加0是多少。", "計算器目前不可用，請開啟後再計算。"),
    ("請預測明天天氣。", "這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。"),
]
selected = []
for question, answer in pairs:
    matches = [r for r in train_text if r["messages"][-2]["content"] == question and r["messages"][-1]["content"] == answer]
    assert matches, question
    selected.append({"question": question, "answer": answer, "matches": len(matches), "example": matches[0]})
out["sample_scoring"] = [score_reply(item["example"], item["answer"], {"tool_call": None, "status": "no_tool"}) for item in selected]
assert all(item["semantic"] and item["format"] for item in out["sample_scoring"])
history = "我先選褲子，後來改選短靴，接下來用一句回答。"
pref = [r for r in train_text if any(m["content"] == history for m in r["messages"])
        and r["messages"][-1]["content"] == "你選了短靴。"]
assert pref
groups, families = defaultdict(set), defaultdict(set)
for record in records:
    groups[record["group_id"]].add(record["split"])
    family = record.get("supervision", {}).get("template_family")
    if family:
        families[family].add(record["split"])
assert all(len(value) == 1 for value in groups.values())
assert all(len(value) == 1 for value in families.values())
full_tok = train_tokenizer(records)
assert full_tok.vocab_size == 550
out["fixed_package"] = {"download": receipt, "record_files": file_receipts,
                         "total_rows": len(records), "table_examples": selected,
                         "preference_example": pref[0],
                         "preference_exact_question_count": sum(r["messages"][-2]["content"] == "請說出我的選項。" for r in pref),
                         "preference_scope": "核對最新選項規則；正文例句的問句與歷史組合不是逐字資料摘錄。",
                         "cross_split_group_collisions": 0,
                         "cross_split_template_family_collisions": 0,
                         "vocab_size_from_all_train_text": full_tok.vocab_size,
                         "sft_train_rows": len(stage_records(records, "sft", "train")),
                         "task_counts": dict(Counter(r["task"] for r in records)),
                         "note": "資料包只在記憶體中讀取；不保存第三方資料或完整壓縮包。"}
smoke_path = ROOT / "docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json"
smoke = json.loads(smoke_path.read_text())
invocation = next(item for item in smoke["invocations"] if item["task"] == "text")
public_messages = json.loads((ROOT / "docs/selftrained/examples/v2/text.messages.json").read_text())
actual = invocation["actual_generations"][0]
assert actual["prompt_messages"] == public_messages
assert actual["raw_output"] == invocation["actual_answer"] == "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
assert invocation["argv"][invocation["argv"].index("--device") + 1] == "cpu"
assert full_tok.decode(actual["generated_ids"], False) == invocation["actual_answer"] + "<eos>"
assert actual["prompt_ids"] == RecordEncoder(full_tok, ".").encode(row, generation=True, messages=public_messages)["input_ids"]
url = "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/" + smoke["HF_commit"] + "/selftrained/v2/moe-joint/tokenizer.json"
tokenizer_raw, tokenizer_receipt = fetch(url)
assert sha(tokenizer_raw) == invocation["model"]["files"]["tokenizer.json"]
assert CharacterTokenizer.from_dict(json.loads(tokenizer_raw)).to_dict() == full_tok.to_dict()
out["historical_cpu_generation_audit"] = {"receipt_path": str(smoke_path.relative_to(ROOT)),
                                         "receipt_sha256": sha(smoke_path.read_bytes()),
                                         "actual_answer": invocation["actual_answer"],
                                         "prompt_matches_public_input": True,
                                         "prompt_and_generated_ids_match_pinned_tokenizer": True,
                                         "public_tokenizer_fetch": tokenizer_receipt,
                                         "recorded_device": "cpu", "recorded_generation_count": invocation["generation_count"],
                                         "recorded_eos": actual["eos"],
                                         "new_weighted_model_generation_runs": 0,
                                         "note": "核對已存的原始 CPU 執行紀錄與生成路徑；本回合未重跑公開權重。"}

primary = []
for url, version, selections, locators in [
    ("https://arxiv.org/html/2203.02155v1", "arXiv:2203.02155v1, 2022-03-04",
     ["Our labelers provide demonstrations of the desired behavior", "Our models are neither fully aligned nor fully safe"],
     ["High-level methodology / Collect demonstration data", "Limitations / Models"]),
    ("https://arxiv.org/html/2305.18290v3", "arXiv:2305.18290v3, 2024-07-29",
     ["without explicit reward modeling or reinforcement learning"], ["Introduction / Direct Preference Optimization"]),
    (f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/nn/functional.py",
     f"PyTorch {torch.__version__}; git {torch.version.git_version}",
     ["ignore_index: int = -100", "does not contribute to the input gradient"], ["cross_entropy signature", "cross_entropy docstring / ignore_index"]),
]:
    body, source_receipt = fetch(url)
    if url.endswith(".py"):
        original = body.decode()
        start = original.index("def cross_entropy(")
        inspected = original[start:original.index("\ndef ", start + 1)]
    else:
        parser = TextParser()
        parser.feed(body.decode())
        original = " ".join(" ".join(parser.parts).split())
        inspected = original
    for selected_quote in selections:
        assert selected_quote in inspected
    primary.append({**source_receipt, "version": version, "locators": locators,
                    "short_excerpts": selections,
                    "read_scope": "讀取原文方法與限制段落；只保存支持主張的小段摘錄。"})
    print(json.dumps({"original_source": url, "received_sha256": source_receipt["sha256"], "checked_excerpt_count": len(selections)}, ensure_ascii=False))
save("primary-sources.json", primary)
save("verification.json", out)
print(json.dumps({"verification": "完成", "vocabulary": [57, 550], "archive_record_files": len(file_receipts),
                  "primary_sources": len(primary), "parameter_updates": 0}, ensure_ascii=False))
