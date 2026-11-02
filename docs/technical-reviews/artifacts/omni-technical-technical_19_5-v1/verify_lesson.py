"""19.5 的離線核對：執行教材原碼；只檢查既有資料與歷史生成，不訓練或推論模型。"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import loss_sum
from tiny_perceptron.selftrained.dataset import RecordEncoder, train_tokenizer
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS, SPECIAL_NAMES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_blocks(blocks):
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for block in blocks:
            exec(compile(block, "19.5-frozen-inline", "exec"), namespace)
    return namespace, output.getvalue()


def main():
    torch.set_num_threads(2)
    started = dt.datetime.now(dt.UTC).isoformat()
    packet_path = ROOT / "docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v10/technical_19_5.json"
    packet = json.loads(packet_path.read_text())
    freeze = json.loads((ROOT / packet["source_freeze"]).read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "19.5")
    body = (ROOT / page["snapshot"]).read_bytes()
    assert sha(body) == page["source_sha256"] == packet["page"]["source_sha256"]
    current = re.search(r"^## 19\.5 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S).group(0)
    assert current.encode() == body
    assert page["figures_sha256"] == {} and not re.search(r"!\[.*?\]\(", body.decode())
    blocks = re.findall(r"```python\n(.*?)```", body.decode(), re.S)
    assert len(blocks) == 3
    ns, printed = run_blocks(blocks)
    tok, library, ours, messages = (ns[k] for k in ["tok", "library", "ours", "messages"])
    assert tok.vocab_size == 57 and len(tok.characters) == 46
    assert ns["same"] and ours["input_ids"] == library["input_ids"] and ours["labels"] == library["labels"]
    answer = "請說明是地址、App還是卡片的問題。<eos>"
    assert tok.decode([x for x in library["labels"] if x != -100], False) == answer
    assert len(library["input_ids"]) == len(library["labels"])
    assert sum(x != -100 for x in library["labels"]) == len(messages[-1]["content"]) + 1
    expected_output = (
        "字表大小 57\nBOS 1 EOS 2 user 3 assistant 4\n" + answer + "\n"
        "與函式庫相同： True\n讀到 <eos> → 目標 不計\n"
        "讀到 <assistant> → 目標 請\n讀到 請 → 目標 說\n讀到 說 → 目標 明\n"
    )
    assert printed == expected_output
    exercise = list(blocks)
    exercise[0] = exercise[0].replace('{"role": "user"', '{"role": "assistant"', 1)
    ex_ns, exercise_printed = run_blocks(exercise)
    assert tok.decode([x for x in ex_ns["library"]["labels"] if x != -100], False) == "那個問題要怎麼處理？<eos>" + answer

    encoder = RecordEncoder(tok, ".")
    pretrain = encoder.encode(ns["row"], pretrain=True)
    assert pretrain["labels"] == pretrain["input_ids"][1:] + [-100]
    generation = encoder.encode(ns["row"], generation=True)
    explicit = encoder.encode(ns["row"], generation=True, messages=messages)
    assert generation["input_ids"][-1] == tok.assistant_id
    assert tok.decode(generation["input_ids"], False) == "<bos><user>那個問題要怎麼處理？<eos><assistant>"
    assert explicit["input_ids"] == library["input_ids"] + [tok.assistant_id]
    try:
        RecordEncoder(tok, ".", context=2).encode(ns["row"])
    except ValueError as e:
        context_error = str(e)
    else:
        raise AssertionError("context must reject, not truncate")

    class TensorTokenizer:
        def __getattr__(self, name):
            return getattr(tok, name)

        def encode(self, text):
            return torch.tensor(tok.encode(text), dtype=torch.long)

    tensor_row = RecordEncoder(TensorTokenizer(), ".").encode(ns["row"])
    assert tensor_row["input_ids"] == library["input_ids"] and tensor_row["labels"] == library["labels"]
    short = {**ns["row"], "messages": [{"role": "user", "content": "那"}, {"role": "assistant", "content": "請"}]}
    batch = encoder.batch([ns["row"], short])
    assert batch["input_ids"].dtype == batch["labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert (batch["labels"][~batch["attention_mask"]] == -100).all()
    cache = Path("/workspace/work/reference/omni-technical-technical_19_5-v1")
    cache.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), 128).save(cache / "synthetic-input.png")
    media_row = {**ns["row"], "task": "vision_clothing", "image": "synthetic-input.png"}
    media_encoder = RecordEncoder(tok, cache)
    media = media_encoder.encode(media_row)
    zero = media_encoder.encode(media_row, modality_override="zero")
    assert media["input_ids"].count(tok.image_id) == 2
    assert media["modalities"][0]["values"].count_nonzero() > 0
    assert zero["modalities"][0]["values"].count_nonzero() == 0
    assert all(v.device.type == "cpu" for v in media_encoder.to_device(media["modalities"])[0].values() if isinstance(v, torch.Tensor))

    labels = torch.tensor([library["labels"]], dtype=torch.long)
    logits = torch.linspace(-1, 1, len(library["labels"]) * tok.vocab_size, dtype=torch.float64).reshape(1, -1, tok.vocab_size).requires_grad_()
    total, count = loss_sum(logits, labels)
    (total / count).backward()
    ignored_gradient = float(logits.grad[labels == -100].abs().max())
    changed = logits.detach().clone()
    changed[labels == -100] = torch.linspace(-100, 100, tok.vocab_size, dtype=torch.float64)
    changed_total, changed_count = loss_sum(changed, labels)
    assert ignored_gradient == 0 and torch.equal(total.detach(), changed_total) and count == changed_count
    assert tok.decode(library["input_ids"], False).startswith("<bos><user>那個問題要怎麼處理？")
    train = load_module("review_19_5_train", "scripts/selftrained/train.py")
    stub = torch.nn.Module()
    for name in ("lm", "vision_encoder", "ocr_encoder", "audio_encoder"):
        stub.add_module(name, torch.nn.Linear(1, 1))
    train.set_trainable(stub, "sft")
    trainable_names = [n for n, p in stub.named_parameters() if p.requires_grad]
    assert trainable_names == ["lm.weight", "lm.bias"]

    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_text())
    archive = ROOT / manifest["package"]["path"]
    assert archive.stat().st_size == manifest["package"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["package"]["sha256"]
    all_records, file_receipts = [], []
    with tarfile.open(archive) as tar:
        for entry in manifest["records"]:
            raw = tar.extractfile(entry["path"]).read()
            assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"]
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            all_records.extend(rows)
            file_receipts.append({**entry, "rows": len(rows)})
    groups, text_families = defaultdict(set), defaultdict(set)
    for row in all_records:
        groups[row["group_id"]].add(row["split"])
        family = row.get("supervision", {}).get("template_family")
        if family:
            text_families[family].add(row["split"])
    assert not any(len(splits) > 1 for splits in groups.values())
    assert not any(len(splits) > 1 for splits in text_families.values())
    product_tok = train_tokenizer(all_records)
    assert product_tok.vocab_size == 550
    serialized_tok = (json.dumps(product_tok.to_dict(), ensure_ascii=False, indent=2) + "\n").encode()
    wanted = [
        ("App打不開，該怎麼辦？", "先檢查網路並重新啟動App，仍失敗再詢問官方客服。"),
        ("那個問題要怎麼處理？", "請說明是地址、App還是卡片的問題。"),
        ("算0加0是多少。", "計算器目前不可用，請開啟後再計算。"),
        ("請預測明天天氣。", "這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。"),
    ]
    matches = []
    for question, target in wanted:
        candidates = [r for r in all_records if r["split"] == "train" and r["messages"][-2]["content"] == question and r["messages"][-1]["content"] == target]
        assert candidates, (question, target)
        row = candidates[0]
        matches.append({"question": question, "target": target, "count": len(candidates), "first_id": row["id"], "group_id": row["group_id"], "task": row["task"], "system_and_history": row["messages"][:-2]})
    preference_rows = [r for r in all_records if r["split"] == "train" and r["messages"][-1]["content"] == "你選了短靴。" and any("我先選褲子，後來改選短靴" in m["content"] for m in r["messages"])]
    assert preference_rows
    train_task_intents = Counter((r["task"], r.get("supervision", {}).get("intent", "")) for r in all_records if r["split"] == "train")
    assert train_task_intents[("text", "preference")] and train_task_intents[("text", "format_change")]
    assert any(r["task"] == "tool_call" for r in all_records if r["split"] == "train")
    assert any(r["task"] == "tool_reply" for r in all_records if r["split"] == "train")

    raw_dir = ROOT / "docs/selftrained/results/public-cpu-raw"
    raw_stdout = (raw_dir / "text-1-stdout.txt").read_bytes()
    raw_result = json.loads((raw_dir / "text-1-result.json").read_text())
    raw_argv = json.loads((raw_dir / "text-1-argv.json").read_text())
    captured = json.loads(raw_stdout)
    public_input = json.loads((ROOT / "docs/selftrained/examples/v2/text.messages.json").read_text())
    assert sha(raw_stdout) == raw_result["stdout_sha256"] == "7d4a38aeaa0268baedd3cc21cbdc48e205a3b078a06fdd008c329bc2efd86ca4"
    assert raw_result["returncode"] == 0 and raw_argv["argv"][raw_argv["argv"].index("--device") + 1] == "cpu"
    assert raw_argv["public_messages_before_execution"] == public_input
    calls = captured["generations"]
    assert len(calls) == 1 and calls[0]["prompt_messages"] == public_input
    assert sha(serialized_tok) == captured["model"]["files"]["tokenizer.json"]
    regenerated_prompt = RecordEncoder(product_tok, ".").encode({"id": "check-history", "task": "text", "messages": public_input}, generation=True, messages=public_input)
    assert regenerated_prompt["input_ids"] == calls[0]["prompt_ids"]
    decoded = product_tok.decode(calls[0]["generated_ids"])
    assert decoded == captured["answer"] == raw_result["actual_answer"] == "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
    assert captured["answer"] not in [m["content"] for m in public_input]
    assert calls[0]["generated_ids"][-1] == product_tok.eos_id and calls[0]["eos"]
    evaluate = load_module("review_19_5_evaluate", "scripts/selftrained/evaluate.py")
    assert evaluate.format_pass(decoded, "two_points") is True

    paths = [
        "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/selftrained/dataset.py",
        "tiny_perceptron/selftrained/model.py", "tiny_perceptron/model.py", "scripts/selftrained/train.py",
        "scripts/selftrained/prepare_text_tools.py", "tiny_perceptron/selftrained/inference.py",
        "scripts/selftrained/chat.py", "scripts/selftrained/evaluate.py", "docs/selftrained/v2-manifest.json",
        "docs/selftrained/examples/v2/text.messages.json", "docs/selftrained/V2-DATA-NOTICE.md",
    ]
    sources = {p: sha((ROOT / p).read_bytes()) for p in paths}
    result = {
        "started_at": started, "completed_at": dt.datetime.now(dt.UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_19_5",
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_19_5-v1/verify_lesson.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "threads": str(torch.get_num_threads())},
        "frozen_source_sha256": sha(body), "code_block_sha256": [sha(b.encode()) for b in blocks],
        "inline_program_stdout": printed, "exercise_stdout": exercise_printed,
        "demo": {"vocab_size": tok.vocab_size, "ordinary_characters": len(tok.characters), "fixed_specials": list(SPECIAL_NAMES), "ocr_characters": OCR_CHARACTERS, "input_positions": len(library["input_ids"]), "supervised_positions": int(count), "assistant_content_length": len(messages[-1]["content"]), "input_and_labels_equal_library": True},
        "api_checks": {"pretrain_next_token_labels": True, "generation_implicit_drops_last": True, "generation_explicit_retains_all": True, "tensor_to_list": True, "context_error": context_error, "batch_shape": list(batch["input_ids"].shape), "ids_labels_dtype": str(batch["labels"].dtype), "attention_dtype": str(batch["attention_mask"].dtype), "padding_ignored": True, "modality_slots": 2, "modality_override_zero": True, "to_device_cpu": True, "sft_trainable_stub_names": trainable_names},
        "loss_check": {"ignored_logit_gradient_max": ignored_gradient, "ignored_logit_change_preserves_loss": True, "total_loss": float(total.detach()), "denominator": int(count), "mean_loss": float((total / count).detach()), "no_model_forward_or_optimizer_step": True},
        "package": {**manifest["package"], "files": file_receipts, "rows_by_split": dict(Counter(r["split"] for r in all_records)), "group_count": len(groups), "cross_split_group_collisions": 0, "template_family_count": len(text_families), "cross_split_template_family_collisions": 0, "matching_train_demonstrations": matches, "preference_policy_history_train_examples": len(preference_rows), "preference_example_scope": "The final paragraph illustrates the expected policy by recombining a training history with a training question; it is not a claim that this exact full dialogue is a stored training row or an observed generation.", "train_tasks_and_intents": {f"{task}:{intent}": n for (task, intent), n in sorted(train_task_intents.items())}, "train_tokenizer_vocab_size": product_tok.vocab_size, "train_tokenizer_file_sha256": sha(serialized_tok)},
        "historical_text": {"stdout_sha256": sha(raw_stdout), "raw_result_sha256": sha((raw_dir / "text-1-result.json").read_bytes()), "raw_argv_sha256": sha((raw_dir / "text-1-argv.json").read_bytes()), "answer": decoded, "prompt_matches_public_input": True, "reencoded_prompt_ids_match": True, "generated_ids_decode_to_answer": True, "target_absent_from_prompt": True, "capture_device": "cpu", "capture_returncode": raw_result["returncode"], "captured_generation_calls": len(calls), "captured_generated_tokens_including_eos": len(calls[0]["generated_ids"]), "eos": True, "two_points_format": True, "public_revision": captured["public_source"]["revision"], "scope": "one selected historical validation example; no new inference or generalization estimate"},
        "repository_source_sha256": sources,
        "result": "all assertions passed; no training, model inference, GPU, downloads of weights, or external writes",
    }
    output = Path(__file__).with_name("execution-v1.json")
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"result": result["result"], "output": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "demo_stdout": printed, "product_vocab": product_tok.vocab_size, "historical_captured_calls": len(calls)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
