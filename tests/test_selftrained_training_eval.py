"""CPU 正確性證據；這些小樣本不代表成品能力通過。"""

import copy
import json
import signal
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from scripts.selftrained import evaluate, train
from tiny_perceptron.selftrained.dataset import RecordEncoder, perception_loss, read_records, train_tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tools import run_tool_loop, serialize_tool_call


@pytest.fixture
def corpus(tmp_path):
    Image.fromarray(np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)).save(tmp_path / "pixels.png")
    sf.write(tmp_path / "wave.wav", np.sin(np.arange(8000) * 0.05).astype("float32"), 8000)
    records = []
    for split in ("train", "validation", "test"):
        for task in ("text", "vision_clothing", "ocr", "voice_qa", "tool_call"):
            record = {
                "id": f"{split}-{task}",
                "group_id": f"source-{split}-{task}",
                "split": split,
                "task": task,
                "messages": [{"role": "user", "content": "請回答。"}, {"role": "assistant", "content": "好。"}],
                "supervision": {"semantic_all": ["好"]},
            }
            if task == "vision_clothing":
                record.update(
                    image="pixels.png", image_layout={"axis": "horizontal", "slots": [[0, 0, 64, 64], [64, 0, 128, 64]]}
                )
                record["supervision"].update(vision_labels=[0, 1])
            elif task == "ocr":
                record.update(image="pixels.png", roi=[0, 0, 100, 32])
                record["supervision"].update(ocr_text="大小")
                record["messages"][-1]["content"] = "大小"
            elif task == "voice_qa":
                record.update(audio="wave.wav", modality_message_index=0)
                record["supervision"].update(intent="address", intent_id=0)
            elif task == "tool_call":
                call = serialize_tool_call("add", 2, 3)
                record["messages"] = [
                    {"role": "user", "content": "幫我計算2加3。"},
                    {"role": "assistant", "content": call},
                ]
                record["supervision"].update(expected_call=json.loads(call), expected_result=5)
            records.append(record)
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return tmp_path, path, records


def config_file(root):
    path = root / "config.json"
    path.write_text(json.dumps({"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2}))
    return path


def test_labels_are_shifted_and_only_assistant_is_supervised(corpus):
    root, _, records = corpus
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 256)
    row = encoder.encode(records[0])
    position = row["input_ids"].index(tokenizer.assistant_id)
    assert row["labels"][position : position + 3] == tokenizer.encode("好。") + [tokenizer.eos_id]
    assert all(label == -100 for label in row["labels"][:position])
    assert row["labels"][-1] == -100
    empty = encoder.encode(records[0], generation=True, messages=[])
    assert empty["input_ids"] == [tokenizer.bos_id, tokenizer.assistant_id]


def test_final_gold_and_supervision_cannot_change_generation_inputs(corpus):
    root, _, records = corpus
    encoder = RecordEncoder(train_tokenizer(records), root, 256)
    for record in records:
        changed = copy.deepcopy(record)
        changed["supervision"] = {
            "vision_labels": [2, 2],
            "intent_id": 2,
            "ocr_text": "入口",
            "transcription": "秘密答案",
        }
        changed["messages"][-1]["content"] = "秘密答案"
        first, second = encoder.encode(record, generation=True), encoder.encode(changed, generation=True)
        assert first["input_ids"] == second["input_ids"]
        for a, b in zip(first["modalities"], second["modalities"]):
            assert a.keys() == b.keys()
            assert set(a) <= {"kind", "values", "valid", "coordinates"}
            assert torch.equal(a["values"], b["values"])


def test_no_silent_truncation_or_path_escape(corpus):
    root, _, records = corpus
    encoder = RecordEncoder(train_tokenizer(records), root, 4)
    with pytest.raises(ValueError, match="禁止靜默截斷"):
        encoder.encode(records[0])
    encoder.context = 256
    image = copy.deepcopy(records[1])
    image["image"] = "../escape.png"
    with pytest.raises(ValueError, match="相對路徑"):
        encoder.encode(image)


def test_full_audio_and_public_message_placement(corpus):
    root, _, records = corpus
    encoder = RecordEncoder(train_tokenizer(records), root, 256)
    voice = next(r for r in records if r["task"] == "voice_qa")
    row = encoder.encode(voice, generation=True)
    assert row["modalities"][0]["values"].shape == (101, 40)
    assert row["modalities"][0]["valid"].all()
    assert row["input_ids"].count(encoder.tokenizer.audio_id) == 16


def test_mixed_batch_gives_lm_and_all_perception_gradients(corpus):
    root, _, records = corpus
    torch.set_num_threads(2)
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 256)
    model = LimitedAssistant(
        SelftrainedConfig(
            vocab_size=tokenizer.vocab_size,
            width=16,
            layers=1,
            heads=2,
            kv_heads=1,
            ffn_hidden=32,
            experts=2,
            max_length=256,
        )
    )
    examples = [r for r in records if r["split"] == "train"]
    result = model(**encoder.batch(examples))
    head_loss, _ = perception_loss(result["perception"], examples)
    loss = result["loss"] + head_loss
    assert torch.isfinite(loss)
    loss.backward()
    for prefix in ("lm.", "vision_encoder.", "ocr_encoder.", "audio_encoder."):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        )


@pytest.mark.parametrize("stage,task", [("vision", "vision_clothing"), ("ocr", "ocr"), ("audio", "voice_qa")])
def test_perception_stage_updates_its_own_encoder_and_keeps_lm_frozen(corpus, stage, task):
    root, _, records = corpus
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 256)
    model = LimitedAssistant(
        SelftrainedConfig(
            vocab_size=tokenizer.vocab_size,
            width=16,
            layers=1,
            heads=2,
            kv_heads=1,
            ffn_hidden=32,
            experts=2,
            max_length=256,
        )
    )
    parameters = train.set_trainable(model, stage)
    before = model.lm.embedding.weight.detach().clone()
    example = next(record for record in records if record["task"] == task and record["split"] == "train")
    loss, detail = train.objective(model, encoder, [example], stage, 1.0, 0.01)
    assert detail["language_loss"] is None and detail["perception_loss"] is not None
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(f"{stage}_encoder.")
    )
    torch.optim.SGD(parameters, lr=0.01).step()
    assert torch.equal(model.lm.embedding.weight, before)


def test_sampler_is_independent_of_model_rng_and_resumable(corpus):
    _, _, records = corpus
    first = train.BalancedSampler(records, 9)
    second = train.BalancedSampler(records, 9)
    torch.rand(100)
    assert [r["id"] for r in first.batch(8)] == [r["id"] for r in second.batch(8)]
    state = first.state_dict()
    expected = [r["id"] for r in first.batch(4)]
    second.load_state_dict(state)
    assert [r["id"] for r in second.batch(4)] == expected


def test_trainer_resume_matches_uninterrupted_and_keeps_best(corpus, capsys):
    root, records_path, _ = corpus
    common = [
        "--records",
        str(records_path),
        "--asset-dir",
        str(root),
        "--stage",
        "joint",
        "--batch-size",
        "5",
        "--context",
        "256",
        "--config",
        str(config_file(root)),
        "--eval-every",
        "10",
        "--save-every",
        "1",
        "--threads",
        "2",
        "--export-inference",
    ]
    train.main(common + ["--output-dir", str(root / "whole"), "--steps", "2"])
    train.main(common + ["--output-dir", str(root / "part"), "--steps", "1"])
    train.main(
        common + ["--output-dir", str(root / "resumed"), "--steps", "2", "--resume", str(root / "part/latest.pt")]
    )
    whole = torch.load(root / "whole/latest.pt", weights_only=False)
    resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
    assert whole["tokens"] == resumed["tokens"]
    assert whole["sampler"]["draws"] == resumed["sampler"]["draws"]
    assert all(torch.equal(weight, resumed["model"][name]) for name, weight in whole["model"].items())
    assert (root / "resumed/best.pt").exists()
    assert (root / "resumed/model.safetensors").exists()
    # 恢復已完成步數且 validation 沒有改善，仍要保留選定權重。
    train.main(
        common + ["--output-dir", str(root / "no-improve"), "--steps", "2", "--resume", str(root / "resumed/latest.pt")]
    )
    assert (root / "no-improve/best.pt").exists()
    capsys.readouterr()


def test_evaluator_uses_true_executor_and_second_model_generation(corpus):
    root, _, records = corpus
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 256)
    call = serialize_tool_call("add", 2, 3)

    class ObservableModel:
        def generate(self, input_ids, **kwargs):
            prompt = tokenizer.decode(input_ids[0])
            output = "5。" if tokenizer.tool_id in input_ids[0].tolist() else call
            assert "秘密答案" not in prompt
            return torch.tensor([tokenizer.encode(output)])

    generator = evaluate.Generator(ObservableModel(), encoder, 32)
    record = copy.deepcopy(next(r for r in records if r["task"] == "tool_call"))
    record["messages"][-1]["content"] = "秘密答案"
    trace = run_tool_loop(lambda messages: generator.generate(record, messages), record["messages"][:-1])
    score = evaluate.score_reply(record, trace["final_output"], trace)
    assert trace["executed"] and trace["tool_result"]["result"] == 5
    assert len(generator.calls) == 2
    assert generator.calls[1]["prompt_messages"][-1]["role"] == "tool"
    assert score["tool_roundtrip"]


def test_wrong_tool_and_all_class_keyword_lists_do_not_pass(corpus):
    _, _, records = corpus
    vision = copy.deepcopy(next(r for r in records if r["task"] == "vision_clothing"))
    vision["messages"][-1]["content"] = "左邊是包。"
    vision["supervision"].update(semantic_all=["左", "包"], expected_tool=False)
    empty_trace = {"tool_call": None, "status": "no_tool", "executed": False}
    assert evaluate.score_reply(vision, "左邊是包。", empty_trace)["semantic"]
    assert not evaluate.score_reply(vision, "左、右、上、下、褲子、包、短靴。", empty_trace)["semantic"]
    assert not evaluate.score_reply(vision, "左邊不是包。", empty_trace)["semantic"]
    assert not evaluate.score_reply(vision, "左邊是包？", empty_trace)["semantic"]
    non_tool = next(r for r in records if r["task"] == "text")
    wrong_trace = {
        "tool_call": {"tool": "calculator", "operation": "add", "a": 2, "b": 3},
        "status": "executed",
        "executed": True,
    }
    score = evaluate.score_reply(non_tool, "好。", wrong_trace)
    assert not score["semantic"] and score["unexpected_tool_requested"]
    call_record = next(r for r in records if r["task"] == "tool_call")
    correct_trace = {**wrong_trace, "tool_result": {"tool": "calculator", "ok": True, "result": 5}}
    assert not evaluate.score_reply(call_record, "結果不是5。", correct_trace)["tool_roundtrip"]
    assert not evaluate.score_reply(call_record, "結果可能是5。", correct_trace)["tool_roundtrip"]
    assert evaluate.score_reply(call_record, "1. 計算器已完成計算。\n2. 結果是5。", correct_trace)["tool_roundtrip"]
    ocr = next(r for r in records if r["task"] == "ocr")
    score = evaluate.score_reply(ocr, "大小", wrong_trace)
    assert not score["semantic"] and score["unexpected_tool_requested"]


def test_tool_loop_preserves_message_level_public_modalities(corpus):
    root, _, records = corpus
    image = copy.deepcopy(next(r for r in records if r["task"] == "vision_clothing"))
    image["messages"][0].update(image=image.pop("image"), image_layout=image.pop("image_layout"))
    tokenizer = train_tokenizer(records)
    encoder = RecordEncoder(tokenizer, root, 256)
    observed = []

    def generate(messages):
        row = encoder.encode(image, generation=True, messages=messages)
        observed.append(row)
        return "好。"

    run_tool_loop(generate, image["messages"][:-1])
    assert len(observed[0]["modalities"]) == 1
    assert observed[0]["input_ids"].count(tokenizer.image_id) == 2


@pytest.mark.parametrize("task", ["vision_clothing", "ocr"])
def test_mixed_visual_history_uses_each_messages_public_roi(corpus, task):
    root, _, records = corpus
    encoder = RecordEncoder(train_tokenizer(records), root, 256)
    record = {
        "id": "mixed-visual",
        "task": task,
        "messages": [
            {
                "role": "user",
                "content": "圖片。",
                "image": "pixels.png",
                "image_layout": {"slots": [[0, 0, 64, 64], [64, 0, 128, 64]]},
            },
            {"role": "user", "content": "讀字。", "image": "pixels.png", "roi": [0, 0, 100, 32]},
            {"role": "assistant", "content": "大小"},
        ],
    }
    encoded = encoder.encode(record, generation=True)
    assert [payload["kind"] for payload in encoded["modalities"]] == ["image", "ocr"]


def test_voice_continuation_uses_actual_first_reply_not_demonstration(corpus, monkeypatch, capsys):
    root, _, records = corpus
    selected = []
    for split in ("train", "validation", "test"):
        record = copy.deepcopy(next(r for r in records if r["task"] == "voice_qa" and r["split"] == split))
        record["task"] = "voice_topic_continuation"
        record["modality_message_index"] = 3
        record["messages"] = [
            {"role": "system", "content": "有限客服。"},
            {"role": "user", "content": "請用一句。"},
            {"role": "assistant", "content": "好，會用一句。"},
            {"role": "user", "content": "請回答語音。"},
            {"role": "assistant", "content": "絕不可輸入示範答案。"},
            {"role": "user", "content": "改成兩點。"},
            {"role": "assistant", "content": "1. 好。\n2. 好。"},
        ]
        record["supervision"].update(format="two_points", initial_format="one_sentence")
        record["evaluation"] = {
            "mode": "voice_topic_continuation",
            "first_prompt_message_count": 4,
            "first_target_message_index": 4,
            "continuation_user_message_index": 5,
            "initial_format": "one_sentence",
            "final_format": "two_points",
        }
        selected.append(record)
    path = root / "continuation.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n", encoding="utf-8")
    train.main(
        [
            "--records",
            str(path),
            "--asset-dir",
            str(root),
            "--output-dir",
            str(root / "voice-run"),
            "--stage",
            "audio",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--context",
            "256",
            "--config",
            str(config_file(root)),
        ]
    )
    observations = []

    def observed_generate(self, record, messages, override=None):
        assert all("絕不可輸入示範答案" not in m["content"] for m in messages)
        self.encoder.encode(record, generation=True, messages=messages)
        observations.append(copy.deepcopy(messages))
        if len(messages) == 4:
            return "好。"
        assert messages[-2] == {"role": "assistant", "content": "好。"}
        return "1. 好。\n2. 好。"

    monkeypatch.setattr(evaluate.Generator, "generate", observed_generate)
    metric = evaluate.main(
        [
            "--records",
            str(path),
            "--asset-dir",
            str(root),
            "--checkpoint",
            str(root / "voice-run/best.pt"),
            "--output-dir",
            str(root / "voice-val"),
            "--max-new-tokens",
            "4",
            "--controls",
            "none",
        ]
    )
    assert len(observations) == 2
    assert metric["voice_topic_continuation"]["end_to_end_correct"] == 1
    assert metric["voice_topic_continuation"]["first_semantic_correct"] == 1
    capsys.readouterr()


def test_frozen_protocol_rejects_changed_conditions_and_second_test(corpus, capsys):
    root, records_path, _ = corpus
    train.main(
        [
            "--records",
            str(records_path),
            "--asset-dir",
            str(root),
            "--output-dir",
            str(root / "run"),
            "--stage",
            "joint",
            "--steps",
            "1",
            "--batch-size",
            "5",
            "--context",
            "256",
            "--config",
            str(config_file(root)),
        ]
    )
    common = [
        "--records",
        str(records_path),
        "--asset-dir",
        str(root),
        "--checkpoint",
        str(root / "run/best.pt"),
        "--max-new-tokens",
        "4",
        "--controls",
        "none",
    ]
    protocol = root / "frozen.json"
    metrics = evaluate.main(common + ["--output-dir", str(root / "val"), "--freeze-protocol", str(protocol)])
    assert metrics["count"] == 5 and not metrics["teacher_forcing_used_for_generation"]
    test_args = common + ["--split", "test", "--output-dir", str(root / "test"), "--protocol", str(protocol)]
    evaluate.main(test_args)
    with pytest.raises(FileExistsError):
        evaluate.main(test_args)
    current = json.loads(protocol.read_text())
    changed = {**current, "max_new_tokens": 5}
    with pytest.raises(ValueError, match="max_new_tokens"):
        evaluate.validate_protocol(current, changed)
    capsys.readouterr()


@pytest.mark.parametrize("recover_tail", [False, True])
def test_interrupted_test_resumes_same_raw_prefix_without_regenerating(corpus, monkeypatch, capsys, recover_tail):
    root, records_path, _ = corpus
    train.main(
        [
            "--records",
            str(records_path),
            "--asset-dir",
            str(root),
            "--output-dir",
            str(root / "resume-run"),
            "--stage",
            "joint",
            "--steps",
            "1",
            "--batch-size",
            "5",
            "--context",
            "256",
            "--config",
            str(config_file(root)),
        ]
    )
    common = [
        "--records",
        str(records_path),
        "--asset-dir",
        str(root),
        "--checkpoint",
        str(root / "resume-run/best.pt"),
        "--max-new-tokens",
        "4",
        "--controls",
        "none",
    ]
    protocol = root / "resume-frozen.json"
    evaluate.main(common + ["--output-dir", str(root / "resume-val"), "--freeze-protocol", str(protocol)])
    original_append = evaluate.EvaluationJournal.append
    first_receipt = None

    def stop_after_first_commit(journal, row):
        nonlocal first_receipt
        original_append(journal, row)
        if len(journal.rows) == 1:
            first_receipt = journal.receipt_path.read_bytes()
        if len(journal.rows) == (2 if recover_tail else 1):
            journal.request_stop(signal.SIGTERM, None)

    monkeypatch.setattr(evaluate.EvaluationJournal, "append", stop_after_first_commit)
    first = evaluate.main(
        common + ["--split", "test", "--protocol", str(protocol), "--output-dir", str(root / "partial-test")]
    )
    expected_prior_rows = 2 if recover_tail else 1
    assert not first["evaluation_complete"] and first["count"] == expected_prior_rows
    if recover_tail:
        # 模擬硬停：第二列 raw/fsync 完成，但 receipt replace 尚未完成。
        (root / "partial-test/evaluation-receipt.json").write_bytes(first_receipt)
    prefix = (root / "partial-test/outputs.jsonl").read_bytes()
    completed_ids = [json.loads(line)["record"]["id"] for line in prefix.splitlines()]
    monkeypatch.setattr(evaluate.EvaluationJournal, "append", original_append)
    generated_ids = []
    original_generate = evaluate.Generator.generate

    def record_generate(generator, record, messages, override=None):
        generated_ids.append(record["id"])
        return original_generate(generator, record, messages, override)

    monkeypatch.setattr(evaluate.Generator, "generate", record_generate)
    resumed = evaluate.main(
        common
        + [
            "--split",
            "test",
            "--protocol",
            str(protocol),
            "--output-dir",
            str(root / "resumed-test"),
            "--resume-output",
            str(root / "partial-test/outputs.jsonl"),
        ]
    )
    assert resumed["evaluation_complete"] and resumed["count"] == 5
    assert resumed["resumed_completed_count"] == expected_prior_rows
    assert resumed["recovered_complete_tail_count"] == int(recover_tail)
    assert not set(completed_ids) & set(generated_ids)
    assert (root / "resumed-test/outputs.jsonl").read_bytes().startswith(prefix)
    receipt = json.loads((root / "resumed-test/evaluation-receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["completed_count"] == 5
    capsys.readouterr()


def test_split_groups_must_stay_together(corpus):
    root, _, records = corpus
    records[-1]["group_id"] = records[0]["group_id"]
    path = root / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="跨 split"):
        read_records([path])


@pytest.mark.parametrize(
    "shared_path,before,after",
    [
        (
            "tiny_perceptron/attention.py",
            "key_positions[None, :] <= query_positions[:, None]",
            "key_positions[None, :] < query_positions[:, None]",
        ),
        ("tiny_perceptron/multimodal.py", "clamp(min=1e-8).log()", "clamp(min=1e-4).log()"),
    ],
)
def test_protocol_binds_shared_attention_logmel_and_inference_sources(
    tmp_path, monkeypatch, shared_path, before, after
):
    repository = Path(evaluate.__file__).resolve().parents[2]
    snapshot = tmp_path / "runtime"
    paths = list((repository / "tiny_perceptron").rglob("*.py"))
    paths += [repository / "scripts/selftrained" / name for name in ("train.py", "evaluate.py", "chat.py")]
    for source in paths:
        destination = snapshot / source.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(evaluate, "__file__", str(snapshot / "scripts/selftrained/evaluate.py"))
    original = evaluate.code_fingerprints()
    for dependency in (
        "tiny_perceptron/model.py",
        "tiny_perceptron/modern.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/multimodal.py",
        "tiny_perceptron/attention.py",
        "tiny_perceptron/selftrained/inference.py",
        "scripts/selftrained/chat.py",
    ):
        assert dependency in original
    operations = snapshot / "scripts/selftrained/modal_runner.py"
    operations.write_text("# 操作收據變動不改模型執行契約。\n", encoding="utf-8")
    assert evaluate.code_fingerprints() == original
    dependency = snapshot / shared_path
    source = dependency.read_text(encoding="utf-8")
    assert before in source
    dependency.write_text(source.replace(before, after), encoding="utf-8")
    current = evaluate.code_fingerprints()
    assert original[shared_path] != current[shared_path]
    with pytest.raises(ValueError, match="code_sha256"):
        evaluate.validate_protocol({"code_sha256": original}, {"code_sha256": current})
