#!/usr/bin/env python3
"""實際自回歸生成、感知及工具往返評估；test 須先凍結協定。"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import signal
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiny_perceptron.selftrained.dataset import (  # noqa: E402
    PREPROCESS_VERSION,
    RecordEncoder,
    asset_fingerprints,
    data_fingerprints,
    decode_ocr,
    file_sha256,
    read_records,
    record_asset_paths,
)
from tiny_perceptron.selftrained.tools import (  # noqa: E402
    run_tool_loop,
    serialize_tool_result,
)

PROTOCOL_VERSION = "selftrained-generation-v1"
THRESHOLDS = {
    "text_semantic_per_intent": 0.8,
    "format": 0.9,
    "vision_per_class_relation": 0.8,
    "ocr_cer_max": 0.1,
    "ocr_exact": 0.8,
    "voice_per_intent": 0.8,
    "tool_roundtrip": 0.95,
    "unsupported_clarification": 0.8,
}


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EvaluationJournal:
    """每列 commit 原始 bytes 與收據；續跑不再次生成已 commit 的 record。"""

    def __init__(self, args, current, selected):
        self.output_dir = Path(args.output_dir)
        self.output_path = self.output_dir / "outputs.jsonl"
        self.receipt_path = self.output_dir / "evaluation-receipt.json"
        self.rows, self.prefix = [], b""
        self.recovered_complete_tail_count = 0
        self.incomplete_tail_bytes = 0
        self.expected_ids = [record["id"] for record in selected]
        conditions = {key: value for key, value in current.items() if key != "created_at"}
        conditions.update(split=args.split, selected_ids=self.expected_ids, limit=args.limit)
        self.base = {
            "schema": "selftrained-evaluation-journal-v1",
            "split": args.split,
            "conditions_sha256": canonical_sha(conditions),
            "checkpoint_sha256": current["checkpoint_sha256"],
            "protocol_sha256": file_sha256(args.protocol) if args.protocol else None,
            "expected_count": len(selected),
            "resumed_from": args.resume_output,
        }
        if args.resume_output:
            previous = Path(args.resume_output)
            prior_receipt = json.loads(previous.with_name("evaluation-receipt.json").read_text(encoding="utf-8"))
            for key in (
                "schema",
                "split",
                "conditions_sha256",
                "checkpoint_sha256",
                "protocol_sha256",
                "expected_count",
            ):
                if prior_receipt.get(key) != self.base[key]:
                    raise ValueError(f"evaluation resume 條件不同: {key}")
            raw = previous.read_bytes()
            size = prior_receipt["output_byte_count"]
            if size < 0 or len(raw) < size:
                raise ValueError("resume 原始 bytes 不完整")
            self.prefix = raw[:size]
            if (self.prefix and not self.prefix.endswith(b"\n")) or hashlib.sha256(
                self.prefix
            ).hexdigest() != prior_receipt["outputs_sha256"]:
                raise ValueError("resume 已 commit 的原始 bytes 指紋不符")
            self.rows = [json.loads(line) for line in self.prefix.decode("utf-8").splitlines()]
            ids = [row["record"]["id"] for row in self.rows]
            if ids != self.expected_ids[: len(ids)] or len(ids) != prior_receipt["completed_count"]:
                raise ValueError("resume 完成 IDs 並非原凍結順序的唯一 prefix")
            if (
                hashlib.sha256("".join(identifier + "\n" for identifier in ids).encode()).hexdigest()
                != prior_receipt["completed_ids_sha256"]
            ):
                raise ValueError("resume 完成 IDs 指紋不同")
            expected = {record["id"]: record for record in selected}
            if any(row["record"] != expected[row["record"]["id"]] for row in self.rows):
                raise ValueError("resume record 與凍結資料不同")
            # crash 可發生在 row fsync 與 receipt replace 之間；完整 raw row 不能重生。
            tail = raw[size:]
            complete_size = tail.rfind(b"\n") + 1
            complete_tail = tail[:complete_size]
            self.incomplete_tail_bytes = len(tail) - complete_size
            if complete_tail:
                try:
                    recovered = [json.loads(line) for line in complete_tail.decode("utf-8").splitlines()]
                    recovered_ids = [row["record"]["id"] for row in recovered]
                    if recovered_ids != self.expected_ids[len(ids) : len(ids) + len(recovered_ids)]:
                        raise ValueError("完整 tail 並非下一批 expected IDs")
                    for row in recovered:
                        if row["record"] != expected[row["record"]["id"]]:
                            raise ValueError("完整 tail 的 record 非凍結資料")
                        if row["score"] != score_reply(row["record"], row["trace"]["final_output"], row["trace"]):
                            raise ValueError("完整 tail 的 score 與凍結 grader 不符")
                        if not {"model_generations", "perception", "routing"} <= row.keys():
                            raise ValueError("完整 tail 缺原始生成或感知證據")
                except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("完整 raw tail 無法核對，禁止丟棄後重新生成") from exc
                self.rows.extend(recovered)
                self.prefix += complete_tail
                self.recovered_complete_tail_count = len(recovered)
            if args.split == "test" and prior_receipt["status"] == "complete" and len(ids) == len(self.expected_ids):
                raise ValueError("test 已全部完成，禁止重新生成")
        self.digest = hashlib.sha256(self.prefix)
        self.byte_count = len(self.prefix)
        self.id_digest = hashlib.sha256()
        for row in self.rows:
            self.id_digest.update((row["record"]["id"] + "\n").encode())
        self.stop_requested = False

    def save_receipt(self, status):
        receipt = {
            **self.base,
            "status": status,
            "completed_count": len(self.rows),
            "output_byte_count": self.byte_count,
            "outputs_sha256": self.digest.hexdigest(),
            "completed_ids_sha256": self.id_digest.hexdigest(),
            "recovered_complete_tail_count": self.recovered_complete_tail_count,
            "incomplete_tail_bytes_retained_in_source": self.incomplete_tail_bytes,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        temporary = self.receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.receipt_path)

    def request_stop(self, signum, frame):
        self.stop_requested = True
        print(
            json.dumps({"event": "evaluation_stop_requested", "signal": signum, "saving_at": "next_record_boundary"}),
            flush=True,
        )

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("xb")
        self.handle.write(self.prefix)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handlers = {number: signal.signal(number, self.request_stop) for number in (signal.SIGTERM, signal.SIGINT)}
        self.save_receipt("running")
        return self

    def append(self, row):
        raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        self.handle.write(raw)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.rows.append(row)
        self.digest.update(raw)
        self.byte_count += len(raw)
        self.id_digest.update((row["record"]["id"] + "\n").encode())
        self.save_receipt("running")

    def __exit__(self, exc_type, exc, traceback):
        status = "failed" if exc_type else ("complete" if len(self.rows) == len(self.expected_ids) else "interrupted")
        self.save_receipt(status)
        self.handle.close()
        for number, handler in self.handlers.items():
            signal.signal(number, handler)


def normalize(text):
    return re.sub(r"\s+", "", text).strip("。！!？?\"'「」")


def positive_visual_answer(text):
    # 問號不當作句號移除；「不是／可能／嗎」也不會和固定正向答案相等。
    return re.sub(r"\s+", "", text).strip("。！!\"'「」")


def numeric_reply_value(text):
    """保守的有限正向計算答覆文法，拒絕否定、猜測與多個矛盾數字。"""
    value = re.sub(r"\s+", "", text).strip("\"'「」")
    two_points = r"(?:1[.、)]計算器已完成計算[。.!]?2[.、)])?"
    affirmative = r"(?:結果(?:是|為)|計算結果(?:是|為)|答案(?:是|為))?"
    match = re.fullmatch(two_points + affirmative + r"(-?\d+)[。.!]?", value)
    return int(match.group(1)) if match else None


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def format_pass(text, requested):
    if not requested:
        return None
    if requested in ("one_sentence", "sentence", "一句"):
        return (
            "\n" not in text.strip()
            and not re.search(r"(?:^|\s)[12][.、)]", text)
            and len(re.findall(r"[。！？!?]", text)) <= 1
        )
    if requested in ("two_points", "two_bullets", "兩點"):
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == 2 and all(re.match(r"\s*(?:[12][.、)]|[-•])", line) for line in lines)
    return None


def score_reply(record, output, trace):
    supervision = record.get("supervision", {})
    target = record["messages"][-1]["content"]
    rubric = supervision.get("rubric", {})
    if not isinstance(rubric, dict):
        rubric = {}
    all_terms = supervision.get("semantic_all", rubric.get("semantic_all", []))
    any_terms = supervision.get("semantic_any", rubric.get("semantic_any", []))
    forbidden = supervision.get("semantic_forbidden", rubric.get("semantic_forbidden", []))
    if all_terms or any_terms:
        semantic = (
            all(term in output for term in all_terms)
            and (not any_terms or any(term in output for term in any_terms))
            and not any(term in output for term in forbidden)
        )
        scoring = "frozen keyword rubric"
    else:
        semantic = normalize(output) == normalize(target)
        scoring = "normalized exact target (no semantic model judge)"
    exact = normalize(output) == normalize(target)
    if record["task"].startswith("vision"):
        # 成品只教有限正向短答；保守 exact 避免羅列、否定或疑問灌分。
        semantic = exact = positive_visual_answer(output) == positive_visual_answer(target)
        scoring = "conservative exact finite positive visual answer"
    requested_tool = trace.get("tool_call") is not None or trace.get("status") == "invalid_call"
    expected_tool = supervision.get("expected_tool", record["task"] == "tool_call")
    requested = supervision.get("format", rubric.get("format"))
    scored = {
        "exact": exact,
        "semantic": semantic,
        "format": format_pass(output, requested),
        "scoring": scoring,
        "tool_requested": requested_tool,
        "unexpected_tool_requested": bool(requested_tool and not expected_tool),
    }
    if record["task"] == "ocr":
        gold = supervision.get("ocr_text", target)
        predicted = normalize(output)
        scored.update(semantic=predicted == gold, ocr_errors=edit_distance(predicted, gold), ocr_characters=len(gold))
    if record["task"] == "tool_call":
        expected = supervision.get("expected_call")
        if isinstance(expected, str):
            expected = json.loads(expected)
        parameters_correct = expected is not None and trace["tool_call"] == expected
        returned = trace.get("tool_result") or {}
        # 最後數值須來自真回填；JSON 呼叫本身不算最終答覆。
        final_correct = returned.get("ok") and numeric_reply_value(output) == returned.get("result")
        roundtrip = bool(parameters_correct and trace["executed"] and final_correct)
        scored.update(
            semantic=roundtrip,
            tool_parameters=parameters_correct,
            tool_executed=trace["executed"],
            tool_valid=trace["tool_call"] is not None,
            tool_final=bool(final_correct),
            tool_roundtrip=roundtrip,
        )
    if expected_tool is False:
        scored["semantic"] = bool(scored["semantic"] and not requested_tool)
    return scored


def code_fingerprints():
    root = Path(__file__).resolve().parents[2]
    # 明定執行邊界：包含重用 backbone/attention/log-mel 與公開 inference。
    # Modal、發布、封裝與 jobs 屬操作流程，不放入最終測試的模型契約。
    names = [str(path.relative_to(root)) for path in sorted((root / "tiny_perceptron").rglob("*.py"))]
    names.extend(("scripts/selftrained/train.py", "scripts/selftrained/evaluate.py", "scripts/selftrained/chat.py"))
    return {name: file_sha256(root / name) for name in names}


def protocol_contents(args, checkpoint, records):
    return {
        "version": PROTOCOL_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "architecture": checkpoint["config"]["architecture"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "data_sha256": data_fingerprints(args.records),
        "asset_sha256": asset_fingerprints(records, args.asset_dir),
        "code_sha256": code_fingerprints(),
        "preprocess_version": PREPROCESS_VERSION,
        "context": checkpoint["config"]["max_length"],
        "max_new_tokens": args.max_new_tokens,
        "controls": args.controls,
        "thresholds": THRESHOLDS,
        "prompt_policy": "ordered messages without last assistant target; public ROI/layout; no supervision/transcript/path text",
        "selection": "validation only",
        "test_once": True,
    }


def validate_protocol(protocol, current):
    for key in (
        "version",
        "checkpoint_sha256",
        "architecture",
        "tokenizer_sha256",
        "data_sha256",
        "asset_sha256",
        "code_sha256",
        "preprocess_version",
        "context",
        "max_new_tokens",
        "controls",
        "thresholds",
        "prompt_policy",
    ):
        if protocol.get(key) != current.get(key):
            raise ValueError(f"凍結協定不匹配: {key}；不能看 test 後更改")


class Generator:
    def __init__(self, model, encoder, maximum):
        self.model, self.encoder, self.maximum = model, encoder, maximum
        self.calls = []

    @torch.no_grad()
    def generate(self, record, messages, override=None):
        encoded = self.encoder.encode(record, generation=True, messages=messages, modality_override=override)
        ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.encoder.device)
        if len(encoded["input_ids"]) + self.maximum > self.encoder.context:
            raise ValueError(f"{record['id']} prompt+generation 超出 context；請在 validation 確定合法預算後凍結")
        modalities = [self.encoder.to_device(encoded["modalities"])]
        output_ids = self.model.generate(
            ids, modalities=modalities, max_new_tokens=self.maximum, eos_id=self.encoder.tokenizer.eos_id
        )
        tokens = output_ids[0].detach().cpu().tolist()
        output = self.encoder.tokenizer.decode(tokens, skip_special_tokens=True)
        self.calls.append(
            {
                "prompt_messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "prompt_ids": encoded["input_ids"],
                "generated_ids": tokens,
                "raw_output": output,
                "modality_kinds": [m["kind"] for m in encoded["modalities"]],
                "invalid_control_tokens": [
                    token for token in tokens if token < 11 and token not in (self.encoder.tokenizer.eos_id,)
                ],
            }
        )
        return output

    @torch.no_grad()
    def perception(self, record, messages=None):
        encoded = self.encoder.encode(record, generation=True, messages=messages)
        ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.encoder.device)
        result = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            modalities=[self.encoder.to_device(encoded["modalities"])],
        )
        predictions = []
        for item in result.get("perception", []):
            kind, logits = item["kind"], item["logits"]
            predicted = decode_ocr(logits) if kind == "ocr" else logits.argmax(-1).detach().cpu().tolist()
            predictions.append({"kind": kind, "prediction": predicted, "logits": logits.detach().cpu().tolist()})
        routing = []
        for layer in result.get("routing", []):
            routing.append(
                None
                if layer is None
                else {
                    key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                    for key, value in layer.items()
                    if key != "chosen"
                }
            )
        return predictions, routing


def summarize(rows):
    buckets = collections.defaultdict(list)
    for row in rows:
        record = row["record"]
        buckets[record["task"]].append(row)
    metrics = {}
    for task, items in sorted(buckets.items()):
        groups = {item["record"]["group_id"] for item in items}
        task_metric = {
            "count": len(items),
            "source_groups": len(groups),
            "recording_assets": len(set().union(*(record_asset_paths(item["record"], "audio") for item in items))),
            "image_assets": len(set().union(*(record_asset_paths(item["record"], "image") for item in items))),
        }
        for name in (
            "exact",
            "semantic",
            "format",
            "tool_parameters",
            "tool_valid",
            "tool_executed",
            "tool_final",
            "tool_roundtrip",
            "tool_requested",
            "unexpected_tool_requested",
        ):
            values = [item["score"][name] for item in items if item["score"].get(name) is not None]
            if values:
                task_metric[name] = {
                    "numerator": sum(values),
                    "denominator": len(values),
                    "rate": sum(values) / len(values),
                }
        if task == "ocr":
            characters = sum(item["score"]["ocr_characters"] for item in items)
            task_metric["final_cer"] = sum(item["score"]["ocr_errors"] for item in items) / max(1, characters)
        metrics[task] = task_metric
    strata = collections.defaultdict(list)
    for row in rows:
        record, supervision = row["record"], row["record"].get("supervision", {})
        for field in ("intent", "format", "history_variant", "case", "font_family", "condition", "control_type"):
            if field in supervision:
                strata[f"{record['task']}/{field}={supervision[field]}"].append(row)
        if "vision_labels" in supervision:
            labels = supervision["vision_labels"]
            query = supervision.get("query_slot", 0)
            if 0 <= query < len(labels) and labels[query] >= 0:
                strata[f"{record['task']}/query_class={labels[query]}"].append(row)
            if record["task"] == "vision_relation":
                axis = record.get("image_layout", {}).get("axis")
                if axis in ("horizontal", "vertical"):
                    position = (("left", "right") if axis == "horizontal" else ("up", "down"))[query]
                    strata[f"vision_relation/axis={axis}"].append(row)
                    strata[f"vision_relation/query_position={position}"].append(row)
    stratified = {}
    for name, items in sorted(strata.items()):
        stratified[name] = {
            "count": len(items),
            "source_groups": len({item["record"]["group_id"] for item in items}),
            "recording_assets": len(set().union(*(record_asset_paths(item["record"], "audio") for item in items))),
            "semantic_numerator": sum(item["score"]["semantic"] for item in items),
            "semantic_rate": sum(item["score"]["semantic"] for item in items) / len(items),
            "exact_numerator": sum(item["score"]["exact"] for item in items),
            "exact_rate": sum(item["score"]["exact"] for item in items) / len(items),
            "format_numerator": sum(item["score"].get("format") is True for item in items),
            "format_denominator": sum(item["score"].get("format") is not None for item in items),
        }
        if items[0]["record"]["task"] == "ocr":
            characters = sum(item["score"]["ocr_characters"] for item in items)
            errors = sum(item["score"]["ocr_errors"] for item in items)
            stratified[name].update(ocr_characters=characters, ocr_errors=errors, final_cer=errors / max(1, characters))
        initial = [item for item in items if "first_reply_score" in item]
        if initial:
            stratified[name].update(
                initial_semantic_numerator=sum(item["first_reply_score"]["semantic"] for item in initial),
                initial_format_numerator=sum(item["first_reply_score"].get("format") is True for item in initial),
                end_to_end_semantic_numerator=sum(item["end_to_end_semantic"] for item in initial),
                end_to_end_format_numerator=sum(item["end_to_end_format"] for item in initial),
            )
    # head 指標與最終生成分開；按類/意圖保留分母。
    head_buckets = collections.defaultdict(list)
    ocr_errors, ocr_characters, ocr_exact = 0, 0, []
    for row in rows:
        supervision = row["record"].get("supervision", {})
        for prediction in row["perception"]:
            if prediction["kind"] == "image":
                for expected, actual in zip(supervision.get("vision_labels", []), prediction["prediction"]):
                    if expected >= 0:
                        head_buckets[f"image_class_{expected}"].append(actual == expected)
            elif prediction["kind"] == "audio" and "intent_id" in supervision:
                head_buckets[f"audio_intent_{supervision['intent_id']}"].append(
                    prediction["prediction"] == supervision["intent_id"]
                )
            elif prediction["kind"] == "ocr":
                gold = supervision["ocr_text"]
                ocr_errors += edit_distance(prediction["prediction"], gold)
                ocr_characters += len(gold)
                ocr_exact.append(prediction["prediction"] == gold)
    perception = {
        name: {"numerator": sum(values), "denominator": len(values), "rate": sum(values) / len(values)}
        for name, values in sorted(head_buckets.items())
    }
    if ocr_exact:
        perception["ocr"] = {
            "cer": ocr_errors / max(1, ocr_characters),
            "exact_numerator": sum(ocr_exact),
            "denominator": len(ocr_exact),
            "exact_rate": sum(ocr_exact) / len(ocr_exact),
        }
    pairs = collections.defaultdict(list)
    for row in rows:
        supervision = row["record"].get("supervision", {})
        pair = row["record"].get("pair_id", supervision.get("pair_id"))
        if pair:
            pairs[(supervision.get("control_type", "paired"), pair)].append(row)
    paired = collections.defaultdict(
        lambda: {"pairs": 0, "all_correct": 0, "output_changed_when_gold_changed": 0, "required_change_pairs": 0}
    )
    for (kind, _), members in pairs.items():
        if len(members) < 2:
            continue
        bucket = paired[kind]
        bucket["pairs"] += 1
        bucket["all_correct"] += all(
            member["score"]["semantic"] and member["score"].get("format") is not False for member in members
        )
        golds = {normalize(member["record"]["messages"][-1]["content"]) for member in members}
        if len(golds) > 1:
            bucket["required_change_pairs"] += 1
            bucket["output_changed_when_gold_changed"] += (
                len({normalize(member["trace"]["final_output"]) for member in members}) > 1
            )
    controls = {
        "paired": dict(paired),
        "zero_modality": {
            "count": sum("zero_output" in row for row in rows),
            "changed": sum(
                row.get("zero_output") is not None
                and normalize(row["zero_output"]) != normalize(row["trace"]["final_output"])
                for row in rows
            ),
        },
        "tool_return_dependency": {
            "count": sum("replay_score" in row for row in rows),
            "correct": sum(row.get("replay_score", False) for row in rows),
        },
    }
    route_counts = []
    for row in rows:
        for index, layer in enumerate(row.get("routing", [])):
            if layer is None:
                continue
            while len(route_counts) <= index:
                route_counts.append([0] * len(layer["counts"]))
            route_counts[index] = [a + b for a, b in zip(route_counts[index], layer["counts"])]
    routing = []
    for counts in route_counts:
        total = sum(counts)
        fractions = [count / max(1, total) for count in counts]
        routing.append(
            {
                "counts": counts,
                "fractions": fractions,
                "max_expert_fraction": max(fractions),
                "unused_experts": sum(count == 0 for count in counts),
            }
        )
    continuation_rows = [row for row in rows if "first_reply_score" in row]
    continuation = {
        "count": len(continuation_rows),
        "source_groups": len({row["record"]["group_id"] for row in continuation_rows}),
        "recording_assets": len(
            set().union(*(record_asset_paths(row["record"], "audio") for row in continuation_rows))
        ),
        "first_semantic_correct": sum(row["first_reply_score"]["semantic"] for row in continuation_rows),
        "first_format_correct": sum(row["first_reply_score"].get("format") is True for row in continuation_rows),
        "final_semantic_correct": sum(row["score"]["semantic"] for row in continuation_rows),
        "final_format_correct": sum(row["score"].get("format") is True for row in continuation_rows),
        "end_to_end_semantic_correct": sum(row["end_to_end_semantic"] for row in continuation_rows),
        "end_to_end_format_correct": sum(row["end_to_end_format"] for row in continuation_rows),
        "end_to_end_correct": sum(row["end_to_end_success"] for row in continuation_rows),
        "history_source": "actual first model generation, no demonstrated answer",
    }
    return {
        "per_task_final_reply": metrics,
        "stratified_final_reply": stratified,
        "perception": perception,
        "controls": controls,
        "routing": routing,
        "voice_topic_continuation": continuation,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--records", action="append", required=True)
    result.add_argument("--asset-dir", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--split", choices=("validation", "test"), default="validation")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--limit", type=int, help="只用 validation smoke；正式 test 不允許限制")
    result.add_argument("--controls", choices=("all", "none"), default="all")
    result.add_argument("--freeze-protocol", help="validation 完成後保存具體協定；不覆寫既有檔")
    result.add_argument("--protocol", help="test 必需的凍結 JSON")
    result.add_argument(
        "--resume-output", help="舊 outputs.jsonl；旁邊須有同條件 evaluation-receipt.json，只生成未完成列"
    )
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=2)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_new_tokens < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("generation/threads/limit 必須為正")
    if args.split == "test" and (not args.protocol or args.limit is not None or args.freeze_protocol):
        raise ValueError("test 需要凍結 protocol，不能 limit 或重新凍結")
    if args.freeze_protocol and Path(args.freeze_protocol).exists():
        raise FileExistsError("協定已存在，禁止覆寫")
    if args.freeze_protocol and args.limit is not None:
        raise ValueError("有限 smoke 不能凍結正式 test；請完成 validation")
    torch.set_num_threads(args.threads)
    from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
    from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "selftrained-random-v1":
        raise ValueError("只評估本從零管線 checkpoint")
    if checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("checkpoint preprocess_version 與目前模態前處理不同")
    records = read_records(args.records)
    current = protocol_contents(args, checkpoint, records)
    if checkpoint["data_sha256"] != current["data_sha256"] or checkpoint["asset_sha256"] != current["asset_sha256"]:
        raise ValueError("checkpoint 訓練資料/資產指紋與評估不同")
    if args.protocol:
        protocol_path = Path(args.protocol)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_protocol(protocol, current)
        # 同協定的 test 只能啟動一次；續跑須以 journal 證明已完成 prefix。
        test_marker = protocol_path.with_suffix(protocol_path.suffix + ".test-started.json")
        if args.split == "test" and test_marker.exists():
            if not args.resume_output:
                raise FileExistsError("test 已啟動；中斷時只可用 --resume-output 續跑未完成列")
            marker = json.loads(test_marker.read_text(encoding="utf-8"))
            if marker["protocol_sha256"] != file_sha256(protocol_path):
                raise ValueError("既有 test marker 協定不同")
        elif args.split == "test" and not args.resume_output:
            with test_marker.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "started_at": dt.datetime.now(dt.UTC).isoformat(),
                        "output_dir": args.output_dir,
                        "protocol_sha256": file_sha256(protocol_path),
                    },
                    handle,
                    indent=2,
                )
    selected = [record for record in records if record["split"] == args.split]
    if args.limit:
        # 依 task 輪流選，讓 CPU smoke 不被單一 task 排序占滿。
        tasks = collections.defaultdict(list)
        for record in selected:
            tasks[record["task"]].append(record)
        ordered = []
        while any(tasks.values()) and len(ordered) < args.limit:
            for task in sorted(tasks):
                if tasks[task] and len(ordered) < args.limit:
                    ordered.append(tasks[task].pop(0))
        selected = ordered
    if not selected:
        raise ValueError("沒有評估列")
    model = LimitedAssistant(SelftrainedConfig(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    encoder = RecordEncoder(tokenizer, args.asset_dir, checkpoint["config"]["max_length"], args.device)
    generator = Generator(model, encoder, args.max_new_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "outputs.jsonl"
    if output_path.exists():
        raise FileExistsError("評估 outputs 已存在，請使用新的 output-dir")
    journal = EvaluationJournal(args, current, selected)
    rows = journal.rows
    with journal:
        for record in selected[len(rows) :]:
            if journal.stop_requested:
                break
            generator.calls = []
            messages = record["messages"][:-1]
            first_score = None
            if record.get("evaluation", {}).get("mode") == "voice_topic_continuation":
                specification = record["evaluation"]
                count = specification["first_prompt_message_count"]
                continuation_index = specification["continuation_user_message_index"]
                prefix = record["messages"][:count]
                continuation = record["messages"][continuation_index]
                if not prefix or prefix[-1]["role"] != "user" or continuation["role"] != "user":
                    raise ValueError("voice continuation 必須先 user 音訊、再 user 改格式")
                first_output = generator.generate(record, prefix)
                messages = prefix + [{"role": "assistant", "content": first_output}, continuation]
                final_output = generator.generate(record, messages)
                trace = {
                    "initial_output": first_output,
                    "final_output": final_output,
                    "tool_call": None,
                    "tool_result": None,
                    "executed": False,
                    "status": "voice_topic_continuation",
                    "continuation_messages": messages,
                }
                reference_index = specification.get("first_target_message_index", count)
                first_reference = record["messages"][reference_index]
                if first_reference["role"] != "assistant":
                    raise ValueError("voice first reference 必須是 assistant，只供評分")
                first_record = {
                    **record,
                    "messages": prefix + [first_reference],
                    "supervision": {**record["supervision"], "format": specification["initial_format"]},
                }
                first_score = score_reply(first_record, first_output, {**trace, "final_output": first_output})
            else:
                trace = run_tool_loop(
                    lambda history: generator.generate(record, history),
                    messages,
                    tools_enabled=record["task"] != "tool_unavailable",
                )
            perception, routing = generator.perception(record, messages=messages)
            row = {
                "record": record,
                "trace": trace,
                "score": score_reply(record, trace["final_output"], trace),
                "model_generations": list(generator.calls),
                "perception": perception,
                "routing": routing,
            }
            if first_score is not None:
                row["first_reply_score"] = first_score
                row["end_to_end_semantic"] = bool(first_score["semantic"] and row["score"]["semantic"])
                row["end_to_end_format"] = bool(first_score["format"] is True and row["score"]["format"] is True)
                row["end_to_end_success"] = bool(
                    first_score["semantic"]
                    and first_score["format"] is not False
                    and row["score"]["semantic"]
                    and row["score"]["format"] is not False
                )
            if args.controls == "all":
                if record_asset_paths(record, "image") or record_asset_paths(record, "audio"):
                    if first_score is not None:
                        zero_first = generator.generate(record, prefix, override="zero")
                        zero_messages = prefix + [{"role": "assistant", "content": zero_first}, continuation]
                        row["zero_initial_output"] = zero_first
                        row["zero_output"] = generator.generate(record, zero_messages, override="zero")
                    else:
                        row["zero_output"] = generator.generate(record, messages, override="zero")
                if trace["executed"]:
                    returned = trace["tool_result"]["result"]
                    changed = returned + 1 if returned < 9801 else returned - 1
                    replay = {"tool": "calculator", "ok": True, "result": changed}
                    replay_messages = messages + [
                        {"role": "assistant", "content": trace["initial_output"]},
                        {"role": "tool", "content": serialize_tool_result(replay)},
                    ]
                    replay_output = generator.generate(record, replay_messages)
                    row.update(
                        replay_tool_message=replay_messages[-1],
                        replay_output=replay_output,
                        replay_score=numeric_reply_value(replay_output) == changed,
                    )
                row["control_generations"] = generator.calls[len(row["model_generations"]) :]
            journal.append(row)
    metrics = {
        "version": PROTOCOL_VERSION,
        "split": args.split,
        "count": len(rows),
        "limited_smoke": args.limit is not None,
        "evaluation_complete": len(rows) == len(selected),
        "expected_count": len(selected),
        "resumed_completed_count": len(journal.prefix.decode("utf-8").splitlines()),
        "recovered_complete_tail_count": journal.recovered_complete_tail_count,
        "interrupted": journal.stop_requested,
        "checkpoint_sha256": current["checkpoint_sha256"],
        "protocol": args.protocol,
        "teacher_forcing_used_for_generation": False,
        "thresholds": THRESHOLDS,
        **summarize(rows),
        "voice_scope": "MInDS-14 source-internal finite intent demonstration; no speaker-disjoint generalization claim",
        "notes": "Keyword/exact rubrics are mechanically scored and raw outputs remain available for review; test never selects checkpoint.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.freeze_protocol and len(rows) == len(selected):
        path = Path(args.freeze_protocol)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                {**current, "validation_metrics_sha256": file_sha256(output_dir / "metrics.json")},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


if __name__ == "__main__":
    main()
