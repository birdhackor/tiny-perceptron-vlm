"""A/B/C：真實小模型的檢索、工具協議與可驗證算術實驗。

所有「generated」欄位都來自模型抽樣；規則程式只建立訓練資料、執行工具與評分。
這些受控 ASCII 小世界讓 ByteTokenizer 的實驗能在單張 L4 的十分鐘內完成。
"""

import copy
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict

import torch
from torch import nn

from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum
from tiny_perceptron.retrieval import call_tool, retrieve
from tiny_perceptron.training import save_checkpoint

from .common import extract_asset, fit, fit_lm, new_lm, records_sha256, seed, split_records, text_examples, write_json


def _sync(ctx):
    if str(ctx.device).startswith("cuda"):
        torch.cuda.synchronize(ctx.device)


def _prompt_ids(messages, tokenizer):
    roles = {"system": tokenizer.system_id, "user": tokenizer.user_id, "assistant": tokenizer.assistant_id}
    ids = [tokenizer.bos_id]
    for message in messages:
        ids += [roles[message["role"]]] + tokenizer.encode(message["content"]) + [tokenizer.eos_id]
    return ids + [tokenizer.assistant_id]


@torch.no_grad()
def _sample(model, messages, ctx, count=1, tokens=32, temperature=0.0):
    """實際 batch 抽樣，保留 EOS、非法特殊 token 與測得的生成成本。"""
    tokenizer = ByteTokenizer()
    prompt = _prompt_ids(messages, tokenizer)
    if len(prompt) + tokens > model.config.max_length:
        raise ValueError(f"輸入 {len(prompt)} + 答案預留 {tokens} > {model.config.max_length}")
    was_training = model.training
    model.eval()
    current = torch.tensor([prompt] * count, device=ctx.device)
    active = torch.ones(count, dtype=torch.bool, device=ctx.device)
    outputs = [[] for _ in range(count)]
    state = None
    forward_tokens = 0
    _sync(ctx)
    started = time.perf_counter()
    try:
        for _ in range(tokens):
            result = model(current, cache=state)
            forward_tokens += current.numel()
            scores = result["logits"][:, -1]
            chosen = (
                scores.argmax(-1)
                if temperature <= 0
                else torch.multinomial((scores / temperature).softmax(-1), 1).squeeze(-1)
            )
            was_active = active.tolist()
            for row, (keep, token) in enumerate(zip(was_active, chosen.tolist(), strict=True)):
                if keep:
                    outputs[row].append(token)
            active = active & (chosen != tokenizer.eos_id)
            if not active.any():
                break
            current = chosen.masked_fill(~active, tokenizer.pad_id).unsqueeze(1)
            state = result["cache"]
    finally:
        model.train(was_training)
    _sync(ctx)
    seconds = time.perf_counter() - started
    samples = []
    for output in outputs:
        raw = output[:-1] if output and output[-1] == tokenizer.eos_id else output
        samples.append(
            {
                "generated": tokenizer.decode(raw),
                "generated_ids": output,
                "eos": bool(output and output[-1] == tokenizer.eos_id),
                "invalid_special_tokens": [token for token in raw if token < 8],
                "generated_tokens": len(output),
            }
        )
    return {
        # 外層協議會繼續加入請求與工具回傳；此欄必須保留本次生成時的輸入。
        "messages": copy.deepcopy(messages),
        "input_ids": prompt,
        "input_tokens": len(prompt),
        "samples": samples,
        "candidate_count": count,
        "temperature": temperature,
        "max_new_tokens": tokens,
        "generated_tokens": sum(len(output) for output in outputs),
        "forward_input_tokens": forward_tokens,
        "seconds": seconds,
    }


def _rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _split_summary(splits):
    return {
        name: {
            "records": len(rows),
            "families": len({row["family"] for row in rows}),
            "sha256": records_sha256(rows),
        }
        for name, rows in splits.items()
    }


def _conversation(question, answer):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def _training_steps(ctx, planned):
    scale = float(getattr(ctx, "step_scale", 1.0))
    if scale <= 0 or scale > 1:
        raise ValueError("step_scale 必須在 (0, 1]；小於 1 只作流程 smoke")
    return max(1, round(planned * scale)), scale


def _fit_lm(model, records, ctx, *, steps, **kwargs):
    actual, scale = _training_steps(ctx, steps)
    return {**fit_lm(model, records, ctx, steps=actual, **kwargs), "planned_steps": steps, "step_scale": scale}


def _fit(model, loss_fn, ctx, *, steps, **kwargs):
    actual, scale = _training_steps(ctx, steps)
    return {**fit(model, loss_fn, ctx, steps=actual, **kwargs), "planned_steps": steps, "step_scale": scale}


@torch.no_grad()
def _nll_examples(model, examples, ctx):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for start in range(0, len(examples), 24):
            x, labels, valid = (tensor.to(ctx.device) for tensor in pad_batch(examples[start : start + 24]))
            loss, amount = loss_sum(model(x, valid=valid)["logits"], labels)
            total += float(loss)
            count += int(amount)
    finally:
        model.train(was_training)
    return {"nll": total / count, "nll_sum": total, "effective_tokens": count, "examples": len(examples)}


def _sft_nll(model, rows, ctx):
    return _nll_examples(model, text_examples(rows, mode="sft", max_length=model.config.max_length), ctx)


def _rag_question(key, documents):
    context = "\n".join(f"[{document['id']}] {document['text']}" for document in documents) or "EMPTY"
    return f"Docs:\n{context}\nFind:{key}\nReply:address[source] or UNKNOWN"


def _rag_records(seed_value):
    rng = random.Random(seed_value)
    records = []
    # 同一店名的全部地址、來源及提示變形共用 family，避免逐筆切分洩漏。
    for index in range(120):
        key = f"K{index:03d}"
        for variant in range(8):
            address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
            source = f"D{rng.randrange(10)}"
            correct = {"id": source, "text": f"{key} address={address}"}
            distractor = {
                "id": f"D{(int(source[1:]) + 1) % 10}",
                "text": f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}",
            }
            documents = [correct] if variant < 2 else [correct, distractor]
            if variant == 6:
                documents = []
            elif variant == 7:
                documents = [distractor]
            elif variant % 2:
                documents.reverse()
            answer = f"{address}[{source}]" if variant < 6 else "UNKNOWN"
            records.append(
                {
                    "family": key,
                    "key": key,
                    "address": address,
                    "source": source,
                    "documents": documents,
                    "variant": variant,
                    "messages": _conversation(_rag_question(key, documents), answer),
                }
            )
    return records


def _rag_splits(seed_value):
    splits = split_records(_rag_records(seed_value), seed_value)
    for records in splits.values():
        keys = sorted({row["key"] for row in records})
        for row in records:
            # 干擾店也使用 Kxxx；只從同 split 選店，不能把 heldout 店放進訓練。
            unrelated = keys[(keys.index(row["key"]) + 1) % len(keys)]
            for document in row["documents"]:
                if document["text"].startswith("Z"):
                    document["text"] = unrelated + document["text"][4:]
            row["messages"][0]["content"] = _rag_question(row["key"], row["documents"])
    return splits


def _icl_question(value, offset=None, reversed_examples=False):
    if offset is None:
        examples = "EMPTY"
    else:
        pairs = [f"0->{offset}", f"1->{(1 + offset) % 10}"]
        if reversed_examples:
            pairs.reverse()
        examples = ";".join(pairs)
    return f"Examples:{examples}\nMap:{value}\nReply:digit or UNKNOWN"


def _icl_records():
    records = []
    for value in range(2, 10):
        for offset in range(10):
            for reversed_examples in (False, True):
                records.append(
                    {
                        "family": f"map:{value}",
                        "value": value,
                        "offset": offset,
                        "messages": _conversation(
                            _icl_question(value, offset, reversed_examples), str((value + offset) % 10)
                        ),
                    }
                )
        records.append(
            {
                "family": f"map:{value}",
                "value": value,
                "offset": None,
                "messages": _conversation(_icl_question(value), "UNKNOWN"),
            }
        )
    return records


def _grounding(sample, documents, expected):
    text = sample["generated"]
    parsed = re.fullmatch(r"([A-E][0-9])\[(D[0-9])\]", text)
    citation = parsed[2] if parsed else None
    address = parsed[1] if parsed else None
    valid_ids = {document["id"] for document in documents}
    supported = bool(
        parsed
        and any(document["id"] == citation and f"address={address}" in document["text"] for document in documents)
    )
    clean = not sample["invalid_special_tokens"]
    return {
        "expected": expected,
        "exact_match": clean and text == expected,
        "parsed_address": address,
        "parsed_citation": citation,
        "citation_valid": clean and citation in valid_ids,
        "supported_by_cited_source": clean and supported,
        "unknown": clean and text == "UNKNOWN",
    }


def _rag_counterfactual_contexts(fact):
    """只修改這次公告的地址或來源；原始 split 與事實物件保持不動。"""
    changed_address = fact["address"][0] + str((int(fact["address"][1]) + 1) % 10)
    changed_source = "D" + str((int(fact["source"][1:]) + 1) % 10)
    changed = {
        "changed_address_context": {**fact, "address": changed_address},
        "changed_source_context": {**fact, "source": changed_source},
    }
    return {
        mode: {
            "fact": current,
            "documents": [{"id": current["source"], "text": f"{current['key']} address={current['address']}"}],
        }
        for mode, current in changed.items()
    }


def run_rag(ctx):
    """同一權重閉卷／正確文件／實際檢索／干擾文件比較。"""
    started = time.perf_counter()
    seed(ctx.seed)
    ctx.output.mkdir(parents=True, exist_ok=True)
    splits = _rag_splits(ctx.seed)
    write_json(ctx.output / "dataset.json", splits)
    icl_splits = split_records(_icl_records(), ctx.seed)
    write_json(ctx.output / "icl-dataset.json", icl_splits)
    model = new_lm(ctx, width=64, layers=2, max_length=160, heads=2, backend="sdpa")
    training = _fit_lm(model, splits["train"] + icl_splits["train"], ctx, mode="sft", steps=1000, batch_size=24)
    rng = random.Random(ctx.seed + 100)
    heldout_keys = sorted({row["key"] for row in splits["test"]})
    facts = []
    for key in heldout_keys:
        # 測試時再建立新地址；模型不可能在訓練答案看過這份 key/value 配對。
        source = f"D{rng.randrange(10)}"
        address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
        facts.append({"family": key, "key": key, "address": address, "source": source})
    corpus = [{"id": f"doc-{row['key']}", "text": f"{row['key']} address={row['address']}"} for row in facts]
    corpus += [
        {"id": f"noise-{index}", "text": f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"}
        for index in range(60)
    ]
    write_json(ctx.output / "retrieval-corpus.json", corpus)
    rows = []
    for index, fact in enumerate(facts):
        key, address, source = fact["key"], fact["address"], fact["source"]
        correct = {"id": source, "text": f"{key} address={address}"}
        distractor = {
            "id": f"D{(int(source[1:]) + 1) % 10}",
            "text": f"{facts[(index + 1) % len(facts)]['key']} address={rng.choice('ABCD')}{rng.randrange(10)}",
        }
        # 四分之一改用這個字詞檢索器不懂的別名，留下真正 retrieval miss。
        query = key if index % 4 else key.replace("K", "shop")
        hits = retrieve(query, corpus, k=1)
        retrieved = [{"id": source, "text": hit["text"]} for hit in hits]
        # corpus 的持久 id 與模型使用的短引用 id 的映射明列在結果。
        modes = {
            "without_context": [],
            "correct_context": [correct],
            "retrieved_context": retrieved,
            "distractor_only": [distractor],
            "correct_with_distractor": [distractor, correct],
        }
        interventions = _rag_counterfactual_contexts(fact)
        modes.update({mode: case["documents"] for mode, case in interventions.items()})
        for mode, documents in modes.items():
            context_fact = interventions[mode]["fact"] if mode in interventions else fact
            fact_answer = f"{context_fact['address']}[{context_fact['source']}]"
            expected = (
                "UNKNOWN"
                if mode in ("without_context", "distractor_only") or not hits and mode == "retrieved_context"
                else fact_answer
            )
            generation = _sample(model, [{"role": "user", "content": _rag_question(key, documents)}], ctx, tokens=16)
            sample = generation["samples"][0]
            rows.append(
                {
                    "family": key,
                    "mode": mode,
                    "fact": fact,
                    "context_fact": context_fact,
                    "paired_baseline_mode": "correct_context" if mode in interventions else None,
                    "query": query,
                    "retrieved_ids": [hit["id"] for hit in hits],
                    "retrieval_hit": any(hit["id"] == f"doc-{key}" for hit in hits),
                    "citation_id_map": {hit["id"]: source for hit in hits},
                    "documents": documents,
                    "generation": generation,
                    **_grounding(sample, documents, expected),
                    "fact_answer_correct": sample["generated"] == fact_answer and not sample["invalid_special_tokens"],
                }
            )
    metrics = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        metrics[mode] = {
            field: _rate(sum(row[field] for row in selected), len(selected))
            for field in (
                "exact_match",
                "fact_answer_correct",
                "citation_valid",
                "supported_by_cited_source",
                "unknown",
            )
        }
        metrics[mode]["generated_tokens"] = sum(row["generation"]["generated_tokens"] for row in selected)
        metrics[mode]["seconds"] = sum(row["generation"]["seconds"] for row in selected)
        metrics[mode]["eos_rate"] = _rate(
            sum(row["generation"]["samples"][0]["eos"] for row in selected), len(selected)
        )
        metrics[mode]["invalid_control_rate"] = _rate(
            sum(bool(row["generation"]["samples"][0]["invalid_special_tokens"]) for row in selected), len(selected)
        )
    baseline = {row["family"]: row for row in rows if row["mode"] == "correct_context"}
    paired = {}
    for mode in ("changed_address_context", "changed_source_context"):
        changed = [row for row in rows if row["mode"] == mode]
        paired[mode] = {
            "both_answers_correct": _rate(
                sum(row["fact_answer_correct"] and baseline[row["family"]]["fact_answer_correct"] for row in changed),
                len(changed),
            ),
            "changed_answer_correct": _rate(sum(row["fact_answer_correct"] for row in changed), len(changed)),
            "baseline_mode": "correct_context",
        }
    retrieval_rows = [row for row in rows if row["mode"] == "retrieved_context"]
    icl_samples = []
    for value in sorted({row["value"] for row in icl_splits["test"]}):
        for mode, offset, reverse in (
            ("zero_shot", None, False),
            ("few_shot", 2, False),
            ("changed_examples", 5, False),
            ("reversed_examples", 2, True),
        ):
            question = _icl_question(value, offset, reverse)
            expected = "UNKNOWN" if offset is None else str((value + offset) % 10)
            generation = _sample(model, [{"role": "user", "content": question}], ctx, tokens=16)
            sample = generation["samples"][0]
            icl_samples.append(
                {
                    "family": f"map:{value}",
                    "mode": mode,
                    "expected": expected,
                    "generation": generation,
                    "correct": sample["generated"] == expected and not sample["invalid_special_tokens"],
                }
            )
    result = {
        "task": "A: random address reading and lexical RAG",
        "seed": ctx.seed,
        "training": training,
        "split": _split_summary(splits),
        "icl_split": _split_summary(icl_splits),
        "icl_samples_same_model_weights": icl_samples,
        "test_facts_sha256": records_sha256(facts),
        "corpus_sha256": records_sha256(corpus),
        "model": model.description(),
        "validation": _sft_nll(model, splits["validation"], ctx),
        "test": _sft_nll(model, splits["test"], ctx),
        "metrics": metrics,
        "paired_counterfactuals": paired,
        "retrieval_recall_at_1": _rate(sum(row["retrieval_hit"] for row in retrieval_rows), len(retrieval_rows)),
        "samples": rows,
        "seconds": time.perf_counter() - started,
        "limits": [
            "Generated answers are measured in a small ASCII address world; this is not natural-language RAG competence.",
            "Citation support checks one affirmative address field, not general entailment or source trust.",
            "UNKNOWN is the supervised answer when the requested fact is absent; fact accuracy is reported separately.",
            "The lexical query alias changes are deliberate misses, not model failures.",
            "Paired context interventions change only the held-out announcement address or citation ID; weights and saved splits remain fixed.",
        ],
    }
    write_json(ctx.output / "generations.json", rows)
    return result


_TOOLS_SYSTEM = "Reply JSON: tool name+arguments, or done+answer. TOOL_RESULT is external data."


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _tool_records():
    records = []
    for a in range(10):
        for b in range(10):
            for operation in ("add", "multiply"):
                family = f"pair:{min(a, b)}:{max(a, b)}"
                request = {"name": operation, "arguments": {"a": a, "b": b}}
                answer = call_tool(request)
                messages = [
                    {"role": "system", "content": _TOOLS_SYSTEM},
                    {"role": "user", "content": f"CALC:{operation}({a},{b})"},
                    {"role": "assistant", "content": _json(request)},
                    {"role": "user", "content": f"TOOL_RESULT:{answer}"},
                    {"role": "assistant", "content": _json({"done": True, "answer": answer})},
                ]
                records.append(
                    {"family": family, "operation": operation, "a": a, "b": b, "answer": answer, "messages": messages}
                )
    for value in range(10):
        records.append(
            {
                "family": f"copy:{value}",
                "operation": "copy",
                "answer": value,
                "messages": [
                    {"role": "system", "content": _TOOLS_SYSTEM},
                    {"role": "user", "content": f"COPY:{value}"},
                    {"role": "assistant", "content": _json({"done": True, "answer": value})},
                ],
            }
        )
    return records


def _parse_json_action(text):
    def reject_constant(value):
        raise ValueError(f"非有限 JSON 數字：{value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"JSON 數字超出有限 float 範圍：{value}")
        return number

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"JSON object 含重複欄位：{key}")
            value[key] = item
        return value

    action = json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_object)
    if not isinstance(action, dict):
        raise ValueError("動作最外層必須是 JSON object")
    if "done" in action:
        if set(action) != {"done", "answer"} or action["done"] is not True:
            raise ValueError("完成動作必須恰有 done=true 與 answer")
        if type(action["answer"]) not in (int, float, str):
            raise ValueError("answer 必須是數字或文字")
        if isinstance(action["answer"], float) and not math.isfinite(action["answer"]):
            raise ValueError("answer 必須有限")
    return action


def _tool_episode(model, record, ctx, max_steps=3):
    messages = copy.deepcopy(record["messages"][:2])
    trace = []
    answer = None
    status = "step_limit"
    started = time.perf_counter()
    for step in range(max_steps):
        try:
            generation = _sample(model, messages, ctx, tokens=64)
        except ValueError as error:
            status = "context_limit"
            trace.append({"step": step + 1, "error": str(error), "messages": copy.deepcopy(messages)})
            break
        raw = generation["samples"][0]
        event = {"step": step + 1, "generation": generation, "executed": False}
        trace.append(event)
        try:
            if raw["invalid_special_tokens"]:
                raise ValueError("模型產生未允許的特殊 token")
            action = _parse_json_action(raw["generated"])
            event["parsed"] = action
            if "done" in action:
                status, answer = "done", action["answer"]
                event["done"] = True
                break
            # 唯一真正執行工具的位置；參數不從標準答案替換或修正。
            result = call_tool(action)
            finite_result = type(result) is not float or math.isfinite(result)
            event.update(
                {
                    "executed": True,
                    "finite_result": finite_result,
                    "tool_result": result if finite_result else repr(result),
                }
            )
            event["request_matches_question"] = record["operation"] != "copy" and action == {
                "name": record["operation"],
                "arguments": {"a": record["a"], "b": record["b"]},
            }
            if not finite_result:
                event["error"] = "工具確實已執行，但運算 overflow 產生非有限結果；保留文字證據並停止。"
                status = "invalid_tool_result"
                break
            messages += [
                {"role": "assistant", "content": raw["generated"]},
                {"role": "user", "content": f"TOOL_RESULT:{result}"},
            ]
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            event["error"] = str(error)
            status = "invalid_request"
            break
    completed_answer_correct = status == "done" and type(answer) in (int, float) and answer == record["answer"]
    executions = [event for event in trace if event.get("executed")]
    required_call_satisfied = (
        not executions
        if record["operation"] == "copy"
        else bool(
            executions
            and executions[-1]["finite_result"]
            and any(event["request_matches_question"] for event in executions)
            and answer == executions[-1]["tool_result"]
        )
    )
    return {
        "family": record["family"],
        "operation": record["operation"],
        "question": record["messages"][1]["content"],
        "expected": record["answer"],
        "status": status,
        "answer": answer,
        "answer_correct": completed_answer_correct,
        "required_tool_behavior": required_call_satisfied,
        "correct": completed_answer_correct and required_call_satisfied,
        "trace": trace,
        "max_steps": max_steps,
        "actual_tool_calls": sum(event["executed"] for event in trace if "executed" in event),
        "generated_tokens": sum(event["generation"]["generated_tokens"] for event in trace if "generation" in event),
        "seconds": time.perf_counter() - started,
    }


def run_tools(ctx):
    """模型每步真實產生 JSON，validate→執行→回填→真實 final。"""
    started = time.perf_counter()
    seed(ctx.seed)
    ctx.output.mkdir(parents=True, exist_ok=True)
    splits = split_records(_tool_records(), ctx.seed)
    write_json(ctx.output / "dataset.json", splits)
    model = new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
    training = _fit_lm(model, splits["train"], ctx, mode="sft", steps=1200, batch_size=16)
    episodes = [_tool_episode(model, row, ctx) for row in splits["test"]]
    capped = [_tool_episode(model, row, ctx, max_steps=1) for row in splits["test"] if row["operation"] != "copy"]
    # 這組是外層協議的故障注入；和上方模型成績分開保存。
    injected = []
    for raw in (
        "{",
        "[]",
        '{"name":"delete_all","arguments":{"a":2,"b":3}}',
        '{"name":"add","arguments":{"a":true,"b":3}}',
        '{"name":"add","arguments":{"a":NaN,"b":3}}',
        '{"name":"add","arguments":{"a":1e999,"b":3}}',
        '{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}',
    ):
        try:
            action = _parse_json_action(raw)
            value = call_tool(action)
            injected.append({"input": raw, "rejected": False, "result": value})
        except (ValueError, TypeError, KeyError) as error:
            injected.append({"input": raw, "rejected": True, "error": str(error)})
    actual_calls = [event for episode in episodes for event in episode["trace"] if event.get("executed")]
    result = {
        "task": "B: generated JSON calculator requests and completions",
        "seed": ctx.seed,
        "training": training,
        "split": _split_summary(splits),
        "model": model.description(),
        "validation": _sft_nll(model, splits["validation"], ctx),
        "test": _sft_nll(model, splits["test"], ctx),
        "protocol": {
            "roles": ["system", "user", "assistant"],
            "external_result_serialization": "user content TOOL_RESULT:<number>",
            "reason": "render_chat supports no tool role; the system and training data declare this explicit protocol.",
            "allowlist": ["add", "multiply"],
            "final": {"done": True, "answer": "number"},
            "max_steps_including_done": 3,
        },
        "accuracy": _rate(sum(episode["correct"] for episode in episodes), len(episodes)),
        "answer_accuracy": _rate(sum(episode["answer_correct"] for episode in episodes), len(episodes)),
        "completion_rate": _rate(sum(episode["status"] == "done" for episode in episodes), len(episodes)),
        "first_action_json_rate": _rate(
            sum(bool(episode["trace"] and "parsed" in episode["trace"][0]) for episode in episodes), len(episodes)
        ),
        "correct_parameters_given_executed_call": _rate(
            sum(event["request_matches_question"] for event in actual_calls), len(actual_calls)
        ),
        "status_counts": dict(Counter(episode["status"] for episode in episodes)),
        "actual_tool_calls": sum(episode["actual_tool_calls"] for episode in episodes),
        "generated_tokens": sum(episode["generated_tokens"] for episode in episodes + capped),
        "samples": episodes,
        "one_step_cap_samples": capped,
        "injected_protocol_checks_not_model_scores": injected,
        "seconds": time.perf_counter() - started,
        "limits": [
            "Arithmetic operand-pair families, including reversed operands and both operations, remain in one split.",
            "done is a protocol signal; correctness is independently compared against the original question.",
            "The finite local allowlist does not establish general agent or arbitrary-code execution ability.",
        ],
    }
    write_json(ctx.output / "generations.json", {"episodes": episodes, "one_step_cap": capped, "injected": injected})
    return result


def _reasoning_records():
    rows = []
    for a in range(6):
        for b in range(6):
            for c in range(6):
                rows.append(
                    {
                        # 相同數字的所有排列共用 family，測試沒有訓練題的重新排列。
                        "family": ":".join(map(str, sorted((a, b, c)))),
                        "a": a,
                        "b": b,
                        "c": c,
                        "question": f"({a}+{b})+{c}=?",
                        "truth": a + b + c,
                    }
                )
    return rows


def _reasoning_sft(rows, mode):
    result = []
    for row in rows:
        a, b, c, truth = row["a"], row["b"], row["c"], row["truth"]
        answer = str(truth) if mode == "direct" else f"{a}+{b}={a + b};{a + b}+{c}={truth};answer={truth}"
        result.append({**row, "messages": _conversation(row["question"], answer)})
    return result


def _verify_reasoning(sample, row, mode):
    text = sample["generated"].strip()
    clean = not sample["invalid_special_tokens"]
    if mode == "direct":
        parsed = int(text) if clean and re.fullmatch(r"-?[0-9]+", text) else None
        return {
            "final_answer": parsed,
            "parsed": parsed is not None,
            "final_correct": parsed == row["truth"],
            "fully_verified": parsed == row["truth"],
        }
    matched = re.fullmatch(
        r"(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)", text
    )
    if not matched or not clean:
        # 結尾格式可獨立解析，讓「最終對但過程錯」保留在資料中。
        final_match = re.search(r";answer=(-?[0-9]+)$", text) if clean else None
        parsed = int(final_match[1]) if final_match else None
        return {
            "final_answer": parsed,
            "parsed": False,
            "final_correct": parsed == row["truth"],
            "equations_valid": False,
            "linked": False,
            "task_operands_valid": False,
            "fully_verified": False,
        }
    a, b, subtotal, previous, c, total, final = map(int, matched.groups())
    equations = a + b == subtotal and previous + c == total
    linked = previous == subtotal and total == final
    task = (a, b, c) == (row["a"], row["b"], row["c"])
    return {
        "final_answer": final,
        "parsed": True,
        "steps": [[a, b, subtotal], [previous, c, total]],
        "final_correct": final == row["truth"],
        "equations_valid": equations,
        "linked": linked,
        "task_operands_valid": task,
        "fully_verified": equations and linked and task and final == row["truth"],
    }


def _reasoning_samples(model, rows, ctx, mode):
    budgets = []
    for candidate_count in (1, 2, 4, 8):
        samples = []
        for row in rows:
            generation = _sample(
                model,
                [{"role": "user", "content": row["question"]}],
                ctx,
                count=candidate_count,
                tokens=8 if mode == "direct" else 48,
                temperature=0.7,
            )
            candidates = [{**sample, **_verify_reasoning(sample, row, mode)} for sample in generation["samples"]]
            answers = [candidate["final_answer"] for candidate in candidates if candidate["final_answer"] is not None]
            majority = Counter(answers).most_common(1)[0][0] if answers else None
            verified = next(
                (candidate["final_answer"] for candidate in candidates if candidate["fully_verified"]), None
            )
            samples.append(
                {
                    **row,
                    "generation": {key: value for key, value in generation.items() if key != "samples"},
                    "candidates": candidates,
                    "oracle_coverage": any(candidate["final_correct"] for candidate in candidates),
                    "majority_answer": majority,
                    "majority_correct": majority == row["truth"],
                    "verified_answer": verified,
                    "verifier_correct": verified == row["truth"],
                    "selection_failed_despite_coverage": any(candidate["final_correct"] for candidate in candidates)
                    and majority != row["truth"],
                }
            )
        candidates = [candidate for sample in samples for candidate in sample["candidates"]]
        covered = [sample for sample in samples if sample["oracle_coverage"]]
        budgets.append(
            {
                "candidate_count": candidate_count,
                "oracle_coverage": _rate(len(covered), len(samples)),
                "majority_accuracy": _rate(sum(sample["majority_correct"] for sample in samples), len(samples)),
                "verifier_accuracy": _rate(sum(sample["verifier_correct"] for sample in samples), len(samples)),
                "majority_accuracy_given_coverage": _rate(
                    sum(sample["majority_correct"] for sample in covered), len(covered)
                ),
                "parsed_candidates": _rate(sum(candidate["parsed"] for candidate in candidates), len(candidates)),
                "candidate_final_accuracy": _rate(
                    sum(candidate["final_correct"] for candidate in candidates), len(candidates)
                ),
                "fully_verified_candidates": _rate(
                    sum(candidate["fully_verified"] for candidate in candidates), len(candidates)
                ),
                "final_correct_but_steps_invalid": _rate(
                    sum(candidate["final_correct"] and not candidate["fully_verified"] for candidate in candidates),
                    len(candidates),
                ),
                "generated_tokens": sum(sample["generation"]["generated_tokens"] for sample in samples),
                "forward_input_tokens": sum(sample["generation"]["forward_input_tokens"] for sample in samples),
                "generation_seconds": sum(sample["generation"]["seconds"] for sample in samples),
                "samples": samples,
            }
        )
    return budgets


class _FinitePolicy(nn.Module):
    """只有 17 種文字動作的小策略；不是把語言模型的 CoT 宣稱成 RLVR。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(18, 48), nn.Tanh(), nn.Linear(48, 17))

    def forward(self, features):
        return self.net(features)


def _policy_features(rows, ctx):
    features = torch.zeros(len(rows), 18, device=ctx.device)
    for i, row in enumerate(rows):
        for offset, key in ((0, "a"), (6, "b"), (12, "c")):
            features[i, offset + row[key]] = 1
    return features


_POLICY_ACTIONS = [str(number) for number in range(16)] + [" ".join(map(str, range(16)))]


def _policy_reward(action, truth, proxy=False):
    text = _POLICY_ACTIONS[action]
    if proxy:
        # 不良評分器只掃任何一個正確數字，列出所有答案即得分。
        return float(str(truth) in text.split())
    return float(bool(re.fullmatch(r"-?[0-9]+", text)) and int(text) == truth)


@torch.no_grad()
def _evaluate_policy(model, rows, ctx):
    logits = model(_policy_features(rows, ctx))
    probabilities = logits.softmax(-1)
    sampled = torch.multinomial(probabilities, 16, replacement=True).tolist()
    greedy = logits.argmax(-1).tolist()
    traces = []
    for row, actions, chosen, probs in zip(rows, sampled, greedy, probabilities.tolist(), strict=True):
        traces.append(
            {
                **row,
                "probabilities": probs,
                "greedy_action": _POLICY_ACTIONS[chosen],
                "greedy_strict_correct": bool(_policy_reward(chosen, row["truth"])),
                "samples": [
                    {
                        "generated": _POLICY_ACTIONS[action],
                        "action_id": action,
                        "strict_reward": _policy_reward(action, row["truth"]),
                        "proxy_reward": _policy_reward(action, row["truth"], proxy=True),
                    }
                    for action in actions
                ],
            }
        )
    candidates = [candidate for trace in traces for candidate in trace["samples"]]
    return {
        "greedy_accuracy": _rate(sum(trace["greedy_strict_correct"] for trace in traces), len(traces)),
        "sample_accuracy": _rate(sum(int(candidate["strict_reward"]) for candidate in candidates), len(candidates)),
        "sample_proxy_reward": _rate(sum(int(candidate["proxy_reward"]) for candidate in candidates), len(candidates)),
        "enumeration_action_rate": _rate(
            sum(candidate["action_id"] == 16 for candidate in candidates), len(candidates)
        ),
        "expected_strict_accuracy": sum(probabilities[i, row["truth"]].item() for i, row in enumerate(rows))
        / len(rows),
        "samples": traces,
    }


def _reinforce(splits, ctx):
    seed(ctx.seed)
    base = _FinitePolicy().to(ctx.device)
    initial = copy.deepcopy(base.state_dict())
    outcome = {}
    for label, proxy in (("strict", False), ("weak_proxy", True)):
        seed(ctx.seed)
        policy = _FinitePolicy().to(ctx.device)
        policy.load_state_dict(initial)
        before = _evaluate_policy(policy, splits["test"], ctx)
        rng = random.Random(ctx.seed)
        baseline = 0.0
        rewards = []
        sampled_actions = 0

        def loss_fn(step):
            nonlocal baseline, sampled_actions
            batch = rng.choices(splits["train"], k=64)
            distribution = torch.distributions.Categorical(logits=policy(_policy_features(batch, ctx)))
            actions = distribution.sample()
            reward = torch.tensor(
                [
                    _policy_reward(action, row["truth"], proxy)
                    for action, row in zip(actions.tolist(), batch, strict=True)
                ],
                device=ctx.device,
            )
            advantage = reward - baseline
            baseline = 0.95 * baseline + 0.05 * float(reward.mean())
            sampled_actions += len(batch)
            if step == 0 or (step + 1) % 100 == 0:
                rewards.append(
                    {"step": step + 1, "sampled_mean_reward": float(reward.mean()), "moving_baseline": baseline}
                )
            return -(advantage.detach() * distribution.log_prob(actions)).mean() - 0.005 * distribution.entropy().mean()

        training = _fit(
            policy,
            loss_fn,
            ctx,
            steps=1200,
            lr=0.003,
            name=f"policy-{label}",
            metadata={
                "actions": _POLICY_ACTIONS,
                "reward": label,
                "input_features": "three six-way one-hot operands",
                "train_sha256": records_sha256(splits["train"]),
                "seed": ctx.seed,
            },
        )
        outcome[label] = {
            "training": {
                **training,
                "effective_tokens": 0,
                "sampled_actions": sampled_actions,
                "reward_history": rewards,
            },
            "before": before,
            "after": _evaluate_policy(policy, splits["test"], ctx),
        }
    outcome["limits"] = [
        "This REINFORCE updates a finite 17-action policy, not the language models above.",
        "An enumeration action exposes reward hacking in the weak any-number parser; it is never counted as a strict correct answer.",
        "Sampled actions and token counts have different units; this policy generates zero autoregressive tokens.",
    ]
    return outcome


def _gsm8k_pilot(ctx):
    root = extract_asset(ctx, "gsm8k")
    path = root / "reasoning-initial" / "gsm8k-train-first200.jsonl"
    # 資料包可能保留 data/training 前綴；只在已驗證的解包目錄找同名檔。
    if not path.is_file():
        candidates = list(root.rglob("gsm8k-train-first200.jsonl"))
        if len(candidates) != 1:
            raise FileNotFoundError("reasoning-initial 沒有唯一 GSM8K training pilot")
        path = candidates[0]
    raw_bytes = path.read_bytes()
    original = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    records = [
        {
            "family": hashlib.sha256(row["question"].encode()).hexdigest(),
            "source_index": i,
            "provenance": "GSM8K original human-authored training answer; not teacher generation",
            "messages": _conversation(row["question"], row["answer"]),
        }
        for i, row in enumerate(original)
    ]
    splits = split_records(records, ctx.seed)
    write_json(ctx.output / "gsm8k-dataset.json", splits)
    tokenizer = ByteTokenizer()
    examples = {}
    packing = {}
    for name, rows in splits.items():
        examples[name] = []
        metadata = []
        skipped = 0
        for row in rows:
            x, labels = render_chat(row["messages"], tokenizer)
            for start in range(0, len(x), 128):
                segment = (x[start : start + 128], labels[start : start + 128])
                count = int((segment[1] != IGNORE).sum())
                if not count:
                    skipped += 1
                    continue
                examples[name].append(segment)
                metadata.append(
                    {
                        "family": row["family"],
                        "source_index": row["source_index"],
                        "start": start,
                        "length": len(segment[0]),
                        "effective_tokens": count,
                    }
                )
        packing[name] = {
            "segments": metadata,
            "question_only_segments_skipped": skipped,
            "effective_tokens": sum(row["effective_tokens"] for row in metadata),
        }
    write_json(ctx.output / "gsm8k-packing.json", packing)
    model = new_lm(ctx, width=32, layers=1, max_length=128, heads=1, backend="sdpa")
    initial = _nll_examples(model, examples["train"], ctx)
    sampler = random.Random(ctx.seed)
    used_tokens = 0

    def loss_fn(_step):
        nonlocal used_tokens
        batch = sampler.choices(examples["train"], k=16)
        x, labels, valid = (tensor.to(ctx.device) for tensor in pad_batch(batch))
        loss, count = loss_sum(model(x, valid=valid)["logits"], labels)
        used_tokens += int(count)
        return loss / count

    training = _fit(
        model,
        loss_fn,
        ctx,
        steps=150,
        name="gsm8k-pilot",
        metadata={
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "human_targets": True,
            "packing": "128-token segments with reset context; no target token truncated",
        },
    )
    final = _nll_examples(model, examples["train"], ctx)
    # 只續寫 heldout 的人寫解答前綴，不當成獨立解題成績。
    continuations = []
    for row in splits["test"]:
        prefix = row["messages"][1]["content"].encode()[:40].decode("utf-8", errors="ignore")
        generation = _sample(model, [{"role": "user", "content": f"Continue human answer: {prefix}"}], ctx, tokens=32)
        continuations.append(
            {
                "family": row["family"],
                "prompt_mode": "human-answer prefix continuation, not problem solving",
                "generation": generation,
            }
        )
    return {
        "source": "OpenAI GSM8K train rows 0-199, commit 3101c7d5072418e28b9008a6636bde82a006892c",
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "records": len(original),
        "provenance": "complete original human questions and worked answers, not a teacher model",
        "split": _split_summary(splits),
        "training": {
            **training,
            "initial_loss": initial["nll"],
            "final_loss": final["nll"],
            "effective_tokens": used_tokens,
        },
        "validation": _nll_examples(model, examples["validation"], ctx),
        "test": _nll_examples(model, examples["test"], ctx),
        "packing": {
            name: {key: value for key, value in data.items() if key != "segments"} | {"segments": len(data["segments"])}
            for name, data in packing.items()
        },
        "continuations": continuations,
        "limits": [
            "All 200 original records are preserved; family splitting happens before context-reset segmentation.",
            "All human answer target tokens occur once in the segment dataset; question-only chunks have no supervised target and are omitted.",
            "Context resets inside long records make this a loss/dataflow pilot, not a GSM8K benchmark or demonstrated complex reasoning.",
            "The continuation prompts expose human-answer prefixes and are explicitly excluded from answer-accuracy scoring.",
        ],
    }


def run_reasoning(ctx):
    """同初始權重的短答／可驗證步驟，真實多候選，有限策略 REINFORCE。"""
    started = time.perf_counter()
    seed(ctx.seed)
    ctx.output.mkdir(parents=True, exist_ok=True)
    splits = split_records(_reasoning_records(), ctx.seed)
    write_json(ctx.output / "dataset.json", splits)
    base = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend="sdpa")
    base_state = copy.deepcopy(base.state_dict())
    save_checkpoint(
        ctx.output / "same-base.pt",
        base,
        step=0,
        metadata={"seed": ctx.seed, "purpose": "identical direct/steps initialization"},
    )
    base_hash = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in base_state.values())
    ).hexdigest()
    models = {}
    for mode in ("direct", "steps"):
        seed(ctx.seed)
        model = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend="sdpa")
        model.load_state_dict(base_state)
        mode_splits = {name: _reasoning_sft(rows, mode) for name, rows in splits.items()}
        write_json(ctx.output / f"{mode}-dataset.json", mode_splits)
        training = _fit_lm(model, mode_splits["train"], ctx, mode="sft", steps=900, batch_size=24, name=mode)
        models[mode] = {
            "base_state_sha256": base_hash,
            "training": training,
            "validation": _sft_nll(model, mode_splits["validation"], ctx),
            "test": _sft_nll(model, mode_splits["test"], ctx),
            "budgets": _reasoning_samples(model, splits["test"], ctx, mode),
        }
    result = {
        "task": "C: controlled two-addition arithmetic, candidate selection, and finite-policy RLVR",
        "seed": ctx.seed,
        "split": _split_summary(splits),
        "model_config": asdict(base.config),
        "comparison": models,
        "reinforce": _reinforce(splits, ctx),
        "gsm8k_training_pilot": _gsm8k_pilot(ctx),
        "seconds": time.perf_counter() - started,
        "limits": [
            "Direct and step models start from identical random weights and receive equal optimizer steps and sampled example counts, not equal supervised-token budgets.",
            "Each N budget really samples a separate batch; candidate generations and measured batch times are preserved.",
            "Oracle coverage is scored using known arithmetic answers; it is not a deployable general selector.",
            "The step verifier checks both equations, linking, original operands, and final answer in this narrow task.",
            "Visible valid steps and self-consistency do not establish faithfulness of an internal chain of thought.",
            "Weak arithmetic generalization or GSM8K loss reduction must not be presented as natural-language complex reasoning.",
        ],
    }
    write_json(ctx.output / "generations.json", models)
    return result
