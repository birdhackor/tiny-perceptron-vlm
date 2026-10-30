"""固定分母的量測函式；真正final開封由另外的入口控制。"""

import hashlib
import itertools
import math
import re
import time
import wave
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_perceptron.omni_products.av import collate_av
from tiny_perceptron.omni_products.choice import Candidate, collate_questions
from tiny_perceptron.omni_products.controlled_diagnostics import controlled_tool_reply
from tiny_perceptron.omni_products.data import Event, av_pair, av_waveform_swap_control, sound_wave
from tiny_perceptron.omni_products.inference import SOUND_MAP
from tiny_perceptron.omni_products.materialize import auto_input, public_old_a, public_question
from tiny_perceptron.omni_products.media_bindings import retention_result
from tiny_perceptron.omni_products.reliability import accepted_rows, calibrated, wilson
from tiny_perceptron.omni_products.routing import b_sum_adapter
from tiny_perceptron.omni_products.signal_metrics import active_reference_snr, program_success


def proportion(correct, total):
    return {
        "correct": correct,
        "families": total,
        "accuracy": correct / total if total else None,
        "wilson95_family": wilson(correct, total),
    }


def error_record(error):
    return {"error_type": type(error).__name__, "message": str(error)[:500]}


@torch.inference_mode()
def av_predictions(model, examples, batch_size=64):
    answers = []
    for start in range(0, len(examples), batch_size):
        batch = collate_av(examples[start : start + batch_size], "cpu", model.config)
        answers.extend(model(batch).argmax(-1).tolist())
    return answers


@torch.inference_mode()
def evaluate_av(model, definitions):
    examples, pairs, missing_image, missing_audio, controls = [], [], [], [], []
    for row in definitions:
        pair = av_pair(row["family"], row["variant"], row["panel"])
        if [label for _, label in pair] != row["labels"]:
            raise ValueError("固定資料與現場生成標籤不同")
        pairs.append(pair)
        examples.extend(pair)
        item, _ = pair[row["predeclared_chain_pair_index"]]
        missing_image.append((replace(item, image_present=False), 2))
        missing_audio.append((replace(item, audio_present=False), 2))
        controls.extend(av_waveform_swap_control(row["family"], row["panel"]))
    predicted = av_predictions(model, examples)
    removed_image = av_predictions(model, missing_image)
    removed_audio = av_predictions(model, missing_audio)
    swapped = av_predictions(model, controls)
    records = []
    for i, row in enumerate(definitions):
        answers = predicted[2 * i : 2 * i + 2]
        records.append(
            {
                "family": row["family"],
                "panel": row["panel"],
                "answers": answers,
                "gold": row["labels"],
                "paired_correct": answers == row["labels"],
                "remove_image_answer": removed_image[i],
                "remove_audio_answer": removed_audio[i],
                "waveform_swap_answers": swapped[2 * i : 2 * i + 2],
                "waveform_swap_gold": [label for _, label in controls[2 * i : 2 * i + 2]],
            }
        )
    panels = {
        str(panel): proportion(
            sum(row["paired_correct"] for row in records if row["panel"] == panel),
            sum(row["panel"] == panel for row in records),
        )
        for panel in range(4)
    }
    return {
        "primary": proportion(sum(row["paired_correct"] for row in records), len(records)),
        "panels": panels,
        "remove_image": proportion(sum(value == 2 for value in removed_image), len(records)),
        "remove_audio": proportion(sum(value == 2 for value in removed_audio), len(records)),
        "same_metadata_waveform_swap": {
            "paired": proportion(
                sum(row["waveform_swap_answers"] == row["waveform_swap_gold"] for row in records), len(records)
            ),
            "correct_observations": sum(pred == gold for pred, (_, gold) in zip(swapped, controls)),
            "observations": len(controls),
        },
        "swap_statistical_unit": "secondary two-observation view of the same primary families; no extra independent n",
        "chance_pair_success": 0.25,
        "records": records,
    }


def waveform_receipt(waveform, output=None):
    samples = waveform.detach().float().cpu().contiguous()
    value = {
        "sample_rate": 8000,
        "samples": len(samples),
        "float32_samples_sha256": hashlib.sha256(samples.numpy().tobytes()).hexdigest(),
    }
    if output is not None and len(samples) and torch.isfinite(samples).all():
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pcm = (samples.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
        with wave.open(str(output), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(pcm)
        value.update(
            wav_path=str(output),
            wav_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            wav_encoding="PCM16 playable export; numerical metrics use original float32",
        )
    return value


@torch.inference_mode()
def generated_sound(codec, talker, program, output=None):
    generated = talker.generate(program)
    codes = generated["codes"]
    waveform = codec.decode_codes(codes) if len(codes) else torch.empty(0)
    measured = program_success(waveform, program)
    result = {
        "eos": generated["eos"],
        "capped": generated["capped"],
        "codes": codes.cpu().tolist(),
        "legal": bool(generated["eos"] and not generated["capped"] and measured["measured"]["legal"]),
        "program_success": bool(generated["eos"] and not generated["capped"] and measured["success"]),
        "analysis": measured,
        "waveform": waveform_receipt(waveform, output),
    }
    return result


@torch.inference_mode()
def evaluate_sound(codec, talker, programs, output):
    rows, snrs = [], []
    prefix_checks = []
    for i, value in enumerate(programs):
        program = tuple(Event(**event) for event in value)
        row = {"family_index": i, "program": value}
        prefix = None
        if i < 10:
            prefix = {"family_index": i, "chunks_bit_exact": False, "future_code_prefix_bit_exact": False}
            prefix_checks.append(prefix)
        try:
            reference = sound_wave(program)
            codes, length = codec.encode_wave(reference)
            reconstructed = codec.decode_codes(codes)[:length]
            snr = active_reference_snr(reference, reconstructed, program)
            row["active_reference_snr_db"] = snr
            snrs.append(snr)
            row["generation"] = generated_sound(codec, talker, program, output / f"sound-{i:03}.wav")
            if i < 10:
                whole = codec.decode_codes(codes)
                chunks = torch.cat(list(codec.chunks(codes)))
                stop = min(3, len(codes) - 1)
                altered = codes.clone()
                altered[stop:] = (altered[stop:] + 1) % codec.config.codes
                prefix.update(
                    chunks_bit_exact=bool(torch.equal(whole, chunks)),
                    future_code_prefix_bit_exact=bool(
                        torch.equal(
                            whole[: stop * codec.config.hop], codec.decode_codes(altered)[: stop * codec.config.hop]
                        )
                    ),
                )
        except Exception as error:
            if "active_reference_snr_db" not in row:
                snrs.append(None)
            row["error"] = error_record(error)
            if prefix is not None:
                prefix["error"] = error_record(error)
        rows.append(row)
    invalid = sum(value is None or not math.isfinite(value) for value in snrs)
    # 任一無效列都不能被丟掉後提高SNR均值。
    mean_snr = sum(snrs) / len(programs) if not invalid and len(snrs) == len(programs) else None
    return {
        "families": len(programs),
        "reference_wave": "canonical phase0/amplitude0.6, same full held-out program family",
        "mean_active_reference_snr_db": mean_snr,
        "invalid_snr_families": invalid,
        "gold_program": proportion(
            sum(row.get("generation", {}).get("program_success", False) for row in rows), len(rows)
        ),
        "legal_output": proportion(sum(row.get("generation", {}).get("legal", False) for row in rows), len(rows)),
        "chunk_controls": prefix_checks,
        "rows": rows,
    }


@torch.inference_mode()
def evaluate_chain(assistant, definitions, output):
    rows = []
    for i, row in enumerate(definitions):
        item, label = av_pair(row["family"], row["variant"], row["panel"])[row["predeclared_chain_pair_index"]]
        gold = ("circle", "square")[label]
        record = {
            "family": row["family"],
            "panel": row["panel"],
            "which": row["predeclared_chain_pair_index"],
            "gold": gold,
            "semantic_correct": False,
            "mapping_correct": False,
            "mapping_attempted": False,
            "actual_condition_correct": False,
            "gold_semantic_condition_correct": False,
            "chain_correct": False,
            "legal": False,
        }
        try:
            answer = assistant.video(item, with_sound=True)
            record.update(answer=answer["answer"], semantic_correct=answer["answer"] == gold)
            if "sound" in answer:
                sound = answer["sound"]
                gold_program = SOUND_MAP[gold]
                actual_program = SOUND_MAP.get(answer["answer"])
                scored = program_success(sound["waveform"], gold_program)
                mapping = actual_program is not None and tuple(sound["program"]) == actual_program
                actual_scored = (
                    program_success(sound["waveform"], actual_program) if actual_program is not None else None
                )
                legal = sound["eos"] and not sound["capped"] and scored["measured"]["legal"]
                record.update(
                    mapping_correct=mapping,
                    mapping_attempted=actual_program is not None,
                    actual_condition_correct=bool(mapping and legal and actual_scored["success"]),
                    analysis_against_actual_condition=actual_scored,
                    analysis_against_gold_program=scored,
                    legal=bool(legal),
                    waveform=waveform_receipt(sound["waveform"], output / f"chain-{i:03}.wav"),
                    chain_correct=bool(record["semantic_correct"] and mapping and legal and scored["success"]),
                )
        except Exception as error:
            record["error"] = error_record(error)
        # Gold semantic另跑自由生成，絕不補到實際A輸出。
        try:
            gold_generation = generated_sound(
                assistant.codec, assistant.talker, SOUND_MAP[gold], output / f"gold-semantic-{i:03}.wav"
            )
            record.update(
                gold_semantic_condition=gold_generation,
                gold_semantic_condition_correct=gold_generation["program_success"],
            )
        except Exception as error:
            record["gold_condition_error"] = error_record(error)
        rows.append(record)
    return {
        "primary": proportion(sum(row["chain_correct"] for row in rows), len(rows)),
        "A_semantic": proportion(sum(row["semantic_correct"] for row in rows), len(rows)),
        "mapping": proportion(sum(row["mapping_correct"] for row in rows), len(rows)),
        "mapping_attempts": sum(row["mapping_attempted"] for row in rows),
        "mapping_scope": "dictionary correctness against actual predicted semantic; unknown stays in full chain denominator",
        "actual_condition_Talker": proportion(sum(row["actual_condition_correct"] for row in rows), len(rows)),
        "gold_semantic_condition_Talker": proportion(
            sum(row["gold_semantic_condition_correct"] for row in rows), len(rows)
        ),
        "distinct_semantic_sound_programs": 2,
        "legal": proportion(sum(row["legal"] for row in rows), len(rows)),
        "all_failures_remain_in_denominator": True,
        "sound_programs_seen_in_Talker_train": True,
        "rows": rows,
    }


def classification_metrics(records, temperature):
    losses, briers, correct, bins = [], [], [], defaultdict(list)
    for row in records:
        logits = torch.tensor(row["logits"], dtype=torch.float64)
        p = calibrated(row["logits"], temperature)
        gold = row["ids"].index(row["gold"])
        answer = min(range(len(p)), key=lambda i: (-row["logits"][i], row["ids"][i]))
        hit = answer == gold
        losses.append(float(-F.log_softmax(logits / temperature, -1)[gold]))
        briers.append(sum((value - int(i == gold)) ** 2 for i, value in enumerate(p)))
        correct.append(hit)
        bins[min(9, int(max(p) * 10))].append((max(p), hit))
    n = len(records)
    ece = sum(
        len(values) / n * abs(sum(x for x, _ in values) / len(values) - sum(y for _, y in values) / len(values))
        for values in bins.values()
    )
    return {
        "observations": n,
        "accuracy": sum(correct) / n,
        "NLL": sum(losses) / n,
        "Brier": sum(briers) / n,
        "ECE10": ece,
    }


def evaluate_c(families, temperature, threshold):
    types = {}
    for kind in ("choice", "info", "rubric"):
        records = [row["tasks"][kind] for row in families]
        correct = sum(row["answer"] == row["gold"] for row in records)
        types[kind] = {
            "primary": proportion(correct, len(families)),
            "raw": classification_metrics(records, 1),
            "calibrated": classification_metrics(records, temperature),
        }
    policy = (
        accepted_rows(families, temperature, threshold)
        if threshold is not None
        else {
            "families": len(families),
            "accepted": 0,
            "errors": 0,
            "coverage": 0,
            "empirical_risk": None,
            "rows": [
                {
                    "family": row["family"],
                    "accepted": False,
                    "error": False,
                    "answer": row["tasks"]["choice"]["answer"],
                    "p_max": max(calibrated(row["tasks"]["choice"]["logits"], temperature)),
                }
                for row in families
            ],
        }
    )
    policy["error_wilson95_among_accepts"] = wilson(policy["errors"], policy["accepted"])
    panels = {
        panel: proportion(
            sum(
                row["tasks"]["choice"]["answer"] == row["tasks"]["choice"]["gold"]
                for row in families
                if row["panel"] == panel
            ),
            sum(row["panel"] == panel for row in families),
        )
        for panel in ("complete", "removed", "missing")
    }
    return {
        "types": types,
        "policy": policy,
        "choice_panels": panels,
        "types_correlated_not_independent180": True,
        "rows": families,
    }


def direct_old_reply(old_a, public):
    return old_a.reply(
        public["messages"],
        task=public["task"],
        max_new_tokens=128,
        tools=public.get("tools", False),
        **public.get("metadata", {}),
    )


def evaluate_old_a(assistant, records, counterfactual):
    rows = []
    for row in records:
        record = {
            "task": row["task"],
            "family": row["family"],
            "source_record": row["record"],
            "old_correct": False,
            "wrapper_correct": False,
        }
        public = public_old_a(row["input"])
        for name, call in (
            ("old", lambda: direct_old_reply(assistant.old_a, public)),
            ("wrapper", lambda: assistant.old_reply(public)),
        ):
            try:
                response = call()
                record[name] = retention_result(row, response)
                record[f"{name}_correct"] = record[name]["correct"]
            except Exception as error:
                record[name] = {"error": error_record(error)}
        rows.append(record)
    tasks = {}
    for task in sorted({row["task"] for row in rows}):
        subset = [row for row in rows if row["task"] == task]
        old = sum(row["old_correct"] for row in subset)
        wrapper = sum(row["wrapper_correct"] for row in subset)
        tasks[task] = {
            "families": len(subset),
            "old_correct": old,
            "wrapper_correct": wrapper,
            "difference_percentage_points": 100 * (wrapper - old) / len(subset),
        }
    controlled = []
    for row in counterfactual:
        record = {
            key: row[key] for key in ("family", "record", "gold", "actual_executor_result", "provided_tool_result")
        }
        try:
            result = controlled_tool_reply(assistant.old_a, row["controlled_prefix"])
            record.update(actual=result["answer"], correct=result["answer"] == row["gold"], actual_executor_used=False)
        except Exception as error:
            record.update(correct=False, error=error_record(error))
        controlled.append(record)
    return {
        "tasks": tasks,
        "observations": len(rows),
        "parent_families": len({row["family"] for row in rows}),
        "shared_cross_task_parents_not_independent": True,
        "rows": rows,
        "counterfactual_diagnostics": {
            "observations": len(controlled),
            "correct": sum(row["correct"] for row in controlled),
            "actual_executor_used": False,
            "same_live_parent_families": True,
            "rows": controlled,
        },
    }


def evaluate_b(assistant, values):
    from tiny_perceptron.lesson_products.inference import rule_depth

    depth = rule_depth(assistant.old_b.auto_policy, "sum")
    rows = []
    for numbers in values:
        record = {"parent": numbers, "gold": sum(numbers), "old_correct": False, "wrapper_correct": False}
        public = b_sum_adapter(numbers)
        for name, call in (
            ("old", lambda: assistant.old_b.solve(public, loops=depth)),
            ("wrapper", lambda: assistant.dispatch("B", {"numbers": numbers})),
        ):
            try:
                answer = call()
                record[name] = answer
                record[f"{name}_correct"] = answer["answer"] == record["gold"]
            except Exception as error:
                record[name] = {"error": error_record(error)}
        rows.append(record)
    ordered = []
    for numbers in itertools.product(range(7), repeat=3):
        record = {"numbers": list(numbers), "parent": sorted(numbers), "gold": sum(numbers), "correct": False}
        try:
            record["actual"] = assistant.dispatch("B", {"numbers": numbers})
            record["correct"] = record["actual"]["answer"] == record["gold"]
        except Exception as error:
            record["error"] = error_record(error)
        ordered.append(record)
    n = len(rows)
    old = sum(row["old_correct"] for row in rows)
    wrapper = sum(row["wrapper_correct"] for row in rows)
    return {
        "population_families": n,
        "old_correct": old,
        "wrapper_correct": wrapper,
        "difference_percentage_points": 100 * (wrapper - old) / n,
        "frozen_depth": depth,
        "rows": rows,
        "ordered_diagnostic": {
            "observations": len(ordered),
            "parent_families": n,
            "correct": sum(row["correct"] for row in ordered),
            "not_independent343": True,
            "rows": ordered,
        },
    }


def score_auto_result(record, result):
    route = record["gold_route"]
    if route == "none":
        return {"correct": result.get("status") == "unsupported"}
    if route == "B":
        return {"correct": result.get("answer") == record["gold"]}
    if route == "C":
        actual = {
            kind: result[kind]["prediction"]["answer"] if kind == "choice" else result[kind]["answer"]
            for kind in ("choice", "info", "rubric")
        }
        return {
            "correct": actual == record["gold"],
            "actual": actual,
            "choice_accepted": result["choice"]["accepted"],
            "choice_decision": result["choice"]["decision"],
            "scope": "three representative predictions correct; accept/handoff separate, not automatic-action assurance",
        }
    if record["family"].startswith("auto-A-calculator"):
        _, operation, a, b = record["parent"]
        trace = result["tool_trace"]
        expected = {"tool": "calculator", "operation": operation, "a": a, "b": b}
        answer = re.fullmatch(
            r"\s*(?:(?:答案|結果|計算結果)(?:是|為|[:：]))?\s*([+-]?\d+)\s*[。.!！]?\s*", result["answer"]
        )
        return {
            "correct": bool(
                trace["executed"]
                and trace["tool_call"] == expected
                and answer is not None
                and int(answer.group(1)) == record["gold"]
            ),
            "actual": result["answer"],
            "tool_trace": trace,
        }
    if "video_audio_definition" in record["payload"]:
        correct = result["answer"] == record["gold"]
        sound = result.get("sound")
        analysis = program_success(sound["waveform"], SOUND_MAP[record["gold"]]) if sound else None
        return {
            "correct": bool(correct and sound and sound["eos"] and not sound["capped"] and analysis["success"]),
            "semantic_correct": correct,
            "actual": result["answer"],
            "sound_analysis": analysis,
        }
    return {"correct": result["answer"] == record["gold"], "actual": result["answer"]}


def evaluate_auto(assistant, records):
    rows = []
    for record in records:
        request, payload = auto_input(record)
        row = {
            "family": record["family"],
            "gold_route": record["gold_route"],
            "gold_route_end_to_end_correct": False,
            "real_route_end_to_end_correct": False,
            "route_correct": False,
        }
        try:
            result = assistant.dispatch(record["gold_route"], payload)
            row["gold_route_result"] = score_auto_result(record, result)
            row["gold_route_end_to_end_correct"] = row["gold_route_result"]["correct"]
        except Exception as error:
            row["gold_route_error"] = error_record(error)
        try:
            actual = assistant.auto(request, payload)
            row.update(
                actual_route=actual["route"],
                route_correct=actual["route"] == record["gold_route"],
                contract_error=actual["contract_error"],
            )
            row["route_logits"] = {
                key: actual["route_prediction"][key] for key in ("answer", "ids", "logits", "probabilities", "p_max")
            }
            if row["route_correct"] and not actual["contract_error"]:
                row["real_route_result"] = score_auto_result(record, actual["result"])
                row["real_route_end_to_end_correct"] = row["real_route_result"]["correct"]
        except Exception as error:
            row["real_route_error"] = error_record(error)
        rows.append(row)
    return {
        "families": len(rows),
        "route_accuracy": proportion(sum(row["route_correct"] for row in rows), len(rows)),
        "gold_route_end_to_end": proportion(sum(row["gold_route_end_to_end_correct"] for row in rows), len(rows)),
        "real_route_end_to_end": proportion(sum(row["real_route_end_to_end_correct"] for row in rows), len(rows)),
        "counts": {name: sum(row["gold_route"] == name for row in rows) for name in ("A", "B", "C", "none")},
        "rows": rows,
    }


def measure_playable_chunks(assistant, program):
    started = time.perf_counter()
    rows = list(assistant.sound_chunks(program))
    return {
        "chunks": len(rows),
        "first_playable_seconds": rows[0]["ready_seconds"] if rows else None,
        "whole_code_generation_seconds": rows[0]["whole_code_generation_seconds"] if rows else None,
        "whole_completion_seconds": time.perf_counter() - started,
        "sample_rate": 8000,
        "samples": sum(len(row["samples"]) for row in rows),
        "scope": "actual generator yielded playable chunks; code generation completes before chunk decoder starts",
    }


@torch.inference_mode()
def choice_invariants(model, development_rows):
    measurements = []
    for row in development_rows:
        q1 = public_question(row["questions"]["choice"])
        q2 = public_question(row["questions"]["info"])
        batch = collate_questions([q1])
        shared, recomputed = model(batch), model(batch, recompute=True)
        n = len(q1.candidates) + 1
        first = model.predict(q1)
        model.predict(q2)
        last = model.predict(q1)
        reverse = model.predict(replace(q1, candidates=tuple(reversed(q1.candidates))))
        before = {name: p for name, p in zip(first["ids"], first["probabilities"])}
        after = {name: p for name, p in zip(reverse["ids"], reverse["probabilities"])}
        measurements.append(
            {
                "family": row["family"],
                "shared_recompute_logits_max_abs": float(
                    (shared["logits"][:, :n] - recomputed["logits"][:, :n]).abs().max()
                ),
                "shared_recompute_hidden_max_abs": float(
                    (shared["candidate_hidden"] - recomputed["candidate_hidden"]).abs().max()
                ),
                "shared_recompute_memory_max_abs": float((shared["memory"] - recomputed["memory"]).abs().max()),
                "Q1_Q2_Q1_logits_max_abs": max(abs(a - b) for a, b in zip(first["logits"], last["logits"])),
                "permutation_probability_max_abs": max(abs(before[name] - after[name]) for name in before),
                "permutation_answer_same": first["answer"] == reverse["answer"],
            }
        )
    numeric = [value for row in measurements for key, value in row.items() if key.endswith("max_abs")]
    return {
        "development_families": len(measurements),
        "atol": 1e-5,
        "all_passed": all(value <= 1e-5 for value in numeric)
        and all(row["permutation_answer_same"] for row in measurements),
        "rows": measurements,
    }


def choice_candidate_diagnostics(judge, rows, pool):
    results = []
    candidates = tuple(Candidate(item["id"], item["description"]) for item in pool)
    for row in rows:
        primary = public_question(row["questions"]["choice"])
        for condition in row["candidate_count_diagnostics"]:
            total = condition["total_choices_including_none"]
            question = replace(primary, candidates=candidates[: total - 1])
            predicted = judge.predict(question)
            results.append(
                {
                    "family": row["family"],
                    "total_choices_including_none": total,
                    "gold": condition["gold"],
                    **{key: predicted[key] for key in ("ids", "logits", "answer", "probabilities", "p_max")},
                }
            )
    return {
        "parent_families": len(rows),
        "secondary_observations": len(results),
        "independent_n_increase": False,
        "rows": results,
    }


def paired_AV_comparison(models, replicates=1000, seed=20261029):
    """按共同family重抽；三seed的同情境不加成600個獨立家族。"""
    common = None
    by_seed = {}
    accuracies = {}
    for training_seed in (11, 29, 47):
        a_rows = models[f"av-additive-s{training_seed}"]["records"]
        b_rows = models[f"av-time-rope-s{training_seed}"]["records"]
        left = {row["family"]: row["paired_correct"] for row in a_rows}
        right = {row["family"]: row["paired_correct"] for row in b_rows}
        if (
            len(a_rows) != 200
            or len(b_rows) != 200
            or len(left) != 200
            or set(left) != set(right)
            or (common is not None and set(left) != set(common))
            or any(type(value) is not bool for value in [*left.values(), *right.values()])
        ):
            raise ValueError("配對比較須六個完整、相同的200家族")
        common = sorted(left)
        by_seed[training_seed] = torch.tensor(
            [int(right[family]) - int(left[family]) for family in common], dtype=torch.float64
        )
        accuracies[str(training_seed)] = {
            "additive": sum(left.values()) / 200,
            "time_rope": sum(right.values()) / 200,
        }
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(200, (replicates, 200), generator=generator)

    def interval(values):
        means = values[indices].mean(1)
        return {
            "mean_time_rope_minus_additive": float(values.mean()),
            "paired_family_bootstrap95": torch.quantile(
                means, torch.tensor([0.025, 0.975], dtype=torch.float64)
            ).tolist(),
        }

    average = torch.stack(list(by_seed.values())).mean(0)
    return {
        "families": 200,
        "replicates": replicates,
        "resampling_seed": seed,
        "per_training_seed": {str(key): interval(value) for key, value in by_seed.items()},
        "paired_accuracy_per_training_seed": accuracies,
        "paired_accuracy_three_seed_mean": {
            kind: sum(row[kind] for row in accuracies.values()) / 3 for kind in ("additive", "time_rope")
        },
        "three_seed_mean": interval(average),
        "unit": "same200 families, conditional on these3 downstream seeds; not600 independent scenarios",
    }
