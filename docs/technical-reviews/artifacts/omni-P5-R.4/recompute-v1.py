"""R.4 reviewer: portable arithmetic replay of preserved fields; no neural inference."""

import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def winner(record):
    return min(zip(record["ids"], record["logits"]), key=lambda pair: (-pair[1], pair[0]))[0]


def probability(record, temperature):
    scaled = [value / temperature for value in record["logits"]]
    maximum = max(scaled)
    values = [math.exp(value - maximum) for value in scaled]
    return max(values) / sum(values)


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "measurement-projection-v1.json").read_text())
    gates = data["freeze"]["metrics"]
    jobs = data["core_jobs"]
    assert len(jobs) == 10 and len({job["job"] for job in jobs}) == 10
    assert all(job["success"] and job["complete"] and job["optimizer_updates_observed"] for job in jobs)
    assert all(job["updates"] == job["requested_updates"] == job["last_logged_update"] for job in jobs)
    assert all(job["initial_parameter_sha256"] != job["latest_parameter_sha256"] for job in jobs)
    assert all(not job["final_opened"] for job in jobs)
    assert len(data["final_parts"]) == 4 and all(part["measurement_completed"] for part in data["final_parts"])
    assert data["freeze"]["metrics_created_at"] < data["freeze"]["selection_created_at"] < data["freeze"]["final_opened_at"]
    assert max(job["finished_at"] for job in jobs) < data["freeze"]["selection_created_at"]

    av = data["av"]["records"]
    assert len(av) == len({row["family"] for row in av}) == gates["AV"]["primary_families"] == 200
    paired = sum(row["answers"] == row["gold"] for row in av)
    panels = {str(panel): {"correct": sum(row["answers"] == row["gold"] for row in av if row["panel"] == panel),
                           "families": sum(row["panel"] == panel for row in av)} for panel in range(4)}
    remove_image = sum(row["remove_image_answer"] == 2 for row in av)
    remove_audio = sum(row["remove_audio_answer"] == 2 for row in av)
    swaps = sum(row["waveform_swap_answers"] == row["waveform_swap_gold"] for row in av)
    swap_observations = sum(a == g for row in av for a, g in zip(row["waveform_swap_answers"], row["waveform_swap_gold"], strict=True))
    av_pass = paired >= gates["AV"]["paired_minimum_correct"] and all(panel["families"] == 50 and panel["correct"] / 50 >= gates["AV"]["panel_minimum_accuracy"] for panel in panels.values())
    av_pass = av_pass and remove_image / 200 >= gates["AV"]["remove_image_minimum_accuracy"] and remove_audio / 200 >= gates["AV"]["remove_audio_minimum_accuracy"]
    assert (paired, remove_image, remove_audio, swaps, swap_observations) == (200, 200, 200, 148, 348)

    sound = data["sound"]["rows"]
    assert len(sound) == len({row["family"] for row in sound}) == 200
    mean_snr = sum(row["snr_db"] for row in sound) / len(sound)
    sound_success = sum(row["program_success"] for row in sound)
    sound_legal = sum(row["legal"] for row in sound)
    measured_program_success = 0
    for row in sound:
        measured, expected = row["measured_events"], row["expected_events"]
        ok = len(measured) == len(expected)
        for actual, gold in zip(measured, expected):
            ok = ok and abs(actual["frequency"] - gold["frequency"]) / gold["frequency"] <= gates["sound"]["frequency_relative_tolerance"]
            ok = ok and abs(actual["onset_ms"] - gold["onset_ms"]) <= gates["sound"]["timing_tolerance_ms"]
            ok = ok and abs(actual["offset_ms"] - gold["offset_ms"]) <= gates["sound"]["timing_tolerance_ms"]
        measured_program_success += bool(ok)
        assert bool(ok) == row["program_success"] == row["analysis_success"]
    chain = data["sound"]["chain"]
    assert len(chain) == len({row["family"] for row in chain}) == 200
    chain_success = sum(row["chain_correct"] for row in chain)
    assert all(row["chain_correct"] == (row["semantic_correct"] and row["mapping_correct"] and row["actual_condition_correct"] and row["legal"]) for row in chain)
    chunk_controls = data["sound"]["chunk_controls"]
    chunks_pass = all(row["chunks_bit_exact"] and row["future_code_prefix_bit_exact"] for row in chunk_controls)
    sound_pass = mean_snr >= gates["sound"]["active_reference_snr_minimum_db"] and sound_success / 200 >= gates["sound"]["gold_program_minimum_accuracy"] and sound_legal / 200 >= gates["sound"]["legal_output_minimum_accuracy"] and chain_success / 200 >= gates["sound"]["actual_AV_chain_minimum_accuracy"] and chunks_pass
    assert sound_success == sound_legal == measured_program_success == chain_success == 200

    rows = data["choice"]["rows"]
    assert len(rows) == len({row["family"] for row in rows}) == 60
    temperature, threshold = data["freeze"]["temperature"], data["freeze"]["threshold"]
    correct = Counter()
    accepted = errors = 0
    probability_max_abs_error = 0.0
    for row in rows:
        for kind, prediction in row["tasks"].items():
            answer = winner(prediction)
            assert answer == prediction["answer"]
            correct[kind] += answer == prediction["gold"]
            probability_max_abs_error = max(probability_max_abs_error, abs(probability(prediction, temperature) - prediction["p_max"]))
        choice = row["tasks"]["choice"]
        accept = probability(choice, temperature) >= threshold and winner(choice) != "none" and winner(row["choice_info"]) == "yes"
        error = winner(choice) != choice["gold"] or not row["world_supported"] or not row["choice_information_sufficient"]
        accepted += bool(accept)
        errors += bool(accept and error)
    assert dict(correct) == {"choice": 39, "info": 55, "rubric": 35}
    assert (accepted, errors) == (53, 19)
    assert probability_max_abs_error < 1e-6
    risk = errors / accepted
    choice_pass = correct["choice"] >= gates["C"]["choice_minimum_correct"]
    risk_pass = accepted >= gates["C"]["minimum_accepted"] and risk <= gates["C"]["maximum_empirical_accepted_risk"]
    invariants = data["choice"]["development_invariants"]
    invariant_keys = ["shared_recompute_logits_max_abs", "shared_recompute_hidden_max_abs", "shared_recompute_memory_max_abs", "Q1_Q2_Q1_logits_max_abs", "permutation_probability_max_abs"]
    invariant_maxima = {key: max(row[key] for row in invariants) for key in invariant_keys}
    invariant_pass = len(invariants) == 8 and all(value <= data["choice"]["atol"] for value in invariant_maxima.values()) and all(row["permutation_answer_same"] for row in invariants)

    old_a = data["retention"]["old_A_rows"]
    old_b = data["retention"]["B_rows"]
    a_tasks = {}
    for task in sorted({row["task"] for row in old_a}):
        selected = [row for row in old_a if row["task"] == task]
        a_tasks[task] = {"observations": len(selected), "old_correct": sum(row["old_correct"] for row in selected), "wrapper_correct": sum(row["wrapper_correct"] for row in selected)}
        assert a_tasks[task]["old_correct"] == a_tasks[task]["wrapper_correct"]
    assert len(old_a) == 258 and len(old_b) == 84
    assert sum(row["old_correct"] for row in old_b) == sum(row["wrapper_correct"] for row in old_b) == 60
    assert all(row["old_correct"] == row["wrapper_correct"] for row in old_b)
    assert not any(job["job"].lower().startswith("b-") for job in jobs)

    arithmetic = [[sum(a * b for a, b in zip(recipe, column)) for column in ([10, 20, 30], [0, 1, 4])] for recipe in ([2, 1, 0], [1, 2, 1])]
    assert arithmetic == [[40, 1], [80, 6]]
    legacy = data["legacy_training"]
    assert all(row["origin"] == "all-neural-weights-random" for row in legacy if "origin" in row)
    assert all(row["random_initialization"] and row["resume_source"] is None and row["weights_changed"] for row in legacy if "product" in row)
    assert all(row["initial_weights_sha256"] != row["selected_weights_sha256"] for row in legacy if "product" in row)
    legacy_scores = data["legacy_scores"]
    old_tool = legacy_scores["MoE"]["per_task"]["tool_call"]["tool_roundtrip"]
    old_sum = legacy_scores["B_5"]["sum"]
    failure_examples = legacy_scores["C_failure_examples"]
    assert (old_tool["numerator"], old_tool["denominator"]) == (0, 276)
    assert (old_sum["correct"], old_sum["count"]) == (2, 42)
    assert all(row["candidates"][max(range(len(row["raw_scores"])), key=lambda j: row["raw_scores"][j])] == row["predicted"] != row["gold"] for row in failure_examples)
    result = {
        "schema": "reviewer-independent-arithmetic-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_r_4",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU; Python arithmetic only; no torch import or neural forward"},
        "training": {"completed_jobs": len(jobs), "by_kind": dict(Counter(job["kind"] for job in jobs)), "updates": {job["job"]: job["updates"] for job in jobs}, "four_final_parts_complete": True, "pre_final_freeze_order_verified": True},
        "AV": {"selected": data["av"]["selected_job"], "primary_correct": paired, "families": len(av), "panels": panels, "remove_image_correct": remove_image, "remove_audio_correct": remove_audio, "waveform_swap_pairs_correct": swaps, "waveform_swap_observations_correct": swap_observations, "waveform_swap_observations": 400, "primary_gates_pass": av_pass},
        "sound": {"families": len(sound), "mean_preserved_active_snr_db": mean_snr, "gold_program_correct": sound_success, "legal": sound_legal, "independent_event_tolerance_recheck_correct": measured_program_success, "chain_correct": chain_success, "chain_families": len(chain), "chunk_controls": len(chunk_controls), "chunk_controls_pass": chunks_pass, "primary_gates_pass": sound_pass, "distinct_chain_sound_programs": data["sound"]["distinct_semantic_sound_programs"]},
        "C_v2": {"families": len(rows), "correct": dict(correct), "accepted": accepted, "accepted_errors": errors, "empirical_risk": risk, "T": temperature, "tau": threshold, "maximum_preserved_probability_error": probability_max_abs_error, "choice_gate_pass": choice_pass, "risk_gate_pass": risk_pass, "development_invariants": len(invariants), "invariant_maxima": invariant_maxima, "development_invariants_pass": invariant_pass},
        "retention": {"old_A_observations": len(old_a), "old_A_tasks": a_tasks, "B_population_families": len(old_b), "B_old_correct": 60, "B_wrapper_correct": 60, "B_depth": data["retention"]["B_depth"], "no_B_training_job": True},
        "W5_arithmetic": {"weighted_sum": arithmetic, "bias_5": [[row[0]+5, row[1]] for row in arithmetic]},
        "legacy": {"MoE_random_origin_records": 2, "B_C_published_random_origin_receipts": 2, "MoE_tool_roundtrip": {"correct": old_tool["numerator"], "questions": old_tool["denominator"]}, "B_five_loop_sum": {"correct": old_sum["correct"], "questions": old_sum["count"]}, "C_wording_failure_examples_rechecked_by_raw_argmax": len(failure_examples)},
        "limitations": data["limitations"] + ["Sound event tolerance replay uses preserved measured events; it does not reanalyze original float32 waveform samples.", "Cache equivalence results are preserved checks on eight development families; this replay never runs the trained network."],
        "all_replay_assertions_pass": av_pass and sound_pass and not choice_pass and not risk_pass and invariant_pass,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
