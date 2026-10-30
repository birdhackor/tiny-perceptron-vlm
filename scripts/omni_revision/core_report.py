"""只套用final前固定門檻；不改候選、分母、T、tau或失敗標準。"""

import argparse
import itertools
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from scripts.omni_revision.formal_local import FROZEN
from tiny_perceptron.omni_products.training_runtime import file_sha


def require(condition, message):
    if not condition:
        raise ValueError(f"final量測不完整：{message}")


def keyed_rows(rows, expected, key, scope):
    keys = [key(row) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected), scope)


def check_proportion(value, total, correct, scope):
    from scripts.omni_revision.core_evaluation import proportion

    require(value == proportion(correct, total), scope)


def validate_prediction(row, gold, ids):
    require(row["gold"] == gold and row["ids"] == ids, "C固定gold／候選ID")
    require(len(row["logits"]) == len(ids), "C每個候選的logit")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in row["logits"]), "C有限logits")
    answer = min(range(len(ids)), key=lambda i: (-row["logits"][i], ids[i]))
    require(row["answer"] == ids[answer], "C原始argmax答案")


def validate_complete(results, selection, metrics, data):
    """核對原始列、固定家族與全部視角；性能失敗仍保留完整分母。"""
    from scripts.omni_revision import core_evaluation as measured

    av_models = results["av"]["models"]
    a_names = {f"av-{kind}-s{seed}" for kind in ("additive", "time-rope") for seed in (11, 29, 47)}
    require(set(av_models) == a_names, "六個AV模型")
    av_expected = {row["family"]: row for row in data["AV"]["final"]}
    for value in av_models.values():
        rows = value["records"]
        keyed_rows(rows, av_expected, lambda row: row["family"], "AV200固定家族")
        for row in rows:
            fixed = av_expected[row["family"]]
            require(row["panel"] == fixed["panel"] and row["gold"] == fixed["labels"], "AV固定panel／gold")
            require(len(row["answers"]) == 2 and len(row["waveform_swap_answers"]) == 2, "AV配對及交換音訊兩次觀察")
            require(len(row["waveform_swap_gold"]) == 2, "AV交換音訊gold")
            require(
                type(row["paired_correct"]) is bool and row["paired_correct"] == (row["answers"] == row["gold"]),
                "AV配對正確標記",
            )
        check_proportion(value["primary"], 200, sum(row["paired_correct"] for row in rows), "AV主分母")
        require(set(value["panels"]) == {"0", "1", "2", "3"}, "AV四個panel")
        for panel in range(4):
            subset = [row for row in rows if row["panel"] == panel]
            require(len(subset) == 50, "AV各panel50家族")
            check_proportion(
                value["panels"][str(panel)], 50, sum(row["paired_correct"] for row in subset), "AV各panel分母"
            )
        for kind in ("image", "audio"):
            check_proportion(
                value[f"remove_{kind}"],
                200,
                sum(row[f"remove_{kind}_answer"] == 2 for row in rows),
                "AV移除模態200家族",
            )
        swapped = value["same_metadata_waveform_swap"]
        check_proportion(
            swapped["paired"],
            200,
            sum(row["waveform_swap_answers"] == row["waveform_swap_gold"] for row in rows),
            "交換音訊200配對",
        )
        require(swapped["observations"] == 400, "交換音訊400相關觀察")
        require(
            swapped["correct_observations"]
            == sum(a == b for row in rows for a, b in zip(row["waveform_swap_answers"], row["waveform_swap_gold"])),
            "交換音訊觀察計數",
        )
    require(
        results["av"]["paired_family_comparison"] == measured.paired_AV_comparison(av_models),
        "固定配對bootstrap／各seed與三seed均值",
    )

    sound = results["sound"]["codec_and_gold_program"]
    keyed_rows(sound["rows"], range(200), lambda row: row["family_index"], "sound200完整原始列")
    require(sound["families"] == 200, "sound200分母")
    for row in sound["rows"]:
        require(row["program"] == data["sound"]["final"][row["family_index"]], "sound固定完整program")
        require("generation" in row or "error" in row, "sound逐列生成結果或失敗")
        require("active_reference_snr_db" in row or "error" in row, "sound逐列SNR或失敗")
    for name, key in (("gold_program", "program_success"), ("legal_output", "legal")):
        check_proportion(
            sound[name], 200, sum(row.get("generation", {}).get(key, False) for row in sound["rows"]), "sound完整成功率"
        )
    keyed_rows(sound["chunk_controls"], range(10), lambda row: row["family_index"], "十次chunk／future-prefix控制")
    for row in sound["chunk_controls"]:
        require(
            type(row["chunks_bit_exact"]) is bool and type(row["future_code_prefix_bit_exact"]) is bool,
            "chunk控制結果或保留失敗",
        )
    snrs = [row.get("active_reference_snr_db") for row in sound["rows"]]
    invalid = sum(x is None or not math.isfinite(x) for x in snrs)
    require(sound["invalid_snr_families"] == invalid, "無效SNR不可丟棄")
    require(sound["mean_active_reference_snr_db"] == (None if invalid else sum(snrs) / 200), "SNR完整200均值")
    times = results["sound"]["actual_playable_timing"]
    keyed_rows(times, range(10), lambda row: row["family_index"], "十次實際首段延遲嘗試")
    for row in times:
        require(
            "error" in row
            or {
                "chunks",
                "first_playable_seconds",
                "whole_code_generation_seconds",
                "whole_completion_seconds",
                "samples",
                "sample_rate",
            }
            <= row.keys(),
            "實際首段延遲欄位或失敗",
        )

    chain = results["sound"]["actual_A_chain"]
    keyed_rows(chain["rows"], av_expected, lambda row: row["family"], "實際A音訊鏈200家族")
    for name, key in (
        ("primary", "chain_correct"),
        ("A_semantic", "semantic_correct"),
        ("mapping", "mapping_correct"),
        ("actual_condition_Talker", "actual_condition_correct"),
        ("gold_semantic_condition_Talker", "gold_semantic_condition_correct"),
        ("legal", "legal"),
    ):
        require(all(type(row[key]) is bool for row in chain["rows"]), "音訊鏈保留每列成敗")
        check_proportion(chain[name], 200, sum(row[key] for row in chain["rows"]), "音訊鏈各階段完整分母")
    require(
        chain["mapping_attempts"] == sum(row["mapping_attempted"] for row in chain["rows"]), "實際semantic字典映射嘗試"
    )
    require(
        chain["distinct_semantic_sound_programs"] == 2 and chain["all_failures_remain_in_denominator"] is True,
        "音訊鏈限制說明",
    )

    c_expected = {row["family"]: row for row in data["C"]["panels"]["final"]}
    for name, temperature, threshold in (
        ("primary", selection["temperature"], selection["threshold"]),
        ("data_control_raw_only", 1.0, None),
    ):
        value = results["choice"][name]
        rows = value["rows"]
        keyed_rows(rows, c_expected, lambda row: row["family"], "C60固定家族／兩個資料對照")
        for row in rows:
            fixed = c_expected[row["family"]]
            require(set(row["tasks"]) == {"choice", "info", "rubric"}, "C三種題型")
            for kind, predicted in row["tasks"].items():
                question = fixed["questions"][kind]
                ids = [x["id"] for x in question["candidates"]] + (["none"] if kind == "choice" else [])
                validate_prediction(predicted, question["gold"], ids)
            for key in ("source", "panel", "world_supported", "choice_information_sufficient"):
                require(row[key] == fixed[key], "C固定panel／可支援資訊")
            info = row["choice_info"]
            validate_prediction(
                {**info, "gold": fixed["questions"]["choice_info"]["gold"]},
                fixed["questions"]["choice_info"]["gold"],
                ["yes", "no"],
            )
        require(
            value == measured.evaluate_c(rows, temperature, threshold),
            "C三種raw／calibrated NLL、Brier、ECE10、policy及panel完整",
        )
    diagnostics = results["choice"]["candidate_count_secondary"]
    expected_conditions = {
        (row["family"], item["total_choices_including_none"]): item
        for row in c_expected.values()
        for item in row["candidate_count_diagnostics"]
    }
    keyed_rows(
        diagnostics["rows"],
        expected_conditions,
        lambda row: (row["family"], row["total_choices_including_none"]),
        "C候選2至8的420筆相關診斷",
    )
    require(
        diagnostics["parent_families"] == 60
        and diagnostics["secondary_observations"] == 420
        and diagnostics["independent_n_increase"] is False,
        "候選診斷分母",
    )
    for row in diagnostics["rows"]:
        total = row["total_choices_including_none"]
        ids = [x["id"] for x in data["C"]["diagnostic_candidate_pool"][: total - 1]] + ["none"]
        validate_prediction(row, expected_conditions[(row["family"], total)]["gold"], ids)
    invariants = results["choice"]["development_invariants"]
    keyed_rows(
        invariants["rows"],
        [row["family"] for row in data["C"]["panels"]["dev"][:8]],
        lambda row: row["family"],
        "C八個dev等價性控制",
    )
    invariant_keys = (
        "shared_recompute_logits_max_abs",
        "shared_recompute_hidden_max_abs",
        "shared_recompute_memory_max_abs",
        "Q1_Q2_Q1_logits_max_abs",
        "permutation_probability_max_abs",
    )
    numeric = [row[key] for row in invariants["rows"] for key in invariant_keys]
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in numeric), "C等價性實際有限誤差")
    require(
        invariants["development_families"] == 8 and invariants["atol"] == metrics["C"]["equivalence_atol"],
        "C固定控制規格",
    )
    require(
        invariants["all_passed"]
        == (all(x <= 1e-5 for x in numeric) and all(row["permutation_answer_same"] for row in invariants["rows"])),
        "C等價性控制總結",
    )

    old = results["integration"]["old_A_retention"]
    old_expected = {(row["task"], row["family"], row["record"]) for row in data["old_A_retention"]}
    keyed_rows(
        old["rows"], old_expected, lambda row: (row["task"], row["family"], row["source_record"]), "舊A258固定觀察"
    )
    task_counts = Counter(row["task"] for row in data["old_A_retention"])
    require(set(old["tasks"]) == set(task_counts), "舊A全部12種任務")
    for task, count in task_counts.items():
        subset = [row for row in old["rows"] if row["task"] == task]
        for row in subset:
            require(type(row["old_correct"]) is bool and type(row["wrapper_correct"]) is bool, "舊A逐列成敗")
            require("old" in row and "wrapper" in row, "舊A及包裝層各自結果或失敗")
        correct_old = sum(row["old_correct"] for row in subset)
        correct_new = sum(row["wrapper_correct"] for row in subset)
        require(
            old["tasks"][task]
            == {
                "families": count,
                "old_correct": correct_old,
                "wrapper_correct": correct_new,
                "difference_percentage_points": 100 * (correct_new - correct_old) / count,
            },
            "舊A每任務固定分母及差異",
        )
    require(
        old["observations"] == 258 and old["parent_families"] == len({row["family"] for row in old["rows"]}) == 180,
        "舊A258觀察／180共同父家族",
    )
    cf = old["counterfactual_diagnostics"]
    keyed_rows(
        cf["rows"],
        [(row["family"], row["record"]) for row in data["counterfactual_tool_diagnostics"]],
        lambda row: (row["family"], row["record"]),
        "24個改寫工具結果控制",
    )
    require(
        cf["observations"] == 24
        and cf["correct"] == sum(row["correct"] for row in cf["rows"])
        and cf["actual_executor_used"] is False,
        "反事實工具結果與實際工具分開",
    )

    b = results["integration"]["B_retention"]
    keyed_rows(
        b["rows"], map(tuple, data["B_retention"]["values"]), lambda row: tuple(row["parent"]), "B84固定有限母體"
    )
    require(b["population_families"] == 84 and b["frozen_depth"] == 1, "B84及固定深度")
    for name in ("old", "wrapper"):
        require(all(f"{name}_correct" in row and name in row for row in b["rows"]), "B原模型及包裝層逐列結果")
        require(b[f"{name}_correct"] == sum(row[f"{name}_correct"] for row in b["rows"]), "B成功計數")
    require(b["difference_percentage_points"] == 100 * (b["wrapper_correct"] - b["old_correct"]) / 84, "B保留能力差異")
    ordered = b["ordered_diagnostic"]
    keyed_rows(
        ordered["rows"], itertools.product(range(7), repeat=3), lambda row: tuple(row["numbers"]), "B343有序相關診斷"
    )
    require(
        ordered["observations"] == 343
        and ordered["parent_families"] == 84
        and ordered["correct"] == sum(row["correct"] for row in ordered["rows"])
        and ordered["not_independent343"] is True,
        "B診斷分母與成功計數",
    )

    auto = results["integration"]["auto"]
    auto_expected = {row["family"]: row["gold_route"] for row in data["Auto"]["families"]}
    keyed_rows(auto["rows"], auto_expected, lambda row: row["family"], "Auto100相同公開payload配對")
    require(auto["families"] == 100 and auto["counts"] == metrics["Auto"]["counts"], "Auto固定各路由分母")
    for row in auto["rows"]:
        require(row["gold_route"] == auto_expected[row["family"]], "Auto固定gold route")
        for key in ("gold_route_end_to_end_correct", "real_route_end_to_end_correct", "route_correct"):
            require(type(row[key]) is bool, "Auto每列成敗")
        require("gold_route_result" in row or "gold_route_error" in row, "Auto gold-route嘗試")
        require("actual_route" in row or "real_route_error" in row, "Auto實際路由嘗試")
        require(
            not row["real_route_end_to_end_correct"]
            or (row["route_correct"] and not row["contract_error"] and row["real_route_result"]["correct"]),
            "錯誤路由不可以gold補答",
        )
    for name, key in (
        ("route_accuracy", "route_correct"),
        ("gold_route_end_to_end", "gold_route_end_to_end_correct"),
        ("real_route_end_to_end", "real_route_end_to_end_correct"),
    ):
        check_proportion(auto[name], 100, sum(row[key] for row in auto["rows"]), "Auto配對評測完整分母")


def verdict(results, selection, metrics, data):
    validate_complete(results, selection, metrics, data)
    a = results["av"]["models"][selection["selected"]["av"]]
    sound = results["sound"]["codec_and_gold_program"]
    chain = results["sound"]["actual_A_chain"]
    c = results["choice"]["primary"]
    old = results["integration"]["old_A_retention"]
    b = results["integration"]["B_retention"]
    av = metrics["AV"]
    tone = metrics["sound"]
    choice = metrics["C"]
    checks = {
        "A_paired_families": a["primary"]["families"] == av["primary_families"],
        "A_pair_correct": a["primary"]["correct"] >= av["paired_minimum_correct"],
        "A_four_panels": all(
            a["panels"][str(panel)]["families"] == av["panel_families"]
            and a["panels"][str(panel)]["accuracy"] >= av["panel_minimum_accuracy"]
            for panel in range(4)
        ),
        "A_remove_image": a["remove_image"]["families"] == 200
        and a["remove_image"]["accuracy"] >= av["remove_image_minimum_accuracy"],
        "A_remove_audio": a["remove_audio"]["families"] == 200
        and a["remove_audio"]["accuracy"] >= av["remove_audio_minimum_accuracy"],
        "A_all_six_reported": len(results["av"]["models"]) == 6,
        "codec_SNR": sound["families"] == tone["final_families"]
        and sound["mean_active_reference_snr_db"] is not None
        and sound["mean_active_reference_snr_db"] >= tone["active_reference_snr_minimum_db"],
        "Talker_gold_program": sound["gold_program"]["families"] == 200
        and sound["gold_program"]["accuracy"] >= tone["gold_program_minimum_accuracy"],
        "sound_format": sound["legal_output"]["families"] == 200
        and sound["legal_output"]["accuracy"] == tone["legal_output_minimum_accuracy"],
        "sound_causal_chunks": len(sound["chunk_controls"]) == 10
        and all(row["chunks_bit_exact"] and row["future_code_prefix_bit_exact"] for row in sound["chunk_controls"]),
        "actual_AV_chain": chain["primary"]["families"] == 200
        and chain["primary"]["accuracy"] >= tone["actual_AV_chain_minimum_accuracy"],
        "actual_AV_chain_format": chain["legal"]["families"] == 200 and chain["legal"]["accuracy"] == 1,
        "C_Choice": c["types"]["choice"]["primary"]["families"] == 60
        and c["types"]["choice"]["primary"]["correct"] >= choice["choice_minimum_correct"],
        "C_information": c["types"]["info"]["primary"]["families"] == 60
        and c["types"]["info"]["primary"]["correct"] >= choice["info_minimum_correct"],
        "C_rubric": c["types"]["rubric"]["primary"]["families"] == 60
        and c["types"]["rubric"]["primary"]["correct"] >= choice["rubric_minimum_correct"],
        "C_fixed_policy": c["policy"]["families"] == 60
        and c["policy"]["accepted"] >= choice["minimum_accepted"]
        and c["policy"]["empirical_risk"] is not None
        and c["policy"]["empirical_risk"] <= choice["maximum_empirical_accepted_risk"],
        "C_invariants": results["choice"]["development_invariants"]["all_passed"],
        "Auto_complete": results["integration"]["auto"]["families"] == 100,
        "old_A_retention_complete": old["observations"] == 258 and old["parent_families"] == 180,
        "old_A_no_excess_drop": all(
            row["difference_percentage_points"] >= -metrics["retention"]["maximum_old_task_drop_percentage_points"]
            for row in old["tasks"].values()
        ),
        "B_population_complete": b["population_families"] == 84 and b["ordered_diagnostic"]["observations"] == 343,
        "B_no_excess_drop": b["difference_percentage_points"]
        >= -metrics["retention"]["maximum_old_task_drop_percentage_points"],
    }
    standalone = all(checks[key] for key in ("codec_SNR", "Talker_gold_program", "sound_format", "sound_causal_chunks"))
    retention = all(
        checks[key]
        for key in ("old_A_retention_complete", "old_A_no_excess_drop", "B_population_complete", "B_no_excess_drop")
    )
    new_a = (
        standalone
        and retention
        and all(value for key, value in checks.items() if key.startswith("A_") or key.startswith("actual_AV_chain"))
    )
    new_c = retention and all(value for key, value in checks.items() if key.startswith("C_"))
    return {
        "schema": "omni-core-capability-gates-v1",
        "checks": checks,
        "promote_finite_sound_module": standalone,
        "promote_A_AV_sound_extension": new_a,
        "promote_C_service": new_c,
        "B_weights_unchanged": True,
        "Auto100_is_measurement_not_guaranteed_capability": True,
        "research_measurements_complete": True,
        "global_targets_all_met": all(checks.values()),
        "failed_module_default": "retain historical product; publish new weights as labelled research candidate",
    }


def validate_part(part, directory, opening, opening_sha):
    receipt = json.loads((directory / "part-completion.json").read_bytes())
    origin = json.loads((directory / "part-origin.json").read_bytes())
    require(origin["part"] == part and receipt["part"] == part, "部分名稱與所在目錄")
    require(origin["final_opened"] is True and receipt["measurement_completed"] is True, "部分需由已開封量測完整結束")
    for value in (origin, receipt):
        require(value["final_opening_sha256"] == opening_sha, "部分來自同一次開封")
        require(value["model_selection_sha256"] == opening["model_selection_sha256"], "部分來自同一固定候選")
    for key in ("data_definition_sha256", "metrics_sha256", "evaluation_sources_sha256"):
        require(origin[key] == opening[key], "部分使用同一資料／度量／執行來源")
    require(receipt["results_sha256"] == file_sha(directory / "results.json"), "部分結果SHA")
    require(receipt["part_origin_sha256"] == file_sha(directory / "part-origin.json"), "部分原始收據SHA")
    started = datetime.fromisoformat(origin["created_at"])
    finished = datetime.fromisoformat(receipt["created_at"])
    opened = datetime.fromisoformat(opening["created_at"])
    require(
        started.tzinfo is not None and finished.tzinfo is not None and opened.tzinfo is not None, "部分時間需有時區"
    )
    require(opened <= started <= finished, "開封／量測開始／結束順序")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    opening = json.loads((FROZEN / "final-opening.json").read_bytes())
    if (
        opening.get("final_opened") is not True
        or opening["model_selection_sha256"] != file_sha(FROZEN / "model-selection.json")
        or opening["metrics_sha256"] != file_sha(FROZEN / "metrics.json")
        or opening["data_definition_sha256"] != file_sha(FROZEN / "data-definition.json")
        or opening["evaluation_sources_sha256"]["scripts/omni_revision/core_report.py"] != file_sha(__file__)
    ):
        raise ValueError("開final後不可更改門檻、候選或報表判定程式")
    values = {
        part: json.loads((args.results / part / "results.json").read_bytes())
        for part in ("av", "sound", "choice", "integration")
    }
    for part in values:
        validate_part(part, args.results / part, opening, file_sha(FROZEN / "final-opening.json"))
    selection = json.loads((FROZEN / "model-selection.json").read_bytes())
    metrics = json.loads((FROZEN / "metrics.json").read_bytes())
    data = json.loads((FROZEN / "data-definition.json").read_bytes())
    # 所有來源仍需與開封時逐檔相同，不能只核對報表檔本身。
    from scripts.omni_revision.core_final import evaluation_sources

    if opening["evaluation_sources_sha256"] != evaluation_sources():
        raise ValueError("開final後量測來源已漂移")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(verdict(values, selection, metrics, data), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
