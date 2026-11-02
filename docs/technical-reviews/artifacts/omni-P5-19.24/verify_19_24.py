"""Independent 19.24 review: artificial CPU probes and arithmetic on preserved rows.

No model is constructed, loaded, trained or called. collect only inspects original
JSON/ZIP bytes. replay works from the small committed projection and this repo.
"""

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import pickletools
import platform
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SNAPSHOT = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.24.md"
FINAL_SHA = "50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3"
CORE_SHA = "107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def select(value, names):
    return {name: value[name] for name in names}


def trim(value):
    if isinstance(value, dict):
        return {key: trim(item) for key, item in value.items() if key not in {"seconds", "scope", "wav_path"}}
    if isinstance(value, list):
        return [trim(item) for item in value]
    return value


def collect(raw_root, core_archive):
    raw_root = Path(raw_root)
    final_archive = raw_root.parent / "final-results.zip"
    assert sha(final_archive.read_bytes()) == FINAL_SHA
    manifest = read(raw_root / "package-file-manifest.json")
    checked = []
    for name, expected in manifest.items():
        raw = (raw_root / name).read_bytes()
        assert len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], name
        checked.append(name)
    definition = read(raw_root / "frozen/data-definition.json")
    integration = read(raw_root / "measurements/integration/results.json")
    selection = read(raw_root / "frozen/model-selection.json")
    metrics = read(raw_root / "frozen/metrics.json")
    av = read(raw_root / "measurements/av/results.json")
    sound = read(raw_root / "measurements/sound/results.json")
    choice = read(raw_root / "measurements/choice/results.json")
    raw_names = [
        "frozen/data-definition.json", "frozen/metrics.json", "frozen/model-selection.json",
        "frozen/final-opening.json", "execution-sources.json", "package-file-manifest.json",
        "measurements/integration/results.json", "measurements/integration/part-origin.json",
        "measurements/integration/part-completion.json", "measurements/av/results.json",
        "measurements/sound/results.json", "measurements/choice/results.json", "final-report.json",
    ]
    identities = {name: {"bytes": (raw_root / name).stat().st_size, "sha256": sha((raw_root / name).read_bytes())}
                  for name in raw_names}
    sources = read(raw_root / "execution-sources.json")
    opening = read(raw_root / "frozen/final-opening.json")
    bound_names = [
        "tiny_perceptron/omni_products/routing.py", "tiny_perceptron/omni_products/inference.py",
        "tiny_perceptron/omni_products/materialize.py", "tiny_perceptron/omni_products/media_bindings.py",
        "tiny_perceptron/omni_products/choice.py", "tiny_perceptron/omni_products/data.py",
        "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/omni_products/training_runtime.py", "tiny_perceptron/selftrained/tools.py",
        "tiny_perceptron/selftrained/inference.py", "scripts/omni_revision/core_evaluation.py",
        "scripts/omni_revision/core_report.py", "scripts/omni_revision/core_final.py",
        "scripts/omni_revision/data_freeze.py",
    ]
    source_bindings = {}
    for name in bound_names:
        actual = sha((ROOT / name).read_bytes())
        assert actual == sources["files"][name], name
        if name in opening["evaluation_sources_sha256"]:
            assert actual == opening["evaluation_sources_sha256"][name], name
        source_bindings[name] = actual
    assert sha(Path(core_archive).read_bytes()) == CORE_SHA
    checkpoints = []
    # Inspect pickle string keys without unpickling tensors or Python objects.
    with zipfile.ZipFile(core_archive) as archive:
        for job in (selection["selected"]["av"], "c-augmented", "codec", "talker"):
            prefix = "core/" + job + "/training/"
            record = json.loads(archive.read(prefix + "training-result.json"))
            for filename, expected in record["files"].items():
                content = archive.read(prefix + filename)
                assert len(content) == expected["bytes"] and sha(content) == expected["sha256"]
            keys = {}
            for filename in ("best-continuation.pt", "latest-continuation.pt"):
                with zipfile.ZipFile(io.BytesIO(archive.read(prefix + filename))) as nested:
                    pkl = next(name for name in nested.namelist() if name.endswith("/data.pkl"))
                    words = {arg for op, arg, pos in pickletools.genops(nested.read(pkl))
                             if op.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}}
                expected_keys = {"model", "optimizer", "step", "python_rng", "torch_rng", "cuda_rng", "sampler_rng", "metadata", "auxiliary"}
                assert expected_keys <= words
                keys[filename] = sorted(expected_keys)
            checkpoints.append({"job": job, "model_config": record["config"]["model"],
                                "files": record["files"], "pickle_keys_observed_without_unpickling": keys})
    auto_defs = []
    for row in definition["Auto"]["families"]:
        item = select(row, ["family", "parent", "request", "gold_route", "gold"])
        payload = row["payload"]
        item["payload_kind"] = next(iter(payload), "empty")
        if "old_A_input" in payload:
            item["old_A_task"] = payload["old_A_input"]["task"]
        if "numbers" in payload:
            item["numbers"] = payload["numbers"]
        if "video_audio_definition" in payload:
            item["video_audio_definition"] = payload["video_audio_definition"]
        auto_defs.append(item)
    old_defs = [select(row, ["task", "family", "record", "gold", "gold_tool_call", "scoring_field"])
                for row in definition["old_A_retention"]]
    snd = sound["codec_and_gold_program"]
    sound_rows = [{"family_index": row["family_index"], "snr": row["active_reference_snr_db"],
                   "eos": row["generation"]["eos"], "capped": row["generation"]["capped"],
                   "legal": row["generation"]["legal"], "analysis": row["generation"]["analysis"]}
                  for row in snd["rows"]]
    chain_rows = [select(row, ["family", "panel", "which", "gold", "answer", "legal",
                              "analysis_against_gold_program", "chain_correct"])
                  for row in sound["actual_A_chain"]["rows"]]
    provider = []
    for index in range(3):
        path = Path("/workspace/work/reference/technical-19.24") / f"github-release-{index}.json"
        response = read(path)
        fields = ["id", "tag_name", "target_commitish", "draft", "prerelease", "created_at", "published_at"]
        projection = {key: response[key] for key in fields if key in response}
        projection["assets"] = [{key: asset[key] for key in ("id", "name", "size", "digest", "browser_download_url")}
                                for asset in response.get("assets", [])]
        if "error" in response:
            projection["error"] = response["error"]
        provider.append({"tool": "mcp__codex_apps__github_fetch", "url": (
            "https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/tags/omni-core-candidate-cpu-core-40988311100-1",
            "https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/tags/omni-final-candidate-cpu-final-41004229262-1",
            "https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/latest")[index],
            "local_external_cache": str(path), "cached_tool_serialized_body_bytes": path.stat().st_size,
            "cached_tool_serialized_body_sha256": sha(path.read_bytes()), "projection": projection})
    result = {
        "schema": "reviewer-19.24-portable-row-projection-v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {"source_revision": sources["revision"], "final_run_id": 41004229262,
            "final_archive_url": "https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/omni-final-candidate-cpu-final-41004229262-1/final-results.zip",
            "final_archive_bytes": final_archive.stat().st_size, "final_archive_sha256": FINAL_SHA,
            "original_manifest_members_checked": len(checked), "raw_members": identities,
            "core_archive_url": "https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/omni-core-candidate-cpu-core-40988311100-1/core-checkpoints.zip",
            "core_archive_bytes": Path(core_archive).stat().st_size, "core_archive_sha256": CORE_SHA,
            "current_source_bindings_equal_original_execution": source_bindings,
            "provider_metadata": provider},
        "auto_definitions": auto_defs, "auto_rows": trim(integration["auto"]["rows"]),
        "old_A_definitions": old_defs, "old_A_rows": trim(integration["old_A_retention"]["rows"]),
        "counterfactual_rows": integration["old_A_retention"]["counterfactual_diagnostics"]["rows"],
        "B_rows": trim(integration["B_retention"]["rows"]),
        "B_ordered_rows": trim(integration["B_retention"]["ordered_diagnostic"]["rows"]),
        "B_expected_representatives": definition["B_retention"]["values"],
        "C_core_final_parents": definition["C"]["families"]["final"],
        "C_auto_parents": definition["C"]["families"]["auto"],
        "C_primary_rows": choice["primary"]["rows"],
        "AV_selected_records": av["models"][selection["selected"]["av"]]["records"],
        "AV_all_model_ids": list(av["models"]), "sound_rows": sound_rows,
        "sound_chunk_controls": snd["chunk_controls"], "chain_rows": chain_rows,
        "selection": select(selection, ["created_at", "selected", "temperature", "threshold", "policy_feasible"]),
        "metrics": metrics, "checkpoint_inspection": checkpoints,
        "projection_limits": "No weights/WAV/full third-party texts copied. Auto B responses are only correctness flags in original integration JSON; actual B answers cross-check against the linked fixed B84 panel. Auto C per-service scores/choice_info logits and adapter outputs were not retained. Gate recheck uses preserved predictions/event analyses/SNR statistics, not new waveform analysis or model inference.",
    }
    save(HERE / "measurement_projection_v1.json", result)
    return {"projection_bytes": (HERE / "measurement_projection_v1.json").stat().st_size,
            "projection_sha256": sha((HERE / "measurement_projection_v1.json").read_bytes()),
            "all_original_final_members_checked": len(checked)}


def softmax(logits, temperature):
    values = [value / temperature for value in logits]
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def analysis_ok(analysis, metrics):
    measured, expected = analysis["measured"], analysis["expected"]
    if not measured["legal"] or len(measured["events"]) != len(expected):
        return False
    return all(abs(a["frequency"] - b["frequency"]) <= metrics["frequency_relative_tolerance"] * b["frequency"]
               and abs(a["onset_ms"] - b["onset_ms"]) <= metrics["timing_tolerance_ms"]
               and abs(a["offset_ms"] - b["offset_ms"]) <= metrics["timing_tolerance_ms"]
               for a, b in zip(measured["events"], expected, strict=True))


def artificial_probes():
    import torch
    from tiny_perceptron.omni_products.routing import Request, routing_question, b_sum_adapter
    from tiny_perceptron.omni_products.inference import CJudge, OmniAssistant
    from tiny_perceptron.omni_products.choice import Candidate, Question, text_bytes
    from tiny_perceptron.lesson_products.inference import parse_state, rule_depth
    from tiny_perceptron.lesson_products.model import Tokenizer, padded

    body = (ROOT / SNAPSHOT).read_text()
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(blocks) == 2
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        for block in blocks:
            exec(compile(block, SNAPSHOT, "exec"), namespace)
    assert (namespace["gold_ok"], namespace["route_ok"], namespace["whole_ok"]) == (3, 2, 1)
    adapter = namespace["adapt_sum"]
    tokenizer = Tokenizer()
    lengths = []
    for numbers in itertools.product(range(7), repeat=3):
        string = adapter(list(numbers))
        assert string == b_sum_adapter(numbers)
        assert parse_state(string) == {"task": "sum"}
        tokens = tokenizer.encode(string)
        padded([tokens], 128)
        lengths.append(len(tokens))
    illegal = [None, [], [1, 2], [1, 2, 3, 4], [True, 2, 3], [1., 2, 3], [1, 2, 7], [-1, 2, 3], ["1", 2, 3]]
    for numbers in illegal:
        try:
            adapter(numbers)
        except ValueError:
            pass
        else:
            raise AssertionError(numbers)
    dispatch = namespace["select_and_dispatch"]
    fixture = {"answer": "none", "p_max": .96}
    exercise = [dispatch(fixture, {"numbers": [1, 2, 3]}, namespace["demo_dispatch"]),
                dispatch(fixture, {"numbers": [1, 2, 3]}, namespace["demo_dispatch"], manual="B")]
    assert exercise[0]["route"] == "none" and exercise[1]["route"] == "B"
    assert exercise[1]["result"]["loops"] == 1
    # Actual dispatcher methods, with recording stubs replacing every learned backend.
    calls = []
    fake_b = SimpleNamespace(auto_policy={"version": 1, "sum_depth": 1},
                             solve=lambda state, loops: calls.append((state, loops)) or {"stub": True, "loops": loops})
    product = object.__new__(OmniAssistant)
    product.old_b = fake_b
    product.c = SimpleNamespace(route=lambda request: {"answer": "B", "p_max": .01})
    success = product.auto(Request("三個整數相加"), {"numbers": [1, 2, 3]})
    failure = product.auto(Request("三個整數相加"), {"numbers": [1, 2, 7]})
    assert success["route"] == "B" and not success["contract_error"]
    assert failure["route"] == "B" and failure["contract_error"] and len(calls) == 1
    assert calls == [("三數加法;x=1;y=2;z=3", 1)]
    assert set(routing_question(Request("判斷片段", audio_present=True, video_present=True)).state.splitlines()) == {
        "request: 判斷片段", "image=0 audio=1 video=1"}
    assert text_bytes("新說法🙂", 512, "state") == "新說法🙂".encode("utf-8")
    state = "A: task=record; permission=yes; ready=yes."
    choice = Question(state, "choice", "Choose A's action. Require permission and ready known; no permission: stop; not ready: wait; else requested task.",
                      (Candidate("record", "Device A: record"),))
    info = Question(state, "info", "Are permission and ready both known for A's action?",
                    (Candidate("yes", "Required information is sufficient"), Candidate("no", "Required information is missing")))
    judge = object.__new__(CJudge)
    judge.threshold = .6
    judge.predict = lambda question: {"answer": "record" if question.kind == "choice" else "yes", "p_max": .6}
    assert judge.judge(choice, info)["accepted"]
    judge.predict = lambda question: {"answer": "record" if question.kind == "choice" else "no", "p_max": 1.}
    assert not judge.judge(choice, info)["accepted"]
    return {"exact_inline_stdout": output.getvalue(), "legal_inline_and_actual_adapter_inputs_checked": 343,
            "rejected_inline_cases": len(illegal), "boolean_is_int_subclass": isinstance(True, int),
            "boolean_passes_exact_int_check": type(True) is int, "old_B_token_lengths": sorted(set(lengths)),
            "exercise": exercise, "actual_dispatch_recording_stub": {"success": success, "failure": failure, "calls": calls},
            "C_same_need_acceptance_recording_stubs": "p=.6/info=yes accepted; info=no handed off",
            "model_weights_loaded": 0, "neural_forward_calls": 0, "training_updates": 0,
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"}}


def replay():
    from tiny_perceptron.selftrained.tools import parse_tool_call, ToolCallError

    data = read(HERE / "measurement_projection_v1.json")
    defs = {row["family"]: row for row in data["auto_definitions"]}
    assert len(defs) == len(data["auto_rows"]) == 100
    assert len({json.dumps(row["request"], sort_keys=True) for row in defs.values()}) == 100
    assert not (set(map(tuple, data["C_core_final_parents"])) & set(map(tuple, data["C_auto_parents"])))
    b_rows = {tuple(row["parent"]): row for row in data["B_rows"]}
    assert set(b_rows) == set(itertools.combinations_with_replacement(range(7), 3))
    assert set(map(tuple, data["B_expected_representatives"])) == set(b_rows)
    b_correct = {name: 0 for name in ("old", "wrapper")}
    for parent, row in b_rows.items():
        assert row["gold"] == sum(parent)
        for name in b_correct:
            actual = row[name]["answer"] == sum(parent)
            assert actual == row[name + "_correct"] and row[name]["loops"] == 1
            b_correct[name] += actual
    ordered = data["B_ordered_rows"]
    assert {tuple(row["numbers"]) for row in ordered} == set(itertools.product(range(7), repeat=3))
    for row in ordered:
        assert row["parent"] == sorted(row["numbers"])
        assert row["gold"] == sum(row["numbers"])
        assert row["correct"] == (row["actual"]["answer"] == row["gold"])
        assert row["actual"]["loops"] == 1
    table = {name: Counter() for name in ("A", "B", "C", "none")}
    composition = Counter()
    route_disagreements = []
    max_probability_delta = 0.
    missing = Counter()
    for row in data["auto_rows"]:
        definition = defs[row["family"]]
        route = definition["gold_route"]
        assert route == row["gold_route"]
        correct_route = route == row["actual_route"]
        assert correct_route == row["route_correct"]
        prediction = row["route_logits"]
        p = softmax(prediction["logits"], data["selection"]["temperature"])
        max_probability_delta = max(max_probability_delta, *(abs(a - b) for a, b in zip(p, prediction["probabilities"], strict=True)))
        chosen = min(range(len(p)), key=lambda i: (-p[i], prediction["ids"][i]))
        assert prediction["answer"] == row["actual_route"] == prediction["ids"][chosen]
        assert abs(max(p) - prediction["p_max"]) < 1e-12
        for key in ("gold_route_result", "real_route_result"):
            if key not in row:
                continue
            result = row[key]
            if route == "C":
                assert result["correct"] == (result["actual"] == definition["gold"])
                if "logits" not in result:
                    missing["C_service_per_question_logits"] += 1
                if "information" not in result and "choice_info" not in result:
                    missing["C_service_corresponding_information_prediction"] += 1
            elif route == "B":
                linked = b_rows[tuple(definition["numbers"])]
                assert result["correct"] == linked["wrapper_correct"]
                if "answer" not in result and "actual" not in result:
                    missing["B_actual_answer_in_Auto_result"] += 1
            elif route == "A":
                if row["family"].startswith("auto-A-calculator"):
                    trace = result["tool_trace"]
                    assert trace["executed"] is False and result["correct"] is False
                elif definition["payload_kind"] == "video_audio_definition":
                    assert result["actual"] == definition["gold"]
                    assert result["correct"] == analysis_ok(result["sound_analysis"], data["metrics"]["sound"])
                else:
                    assert result["correct"] == (result["actual"] == definition["gold"])
        if route == "A":
            composition[definition.get("old_A_task", definition["payload_kind"])] += 1
        if not correct_route:
            route_disagreements.append({"family": row["family"], "gold_route": route, "actual_route": row["actual_route"],
                                        "gold_success": row["gold_route_end_to_end_correct"]})
        real = correct_route and not row["contract_error"] and row.get("real_route_result", {}).get("correct", False)
        assert real == row["real_route_end_to_end_correct"]
        assert row["gold_route_result"]["correct"] == row["gold_route_end_to_end_correct"]
        table[route].update(n=1, route_correct=correct_route, gold_chain=row["gold_route_end_to_end_correct"], real_chain=real)
    assert max_probability_delta < 1e-12
    old_defs = {(row["task"], row["family"], row["record"]): row for row in data["old_A_definitions"]}
    tasks = {}
    for row in data["old_A_rows"]:
        definition = old_defs[(row["task"], row["family"], row["source_record"])]
        task = tasks.setdefault(row["task"], Counter())
        task["observations"] += 1
        for name in ("old", "wrapper"):
            actual = row[name].get("actual")
            if row["task"] == "tool_call":
                try:
                    correct = parse_tool_call(actual).to_dict() == definition["gold_tool_call"]
                except ToolCallError:
                    correct = False
            else:
                correct = actual == definition["gold"]
                # All tool_reply strings already fail, independent of omitted executor traces.
                if row["task"] == "tool_reply":
                    assert not correct
            assert correct == row[name + "_correct"]
            task[name + "_correct"] += correct
    for task in tasks.values():
        task["difference_percentage_points"] = 100 * (task["wrapper_correct"] - task["old_correct"]) / task["observations"]
    cf = data["counterfactual_rows"]
    assert len(cf) == 24 and all(row["actual"] == row["gold"] and row["correct"] and not row["actual_executor_used"] for row in cf)
    av = data["AV_selected_records"]
    av_pair = sum(row["answers"] == row["gold"] for row in av)
    panels = {str(i): {"n": sum(row["panel"] == i for row in av),
                      "correct": sum(row["panel"] == i and row["answers"] == row["gold"] for row in av)} for i in range(4)}
    removed = {name: sum(row[name + "_answer"] == 2 for row in av) for name in ("remove_image", "remove_audio")}
    swap = sum(row["waveform_swap_answers"] == row["waveform_swap_gold"] for row in av)
    sound = data["sound_rows"]
    snd_correct = sum(row["eos"] and not row["capped"] and analysis_ok(row["analysis"], data["metrics"]["sound"]) for row in sound)
    snr = math.fsum(row["snr"] for row in sound) / len(sound)
    chain = sum(row["answer"] == row["gold"] and row["legal"] and analysis_ok(row["analysis_against_gold_program"], data["metrics"]["sound"]) for row in data["chain_rows"])
    c = data["C_primary_rows"]
    c_counts = {kind: sum(row["tasks"][kind]["answer"] == row["tasks"][kind]["gold"] for row in c) for kind in ("choice", "info", "rubric")}
    accepted, errors = 0, 0
    for row in c:
        choice = row["tasks"]["choice"]
        prob = softmax(choice["logits"], data["selection"]["temperature"])
        accept = max(prob) >= data["selection"]["threshold"] and choice["answer"] != "none" and row["choice_info"]["answer"] == "yes"
        error = choice["answer"] != choice["gold"] or not row["world_supported"] or not row["choice_information_sufficient"]
        accepted += accept
        errors += accept and error
    m = data["metrics"]
    a_pass = (av_pair >= m["AV"]["paired_minimum_correct"] and len(av) == 200
              and all(row["n"] == 50 and row["correct"] / row["n"] >= .75 for row in panels.values())
              and all(value / 200 >= .9 for value in removed.values()) and len(data["AV_all_model_ids"]) == 6
              and snr >= 10 and snd_correct / 200 >= .85 and all(row["legal"] for row in sound)
              and chain / 200 >= .8 and all(row["legal"] for row in data["chain_rows"])
              and len(data["sound_chunk_controls"]) == 10
              and all(row["chunks_bit_exact"] and row["future_code_prefix_bit_exact"] for row in data["sound_chunk_controls"])
              and all(row["difference_percentage_points"] >= -5 for row in tasks.values())
              and b_correct["old"] == b_correct["wrapper"])
    c_pass = (c_counts["choice"] >= 54 and c_counts["info"] >= 51 and c_counts["rubric"] >= 51
              and accepted >= 30 and errors / accepted <= .05)
    assert a_pass and not c_pass
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/omni_p5_technical_19_24",
        "projection_sha256": sha((HERE / "measurement_projection_v1.json").read_bytes()),
        "auto_table": table, "auto_A_composition": composition, "route_disagreements": route_disagreements,
        "max_route_probability_recalculation_abs_delta": max_probability_delta,
        "auto_C_core_parent_overlap": 0, "auto_trace_missing_fields_count_per_recorded_route": missing,
        "old_A_table": tasks, "old_A_observations": len(data["old_A_rows"]),
        "old_A_unique_parent_families": len({row["family"] for row in data["old_A_rows"]}),
        "counterfactual_correct": sum(row["correct"] for row in cf), "counterfactual_n": len(cf),
        "counterfactual_example": cf[0], "B_representatives": {"n": len(b_rows), **b_correct, "loops": 1},
        "B_ordered": {"n": len(ordered), "correct": sum(row["correct"] for row in ordered), "parents": len(b_rows)},
        "A_gate_support": {"paired_correct": av_pair, "n": len(av), "panels": panels, "deleted_cue_correct": removed,
            "waveform_swap_pair_correct": swap, "sound_gold_correct": snd_correct, "sound_n": len(sound),
            "mean_preserved_snr_db": snr, "actual_AV_chain_correct": chain, "actual_AV_chain_n": len(data["chain_rows"]), "passed": a_pass},
        "C_gate_support": {"correct": c_counts, "n_per_correlated_type": len(c), "accepted": accepted, "errors": errors, "risk": errors / accepted, "passed": c_pass},
        "thresholds_were_not_reselected": data["selection"], "artificial_CPU_probes": artificial_probes(),
        "limits": data["projection_limits"], "NN_inference_calls": 0, "training_updates": 0, "Modal_calls": 0}
    save(HERE / "replay_result_v1.json", result)
    return {key: result[key] for key in ("auto_table", "auto_A_composition", "old_A_observations", "old_A_unique_parent_families", "B_representatives", "B_ordered", "A_gate_support", "C_gate_support", "auto_trace_missing_fields_count_per_recorded_route")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "replay"))
    parser.add_argument("--raw-root")
    parser.add_argument("--core-archive")
    args = parser.parse_args()
    if args.mode == "collect":
        result = collect(args.raw_root, args.core_archive)
    else:
        sys.path.insert(0, str(ROOT))
        result = replay()
    print(json.dumps(result, ensure_ascii=False, indent=2))
