"""只核研究原始觀察與固定分母完整；全部失敗也可完成量測，不增加能力門檻。"""

import hashlib
import io
import json
import math
import re
import string
import wave
import zipfile
from pathlib import PurePosixPath

from scripts.omni_revision.core_evaluation import evaluate_c
from scripts.omni_revision.formal_local import FROZEN
from tiny_perceptron.omni_products.reliability import select_policy, select_temperature


def require(condition, message):
    if not condition:
        raise ValueError("成熟研究量測不完整：" + message)


def same(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def keyed(rows, expected, key, name):
    keys = [key(row) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected), name)
    return dict(zip(keys, rows))


def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def logits_prediction(value, question, include_gold=True):
    ids = [x["id"] for x in question["candidates"]] + (["none"] if question["kind"] == "choice" else [])
    require(value["ids"] == ids and len(value["logits"]) == len(ids), "Kev固定候選與所有logits")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value["logits"]), "Kev有限logits")
    best = min(range(len(ids)), key=lambda i: (-value["logits"][i], ids[i]))
    require(value["answer"] == ids[best], "Kev raw argmax與stable ID")
    if include_gold:
        require(value["gold"] == question["gold"], "Kev固定gold")


def kev_families(rows, definitions, language):
    fixed = {row["family"]: row for row in definitions}
    indexed = keyed(rows, fixed, lambda row: row["family"], "Kev固定家族不可缺漏或重複")
    for family, row in indexed.items():
        source = fixed[family]
        require(set(row["tasks"]) == {"choice", "info", "rubric"}, "Kev完整三型代表題")
        for kind, prediction in row["tasks"].items():
            logits_prediction(prediction, source["questions"][kind])
        logits_prediction(row["choice_info"], source["questions"]["choice_info"], include_gold=False)
        for key in ("panel", "world_supported", "choice_information_sufficient"):
            require(row[key] == source[key], "Kev固定panel與可支援資訊")
        require(row["source"] == source["source"] + "/" + language + "-state-view", "Kev表面語言身分")
        require(type(row["context_tokens"]) is int and 0 < row["context_tokens"] <= 2048, "Kev完整context")
        require(type(row["state_tokens"]) is int and 0 < row["state_tokens"] <= 384, "Kev完整state")
        require(finite(row["forward_seconds"]), "Kev實際forward時間")
    return rows


def check_kev(archive, result, job):
    from scripts.omni_revision.kev_research import ADAPTER, BASE, BASE_REVISION

    def read(name):
        return json.loads(archive.read(name))

    data = json.loads((FROZEN / "data-definition.json").read_bytes())
    calibration = kev_families(read("kev-calibration.json"), data["C"]["panels"]["calibration"], "english")
    policy = kev_families(read("kev-policy-predictions.json"), data["C"]["panels"]["policy"], "english")
    selection = read("kev-reliability-selection.json")
    temperature = select_temperature(calibration)
    selected_policy = select_policy(policy, temperature["temperature"])
    require(
        same(selection["temperature"], temperature) and same(selection["policy"], selected_policy),
        "Kev只由cal80／policy60選T／tau",
    )
    require(
        selection["head_temperature_during_raw_logits"] == 1.0
        and selection["upstream_temperature"] == 2.406050072164233
        and selection["core_selection_sha256"] == job["core_selection_sha256"]
        and selection["final_opened"] is False,
        "Kev raw／upstream／new T身分與選定順序",
    )
    raw_expected = [("calibration", "english", row) for row in calibration] + [
        ("policy", "english", row) for row in policy
    ]
    for language in ("english", "chinese"):
        views = read(f"kev-final-{language}.json")
        require(set(views) == {"raw", "upstream_temperature", "new_temperature_policy"}, "Kev每語言三個實際視角")
        rows = kev_families(views["raw"]["rows"], data["C"]["panels"]["final"], language)
        for name, value, threshold in (
            ("raw", 1.0, None),
            ("upstream_temperature", selection["upstream_temperature"], None),
            ("new_temperature_policy", temperature["temperature"], selected_policy["threshold"]),
        ):
            require(same(views[name], evaluate_c(rows, value, threshold)), "Kev每視角的raw列／三型度量／完整policy一致")
        raw_expected.extend(("final", language, row) for row in rows)
    raw = [json.loads(line) for line in archive.read("kev-raw-predictions.jsonl").decode().splitlines()]
    expected = {(split, language, row["family"]): row for split, language, row in raw_expected}
    indexed = keyed(
        raw, expected, lambda row: (row["split"], row["language"], row["prediction"]["family"]), "Kev260筆家族原始預測"
    )
    require(all(same(row["prediction"], expected[key]) for key, row in indexed.items()), "Kev raw JSONL與報表同一預測")
    cache = result["cache"]
    keyed(
        cache["rows"],
        [row["family"] for row in data["C"]["panels"]["dev"][:8]],
        lambda row: row["family"],
        "Kev八個dev cache控制",
    )
    for row in cache["rows"]:
        require(
            all(isinstance(row[k], str) and len(row[k]) == 64 for k in ("prefix_before", "prefix_after")),
            "Kev實際prefix fingerprint",
        )
        require(
            finite(row["uncached_cached_max_abs"]) and finite(row["Q1_Q2_Q1_probability_max_abs"]), "Kev有限cache誤差"
        )
    passed = all(
        row["prefix_before"] == row["prefix_after"]
        and row["uncached_cached_max_abs"] <= 1e-5
        and row["Q1_Q2_Q1_probability_max_abs"] <= 1e-5
        for row in cache["rows"]
    )
    require(
        cache["development_families"] == 8
        and cache["atol"] == 1e-5
        and cache["all_observed_within_tolerance"] is passed,
        "Kev cache總結保留失敗",
    )
    require(
        result["adapter"] == ADAPTER
        and result["base"] == BASE
        and result["base_revision"] == BASE_REVISION
        and result["families_per_language"] == 60
        and result["language_views_share_same60_families"] is True
        and result["own_training_used_Kev_outputs"] is False
        and result["option_isolation"] is False
        and same(result["selection"], selection),
        "Kev版本／完整分母與選擇報告",
    )


def normalized(text):
    words = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    text = " ".join(text.strip().lower().strip(string.punctuation + " ").split())
    return words[int(text)] if text.isdigit() and 0 <= int(text) < len(words) else text


def qwen_prediction(value, *, seed, speech=False, av=False):
    require(
        isinstance(value["text"], str)
        and type(value["new_tokens"]) is int
        and 0 <= value["new_tokens"] <= (16 if speech else 32),
        "Qwen有界實際文字輸出",
    )
    require(
        len(value["generated_ids"]) == value["new_tokens"]
        and all(type(x) is int and x >= 0 for x in value["generated_ids"]),
        "Qwen所有新token IDs",
    )
    require(value["seed"] == seed and finite(value["whole_completion_seconds"]), "Qwen固定seed與完整生成時間")
    expected = {
        "return_audio": speech,
        "use_audio_in_video": av,
        "thinker_max_new_tokens": 16 if speech else 32,
        "talker_max_new_tokens": 192,
        "thinker_do_sample": False,
        "talker_do_sample": False,
        "speaker": "Chelsie",
    }
    require(value["generation"] == expected, "Qwen事前固定generate設定")
    inputs = value["input"]
    require(
        inputs["input_ids"]["shape"][0] == 1 and 0 < inputs["input_ids"]["shape"][1] <= 1024,
        "Qwen processor實際input界",
    )
    for tensor in inputs.values():
        require(
            isinstance(tensor["shape"], list)
            and isinstance(tensor["dtype"], str)
            and isinstance(tensor["sha256"], str)
            and len(tensor["sha256"]) == 64,
            "Qwen每個實際processor tensor收據",
        )
    if av:
        require(inputs["video_grid_thw"]["values"] == [[4, 8, 8]], "Qwen8個112平方frame的實際grid")


def fixed_model_files(note, model, revision):
    require(
        note["schema"] == "omni-fixed-public-model-input-files-v1"
        and note["model"] == model
        and note["revision"] == revision
        and note["local_files_only_for_neural_load"] is True
        and note["implicit_tokens_permitted"] is False
        and note["ordinary_git_weight_bytes"] == 0,
        "完整公開模型snapshot固定來源",
    )
    files = note["files"]
    require(
        "config.json" in files
        and any(name.endswith(".safetensors") for name in files)
        and (not model.startswith("Qwen/") or "spk_dict.pt" in files),
        "固定snapshot含權重／config／speaker",
    )
    for name, row in files.items():
        path = PurePosixPath(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and type(row["bytes"]) is int
            and row["bytes"] >= 0
            and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None
            and (row["source_lfs_sha256"] is None or row["source_lfs_sha256"] == row["sha256"]),
            "固定snapshot逐檔bytes／sha收據",
        )


def parameter_precision(result):
    before = result["parameter_numel_by_dtype_after_load"]
    after = result["parameter_numel_by_dtype_after_speech"]
    require(set(before) == set(after) == {"thinker", "talker", "token2wav"}, "Qwen逐零件實際參數精度")
    for name in before:
        for counts in (before[name], after[name]):
            require(
                bool(counts)
                and set(counts) <= {"torch.bfloat16", "torch.float32", "torch.float16"}
                and all(type(value) is int and value > 0 for value in counts.values()),
                "Qwen實際numel／dtype，不是整個模型一個dtype",
            )
        require(sum(before[name].values()) == sum(after[name].values()), "Qwen精度轉換不更換參數數量")
    if any(row["generated"] for row in result["speech_outputs"]):
        require(set(after["token2wav"]) == {"torch.float32"}, "官方完成speech的Token2Wav實際FP32")


def observation(row, gold, seed, av=False):
    require(type(row["correct"]) is bool and row["gold"] == gold, "Qwen固定gold與每列成敗")
    if "prediction" in row:
        require("error_type" not in row, "Qwen成功與錯誤不可矛盾")
        qwen_prediction(row["prediction"], seed=seed, av=av)
        require(row["correct"] == (normalized(row["prediction"]["text"]) == gold), "Qwen逐列內容評分")
    else:
        require(
            isinstance(row["error_type"], str)
            and bool(row["error_type"])
            and isinstance(row["error"], str)
            and row["correct"] is False,
            "Qwen失敗仍保留原始列",
        )


def check_qwen(archive, result, job, input_zip):
    from scripts.omni_revision.qwen_inputs import MODEL, REVISION

    fixed_model_files(json.loads(archive.read("qwen-frozen-input-files.json")), MODEL, REVISION)
    fixed_model_files(
        json.loads(archive.read("asr-frozen-input-files.json")),
        "openai/whisper-small",
        "973afd24965f72e36ca33b3055d56a652f456b4d",
    )

    with zipfile.ZipFile(input_zip) as inputs:
        raw_cases = inputs.read("cases/cases.json")
    binding = job["input_files"]["cases/cases.json"]
    require(
        len(raw_cases) == binding["bytes"] and hashlib.sha256(raw_cases).hexdigest() == binding["sha256"],
        "Qwen事前固定材料manifest",
    )
    cases = json.loads(raw_cases)
    raw = [json.loads(line) for line in archive.read("qwen-raw-predictions.jsonl").decode().splitlines()]
    require(
        len(raw) == 96 and all(row["panel"] in ("audio-text", "AV-shift") for row in raw), "Qwen96個文字／影音原始嘗試"
    )
    audio = keyed(
        [row for row in raw if row["panel"] == "audio-text"],
        [(case["id"], modality) for case in cases["audio_text_pairs"] for modality in ("audio", "text")],
        lambda row: (row["id"], row["modality"]),
        "Qwen24語音／文字家族各兩筆",
    )
    av = keyed(
        [row for row in raw if row["panel"] == "AV-shift"],
        [(case["id"], which) for case in cases["av_shift_pairs"] for which in (0, 1)],
        lambda row: (row["id"], row["which"]),
        "Qwen24影音家族各兩筆",
    )
    expected_audio, expected_av = [], []
    for i, case in enumerate(cases["audio_text_pairs"]):
        for modality in ("audio", "text"):
            observation(audio[(case["id"], modality)], case["gold"], 20261029 + i)
        a, t = (audio[(case["id"], m)]["correct"] for m in ("audio", "text"))
        expected_audio.append({"id": case["id"], "gold": case["gold"], "audio": a, "text": t, "both_correct": a and t})
    for i, case in enumerate(cases["av_shift_pairs"]):
        for which, item in enumerate(case["pair"]):
            observation(av[(case["id"], which)], item["gold"], 20262029 + i, av=True)
        hits = [av[(case["id"], which)]["correct"] for which in (0, 1)]
        expected_av.append({"id": case["id"], "observations": hits, "both_correct": all(hits)})
    require(
        len(expected_audio) == len(expected_av) == 24
        and result["audio_text_pairs"] == expected_audio
        and result["AV_shift_pairs"] == expected_av,
        "Qwen配對與原始96筆同一分母",
    )
    for key, expected in (("audio_text_counts", expected_audio), ("AV_counts", expected_av)):
        require(
            result[key] == {"families": 24, "both_correct": sum(row["both_correct"] for row in expected)},
            "Qwen配對摘要計數",
        )
    speech = keyed(
        result["speech_outputs"],
        [case["id"] for case in cases["speech_output_prompts"]],
        lambda row: row["id"],
        "Qwen8個語音輸出原始嘗試",
    )
    original = keyed(
        [json.loads(x) for x in archive.read("qwen-speech-raw.jsonl").decode().splitlines()],
        speech,
        lambda row: row["id"],
        "Qwen8個原始生成紀錄",
    )
    proxy = keyed(
        [json.loads(x) for x in archive.read("qwen-asr-raw.jsonl").decode().splitlines()],
        speech,
        lambda row: row["id"],
        "Qwen8個ASR內容proxy結果或未嘗試原因",
    )
    for i, case in enumerate(cases["speech_output_prompts"]):
        row = speech[case["id"]]
        require(
            type(row["generated"]) is bool and row["gold_word"] == case["gold_word"] and same(row, proxy[case["id"]]),
            "Qwen語音列／proxy同一結果",
        )
        require(all(k in row and same(v, row[k]) for k, v in original[case["id"]].items()), "Qwen原始語音生成不被重寫")
        if row["generated"]:
            qwen_prediction(row["prediction"], seed=20263029 + i, speech=True)
            audio_result = row["audio"]
            require(
                type(audio_result["legal"]) is bool
                and audio_result["sample_rate"] == 24000
                and type(audio_result["samples"]) is int
                and audio_result["samples"] >= 0,
                "Qwen實際waveform格式",
            )
            if audio_result["legal"]:
                require(0 < audio_result["samples"] <= 192000, "Qwen語音8秒界")
                require(audio_result["path"] == f"result/speech/{i}.wav", "Qwen輸出waveform路徑")
                wav = archive.read(f"speech/{i}.wav")
                require(hashlib.sha256(wav).hexdigest() == audio_result["sha256"], "Qwen可播放檔案SHA")
                with wave.open(io.BytesIO(wav)) as source:
                    require(
                        (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes())
                        == (1, 2, 24000, audio_result["samples"]),
                        "Qwen實際PCM16檔案格式",
                    )
        else:
            require(
                isinstance(row["error_type"], str) and bool(row["error_type"]) and isinstance(row["error"], str),
                "Qwen生成失敗仍有原始原因",
            )
        attempted = row["generated"] and row["audio"]["legal"]
        require(row["ASR_proxy_attempted"] is attempted, "ASR只對合法已生成聲音嘗試")
        if not attempted:
            require(
                row["ASR_proxy_correct"] is None and isinstance(row["ASR_proxy_skip_reason"], str),
                "未有合法聲音不可捏造ASR成功",
            )
        elif "ASR_proxy_error_type" in row:
            require(
                row["ASR_proxy_correct"] is False and isinstance(row["ASR_proxy_error"], str), "ASR失敗留在原8筆分母"
            )
        else:
            require(
                isinstance(row["ASR_proxy_text"], str)
                and type(row["ASR_proxy_correct"]) is bool
                and row["ASR_proxy_correct"] == (normalized(row["ASR_proxy_text"]) == case["gold_word"]),
                "ASR文字內容proxy評分",
            )
    expected_counts = {
        "prompts": 8,
        "generated": sum(row["generated"] for row in speech.values()),
        "legal": sum(row.get("audio", {}).get("legal", False) for row in speech.values()),
        "ASR_proxy_attempted": sum(row["ASR_proxy_attempted"] for row in speech.values()),
        "ASR_proxy_correct_out_of_all8": sum(row["ASR_proxy_correct"] is True for row in speech.values()),
    }
    require(
        len(speech) == result["speech_count"] == 8 and result["speech_counts"] == expected_counts,
        "Qwen8個聲音及proxy完整分母",
    )
    require(
        result["model"] == MODEL
        and result["revision"] == REVISION
        and result["teacher_outputs_used_for_own_training"] is False
        and result["ASR_proxy"]["model"] == "openai/whisper-small"
        and result["ASR_proxy"]["revision"] == "973afd24965f72e36ca33b3055d56a652f456b4d",
        "Qwen／ASR固定權重與無自家訓練",
    )
    require(
        result["dtype"]
        == "BF16 requested neural load; official Token2Wav converts to FP32 when decoding speech; no quantization",
        "Qwen載入請求與官方精度路徑分開",
    )
    parameter_precision(result)
