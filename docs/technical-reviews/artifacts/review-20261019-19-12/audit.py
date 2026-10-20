"""唯讀核對凍結評測的數字、評分契約與原始來源。"""

import ast
import collections
import hashlib
import json
import operator
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.selftrained import evaluate as ev
from scripts.selftrained import prepare_text_tools as text
from scripts.selftrained import prepare_vision_ocr as vision
from tiny_perceptron.selftrained.dataset import OCR_CHARACTERS, decode_ocr
from tiny_perceptron.selftrained.tools import execute_tool_call, run_tool_loop


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def write_new(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def fetch(url, version, locator, excerpt, note):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    content = raw.decode("utf-8")
    assert excerpt in content
    return {
        "url": url,
        "version": version,
        "locator": locator,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command": command,
        "full_response_sha256": sha(raw),
        "full_response_bytes": len(raw),
        "excerpt": excerpt,
        "inspection_note": note,
    }, content


def main():
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-12/audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "source_sha256": sha((ROOT / "docs/course-review-20261019/freeze-04/sources/19.12.md").read_bytes()),
        "inspection_scope": "只讀教材、實作、資料生成器與公開評測彙總；未讀其他角色報告，未重新生成模型回答。",
    }
    original_path = "docs/course-repair-20261008/reviews/freeze-01/evidence/technical-integration/evaluate-original-bf43be.py"
    original = (ROOT / original_path).read_bytes()
    old_tree = ast.parse(original)
    new_tree = ast.parse((ROOT / "scripts/selftrained/evaluate.py").read_bytes())
    old_symbols = {n.name: ast.dump(n, include_attributes=False) for n in old_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    new_symbols = {n.name: ast.dump(n, include_attributes=False) for n in new_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    same = [name for name in old_symbols if old_symbols[name] == new_symbols.get(name)]
    for name in ["score_reply", "summarize", "normalize", "positive_visual_answer", "numeric_reply_value", "format_pass", "edit_distance", "Generator", "validate_protocol"]:
        assert name in same
    result["evaluator_version"] = {"original_path": original_path, "original_sha256": sha(original), "identical_symbols": same,
        "changed_symbols": [name for name in old_symbols if name not in same],
        "difference_scope": "code_fingerprints 將路徑轉 POSIX 字串；main 將 fsync 的檔案開啟模式由 rb 改為 r+b。評分與彙總 AST 相同。"}
    index = read("docs/selftrained/results/public-raw/index.json")
    checked_files = []
    for entry in index["files"]:
        raw = (ROOT / entry["destination_path"]).read_bytes()
        assert sha(raw) == entry["sha256"] and len(raw) == entry["bytes"]
        checked_files.append({"path": entry["destination_path"], "sha256": sha(raw), "bytes": len(raw)})
    result["archive_integrity"] = checked_files
    capsule = read("docs/selftrained/results/v2-final-public-results.json")
    selection = read("docs/selftrained/results/v2-final-training-source-selection.json")
    metrics = {}
    result["architectures"] = {}
    expected = {
        "moe": {"text": 464, "ocr": 252, "vision_clothing": 357, "vision_relation": 1427, "voice_qa": 42, "voice_topic_continuation": 26, "tool_call": 0, "tool_reply": 551, "tool_concept": 6, "tool_missing": 5, "tool_unavailable": 29, "tool_unsupported": 4},
        "dense": {"text": 429, "ocr": 281, "vision_clothing": 356, "vision_relation": 1426, "voice_qa": 37, "voice_topic_continuation": 24, "tool_call": 7, "tool_reply": 468, "tool_concept": 3, "tool_missing": 8, "tool_unavailable": 24, "tool_unsupported": 4},
    }
    counts = {"text": 560, "ocr": 324, "vision_clothing": 360, "vision_relation": 1440, "voice_qa": 90, "voice_topic_continuation": 60, "tool_call": 276, "tool_reply": 552, "tool_concept": 8, "tool_missing": 8, "tool_unavailable": 48, "tool_unsupported": 8}
    for arch in ["moe", "dense"]:
        prefix = f"docs/selftrained/results/public-raw/{arch}"
        m = read(prefix + "/test/metrics.json")
        protocol = read(prefix + "/freeze/frozen.json")
        receipt = read(prefix + "/test/receipt.json")
        journal = read(prefix + "/test/evaluation-receipt.json")
        stage = read(f"docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json")
        training = read(f"docs/selftrained/results/training-raw/{'moe-native' if arch == 'moe' else 'dense-joint'}/raw/train-receipt.json")
        metrics[arch] = m
        assert protocol["code_sha256"]["scripts/selftrained/evaluate.py"] == sha(original)
        assert protocol["thresholds"] == m["thresholds"] == ev.THRESHOLDS
        assert protocol["test_once"] and protocol["selection"] == "validation only"
        assert protocol["created_at"] < receipt["started_at"] < journal["updated_at"]
        assert journal["protocol_sha256"] == sha((ROOT / prefix / "freeze/frozen.json").read_bytes())
        assert journal["completed_count"] == journal["expected_count"] == 3734
        assert journal["status"] == "complete" and journal["resumed_from"] is None
        assert m["resumed_completed_count"] == 0 and m["recovered_complete_tail_count"] == 0
        assert m["checkpoint_sha256"] == protocol["checkpoint_sha256"] == journal["checkpoint_sha256"]
        assert m["per_task_final_reply"] == capsule["architectures"][arch]["final_test"]["per_task_final_reply"]
        assert training["test_used_for_selection"] is False
        assert training["origin"]["kind"] == "all-neural-weights-random"
        selected = selection[arch + "_joint"]
        assert selected.get("sha256", selected.get("best_sha256")) == protocol["selected_checkpoint_sha256"]
        for task, count in counts.items():
            item = m["per_task_final_reply"][task]
            assert item["count"] == item["semantic"]["denominator"] == count
            assert item["semantic"]["numerator"] == expected[arch][task]
        assert sum(item["count"] for item in m["per_task_final_reply"].values()) == 3734
        strata = m["stratified_final_reply"]
        text_intents = {k: [v["semantic_numerator"], v["count"]] for k, v in strata.items() if k.startswith("text/intent=")}
        failed_text = [k for k, (n, d) in text_intents.items() if n / d < 0.8]
        assert len(text_intents) == 9 and len(failed_text) == (4 if arch == "moe" else 3)
        visual = {k: [v["semantic_numerator"], v["count"]] for k, v in strata.items() if "/query_class=" in k or "/query_position=" in k}
        assert all(n / d >= 0.8 for n, d in visual.values())
        voices = {k: [v.get("end_to_end_semantic_numerator", v["semantic_numerator"]), v["count"]] for k, v in strata.items() if k.startswith(("voice_qa/intent=", "voice_topic_continuation/intent="))}
        assert len(voices) == 6 and all(n / d < 0.8 for n, d in voices.values())
        formats = {k: [v["format_numerator"], v["format_denominator"]] for k, v in strata.items() if "/format=" in k}
        assert all(n / d >= 0.9 for n, d in formats.values())
        ocr = m["per_task_final_reply"]["ocr"]
        errors = sum(v["ocr_errors"] for k, v in strata.items() if k.startswith("ocr/case="))
        characters = sum(v["ocr_characters"] for k, v in strata.items() if k.startswith("ocr/case="))
        assert characters == 984 and errors == (72 if arch == "moe" else 43)
        assert errors / characters == ocr["final_cer"]
        assert m["perception"]["ocr"] == {"cer": 0.0, "exact_numerator": 324, "denominator": 324, "exact_rate": 1.0}
        for name, value in stage["tool_call_chain"].items():
            assert value == m["per_task_final_reply"]["tool_call"][name]
        statuses = stage["tool_statuses_by_task"]["tool_call"]
        assert statuses == ({"no_tool": 160, "invalid_call": 2, "executed": 114} if arch == "moe" else {"no_tool": 173, "invalid_call": 1, "executed": 102})
        assert m["voice_topic_continuation"]["end_to_end_correct"] == (26 if arch == "moe" else 24)
        result["architectures"][arch] = {
            "task_counts": {k: {f: v[f] for f in ["count", "source_groups", "recording_assets", "image_assets", "semantic", "format", "final_cer", "tool_parameters", "tool_executed", "tool_final"] if f in v} for k, v in m["per_task_final_reply"].items()},
            "total_questions": sum(counts.values()), "text_intents": text_intents, "failed_text_intents": failed_text,
            "visual_strata": visual, "voice_strata": voices, "format_strata": formats,
            "ocr_errors": errors, "ocr_characters": characters, "perception_ocr": m["perception"]["ocr"],
            "controls": m["controls"]["paired"], "tool_chain": stage["tool_call_chain"], "tool_statuses": statuses,
            "continuation": m["voice_topic_continuation"],
            "protocol": {k: protocol[k] for k in ["created_at", "thresholds", "checkpoint_sha256", "selected_checkpoint_sha256", "data_sha256", "max_new_tokens", "context", "selection", "test_once"]},
            "test_started_at": receipt["started_at"], "completed_ids_sha256": journal["completed_ids_sha256"],
            "training_selection": training["selection"], "training_seed": training["seed"],
            "origin_kind": training["origin"]["kind"], "training_test_used_for_selection": training["test_used_for_selection"],
        }
    assert result["architectures"]["moe"]["completed_ids_sha256"] == result["architectures"]["dense"]["completed_ids_sha256"]
    assert result["architectures"]["moe"]["protocol"]["data_sha256"] == result["architectures"]["dense"]["protocol"]["data_sha256"]
    text_rows = text.text_records()
    tool_rows = text.tool_records()
    test_rows = [r for r in text_rows + tool_rows if r["split"] == "test"]
    pairs = collections.defaultdict(list)
    for row in test_rows:
        sup = row["supervision"]
        if "pair_id" in sup:
            pairs[(sup["control_type"], sup["pair_id"])].append(row)
    sizes = collections.defaultdict(collections.Counter)
    for (kind, _), members in pairs.items():
        sizes[kind][len(members)] += 1
        if kind in ["history_topic", "history_format", "history_choice"]:
            assert len({r["messages"][-2]["content"] for r in members}) == 1
        if kind == "history_format":
            assert len({r["supervision"]["format"] for r in members}) == 2
        if kind == "history_choice":
            assert len({r["supervision"]["selected_option"] for r in members}) == 2
        if kind == "history_topic":
            assert len({r["supervision"]["topic"] for r in members}) == 3
    result["reconstructed_text_tools"] = {"test_counts": dict(collections.Counter(r["task"] for r in test_rows)), "control_group_sizes": {k: dict(v) for k, v in sizes.items()}, "history_control_last_question_fixed": True}
    app = next(r for r in test_rows if r["task"] == "text" and r["supervision"]["intent"] == "app_error" and r["supervision"]["format"] == "two_points")
    trace = {"tool_call": None, "status": "no_tool"}
    reply = "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
    result["app_example"] = ev.score_reply(app, reply, trace)
    assert result["app_example"]["semantic"] and result["app_example"]["format"]
    voice_source = (ROOT / "scripts/selftrained/prepare_voice.py").read_text()
    assert '"speaker_id": None' in voice_source and '"speaker_disjoint": False' in voice_source
    words = vision.word_pools()["test"]
    lengths = collections.Counter(map(len, words))
    result["ocr_construction"] = {"characters": OCR_CHARACTERS, "vocabulary_size": len(OCR_CHARACTERS), "test_composition_lengths": dict(lengths), "rows": 2 * len(words) + 2 * len(OCR_CHARACTERS), "reference_characters": 2 * sum(map(len, words)) + 2 * len(OCR_CHARACTERS), "roi_control_groups": len(words)}
    assert result["ocr_construction"]["rows"] == 324 and result["ocr_construction"]["reference_characters"] == 984
    result["derived_numbers"] = {"ocr_exact_percent": [round(252 / 324 * 100, 1), round(281 / 324 * 100, 1)], "ocr_cer_percent": [round(72 / 984 * 100, 1), round(43 / 984 * 100, 1)], "constant_bag_baseline": 120 / 360, "by_question": 12 / 18, "by_task": (1 + 0) / 2}
    calculator_count = 0
    for operation, function in [("add", operator.add), ("subtract", operator.sub), ("multiply", operator.mul)]:
        for a in range(100):
            for b in range(100):
                call = json.dumps({"tool": "calculator", "operation": operation, "a": a, "b": b})
                assert execute_tool_call(call)["result"] == function(a, b)
                calculator_count += 1
    row = next(r for r in tool_rows if r["task"] == "tool_call" and r["split"] == "test")
    expected_call = row["supervision"]["expected_call"]
    changed = dict(expected_call, a=(expected_call["a"] + 1) % 100)
    scripted = iter([json.dumps(changed), str(execute_tool_call(json.dumps(changed))["result"])])
    actual_trace = run_tool_loop(lambda history: next(scripted), row["messages"][:-1])
    chain_score = ev.score_reply(row, actual_trace["final_output"], actual_trace)
    assert chain_score["tool_final"] and not chain_score["tool_roundtrip"]
    logits = torch.full((7, 13), -10.0)
    for i, token in enumerate([1, 1, 0, 1, 2, 2, 0]):
        logits[i, token] = 10.0
    assert decode_ocr(logits) == OCR_CHARACTERS[0] * 2 + OCR_CHARACTERS[1]
    result["mechanism_probes"] = {"calculator_valid_cases": calculator_count, "wrong_parameters_correct_return_score": chain_score, "ctc_sequence": [1, 1, 0, 1, 2, 2, 0], "ctc_decoded": decode_ocr(logits), "normalization": ev.normalize(' 「大 小。 」\n'), "edit_distance_example": ev.edit_distance("大小入", "大小出")}
    urls = [
        ("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md", "89644892e4d85e24eaac8bacfd4f463576704203", "Model card：架構與權重倉庫說明", "This is the weight repository for Qwen3-VL-2B-Instruct.", "核對它是 Qwen 團隊發布的圖文模型權重倉庫；未用上游自述推論本課成品品質。"),
        ("https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "Whisper 與 Model details", "Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation", "核對這是 OpenAI 已訓練並發布的 ASR 模型；不把它當本課 MoE 的延續。"),
        ("https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md", "40ce77cb32a384e4d50a568e1ec39ac804019d33", "Data Fields", "### Data Fields\nThe data fields are the same among all splits.", "完整讀取 Data Fields 與 zh-CN 的 features，欄位為 path、audio、transcription、english_transcription、intent_class、lang_id，無 speaker/session ID。"),
        ("https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/cer.py", "v1.8.2", "_cer_update 與 _cer_compute", "errors += _edit_distance(list(pred_tokens), list(tgt_tokens))\n        total += len(tgt_tokens)", "逐題字元編輯距離相加，再除以所有 reference 字元數；與教材及本專案 evaluator 的公式一致。"),
    ]
    result["external_sources"] = []
    for args in urls:
        receipt, content = fetch(*args)
        if "minds14" in args[0]:
            receipt["field_names"] = ["path", "audio", "transcription", "english_transcription", "intent_class", "lang_id"]
            fields = content.split("### Data Fields", 1)[1].split("### Data Splits", 1)[0]
            assert "speaker" not in fields.lower() and "session" not in fields.lower()
        result["external_sources"].append(receipt)
    paths = [
        "scripts/selftrained/evaluate.py", original_path, "scripts/selftrained/prepare_text_tools.py", "scripts/selftrained/prepare_vision_ocr.py", "scripts/selftrained/prepare_voice.py", "scripts/selftrained/augment_ocr_v2.py", "scripts/selftrained/train.py", "tiny_perceptron/selftrained/dataset.py", "tiny_perceptron/selftrained/tools.py", "tiny_perceptron/natural_assistant.py", "scripts/natural_assistant.py", "docs/natural-assistant/v4/public-release.json", "docs/selftrained/v2-manifest.json", "docs/selftrained/results/v2-final-public-results.json", "docs/selftrained/results/v2-final-training-source-selection.json", "docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json", "docs/selftrained/results/training-raw/dense-joint/raw/train-receipt.json", "docs/selftrained/results/v2-moe-public-final-test-stage-index.json", "docs/selftrained/results/v2-dense-public-final-test-stage-index.json", "docs/selftrained/results/public-raw/index.json",
    ]
    paths += [item["path"] for item in checked_files]
    result["inputs"] = [{"path": p, "sha256": sha((ROOT / p).read_bytes())} for p in dict.fromkeys(paths)]
    result["limits"] = ["未下載權重或重新訓練／重新生成最後測試。", "公開封存僅保存彙總與收據，未保存逐題 raw outputs；工具狀態分類依 stage index，不能在這裡獨立重算逐題回答。", "本機教學資料包為 Git LFS pointer；以資料生成器與 frozen metadata 核對設計、計數與範圍，未解包逐像素／逐錄音複查。", "有限測試結果不能推出一般中文、任意照片、陌生字體或新說話者的能力。"]
    result["result"] = "所有執行的數值、版本與機制核對均符合預期。"
    write_new(ARTIFACT / "audit-results.json", result)
    print(json.dumps({"result": result["result"], "archive_files_checked": len(checked_files), "calculator_cases": calculator_count, "external_originals_checked": len(urls)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
