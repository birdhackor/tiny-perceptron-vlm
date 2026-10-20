"""核對工具請求範例、資料家族及已保存的生成紀錄；不執行訓練。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import applications, common, tool_choice
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (ARTIFACT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def originals():
    specifications = [
        ("json", "https://docs.python.org/3.13/library/json.html", "Python 3.13.16；頁面更新 2026-10-01", "json.dumps", "Serialize obj to a JSON formatted str", "核對 dumps 的回傳型別與 dict 至 JSON object 的轉換；範例僅用有限整數與字串鍵。"),
        ("json_standard", "https://www.rfc-editor.org/rfc/rfc8259.txt", "RFC 8259，2017-12", "Objects；Strings；JSON Grammar", "An object structure is represented as a pair of curly brackets", "本節的大括號描述適用於請求的 object 形式；規範也允許 array、number、string、boolean 與 null。字串及 object 欄位名以 U+0022 雙引號界定。"),
        ("parameter", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.parameter.Parameter.html", "PyTorch 2.14 文件", "Parameter", "A kind of Tensor that is to be considered a module parameter", "模型的 Parameter 是註冊在 Module 的 Tensor，與本地工具呼叫的普通 JSON 輸入值分屬不同角色。"),
        ("ascii", "https://www.rfc-editor.org/rfc/rfc20.txt", "RFC 20，1969-10-16", "ASCII format for Network Interchange；Standard Code", "standard 7-bit ASCII embedded in an 8 bit byte whose high order bit is always 0", "核對字元表中的英文字母、數字及 CALC/COPY 所用標點；此編碼不涵蓋中文字元。"),
        ("toolformer", "https://arxiv.org/html/2302.04761v1", "arXiv:2302.04761v1，2023-02-09", "Approach：Executing API Calls；Model Finetuning；Inference", "what arguments to pass, and how to best incorporate the results into future token prediction", "原文把模型生成呼叫、外部執行 API、插入回傳後繼續生成分開。這支持分工概念，不表示本專案使用 Toolformer 的訓練法，也不證明只放入示範就必然學會。"),
    ]
    records = []
    for name, url, version, locator, excerpt, note in specifications:
        argv = ["curl", "--fail", "--silent", "--show-error", "--location", url]
        result = subprocess.run(argv, capture_output=True, check=True)
        raw = result.stdout
        text = raw.decode("utf-8")
        if "<html" in text[:4000].lower() or "<!doctype" in text[:4000].lower():
            parser = Text()
            parser.feed(text)
            text = " ".join(parser.parts)
        text = re.sub(r"\s+", " ", text)
        assert excerpt in text, "來源中找不到預定核對的摘錄"
        if name == "toolformer":
            start = text.index("Model Finetuning", text.index("Executing API Calls As a next step"))
            assert text.index("Inference", start) > start, "未定位到訓練與推論說明"
        print(json.dumps({"source": name, "curl_exit_code": result.returncode, "locator": locator}, ensure_ascii=False))
        records.append({"id": name, "url": url, "version": version, "locator": locator,
                        "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": argv,
                        "curl_exit_code": result.returncode, "original_sha256": sha(raw),
                        "original_bytes": len(raw), "excerpt": excerpt, "inspection_note": note})
    save("original-sources.json", records)


def audit():
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                   "platform": platform.platform()}
    torch.set_num_threads(1)
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/B.1.md").read_bytes()
    assert sha(frozen) == "ca9a250652f416d1355dc8e1d333087fbcb93cfbcb9df9f32dd0554b3f791b23", "凍結原文不符"
    run = ROOT / "outputs/reviewer-tools/runs/20261020T073311635182Z"
    section = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0B.md#B.1" --execute',
               "execution": json.loads((run / "execution.json").read_bytes()),
               "environment": json.loads((run / "environment.json").read_bytes()),
               "extraction": json.loads((run / "extraction.json").read_bytes()),
               "stdout": (run / "stdout.txt").read_text(), "stderr": (run / "stderr.txt").read_text(),
               "code": (run / "fence-1.py").read_text()}
    assert (run / "section.md").read_bytes() == frozen, "執行的小節和凍結版不同"
    assert section["execution"]["exit_code"] == 0 and not section["stderr"], "小節執行失敗"
    save("section-execution.json", section)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(section["code"], namespace)
    assert isinstance(namespace["request"], dict) and isinstance(namespace["text"], str), "資料型別不符"
    assert json.loads(namespace["text"]) == namespace["request"], "序列化往返不符"
    request = copy.deepcopy(namespace["request"])
    assert call_tool(request) == 5535, "原題計算不符"
    request["arguments"]["b"] = 54
    changed = json.dumps(request)
    assert json.loads(changed) == request and call_tool(request) == 6642, "錯誤參數案例不符"
    diagnoses = {}
    for name, raw in [("format", "{"), ("tool", '{"name":"divide","arguments":{"a":123,"b":45}}')]:
        try:
            action = applications._parse_json_action(raw)
            call_tool(action)
        except (ValueError, TypeError, KeyError) as error:
            diagnoses[name] = {"raw_request": raw, "error": str(error)}
        else:
            raise AssertionError("錯誤請求未被攔下")
    diagnoses["arguments"] = {"question": "123乘45", "raw_request": changed, "result": call_tool(request), "matches_question": False}

    records = applications._tool_records()
    splits = common.split_records(records, 42)
    saved_path = ROOT / "docs/course-experiments/results/tools.json"
    saved = json.loads(saved_path.read_bytes())
    summaries = applications._split_summary(splits)
    assert summaries == saved["results"]["split"], "重建資料切分和已保存紀錄不同"
    families = {key: {row["family"] for row in rows} for key, rows in splits.items()}
    intersections = {f"{a}/{b}": sorted(families[a] & families[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    assert not any(intersections.values()), "家族跨集合"
    assert len(records) == 210 and len(set(row["family"] for row in records)) == 65, "資料列數或家族數不符"
    assert all(message["content"].isascii() for row in records for message in row["messages"]), "工具資料含非 ASCII 文字"
    assert all(0 <= row["a"] <= 9 and 0 <= row["b"] <= 9 for row in records if row["operation"] != "copy"), "運算數字超出範圍"
    example_family = [row for row in records if row["family"] == "pair:2:3"]
    assert {(row["operation"], row["a"], row["b"]) for row in example_family} == {("add", 2, 3), ("add", 3, 2), ("multiply", 2, 3), ("multiply", 3, 2)}, "家族未涵蓋運算和交換順序"
    copies = [row for row in records if row["operation"] == "copy"]
    assert len(copies) == 10 and all(json.loads(row["messages"][-1]["content"]) == {"done": True, "answer": row["answer"]} for row in copies), "COPY 標籤不符"

    historic_oid = "99393a5d6ab342099e6d9cc267895acb25256ef6"
    historic = subprocess.run(["git", "cat-file", "blob", historic_oid], capture_output=True, check=True).stdout
    expected_hash = saved["code_sha256"]["scripts/course_experiments/applications.py"]
    assert sha(historic) == expected_hash, "歷史程式與實驗紀錄的 SHA 不同"
    old_tree, new_tree = ast.parse(historic), ast.parse((ROOT / "scripts/course_experiments/applications.py").read_bytes())
    functions = ("_tool_records", "_tool_episode", "run_tools")
    equivalent = {}
    excerpts = []
    for name in ("_sample",) + functions:
        old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        excerpts.append(ast.get_source_segment(historic.decode(), old))
        if name in functions:
            new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            equivalent[name] = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    assert all(equivalent.values()), "工具資料、執行或訓練流程已改變"
    (ARTIFACT / "historical-tools-source.txt").write_text(
        "實驗紀錄所綁定的程式摘錄\nGit blob: " + historic_oid + "\n完整檔 SHA-256: " + sha(historic) + "\n\n" + "\n\n".join(excerpts) + "\n", encoding="utf-8")

    tok = ByteTokenizer()
    lookup = {row["messages"][1]["content"]: row for row in records}
    event_count, call_count, final_count, message_alias_rows, invalid_json_events = 0, 0, 0, 0, 0
    for episode in saved["results"]["samples"] + saved["results"]["one_step_cap_samples"]:
        row = lookup[episode["question"]]
        messages = copy.deepcopy(row["messages"][:2])
        for event in episode["trace"]:
            generation = event["generation"]
            assert generation["input_ids"] == applications._prompt_ids(messages, tok), "原始輸入 token 與重建對話不符"
            message_alias_rows += generation["messages"] != messages
            raw = generation["samples"][0]
            ids = raw["generated_ids"][:-1] if raw["eos"] else raw["generated_ids"]
            assert tok.decode(ids) == raw["generated"], "原始生成 token 與文字不符"
            try:
                action = applications._parse_json_action(raw["generated"])
            except ValueError:
                assert not event["executed"] and "parsed" not in event and "error" in event, "無效 JSON 沒有保留拒絕證據"
                invalid_json_events += 1
                event_count += 1
                continue
            assert action == event["parsed"], "解析後請求與原始文字不符"
            if event["executed"]:
                result = call_tool(action)
                assert result == event["tool_result"], "工具回傳和實際請求計算不符"
                messages += [{"role": "assistant", "content": raw["generated"]}, {"role": "user", "content": f"TOOL_RESULT:{result}"}]
                call_count += 1
            elif event.get("done"):
                assert action["answer"] == episode["answer"], "完成答案與原始生成不符"
                final_count += 1
            event_count += 1

    probe_messages = []
    actions = iter(['{"name":"multiply","arguments":{"a":2,"b":4}}', '{"done":true,"answer":8}'])

    def wrong_parameters(model, messages, ctx, **kwargs):
        probe_messages.append(copy.deepcopy(messages))
        return {"samples": [{"generated": next(actions), "invalid_special_tokens": []}], "generated_tokens": 1}

    probe_row = next(row for row in records if row.get("a") == 2 and row.get("b") == 3 and row["operation"] == "multiply")
    with patch.object(applications, "_sample", wrong_parameters):
        probe = applications._tool_episode(None, probe_row, None)
    assert probe["answer"] == 8 and not probe["correct"] and not probe["trace"][0]["request_matches_question"], "外層意外改正了錯誤參數"
    assert probe_messages[1][-1]["content"] == "TOOL_RESULT:8", "回填的不是實際工具結果"

    ctx = common.Context("cpu", ARTIFACT, ARTIFACT, ROOT / "assets/training", seed=42)
    model_a = common.new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
    model_b = common.new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
    ctx.seed = 43
    model_c = common.new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
    assert torch.equal(model_a.embedding.weight, model_b.embedding.weight), "相同 seed 未重現初始化"
    assert not torch.equal(model_a.embedding.weight, model_c.embedding.weight), "更換 seed 未改變隨機權重"
    choices = tool_choice.build_records()
    assert {row["expected_action"] for row in choices} == {"DIRECT", "TOOL", "ASK"}, "選卡輸出標籤不符"
    assert all(row["messages"][-1]["content"] == row["expected_action"] for row in choices), "選卡模型混入工具參數或最終答案"
    choice_model = common.new_lm(ctx, width=64, layers=2, max_length=160, heads=2)
    assert choice_model is not model_a, "兩個實驗沒有各自建立模型"
    choice_saved = json.loads((ROOT / "docs/course-experiments/results/tool_choice.json").read_bytes())
    choice_summary = {"experiment_id": choice_saved["experiment_id"], "device": choice_saved["device"], "seed": choice_saved["seed"],
                      "steps": choice_saved["results"]["training"]["steps"], "checkpoint": choice_saved["results"]["training"]["checkpoint"],
                      "parameters": choice_saved["results"]["training"]["parameters"], "records": choice_saved["results"]["training"]["records"]}
    assert choice_summary["steps"] == 900 and choice_summary["parameters"] == choice_model.description()["parameters"], "選卡模型紀錄與配方不符"

    cli = []
    for arguments in (["--help"], ["--list-assets", "tools"]):
        argv = [str(ROOT / ".venv/bin/python"), "scripts/course_experiments/run.py", *arguments]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        cli.append({"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    source_files = ["scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py", "scripts/course_experiments/tool_choice.py", "scripts/build_course.py", "tiny_perceptron/retrieval.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "docs/course-experiments/plan.json", "docs/course-experiments/results/tools.json", "docs/course-experiments/results/tool_choice.json"]
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-1/audit.py", "environment": environment,
              "source_sha256": sha(frozen), "result": "全部斷言通過；未訓練模型", "numeric": {"123*45": 5535, "123*54": 6642, "difference": 1107},
              "serialization": {"request_type": type(namespace["request"]).__name__, "text_type": type(namespace["text"]).__name__, "text": namespace["text"], "changed_text": changed},
              "diagnoses": diagnoses, "dataset": {"records": len(records), "families": len(set(row["family"] for row in records)), "operations": dict(Counter(row["operation"] for row in records)), "split": summaries, "family_intersections": intersections, "example_family": [{k: row[k] for k in ("family", "operation", "a", "b")} for row in example_family], "all_messages_ascii": True, "contains_123_times_45": False},
              "historical_code": {"git_blob": historic_oid, "full_source_sha256": sha(historic), "matches_recorded_sha256": True, "current_function_ast_equivalent": equivalent, "prior_messages_alias_events": message_alias_rows, "note": "歷史 messages 欄位可能因後續回填共享 list 而變長；本次以不變的 input_ids 重建當時真實輸入，沒有把後來的 list 當作當時輸入。"},
              "saved_experiment": {"device": saved["device"], "gpu": saved["gpu"], "python": saved["python_version"], "torch": saved["torch_version"], "seed": saved["seed"], "training_steps": saved["results"]["training"]["steps"], "training_records": saved["results"]["training"]["records"], "evidence_status": saved["evidence_status"], "regular_episodes": len(saved["results"]["samples"]), "capped_episodes": len(saved["results"]["one_step_cap_samples"]), "events_verified": event_count, "invalid_json_events": invalid_json_events, "tool_calls_recomputed": call_count, "model_final_actions_verified": final_count, "quality_not_remeasured": True},
              "wrong_parameter_probe": {"sampling": "以預設錯誤輸出替換抽樣器，單獨核對外層；不是模型品質測量", "episode": probe, "observed_messages": probe_messages},
              "random_initialization": {"same_seed_equal": True, "changed_seed_differs": True, "parameters": model_a.description()["parameters"], "pretrained_checkpoint_loaded": False},
              "separate_action_model": {"labels": sorted({row["expected_action"] for row in choices}), "records": len(choices), "max_length": choice_model.config.max_length, "shared_instance": False, "saved_record": choice_summary},
              "cli": cli, "source_files_sha256": {name: sha((ROOT / name).read_bytes()) for name in source_files},
              "limitations": ["沒有重跑訓練，也沒有重新量測 NVIDIA L4 的品質或速度。", "CLI 僅執行 help 與列資產；CPU 選項及預設輸出路徑另以 main/execute 原始碼核對。", "隨機初始化與故障注入僅核對機制；既有 GPU 執行事實依已保存結果、資料 hash、原始 token 與匹配的歷史程式核對。"]}
    save("audit.json", result)
    originals()
    print(json.dumps({"result": result["result"], "dataset_records": len(records), "events_verified": event_count, "tool_calls_recomputed": call_count, "environment": environment}, ensure_ascii=False))


if __name__ == "__main__":
    audit()
