"""核對引用頁的變動、既有證據版本與目前的小節執行。"""

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.training import choose_device, load_checkpoint

ARTIFACT = ROOT / "docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-01.json"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_14-1/technical-14-1-original-01.json"
TRAINING = "course/training.md"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def section(raw, name):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, item in enumerate(headers) if item.group().startswith(("## " + name + " ").encode()))
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


class CapturedArguments(Exception):
    pass


def capture_arguments(module_name, arguments):
    module = importlib.import_module(module_name)
    parsed = {}
    parse = argparse.ArgumentParser.parse_args

    def capture(parser, args=None, namespace=None):
        value = parse(parser, args, namespace)
        parsed["arguments"] = {key: str(item) if isinstance(item, Path) else item for key, item in vars(value).items()}
        parsed["options"] = sorted(option for action in parser._actions for option in action.option_strings)
        raise CapturedArguments

    with patch.object(sys, "argv", [module_name, *arguments]), patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            module.main()
        except CapturedArguments:
            pass
    assert parsed
    return parsed


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def main():
    original_raw = ORIGINAL.read_bytes()
    assert sha(original_raw) == "c01c1032d61220c9325c25c7bee312c02fe5a69fbdff89039726310e8a3e499c"
    original = json.loads(original_raw)
    assert original["issues"] == []
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-01.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "original_report": {"path": ORIGINAL.relative_to(ROOT).as_posix(), "sha256": sha(original_raw)},
        "resolved_issue_ids": [],
    }
    old_artifacts = []
    for item in original["artifacts"]:
        observed = sha((ROOT / item["path"]).read_bytes())
        assert observed == item["sha256"]
        old_artifacts.append({"path": item["path"], "sha256": observed, "unchanged": True})
    result["previous_artifacts"] = old_artifacts
    checked_sources = []
    for item in original["sources"]:
        if "path" not in item:
            continue
        observed = sha((ROOT / item["path"]).read_bytes())
        checked_sources.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"],
                                "current_sha256": observed, "unchanged": observed == item["sha256"]})
        assert item["path"] == TRAINING or observed == item["sha256"]
    result["repository_sources"] = checked_sources
    current_training = (ROOT / TRAINING).read_bytes()
    assert sha(current_training) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    old_training = subprocess.run(["git", "show", "5ef56bc756f3:" + TRAINING], cwd=ROOT, capture_output=True, check=True).stdout
    new_training = subprocess.run(["git", "show", "6a7aa1de5227:" + TRAINING], cwd=ROOT, capture_output=True, check=True).stdout
    assert sha(old_training) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert new_training == current_training
    change = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", TRAINING])
    old_part, new_part = section(old_training, "T.8"), section(current_training, "T.8")
    assert old_part == new_part
    result["training_change"] = {"diff": change, "previous_sha256": sha(old_training), "current_sha256": sha(current_training),
                                 "referenced_section": "架構比較一次只換一個條件",
                                 "referenced_section_unchanged": True, "referenced_section_sha256": sha(new_part),
                                 "referenced_section_text": new_part.decode(),
                                 "impact": "引用的固定配方與本機練習沒有變動；裝置說明改為逐工具區分，與明確指定CUDA的固定配方相容。"}
    parsed = {}
    for name, args in (
        ("scripts.train", []), ("scripts.evaluate", ["unread.pt"]), ("scripts.infer", ["unread.pt"]),
        ("scripts.quantize", ["unread.pt"]), ("scripts.prepare_data", []), ("scripts.fetch_training_assets", []),
        ("scripts.course_experiments.run", ["--experiment", "modern"]),
    ):
        parsed[name] = capture_arguments(name, args)
    for name in ("scripts.train", "scripts.evaluate", "scripts.infer"):
        assert parsed[name]["arguments"]["device"] == "auto"
    assert parsed["scripts.course_experiments.run"]["arguments"]["device"] == "cpu"
    explicit = capture_arguments("scripts.course_experiments.run", ["--experiment", "modern", "--device", "cuda"])
    assert explicit["arguments"]["device"] == "cuda"
    for name in ("scripts.quantize", "scripts.prepare_data", "scripts.fetch_training_assets"):
        assert "--device" not in parsed[name]["options"]
    paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py",
             "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py",
             "tiny_perceptron/training.py", "tiny_perceptron/quantization.py"]
    source_details = {}
    for name in paths:
        raw = (ROOT / name).read_bytes()
        lines = raw.decode().splitlines()
        source_details[name] = {"sha256": sha(raw), "relevant_current_lines": [line for line in lines
            if any(term in line for term in ("choose_device", "load_checkpoint", '"--device"', "device=", 'map_location="cpu"'))]}
    auto = []
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
            observed = str(choose_device())
            explicit_cpu = str(choose_device("cpu"))
        assert observed == expected and explicit_cpu == "cpu"
        auto.append({"mock_cuda_available": cuda, "mock_mps_available": mps, "auto": observed, "explicit_cpu": explicit_cpu})
    quantize_module = importlib.import_module("scripts.quantize")
    quantize_load = {}

    def no_read(*args, **kwargs):
        bound = inspect.signature(load_checkpoint).bind(*args, **kwargs)
        bound.apply_defaults()
        quantize_load.update({key: str(value) for key, value in bound.arguments.items()})
        raise CapturedArguments

    with patch.object(sys, "argv", ["quantize", "unread.pt"]), patch.object(quantize_module, "load_checkpoint", no_read):
        try:
            quantize_module.main()
        except CapturedArguments:
            pass
    assert quantize_load["device"] == "cpu"
    from scripts.course_experiments.run import experiment_spec
    route = experiment_spec("modern")
    assert route["module"] == "architecture" and route["function"] == "run_modern"
    result["device_and_route"] = {"parsed": parsed, "explicit_fixed_recipe": explicit,
                                   "choose_device_mock_branches": auto, "quantize_load_arguments": quantize_load,
                                   "modern_spec": route, "source_details": source_details,
                                   "help": command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]),
                                   "scope": "只執行參數解析與裝置物件選擇，不讀檢查點、不訓練、不呼叫GPU。"}
    url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    response = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    parser = Text()
    parser.feed(response.stdout.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
    assert excerpt in text
    title = re.search(r"<title>(.*?)</title>", response.stdout.decode(), re.S).group(1)
    result["reproducibility_source"] = {"url": url, "version": "PyTorch 2.14 官方文件；" + title,
                                         "accessed_on": datetime.now(timezone.utc).date().isoformat(),
                                         "command": argv, "sha256": sha(response.stdout), "short_excerpt": excerpt,
                                         "locator": "Reproducibility 開頭的跨版本與平台限制",
                                         "inspection_note": "讀開頭限制：同seed亦無跨平台完全重現保證，支持『不同電腦可能有不同結果』，沒有因此推定任何特定電腦必然不同。"}
    result["dependency_sentence_checks"] = [
        {"sentence": "訓練、評估與推論工具省略device時自動選擇裝置；可明確指定CPU。", "status": "verified",
         "basis": "CLI預設auto；choose_device已讀原碼並模擬CUDA、MPS、CPU分支。"},
        {"sentence": "scripts.course_experiments.run 省略裝置參數時使用CPU。", "status": "verified",
         "basis": "實際解析 --experiment modern，device=cpu；固定配方明確傳cuda時為cuda。"},
        {"sentence": "quantize.py使用CPU；資料準備、資料下載與量化三個工具均沒有device參數。", "status": "verified",
         "basis": "量化實際到load_checkpoint呼叫前攔截得到device=cpu；三份parser均無--device選項，已讀其處理流程。"},
        {"sentence": "相同固定配方在不同電腦上可能得到不同訓練結果。", "status": "verified",
         "basis": "PyTorch官方重現性文件明示跨平台無完全重現保證；僅支持可能性，沒有重跑多電腦實測。"},
        {"sentence": "三個範例答案有差異時先核對命令及檢查點，再以完整驗證題判讀。", "status": "verified",
         "basis": "這是診斷與比較範圍的建議，不是完整驗證必定通過的保證；三個提示不足代表完整資料集。", "scope": "不重審多模態實驗結果。"},
        {"sentence": "原模型若未回答blue，量化後也不必回答blue；此例應比較前後答案。", "status": "verified",
         "basis": "量化處理既有權重，不以blue為正確標籤重新訓練；答案保持與答對blue是不同判準。",
         "derivation": [{"before": "green", "after": "green", "answer_preserved": True, "equals_blue": False},
                        {"before": "green", "after": "blue", "answer_preserved": False, "equals_blue": True}],
         "scope": "只核對條件與比較目的，不宣稱量化必然保持答案或必然不能改善個別題。"},
    ]
    section_run = command([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/14.md#14.1", "--execute"])
    driver = json.loads(section_run["stdout"])
    assert driver["source_sha256"] == original["source_sha256"] and driver["execution_exit_code"] == 0
    run_dir = Path(driver["artifact_directory"]).resolve()
    assert run_dir.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    copied = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
        raw = (run_dir / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "text": raw.decode()}
    expected_output = "-0.7071\n-0.7071\n0.0\n2對5 3.9298 12對15 3.9298\n3對5 5.1274\n與函式庫版本相同： True\n"
    assert copied["stdout.txt"]["text"] == expected_output and copied["stderr.txt"]["text"] == ""
    result["section_execution"] = {"driver": section_run, "files": copied}
    current_raw = section((ROOT / "course/chapters/14.md").read_bytes(), "14.1")
    freeze_raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/14.1.md").read_bytes()
    assert current_raw == freeze_raw and sha(current_raw) == original["source_sha256"]
    result["unchanged_page"] = {"source_sha256": sha(current_raw), "all_original_claim_ids": [item["id"] for item in original["claims"]],
                                "claim_count": len(original["claims"]), "page_sentences_changed": []}
    figures = []
    receipts = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
    for record in receipts["records"]:
        if record.get("page_id") != "14.1":
            continue
        path = record["figure"]
        assert sha((ROOT / path).read_bytes()) == original["figure_sha256"][path]
        assert sha((ROOT / record["png"]["path"]).read_bytes()) == record["png"]["sha256"]
        figures.append(record)
    result["figures"] = {"records": figures, "visual_inspection": "本回合已用view_image重看兩張指定PNG。單位分量的方向、正負、共同旋轉角與點積均與原判斷相同。PNG未複製或列為artifact。"}
    result["result"] = "所有斷言通過；引用區未變，裝置例外與現行工具一致，原始程式輸出仍符合正文。"
    with ARTIFACT.open("x", encoding="utf-8") as target:
        target.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": ARTIFACT.relative_to(ROOT).as_posix(), "result": result["result"],
                      "referenced_section_unchanged": True, "claim_count": len(original["claims"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
