"""核對公開 CPU 示範、輸入邊界、資料範圍與固定來源；大型下載只暫存。"""

import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def curl(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    proc = subprocess.run(command, capture_output=True, check=True)
    return proc.stdout, {"command": command, "exit_code": proc.returncode, "bytes": len(proc.stdout),
                         "sha256": sha(proc.stdout)}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    import torch
    import safetensors
    from tiny_perceptron.selftrained.tokenizer import OCR_CHARACTERS
    from tiny_perceptron.selftrained.tools import execute_tool_call, parse_tool_call, ToolCallError

    torch.set_num_threads(2)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-1/verify.py",
        "environment": {"python": sys.version, "torch": torch.__version__, "safetensors": safetensors.__version__,
                        "platform": platform.platform(), "device": "cpu", "threads": "2"},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tag": subprocess.check_output(["git", "rev-parse", "selftrained-v2"], cwd=ROOT, text=True).strip(),
    }
    files = [
        "docs/course-review-20261019/freeze-01/sources/19.1.md",
        "docs/course-review-20261019/freeze-04.json", "course/figures/p6-19-start-core.svg",
        "scripts/selftrained/chat.py", "tiny_perceptron/selftrained/inference.py",
        "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/dataset.py",
        "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/selftrained/tools.py",
        "tiny_perceptron/model.py", "tiny_perceptron/modern.py",
        "scripts/selftrained/prepare_vision_ocr.py", "scripts/selftrained/augment_ocr_v2.py",
        "scripts/selftrained/prepare_voice.py", "scripts/selftrained/augment_text_tools_v2.py",
        "scripts/selftrained/train.py", "scripts/selftrained/evaluate.py", "docs/selftrained/v2-manifest.json",
        "docs/selftrained/examples/v2/text.messages.json", "notebooks/19/19.1.ipynb",
        "pyproject.toml", "uv.lock", "docs/selftrained/v2-jobs/moe-joint-native-balanced-4000.json",
        "docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json",
        "docs/selftrained/results/public-cpu-raw/text-1-result.json",
        "docs/selftrained/results/public-raw/moe/test/metrics.json",
    ]
    result["file_sha256"] = {p: sha(read(p)) for p in files}
    result["tag_comparison"] = {}
    for path in files[3:16] + ["pyproject.toml", "uv.lock"]:
        original = subprocess.check_output(["git", "show", f"selftrained-v2:{path}"], cwd=ROOT)
        result["tag_comparison"][path] = {"tag_sha256": sha(original), "same_as_current": original == read(path)}
    notebook = json.loads(read("notebooks/19/19.1.ipynb"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    result["notebook_setup"] = {"code_cells": code, "pip_base_install": '"--quiet", "-e", str(root)' in code[0],
                                "selftrained_extra_in_setup": "selftrained" in code[0],
                                "uv_sync_in_setup": "uv sync" in code[0]}
    messages = json.loads(read("docs/selftrained/examples/v2/text.messages.json"))
    result["input_messages"] = messages
    result["ocr_alphabet"] = {"text": OCR_CHARACTERS, "length": len(OCR_CHARACTERS),
                              "unique": len(set(OCR_CHARACTERS))}
    result["calculator"] = {}
    for operation, a, b in [("add", 12, 7), ("subtract", 0, 99), ("multiply", 99, 99)]:
        call = json.dumps({"tool": "calculator", "operation": operation, "a": a, "b": b})
        result["calculator"][call] = execute_tool_call(call)
    try:
        parse_tool_call('{"tool":"calculator","operation":"add","a":100,"b":1}')
    except ToolCallError as error:
        result["calculator"]["outside_boundary"] = str(error)

    snapshots = []
    fashion_url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md"
    raw, receipt = curl(fashion_url)
    text = raw.decode()
    excerpt = "Each example is a 28x28 grayscale image, associated with a label from 10 classes."
    assert excerpt in text
    assert all(s in text for s in ["| 1 | Trouser |", "| 8 | Bag |", "| 9 | Ankle boot |"])
    snapshots.append({"id": "fashion", "url": fashion_url, "version": "b2617bb6d3ffa2e429640350f613e3291e10b141",
                      "locator": "Introduction; Labels table", "excerpt": excerpt,
                      "label_mapping": {"1": "Trouser", "8": "Bag", "9": "Ankle boot"}, "receipt": receipt})
    minds_url = "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md"
    raw, receipt = curl(minds_url)
    text = raw.decode()
    excerpt = "MINDS-14 is training and evaluation resource for intent detection task with spoken data."
    assert excerpt in text and "config_name: zh-CN" in text and "crowdsourced" in text
    snapshots.append({"id": "minds", "url": minds_url, "version": "40ce77cb32a384e4d50a568e1ec39ac804019d33",
                      "locator": "Dataset Description; YAML config_name zh-CN and crowdsourced tags", "excerpt": excerpt,
                      "inspection": "閱讀資料說明、音訊與轉寫欄位及 zh-CN 設定；原文描述真人口說意圖資料，非合成 TTS。",
                      "receipt": receipt})
    checklist_url = "https://aclanthology.org/2020.acl-main.442/"
    raw, receipt = curl(checklist_url)
    assert b"CheckList includes a matrix of general linguistic capabilities and test types" in raw
    snapshots.append({"id": "checklist", "url": checklist_url, "version": "ACL 2020 proceedings, DOI 10.18653/v1/2020.acl-main.442",
                      "locator": "Abstract", "excerpt": "CheckList includes a matrix of general linguistic capabilities and test types that facilitate comprehensive test ideation",
                      "inspection": "讀取出版方 abstract：以多種能力與測試類型補足單一整體分數，支持逐項行為測試；未借用論文的效益數字。",
                      "receipt": receipt})
    for snapshot in snapshots:
        snapshot["accessed_on"] = datetime.now(timezone.utc).date().isoformat()
    dump(ART / "primary-sources.json", snapshots)

    remote = "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint/inference-manifest.json"
    raw, receipt = curl(remote)
    assert sha(raw) == "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"
    manifest = json.loads(raw)
    result["public_manifest"] = {"receipt": receipt, "excerpt": {k: manifest[k] for k in
        ["schema", "files", "stage", "selected_step", "selected_checkpoint_sha256", "origin", "selection", "preprocess_version"]}}
    training = json.loads(read("docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json"))
    result["native_training"] = {k: training.get(k) for k in
        ["stage", "architecture", "steps", "best_step", "best_checkpoint_sha256", "selected_step", "selected_checkpoint_sha256",
         "native_voice_loss_weight", "tool_loss_weight", "numeric_run_loss_weight", "completed_requested_steps"]}
    result["native_training"]["checkpoint_fields"] = {k: v for k, v in training.items()
        if "checkpoint" in k and isinstance(v, (str, int, bool))}
    result["native_training"]["data_matches_export"] = training["data_sha256"] == manifest["data_sha256"]
    selected = json.loads(read("docs/selftrained/results/public-raw/moe/freeze/frozen.json"))
    result["frozen_selection"] = {k: v for k, v in selected.items()
        if k in ["stage", "selected_step", "checkpoint_sha256", "model_sha256", "selected_checkpoint_sha256", "selection"]}
    result["remote_tag"] = subprocess.check_output(
        ["git", "ls-remote", "https://github.com/birdhackor/tiny-perceptron-vlm.git", "refs/tags/selftrained-v2"],
        cwd=ROOT, text=True).strip()

    with tempfile.TemporaryDirectory(prefix="runtime-", dir=ART) as runtime:
        runtime = Path(runtime)
        os.environ["HF_HOME"] = str(runtime / "hf")
        os.environ["HF_HUB_CACHE"] = str(runtime / "hf" / "hub")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "45"
        from scripts.selftrained import chat
        arguments = ["--model-dir", str(runtime / "export"), "--asset-dir", str(runtime / "no-assets"),
            "--repo", "birdhackor/tiny-perceptron-course-models", "--revision", "979cdfacc588ad0536f1c64fff96f264571cf054",
            "--prefix", "selftrained/v2/moe-joint", "--manifest-sha256", sha(raw), "--messages",
            "docs/selftrained/examples/v2/text.messages.json", "--task", "text", "--device", "cpu", "--max-new-tokens", "128", "--threads", "2"]
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            answer = chat.main(arguments)
        result["cli"] = {"arguments": [s.replace(str(runtime), "{runtime}") for s in arguments],
                          "path_note": "只將模型與 HF 快取根目錄改到 artifact 目錄內的暫存子目錄；離開後刪除下載，未改推論程式。",
                          "stdout_json_matches_return": json.loads(capture.getvalue()) == answer,
                          "result": answer,
                          "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                                    for p in (runtime / "export").iterdir()}}
        expected = "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
        result["cli"]["expected_answer_match"] = answer["answer"] == expected
        result["cli"]["eos_match"] = answer["generations"][0]["eos"] is True
        before = {p.name: sha(p.read_bytes()) for p in (runtime / "export").iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = chat.main(arguments)
        after = {p.name: sha(p.read_bytes()) for p in (runtime / "export").iterdir()}
        result["repeat"] = {"same_answer": repeated["answer"] == answer["answer"], "payloads_unchanged": before == after,
                            "eos": repeated["generations"][0]["eos"], "asset_directory_created": (runtime / "no-assets").exists()}
        assistant = chat.InferenceAssistant(runtime / "export", runtime / "no-assets", device="cpu", manifest_sha256=sha(raw))
        changed_format = [dict(m) for m in messages]
        for message in changed_format:
            message["content"] = message["content"].replace("兩點", "一句")
        changed_content = [dict(m) for m in messages]
        changed_content[-1]["content"] = "我想更新地址。"
        result["counterfactuals"] = {}
        for name, inputs in [("format_one_sentence", changed_format), ("content_address", changed_content)]:
            changed = assistant.reply(inputs, task="text", max_new_tokens=128)
            result["counterfactuals"][name] = {"messages": inputs, "answer": changed["answer"],
                "eos": changed["generations"][0]["eos"], "generated_ids": changed["generations"][0]["generated_ids"]}
        result["encoder_probes"] = {}
        for kind, token, payload in [
            ("image", 7, {"kind": "image", "values": torch.zeros(2, 1, 32, 32), "coordinates": torch.tensor([[.25,.5],[.75,.5]])}),
            ("ocr", 8, {"kind": "ocr", "values": torch.ones(1, 32, 128)}),
            ("audio", 9, {"kind": "audio", "values": torch.zeros(40, 40)}),
        ]:
            count = assistant.model.modality_token_count(kind)
            ids = torch.tensor([[assistant.encoder.tokenizer.bos_id] + [token] * count], dtype=torch.long)
            with torch.no_grad():
                embedded, perceptions = assistant.model._embeddings(ids, None, [[payload]])
            result["encoder_probes"][kind] = {"slots": count, "embedding_shape": list(embedded.shape),
                "finite": bool(torch.isfinite(embedded).all()), "logit_shape": list(perceptions[0]["logits"].shape),
                "shared_language_core": assistant.model.lm is assistant.model.lm}
    old = json.loads(read("docs/selftrained/results/public-cpu-raw/text-1-result.json"))
    result["saved_cpu_comparison"] = {"same_answer": old["actual_answer"] == answer["answer"],
                                      "same_weight_sha256": old["model"]["files"]["model.safetensors"] == answer["model"]["files"]["model.safetensors"],
                                      "saved_eos": old["actual_generations"][0]["eos"]}
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    dump(ART / "verification.json", result)
    print(json.dumps({"answer_match": result["cli"]["expected_answer_match"], "eos": result["cli"]["eos_match"],
                      "repeat": result["repeat"], "files": result["cli"]["files"], "native_training": result["native_training"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
