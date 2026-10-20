"""唯讀核對教材引用的原始來源與本機資料；只向 stdout 輸出查核結果。"""

import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "docs/natural-assistant/v4/data"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def lines(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def curl(url):
    return subprocess.run(
        ["curl", "-fLsS", "--retry", "1", "--max-time", "55", url],
        capture_output=True,
        check=True,
    ).stdout


def local():
    result = {}
    manifest = load("docs/natural-assistant/v4/manifest.json")
    chat = lines("docs/natural-assistant/v4/data/chat-labels.jsonl")
    authored = [r for r in chat if r["synthetic"]]
    conversations = load("docs/natural-assistant/v4/data/chat-original-conversations.json")
    result["project_chat"] = {
        "conversation_count": len(conversations["conversations"]),
        "answer_turn_count": sum(len(c) // 2 for c in conversations["conversations"]),
        "followup_turn_count": sum(len(c) // 2 - 1 for c in conversations["conversations"]),
        "label_count": len(authored),
        "label_with_history_count": sum(bool(r["history"]) for r in authored),
        "splits": dict(Counter(r["split"] for r in authored)),
        "licenses": dict(Counter(r["license"] for r in authored)),
        "sources": dict(Counter(r["source"] for r in authored)),
        "all_synthetic": all(r["synthetic"] is True for r in authored),
        "provenance": conversations["provenance"],
    }
    heldout_chat = [r for r in manifest["rows"] if r["task"] == "chat" and r["split"] != "train"]
    result["heldout_chat"] = {
        "count": len(heldout_chat),
        "splits": dict(Counter(r["split"] for r in heldout_chat)),
        "sources": dict(Counter(r["source"] for r in heldout_chat)),
        "all_have_source_message_ids": all(bool(r.get("source_message_ids")) for r in heldout_chat),
    }
    selected = lines("docs/natural-assistant/evidence/v4-research/chat-speech/oasst2-selected-original-messages.jsonl")
    result["oasst_source_snapshot"] = {
        "count": len(selected),
        "synthetic_annotations": dict(Counter(str(r["synthetic_annotation"]) for r in selected)),
        "roles": dict(Counter(r["role"] for r in selected)),
        "all_chinese": all(r["lang"] == "zh" for r in selected),
    }
    cats = [r for r in manifest["rows"] if isinstance(r.get("source"), dict) and r["source"].get("original_id") == "train_01827"]
    result["cat_labels"] = [
        {k: r[k] for k in ["id", "split", "user", "answer", "image", "source"]} for r in cats
    ]
    svg = ET.fromstring((ROOT / "course/figures/natural-v4-training-cat.svg").read_bytes())
    image = svg.find("{http://www.w3.org/2000/svg}image")
    jpg = base64.b64decode(image.attrib["href"].partition(",")[2])
    result["figure"] = {
        "embedded_jpeg_sha256": digest(jpg),
        "image_box": {k: image.attrib[k] for k in ["x", "y", "width", "height"]},
        "svg_viewBox": svg.attrib["viewBox"],
        "metadata": json.loads(svg.find("{http://www.w3.org/2000/svg}metadata").text),
    }
    result["audio"] = {
        "group_counts": dict(Counter((r["source"], r["task"], r["split"]).__repr__() for r in manifest["audio_rows"])),
        "all_fleurs_answers_null": all(r["answer"] is None for r in manifest["audio_rows"] if r["source"].startswith("fleurs")),
        "all_aishell_have_project_rubric": all(bool(r["references"]["rubric"]) for r in manifest["audio_rows"] if r["source"].startswith("aishell")),
        "no_audio_training_rows": all(r["split"] != "train" for r in manifest["audio_rows"]),
    }
    tree = ast.parse((ROOT / "tiny_perceptron/natural_assistant.py").read_text())
    dispatch = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == "(row, observation)")
    calls = []

    def generate(model, processor, row, data_root, options):
        calls.append({"id": row["id"], "task": row["task"], "user": row["user"], "history": row.get("history"), "system": row.get("system")})
        return {}

    observations = [{"transcript": "ASR::" + r["id"]} for r in manifest["audio_rows"]]
    namespace = {
        "audio_rows": manifest["audio_rows"], "transcripts": observations, "zip": zip,
        "time": time, "deadline": float("inf"), "generate": generate,
        "model": None, "processor": None, "data_root": None, "options": types.SimpleNamespace(),
        "variant": "source-dispatch-probe", "records": [],
    }
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "source_dispatch", "exec"), namespace)
    questions = {r["id"]: r for r in manifest["audio_rows"] if r["source"].startswith("aishell")}
    result["audio_dispatch_probe"] = {
        "method": "執行原始 AST 中的 row/observation 迴圈；generate 只記錄輸入，無模型載入或推論。",
        "generate_call_count": len(calls),
        "tasks": dict(Counter(r["task"] for r in calls)),
        "no_fleurs_generation": all(r["id"] in questions for r in calls),
        "typed_uses_source_text": all(r["user"] == questions[r["id"]]["user"] for r in calls if r["task"] == "typed_chat"),
        "speech_uses_observed_transcript": all(r["user"] == "ASR::" + r["id"] for r in calls if r["task"] == "speech_chat"),
        "both_preserve_system_and_history": all(r["history"] == questions[r["id"]].get("history") and r["system"] == questions[r["id"]].get("system") for r in calls),
    }
    ocr = load("docs/natural-assistant/v4/data/ocr-sources.json")
    commons = [r for r in ocr["sources"] if "Commons" in r["dataset"]]
    result["commons"] = {
        "source_count": len(commons),
        "license_counts": dict(Counter(r["license"] for r in commons)),
        "all_have_artist_url_and_transform": all(r.get("artist_html") and r.get("source_page") and r.get("transform") for r in commons),
    }
    labels = lines("docs/natural-assistant/v4/data/ocr-labels.jsonl")
    result["ocr"] = {
        "annotation_methods": dict(Counter(r.get("annotation_method") for r in labels)),
        "nvidia_metadata": ocr["nvidia"],
        "all_nvidia_labels_have_original_text_and_source": all(r.get("text") is not None and r.get("source_id") for r in labels if r["synthetic"]),
    }
    spec = importlib.util.spec_from_file_location("assets", ROOT / "scripts/build_natural_v4_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, audio = module.assemble()
    result["assembly"] = {
        "rows_equal_manifest": sorted(rows, key=lambda r: r["id"]) == sorted(manifest["rows"], key=lambda r: r["id"]),
        "audio_equal_manifest": sorted(audio, key=lambda r: r["id"]) == sorted(manifest["audio_rows"], key=lambda r: r["id"]),
        "row_count": len(rows),
        "audio_count": len(audio),
    }
    selftrained = load("docs/selftrained/v2-manifest.json")
    result["separate_data_packages"] = {
        "selftrained_initialization": selftrained["initialization"],
        "selftrained_scope": selftrained["scope"],
        "selftrained_package": selftrained["package"]["path"],
        "natural_packages": [r.get("path") for r in manifest["archives"]],
        "asset_hash_overlap": len({r["sha256"] for r in selftrained["assets"]} & {r["sha256"] for r in manifest["files"]}),
    }
    original = ("train_01827", "桌上有幾隻貓？", "兩隻貓。")
    duplicate_rows = [original] * 10
    losses = [2.0, 5.0]
    duplicated_loss = sum([losses[0]] * 10 + [losses[1]]) / 11
    result["duplicate_derivation"] = {
        "copies": 10,
        "unique_scenarios": len(set(duplicate_rows)),
        "new_scenarios": len(set(duplicate_rows)) - len({original}),
        "original_relative_weights": [1, 1],
        "duplicated_relative_weights": [10, 1],
        "original_mean_loss": sum(losses) / 2,
        "duplicated_mean_loss": duplicated_loss,
        "expected_duplicated_mean_loss": 25 / 11,
        "scope": "固定兩題的逐題代價、均勻抽樣或直接平均；不宣稱實際 SGD 軌跡或泛化品質相同。",
    }
    paths = [
        "docs/course-review-20261019/freeze-03/sources/20.4.md",
        "docs/course-review-20261019/freeze-04.json",
        "course/figures/natural-v4-training-cat.svg",
        "docs/natural-assistant/v4/manifest.json",
        "docs/natural-assistant/v4/DATA.md",
        "docs/natural-assistant/v4/data/chat-labels.jsonl",
        "docs/natural-assistant/v4/data/chat-original-conversations.json",
        "docs/natural-assistant/v4/data/chat-curation-ledger.jsonl",
        "docs/natural-assistant/v4/data/vision-labels/train-02.jsonl",
        "docs/natural-assistant/v4/data/ocr-sources.json",
        "docs/natural-assistant/v4/data/ocr-labels.jsonl",
        "docs/natural-assistant/v4/data/voice-sources.json",
        "docs/natural-assistant/v4/data/voice-question-sources.json",
        "docs/natural-assistant/evidence/v4-research/chat-speech/oasst2-selected-original-messages.jsonl",
        "docs/selftrained/V2-DATA-NOTICE.md",
        "docs/selftrained/v2-manifest.json",
        "scripts/build_natural_v4_assets.py",
        "scripts/prepare_natural_v4_chat.py",
        "scripts/prepare_natural_v4_ocr.py",
        "scripts/replay_natural_v4_voice.py",
        "tiny_perceptron/natural_assistant.py",
        "LICENSE",
    ]
    result["input_hashes"] = {p: digest((ROOT / p).read_bytes()) for p in paths}
    return result


def originals():
    records = []
    definitions = [
        ("docci", "https://raw.githubusercontent.com/google/docci/c190366529943195d90c141899abb2ec68dc345d/index.html", "c190366529943195d90c141899abb2ec68dc345d", ["human-annotated English descriptions", "The annotations and images are licensed by Google LLC under"], "Abstract 與下載區授權聲明"),
        ("nvidia", "https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/resolve/69696a1cc543ef3a0f8e9892a89c17293e915263/README.md", "69696a1cc543ef3a0f8e9892a89c17293e915263", ["license: cc-by-4.0", "Large-scale synthetically generated OCR training dataset"], "Dataset Description、License/Terms of Use、Dataset Format"),
        ("oasst", "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md", "179dd21fc55192153d94adb0e0ce8f69e222bf75", ["license: apache-2.0", "This dataset contains message trees."], "Dataset Structure、JSON Example: Message、JSON Example: Conversation Tree"),
        ("fleurs", "https://huggingface.co/datasets/google/fleurs/resolve/d7c758a6dceecd54a98cac43404d3d576e721f07/README.md", "d7c758a6dceecd54a98cac43404d3d576e721f07", ["- cc-by-4.0", "The dataset has a particular focus on read-speech"], "metadata license、Data Fields、Dataset Creation 與 Known Limitations"),
        ("aishell", "https://www.openslr.org/33/", "SLR33；本次取得之頁面雜湊另列", ["Apache License v.2.0", "400 people from different accent areas in China"], "License、About this resource"),
        ("llava", "https://arxiv.org/html/2304.08485v2", "arXiv:2304.08485v2", ["Visual Instruction Tuning"], "GPT-assisted Visual Instruction Data Generation、Instruction Tuning、Table 1、Table 2、Equation (3)"),
        ("docci_attribution", "https://huggingface.co/datasets/google/docci/resolve/a0a43eaf34676ffd008fb6565dd8c2ba00d09100/README.md", "a0a43eaf34676ffd008fb6565dd8c2ba00d09100", ["Jason Baldridge"], "Dataset card 作者署名與授權"),
        ("docci_paper", "https://arxiv.org/html/2404.19753v1", "arXiv:2404.19753v1", ["Jason Baldridge"], "Images 與 Image Collection and Curation；Jason Baldridge 與家人拍攝照片"),
        ("evaluation", "https://scikit-learn.org/1.7/modules/cross_validation.html", "scikit-learn 1.7 文件", ["methodological mistake"], "Cross-validation: evaluating estimator performance 的開頭：訓練資料不能用來估計未見資料表現"),
    ]
    for identifier, url, version, excerpts, locator in definitions:
        raw = curl(url)
        text = raw.decode()
        records.append({
            "id": identifier, "url": url, "version": version,
            "download_sha256": digest(raw), "download_bytes": len(raw), "locator": locator,
            "short_excerpts": [{"text": e, "present": e in text} for e in excerpts],
            "checked_original": True,
        })
    url = "https://storage.googleapis.com/docci/thumbnails/train_01827.jpg?generation=1713028329088595"
    raw = curl(url)
    records.append({"id": "cat_jpeg", "url": url, "version": "GCS generation 1713028329088595", "download_sha256": digest(raw), "download_bytes": len(raw)})
    url = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines?generation=1714384012999810"
    raw = curl(url)
    captions = [json.loads(line) for line in raw.splitlines()]
    cat = next(r for r in captions if r["example_id"] == "train_01827")
    saved_cat = next(r for r in lines("docs/natural-assistant/v4/data/vision-labels/train-02.jsonl") if r["source"]["original_id"] == "train_01827")
    records.append({
        "id": "docci_caption", "url": url, "version": "GCS generation 1714384012999810", "download_sha256": digest(raw), "download_bytes": len(raw),
        "keys": list(cat), "description_sha256": digest(cat["description"].encode()),
        "saved_description_matches_original": saved_cat["references"]["official_description"] == cat["description"],
        "short_excerpt": "A rear view of two cats on top of a wooden desk with black metal legs.",
    })
    ocr = load("docs/natural-assistant/v4/data/ocr-sources.json")
    commons = [r for r in ocr["sources"] if "Commons" in r["dataset"]]
    for start in range(0, len(commons), 50):
        batch = commons[start:start + 50]
        url = "https://commons.wikimedia.org/w/api.php?" + urlencode({
            "action": "query", "format": "json", "formatversion": 2, "prop": "imageinfo|info",
            "pageids": "|".join(str(r["source_page_id"]) for r in batch), "iiprop": "extmetadata|timestamp",
        })
        raw = curl(url)
        pages = json.loads(raw)["query"]["pages"]
        metadata = []
        for page in pages:
            image = page["imageinfo"][0]
            ext = image["extmetadata"]
            source = next(r for r in batch if r["source_page_id"] == page["pageid"])
            metadata.append({
                "page_id": page["pageid"], "title": page["title"], "file_timestamp": image["timestamp"],
                "artist": ext.get("Artist", {}).get("value"),
                "license": ext.get("LicenseShortName", {}).get("value"),
                "license_url": ext.get("LicenseUrl", {}).get("value"),
                "recorded_license": source["license"],
                "license_matches": ext.get("LicenseShortName", {}).get("value") == source["license"],
            })
        records.append({"id": "commons_" + str(start), "url": url, "version": "API 取得時間及每張圖片時間戳；完整回應雜湊另列", "download_sha256": digest(raw), "pages": metadata})
    voices = load("docs/natural-assistant/v4/data/voice-sources.json")
    for split in ["dev", "test"]:
        info = voices["source_files"][split]["transcripts"]
        raw = curl(info["url"])
        rows = [r for r in voices["audio_rows"] if r["source_split"] == split]
        source_lines = raw.decode().splitlines()
        records.append({
            "id": "fleurs_" + split, "url": info["url"], "version": voices["revision"],
            "download_sha256": digest(raw), "matches_recorded_sha256": digest(raw) == info["sha256"],
            "selected_count": len(rows),
            "all_selected_raw_transcripts_present": all(any(r["raw_transcription"] in line and Path(r["source_member"]).name in line for line in source_lines) for r in rows),
        })
    url = "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/bbe295d530192a4cd41644b711c9aecd087df653/data_aishell/transcript/aishell_transcript_v0.8.txt"
    raw = curl(url)
    transcript = dict(line.split(maxsplit=1) for line in raw.decode().splitlines())
    rows = load("docs/natural-assistant/v4/data/voice-question-sources.json")["audio_rows"]
    records.append({"id": "aishell_transcripts", "url": url, "version": "bbe295d530192a4cd41644b711c9aecd087df653", "download_sha256": digest(raw), "selected_count": len(rows), "all_selected_questions_match_original": all(transcript[r["source_utterance_id"]].replace(" ", "") == r["user"] for r in rows)})
    url = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.messages.jsonl.gz"
    raw = curl(url)
    selected = lines("docs/natural-assistant/evidence/v4-research/chat-speech/oasst2-selected-original-messages.jsonl")
    wanted = {r["message_id"]: r for r in selected}
    found = {}
    for line in gzip.decompress(raw).splitlines():
        r = json.loads(line)
        if r["message_id"] in wanted:
            s = wanted[r["message_id"]]
            found[r["message_id"]] = {"text_matches": r["text"] == s["text"], "parent_matches": r["parent_id"] == s["parent_id"], "synthetic": r["synthetic"], "role": r["role"]}
    records.append({"id": "oasst_messages", "url": url, "version": "179dd21fc55192153d94adb0e0ce8f69e222bf75", "download_sha256": digest(raw), "download_bytes": len(raw), "selected_count": len(selected), "found_count": len(found), "all_text_and_parent_match": all(r["text_matches"] and r["parent_matches"] for r in found.values()), "synthetic_annotations": dict(Counter(str(r["synthetic"]) for r in found.values()))})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local", "originals"])
    args = parser.parse_args()
    print(json.dumps({
        "command": f"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python {Path(__file__).relative_to(ROOT)} {args.mode}",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；只讀資料與原始來源，無模型推論", "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "result": local() if args.mode == "local" else originals(),
    }, ensure_ascii=False, indent=2))
