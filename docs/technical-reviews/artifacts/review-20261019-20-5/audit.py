"""保存凍結範例的執行結果、反例與原始來源的小段摘錄。"""

import ast
import contextlib
import copy
import hashlib
import html
import io
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/20.5.md"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (ARTIFACT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment():
    return {"python": sys.version, "platform": platform.platform(), "device": "CPU", "executable": sys.executable}


def section():
    command = [".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/20.md#20.5", "--execute"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    receipt = {"command": "PYTHONDONTWRITEBYTECODE=1 " + " ".join(command), "environment": environment(), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode == 0:
        directory = Path(json.loads(result.stdout)["artifact_directory"])
        receipt["copied_outputs"] = {}
        for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
            raw = (directory / name).read_bytes()
            receipt["copied_outputs"][name] = {"sha256": sha(raw), "content": raw.decode("utf-8")}
        extracted = (directory / "section.md").read_bytes()
        receipt["frozen_match"] = extracted == FROZEN.read_bytes()
        assert receipt["frozen_match"]
    save("section-execution.json", receipt)
    print(json.dumps({"file": "section-execution.json", "exit_code": result.returncode, "stdout": result.stdout}, ensure_ascii=False))
    assert result.returncode == 0


def examples():
    raw = FROZEN.read_bytes()
    code = re.search(r"```python\n(.*?)\n```", raw.decode("utf-8"), re.S).group(1)
    tree = ast.parse(code)
    initial = ast.literal_eval(tree.body[0].value)
    tail = ast.Module(body=tree.body[1:], type_ignores=[])
    cases = {}
    for label in ("original", "second_record_to_test", "whole_family_to_test", "renamed_family_only"):
        records = copy.deepcopy(initial)
        if label == "second_record_to_test":
            records[1]["split"] = "test"
        elif label == "whole_family_to_test":
            for row in records:
                if row["family"] == initial[0]["family"]:
                    row["split"] = "test"
        elif label == "renamed_family_only":
            records[1]["split"] = "test"
            records[1]["family"] = "另一個名字"
        before = copy.deepcopy(records)
        namespace = {"records": records}
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(tail, str(FROZEN), "exec"), namespace)
        train = namespace["train_families"]
        test = namespace["test_families"]
        cases[label] = {"records": records, "stdout": stream.getvalue(), "train_families": sorted(train), "test_families": sorted(test), "intersection": sorted(train & test), "records_unchanged": records == before}
        assert records == before
    assert cases["original"]["intersection"] == []
    assert cases["original"]["train_families"] == ["貓照片A"]
    assert cases["original"]["test_families"] == ["招牌照片B"]
    assert cases["second_record_to_test"]["intersection"] == ["貓照片A"]
    assert cases["whole_family_to_test"]["intersection"] == []
    assert cases["renamed_family_only"]["intersection"] == []
    probes = {"set_deduplicates": len({"A", "A"}) == 1, "filter_selects_only_train": all(row["split"] == "train" for row in initial if row["family"] in cases["original"]["train_families"]), "intersection": sorted({"A", "B"} & {"B", "C"}), "sorted_returns_list": isinstance(sorted({"B", "A"}), list)}
    save("example-verification.json", {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " examples", "environment": environment(), "frozen_sha256": sha(raw), "fence_sha256": sha(code.encode()), "cases": cases, "probes": probes, "result": "所有精確比較均成立；僅改家族名稱會隱藏來源重複，不能作為修正。"})
    print(json.dumps({"file": "example-verification.json", "cases": {name: item["stdout"] for name, item in cases.items()}}, ensure_ascii=False))


def clean(raw):
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"</(?:p|h[1-6]|li|tr|dd|dt|div|section)>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def sources():
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    targets = [
        ("docci-paper", "https://arxiv.org/html/2404.19753v1", ["We grouped images into", "We provide the cluster ID for each image"], 300, 2),
        ("oasst-card", "https://huggingface.co/datasets/OpenAssistant/oasst2/raw/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md", ["This dataset contains message trees", "Full conversation trees"], 350, 2),
        ("split-guide", "https://scikit-learn.org/stable/modules/cross_validation.html", ["When evaluating different settings", "The i.i.d. assumption is broken", "GroupKFold is a variation"], 750, 3),
        ("python-sets", f"https://docs.python.org/{minor}/library/stdtypes.html", ["A set object is an unordered collection", "a set comprehension", "Return a new set with elements common"], 250, 3),
        ("python-sorted", f"https://docs.python.org/{minor}/library/functions.html", ["Return a new sorted list"], 250, 1),
        ("regression-glossary", "https://www.nist.gov/glossary-term/30681", ["Checking to confirm"], 250, 1),
        ("qwen-paper", "https://arxiv.org/html/2511.21631v1", ["predominantly Chinese", "samples from open-source datasets"], 350, 2),
    ]
    records = []
    for identifier, url, needles, radius, maximum in targets:
        command = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "45", url]
        result = subprocess.run(command, capture_output=True)
        item = {"id": identifier, "url": url, "command": command, "accessed_at": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode, "stderr": result.stderr.decode("utf-8", errors="replace"), "response_sha256": sha(result.stdout), "response_bytes": len(result.stdout), "excerpts": []}
        if result.returncode == 0:
            raw = result.stdout.decode("utf-8")
            text = clean(raw) if "<html" in raw[:1000].lower() or "<!doctype" in raw[:1000].lower() else raw
            title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
            item["title"] = clean(title.group(1)) if title else identifier
            starts = []
            for needle in needles:
                for found in re.finditer(re.escape(needle), text, flags=re.I):
                    start, end = max(0, found.start() - radius), min(len(text), found.end() + radius)
                    if any(abs(start - other) < radius for other in starts):
                        continue
                    starts.append(start)
                    item["excerpts"].append({"search_term": needle, "normalized_text_offset": start, "text": text[start:end]})
                    break
                if len(starts) >= maximum:
                    break
        records.append(item)
        print(json.dumps({"id": identifier, "exit_code": result.returncode, "title": item.get("title"), "excerpt_count": len(item["excerpts"])}, ensure_ascii=False))
    save("source-excerpts.json", {"environment": environment(), "retention": "全文僅在記憶體中處理；保存取回全文的 SHA-256 與直接支持查核的小段摘錄。", "sources": records})


def repository():
    base = ROOT / "docs/natural-assistant/v4"
    paths = ["manifest.json", "data/chat-labels.jsonl", "data/chat-curation-ledger.jsonl", "data/vision-sources.json", "data/vision-activity-sources.json", "data/vision-activity-heldout-sources.json", "selection.json", "validation-protocol.json"]
    inputs = {str((base / name).relative_to(ROOT)): sha((base / name).read_bytes()) for name in paths}
    manifest = json.loads((base / "manifest.json").read_text())
    grouped = defaultdict(set)
    for row in manifest["rows"] + manifest["audio_rows"]:
        grouped[row["family"]].add(row["split"])
    overlap = {key: sorted(value) for key, value in grouped.items() if len(value) > 1}
    assert not overlap
    photos = []
    for name in paths[3:6]:
        photos.extend(json.loads((base / name).read_text())["rows"])
    photo_maps = {field: defaultdict(set) for field in ("example_id", "cluster_id", "sha256", "rgb_sha256")}
    for photo in photos:
        for field, groups in photo_maps.items():
            if photo.get(field):
                groups[photo[field]].add(photo["split"])
    photo_overlaps = {field: {key: sorted(value) for key, value in groups.items() if len(value) > 1} for field, groups in photo_maps.items()}
    assert all(not value for value in photo_overlaps.values())
    chat = [json.loads(line) for line in (base / "data/chat-labels.jsonl").read_text().splitlines()]
    trees = defaultdict(set)
    family_trees = defaultdict(set)
    for row in chat:
        if row.get("source_tree_id"):
            trees[row["source_tree_id"]].add(row["split"])
            family_trees[row["family"]].add(row["source_tree_id"])
    assert all(len(splits) == 1 for splits in trees.values())
    assert all(len(tree_ids) == 1 for tree_ids in family_trees.values())
    selection = json.loads((base / "selection.json").read_text())
    assert selection["dataset_manifest_sha256"] == sha((base / "manifest.json").read_bytes())
    protocol_matches = [path for path in base.glob("validation-protocol*.json") if sha(path.read_bytes()) == selection["validation_protocol_sha256"]]
    assert len(protocol_matches) == 1
    selected_protocol = protocol_matches[0]
    inputs[str(selected_protocol.relative_to(ROOT))] = sha(selected_protocol.read_bytes())
    save("repository-verification.json", {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " repository", "environment": environment(), "inputs": inputs, "manifest_rows": len(manifest["rows"]), "manifest_audio_rows": len(manifest["audio_rows"]), "manifest_families": len(grouped), "family_cross_split": overlap, "photo_source_rows": len(photos), "photo_cross_split": photo_overlaps, "chat_row_counts": dict(Counter(row["split"] for row in chat)), "oasst_trees": len(trees), "oasst_tree_cross_split": {key: sorted(value) for key, value in trees.items() if len(value) > 1}, "selection": {"selected_variant": selection["selected_variant"], "manifest_hash_matches": True, "protocol_hash_matches": True, "protocol_path": str(selected_protocol.relative_to(ROOT)), "base_model": selection["base_model"]}, "result": "本地 manifest 的家族、照片來源 ID／群組／已記錄指紋、對話樹均未跨組；選版紀錄綁定同一份 manifest 與驗證規約。", "scope": "只讀來源記錄與本回合實際檢查；沒有重訓、重跑模型推論、讀取其他審閱報告或逐一重解碼照片。已記錄 rgb_sha256 不是本回合重算的像素指紋。"})
    print(json.dumps({"file": "repository-verification.json", "cross_split_families": len(overlap), "cross_split_oasst_trees": 0, "photo_overlaps": photo_overlaps}, ensure_ascii=False))


if __name__ == "__main__":
    {"section": section, "examples": examples, "sources": sources, "repository": repository}[sys.argv[1]]()
