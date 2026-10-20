"""獨立核對裁切算例、圖中格子、實驗計分與原始來源。"""

import hashlib
import html
import io
import json
import platform
import re
import struct
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import PIL
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tiny_perceptron.multimodal import scene


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    if (DESTINATION / name).is_file():
        assert json.loads((DESTINATION / name).read_text()) == value
        return
    with (DESTINATION / name).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def download(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    raw = subprocess.check_output(command)
    return raw, command


def native_zlib_png(raw):
    position = 8
    chunks = []
    while position < len(raw):
        size = struct.unpack(">I", raw[position:position + 4])[0]
        kind = raw[position + 4:position + 8]
        data = raw[position + 8:position + 8 + size]
        chunks.append((kind, data))
        position += size + 12
    assert sum(kind == b"IDAT" for kind, _ in chunks) == 1
    scanlines = zlib.decompress(b"".join(data for kind, data in chunks if kind == b"IDAT"))
    rebuilt = raw[:8]
    for kind, data in chunks:
        if kind == b"IDAT":
            data = zlib.compress(scanlines, level=6)
        rebuilt += struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return rebuilt


environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "pillow": PIL.__version__,
    "numpy": np.__version__,
    "device": "cpu",
    "native_zlib": zlib.ZLIB_RUNTIME_VERSION,
}
run = ROOT / "outputs/reviewer-tools/runs/20261020T054636083333Z"
copied = {
    name: (run / name).read_text(encoding="utf-8")
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py")
}
save("section-run.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/11.md#11.9" --execute',
    "result": "execution_exit_code=0；原圖與裁切的形狀及紅色格數均符合正文。",
    "environment": json.loads(copied["environment.json"]),
    "copied_files_verbatim": copied,
    "copied_file_sha256": {name: digest(content.encode()) for name, content in copied.items()},
})

result = {"environment": environment, "scene_checks": []}
for offset, expected, expected_crop in ((7, 45, 8), (0, 81, 64)):
    image = scene("red", "square", offset=offset)
    crop = image[:, 4:12, 4:12]
    positions = (image[0] > 0).nonzero().tolist()
    crop_positions = (crop[0] > 0).nonzero().tolist()
    count = int((image[0] > 0).sum())
    crop_count = int((crop[0] > 0).sum())
    assert image.shape == (3, 16, 16)
    assert crop.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert count == expected and crop_count == expected_crop
    assert torch.count_nonzero(image[1:]).item() == 0
    result["scene_checks"].append({
        "offset": offset,
        "image_shape": list(image.shape),
        "crop_shape": list(crop.shape),
        "dtype": str(image.dtype),
        "red_pixels": count,
        "crop_red_pixels": crop_count,
        "image_red_rows": sorted({r for r, _ in positions}),
        "image_red_columns": sorted({c for _, c in positions}),
        "crop_red_local_rows": sorted({r for r, _ in crop_positions}),
        "crop_red_local_columns": sorted({c for _, c in crop_positions}),
    })

svg_path = "course/figures/rewrite-11-crop-evidence.svg"
svg = ET.fromstring((ROOT / svg_path).read_bytes())
rectangles = svg.findall("{http://www.w3.org/2000/svg}rect")
grids = []
for side, cell in ((16, 14), (8, 22)):
    cells = [r.attrib for r in rectangles if r.attrib.get("width") == str(cell)]
    assert len(cells) == side * side
    x0 = min(float(r["x"]) for r in cells)
    y0 = min(float(r["y"]) for r in cells)
    red = sorted([
        [int((float(r["y"]) - y0) / cell), int((float(r["x"]) - x0) / cell)]
        for r in cells if r["fill"] == "#ff0000"
    ])
    target = scene("red", "square", offset=7)[0]
    if side == 8:
        target = target[4:12, 4:12]
    assert red == sorted((target > 0).nonzero().tolist())
    grids.append({"side": side, "cell": cell, "origin": [x0, y0], "red_pixels": len(red), "red_positions": red})
yellow = [r.attrib for r in rectangles if r.attrib.get("stroke") == "#ffdc67"]
assert len(yellow) == 1
assert {k: yellow[0][k] for k in ("x", "y", "width", "height")} == {
    "x": "109", "y": "185", "width": "112", "height": "112"
}
result["figure_geometry"] = {"svg": svg_path, "grids": grids, "crop_rectangle": yellow[0]}

experiment_path = "docs/course-experiments/results/vision_ablation.json"
experiment = json.loads((ROOT / experiment_path).read_bytes())
test = experiment["results"]["data"]["splits"]["test"]["records"]
assert len(test) == 12
assert {row["offset"] for row in test} == {2}
assert len({row["family"] for row in test}) == 6
result["experiment"] = {
    "path": experiment_path,
    "sha256": digest((ROOT / experiment_path).read_bytes()),
    "revision": experiment["revision"],
    "seed": experiment["seed"],
    "device": experiment["device"],
    "gpu": experiment["gpu"],
    "python": experiment["python_version"],
    "torch": experiment["torch_version"],
    "step_scale": experiment["step_scale"],
    "training": {k: v for k, v in experiment["results"]["training"].items() if k != "history"},
    "split_counts": {k: v["count"] for k, v in experiment["results"]["data"]["splits"].items()},
    "test_records": test,
    "scores": {},
    "crop_reconstruction": [],
}
for name in ("none", "edge_cropped"):
    intervention = experiment["results"]["interventions"][name]
    samples = intervention["samples"]
    assert len(samples) == intervention["examples"] == 12
    rows = []
    for sample, original in zip(samples, test, strict=True):
        match = sample["generated"] == original["answer"]
        assert match == sample["exact_match"]
        assert sample["target"] == original["answer"]
        assert sample["question"] == original["question"]
        rows.append({k: sample[k] for k in ("row", "family", "question", "target", "generated", "exact_match")})
    correct = sum(row["exact_match"] for row in samples)
    assert correct == intervention["correct"] == 9
    assert correct / len(samples) == intervention["exact_match"] == 0.75
    grouped = {}
    for question in ("shape?", "color?"):
        group = [r for r in samples if r["question"] == question]
        grouped[question] = {"count": len(group), "correct": sum(r["exact_match"] for r in group)}
    assert grouped == intervention["groups"]
    assert grouped == {"shape?": {"count": 6, "correct": 3}, "color?": {"count": 6, "correct": 6}}
    assert {r["generated"] for r in samples if r["question"] == "shape?"} == {"circle"}
    result["experiment"]["scores"][name] = {"correct": correct, "count": len(samples), "groups": grouped, "samples": rows}

stored_hashes = {item["path"]: item["sha256"] for item in experiment["artifacts"]}
for index, row in enumerate(test):
    crop = scene(row["color"], row["shape"], offset=7)[:, 4:12, 4:12]
    pixels = (crop.permute(1, 2, 0).numpy() * 255).astype("uint8")
    enlarged = Image.fromarray(pixels).resize((16, 16))
    buffer = io.BytesIO()
    enlarged.save(buffer, format="PNG")
    png_hash = digest(buffer.getvalue())
    native_png_hash = digest(native_zlib_png(buffer.getvalue()))
    expected_hash = stored_hashes[f"crop-{index}.png"]
    assert expected_hash in (png_hash, native_png_hash)
    result["experiment"]["crop_reconstruction"].append({
        "row": index, "color": row["color"], "shape": row["shape"],
        "source_offset": 7, "crop_size": [8, 8], "resized_size": list(enlarged.size),
        "remaining_lit_pixels": int((crop.sum(0) > 0).sum()),
        "pillow_png_sha256": png_hash,
        "native_zlib_png_sha256": native_png_hash,
        "recorded_png_sha256": expected_hash,
        "matches_recorded_sha256": True,
    })
result["experiment"]["png_encoding_note"] = "初次額外核對假定 Pillow 當前預設 PNG 編碼能逐位元組重現原檔；方塊圖的 hash 斷言失敗。診斷發現 Pillow 壓縮器與 Python 原生 zlib 的輸出不同；保留 IHDR、濾波後的原始掃描列與 IEND，僅用 Python zlib level=6 重新壓縮 IDAT 並重算 CRC，即得到原紀錄的方塊圖 SHA-256。圓形圖的預設 Pillow 輸出直接相符。此比對核實圖片內容，不把壓縮器輸出差異當作像素差異。"

first = torch.zeros(3, 16, 16)
second = first.clone()
second[0, 0, 0] = 1
first_crop = first[:, 4:12, 4:12]
second_crop = second[:, 4:12, 4:12]
assert not torch.equal(first, second) and torch.equal(first_crop, second_crop)
padded = torch.nn.functional.pad(first_crop, (4, 4, 4, 4))
assert padded.shape == first.shape and not torch.equal(padded, second)
line = Image.new("L", (8, 8))
for y in range(8):
    line.putpixel((3, y), 255)
small = np.array(line.resize((1, 1), Image.Resampling.BOX))
assert 0 < int(small[0, 0]) < 255
result["information_checks"] = {
    "different_originals_identical_crops": True,
    "padding_restores_size_but_not_arbitrary_original": True,
    "box_downsample_thin_line_to_background_mixture": int(small[0, 0]),
    "derivation": "兩張原圖僅在裁切範圍外不同時，裁切結果相同。只依賴裁切結果的確定性補白或放大也相同，不可能同時等於兩張不同原圖；恢復尺寸不保證還原資料。常數 circle 答案對三個 circle 樣本得分，對三個 square 樣本失分，形狀得分固定為 3/6；加上六個正確顏色答案即為 9/12。",
}
result["input_sha256"] = {
    name: digest((ROOT / name).read_bytes())
    for name in ("tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py", experiment_path, svg_path)
}
for name in ("tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py"):
    assert result["input_sha256"][name] == experiment["code_sha256"][name]
save("verification.json", result)

receipts = []
for name, url, version, excerpts in (
    (
        "pillow_filters",
        "https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/docs/handbook/concepts.rst",
        "Pillow 12.3.0，concepts.rst 的 Filters",
        ["For geometry operations that may map multiple input pixels to a single output", "Pick one nearest pixel"],
    ),
    (
        "pillow_image",
        "https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/src/PIL/Image.py",
        "Pillow 12.3.0，Image.crop 與 Image.resize",
        ["Returns a rectangular region from this image.", "if resample is None:\n            resample = Resampling.BICUBIC"],
    ),
    (
        "vizwiz_answerability",
        "https://vizwiz.org/tasks-and-datasets/vqa/",
        "Visual Question Answering，讀取時的頁面；Dataset 更新標示 January 10, 2023；Task 2",
        ["Given an image and question about it, the task is to predict if the visual question cannot be answered"],
    ),
):
    raw, command = download(url)
    text = raw.decode()
    if name == "vizwiz_answerability":
        text = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", text)))
    assert all(excerpt in text for excerpt in excerpts)
    receipts.append({
        "id": name, "url": url, "version": version,
        "accessed_on": datetime.now().astimezone().date().isoformat(),
        "retrieval_command": command, "retrieved_bytes": len(raw),
        "retrieved_sha256": digest(raw), "checked_original": True,
        "excerpts": excerpts,
    })
save("original-sources.json", {"sources": receipts, "environment": environment})
print(json.dumps({"result": "全部核對成立", "environment": environment, "scene_cases": 2, "matched_crop_hashes": 12, "scores": {k: {"correct": v["correct"], "count": v["count"], "groups": v["groups"]} for k, v in result["experiment"]["scores"].items()}}, ensure_ascii=False))
