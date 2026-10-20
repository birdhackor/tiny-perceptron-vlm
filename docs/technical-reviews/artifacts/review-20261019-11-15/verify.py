"""核對縮圖數值、資訊碰撞、專案契約，並保存小份原始來源收據。"""

import ast
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tiny_perceptron.multimodal import patchify
from tiny_perceptron.natural_concepts import thin_stroke_report


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    target = HERE / name
    if target.exists():
        if json.loads(target.read_text()) != value:
            raise ValueError("既有證據內容不同，停止覆寫：" + name)
        return
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)


def fetch(url, anchors, excerpt):
    command = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "40", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    text = raw.decode("utf-8")
    if "<html" in text.lower()[:1000] or "<!doctype" in text.lower()[:1000]:
        parser = Text()
        parser.feed(text)
        text = " ".join(" ".join(parser.parts).split())
    for anchor in anchors:
        position = text.find(anchor)
        if position < 0:
            raise AssertionError("原始來源沒有預期定位文字：" + anchor)
        print(url, text[max(0, position - 100) : position + 2400], sep="\n")
    if excerpt not in text:
        raise AssertionError("摘錄與當次原文不一致")
    return {
        "url": url,
        "accessed_at": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": result.returncode,
        "response_sha256": sha(raw),
        "response_bytes": len(raw),
        "locators": anchors,
        "excerpt": excerpt,
        "storage": "全文只在記憶體處理；本檔只保存小段佐證與收據。",
    }


torch.set_num_threads(1)
image = torch.zeros(1, 1, 32, 32)
image[:, :, :, 15] = 1
small = F.interpolate(image, (4, 4), mode="area")
other = torch.zeros_like(image)
other[:, :, :, 14] = 1
other_small = F.interpolate(other, (4, 4), mode="area")
manual = image.reshape(1, 1, 4, 8, 4, 8).mean((3, 5))
restored = F.interpolate(small, (32, 32), mode="nearest")
crop = image[:, :, :, 8:16]
crop_small = F.interpolate(crop, (4, 4), mode="area")

producer_path = ROOT / "scripts/selftrained/prepare_vision_ocr.py"
tree = ast.parse(producer_path.read_text())
charset = next(
    ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "CHARACTERS" for target in node.targets)
)
model_path = ROOT / "tiny_perceptron/natural_assistant.py"
model_tree = ast.parse(model_path.read_text())
model_constants = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in model_tree.body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in ("MODEL_ID", "MODEL_REVISION")
}
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-15/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(), "dtype": str(image.dtype)},
    "report": thin_stroke_report(),
    "bright_column_indices_zero_based": image[0, 0].sum(0).nonzero().flatten().tolist(),
    "bright_column_ordinals_one_based": (image[0, 0].sum(0).nonzero().flatten() + 1).tolist(),
    "small_matrix": small[0, 0].tolist(),
    "equals_manual_nonoverlapping_8_by_8_average": torch.equal(manual, small),
    "different_inputs_same_downsample": {"columns_zero_based": [14, 15], "different_scalar_count": int((image != other).sum()), "same_output": torch.equal(small, other_small)},
    "enlarge_already_small_max": float(restored.max()),
    "crop_original_then_resize": {"original_columns": [8, 16], "right_endpoint_excluded": True, "input_shape": list(crop.shape), "max": float(crop_small.max())},
    "patch_counts_with_fixed_side_four": {str(side): patchify(torch.zeros(1, 3, side, side), 4).shape[1] for side in (16, 32)},
    "ocr_characters": charset,
    "ocr_character_count": len(charset),
    "ocr_charset_disjoint_from_ascii_digits": not bool(set(charset) & set("0123456789")),
    "natural_assistant": model_constants,
    "installed_interpolate_sha256": sha(inspect.getsource(F.interpolate).encode()),
    "installed_functional_file_sha256": sha(Path(F.__file__).read_bytes()),
    "installed_area_dispatch": 'if input.dim() == 4 and mode == "area": ... return adaptive_avg_pool2d(input, output_size)',
    "repository_hashes": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in [producer_path, model_path, ROOT / "tiny_perceptron/natural_concepts.py", ROOT / "tiny_perceptron/multimodal.py", ROOT / "tiny_perceptron/selftrained/tokenizer.py", ROOT / "tiny_perceptron/selftrained/model.py"]},
}
figure_receipts = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
record["figure_receipt"] = next(item for item in figure_receipts["records"] if item["page_id"] == "11.15")
assert record["report"] == {"original_shape": [32, 32], "small_shape": [4, 4], "original_brightest": 1.0, "small_brightest": 0.125, "pixels_at_least_half_bright_before": 32, "pixels_at_least_half_bright_after": 0}, "縮圖數值與手算不同"
assert record["equals_manual_nonoverlapping_8_by_8_average"], "面積縮圖與分塊平均不同"
assert record["different_inputs_same_downsample"]["same_output"], "兩條線沒有得到同樣縮圖"
save("verification.json", record)
print(json.dumps(record, ensure_ascii=False, indent=2))

run = ROOT / "outputs/reviewer-tools/runs/20261020T054652194126Z"
save("section-execution.json", {
    "invocation": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/11.md#11.15\" --execute",
    "files": {name: {"sha256": sha((run / name).read_bytes()), "content": (run / name).read_text()} for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json")},
})

receipts = [
    fetch("https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py", ['if input.dim() == 4 and mode == "area":'], "return adaptive_avg_pool2d(input, output_size)"),
    fetch("https://arxiv.org/html/2010.11929v2", ["To handle 2D images", "When feeding images of higher resolution", "quadratic"], "Position embeddings are added to the patch embeddings to retain positional information."),
    fetch("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md", ["OCR"], "OCR"),
]
for identifier, glyph, excerpt in [(20651, "未", "上橫短，下橫長"), (20653, "末", "上橫長，下橫短")]:
    receipts.append({
        "url": f"https://dict.variants.moe.edu.tw/dictView.jsp?ID={identifier}",
        "accessed_on": "2026-10-20",
        "method": "web__run open 原始教育部字典頁面",
        "version": "教育部《異體字字典》臺灣學術網路十四版（正式七版）2024",
        "locator": glyph + "／字樣說明／楷書寫法",
        "excerpt": excerpt,
        "inspection": "實際閱讀字樣說明中的上下橫長短規定。",
        "curl_attempt": {"command": f"curl -L --fail --silent --show-error --max-time 20 'https://dict.variants.moe.edu.tw/dictView.jsp?ID={identifier}&la=0'", "result": "curl: (22) The requested URL returned error: 404"},
        "storage": "只保存支持字形差異的小段摘錄；沒有保存全文。",
    })
save("source-receipts.json", receipts)
