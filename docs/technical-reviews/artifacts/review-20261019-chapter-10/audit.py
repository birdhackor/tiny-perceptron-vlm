"""核對章首導覽指向的程式路徑，並保存原始論文的最小來源收據。"""

import ast
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import (
    MultiModalLM,
    VisionEncoder,
    expand_modalities,
    patchify,
    scene,
    unpatchify,
)


ROOT = Path(__file__).resolve().parents[4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.current = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.depth += 1
        if self.depth and tag in {"math", "span"}:
            self.current.append(" ")

    def handle_data(self, data):
        if self.depth:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.depth:
            self.depth -= 1
            if not self.depth:
                self.paragraphs.append(" ".join("".join(self.current).split()))
                self.current = []


def original_receipt(url, version, quotations):
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--max-time", "45", url,
    ]
    response = subprocess.run(command, capture_output=True, check=True)
    parser = Paragraphs()
    parser.feed(response.stdout.decode("utf-8"))
    excerpts = []
    for locator, quotation in quotations:
        assert any(quotation in paragraph for paragraph in parser.paragraphs), "原文摘錄必須能定位"
        excerpts.append({"locator": locator, "quotation": quotation})
    return {
        "url": url,
        "version": version,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "command": command,
        "returncode": response.returncode,
        "full_response_sha256": digest(response.stdout),
        "full_response_bytes": len(response.stdout),
        "checked_original": True,
        "excerpts": excerpts,
        "storage": "完整 HTML 僅在記憶體讀取；只保存來源指紋與支持主張的短摘錄。",
    }


torch.manual_seed(0)
torch.set_num_threads(1)
source_path = ROOT / "docs/course-review-20261019/freeze-01/sources/chapter-10.md"
freeze_path = ROOT / "docs/course-review-20261019/freeze-04.json"
source = source_path.read_bytes()
freeze = json.loads(freeze_path.read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-10")
assert digest(source) == "7e5239a9c8c8d65a46c2971f95204bc67b56de3aa6397514aba3a4865b3e5ac7", "凍結原文必須相符"
assert digest(freeze_path.read_bytes()) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單必須相符"
assert page["source_sha256"] == digest(source), "清單與原文必須相符"
assert page["figures_sha256"] == {}, "章首導言不含圖"
chapter = (ROOT / "course/chapters/10.md").read_text()
assert chapter.split("## ", 1)[0].encode("utf-8") == source, "現行章首與凍結章首必須相符"
section_matches = list(re.finditer(r"^## (\d+\.\d+) (.+)$", chapter, re.M))
sections = {}
for index, match in enumerate(section_matches):
    end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(chapter)
    sections[match.group(1)] = chapter[match.start():end]

definitions = {
    "scene": scene,
    "patchify": patchify,
    "VisionEncoder": VisionEncoder,
    "expand_modalities": expand_modalities,
    "MultiModalLM": MultiModalLM,
}
definition_sections = dict(zip(definitions, ["10.1", "10.2", "10.5", "10.7", "10.8"], strict=True))
implementation = (ROOT / "tiny_perceptron/multimodal.py").read_text()
implementation_ast = ast.parse(implementation)
library_nodes = {node.name: node for node in implementation_ast.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
navigation = []
for name, definition in definitions.items():
    section_id = definition_sections[name]
    assert Path(inspect.getsourcefile(definition)).resolve() == ROOT / "tiny_perceptron/multimodal.py", "函式來源必須相符"
    blocks = re.findall(r"```python\n(.*?)```", sections[section_id], re.S)
    matching_nodes = []
    for block in blocks:
        matching_nodes.extend(
            node for node in ast.parse(block).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        )
    assert matching_nodes, "連結小節必須展示指定零件"
    exact_excerpt = any(ast.dump(node) == ast.dump(library_nodes[name]) for node in matching_nodes)
    if name != "MultiModalLM":
        assert exact_excerpt, "完整節錄的實作必須一致"
    navigation.append({"definition": name, "section_id": section_id, "exact_ast_excerpt": exact_excerpt})
assert "精簡" in sections["10.8"] and "聲音" in sections["10.8"], "包裝的圖片精簡版必須有明示範圍"
last_sections = list(sections)[-3:]
assert last_sections == ["10.9", "10.10", "10.11"], "最後三節必須指向圖文配對"
assert "不要求生成句子" in sections["10.9"]
assert "F.cross_entropy" in sections["10.10"]
assert "image_to_text" in sections["10.11"] and "text_to_image" in sections["10.11"]
glossary_path = (ROOT / "course/chapters/../glossary.md").resolve()
glossary = glossary_path.read_text()
assert "embedding" in glossary and "圖片入口" in glossary, "名詞頁必須可用"

image = scene("red", "square")
mask = torch.zeros(16, 16, dtype=torch.bool)
mask[4:13, 4:13] = True
assert tuple(image.shape) == (3, 16, 16), "圖片尺寸必須符合預設"
assert torch.equal(image[0], mask.float()) and torch.count_nonzero(image[1:]).item() == 0, "必須是黑底紅方塊"
patches = patchify(image[None])
manual_patches = torch.stack([
    image[:, row:row + 4, col:col + 4].reshape(-1)
    for row in range(0, 16, 4) for col in range(0, 16, 4)
])[None]
assert torch.equal(patches, manual_patches), "圖塊內容與順序必須相符"
assert torch.equal(unpatchify(patches, 3, 16, 16), image[None]), "切塊後必須能還原"
encoder = VisionEncoder(width=8).eval()
with torch.no_grad():
    encoded = encoder(image[None])
    decomposed = encoder.feature(encoder.projection(patches) + encoder.position)
assert torch.equal(encoded, decomposed), "圖片入口必須使用圖塊投影與位置參數"

tok = ByteTokenizer()
ids = torch.tensor([tok.bos_id, tok.image_id, tok.assistant_id, 73, tok.eos_id])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, tok.eos_id])
embedding = torch.nn.Embedding(tok.vocab_size, 8)
features = torch.arange(24, dtype=torch.float32).reshape(3, 8)
x, y = expand_modalities(ids, labels, embedding, {tok.image_id: features}, {tok.image_id})
expected_x = torch.cat([embedding(ids[:1]), features, embedding(ids[2:-1])])[None]
expected_y = torch.tensor([[IGNORE, IGNORE, IGNORE, IGNORE, 73, tok.eos_id]])
assert torch.equal(x, expected_x) and torch.equal(y, expected_y), "圖片展開後必須對齊輸入與答案"

lm = TinyLM(ModelConfig(width=8)).eval()
model = MultiModalLM(lm).eval()
captured = {}


def record_input(module, args, kwargs):
    if "embeddings" in kwargs:
        captured["embeddings"] = kwargs["embeddings"].detach().clone()


hook = lm.register_forward_pre_hook(record_input, with_kwargs=True)
with torch.no_grad():
    projected = model.image_projector(model.vision(image[None]))[0]
    direct_x, direct_y = expand_modalities(ids, labels, lm.embedding, {tok.image_id: projected}, {tok.image_id})
    result = model(ids, labels, image=image)
    assert torch.equal(captured["embeddings"], direct_x), "文字核心必須收到展開後的圖片與文字特徵"
    assert torch.equal(result["labels"], direct_y), "包裝必須保留對齊後的答案"
    assert torch.isfinite(result["logits"]).all(), "前向計算必須產生有限分數"
    text_ids = torch.tensor([1, 20, 30])
    assert torch.equal(model(text_ids)["logits"], lm(text_ids[None])["logits"]), "純文字路徑必須一致"
hook.remove()

image_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
text_vectors = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
scores = F.normalize(image_vectors, dim=-1) @ F.normalize(text_vectors, dim=-1).T
selected = scores.argmax(-1).tolist()
assert selected == [0, 1], "候選配對必須從現有文字欄選擇"

originals = [
    original_receipt(
        "https://arxiv.org/html/2304.08485v2",
        "Visual Instruction Tuning, arXiv:2304.08485v2，固定修訂版 HTML",
        [
            ("Architecture：將視覺特徵接到文字 embedding 的段落", "a simple linear layer to connect image features into the word embedding space"),
            ("Training／Stage 1: Pre-training for Feature Alignment", "training a compatible visual tokenizer for the frozen LLM"),
        ],
    ),
    original_receipt(
        "https://arxiv.org/html/2103.00020v1",
        "Learning Transferable Visual Models From Natural Language Supervision, arXiv:2103.00020v1，固定修訂版 HTML",
        [
            ("Selecting an Efficient Pre-Training Method：配對目標與逐字預測的區分", "predicting only which text as a whole is paired with which image and not the exact words of that text"),
        ],
    ),
]

paths = [
    "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py",
    "tiny_perceptron/data.py", "course/chapters/10.md", "course/glossary.md",
]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-10/audit.py",
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "seed": "0",
        "torch_threads": str(torch.get_num_threads()), "training_steps": "0",
        "executable": sys.executable,
    },
    "repository_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "repository_sources": [{"path": path, "sha256": digest((ROOT / path).read_bytes())} for path in paths],
    "freeze": {
        "source_sha256": digest(source), "manifest_sha256": digest(freeze_path.read_bytes()),
        "figure_sha256": page["figures_sha256"], "current_introduction_equals_snapshot": True,
    },
    "navigation": navigation,
    "scene": {
        "shape": list(image.shape), "center_rgb": image[:, 8, 8].tolist(),
        "corner_rgb": image[:, 0, 0].tolist(), "foreground_pixels": int(mask.sum()),
        "exact_red_square_on_black": True,
    },
    "patchify": {"shape": list(patches.shape), "manual_slices_equal": True, "roundtrip_equal": True},
    "vision_encoder": {
        "shape": list(encoded.shape), "position_shape": list(encoder.position.shape),
        "projection_plus_position_path_equal": True,
    },
    "expansion": {"input_shape": list(x.shape), "targets": y.tolist(), "input_content_equal": True},
    "multimodal_lm": {
        "projected_image_shape": list(projected.shape), "language_input_shape": list(direct_x.shape),
        "logits_shape": list(result["logits"].shape), "finite_logits": True,
        "input_hook_equals_manual_expansion": True, "text_only_equal": True,
        "interpretation": "僅證明未訓練模型的連接與前向計算，沒有測量辨識或描述品質。",
    },
    "matching_navigation": {
        "last_sections": [{"id": key, "heading": sections[key].splitlines()[0]} for key in last_sections],
        "candidate_scores": scores.tolist(), "selected_candidate_columns": selected,
        "scope": "只核對章首所述候選配對的工作型態，不重審後續小節的數值與實驗。",
    },
    "glossary": {"path": glossary_path.relative_to(ROOT).as_posix(), "relevant_terms_present": True},
    "original_sources": originals,
    "result": "全部定向核對通過；無訓練、無遠端寫入。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
