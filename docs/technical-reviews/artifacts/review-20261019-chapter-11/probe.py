"""保存章首導言的原始來源收據、導覽核對與短程式驗證。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def fetch(url, quote, locator, inspection):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    result = subprocess.run(command, capture_output=True, check=True)
    text = result.stdout.decode("utf-8")
    if "/html/" in url:
        parser = Text()
        parser.feed(text)
        text = " ".join(parser.parts)
    normalized = re.sub(r"\s+", " ", text)
    assert quote in normalized, "指定原文摘錄未出現在取得的來源"
    return {
        "url": url,
        "accessed_on": datetime.now().astimezone().date().isoformat(),
        "command": command,
        "exit_code": result.returncode,
        "response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "response_bytes": len(result.stdout),
        "locator": locator,
        "excerpt": quote,
        "inspection_note": inspection,
        "storage_note": "全文僅在記憶體解析；只保存這份收據與支持主張的短摘錄。",
    }


torch.manual_seed(0)
torch.set_num_threads(1)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
before = {name: value.detach().clone() for name, value in model.named_parameters()}
trace = {}


def capture_projector(module, args, output):
    trace["projector_input_shape"] = list(args[0].shape)
    trace["projector_output_shape"] = list(output.shape)
    trace["projector_output"] = output.detach().clone()


def capture_language(module, args, kwargs):
    trace["language_input_shape"] = list(kwargs["embeddings"].shape)
    trace["language_input"] = kwargs["embeddings"].detach().clone()


projector_hook = model.image_projector.register_forward_hook(capture_projector)
language_hook = model.language.register_forward_pre_hook(capture_language, with_kwargs=True)
output = model(ids, labels, image=scene())
projector_hook.remove()
language_hook.remove()
loss = masked_loss(output["logits"], output["labels"])
count = trace["projector_output"].shape[1]
marker_position = prefix.index(tok.image_id)
trace["projected_features_reach_language_exactly"] = torch.equal(
    trace["language_input"][0, marker_position : marker_position + count], trace["projector_output"][0]
)
del trace["language_input"], trace["projector_output"]
forward = {
    "logits_shape": list(output["logits"].shape),
    "labels_shape": list(output["labels"].shape),
    "valid_targets": int((output["labels"] != -100).sum()),
    "all_parameter_gradients_absent": all(p.grad is None for p in model.parameters()),
    "all_parameters_unchanged": all(torch.equal(before[n], p) for n, p in model.named_parameters()),
}
loss.backward()
backward = {
    "projector_gradient_norm": model.image_projector.weight.grad.norm().item(),
    "language_embedding_gradient_absent": model.language.embedding.weight.grad is None,
    "all_parameters_unchanged": all(torch.equal(before[n], p) for n, p in model.named_parameters()),
}
optimizer = torch.optim.SGD(model.image_projector.parameters(), lr=0.1)
optimizer.step()
step = {
    "changed_parameter_names": [n for n, p in model.named_parameters() if not torch.equal(before[n], p)],
    "frozen_parameters_unchanged": all(
        torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad
    ),
}
assert trace["projected_features_reach_language_exactly"], "接頭特徵沒有逐值送進文字核心"
assert forward["all_parameter_gradients_absent"] and forward["all_parameters_unchanged"], "前向檢查不符"
assert backward["projector_gradient_norm"] > 0 and backward["all_parameters_unchanged"], "梯度檢查不符"
assert step["changed_parameter_names"] == ["image_projector.weight", "image_projector.bias"], "更新名單不符"
assert step["frozen_parameters_unchanged"], "凍結參數發生變動"

frozen_path = "docs/course-review-20261019/freeze-01/sources/chapter-11.md"
chapter_path = "course/chapters/11.md"
chapter = (ROOT / chapter_path).read_text()
introduction = chapter.split("## ", 1)[0]
frozen = (ROOT / frozen_path).read_text()
assert introduction == frozen, "現行導言與凍結導言不同"
assert sha(frozen_path) == "1ab631c8be8a2e2f34b56e93aa79f2d36b13c95f808aecea969ae4ae528469fc"
assert sha("docs/course-review-20261019/freeze-04.json") == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
headings = re.findall(r"^## (.+)$", chapter, re.M)
roadmap = {
    "接通、梯度與參數更新": ["接頭輸出不同，就算理解圖片了嗎？", "怎麼確認真正被更新的是哪一批參數？", "只調圖片接頭，梯度怎麼穿過固定文字核心？"],
    "同圖多問": ["同一張圖片，換問題後應怎麼改答案？"],
    "換圖重測": ["換圖後答對新圖，才能說用了圖片嗎？"],
    "原文字題對照": ["新圖文能力進步時，舊文字能力有沒有退步？"],
    "前處理像素": ["裁切後，模型還收得到作答線索嗎？", "筆畫已被縮掉，較大的模型能確定補回嗎？"],
    "讀字與讀序評分": ["四個字對三個，整串編號就正確嗎？", "字都出現了，為什麼仍可能讀錯順序？"],
    "服飾與短字詞任務設計": ["有限物件與短字詞，如何做成可學的看圖問答？", "指定哪一塊文字，如何只讀那一塊？"],
}
roadmap_matches = {}
for topic, titles in roadmap.items():
    matches = [h for h in headings if any(h.endswith(title) for title in titles)]
    assert len(matches) == len(titles), "導言預告的主題沒有對應標題"
    roadmap_matches[topic] = matches
links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", frozen)
link_checks = []
for link in links:
    path, _, anchor = link.partition("#")
    target = (ROOT / chapter_path).parent / path
    assert target.is_file(), "導言連結的目標檔不存在"
    anchor_found = not anchor or bool(re.search(r"^## " + re.escape(anchor) + r" ", target.read_text(), re.M))
    assert anchor_found, "導言連結的目標標題不存在"
    link_checks.append({"link": link, "file_exists": True, "anchor_found": anchor_found})

sources = [
    fetch(
        "https://arxiv.org/html/2304.08485v2",
        "This stage can be understood as training a compatible visual tokenizer for the frozen LLM.",
        "Architecture；Training 下的 Pre-training for Feature Alignment",
        "已讀架構與第一階段訓練：Vicuna 是語言模型，線性接頭把 CLIP 特徵映到語言 embedding 寬度；以圖片描述目標訓練接頭而凍結兩端。支持本頁的文字核心與模態對齊定義，不證明本課模型已學會看圖。",
    ),
    fetch(
        "https://arxiv.org/html/1612.00837v3",
        "a pair of similar images that result in two different answers to the question.",
        "Abstract；Introduction；Dataset",
        "已讀摘要、導言與資料設計脈絡：相同問題配上應有不同答案的互補圖片，用以減少語言先驗造成的表面成功。支持相關圖片內容變更的對照，不表示任意換圖都應改答。",
    ),
    fetch(
        "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_tensor.py",
        "This function accumulates gradients in the leaves",
        "Tensor.backward",
        "已讀 Tensor.backward 的文件與呼叫：計算並累積葉節點梯度，不在這個方法中執行最佳化器更新。版本與本機 torch 相同。",
    ),
    fetch(
        "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/optim/sgd.py",
        "param.add_(grad, alpha=-lr)",
        "SGD.step；_single_tensor_sgd",
        "已讀 SGD.step 到 sgd 與單 tensor 更新路徑，更新直接作用於參數。正的學習率配合非零梯度才在本次無動量、無衰減配置觀察到變化；不能宣稱所有 step 都必有變化。",
    ),
]
with (ARTIFACTS / "primary-sources.json").open("x", encoding="utf-8") as handle:
    json.dump(sources, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --offline python docs/technical-reviews/artifacts/review-20261019-chapter-11/probe.py",
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "platform": platform.platform(),
        "seed": "0",
        "threads": str(torch.get_num_threads()),
    },
    "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "file_sha256": {
        path: sha(path)
        for path in [frozen_path, "docs/course-review-20261019/freeze-04.json", chapter_path, "course/chapters/10.md", "course/glossary.md", "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py"]
    },
    "source_freeze_matches": True,
    "current_introduction_matches_snapshot": True,
    "image_references_in_introduction": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", frozen),
    "roadmap_matches": roadmap_matches,
    "link_checks": link_checks,
    "trace": trace,
    "after_forward": forward,
    "after_backward": backward,
    "after_optimizer_step": step,
    "limitations": "隨機初始化、單一合成圖片、單次前向與反向、單次 SGD 更新；僅驗證接線及更新機制，不量測生成品質、舊技能保留或跨平台契約。章內其他主題僅核對導覽承諾，不延伸為其他小節的正式技術審閱。",
    "result": "所有斷言通過；來源全文未寫入工作樹。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
