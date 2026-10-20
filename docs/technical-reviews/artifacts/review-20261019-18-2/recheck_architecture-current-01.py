import difflib
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron import attention
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_18-2/technical-18-2-original-01.json"
original = json.loads(original_path.read_bytes())
assert sha(original_path) == "9ceaf4cf4bda25b6ec1cfac3175757924ee65ebdc68ac975f988a3eb360c0b7e"
old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/18.2.md"
new_path = ROOT / "docs/course-review-20261019/freeze-05/sources/18.2.md"
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(new_path) == "77a1584608df1a6c8eca56be1a12ec4a7a578c0ed3c859dea9d0bc83e6bc62a0"
assert sha(freeze_path) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
old_text = old_path.read_text()
new_text = new_path.read_text()
old_sentence = "較窄的學生，embedding表、各個矩陣與中間特徵（activation）都會縮小；較少層則減少重複處理的次數。"
new_sentence = "較窄的學生，embedding表、權重矩陣與每個位置的特徵向量都會縮小，但注意力分數表的大小由序列長度與頭數決定，不跟著變窄；較少層則減少重複處理的次數。"
assert old_text.count(old_sentence) == 1
assert old_text.replace(old_sentence, new_sentence) == new_text
chapter = (ROOT / "course/chapters/18.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
index = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 18.2 "))
section = chapter[headers[index].start():headers[index + 1].start()]
assert section == new_path.read_bytes()
previous_artifacts = []
for item in original["artifacts"]:
    current_hash = sha(ROOT / item["path"])
    assert current_hash == item["sha256"]
    previous_artifacts.append({"path": item["path"], "sha256": current_hash, "unchanged": True})
old_self_check = ROOT / "docs/technical-reviews/artifacts/review-20261019-18-2/self-check.txt"
assert sha(old_self_check) == "9c5c6b474cc3b05f4f8e03c39ee1a07d29ec92d63ae1740920be697c83532796"
repository_sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        current_hash = sha(ROOT / item["path"])
        assert current_hash == item["sha256"]
        repository_sources.append({"path": item["path"], "sha256": current_hash, "unchanged": True})

counts = []
for width, layers, heads in ((16, 2, 1), (8, 1, 1), (16, 1, 1), (64, 2, 1), (16, 1, 2), (32, 1, 2)):
    model = TinyLM(ModelConfig(width=width, layers=layers, heads=heads))
    count = sum(p.numel() for p in model.parameters())
    count_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert count == 658 * width + layers * (12 * width**2 + 9 * width)
    assert count_bytes == count * 4
    counts.append({"width": width, "layers": layers, "heads": heads, "parameters": count, "fp32_bytes": count_bytes})

shape_rows = []
original_attention = attention.manual_attention
for heads in (1, 2):
    for length in (4, 6):
        for width, layers in ((16, 2), (8, 1)):
            row = {"batch_size": 1, "heads": heads, "length": length, "width": width, "layers": layers, "attention": [], "features": {}}
            model = TinyLM(ModelConfig(width=width, layers=layers, heads=heads))
            hooks = []
            for name, module in model.named_modules():
                if name == "embedding" or name.startswith("blocks.") and (name.count(".") == 1 or name.endswith("ffn.up")):
                    def hook(module, args, value, name=name):
                        tensor = value[0] if isinstance(value, tuple) else value
                        row["features"][name] = list(tensor.shape)
                    hooks.append(module.register_forward_hook(hook))

            def capture(q, k, v, allowed):
                out, weights = original_attention(q, k, v, allowed)
                scores = q @ k.transpose(-2, -1)
                row["attention"].append({"score_shape": list(scores.shape), "weights_shape": list(weights.shape), "elements": scores.numel()})
                assert list(scores.shape) == [1, heads, length, length]
                return out, weights

            attention.manual_attention = capture
            try:
                with torch.no_grad():
                    ids = torch.arange(8, 8 + length, dtype=torch.long)[None]
                    row["logits_shape"] = list(model(ids)["logits"].shape)
            finally:
                attention.manual_attention = original_attention
                for handle in hooks:
                    handle.remove()
            assert row["features"]["embedding"] == [1, length, width]
            assert len(row["attention"]) == layers
            shape_rows.append(row)
        assert shape_rows[-1]["attention"][0]["score_shape"] == shape_rows[-2]["attention"][0]["score_shape"]

wide = TinyLM(ModelConfig(width=16, layers=2))
narrow = TinyLM(ModelConfig(width=8, layers=1))
wide_parameters = dict(wide.named_parameters())
matrices = []
for name, parameter in narrow.named_parameters():
    if parameter.ndim == 2:
        larger = wide_parameters[name]
        assert parameter.numel() < larger.numel()
        matrices.append({"name": name, "wide_shape": list(larger.shape), "narrow_shape": list(parameter.shape)})

run = Path(sys.argv[1]).resolve()
files = {name: (run / name).read_text() for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py")}
extraction = json.loads(files["extraction.json"])
execution = json.loads(files["execution.json"])
assert extraction["source_sha256"] == sha(new_path)
assert execution["exit_code"] == 0
assert files["stderr.txt"] == ""
assert extraction["figure_sha256"] == {}
assert files["fence-1.py"] == json.loads((ROOT / original["artifacts"][0]["path"]).read_bytes())["files"]["fence-1.py"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-2/recheck_architecture-current-01.py " + str(run.relative_to(ROOT)),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "float32"},
    "source_sha256": sha(new_path),
    "source_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)},
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path)},
    "diff": {"old_sentence": old_sentence, "new_sentence": new_sentence, "all_other_text_identical": True, "unified_diff": "".join(difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True), fromfile=str(old_path.relative_to(ROOT)), tofile=str(new_path.relative_to(ROOT))))},
    "current_chapter_section_matches_freeze": True,
    "previous_artifacts": previous_artifacts,
    "previous_self_check_sha256": sha(old_self_check),
    "repository_sources": repository_sources,
    "counts": counts,
    "matrix_comparison": matrices,
    "shape_probe": shape_rows,
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/18.md#18.2" --execute', "files": files},
    "restricted_evidence_reads": [item["path"] for item in original["artifacts"]] + [str(old_self_check.relative_to(ROOT)), str(run.relative_to(ROOT))],
    "result": "embedding、權重矩陣與每位置特徵隨寬度縮小；同一輸入長度與頭數下，單層注意力分數表大小不變；層數只改變重複的Block次數。參數與FP32數字全部吻合。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
