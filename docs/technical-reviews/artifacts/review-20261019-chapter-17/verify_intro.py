"""核對凍結導言、導覽與決策反例；外部原文僅在記憶體中處理。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "pre"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def inspect_original(url, version, selections):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    parser = TextParser()
    parser.feed(raw.decode("utf-8"))
    text = re.sub(r"\s+", " ", "".join(parser.parts)).strip()
    excerpts = []
    for locator, excerpt in selections:
        normalized = re.sub(r"\s+", " ", excerpt).strip()
        if normalized not in text:
            raise ValueError(f"找不到來源摘錄：{url}，定位：{locator}")
        excerpts.append({"locator": locator, "excerpt": normalized})
    title = re.search(r"<title>(.*?)</title>", raw.decode("utf-8"), re.S)
    return {
        "url": url,
        "version": version,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "response_bytes": len(raw),
        "response_sha256": sha256(raw),
        "html_title": title.group(1) if title else "",
        "excerpts": excerpts,
        "full_text_saved": False,
    }


def main():
    source_path = "docs/course-review-20261019/freeze-01/sources/chapter-17.md"
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    chapter_path = "course/chapters/17.md"
    raw = (ROOT / source_path).read_bytes()
    freeze_raw = (ROOT / freeze_path).read_bytes()
    freeze = json.loads(freeze_raw)
    page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-17")
    assert sha256(raw) == "829f21e17da1a92d497b20b01f7fb64bf8ab269f2907fe04fbac6af65dc3b947"
    assert sha256(freeze_raw) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
    assert page["snapshot"] == source_path
    assert page["source_sha256"] == sha256(raw)
    chapter_raw = (ROOT / chapter_path).read_bytes()
    intro = chapter_raw.split(b"\n## ", 1)[0] + b"\n"
    assert intro == raw
    body = raw.decode("utf-8")
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)|<img\b", body)
    assert not figures and page["figures_sha256"] == {}
    assert "```" not in body

    links = []
    for label, reference in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", body):
        path, _, fragment = reference.partition("#")
        target = (ROOT / chapter_path).parent / path
        target = target.resolve()
        assert target.is_file()
        entry = {"label": label, "reference": reference, "path": target.relative_to(ROOT).as_posix()}
        entry["sha256"] = sha256(target.read_bytes())
        if fragment:
            heading = re.search(rf"^## {re.escape(fragment)} .+$", target.read_text(), re.M)
            assert heading is not None
            entry["heading"] = heading.group(0)
            linked = next(item for item in freeze["pages"] if item["page_id"] == fragment)
            linked_raw = (ROOT / linked["snapshot"]).read_bytes()
            assert sha256(linked_raw) == linked["source_sha256"]
            definition = next(line for line in linked_raw.decode().splitlines() if "**中間特徵（activation）**" in line)
            entry["frozen_target"] = linked["snapshot"]
            entry["frozen_sha256"] = sha256(linked_raw)
            entry["definition"] = definition
        links.append(entry)

    headings = re.findall(r"^## .+$", chapter_raw.decode(), re.M)
    glossary = (ROOT / "course/glossary.md").read_text()
    assert all(term in glossary for term in ("activation", "quantization", "range calibration"))

    originals = [
        inspect_original(
            "https://docs.pytorch.org/docs/2.8/quantization.html",
            "PyTorch 2.8 documentation; Last Updated On: Jun 17, 2025",
            [
                ("Introduction to Quantization", "storing tensors at lower bitwidths than floating point precision"),
                ("Introduction to Quantization", "a more compact model representation"),
                ("Quantization Aware Training for Static Quantization", "by clamping and rounding to simulate the effects of INT8"),
                ("(Prototype - maintenance mode) FX Graph Mode Quantization", "There are multiple quantization types in post training quantization (weight only, dynamic and static)"),
                ("General Quantization Flow / Dynamic/Weight Only Quantization", "activation dynamically quantized (fp16, int8) or not quantized"),
                ("Post Training Static Quantization", "It requires calibration with a representative dataset"),
                ("Post Training Static Quantization / Diagram", "previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32"),
                ("Quantization Aware Training for Static Quantization", "models the effects of quantization during training"),
                ("Backend/Hardware Support", "without AVX2 some operations have inefficient implementations"),
            ],
        ),
        inspect_original(
            "https://docs.pytorch.org/docs/2.9/generated/torch.argmax.html",
            "PyTorch 2.9 documentation, torch.argmax(input)",
            [("torch.argmax(input)", "Returns the indices of the maximum value of all elements in the input tensor.")],
        ),
    ]
    receipt_path = ARTIFACT_DIR / "source-receipts.json"
    assert not receipt_path.exists()
    receipt_path.write_text(json.dumps(originals, ensure_ascii=False, indent=2) + "\n")

    original_scores = [Fraction("1"), Fraction("1.005")]
    perturbed_scores = [Fraction("1.01"), Fraction("1.005")]
    maximum_index = lambda values: max(range(len(values)), key=values.__getitem__)
    mae = sum(abs(left - right) for left, right in zip(original_scores, perturbed_scores)) / len(original_scores)
    assert mae == Fraction("0.005")
    assert maximum_index(original_scores) == 1
    assert maximum_index(perturbed_scores) == 0
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU; Python 標準函式庫，沒有載入 torch"},
        "source": {"path": source_path, "sha256": sha256(raw), "bytes": len(raw), "matches_current_intro": True},
        "freeze": {"path": freeze_path, "sha256": sha256(freeze_raw)},
        "scope": {"paragraphs": len([part for part in body.strip().split("\n\n") if not part.startswith("#")]), "figures": figures, "code_blocks": 0},
        "navigation": {"links": links, "chapter_headings": headings, "glossary_terms_present": True},
        "counterexample": {
            "purpose": "核對數值接近不能單獨保證離散選擇相同；這不是實際模型品質實驗。",
            "original_scores": [str(value) for value in original_scores],
            "perturbed_scores": [str(value) for value in perturbed_scores],
            "mean_absolute_difference": str(mae),
            "original_maximum_index": maximum_index(original_scores),
            "perturbed_maximum_index": maximum_index(perturbed_scores),
            "derivation": "對任意正數 ε，分數 (0, ε) 與 (2ε, ε) 的平均絕對差是 ε，而最大值位置不同。令 ε 趨近零，數值差可任意小，greedy 選擇仍不同。",
        },
        "source_receipts": {"path": receipt_path.relative_to(ROOT).as_posix(), "sha256": sha256(receipt_path.read_bytes())},
        "result": "所有斷言通過；未進行訓練、模型品質或硬體速度量測。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
