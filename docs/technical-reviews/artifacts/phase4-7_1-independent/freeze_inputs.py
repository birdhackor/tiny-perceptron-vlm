"""Preserve original raw UTF-8 review inputs; no prior reports are read."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
raw = (ROOT / "course/chapters/07.md").read_bytes()
intro = raw[:raw.index(b"## 7.1 ")]
(HERE / "inputs").mkdir(exist_ok=True)
(HERE / "inputs/chapter-intro.md").write_bytes(intro)
paths = [
    "course/chapters/07.md",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "scripts/build_course.py",
    "scripts/check_technical_reviews.py",
    "docs/review-tools/section_facts.py",
    "docs/review-tools/factual-reviewer-instructions.md",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
    "pyproject.toml",
]
manifest = []
for relative in paths:
    original = ROOT / relative
    target = HERE / "inputs" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    manifest.append({"path": relative, "sha256": sha(original), "snapshot": str(target.relative_to(ROOT))})
run = ROOT / "outputs/reviewer-tools/phase4-7_1-independent"
(HERE / "original-fences").mkdir(exist_ok=True)
for p in run.iterdir():
    if p.is_file():
        shutil.copyfile(p, HERE / "original-fences" / p.name)
provenance = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_7_1",
    "source": "course/chapters/07.md#7.1",
    "intro_sha256": hashlib.sha256(intro).hexdigest(),
    "intro_summary": "本章以1+1問答追蹤角色、首回答的預測位置、可讀前文和有代價的答案；再接填充、EOS、截斷、SFT與留出檢查，最後討論材料與共用底座配方及歷史和新回答共用的長度預算。這是章節路线，不是已驗證成品或能力宣告。",
    "actual_read_scope": [
        "course/chapters/07.md raw lines 1-175 read for context; formal review restricted to full 7.1 lines 5-39",
        "course/chapters/06.md lines 203-236 (6.6 boundary contract), plus heading/byte/BOS/EOS search locators",
        "tiny_perceptron/data.py entire file; model.py lines 1-145; build_course.py lines 1-130 and BOOTSTRAP",
        "current review instructions, section_facts.py entire file and check_technical_reviews.py entire file; review-protocol.md entire file",
    ],
    "figure_scope": "7.1 references no SVG or other image; no visual artifact exists to render in this section.",
    "input_provenance": "Section/fence snapshots were extracted from current raw Markdown bytes by the personally read current helper. No historic numerical/model result is cited in 7.1. The exact runtime output is data preprocessing, not model prediction, training or evaluation. No datasets or model weights were loaded.",
    "snapshot_manifest": manifest,
}
(HERE / "input-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(provenance, ensure_ascii=False, indent=2))
