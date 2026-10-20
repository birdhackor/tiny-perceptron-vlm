"""只保存來源收據、小段摘錄與查核定位，不保存第三方全文。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sources = [
    {
        "name": "ctc", "url": "https://www.cs.toronto.edu/~graves/icml_2006.pdf",
        "version": "ICML 2006 proceedings PDF",
        "locator": "From Network Outputs to Labellings; Constructing the Classifier; Maximum Likelihood Training; equations (2), (3), (4), (12)",
        "excerpt": "However it is not guaranteed to find the most probable labelling.",
        "inspection": "以web PDF文字閱讀原始論文的CTC定義、路徑乘積、標籤路徑加總、blank與重複合併、best-path限制及最大概似目標；curl取得同一PDF核對hash。blank代表no label，不代表白色像素。",
    },
    {
        "name": "fashion_readme", "url": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md",
        "version": "b2617bb6d3ffa2e429640350f613e3291e10b141",
        "locator": "Labels; Citing Fashion-MNIST",
        "excerpt": "| 1 | Trouser |\n| 8 | Bag |\n| 9 | Ankle boot |",
        "inspection": "閱讀28×28灰階商品影像說明、官方類別表與作者引用；對照本專案FASHION_CLASSES選取1、8、9。",
    },
    {
        "name": "fashion_license", "url": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE",
        "version": "b2617bb6d3ffa2e429640350f613e3291e10b141", "locator": "License header",
        "excerpt": "The MIT License (MIT) Copyright © 2017 Zalando SE",
        "inspection": "原始檔標明MIT及2017 Zalando SE，教材署名與原始資料授權相符。",
    },
    {
        "name": "noto_sans", "url": "https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/LICENSE",
        "version": "f8d157532fbfaeda587e826d4cd5b21a49186f7c", "locator": "License opening",
        "excerpt": "This Font Software is licensed under the SIL Open Font License,\nVersion 1.1.",
        "inspection": "Sans原始字型授權為SIL OFL 1.1；本頁僅陳述字型來源與授權名稱。",
    },
    {
        "name": "noto_serif", "url": "https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Serif/LICENSE",
        "version": "f8d157532fbfaeda587e826d4cd5b21a49186f7c", "locator": "License opening",
        "excerpt": "SIL OPEN FONT LICENSE Version 1.1",
        "inspection": "Serif原始授權與Sans檔案的SHA-256相同，兩者皆標明SIL OFL 1.1。",
    },
    {
        "name": "minds", "url": "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md",
        "version": "40ce77cb32a384e4d50a568e1ec39ac804019d33", "locator": "YAML license/config zh-CN; Data Fields; Dataset Summary",
        "excerpt": "license:\n- cc-by-4.0",
        "inspection": "閱讀原始PolyAI卡片：中文zh-CN、意圖名稱address/app_error/card_issues；資料欄位只有path、audio、transcription、english_transcription、intent_class、lang_id，沒有speaker ID。授權metadata為cc-by-4.0。",
    },
    {
        "name": "torch_ctc", "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html",
        "version": "PyTorch 2.14 documentation; local runtime 2.14.1", "locator": "CTCLoss definition; Shape; blank; log_probs",
        "excerpt": "The Connectionist Temporal Classification loss.",
        "inspection": "核對CTCLoss加總可對齊目標的路徑、blank預設0、log_probs為(T,N,C)，target長度不得超過input；本專案傳入32×1×13的log_softmax與1–4字target。",
    },
    {
        "name": "public_export", "url": "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint/inference-manifest.json",
        "version": "979cdfacc588ad0536f1c64fff96f264571cf054", "locator": "origin; freeze_perception_backbones; stage; selected_step; files",
        "inspection": "核對公開成品manifest的來源宣告與凍結旗標；這是可稽核的專案訓練出處，不是重跑完整訓練。",
    },
]
for source in sources:
    response = subprocess.run(["curl", "-fLsS", "--max-time", "90", source["url"]], capture_output=True, check=True)
    raw = response.stdout
    source.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), accessed_on="2026-10-20", checked_original=True)
    if source["name"] == "public_export":
        value = json.loads(raw)
        source["selected_values"] = {key: value[key] for key in ("stage", "selected_step", "freeze_perception_backbones", "preprocess_version", "files")}
        source["selected_values"]["origin_kind"] = value["origin"]["kind"]
    elif source["name"] != "ctc":
        text = raw.decode()
        excerpt = source.get("excerpt", "")
        if source["name"] == "fashion_readme":
            assert all(line in text for line in excerpt.splitlines())
        else:
            assert excerpt in text or excerpt in re.sub("<[^>]+>", "", text)
    print(source["name"], source["sha256"])
value = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-6/inspect_sources.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
    "sources": sources,
    "scope": "第三方全文只在curl程序與記憶體中經過，檔案僅保存收據及必要摘錄；CTC的閱讀定位另由本回合web原始PDF擷取確認。",
}
(OUT / "source-inspection.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
