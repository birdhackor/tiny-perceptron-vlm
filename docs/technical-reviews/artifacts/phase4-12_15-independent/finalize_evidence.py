"""Freeze named proof inputs and own inspection record; does not read any review report."""
import hashlib
import json
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]

def main():
    code_scope = {
        "tiny_perceptron/multimodal.py": ["imports L3-9", "tone L44-46", "mel_filter_bank L49-59", "log_mel L62-69", "AudioEncoder L72-81", "expand_modalities L84-100", "MultiModalLM L103-135", "generate_modal L139-174"],
        "tiny_perceptron/model.py": ["ModelConfig L15-28", "Block L31-50", "TinyLM L53-89", "loss_sum L92-100", "masked_loss L103-105"],
        "tiny_perceptron/data.py": ["ByteTokenizer L14-28"],
        "tiny_perceptron/attention.py": ["attention_mask L10-18", "manual_attention L21-28", "CausalAttention L31-74"],
        "tiny_perceptron/modern.py": ["DenseFFN L35-53 (default gelu branch; other branches read as same function, not executed)"],
    }
    copies = []
    for name, scope in code_scope.items():
        origin = ROOT / name
        if origin.is_symlink():
            raise ValueError("Do not follow code symlink: " + name)
        raw = origin.read_bytes()
        target = BASE / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        copies.append({"original_path": name, "snapshot": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "actually_read_locators": scope})

    cpu_command = "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-12_15-independent/bounded_cpu_check.py > docs/technical-reviews/artifacts/phase4-12_15-independent/cpu.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_15-independent/cpu.stderr.txt"
    render_command = "inkscape docs/technical-reviews/artifacts/phase4-12_15-independent/inputs/new-12.15-direct-audio.svg --export-type=png --export-width=1600 --export-filename=docs/technical-reviews/artifacts/phase4-12_15-independent/figure.png > docs/technical-reviews/artifacts/phase4-12_15-independent/figure-render.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_15-independent/figure-render.stderr.txt"
    receipt = {
        "reviewer_task": "/root/phase4_factual_coordinator/factual_12_15", "cwd": str(ROOT),
        "cpu_command": cpu_command, "cpu_exit_code": 0,
        "render_command": render_command, "render_exit_code": 0,
        "render_environment": {"inkscape": "1.4 (e7c3feb100, 2024-10-09)", "device": "CPU", "render_width": "1600 px", "original_svg_viewBox": "0 0 640 760"},
        "pdf_conversion": {"tool": "pdftotext 25.03.0", "commands": [
            "pdftotext -layout docs/technical-reviews/artifacts/phase4-12_15-independent/sources/minds14-2104.08524v1.pdf docs/technical-reviews/artifacts/phase4-12_15-independent/sources/minds14-2104.08524v1.txt",
            "pdftotext -layout docs/technical-reviews/artifacts/phase4-12_15-independent/sources/slu-1904.03670v1.pdf docs/technical-reviews/artifacts/phase4-12_15-independent/sources/slu-1904.03670v1.txt"
        ], "exit_codes": [0, 0]},
        "actual_visual_inspection": "view_image 讀取實際 Inkscape PNG；畫面由系統縮為1439×1709，全部節點與底部說明可見。已核真人錄音→固定log-mel→自行訓練入口+接頭→同一文字核心→希望答案（手寫示例）；底部明載色塊為示意。沒有瀏覽器頁面檢查，不宣稱手機/桌面正文可讀。",
        "render_warnings": "Inkscape PangoFT2FontMap/GtkRecentManager warnings; exit 0 and visually complete PNG, no missing labels seen.",
        "repository_code_snapshots": copies,
        "authority_inspection": {
            "minds14-readme": "親讀固定revision全文（官方資料卡非repo審閱摘要）；再CPU解析/ license與/dataset_info/14/config_name、features、splits/0/num_examples。標示CC-BY-4.0、502、六欄位及三类标签 IDs 1/2/6；沒有客服答覆/speaker_id欄。只支持該公開schema；未下載parquet/音訊，未核三類可用短句數量或說話者隔離。",
            "minds14-paper": "實讀arXiv:2104.08524v1（17 Apr 2021）Abstract、§1、§2、§3、§4 Speech Transcription/Training setup（layout TXT L1-180），Appendix A.1/A.2（L430-467）。§2 subjects given intent/class description/three examples then produce new utterances; Chinese included; §3固定預訓練mUSE/LaBSE; §4 Google ASR; A.2 ZH502。未將論文模型分數當作自行訓練成績。",
            "slu-paper": "實讀arXiv:1904.03670v1（7 Apr 2019）Abstract/§1/§2/§3 collection beginning（TXT L1-107）、§4/§4.1/§4.4/§4.5/§5.1（L197-246）、Table2/§5.2/§5.3/Conclusion（L268-320）。End-to-end無显式逐字稿→意圖；使用詞/音素預訓練；§5.3仍有新措辭/同義詞限制。英文Fluent Speech Commands不是中文MInDS，也不是本課生成客服答覆的驗收。",
            "librosa-docs": "官方0.11.0 melspectrogram API signature、spectrogram mapping、power=2、hop_length/window/center/return (...,n_mels,t)；power_to_db scaling/amin/reference。確認固定算式與保留時間軸；本repo使用自然log而非10log10，不宣稱輸出數值等同librosa預設。未安裝或執行librosa。",
            "pytorch-docs": "官方2.9 Linear API formula/shape/learnable weights/uniform initialization；CrossEntropyLoss input logits/target indices/ignore_index=-100/nonignored mean/backward例。支持probe中投影與答案損失API，安裝執行torch2.14.1+cpu，版本差明標；未聲稱逐版本源碼同一。",
            "sklearn-docs": "官方1.7 GroupKFold class description、non-overlapping groups/each group once test、parameters、See also；支持按原錄音family隔離的原則。未執行sklearn或構造真資料split。"
        },
        "failures_and_limits": ["Pinned minds14.py raw URL HTTP404；該revision公開README包含完整features/schema已足以核教材的metadata主張，沒有冒稱讀到loader。", "沒有原Python/bash fences；未呼叫helper --execute。", "沒有讀取、播放、轉寫真人音訊；没有train/test実测、weight下载、optimizer step或GPU。", "沒有讀舊canonical report/reader verdict/author correction summary；初次filename inventory有其他report檔名但未讀内容。"],
        "independent_reasoning": [
            "保留相同問題/歷史而替換音訊，只改音訊因素，可檢查答覆是否依需求主題變化；這是必要控制，不單獨保證語意正確。",
            "對所有音訊固定輸出同一句客服話的函式不使用音訊，因此語句合理不能證明分辨三類。",
            "裁切/加噪是原錄音的變體；僅以不同檔名分train/test可以有同一原錄音family重疊，不能稱作未見過的原錄音。",
            "未知帳戶資訊不是意圖標籤或所選資料的欄位；設計不連銀行，因此一般提示不能宣稱取得真實餘額/完成操作。",
            "依有限標籤產生回覆與完整逐字轉寫是不同目標；有限需求成功的命題不含任意中文、新名稱與数字辨识，所以不能推出這些能力。"
        ],
    }
    (BASE / "inspection-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"snapshot_count": len(copies), "inspection_receipt": str(BASE / "inspection-receipt.json")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
