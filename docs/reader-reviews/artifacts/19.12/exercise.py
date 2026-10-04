"""Answer the design exercise; this does not evaluate a speech model."""

import json
from pathlib import Path

answer = {
    "chosen_matrix_task": "單獨音高",
    "new_input": "一段真實說話者錄下的中文『紅色方形』語音；問題：這段聲音說了什麼？",
    "claim_to_test": "轉寫未見說話者錄音中的語句，而非把兩群純音分成 high/low。",
    "new_data": "收集有來源、使用許可與逐句文字標註的錄音，包含不同說話者、語速及背景聲。",
    "split_rule": "先按說話者與錄音場次分家族，再切 train/validation/test；同場錄音片段不能跨份。",
    "checks": [
        "validation 先選設定，test 封存到定案；保留每題來源、原始生成 ID、EOS 與轉寫。",
        "固定句子完全相同的比例，以及字元錯誤率；分母與字元插入/刪除/替換定義先寫清楚。",
        "分別報新說話者、語速及背景聲條件，並與總答同一句的基線比較。",
        "保留問題換錄音，按新錄音重新標真值；兩題都對才計成對成功。",
        "重跑原本純音題，檢查新增語音訓練是否降低舊能力。",
    ],
    "representation_reason": "目前沿時間平均的 16 項聲學摘要丟掉順序；需要另研究保留時間資訊的輸入表示。",
    "status": "這是練習設計答案，沒有取得錄音、訓練或測量新能力。",
}
out = Path(__file__).resolve().parent / "exercise-answer.json"
out.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(answer, ensure_ascii=False, indent=2))
