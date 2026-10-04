# 課程權重發布：來源許可與模型卡聲明

核驗日期：2026-10-02。依據 `assets/training/sources/*.json` 的固定 revision，重新讀取官方資料卡、上游 LICENSE、CDLA 與 Creative Commons 法律文本；下載原文及雜湊保存在 ignored `outputs/technical-sources/release-licenses/`。

課程程式碼的 [MIT LICENSE](../../LICENSE) 不會把訓練資料一併改成 MIT。公開模型卡應分別標明 **程式碼許可、該權重的許可、各訓練資料來源及許可**。以下是具體發布配置；本輪 PKU 衍生權重與完整訓練檔案不進公開學生模型包，完整報告備份在私有 HF。較早 Actions 診斷包含選取的公開來源片段與模型生成，詳見下方紀錄。

## 公開模型卡與 notice 配置

| 使用的資料 | 核驗到的資料許可／官方來源 | 模型卡應保留的來源聲明 | 權重發布配置 |
|---|---|---|---|
| TinyStories | [固定資料卡][tinystories-card]：CDLA-Sharing-1.0 | `roneneldan/TinyStories`；Ronen Eldan、Yuanzhi Li；revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`；[原論文](https://arxiv.org/abs/2305.07759)。說明選取完整故事、去除邊緣空白及 JSONL 轉換。 | 獨立發布的合格 Results 可採 `MIT`；公開語料子集仍用 CDLA-Sharing-1.0，保留署名、許可連結及修改聲明。 |
| Chinese-poetry 唐詩三百首 | [固定 LICENSE][poetry-license]／[資料 README][poetry-readme]：MIT | `chinese-poetry/chinese-poetry`；`Copyright (c) 2016 JackeyGao`；revision `b8594f81a89752241442f2ce267d6f66f96704ee`。保留原詩題與作者，說明 JSONL 轉換及移除一筆完全重複資料。 | 課程自行訓練權重採 `MIT`；附上原完整 MIT notice，不把古詩作者改署為模型作者。 |
| UltraChat 200k | [HuggingFaceH4 固定資料卡][ultrachat-card]：`license: mit`；[原 UltraChat LICENSE][ultrachat-license] | 同時署名 `HuggingFaceH4/ultrachat_200k` 與原 UltraChat；資料 revision `8049631c405ae6576f93f445c6b8166f76f5505a`、`train_sft`；保留上游 `Copyright (c) 2023 THUNLP` 及完整 MIT notice，引用 [UltraChat 論文](https://arxiv.org/abs/2305.14233)。 | 課程自行訓練權重採 `MIT`；資料卡的 MIT 是資料許可證據，上游專案 LICENSE 提供其 notice 文本。 |
| UltraFeedback binarized | [HuggingFaceH4 固定資料卡][ultrafeedback-card]：`license: mit`；[原 UltraFeedback LICENSE][ultrafeedback-license] | 同時署名 `HuggingFaceH4/ultrafeedback_binarized` 與原 UltraFeedback；資料 revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`、`train_prefs`；保留上游 `Copyright (c) 2023 THUNLP` 及完整 MIT notice，引用 [UltraFeedback 論文](https://arxiv.org/abs/2310.01377)。 | 課程自行訓練權重採 `MIT`；不要替 H4 加工版本杜撰未提供的 copyright holder。 |
| FSDD | [固定 README 的 License][fsdd-card]：CC-BY-SA-4.0 | `Free Spoken Digit Dataset (FSDD)`；發布者 `Jakobovski/free-spoken-digit-dataset` 與貢獻者；revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`；CC BY-SA 4.0 連結。說明實際錄音／說話者索引及自訂 speaker holdout；原 WAV 未修改。 | 本次可直接採 **權重 CC-BY-SA-4.0、程式碼 MIT** 的配置。這是課程主動選擇的相容發布方式，不是認定所有 FSDD 訓練權重依法必然受 SA 約束。 |
| Fashion-MNIST | [Zalando 原 LICENSE][fashion-license] 與 [固定資料卡][fashion-card]：MIT | `zalandoresearch/fashion-mnist`；原 notice `Copyright © 2017 Zalando SE, https://tech.zalando.com`；repo revision `b2617bb6d3ffa2e429640350f613e3291e10b141`、HF revision `531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2`。說明 first-five-per-label 50 張與自訂 3/1/1 holdout、16×16 RGB 預處理。 | 本課程 Fashion 分支權重採 `MIT`，附完整 Zalando notice；不把程式碼 MIT 當成資料許可證據。 |
| GSM8K | [OpenAI 原 LICENSE][gsm-license] 與 [固定資料卡][gsm-card]：MIT | `openai/grade-school-math`；`Copyright (c) 2021 OpenAI`；repo revision `3101c7d5072418e28b9008a6636bde82a006892c`、HF revision `740312add88f781978c0658806c59bc2815b9866`。按實際分支標明前 200 題、人類原答案、截取／分窗／family split；僅評估使用不得寫成訓練來源。 | 本課程 GSM8K 訓練分支權重採 `MIT`，附完整 OpenAI notice；保留人類答案來源，不稱教師生成。 |
| PKU-SafeRLHF | [固定資料卡][pku-card]：CC-BY-NC-4.0 | 私有模型卡保留 `PKU-Alignment/PKU-SafeRLHF`、revision `9421ffafec3fa40a1f1a7d567b4d525079477ecb`、`alpaca3-8b/train` 與資料卡 citation。 | **本輪衍生權重不公開**：額外選樣、截短、切分樣本及 checkpoint、adapter、merged、distilled 不進公開權重包；既有固定來源 archive 仍按 CC-BY-NC-4.0 提供。這是本項目發布政策；CC BY-NC 本身允許符合條件的非商業公開分享，不能把「私有」寫成該許可的原文規定。 |
