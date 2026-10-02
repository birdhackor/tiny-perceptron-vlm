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

權重包根目錄的 `LICENSE` 寫該權重的許可；`THIRD_PARTY_NOTICES.md` 寫上表的來源、revision、原 copyright、許可連結及修改說明。需附原文的 MIT notice 可分別保存為 `licenses/chinese-poetry-MIT.txt`、`licenses/UltraChat-MIT.txt`、`licenses/UltraFeedback-MIT.txt`。實際附帶的 TinyStories 語料與 FSDD WAV／改編資料各自保持原許可，不能由權重包的 MIT 聲明覆蓋。

模型卡 `license:` 指**該模型權重**，不要填成「所有訓練資料的統一許可」。MIT 模型可用 `license: mit`；上述 FSDD 配置用 `license: cc-by-sa-4.0`。若同一發布目錄包含多個不同許可的模型，按檔案列出許可映射；FSDD 模型卡單獨標明權重許可。

可直接加入發布 manifest 的欄位示例：

```json
{
  "code_license": "MIT",
  "weights_license": "MIT",
  "contains_source_data": false,
  "pku_derived": false,
  "training_sources": [{
    "id": "roneneldan/TinyStories",
    "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    "data_license": "CDLA-Sharing-1.0"
  }]
}
```

FSDD 的對應示例把 `weights_license` 改為 `CC-BY-SA-4.0`、`training_sources` 改為其固定來源。`contains_source_data` 必須按實際附件填寫；此示例不是已發布 artifact 的檢驗結果。

## TinyStories：CDLA-Sharing-1.0 明文排除哪些 Results

[CDLA-Sharing-1.0 官方法律文本](https://cdla.dev/sharing-1-0/) 透過「計算使用 → Results」明確排除結果的共享義務。法律文本沒有逐一列舉「模型權重」，也沒有對所有含原語料的檔案提供無條件模型豁免；訓練結果適用豁免的關鍵是 §1.11 的定義。

§1.2 Computational Use：

> 1.2 “Computational Use” means Your analysis (through the use of computational devices or otherwise) or other interpretation of Data. By way of example and not limitation, “Computational Use” includes the application of any computational analytical technique, the purpose of which is the analysis of any Data in digital form to generate information about Data such as patterns, trends, correlations, inferences, insights and attributes.

§1.3 Data：

> 1.3 “Data” means the information (including copyrightable information, such as images or text), collectively or individually, whether created or gathered by a Data Provider or an Entity acting on its behalf, to which rights are granted under this Agreement.

§1.11 Results：

> 1.11 “Results” means the outcomes or outputs that You obtain from Your Computational Use of Data. Results shall not include more than a de minimis portion of the Data on which the Computational Use is based.

§2.1 對 Data 使用／發布的授權：

> 2.1 Subject to the conditions set forth in Section 3 of this Agreement, Data Provider(s) hereby grant(s) to You a worldwide, non-exclusive, irrevocable (except as provided in Section 5) right to: (a) Use Data; and (b) Publish Data.

§3.5 Results 的使用與發布：

> 3.5 This Agreement imposes no obligations or restrictions on Your Use or Publication of Results.

因此，可把獨立發布且符合 Results 定義的課程模型權重採用 MIT；不因使用 TinyStories 訓練就必須把這些 Results 改授 CDLA。`de minimis` 不是固定百分比：官方 FAQ 解釋，它不能是足以替代原資料的顯著子集。不能用 Results 標籤把語料 archive、故事摘錄包或大量原文一起重新授 MIT。

[官方 FAQ](https://cdla.dev/faq-resources/faq/) 的 **Permissive and Sharing Version 1.0 FAQ**，問題 “In the Sharing version of the Agreement, why are Results excluded from the sharing obligations that apply to Enhanced Data?”：

> Results are separate works from the Data licensed under the Agreement, and therefore are free of any obligation to Publish them under the Agreement – if you choose to publish them at all. You never have any obligation to share Results if You do not want to.

> On the other hand, if You want to share Results from Data Received under the Sharing version of the Agreement, then You may include them with the Data You Publish and the Results will be considered Data, just like any other Data that is Published under the Agreement. Or, You may Publish them separately, under an agreement of Your choosing.

若公開的是 Data／Enhanced Data，§3.1 的條件仍完整適用：

> 3.1 If You Publish Data You Receive or Enhanced Data: (a) The Data (including the Enhanced Data) must be Published under this Agreement in accordance with this Section 3; and (b) You must cause any Data files containing Enhanced Data to carry prominent notices that You have changed those files; and (c) If You Publish Data You Receive, You must preserve all credit or attribution to the Data Provider(s). Such retained credit or attribution includes any of the following to the extent they exist in Data as You have Received it: legal notices or metadata; identification of the Data Provider(s); or hyperlinks to Data to the extent it is practical to do so.

§3.3 要求保留原協議並禁止額外用途限制：

> 3.3 If You Publish Data You Receive, You must do so under an unmodified form of this Agreement and include the text of this Agreement, the name of this Agreement and/or a hyperlink or other method reasonably likely to provide a copy of the text of this Agreement. You may not modify this Agreement or impose any further restrictions on the exercise of the rights granted under this Agreement, including by adding any restriction on commercial or non-commercial Use of Data (including Your Enhanced Data) or by limiting permitted Use of such Data to any particular platform, technology or field of endeavor. Notices that purport to modify this Agreement shall be of no effect.

注意版本：FAQ 頁上半段的 “If you train an ML model on it, that would typically be considered a Result” 及放寬原文輸出的說明屬於 **CDLA-Permissive-2.0**。本文件沒有把 2.0 的較寬定義套到 TinyStories 所用的 Sharing 1.0。

## FSDD：CC BY-SA 的觸發條件與權重邊界

[CC BY-SA 4.0 官方法律文本](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) §1(a) 將 Adapted Material 系於需要權利人許可的改作：

> Adapted Material means material subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor. For purposes of this Public License, where the Licensed Material is a musical work, performance, or sound recording, Adapted Material is always produced where the Licensed Material is synched in timed relation with a moving image.

§3(b) ShareAlike 的完整條件：

> In addition to the conditions in Section 3(a), if You Share Adapted Material You produce, the following conditions also apply. The Adapter’s License You apply must be a Creative Commons license with the same License Elements, this version or later, or a BY-SA Compatible License. You must include the text of, or the URI or hyperlink to, the Adapter's License You apply. You may satisfy this condition in any reasonable manner based on the medium, means, and context in which You Share Adapted Material. You may not offer or impose any additional or different terms or conditions on, or apply any Effective Technological Measures to, Adapted Material that restrict exercise of the rights granted under the Adapter's License You apply.

「使用資料訓練」與「公開分享該資料的 Adapted Material」不是許可中的同一觸發詞。是否某個權重檔本身構成原錄音的改編，不能只由訓練來源名稱決定；4.0 原文沒有 CDLA §3.5 那樣的 Results 豁免，也沒有將一切學習所得權重自動定義為改編。

[CC 官方 FAQ：When is my use considered an adaptation?](https://creativecommons.org/faq/#when-is-my-use-considered-an-adaptation)：

> Whether a modification of licensed material is considered an adaptation for the purpose of CC licenses depends primarily on the applicable copyright law. Copyright law reserves to an original creator the right to create adaptations of the original work. CC licenses that allow for adaptations to be shared—all except BY-ND and BY-NC-ND—grant permission to others to create and redistribute adaptations when doing so would otherwise constitute a violation of applicable copyright law.

[CC 官方 FAQ：AI training](https://creativecommons.org/faq/#what-are-the-limits-on-how-cc-licensed-works-can-be-used-in-the-development-of-new-technologies-such-as-training-of-artificial-intelligence-software) 同樣採條件式，而非另設一條模型專用許可：

> If someone uses a CC-licensed work with any new or developing technology, and if copyright permission is required, then the CC license allows that use without the need to seek permission from the copyright owner so long as the license conditions are respected. This is one of the enduring qualities of our licenses — they have been carefully designed to work with all new technologies where copyright comes into play. No special or explicit permission regarding new technologies from a copyright perspective is required.

因此本次採「FSDD 權重也以 CC-BY-SA-4.0 發布」即可給出具體、支持商業使用的公開方案，程式碼仍為 MIT。若改選 MIT 權重，需要針對實際權重／附件是否包含或改編受保護錄音作出具體判斷；不能引用本文件宣稱 CC BY-SA 絕不及於權重。

公開 FSDD 錄音、聲譜圖等受許可涵蓋的資料時，至少遵守 §3(a)(1) 的署名與改動標示：

> If You Share the Licensed Material (including in modified form), You must: retain the following if it is supplied by the Licensor with the Licensed Material: identification of the creator(s) of the Licensed Material and any others designated to receive attribution, in any reasonable manner requested by the Licensor (including by pseudonym if designated); a copyright notice; a notice that refers to this Public License; a notice that refers to the disclaimer of warranties; a URI or hyperlink to the Licensed Material to the extent reasonably practicable; indicate if You modified the Licensed Material and retain an indication of any previous modifications; and indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License.

## MIT 與 PKU：保留原 notice，區分資料與程式碼

Chinese-poetry 的 MIT 全文明示 `Copyright (c) 2016 JackeyGao`。原 UltraChat 與 UltraFeedback 的官方 LICENSE 均明示 `Copyright (c) 2023 THUNLP`；H4 固定資料卡另外明示 `license: mit`。這些是不同來源的證據，不能僅用本課程程式碼的 `Copyright (c) 2026 birdhackor` 替換。

三份上游 MIT 許可的核心授權／notice 段落相同（以下逐字引 Chinese-poetry）：

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

公開包保留各自的完整 MIT 文本（含其原 copyright 和 disclaimer），模型卡提供來源與修改說明。H4 資料卡未提供獨立 LICENSE 檔這一事實，也不應被寫成已取得 H4 對所有來源權利的新增保證。

PKU 資料卡宣告 CC-BY-NC-4.0。[官方法律文本](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) §1(i) 的 NonCommercial 定義：

> NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation. For purposes of this Public License, the exchange of the Licensed Material for other material subject to Copyright and Similar Rights by digital file-sharing or similar means is NonCommercial provided there is no payment of monetary compensation in connection with the exchange.

§2(a)(1) 的授權段落：

> Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to: reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and produce, reproduce, and Share Adapted Material for NonCommercial purposes only.

本次權重發布 manifest 將本輪 PKU 訓練分支的選取／截短／切分樣本及 checkpoint、adapter、merged、distilled 標記 `pku_derived: true`、`visibility: private`，排除公開權重下載清單。這落實既定政策，不把資料 NC 條款擴寫成無來源支持的「所有訓練模型必須私有」一般規則。

既有的 [PKU 固定來源 archive](../../assets/training/pku-safe-rlhf-v1.tar.gz)（上游訓練 split 前 100 筆）已由 Git LFS 提供，隨包保留 `pku-safe-rlhf-LICENSE-AND-SOURCE.md` 的 CC-BY-NC-4.0 聲明、固定來源版本與 `source-README.md` 的上游署名／引用；它繼續遵循原非商用條件，不屬於 MIT 公開模型權重包，也不因上述本輪分支政策而變成私有。

較早 safety Actions run `37046840520` 的診斷 log／artifact 包含 validation、test 各 6 筆來源截短與模型生成。這 12 組來源問答與上述公開 archive 的 120-byte 前綴一致；它們沿用原資料的 CC-BY-NC-4.0 條件。這些診斷副本仍公開，因此不能宣稱歷史上所有逐筆內容一直私有。後續 runner 在完整私有備份完成後，先移除 `private_only` 支線的逐筆樣本，再寫入公開 Actions 報告、log 與 summary；聚合指標與來源資訊保留。PKU 衍生權重始終不在公開學生模型清單。

[tinystories-card]: https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md
[poetry-license]: https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE
[poetry-readme]: https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/README.md
[ultrachat-card]: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/blob/8049631c405ae6576f93f445c6b8166f76f5505a/README.md
[ultrafeedback-card]: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/blob/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md
[pku-card]: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/blob/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md
[fsdd-card]: https://github.com/Jakobovski/free-spoken-digit-dataset/blob/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md#license
[ultrachat-license]: https://raw.githubusercontent.com/thunlp/UltraChat/1f613e1b8dfc6d1e3a02efb6905de608ed06645b/LICENSE
[ultrafeedback-license]: https://raw.githubusercontent.com/OpenBMB/UltraFeedback/bf80fd46a8c6ceecc86e8babb1ae8771f26a3cbb/LICENSE

## Fashion-MNIST 與 GSM8K：固定原始 MIT notice

2026-10-02 重新取得上述兩個固定 commit 的官方 LICENSE；內容 SHA 與本地來源 manifest／原檔一致。兩份固定官方 HF 資料卡的 YAML 均列出 `license:\n- mit`；GSM8K 資料卡另明示：

> The GSM8K dataset is licensed under the [MIT License](https://opensource.org/licenses/MIT).

公開模型卡保留下列全文，並以實際模型分支填寫來源／改動。程式碼的 birdhackor MIT notice 與資料原 copyright 分開。Fashion/FSDD 混合發布目錄按檔案列出 `fashion-mnist.pt: MIT`、`fsdd.pt: CC-BY-SA-4.0`；若 `model.pt` 是 FSDD 檔案別名，該檔也使用 CC-BY-SA-4.0。非 FSDD 的 synthetic audio 不因此改成 CC BY-SA。此映射須於完成報告中確認所選檔案與 hash，不能只看目錄名稱。

Fashion-MNIST 原全文（Zalando SE）：

```text
The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

GSM8K 原全文（OpenAI）：

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

[fashion-license]: https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE
[fashion-card]: https://huggingface.co/datasets/zalando-datasets/fashion_mnist/blob/531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2/README.md
[gsm-license]: https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE
[gsm-card]: https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md
