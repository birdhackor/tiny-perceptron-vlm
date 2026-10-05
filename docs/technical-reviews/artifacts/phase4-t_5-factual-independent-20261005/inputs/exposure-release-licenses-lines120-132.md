PKU 資料卡宣告 CC-BY-NC-4.0。[官方法律文本](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) §1(i) 的 NonCommercial 定義：

> NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation. For purposes of this Public License, the exchange of the Licensed Material for other material subject to Copyright and Similar Rights by digital file-sharing or similar means is NonCommercial provided there is no payment of monetary compensation in connection with the exchange.

§2(a)(1) 的授權段落：

> Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to: reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and produce, reproduce, and Share Adapted Material for NonCommercial purposes only.

本次權重發布 manifest 將本輪 PKU 訓練分支的選取／截短／切分樣本及 checkpoint、adapter、merged、distilled 標記 `pku_derived: true`、`visibility: private`，排除公開權重下載清單。這落實既定政策，不把資料 NC 條款擴寫成無來源支持的「所有訓練模型必須私有」一般規則。

既有的 [PKU 固定來源 archive](../../assets/training/pku-safe-rlhf-v1.tar.gz)（上游訓練 split 前 100 筆）已由 Git LFS 提供，隨包保留 `pku-safe-rlhf-LICENSE-AND-SOURCE.md` 的 CC-BY-NC-4.0 聲明、固定來源版本與 `source-README.md` 的上游署名／引用；它繼續遵循原非商用條件，不屬於 MIT 公開模型權重包，也不因上述本輪分支政策而變成私有。

較早 safety Actions run `37046840520` 的診斷 log／artifact 包含 validation、test 各 6 筆來源截短與模型生成。這 12 組來源問答與上述公開 archive 的 120-byte 前綴一致；它們沿用原資料的 CC-BY-NC-4.0 條件。這些診斷副本仍公開，因此不能宣稱歷史上所有逐筆內容一直私有。後續 runner 在完整私有備份完成後，先移除 `private_only` 支線的逐筆樣本，再寫入公開 Actions 報告、log 與 summary；聚合指標與來源資訊保留。PKU 衍生權重始終不在公開學生模型清單。
