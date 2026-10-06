## 第19章v2資料：自行訓練的有限多模態主線

`selftrained-v2.tar.gz` 是固定Git LFS快照，大小67,862,431 bytes，解包內容127,161,811 bytes；包含12個JSONL與8,950個其他檔案（包含圖片、錄音與來源說明），共8,962個檔案。訓練／驗證／最後測試題數為28,876／2,435／3,734，題數不等於獨立圖片或錄音數。[v2-manifest.json](../../docs/selftrained/v2-manifest.json)指定Git commit `08761dac87a6ef360db95883d9bcd338c44fe76d`、包SHA-256與逐檔指紋。

文字與工具題由本課編寫；圖片取Fashion-MNIST三類服飾，另組成公開兩格位置圖。OCR使用12個已知繁體字與一個使用者指定的連續1–4字區域；v2訓練已包含Sans與Serif兩種字型。MInDS-14中文錄音限地址、App與卡片三種銀行客服主題；訓練／驗證／測試各62／15／30段錄音，沒有已知speaker ID，不能宣稱說話者隔離。文字與服飾來源採MIT，錄音採CC BY 4.0，字型採SIL OFL 1.1；各來源與原授權保留在包內，不能把全部素材改稱MIT。

只想試成品，按[公開CPU操作](../../docs/selftrained/v2-public-cpu-commands.md)取得示範素材與匿名推論權重；要從隨機初始化重做，用[本機訓練指引](../../docs/selftrained/TRAINING.md)核對整包資料，逐段接自己的checkpoint，不需要作者私人Volume。兩版固定最後測試與未達標項目見[19.12](../../course/chapters/19.md#19.12)和[完整結果](../../docs/selftrained/results/v2-final-public-results.json)。

## 第20章資料：照片、中文文件與真人語音

目前成品的來源、授權、固定下載與逐檔核對見[新版資料說明](../../docs/natural-assistant/v4/DATA.md)，重新訓練見[訓練指引](../../docs/natural-assistant/v4/TRAINING.md)。只想開啟模型時，直接用[學生操作指引](../../docs/natural-assistant/v4/STUDENT.md)，不用先取得全部訓練資料。

需要回查舊版v3資料時，可查看[舊整合清單](../../docs/natural-assistant/manifest.json)；目前第20章使用上方資料說明中的固定清單與下載步驟。
