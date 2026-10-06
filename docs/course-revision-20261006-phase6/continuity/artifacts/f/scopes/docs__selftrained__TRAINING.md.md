## 本輪實際完成步數與選定步數

下表只描述已實核的本輪 production lineage，供比對 checkpoint selection；新執行不硬編碼這些選定步數。`best.pt` 由 unchanged unweighted teacher-forced validation loss 選定，並不代表 generation success。

| Stage | 實際完成 steps | MoE selected step | Dense selected step |
| --- | ---: | ---: | ---: |
| pretrain | 1000 | 250 | 250 |
| sft | 8000 | 8000 | 6000 |
| vision | 1200 | 1200 | 1200 |
| ocr | 3000 | 3000 | 3000 |
| audio | 1000 | 200 | 200 |
| joint | 6000 | 2000 | 2000 |
| weighted tool4/numeric4 | 10000 | 3000 | 1000 |
| native tool4/numeric1/native4 | 4000 | 1000 | — |

來源見 [12 段 baseline index](v2-training-stage-index.json)、[MoE weighted index](results/v2-moe-weighted-10000-training-stage-index.json)、[Dense weighted index](results/v2-dense-weighted-10000-training-stage-index.json) 和 [MoE native index](results/v2-moe-native-balanced-4000-training-stage-index.json)。本輪 MoE native 確實從 completed10000、selected3000 的 own weighted source 啟動新段，沒有將 latest10000 當作改 objective 的 exact resume。

本機 wrapper 的工程驗證使用 width16 合成 CPU 資料、真 batch16/context512：三段 joint 各一步、genuine failure/interrupt、禁止 incomplete promotion 與 ownlatest exact resume。這只驗證執行契約，不代表重跑了全量 production training 或證明模型能力。[Frozen final results](results/v2-final-public-results.json) 記錄兩個 architecture 各 3,734 筆固定 final test；原判準仍未全數達標，不能將本操作指引或 CPU smoke 當作 pass 宣稱。
