# 開發分支與歷史版本

目前開發與新安裝統一使用 `main`：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main https://github.com/birdhackor/tiny-perceptron-vlm.git
```

2026-10-07 將下列工作分支封存為同名 annotated tag。Tag 固定在刪除分支前的原始提交，不會跟著 `main` 移動。

| Tag | 原始提交 | 保存內容 |
| --- | --- | --- |
| `natural-assistant-v4` | `0bde505105fe3dac49ba9488eff297a8fe0f3625` | 舊版助手開發與第一輪自行訓練的十二階段紀錄、驗證結果及接續訓練設定 |
| `natural-v4-data-bootstrap` | `ed0a1ed018c9f71fc2bf3018563bb5fbb336fd5f` | 舊版公開資料重建、核對與 Git LFS 上傳工具 |
| `selftrained-v2` | `5aa019d50a940deae8f8f1cf84bd9b11fed47121` | 已發布 V2 成品的程式、固定資料清單、操作指南與能力紀錄 |

現行本機訓練指引、CPU 操作指南與模型卡原稿改指向 `main`。教材與 notebook 中重現既有結果的命令、已發布模型卡及歷史審閱紀錄保留固定版本引用；GitHub 的同名 tag 讓既有 `blob/selftrained-v2/…` 連結仍可開啟，原始紀錄也不必改寫。

若要重現封存版本，Git 同樣接受 `git clone --branch selftrained-v2`。取得 tag 時會進入 detached HEAD，表示正在查看固定提交；若要以此為基礎修改程式，另建自己的分支，例如 `git switch -c my-experiment`。

這次整理只改 Git 引用與安裝指引，沒有重新訓練、改動權重或訓練資料。
