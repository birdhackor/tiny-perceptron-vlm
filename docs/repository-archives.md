# 開發分支與歷史版本

目前開發與新安裝統一使用 `main`：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main https://github.com/birdhackor/tiny-perceptron-vlm.git
```

## 封存的工作分支

2026-10-07 將下列工作分支封存為同名 annotated tag。2026-10-11 精簡歷史後，前兩個 tag 只保存在私有封存 repo；`selftrained-v2` 仍在公開 repo，改指向重寫後對應的提交。

| Tag | 原始提交 | 目前位置 | 保存內容 |
| --- | --- | --- | --- |
| `natural-assistant-v4` | `0bde505105fe3dac49ba9488eff297a8fe0f3625` | 私有封存 repo | 舊版助手開發與第一輪自行訓練的十二階段紀錄、驗證結果及接續訓練設定 |
| `natural-v4-data-bootstrap` | `ed0a1ed018c9f71fc2bf3018563bb5fbb336fd5f` | 私有封存 repo | 舊版公開資料重建、核對與 Git LFS 上傳工具 |
| `selftrained-v2` | `5aa019d50a940deae8f8f1cf84bd9b11fed47121` | 公開 repo，提交 `1b5fb93c2c8c041181ceb09134923afc03cfedfb` | 已發布 V2 成品的程式、固定資料清單、操作指南與能力紀錄 |

現行本機訓練指引、CPU 操作指南與模型卡原稿改指向 `main`。教材與 notebook 中重現既有結果的命令、已發布模型卡及歷史審閱紀錄保留固定版本引用。`blob/selftrained-v2/…` 連結仍可開啟，但不含下文移除的目錄；以原始提交 hash 組成的連結不再保證能在公開 repo 開啟。

若要重現封存版本，Git 同樣接受 `git clone --branch selftrained-v2`。取得 tag 時會進入 detached HEAD，表示正在查看固定提交；若要以此為基礎修改程式，另建自己的分支，例如 `git switch -c my-experiment`。

2026-10-07 的整理只改 Git 引用與安裝指引，沒有重新訓練、改動權重或訓練資料。

## 2026-10-11 精簡歷史

公開 repo 的 `main` 重寫了歷史，從每個提交移除已被後續審閱輪次取代的紀錄：

- `docs/technical-reviews/` 與 `docs/reader-reviews/`：頂層報告檔保留，其餘移除
- `docs/continuity-reviews/`、`docs/reading-time-reviews/`、`docs/validation-artifacts/`
- `docs/course-revision-20261005/`、`docs/course-revision-20261006-phase6/`、`docs/course-revision-20261007-phase7/`
- `docs/natural-assistant/reviews/`

課文、notebook、README、CI 與審閱準則仍引用的 50 個檔案留在原路徑。生效中的審閱輪次（`docs/course-publish-20261008/`、`docs/course-repair-20261008/`）與 `docs/natural-assistant/evidence/` 沒有移除。保留檔案的內容逐位元組不變，發布前的審閱檢查輸出與重寫前相同；Hugging Face 上的權重與 LFS 資料包不受影響。

重寫前的完整歷史，包括 `main` 的 565 個提交與上表 3 個 tag，原樣保存在私有 repo `birdhackor/tiny-perceptron-vlm-archive`，只有維護者能讀取。公開 repo 的分支與 tag 已不包含這些提交；紀錄與文件中引用的重寫前提交 hash，要到封存 repo 查詢，例如 `git clone --bare git@github.com:birdhackor/tiny-perceptron-vlm-archive.git` 後用 `git show <提交>:<路徑>`。封存 repo 只有 LFS pointer，資料包實體仍在公開 repo 的 LFS 儲存空間。

## 審閱輪次的保存規則

`docs/course-reviews-active.json` 指向生效中的審閱輪次。發布流程每次 push 到 `main` 都核對這一輪，下一輪沿用舊審閱的頁面也從這裡取得證據，所以生效中的輪次必須留在 `main`。

新一輪生效且發布檢查通過後，封存上一輪：

1. 從 `main` 移除上一輪的目錄；課文、notebook、README、CI、審閱準則或生效輪次仍引用的檔案留在原路徑。移除的內容仍保留在 Git 歷史裡。
2. 重跑發布前的審閱檢查與全套測試，通過後再提交。
3. 只有移除的內容在歷史裡占用太多空間時，才依 [AGENTS.md](../AGENTS.md#改寫-git-歷史) 的步驟重寫歷史。推到封存 repo 時使用新的 ref 名稱，不要覆寫封存 repo 已有的 ref。
