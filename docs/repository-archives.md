# 開發分支與歷史版本

新安裝使用 `main`；只學習或執行最新版教材時，可省略完整歷史及可選資料包：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-tags --branch main https://github.com/birdhackor/tiny-perceptron-vlm.git
```

需要資料包時按 [資料下載指引](../assets/training/README.md)取得指定包。若要核對教材釘住的歷史版本，另執行 `git fetch --unshallow --tags`；一般完整 clone 仍會下載舊歷史。

## 2026-11-02 精簡工作樹與發布流程

教材、Notebook、圖解、實作、重現設定、來源、授權和重要實驗結果保留在 `main`。歷史審閱快照、重複 manifest、讀者 session 與截圖不再放在目前工作樹；發布也不再執行舊審閱引擎。

移出的檔案可在[精簡前的完整公開版本](https://github.com/birdhackor/tiny-perceptron-vlm/tree/83630ea1cea23d47c4b77162a31cbb7566686d11)查閱。這個提交仍是 `main` 的祖先，本輪沒有改寫 Git 歷史或刪除舊提交。教材對移出資料的連結改指固定版本；留在目前工作樹的科學結果與來源內容不變。

日後詳細審閱過程使用短期 artifact，PR 留易讀性、正確性、連貫性的簡短結論。長期支持實驗主張的資料另按重現需要保存；權重與完整訓練輸出使用 Release 或模型儲存庫。見 [發布流程](publishing.md)與[編輯準則](editorial-guide.md)。

這次減少的是最新版工作樹與淺層下載量。完整 clone 仍包含舊證據的歷史；若未來需要縮小它，要另行規劃備份與固定版本連結，不能直接 force push。

## 較早的封存紀錄

以下保留先前的整理紀錄；其中「生效中的審閱輪次必須留在 main」等舊發布規則已由上述流程取代。

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

## 2026-10-21 封存 2026-10-17 的審閱輪

2026-10-19 這一輪（`docs/course-review-20261019/`）在 `eaee5ed4` 生效、推到 `main` 後發布檢查通過，於是從 `main` 移除上一輪 `docs/course-repair-20261017/` 的 681 個檔（10.5 MiB）。留在原路徑的 43 個檔，是下列來源以路徑引用的：課文 19.6 與它的 Notebook（`artifacts/primary-audio-head-values-01.json`）、生效輪次的紀錄與技術審閱 artifact（上一輪的部分技術報告與凍結快照、`manifest-final-01.json`、`all-gate-final-01.json`、`active-pointer-before-repair-01.json` 等）、`docs/course-revision-20261016/reader-sessions/` 裡上一輪的讀者 session（它們指向的閱讀計畫與凍結檔；`docs/course-revision-20261016/` 因為三道檢查的重放要整棵留著）、交接文件與一個測試。

移除的檔仍在 Git 歷史裡，完整的目錄最後出現在 `eac81bbc`，例如 `git show eac81bbc:docs/course-repair-20261017/REPAIR-NOTE.md`；這個目錄裡其他紀錄引用的路徑，現在只在那個提交以前有效。移除前後，三道發布審閱檢查的輸出逐位元組相同，全套測試的結果也相同。留下的檔內容不變。
