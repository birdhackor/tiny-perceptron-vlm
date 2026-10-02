# 教材資產存放：Git、LFS 與外部下載

更新日期：2026-10-02。repo 已啟用 `/assets/training/*.tar.gz` 的 LFS 追蹤，首批8個分來源資料快照與普通 Git 的來源／授權／雜湊清單在[訓練資料入口](../assets/training/README.md)。原始 `data/` 快取、權重與執行輸出維持忽略。

8個資料物件已上傳，共30,854,837 bytes；新的LFS快取重新下載與逐檔解包驗證均通過。固定快照與來源／授權清單隨main發布；讀者可依單元按需取得，不需要提供額外的上傳認證。

## 已驗證的能力

- 本環境已有 `git-lfs/3.6.1`，執行檔為 `/usr/bin/git-lfs`。
- 在 `/tmp` 的獨立測試 repo 執行 local install／track／add：4096 bytes 的 binary 在 Git index 中成為 129 bytes 的 LFS pointer；smudge 還原後逐 byte 相同，SHA-256 一致。
- 原 repo 的 `git ls-remote origin HEAD` 成功。
- 使用原 repo 的 GitHub remote 與原生 LFS client，只讀查詢一個刻意不存在的 object；服務回應 `Object does not exist on the server: [404]`。這證明 LFS 物件查詢端點可通，沒有本輪先前其他網站的 CONNECT 403。
- 前四項是首次調查的結果，當時未上傳物件。不把repo擁有者的方案或剩餘額度當成已知條件。
- 2026-10-02已確認8個包在Git中都是LFS pointer，大小與SHA-256符合manifest。GitHub runner從固定公開來源重建所有包，逐檔與壓縮包雜湊皆符合原快照，LFS上傳8/8成功；[執行紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/36946663981)。
- 本環境以全新LFS storage、不使用額外secret或credential helper重新下載全部8個物件，再解包核對全部檔案；結果均通過。報告在ignored的`outputs/lfs-remote-validation.json`。
- 雲端原生上傳仍有傳輸限制：提供有效認證後，LFS batch API回應200，但代理轉送到S3時得到501，指出不支援`Transfer-Encoding`；Release附件上傳同樣回報400 `Bad Content-Length`。本輪由GitHub runner完成物件上傳，普通Git推送與資料下載可用。

初次調查時 `.gitattributes` 只有換行設定。現在固定資料包使用 LFS pointer；來源、授權、manifest 與生成器用普通 Git。`.gitignore` 仍忽略 `/data/`、`/checkpoints/`、實驗輸出及 `*.pt`／`*.safetensors` 等權重，沒有把整個下載快取目錄改為追蹤。

## 建議分工

| 內容 | 存放方式 | 理由 |
| --- | --- | --- |
| Python、Markdown、Notebook、配置、tokenizer、資產 manifest | 普通 Git | 易 diff、review，能與程式一起版本化；大型 tokenizer 另依實際大小選擇。 |
| 少量微小圖像／音訊 fixture | 普通 Git | 離線 smoke 與第一個例子可立即執行；為整批檔案設小預算，不因 binary 就一律放 LFS。 |
| 可重現的合成圖片／音訊／小世界資料 | Git 保存生成器、seed、配置；本機生成 | 避免每個資料版本都存一份，且方便改一個變數做教學。 |
| 課程專用、固定版本的圖像／音訊／混合訓練資料包 | Git LFS，按單元發布 | 讓教學與訓練實驗使用相同資料快照；生成器、seed、split 與 manifest 仍留普通 Git。 |
| 少量必須隨原始碼版本維護的中型 binary | 選擇性 Git LFS | Pointer 跟 commit 一起走，使用方便；限制追蹤目錄及發布版本數量。 |
| 完整資料集、教材 checkpoint、教師／學生權重 | Hugging Face model／dataset repo；以固定 revision 下載 | 按單元取檔，避免 clone 教材時下載所有權重；現有 `huggingface-hub`／`datasets` 可使用。 |
| 已定版的課程資產包 | GitHub Releases 可作替代或備援 | 不增加普通 Git 的 binary 歷史；固定版本、檔名與 checksum，程式自行下載。 |
| 每次訓練 checkpoint、logs、曲線中間檔 | 本機忽略目錄 | 只發布挑選過的教材成果。 |

Notebook 保留教學需要的少量結果；大型 base64 圖像、音訊、互動狀態或完整 logs 改放獨立資產。大量小 binary 可以依單元打包或分 shard，避免把每個樣本都做成獨立遠端物件。

## 教學訓練資料可能多大？

目前已收集的小型 pilot 原始資料約35.3 MiB，共1,447筆整理好的 training 紀錄；尚未覆蓋全部教材的自然資料需求。首批快照保留來源與授權，按資料來源分包，精確壓縮 bytes 見 `assets/training/manifest.json`。主線合成資料由生成器重建，不需要為每節各自複製一份。

下列是明確假設下的 payload 計算，非實測檔案大小：

| 示範規模 | 大小 |
| --- | --- |
| 10,000 筆文字，每筆 1 KiB | 約 9.8 MiB。 |
| 10,000 張 64×64 RGB、uint8 圖片 | 原始像素約 117 MiB；簡單合成圖以 PNG 儲存通常較小，需實測。 |
| 10,000 段 1 秒、16 kHz、單聲道 PCM16 音訊 | 樣本約 305 MiB，另加容器／標註成本。 |

首版核心資料可先以數十至數百 MiB 作發布規劃目標，確認教學效果後再定案。小型文字與評估樣例仍適合普通 Git；固定較大的多模態資料包適合 LFS。由於學生下載計入擁有者流量，300 MiB 資料包下載 100 次約為 29.3 GiB，已超過 GitHub Free／Pro 的 10 GiB 月流量；人數增加時可另發布 HF dataset 副本，或把資料改為按需生成／下載。

自然圖片、完整語音與較大文字 corpus 是獨立延伸，可能達 GB 至數十 GB。原有候選的大小見 [環境研究的候選資料集](environment.md#候選資料集)，那些是概略數字；正式選用時再查精確版本、所需 subset 和授權。大型外部資料保留來源與版本，直接從原發布處取得，不把整份 corpus 搬進 GitHub LFS。

## 權重建議放 Hugging Face

建立專案的 HF **model repo**，發布挑選過的文字 SFT、VLM、音訊整合、MoE、蒸餾與量化里程碑。它可以保存本專案自己寫的 PyTorch 模型，不要求先改成 Transformers 架構；下載後由我們的模型類別與載入程式使用。

每份可用權重附模型配置、匹配的 tokenizer、程式碼 commit、訓練資料版本、評估結果與 model card。純推論權重可用 `safetensors`；需要教「中斷後接續」的小節，另提供包含 optimizer／step／必要隨機狀態的 resume checkpoint。只發布教材所需版本，不把每一步 checkpoint 都當成發布成果。

純 FP32 權重約為參數數量 × 4 bytes：10M 參數約 40 MB，100M 約 400 MB；resume checkpoint 會更大。這些是估算，不代表課程已決定模型大小。

GitHub manifest 固定 HF commit revision、檔名與 SHA-256；讀者按單元下載所需權重。使用現有 `huggingface-hub` 的 API／`hf` CLI 處理大檔，無須把模型掛在本專案的 GitHub LFS 內。

HF 官方目前將免費公開儲存列為 best-effort，並要求大型內容具有社群用途；不是無限免費備份承諾。教材公開里程碑符合其 model／dataset 分享方向，實際容量仍按帳戶方案與最新政策確認。本輪未建立 HF repo 或上傳檔案。

## Git LFS 的成本與界線

檔案數量本身不是唯一判準：總 bytes、修改頻率、需要保留多少版本、每位讀者下載多少，都會影響選擇。

LFS 在 Git 裡存 pointer，實體檔案另外存放；不會讓檔案本身變小。新增 binary 版本會計入完整新物件的儲存量，讀者和 CI 的下載會計入 repo 擁有者的下載流量。公開 repo 也適用，因此教材廣泛被使用後應按需下載，而不是 clone 時拿完所有模型。

截至本次查閱，GitHub 官方資料列出普通 Git 超過 50 MiB 警告、超過 100 MiB 阻擋；Free／Pro 的 LFS 包含 10 GiB 儲存與每月 10 GiB 下載流量，Team／Enterprise 為 250 GiB。單一 LFS 檔案上限依方案為 2／4／5 GB。這些是官方方案值，不代表已確認本 repo 擁有者的方案、剩餘額度或付費設定；使用前以帳戶頁面及最新文件為準。

GitHub Releases 的附件獨立於普通 Git 歷史，官方文件目前表示不限制總附件大小與下載頻寬，單檔仍受方案上限限制。適合已發布的資產，頻繁訓練中間檔不必逐版上傳。

## 可重現與按需取得

建議 Git 中的資產清單至少記錄：

- 資產 ID、對應單元、用途與依賴。
- 檔案大小、來源 repo／URL、不可變 revision、SHA-256 與授權。
- 資料生成器／seed／split，以及模型架構、tokenizer、配置與訓練入口。
- 能力檢查、預期輸出、離線替代方式及本機快取路徑。

下載入口按單元選取檔案，驗證 checksum 後存入目前已忽略的資料／權重目錄。HF 使用固定 commit revision，避免以 `main` 代表教材版本；Release 附件用 checksum 保證取得的是教材指定內容。

目前 `.gitattributes` 僅追蹤 `/assets/training/*.tar.gz`；LFS 追蹤不會覆蓋 `.gitignore`，`data/` 仍是本機快取。可用 `GIT_LFS_SKIP_SMUDGE=1` 先 clone 程式，再由各單元按需取 LFS 物件；不能把僅有 pointer 的狀態當作資產已下載。

發布需要能供Git LFS使用、對本repo有寫入權限的GitHub HTTPS認證與可用的實體檔案傳輸路徑；普通Git或API認證成功不保證LFS上傳可用。先上傳物件，再推送commit，最後以新的LFS cache下載核對SHA-256。禁止略過LFS pre-push hook而發布無法下載的pointer。認證只能透過安全的環境設定輸入，不放進repo、指令參數或對話。

此雲端實例的上傳代理有上述傳輸限制；新增資料版本可由正常開發機或GitHub runner上傳。這次runner只保存重建程式與凍結的來源／授權metadata於暫存分支，從公開的固定revision取回原資料，核對原始逐檔與壓縮包SHA-256後才寫入LFS；沒有把實體訓練資料塞入普通Git。runner使用本repo的`GITHUB_TOKEN`與`contents: write`，沒有把雲端Secret複製到GitHub Actions。

## 官方來源

- [GitHub：Git LFS 與單檔上限](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [GitHub：LFS 儲存／流量計費](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [GitHub：普通 Git 大檔與 Release 分發](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub 官方文件的限制／額度變數](https://github.com/github/docs/blob/main/data/variables/large_files.yml)
- [Hugging Face：Model Hub](https://huggingface.co/docs/hub/models-the-hub)
- [Hugging Face：儲存政策](https://huggingface.co/docs/hub/storage-limits)
- [Hugging Face：repo 與上傳方式](https://huggingface.co/docs/hub/repositories-getting-started)

本次讀取官方 `github/docs` repo 的原文與額度變數，以及 `huggingface/hub-docs` 的原文；政策可能更新，以上記錄查閱日期。
