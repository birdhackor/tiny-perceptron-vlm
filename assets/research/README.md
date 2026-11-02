# 固定研究來源與標註

只讀教材或執行各節短程式，不需要這裡的完整研究明細。重建第19／20章的自然資料，或跑需要真實研究輸入的維護測試時，再按需取得：

```bash
python scripts/research_assets.py fetch --group natural-v4
python scripts/research_assets.py fetch --group selftrained-sources
```

也可用 `--group all` 一次準備兩組。`natural-v4` 包含原始 manifest、來源與標註；`selftrained-sources` 包含來源紀錄與兩份重建配方。來源固定在 Git 提交 `92b8763ca24511f0da1c259429851a314accea19`，下載位置、每檔大小與 SHA-256 見 [catalog.json](catalog.json)。來源的授權與 NOTICE 隨包保存；這些資料不會因此改成專案的 MIT 授權。

若 catalog 尚無下載網址，代表該組仍使用專案附的原始明細；此時 `fetch` 只核對本機。公開驗證成功後才會加入下載網址，再改為按需取得。

工具核對壓縮包及每個檔案後，放回原本的 `docs/` 路徑，讓既有重建程式能讀取。已核對的本機檔案會直接重用；有不同內容的檔案會報錯，不覆寫你的修改。壓縮包快取在 `outputs/research-assets/`。資料與權重是兩種不同的下載，這裡不含模型或完整照片／音訊訓練包。

下載完成後可離線核對：

```bash
python scripts/research_assets.py verify --group all
```

pytest 不會替你下載資料。需要完整資料的測試先準備這兩包，再離線執行；一般教材網站的 CPU Notebook 不需要這兩包。固定實驗的環境與授權仍依其原始程式版本核對，不能把新環境當作舊核准。
