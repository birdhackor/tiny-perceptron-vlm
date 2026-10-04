# 最終公開入口作者凍結交接

這是作者整合與來源／語法核對，不是 fresh reader、正確性或連貫性評阅，不更新正式review registry或閱讀時間。

20.2与STUDENT固定實際試用程式checkout 59a1eda4ed7b6e8609892ec2b9013c821ac93e69，另開tiny-perceptron-natural資料夾保留原工作。學生路線只採已選定BASE＋Whisper-large-v3-turbo；先拿兩份配置文件，再於啟動／第一次辨識取得官方模型。20.2避免在20.3之前使用未定義的LoRA詞。

STUDENT第2節是唯一CPU資源總表：兩份公開文件9,262B、23個官方快照檔5,889,111,977B、服務RSS 13,496,104KiB，及24.24／170.33／四LM／一次ASR的實測計時。區分已有快取與新匿名文件下載、伺服器与瀏覽器、總工作時間与回答延遲；沒有聲稱最低RAM、整環境磁碟、fresh模型下載、GPU四步或其他平台已驗收。

20.13的九列正式能力表、分母、CER、動作與關係子集、唯一依前文題、七份截斷及原Python小例逐字保留。本次只加入真實公開清單与CPU/browser操作已接通的交付說明，資源指向STUDENT，不產生第二張能力表或新品質分數。

DATA開頭明寫本版按驗證保留底座；TRAINING仍教真實0.00003 LoRA候選配方，仅去完成的作者註記並將『披露樣本』改為『底座幾個固定抽查位置』。三份guides和20章沒有作者待填區。

必要入口分清20.9–20.12的概念判尺与20.13正式實測。R.2/R.3、T.1與README/curriculum连接實際配置／資料／操作。沒有重寫其餘已凍結教材、原始證據、選版、gold、manifest、scorer或流水線。

## 作者實際核對

- 59 Python fenced片段AST、80 Bash片段syntax、22組實際argparse；不dispatch任何訓練、測試、模型載入或推論。
- 659本地連結与數字anchor、21 SVG XML；原公開清單native --list成功且顯示base，没有下載或載入模型。
- 完整source-index中的51個raw與34個checkout-copy逐檔bytes/SHA一致。實際UI receipt、public-release與官方cache紀錄讀取並數學重算配套23檔總量，源碼符合清單code_files／requirements指紋。
- source、manifest、selection與正式scores原SHA都保留，20.13能力表逐字與改前相同。
- root在本次工作期間对20intro/.1/.3/.4/.5/.6/.7/.9的独立reader小修保留不還原，依root各凍結SHA核對；changed-owners中特別標為parent simultaneous frozen edit，與本作者變更分開。20.8/.10/.11/.12逐字相同。

## 凍結範圍与hash

全部實際改動及前後hash見changed-owners.json；完整11個文件SHA見files-after.json。curriculum第3節的實際變動只在『第20章』介紹与20.13摘要，不擴改其他章大綱。

- `course/chapters/20.md#20.2`：`c61d8eb0b3258e73d857000e3db0ee1b5761f54e783ed55e62d3ac6f5d6ca2d6`
- `course/chapters/20.md#20.13`：`c22e1db4cea69fedb1a6fa23e540945b9044363df84a57394948fc46ba84d8cf`
- `docs/natural-assistant/v4/STUDENT.md#intro`：`8dc991d5eb058191113650970b24fb5ed3b6bb7db446ce825a71b5ae98d6a855`
- `docs/natural-assistant/v4/STUDENT.md#1.`：`3dc6601395f9b2c0139994813694370a7efb2e82cf20aaf76ae612b5b2a37c4a`
- `docs/natural-assistant/v4/STUDENT.md#2.`：`81f5f3cbff0dccaccac1ce7e96a7123e5e59a0f019a458ed2111995380bf9586`
- `docs/natural-assistant/v4/STUDENT.md#3.`：`8268420d9f7ddf644533314d7faa4a197e08df8641549726f2ef0b903e599c10`
- `docs/natural-assistant/v4/STUDENT.md#4.`：`888f72e7896ace4409f964e27d8f738028e3f672201ccaf6bc26011a33aab906`
- `docs/natural-assistant/v4/DATA.md#intro`：`4e92da5a8e92e355be2c9a9a31395079156f8e69351a4cc20dd61e976b4816b4`
- `docs/natural-assistant/v4/TRAINING.md#intro`：`ebbe8851bd906c5a25d64468780e9ed630aa96d5272c703370a28f2fcba3a36c`
- `docs/natural-assistant/v4/TRAINING.md#5.`：`cc3f6d4d5307af91075722da82a223699eb8df55d3c115ef00ac534c4bf13649`
- `course/chapters/11.md#11.14`：`8a8d6b6441b5e66b01424cbd3b8eeee7d6191ebba47b41afb3a187e7948ce12e`
- `course/chapters/11.md#11.16`：`4d68151f2cae9fe97e60b31480d592ccfcd6d8d5a6a034e41c57837c4dcddc1c`
- `course/chapters/12.md#12.13`：`40e6e2a741e86f3a115b221a20c5cc83c8a1b9d13a6934b3440beda985dea89f`
- `course/chapters/12.md#12.14`：`071af393d99e4f0ce229025bec02b13046b4ddcfdf116ef96d17771f042fb815`
- `course/chapters/19.md#19.12`：`4ae52e4bed3132e793291a91c07daf89266419456f900306e57e91923e904b02`
- `course/README.md#R.2`：`bbccf491cc9e2f421225eab61a6805e725823f27354f5e6a6185aba3813854a1`
- `course/README.md#R.3`：`93c36dab213ff873a98f92a1e73d37d81dbc2be6bf1ac62970398639344bbc66`
- `course/training.md#T.1`：`0ef8c0f2aed055791fd0f270fefdc262e44c600f4a6d9bd3800ad4529fee72cd`
- `README.md#教材提供什麼`：`9bec400ffef2bd1a484410546a324210418a5702f9f1484544d9652bc6007f76`
- `docs/curriculum.md#intro`：`37cb3c011cd520e7f0b4694ea25e9fceb5828afe0cf195d963a9e4cfd9a08135`
- `docs/curriculum.md#3.`：`fdfcc8c287abf5c7d029410b17b2e4433926a423cae689ae0273cca4b074c446`

三份指引完整頁SHA：

- `docs/natural-assistant/v4/STUDENT.md`：`fbcda65f9e7ed6f36f97d137cd91e7940a09cf38b94eb4e38ae1de6845847d89`
- `docs/natural-assistant/v4/DATA.md`：`2557ea143b9b83a5beb6e78284f81b61232d011c42ffbba142b97c97b1a9213e`
- `docs/natural-assistant/v4/TRAINING.md`：`7507af85222ebe222fe26f68a71184ac4393bad5f2b83f34e9a15cd4b8413412`

後續由root/coordinator按這些最終bytes派fresh三階審阅與閱讀時間。沒有pending實驗數字，沒有Git操作或commit。
