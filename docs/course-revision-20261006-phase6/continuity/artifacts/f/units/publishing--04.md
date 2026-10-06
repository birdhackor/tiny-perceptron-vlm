## 4. 第19章v2的公開材料

v2資料快照由[固定manifest](selftrained/v2-manifest.json)指向Git LFS包；四組推論輸出固定在HF revision `979cdfacc588ad0536f1c64fff96f264571cf054` 下的 `selftrained/v2/`，共16個檔案。網站改版不會更新這些權重，也不要求按文稿版次重新訓練全書。教材回填的能力與限制須對照[固定最後測試](selftrained/results/v2-final-public-results.json)，公開操作須對照[已實跑CPU命令](selftrained/v2-public-cpu-commands.md)。

模型庫根卡的完整發布稿保存在[repository-README.md](selftrained/model-cards/repository-README.md)。修改這份本機稿不等於已更新HF根README；模型卡文字發布也不改四組safe exports的固定版本與內容。它保留各章歷史權重索引、v2入口，以及第20章Qwen／Whisper的延伸說明。
