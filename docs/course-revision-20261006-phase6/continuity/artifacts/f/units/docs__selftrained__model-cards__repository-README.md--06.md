## 在自己的CPU試用

先安裝Git LFS與[uv](https://docs.astral.sh/uv/getting-started/installation/)，取得含v2程式與操作頁的版本；在repo根目錄執行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch selftrained-v2 https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
git lfs install --local
uv sync --frozen --extra cpu --extra selftrained
git lfs pull --include="assets/training/selftrained-v2.tar.gz" --exclude=""
```

依[公開CPU命令](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-public-cpu-commands.md)先用checksum-bound helper解開示範資料，再執行文字示範：

```bash
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/text.messages.json \
  --task text --device cpu --max-new-tokens 128 --threads 2
```

入口匿名取得固定revision的四個safe files，核對配對manifest與檔案指紋；不需要HF token、作者私人 `.pt`、optimizer或Modal。完整操作頁另有OCR、服飾、位置、真正工具執行、語音回答與保存history的兩輪對話。

這些是挑選的成功validation情境，已用公開權重完成CPU smoke：8次chat呼叫與1次history append，回答和保存的GPU validation示範相同。這證明當時環境中的下載、載入及介面可運作，不是未知問題的成功率，也沒有改變上表的工具／語音失敗。

