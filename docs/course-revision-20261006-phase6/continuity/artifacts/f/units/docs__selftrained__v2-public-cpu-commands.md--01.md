## 取得依賴、資料和公開權重

先安裝 [Git](https://git-scm.com/downloads)、[Git LFS](https://git-lfs.com/) 與 [uv](https://docs.astral.sh/uv/getting-started/installation/)。第一次使用時，取得包含本頁程式的 `selftrained-v2` 分支，再進入專案根目錄：

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch selftrained-v2 https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
git lfs install --local
```

`GIT_LFS_SKIP_SMUDGE=1` 讓這次 clone 先取得程式與資料指標，下一段才下載需要的資料包。Windows PowerShell 先執行 `$env:GIT_LFS_SKIP_SMUDGE="1"`，再執行不帶前綴的 `git clone`。若已有專案，先確認其中有 `scripts/selftrained/chat.py` 與本頁列出的示範輸入；想重新取得完整配套時，另選新目錄 clone，保留原來的工作。

以下所有命令都從包含 `pyproject.toml` 的專案根目錄執行。`uv sync` 準備 CPU 推論的工具；`git lfs pull` 只下載本頁需要的 V2 資料包。

```sh
uv sync --extra cpu --extra selftrained
git lfs pull --include="assets/training/selftrained-v2.tar.gz" --exclude=""
```

如果還沒解開資料包，使用現有的 checksum-bound unpacker：

```sh
uv run --extra cpu --extra selftrained python - <<'PY'
import json
from pathlib import Path
from scripts.selftrained.hf_transport import unpack_verified_archive

manifest = json.loads(Path("docs/selftrained/v2-manifest.json").read_text())
unpack_verified_archive(
    "assets/training/selftrained-v2.tar.gz",
    "outputs/selftrained-v2/data",
    manifest["package"],
)
PY
```

`chat.py` 會匿名取得四個公開 safe files，不需要私人 `.pt`、optimizer 或 Modal。指定的 HF revision 是 `979cdfacc588ad0536f1c64fff96f264571cf054`；MoE joint 的 inference manifest SHA 是 `f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e`。這份公開權重選自該輪 step 1,000，而該輪訓練實際完成 4,000 steps。

`docs/selftrained/examples/v2` 的 10 個 JSON 只提供 user 訊息、先前格式確認與公開 ROI／位置資料。它們不包含目前的標準答案、評分標籤或保存的實際模型回答。

