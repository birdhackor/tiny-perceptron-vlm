"""用 curl 讀原始來源；只保存定位、指紋與支持主張的小段摘錄。"""

import hashlib
import html
import json
import platform
import re
import subprocess


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return raw, {"url": url, "command_argv": command, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def text(raw):
    value = raw.decode("utf-8")
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value)))


records = []
raw, entry = fetch("https://docs.pytorch.org/docs/2.8/quantization.html")
value = text(raw)
start = value.index("Quantization refers")
entry.update(version="PyTorch 2.8 官方文件", locator="Quantization 的定義段", excerpt=value[start:start + len("Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision.")], inspection_note="已讀定義與轉換流程：低位元數表示減少儲存；沒有採用該頁對硬體速度或品質的概括。概念核對不宣稱此舊版量化 API 與本機新版相容。")
records.append(entry)
raw, entry = fetch("https://raw.githubusercontent.com/huggingface/safetensors/v0.7.0/README.md")
value = raw.decode("utf-8")
start = value.index("## Format")
entry.update(version="safetensors v0.7.0 README", locator="Format", excerpt="8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header", inspection_note="已讀 Format：8-byte little-endian 長度、JSON header 與 tensor buffer。header 每個名稱另記 dtype、shape、data_offsets；本次另以公開檔實際解析核對。只摘錄長度格式。")
records.append(entry)
raw, entry = fetch("https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html")
value = text(raw)
phrases = ["both", "memory", "state_dict", "copy_"]
entry.update(version="PyTorch Tutorials 2.14.0+cu130（2026-10-20 讀取）", locator="Using torch.load(mmap=True)、Using torch.device('meta')、Using load_state_dict(assign=True)", inspection_note="已讀載入方式：一般初始化模型會先配置權重；checkpoint 的 state_dict 另占儲存；meta、mmap、assign 可避免部分配置或複製。本專案 loader 走 load_file、一般初始化和預設 load_state_dict，非這些節省記憶體路徑。")
start = value.find("copy_")
assert start != -1
entry["excerpt"] = "param_in_model.copy_(param_in_state_dict)"
records.append(entry)
raw, entry = fetch("https://aclanthology.org/D16-1139.pdf")
entry.update(version="Kim & Rush，EMNLP 2016 出版版，DOI 10.18653/v1/D16-1139", locator="Knowledge Distillation；Word-Level Knowledge Distillation；Sequence-Level Knowledge Distillation（紙本頁 1318–1320）", inspection_method="curl 取得原始 PDF bytes 並記 SHA-256；web.open 同一 HTTPS URL 讀取 PDF 解析全文。", excerpt="train the student network with cross-entropy on this new dataset", inspection_note="已讀小學生匹配教師機率分布的 cross-entropy，以及以教師 beam-search 產生的文字序列訓練學生。只支持蒸餾的兩種監督來源與較小學生的部署目的，不把論文實驗效益套用到本課成品。")
records.append(entry)
print(json.dumps({"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-10/inspect_sources.py", "environment":{"python":platform.python_version(),"transport":"curl","device":"不適用"},"accessed_on":"2026-10-20","sources":records},ensure_ascii=False,indent=2))
