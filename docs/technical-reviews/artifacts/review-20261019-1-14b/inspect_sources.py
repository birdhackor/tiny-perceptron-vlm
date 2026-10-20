"""以 curl 讀取官方原文，只保存與抽樣核對有關的短摘錄與收據。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DESTINATION = Path(__file__).resolve().parent
SPECS = [
    {
        "id": "multinomial",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html",
        "version": "PyTorch 2.14 官方文件",
        "start": "Returns a tensor where each row",
        "stop": "Example:",
        "quotes": [
            "The rows of input do not need to sum to one",
            "must be non-negative, finite and have a non-zero sum",
        ],
        "inspection_note": "已讀回傳值、列權重條件、向量輸入形狀、replacement 與 generator 參數；本節每次只抽一項，因此單次呼叫的預設不放回不會禁止下一輪再次抽到相同字。",
    },
    {
        "id": "generator",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html",
        "version": "PyTorch 2.14 官方文件",
        "start": "class torch. Generator",
        "stop": "philox_state",
        "quotes": ["Sets the seed for generating random numbers."],
        "inspection_note": "已讀 Generator 類別、device 預設 cpu、manual_seed 回傳 Generator 及種子設定；獨立 generator 的狀態提供 multinomial 使用。",
    },
    {
        "id": "randomness",
        "url": "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
        "version": "PyTorch 2.14 官方文件",
        "start": "Completely reproducible results",
        "stop": "Python",
        "quotes": ["Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."],
        "inspection_note": "已讀 Reproducibility 開頭限制及 Controlling sources of randomness／PyTorch random number generator；同 seed 的重現範圍需限定相同環境，且 CPU/GPU 不保證相同。",
    },
    {
        "id": "item",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html",
        "version": "PyTorch 2.14 官方文件",
        "start": "Returns the value of this tensor",
        "stop": "Example:",
        "quotes": ["Returns the value of this tensor as a standard Python number."],
        "inspection_note": "已讀 item 的單元素限制；本例 multinomial 回傳 long tensor，item 實際產生 Python int。",
    },
    {
        "id": "generation",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/generation_strategies.md",
        "version": "Transformers v4.57.1 tag 的官方文件原稿",
        "start": "### Greedy search",
        "stop": "### Beam search",
        "quotes": [
            "It selects the next most likely token",
            "every token with a non-zero probability has a chance to be selected",
        ],
        "inspection_note": "已讀 Greedy search 與 Sampling，以及本文件 generate.py 的逐步解碼示例：每輪由最後位置的分數挑字、串接回 input_ids、增加長度後再執行。這支持生成流程與兩種挑字方法，不支持此玩具模型的品質。",
    },
]


def main():
    receipts = []
    for spec in SPECS:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", spec["url"]]
        result = subprocess.run(command, capture_output=True, check=True)
        raw = result.stdout
        document = raw.decode("utf-8")
        visible = re.sub(r"<script\b[^>]*>.*?</script>", " ", document, flags=re.S)
        visible = re.sub(r"<style\b[^>]*>.*?</style>", " ", visible, flags=re.S)
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", visible)))
        plain = plain.replace("\u00a0", " ")
        if spec["start"] not in plain:
            raise ValueError("找不到預期的官方來源定位：" + spec["id"])
        for quote in spec["quotes"]:
            if quote not in plain:
                raise ValueError("找不到預期的官方來源摘錄：" + spec["id"])
        start = plain.index(spec["start"])
        stop = plain.find(spec["stop"], start)
        print(json.dumps({"id": spec["id"], "inspected_passage": plain[start:stop if stop > start else start + 3200]}, ensure_ascii=False))
        if spec["id"] == "generation":
            marker = "# Simple greedy decoding loop"
            position = document.index(marker)
            print(document[position:document.index("return input_ids", position) + len("return input_ids")])
        receipts.append({
            **spec,
            "command_argv": command,
            "exit_code": result.returncode,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "accessed_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
            "checked_original": True,
            "storage_note": "全文只在本次行程記憶體內處理；持久檔只保存短摘錄、定位、版本與回應雜湊。",
        })
    destination = DESTINATION / "official-source-receipts.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"sources": receipts}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
