"""核對位置選取、練習、空遮罩與官方來源；不下載模型或訓練。"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


DESTINATION = Path(__file__).resolve().parent
REVISION = str(torch.version.git_version)


def save(name, value):
    with (DESTINATION / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


valid = torch.tensor([[True, True, False, False], [False, True, True, True]])
base = torch.arange(4)
expanded = base[None].expand_as(valid)
masked = expanded.masked_fill(~valid, -1)
last = masked.amax(dim=-1)
logits = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
before = logits.clone()
selected = logits[torch.arange(2), last]
assert base.tolist() == [0, 1, 2, 3]
assert list(base[None].shape) == [1, 4]
assert expanded.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
assert expanded.stride(0) == 0
assert (~valid).tolist() == [[False, False, True, True], [True, False, False, False]]
assert masked.tolist() == [[0, 1, -1, -1], [-1, 1, 2, 3]]
assert last.tolist() == [1, 3]
assert (valid.sum(-1) - 1).tolist() == [1, 2]
assert selected.tolist() == [[5, 6, 7, 8, 9], [35, 36, 37, 38, 39]]
assert list(selected.shape) == [2, 5]
assert torch.equal(selected, torch.stack([logits[0, 1], logits[1, 3]]))
assert torch.equal(logits, before)

exercise_valid = valid.clone()
exercise_valid[0] = torch.tensor([False, False, True, True])
exercise_last = expanded.masked_fill(~exercise_valid, -1).amax(-1)
exercise_selected = logits[torch.arange(2), exercise_last]
assert exercise_last.tolist() == [3, 3]
assert exercise_selected.tolist() == [[15, 16, 17, 18, 19], [35, 36, 37, 38, 39]]
assert exercise_valid.sum(-1).tolist() == valid.sum(-1).tolist() == [2, 3]

empty_valid = torch.zeros_like(valid)
empty_last = expanded.masked_fill(~empty_valid, -1).amax(-1)
empty_selected = logits[torch.arange(2), empty_last]
assert empty_last.tolist() == [-1, -1]
assert torch.equal(empty_selected, logits[:, -1])
assert empty_selected.tolist() == [[15, 16, 17, 18, 19], [35, 36, 37, 38, 39]]

checked = 0
empty_checked = 0
for length in range(1, 9):
    masks = list(itertools.product([False, True], repeat=length))
    batch = torch.tensor(masks)
    positions = torch.arange(length)[None].expand_as(batch)
    observed = positions.masked_fill(~batch, -1).amax(-1).tolist()
    for mask, found in zip(masks, observed, strict=True):
        expected = max((i for i, value in enumerate(mask) if value), default=-1)
        assert found == expected
        checked += int(any(mask))
        empty_checked += int(not any(mask))

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-8/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_revision": REVISION,
        "device": "cpu",
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
    },
    "base": {
        "valid": valid.tolist(),
        "arange": base.tolist(),
        "unsqueezed_shape": list(base[None].shape),
        "expanded": expanded.tolist(),
        "expanded_shape": list(expanded.shape),
        "expanded_stride": list(expanded.stride()),
        "inverted": (~valid).tolist(),
        "masked": masked.tolist(),
        "last": last.tolist(),
        "count_minus_one": (valid.sum(-1) - 1).tolist(),
        "logits_dtype": str(logits.dtype),
        "logits_shape": list(logits.shape),
        "logits": logits.tolist(),
        "selected": selected.tolist(),
        "selected_shape": list(selected.shape),
        "manual_matches": True,
        "logits_unchanged": True,
    },
    "exercise": {
        "valid": exercise_valid.tolist(),
        "last": exercise_last.tolist(),
        "selected": exercise_selected.tolist(),
        "true_counts": exercise_valid.sum(-1).tolist(),
    },
    "all_false": {"last": empty_last.tolist(), "selected": empty_selected.tolist()},
    "exhaustive": {
        "sequence_lengths": list(range(1, 9)),
        "nonempty_masks": checked,
        "empty_masks": empty_checked,
        "comparison": "逐個遮罩與 Python 枚舉 True 位置的最大值精確相等；空遮罩預期為 -1。",
    },
    "result": "全部斷言通過；只驗證 CPU 確定性索引與算例，不代表模型品質。",
}
save("probe-execution.json", result)

prefix = f"https://raw.githubusercontent.com/pytorch/pytorch/{REVISION}/"
specifications = [
    (
        "torch-functions", prefix + "torch/_torch_docs.py", REVISION,
        ["torch.arange,", "torch.amax,", "torch.bitwise_not,"],
        "For bool tensors, it computes the logical NOT.",
        "讀取 arange 的半開區間、整數 dtype 規則、amax 的指定軸最大值與 bitwise_not 的 bool 語義。",
    ),
    (
        "tensor-methods", prefix + "torch/_tensor_docs.py", REVISION,
        ['"expand_as",', '"expand",', '"masked_fill",', '"masked_fill_",'],
        "Expand this tensor to the same size as :attr:`other`.",
        "讀取 expand_as、expand 的 singleton view 與零 stride 語義，及 masked_fill 的非原地填值與 True mask 條件。重複是邏輯展開，並非複製儲存。",
    ),
    (
        "advanced-indexing", prefix + "aten/src/ATen/native/TensorAdvancedIndexing.cpp", REVISION,
        ["All indexes are broadcast together", "result[i_1, ..., i_M]"],
        "All indexes are broadcast together and iterated as *one*.",
        "讀取檔頭的廣播與共同迭代公式；等長相鄰整數索引逐項配對，不形成 Cartesian product。",
    ),
    (
        "tensor-indexing", prefix + "aten/src/ATen/TensorIndexing.h", REVISION,
        ["inline Tensor applySelect", "index.is_none()"],
        "Tensor result = prev_dim_result.unsqueeze(*dim_ptr);",
        "讀取 applySelect 對負索引的處理與 is_none 分支插入軸；實際 tensor 負索引結果另以空遮罩執行核對。",
    ),
    (
        "causal-generation", "https://raw.githubusercontent.com/huggingface/transformers/v4.48.2/docs/source/en/llm_tutorial.md", "transformers v4.48.2",
        ["## Generate text", "### Wrong padding side"],
        "Since LLMs are not trained to continue from pad tokens, your input needs to be left-padded.",
        "讀取 Generate text 的 causal next-token 分布與另行選 token，及 Wrong padding side 的 padding 條件。該來源的 generate 使用左補，不能據此聲稱其現成 generate 支援本節混合左右補；本節核對的是自訂逐筆抽取分數。",
    ),
]
receipts = []
for identifier, url, version, locators, excerpt, note in specifications:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    response = subprocess.run(command, check=True, capture_output=True)
    text = response.stdout.decode("utf-8")
    assert excerpt in text
    for locator in locators:
        assert locator in text
    receipts.append({
        "id": identifier, "url": url, "version": version,
        "accessed_on": "2026-10-20", "command_argv": command,
        "exit_code": response.returncode, "retrieved_bytes": len(response.stdout),
        "retrieved_sha256": hashlib.sha256(response.stdout).hexdigest(),
        "locators_read": locators, "excerpt": excerpt, "inspection_note": note,
        "storage": "全文僅在記憶體讀取；僅保存支持主張的短摘錄、定位與取得收據。",
    })
save("source-excerpts.json", {"sources": receipts})
print(json.dumps({"nonempty_masks": checked, "empty_masks": empty_checked, "assertions": "pass", "original_sources_read": len(receipts)}, ensure_ascii=False))
