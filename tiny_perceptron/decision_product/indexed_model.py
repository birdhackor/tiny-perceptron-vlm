"""以已驗證的裝置識別碼索引共享記憶；欄位語意和答案仍由神經模型判斷。"""

import re
from dataclasses import dataclass

import torch

from .model import collate
from .record_model import RecordConfig, RecordDecisionModel, encode_record_example

ENTITY = re.compile(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])")


@dataclass(frozen=True)
class IndexedConfig(RecordConfig):
    indexed: bool = True


def encode_indexed_example(example, tokenizer):
    result = encode_record_example(example, tokenizer)
    targets = ENTITY.findall(example.question)
    if len(targets) != 1:
        raise ValueError("結構化問題必須明確指出唯一裝置")
    rows = [row.strip() for row in re.split(r"[。.]", example.state) if row.strip()]
    names = [ENTITY.findall(row) for row in rows]
    if any(len(name) != 1 for name in names) or len({name[0] for name in names}) != len(names):
        raise ValueError("每筆紀錄須有唯一且不重複的裝置識別碼")
    identities = [name[0] for name in names]
    if targets[0] not in identities:
        raise ValueError("目標裝置不在共享狀態中")
    # 索引只比較 ID；不查看任務、許可、就緒、gold 或候選正解。
    result["target_row"] = identities.index(targets[0])
    return result


def collate_indexed(rows, device="cpu"):
    batch = collate(rows, device)
    batch["target_rows"] = torch.tensor([row["target_row"] for row in rows], dtype=torch.long, device=device)
    return batch


class IndexedDecisionModel(RecordDecisionModel):
    def forward(self, batch, memory=None, return_auxiliary=False):
        states = batch["state"]
        # state-only encoder 仍一次讀完全部裝置，完全不看未來問題。
        if memory is None:
            memory = self.state_encoder(states)
        row = batch["target_rows"]
        width = self.config.row_tokens
        indices = row[:, None] * width + torch.arange(width, device=states.device)[None]
        if bool((indices >= states.shape[1]).any()) or bool((indices < 0).any()):
            raise ValueError("共享記憶索引超出範圍")
        query_batch = dict(batch)
        query_batch["state"] = states.gather(1, indices)
        selected_memory = memory.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))
        return super().forward(query_batch, memory=selected_memory, return_auxiliary=return_auxiliary)
