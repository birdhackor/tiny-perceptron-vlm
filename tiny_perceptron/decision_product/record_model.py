"""把每筆裝置當作集合中的獨立紀錄，避免先把不同裝置的欄位混在一起。"""

import re
from dataclasses import dataclass

from .model import CLS, PAD, Config, DecisionModel, Encoder, encode_example


@dataclass(frozen=True)
class RecordConfig(Config):
    row_tokens: int = 48


class RecordEncoder(Encoder):
    def __init__(self, config, embedding):
        super().__init__(config, embedding, config.row_tokens, config.state_layers)
        self.row_tokens = config.row_tokens

    def forward(self, tokens):
        if tokens.shape[1] % self.row_tokens:
            raise ValueError("record memory 長度須為固定 row_tokens 的倍數")
        b = tokens.shape[0]
        rows = tokens.reshape(-1, self.row_tokens)
        empty = rows.eq(PAD).all(-1)
        rows = rows.clone()
        # batch 補齊的空紀錄不能把 all-masked attention 送入 softmax。
        rows[empty, 0] = CLS
        values = super().forward(rows).masked_fill(empty[:, None, None], 0)
        # 紀錄間不使用順序位置；換整筆紀錄的排列只是換 memory 的排列。
        return values.reshape(b, -1, values.shape[-1])


class RecordDecisionModel(DecisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.state_encoder = RecordEncoder(config, self.embedding)


def encode_record_example(example, tokenizer):
    result = encode_example(example, tokenizer)
    rows = [row.strip() for row in re.split(r"[。.]", example.state) if row.strip()]
    if not 1 <= len(rows) <= 6:
        raise ValueError("本版支援1–6筆獨立裝置紀錄")
    flattened = []
    for row in rows:
        encoded = [CLS, *tokenizer.encode(row)]
        if len(encoded) > 48:
            raise ValueError("每筆裝置紀錄最多48tokens，不自動截斷")
        flattened.extend(encoded + [PAD] * (48 - len(encoded)))
    result["state"] = flattened
    return result


def create_model(config):
    return RecordDecisionModel(RecordConfig(**config)) if "row_tokens" in config else DecisionModel(Config(**config))


def encode_for_model(example, tokenizer, model):
    return (
        encode_record_example(example, tokenizer)
        if isinstance(model, RecordDecisionModel)
        else encode_example(example, tokenizer)
    )
