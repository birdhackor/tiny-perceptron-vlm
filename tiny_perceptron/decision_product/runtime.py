"""讓訓練、離線驗收與成品入口使用完全相同的架構及前處理。"""

from .indexed_model import ENTITY, IndexedConfig, IndexedDecisionModel, collate_indexed, encode_indexed_example
from .model import Config, DecisionModel, collate, encode_example
from .record_model import RecordConfig, RecordDecisionModel, encode_record_example


def create_model(config):
    if config.get("indexed"):
        return IndexedDecisionModel(IndexedConfig(**config))
    if "row_tokens" in config:
        return RecordDecisionModel(RecordConfig(**config))
    return DecisionModel(Config(**config))


def encode_for_model(example, tokenizer, model):
    if isinstance(model, IndexedDecisionModel):
        return encode_indexed_example(example, tokenizer)
    if isinstance(model, RecordDecisionModel):
        return encode_record_example(example, tokenizer)
    return encode_example(example, tokenizer)


def collate_for_model(rows, model, device="cpu"):
    return collate_indexed(rows, device) if isinstance(model, IndexedDecisionModel) else collate(rows, device)


def candidate_eligibility(example):
    """採用建議前核對對象；只比 ID，不讀取或計算動作規則。"""
    if example.kind != "choice":
        return [True] * len(example.candidates) + [False]
    targets = ENTITY.findall(example.question)
    if len(targets) != 1:
        raise ValueError("問題必須指向唯一裝置")
    return [ENTITY.findall(description) == targets for _, description in example.candidates] + [False]
