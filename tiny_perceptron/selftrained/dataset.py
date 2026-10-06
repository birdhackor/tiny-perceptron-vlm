"""從零成品的資料邊界：只把對話與實際模態內容交給模型。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tiny_perceptron.multimodal import log_mel

ROLES = ("system", "user", "assistant", "tool")
MODALITY_SLOTS = {"image": 2, "ocr": 32, "audio": 16}
OCR_CHARACTERS = "大小上下左右開關入出人口"
PREPROCESS_VERSION = "gray-crops32-full-logmel40-v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_fingerprints(paths):
    result = {}
    for path in sorted(map(str, paths)):
        name = Path(path).name
        if name in result:
            raise ValueError(f"資料檔 basename 重複: {name}")
        result[name] = file_sha256(path)
    return result


def asset_fingerprints(records, asset_dir):
    assets = set()
    for record in records:
        for source in [record, *record["messages"]]:
            assets.update(source[key] for key in ("image", "audio") if source.get(key))
    return {relative: file_sha256(_asset_path(asset_dir, relative)) for relative in sorted(assets)}


def record_asset_paths(record, kind):
    """同時涵蓋 top-level 與合法 message-level 公開資產。"""
    return {item[kind] for item in [record, *record["messages"]] if item.get(kind)}


def read_records(paths):
    """驗證完整資料的群組切分；資料中的 metadata 不會混入 prompt。"""
    records, ids, groups = [], set(), {}
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"id", "group_id", "split", "task", "messages"}
            if not required <= record.keys():
                raise ValueError(f"{path}:{number}: 缺少 {required - record.keys()}")
            if record["id"] in ids:
                raise ValueError(f"重複 record id: {record['id']}")
            ids.add(record["id"])
            split = record["split"]
            if split == "val":
                split = record["split"] = "validation"
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"未知 split: {split}")
            group = record["group_id"]
            if group in groups and groups[group] != split:
                raise ValueError(f"group 跨 split: {group}")
            groups[group] = split
            messages = record["messages"]
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError("每列最後一則必須是 assistant target")
            for message in messages:
                if message.get("role") not in ROLES or not isinstance(message.get("content"), str):
                    raise ValueError("message 必須有合法 role 與文字 content")
                forbidden = set(message) - {"role", "content", "image", "audio", "roi", "image_layout"}
                if forbidden:
                    raise ValueError(f"message 含非輸入欄位: {forbidden}")
            records.append(record)
    return sorted(records, key=lambda item: item["id"])


def train_tokenizer(records):
    from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

    # 固定 ASCII 是格式字元，不是從 test 學得的詞表。
    texts = [m["content"] for r in records if r["split"] == "train" for m in r["messages"]]
    if not texts:
        raise ValueError("沒有 train 文本可建立 tokenizer")
    return CharacterTokenizer.build(texts, required_chars="".join(chr(i) for i in range(32, 127)) + OCR_CHARACTERS)


def _special(tokenizer, name):
    return getattr(tokenizer, f"{name}_id")


def _asset_path(root, relative):
    relative = Path(relative)
    resolved = (Path(root) / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("模態資產必須是 asset-dir 內的相對路徑")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


class RecordEncoder:
    """定長 soft-token slots 與 assistant-only 監督；不靜默截掉歷史。"""

    def __init__(self, tokenizer, asset_dir, context=512, device="cpu"):
        self.tokenizer = tokenizer
        self.asset_dir = Path(asset_dir)
        self.context = context
        self.device = torch.device(device)
        self.cache = {}

    def _modality(self, metadata, kind):
        key = json.dumps({k: metadata.get(k) for k in ("image", "audio", "roi", "image_layout")}, sort_keys=True)
        cache_key = (kind, key)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if kind == "audio":
            import soundfile as sf

            audio, sample_rate = sf.read(_asset_path(self.asset_dir, metadata["audio"]), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if len(audio) == 0 or sample_rate < 1:
                raise ValueError("音訊為空或 sample rate 無效")
            waveform = torch.from_numpy(audio.copy())
            # 全錄音，32 ms FFT / 10 ms hop；不以來源轉寫或意圖裁剪。
            values = log_mel(waveform, sample_rate, round(sample_rate * 0.032), round(sample_rate * 0.01), 40).T
            values = (values - values.mean()) / values.std().clamp(min=1e-5)
            payload = {"kind": kind, "values": values, "valid": torch.ones(len(values), dtype=torch.bool)}
        else:
            with Image.open(_asset_path(self.asset_dir, metadata["image"])) as source:
                image = source.convert("L")
                if kind == "ocr":
                    roi = metadata.get("roi")
                    if not roi or len(roi) != 4:
                        raise ValueError("OCR 需要公開指定的 roi")
                    crops = [image.crop(tuple(roi)).resize((128, 32), Image.Resampling.BILINEAR)]
                    values = torch.from_numpy(np.asarray(crops[0], dtype=np.float32).copy())[None] / 255
                    payload = {"kind": kind, "values": values}
                else:
                    slots = metadata.get("image_layout", {}).get("slots")
                    if slots is None:
                        slots = [[0, 0, image.width, image.height], [0, 0, image.width, image.height]]
                    if len(slots) != 2:
                        raise ValueError("服飾圖需要兩個公開幾何格位")
                    crops = [image.crop(tuple(box)).resize((32, 32), Image.Resampling.BILINEAR) for box in slots]
                    values = torch.stack(
                        [torch.from_numpy(np.asarray(c, dtype=np.float32).copy())[None] / 255 for c in crops]
                    )
                    coordinates = torch.tensor(
                        [[(b[0] + b[2]) / (2 * image.width), (b[1] + b[3]) / (2 * image.height)] for b in slots]
                    )
                    payload = {"kind": kind, "values": values, "coordinates": coordinates}
        self.cache[cache_key] = payload
        return payload

    def encode(self, record, *, generation=False, pretrain=False, messages=None, modality_override=None):
        tokenizer = self.tokenizer
        if messages is None:
            messages = record["messages"][:-1] if generation else record["messages"]
        ids, labels, modalities = [_special(tokenizer, "bos")], [-100], []
        user_indexes = [i for i, m in enumerate(messages) if m["role"] == "user"]
        attach_index = record.get("modality_message_index", user_indexes[-1] if user_indexes else -1)
        if (record.get("image") or record.get("audio")) and attach_index not in user_indexes:
            raise ValueError("公開 modality_message_index 必須指向保留的 user 訊息")
        for index, message in enumerate(messages):
            ids.append(_special(tokenizer, message["role"]))
            labels.append(_special(tokenizer, message["role"]) if pretrain else -100)
            metadata = dict(message)
            if index == attach_index:
                metadata.update({k: record[k] for k in ("image", "audio", "roi", "image_layout") if k in record})
            if (
                index == attach_index
                and record["task"] == "ocr"
                and metadata.get("image")
                and metadata.get("roi") is None
            ):
                raise ValueError("目前 OCR 訊息需要公開指定的 roi")
            kinds = ["ocr" if metadata.get("roi") is not None else "image"] if metadata.get("image") else []
            if metadata.get("audio"):
                kinds.append("audio")
            for kind in kinds:
                ids.extend([_special(tokenizer, kind)] * MODALITY_SLOTS[kind])
                labels.extend([-100] * MODALITY_SLOTS[kind])
                payload = self._modality(metadata, kind)
                if modality_override == "zero":
                    payload = {**payload, "values": torch.zeros_like(payload["values"])}
                modalities.append(payload)
            content_ids = tokenizer.encode(message["content"])
            if isinstance(content_ids, torch.Tensor):
                content_ids = content_ids.tolist()
            ids.extend(content_ids)
            labels.extend(content_ids if message["role"] == "assistant" or pretrain else [-100] * len(content_ids))
            ids.append(_special(tokenizer, "eos"))
            labels.append(_special(tokenizer, "eos") if message["role"] == "assistant" or pretrain else -100)
        if generation:
            ids.append(_special(tokenizer, "assistant"))
            labels.append(-100)
        if len(ids) > self.context:
            raise ValueError(f"record {record['id']} 需要 {len(ids)} tokens，超過 context={self.context}；禁止靜默截斷")
        # forward 的 labels 已對齊「下一個 token」；assistant role 預測首字。
        labels = labels[1:] + [-100]
        return {"input_ids": ids, "labels": labels, "modalities": modalities, "record": record}

    def batch(self, records, *, generation=False, pretrain=False):
        rows = [self.encode(record, generation=generation, pretrain=pretrain) for record in records]
        length = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), length), _special(self.tokenizer, "pad"), dtype=torch.long, device=self.device)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids, dtype=torch.bool)
        modalities = []
        for index, row in enumerate(rows):
            n = len(row["input_ids"])
            ids[index, :n] = torch.tensor(row["input_ids"], device=self.device)
            labels[index, :n] = torch.tensor(row["labels"], device=self.device)
            attention[index, :n] = True
            modalities.append(self.to_device(row["modalities"]))
        return {"input_ids": ids, "labels": labels, "attention_mask": attention, "modalities": modalities}

    def to_device(self, modalities):
        return [
            {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}
            for payload in modalities
        ]


def perception_loss(perceptions, records):
    """gold 只交給 loss；encoder/forward 從未收到 supervision。"""
    losses, metrics = [], {}
    for item in perceptions:
        supervision = records[item["batch_index"]].get("supervision", {})
        kind, logits = item["kind"], item["logits"]
        if kind == "image" and "vision_labels" in supervision:
            target = torch.tensor(supervision["vision_labels"], device=logits.device)
            valid = target >= 0
            if valid.any():
                losses.append(torch.nn.functional.cross_entropy(logits[valid], target[valid]))
                metrics.setdefault("image", []).extend(
                    (logits.argmax(-1)[valid] == target[valid]).detach().cpu().tolist()
                )
        elif kind == "audio" and "intent_id" in supervision:
            target = torch.tensor([supervision["intent_id"]], device=logits.device)
            losses.append(torch.nn.functional.cross_entropy(logits[None], target))
            metrics.setdefault("audio", []).append(int(logits.argmax()) == supervision["intent_id"])
        elif kind == "ocr" and "ocr_text" in supervision:
            target = torch.tensor(
                [OCR_CHARACTERS.index(c) + 1 for c in supervision["ocr_text"]], dtype=torch.long, device=logits.device
            )
            losses.append(
                torch.nn.functional.ctc_loss(
                    logits.log_softmax(-1)[:, None], target, [len(logits)], [len(target)], blank=0, zero_infinity=False
                )
            )
    return (torch.stack(losses).mean() if losses else None), metrics


def decode_ocr(logits):
    sequence = logits.argmax(-1).detach().cpu().tolist()
    chars, previous = [], None
    for symbol in sequence:
        if symbol and symbol != previous:
            chars.append(OCR_CHARACTERS[symbol - 1])
        previous = symbol
    return "".join(chars)
