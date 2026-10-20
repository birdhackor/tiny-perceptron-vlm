"""用合成輸入核對特徵數、角色順序與有效目標，不評估回答品質。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, expand_modalities, tone
from tiny_perceptron.selftrained.dataset import MODALITY_SLOTS, RecordEncoder
from tiny_perceptron.selftrained.model import AudioEncoder, LimitedAssistant, MODALITY_TOKENS, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


ROOT = Path.cwd()
torch.manual_seed(0)
torch.set_num_threads(1)
messages = [
    {"role": "user", "content": "接下來請用兩點回答。"},
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "1. 檢查網路。2. 重新啟動 App。"},
    {"role": "user", "content": "改成一句。"},
    {"role": "assistant", "content": "先檢查網路，再重新啟動 App。"},
]
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=512))).eval()
waveform = tone(440)
with torch.no_grad():
    features = model.audio_projector(model.audio(waveform[None]))[0]
assert features.shape == (11, 8), "單音特徵形狀不符"
ids, labels = [tok.bos_id], [IGNORE]
roles = {"user": tok.user_id, "assistant": tok.assistant_id}
for index, message in enumerate(messages):
    content = [tok.audio_id] if index == 1 else tok.encode(message["content"])
    content = content + [tok.eos_id]
    ids += [roles[message["role"]]] + content
    labels += [IGNORE] + (content if message["role"] == "assistant" else [IGNORE] * len(content))
ids, labels = torch.tensor(ids), torch.tensor(labels)
with torch.no_grad():
    embeddings, targets = expand_modalities(
        ids, labels, model.language.embedding, {tok.audio_id: features}, {tok.audio_id}
    )
    output = model(ids, labels, waveform=waveform)
expected = sum((tok.encode(m["content"]) + [tok.eos_id] for m in messages if m["role"] == "assistant"), [])
observed = targets[targets != IGNORE].tolist()
assert observed == expected, "有效目標應只有助手內容與 EOS"
assert torch.equal(output["labels"], targets), "模型回傳的標籤和展開結果不同"
audio_start = (ids == tok.audio_id).nonzero().item()
assert torch.equal(embeddings[0, audio_start:audio_start + len(features)], features), "聲音特徵未填到原佔位處"
assert all(labels[ids == tok.audio_id] == IGNORE), "聲音佔位不應是有效目標"
assert all(targets[0, audio_start - 1:audio_start + len(features) - 1] == IGNORE), "移位後的聲音目標不應計分"
text_x, text_y = render_chat(messages, tok)
assert text_y[text_y != IGNORE].tolist() == expected, "文字角色規則應給出同一組有效目標"

encoder = AudioEncoder(64).eval()
shapes = []
with torch.no_grad():
    for frames in [173, 600, 1000]:
        mel = torch.randn(frames, 40)
        tokens, scores = encoder(mel)
        assert tokens.shape == (16, 64), "真人語音入口特徵數不符"
        shapes.append({"input": list(mel.shape), "output": list(tokens.shape), "scores": list(scores.shape)})
assert MODALITY_SLOTS["audio"] == MODALITY_TOKENS["audio"] == 16, "資料與模型預留位置不一致"


class SyntheticRecordEncoder(RecordEncoder):
    """只替代讀檔步驟，保留原本的訊息編碼與監督邏輯。"""

    def _modality(self, metadata, kind):
        assert kind == "audio", "本算例只提供聲音"
        return {"kind": "audio", "values": torch.zeros(173, 40), "valid": torch.ones(173, dtype=torch.bool)}


ctok = CharacterTokenizer.build(m["content"] for m in messages)
record = {"id": "synthetic-history", "task": "audio", "messages": [dict(m) for m in messages]}
record["messages"][1]["audio"] = "synthetic.wav"
record_encoder = SyntheticRecordEncoder(ctok, Path(__file__).parent, context=512)
row = record_encoder.encode(record)
generation = record_encoder.encode(record, generation=True, messages=record["messages"][:-1])
reserved = [i for i, token in enumerate(row["input_ids"]) if token == ctok.audio_id]
generation_reserved = [i for i, token in enumerate(generation["input_ids"]) if token == ctok.audio_id]
assert len(reserved) == len(generation_reserved) == 16, "歷史中的聲音應保留相同位置數"
assert len(row["modalities"]) == len(generation["modalities"]) == 1, "聲音 payload 應保留一份"
character_targets = sum((ctok.encode(m["content"]) + [ctok.eos_id] for m in messages if m["role"] == "assistant"), [])
assert [x for x in row["labels"] if x != IGNORE] == character_targets, "成品有效目標應只有助手內容與 EOS"
assert all(label == IGNORE for label in generation["labels"][-len(ctok.encode(messages[3]["content"])) - 2:]), "新要求和生成角色不應是答案"
assistant = LimitedAssistant(SelftrainedConfig(
    vocab_size=ctok.vocab_size, width=64, layers=1, heads=2, kv_heads=1, ffn_hidden=64
)).eval()
with torch.no_grad():
    input_ids = torch.tensor([row["input_ids"]])
    actual_embeddings, _ = assistant._embeddings(input_ids, torch.ones_like(input_ids, dtype=torch.bool), [row["modalities"]])
    expected_features, _ = assistant.audio_encoder(row["modalities"][0]["values"], row["modalities"][0]["valid"])
assert torch.equal(actual_embeddings[0, reserved], expected_features), "聲音位置應換成編碼器數字"
assert ctok.decode(generation["input_ids"]) == "".join(m["content"] for m in messages[:-1]), "後續生成應保留先前文字和助手答覆"

source_paths = [
    "docs/course-review-20261019/freeze-01/sources/12.16.md",
    "docs/course-review-20261019/freeze-04.json",
    "course/figures/new-12.16-shared-history.svg",
    "tiny_perceptron/data.py", "tiny_perceptron/multimodal.py",
    "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/dataset.py",
    "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/selftrained/inference.py",
    "docs/course-review-20261019/figure-renders-04.json",
    "outputs/course-review-20261019/figure-renders-04/db6e4d05e17af1f608470e5a4e6698e9778debcc6c0fdae6f0b900e61e2218dd.png",
]
print(json.dumps({
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "seed": 0,
    "byte_history": {
        "messages": messages, "waveform_samples": waveform.numel(), "sample_rate": 16000,
        "audio_feature_shape": list(features.shape), "audio_start": audio_start,
        "input_positions_before_expansion": len(ids), "positions_after_expansion_before_shift": len(ids) - 1 + len(features),
        "shifted_embeddings_shape": list(embeddings.shape), "logits_shape": list(output["logits"].shape),
        "expected_effective_targets": expected, "observed_effective_targets": observed,
        "effective_target_count": len(observed), "effective_eos_count": observed.count(tok.eos_id),
        "audio_target_ignore_count": int((targets[0, audio_start - 1:audio_start + len(features) - 1] == IGNORE).sum()),
    },
    "selftrained_history": {
        "synthetic_payload": "全零 log-mel；沒有讀取錄音或轉寫，未執行檔案前處理",
        "encoder_shapes": shapes, "audio_slots": reserved, "generation_audio_slots": generation_reserved,
        "role_order": [m["role"] for m in messages], "generation_text": ctok.decode(generation["input_ids"]),
        "effective_target_count": len(character_targets), "effective_eos_count": character_targets.count(ctok.eos_id),
        "features_replace_reserved_embeddings": True,
    },
    "file_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
    "result": "全部斷言通過；核對形狀、內容替換、角色順序、歷史保留及標籤，未驗收回答品質。",
}, ensure_ascii=False, indent=2))
