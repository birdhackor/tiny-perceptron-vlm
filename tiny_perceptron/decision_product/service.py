"""有邊界的裝置決策服務：結構化入口、真實神經推論與校準拒答。"""

import hashlib
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import ACTIONS, Example, render_state
from .model import Tokenizer
from .runtime import candidate_eligibility, collate_for_model, create_model, encode_for_model


@dataclass(frozen=True)
class Device:
    id: str
    task: str
    permission: bool | None
    ready: bool | None


@dataclass(frozen=True)
class Action:
    id: str
    device: str
    action: str


@dataclass(frozen=True)
class Prediction:
    kind: str
    status: str
    selected: str | None
    model_choice: str
    confidence: float
    reason: str

    def to_dict(self):
        return asdict(self)


class DecisionEngine:
    """載入已驗收 bundle；本服務只提出建議，不送出裝置命令。"""

    def __init__(self, bundle, device="cpu"):
        root = Path(bundle)
        manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "qualified":
            raise ValueError("此 bundle 尚未通過成品驗收")
        for name, digest in manifest["files"].items():
            if name not in {"weights.safetensors", "tokenizer.json", "policy.json"}:
                raise ValueError("未知的 bundle 檔案")
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"bundle 檔案不符：{name}")
        if set(manifest["files"]) != {"weights.safetensors", "tokenizer.json", "policy.json"}:
            raise ValueError("bundle 缺少必要檔案")
        self._tokenizer = Tokenizer(json.loads((root / "tokenizer.json").read_text(encoding="utf-8")))
        self._model = create_model(manifest["config"])
        self._model.load_state_dict(load_file(root / "weights.safetensors"))
        if any(not bool(torch.isfinite(value).all()) for value in self._model.state_dict().values()):
            raise ValueError("權重含非有限值")
        self._model.to(device).eval().requires_grad_(False)
        self._policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))
        self._temperatures = self._policy["temperatures"]
        self._threshold = self._policy["threshold"]
        if set(self._temperatures) != {"choice", "info", "rubric"} or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in self._temperatures.values()
        ):
            raise ValueError("temperature 不合法")
        if (
            not isinstance(self._threshold, (int, float))
            or not math.isfinite(self._threshold)
            or not 0 < self._threshold <= 1
        ):
            raise ValueError("threshold 不合法")
        self._tokenizer_digest = manifest["files"]["tokenizer.json"]
        self._cache = OrderedDict()

    def session(self, devices, language="zh"):
        return DeviceSession(self, tuple(devices), language)

    @classmethod
    def _for_development(cls, model, tokenizer, temperatures, threshold):
        """驗收工具使用的記憶體入口；不產生或宣稱已合格的 bundle。"""
        engine = cls.__new__(cls)
        engine._model = model.eval().requires_grad_(False)
        engine._tokenizer = tokenizer
        engine._temperatures = dict(temperatures)
        engine._threshold = threshold
        engine._tokenizer_digest = hashlib.sha256(json.dumps(tokenizer.vocabulary, sort_keys=True).encode()).hexdigest()
        engine._cache = OrderedDict()
        return engine

    def _memory_key(self, state):
        digest = hashlib.sha256(state.encode("utf-8"))
        digest.update(self._tokenizer_digest.encode())
        digest.update(json.dumps(self._tokenizer.vocabulary, sort_keys=True).encode())
        digest.update(json.dumps(self._model.metadata(), sort_keys=True).encode())
        for name, tensor in self._model.state_encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tensor.device).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def _predict(self, example, use_cache=True):
        if self._model.training:
            raise ValueError("推論模型必須固定為 eval")
        row = encode_for_model(example, self._tokenizer, self._model)
        batch = collate_for_model([row], self._model, next(self._model.parameters()).device)
        memory = None
        if use_cache:
            key = self._memory_key(example.state)
            if key not in self._cache:
                self._cache[key] = self._model.state_encoder(batch["state"])
                if len(self._cache) > 32:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            memory = self._cache[key]
        logits = self._model(batch, memory=memory)[0]
        count = len(example.candidates)
        required = logits[:count] if example.kind == "info" else logits[: count + 1]
        if not bool(torch.isfinite(required).all()) or (
            example.kind == "info" and not bool(torch.isneginf(logits[count]))
        ):
            raise RuntimeError("模型輸出異常；停止此筆推論")
        probabilities = (logits / self._temperatures[example.kind]).softmax(-1)
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("模型機率異常；停止此筆推論")
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        ids = [key for key, _ in example.candidates] + ["none"]
        chosen = ids[index]
        if chosen == "none":
            return Prediction(example.kind, "abstained", None, chosen, confidence, "資訊不足或沒有合適的選項")
        if example.kind == "choice" and not candidate_eligibility(example)[index]:
            return Prediction(example.kind, "abstained", None, chosen, confidence, "候選對象與目標裝置不符")
        if example.kind == "choice" and confidence < self._threshold:
            return Prediction(example.kind, "abstained", None, chosen, confidence, "信心未達已驗證的採用門檻")
        return Prediction(
            example.kind,
            "accepted" if example.kind == "choice" else "prediction",
            chosen,
            chosen,
            confidence,
            "已通過採用門檻" if example.kind == "choice" else "僅提供判斷，不執行動作",
        )


class DeviceSession:
    def __init__(self, engine, devices, language):
        if not 3 <= len(devices) <= 6:
            raise ValueError("本版僅支援同時 3–6 個裝置")
        if type(language) is not str or language not in {"zh", "en"}:
            raise ValueError("language 僅支援 zh/en")
        for item in devices:
            if (
                not isinstance(item, Device)
                or type(item.id) is not str
                or not item.id.strip()
                or len(item.id.encode("utf-8")) > 128
            ):
                raise ValueError("裝置須使用 Device，ID 須為非空且不超過128 bytes")
            if (
                type(item.task) is not str
                or item.task not in {"record", "play", "save"}
                or any(type(value) not in {bool, type(None)} for value in (item.permission, item.ready))
            ):
                raise ValueError("task 只支援 record/play/save；欄位須為 bool 或 None")
        if len({item.id for item in devices}) != len(devices):
            raise ValueError("裝置 ID 不可重複")
        self._engine, self._devices, self._language = engine, devices, language
        # 只做 schema 與名稱對應；不計算動作、充分性或分數。
        self._slots = {item.id: chr(ord("A") + i) for i, item in enumerate(devices)}
        facts = [
            (
                ("record", "play", "save").index(item.task),
                -1 if item.permission is None else int(item.permission),
                -1 if item.ready is None else int(item.ready),
            )
            for item in devices
        ]
        self._state = render_state(
            facts,
            list(self._slots.values()),
            list(range(len(devices))),
            "chinese" if language == "zh" else "compact",
            random.Random(0),
        )

    def _slot(self, target):
        if type(target) is not str or target not in self._slots:
            raise ValueError("找不到目標裝置")
        return self._slots[target]

    def choose(self, target, candidates=None, use_cache=True):
        name = self._slot(target)
        candidates = (
            tuple(candidates) if candidates is not None else tuple(Action(action, target, action) for action in ACTIONS)
        )
        if not 1 <= len(candidates) <= 7:
            raise ValueError("候選動作須為1–7項")
        for item in candidates:
            if (
                not isinstance(item, Action)
                or type(item.id) is not str
                or not item.id.strip()
                or item.id == "none"
                or len(item.id.encode("utf-8")) > 128
            ):
                raise ValueError("候選須使用 Action，ID 須為非空、不得為 none 且不超過128 bytes")
            if (
                type(item.device) is not str
                or type(item.action) is not str
                or item.device not in self._slots
                or item.action not in ACTIONS
            ):
                raise ValueError("候選裝置不存在或動作不支援")
        if len({item.id for item in candidates}) != len(candidates) or len(
            {(item.device, item.action) for item in candidates}
        ) != len(candidates):
            raise ValueError("候選 ID 或候選語意重複")
        chinese = self._language == "zh"
        descriptions = tuple(
            (
                item.id,
                f"裝置{self._slots[item.device]}：{item.action}"
                if chinese
                else f"Device {self._slots[item.device]}: {item.action}",
            )
            for item in candidates
        )
        question = (
            f"選擇裝置{name}的動作。許可與就緒都必須已知；許可否先stop；否則就緒否wait；否則執行其任務。"
            if chinese
            else f"Choose {name}'s action. Both permission and ready must be known. No permission: stop; otherwise not ready: wait; otherwise requested task."
        )
        example = Example(self._state, question, descriptions, "", "choice", "", "service", "")
        return self._engine._predict(example, use_cache)

    def information(self, target, use_cache=True):
        name = self._slot(target)
        chinese = self._language == "zh"
        question = (
            f"裝置{name}的必要資訊齊全嗎？record需要許可與就緒；play需要許可；save需要就緒。"
            if chinese
            else f"Is required information known for {name}? record needs permission and ready; play needs permission; save needs ready."
        )
        candidates = (
            (("yes", "必要資訊齊全"), ("no", "必要資訊缺漏"))
            if chinese
            else (("yes", "Required information is sufficient"), ("no", "Required information is missing"))
        )
        return self._engine._predict(
            Example(self._state, question, candidates, "", "info", "", "service", ""), use_cache
        )

    def grade(self, target, use_cache=True):
        name = self._slot(target)
        chinese = self._language == "zh"
        question = (
            f"評分裝置{name}。兩欄須已知。record計算是的欄位數；play許可是得2否則看就緒0或1；save就緒是得2否則看許可0或1。"
            if chinese
            else f"Grade {name}. Both fields must be known. record: count yes fields; play: 2 if permission else ready(0/1); save: 2 if ready else permission(0/1)."
        )
        candidates = tuple(
            (str(i), f"裝置{name}得{i}分" if chinese else f"Device {name} has grade {i}") for i in range(3)
        )
        return self._engine._predict(
            Example(self._state, question, candidates, "", "rubric", "", "service", ""), use_cache
        )
