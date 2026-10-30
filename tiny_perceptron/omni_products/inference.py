"""新版有限產品的推論組合；舊 A/B 的權重與入口保持固定。"""

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from .av import AVConfig, AVModel, collate_av
from .choice import ChoiceConfig, ChoiceModel, Question
from .data import Event, validate_program
from .reliability import calibrated
from .routing import Request, b_sum_adapter, routing_question
from .sound import CodecConfig, ProgramTalker, SoundCodec

SOUND_MAP = {
    "circle": (Event(220, 200, 100), Event(440, 200, 0)),
    "square": (Event(880, 200, 100), Event(440, 200, 0)),
}
CHOICE_ENTITY = {
    text: entity
    for entity in ("A", "B", "C")
    for text in (
        f"Choose {entity}'s action. Require permission and ready known; no permission: stop; not ready: wait; else requested task.",
        f"For {entity}, pick action: both fields required. permission=no -> stop; ready=no -> wait; otherwise do the requested task.",
    )
}


def verify_information_binding(choice, information):
    """自動接受只用已校準服務的精確需求；自由問題仍可predict分數。"""
    entity = CHOICE_ENTITY.get(choice.text)
    if entity is None:
        return False
    expected = f"Are permission and ready both known for {entity}'s action?"
    descriptions = {"yes": "Required information is sufficient", "no": "Required information is missing"}
    if (
        information.kind != "info"
        or information.state != choice.state
        or information.text != expected
        or {item.id: item.description for item in information.candidates} != descriptions
    ):
        raise ValueError("Choice與資訊問題必須綁定同entity、permission+ready需求與同state")
    return True


def load_trained(directory, kind, device="cpu"):
    """只讀自己的 safetensors；完整 continuation .pt 另存，不用於一般推論。"""
    root = Path(directory)
    record = json.loads((root / "training-result.json").read_bytes())
    weight_path = root / "weights.safetensors"
    expected = record["files"][weight_path.name]
    if (
        weight_path.is_symlink()
        or weight_path.stat().st_size != expected["bytes"]
        or hashlib.sha256(weight_path.read_bytes()).hexdigest() != expected["sha256"]
    ):
        raise ValueError("新權重與訓練匯出指紋不同")
    if record.get("final_opened") is not False or record.get("optimizer_updates_observed") is not True:
        raise ValueError("缺少實際訓練／評估隔離來源")
    from .training_runtime import TRAINING_REVISION, source_binding

    if (
        record.get("training_revision") != TRAINING_REVISION
        or record.get("implementation_sha256") != source_binding()
        or record.get("source_unchanged_during_training") is not True
    ):
        raise ValueError("權重所用的模型／訓練實作 revision 不同；不可默默換新程式解讀舊權重")
    config = record["config"]["model"]
    if kind == "av":
        model = AVModel(AVConfig(**config))
    elif kind == "choice":
        model = ChoiceModel(ChoiceConfig(**config))
    elif kind == "codec":
        model = SoundCodec(CodecConfig(**config))
    elif kind == "talker":
        model = ProgramTalker(config["codes"], config["width"], config["hop"])
    else:
        raise ValueError("未知新模組")
    state = load_file(str(weight_path), device="cpu")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("非有限模型權重")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


class CJudge:
    def __init__(self, model, temperature=1.0, threshold=None):
        if model.training or next(model.parameters()).dtype != torch.float32:
            raise ValueError("C 服務固定 fp32 eval")
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or (threshold is not None and (not math.isfinite(threshold) or not 0 < threshold <= 1))
        ):
            raise ValueError("T／threshold 必須是合法有限數值")
        self.model, self.temperature, self.threshold = model, temperature, threshold

    def predict(self, question):
        # 當前標準答案即使在研究記錄中存在，也不經公開推論入口送入模型。
        if not isinstance(question, Question):
            raise ValueError("需明示 typed question")
        result = self.model.predict(replace(question, gold=None), 1.0)
        result["probabilities"] = calibrated(result["logits"], self.temperature)
        result["p_max"] = max(result["probabilities"])
        return result

    def judge(self, question, corresponding_information=None):
        prediction = self.predict(question)
        result = {"prediction": prediction, "accepted": False, "decision": "handoff"}
        if question.kind != "choice":
            return result
        if corresponding_information is None:
            return result
        if not verify_information_binding(question, corresponding_information):
            result["policy_supported"] = False
            return result
        information = self.predict(corresponding_information)
        accepted = (
            self.threshold is not None
            and prediction["p_max"] >= self.threshold
            and prediction["answer"] != "none"
            and information["answer"] == "yes"
        )
        result.update(information=information, accepted=accepted, decision="accept" if accepted else "handoff")
        return result

    def route(self, request):
        return self.predict(routing_question(request))


class OmniAssistant:
    def __init__(self, av, choice, codec, talker, *, old_a=None, old_b=None, temperature=1.0, threshold=None):
        if codec.config.hop != talker.hop or codec.config.codes != talker.codes:
            raise ValueError("Talker／codec 的 hop／codebook 必須配對")
        self.av, self.codec, self.talker = av.eval(), codec.eval(), talker.eval()
        self.c = CJudge(choice.eval(), temperature, threshold)
        self.old_a, self.old_b = old_a, old_b

    def old_reply(self, public_input):
        if self.old_a is None:
            raise ValueError("必須另外載入固定舊 A")
        return self.old_a.reply(
            public_input["messages"],
            task=public_input["task"],
            max_new_tokens=128,
            tools=public_input.get("tools", False),
            **public_input.get("metadata", {}),
        )

    @torch.inference_mode()
    def video(self, item, with_sound=False):
        item.validate()
        device = next(self.av.parameters()).device
        logits = self.av(collate_av([(item, 0)], device, self.av.config))[0]
        answer = ("circle", "square", "unknown")[int(logits.argmax())]
        result = {"answer": answer, "logits": logits.cpu().tolist(), "source": "new learned AV branch"}
        if with_sound and answer in SOUND_MAP:
            program = SOUND_MAP[answer]
            generated = self.talker.generate(program)
            result["sound"] = {**generated, "program": program}
            result["sound"]["waveform"] = (
                self.codec.decode_codes(generated["codes"])
                if len(generated["codes"])
                else torch.empty(0, device=device)
            )
        return result

    def sound_chunks(self, program):
        """本介面先完成 codes 再逐塊解碼；首段等待包含整段 code 生成時間。"""
        validate_program(program)
        started = time.perf_counter()
        generated = self.talker.generate(program)
        if not generated["eos"] or generated["capped"] or not len(generated["codes"]):
            raise ValueError("自由生成未正常 EOS，不能改餵參考 codes")
        code_seconds = time.perf_counter() - started
        chunks = iter(self.codec.chunks(generated["codes"]))
        for index in range(len(generated["codes"])):
            with torch.inference_mode():
                # CPU copy 使 CUDA 真完成後才記可播放時刻；離開 context 才 yield。
                samples = next(chunks).detach().cpu()
            yield {
                "index": index,
                "sample_rate": 8000,
                "samples": samples,
                "ready_seconds": time.perf_counter() - started,
                "whole_code_generation_seconds": code_seconds,
            }

    def dispatch(self, route, payload):
        if route == "none":
            return {"status": "unsupported"}
        if route == "B":
            if self.old_b is None:
                raise ValueError("必須另外載入固定舊 B")
            from tiny_perceptron.lesson_products.inference import rule_depth

            depth = rule_depth(self.old_b.auto_policy, "sum")
            return self.old_b.solve(b_sum_adapter(payload["numbers"]), loops=depth)
        if route == "C":
            questions = payload["questions"]
            return {
                "choice": self.c.judge(questions["choice"], questions["choice_info"]),
                "info": self.c.predict(questions["info"]),
                "rubric": self.c.predict(questions["rubric"]),
            }
        if route == "A":
            if "video_audio" in payload:
                return self.video(payload["video_audio"], with_sound=True)
            return self.old_reply(payload["old_A_input"])
        raise ValueError("只能手動指定 A/B/C/none")

    def auto(self, request, payload):
        if type(request) is not Request:
            raise ValueError("Auto 只收明示的公開 request")
        prediction = self.c.route(request)
        route = prediction["answer"]
        try:
            result = self.dispatch(route, payload)
            return {"route_prediction": prediction, "route": route, "result": result, "contract_error": False}
        except (KeyError, TypeError, ValueError) as error:
            # 選錯路徑不能偷看 gold task 改派；原失敗留在 Auto 全分母。
            return {
                "route_prediction": prediction,
                "route": route,
                "result": None,
                "contract_error": True,
                "error_type": type(error).__name__,
            }
