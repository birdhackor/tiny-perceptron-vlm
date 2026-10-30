"""固定Omni官方processor的公開輸入；CPU可驗前處理，不載神經權重。"""

import hashlib

import torch

SYSTEM = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
MODEL = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"


def make_inputs(processor, *, text=None, audio=None, frames=None):
    if audio is not None:
        if getattr(audio, "ndim", None) != 1 or not 0 < len(audio) <= 128000:
            raise ValueError("研究音訊只收16kHz、一維且<=8秒")
    if frames is not None:
        if len(frames) != 8 or any(frame.size != (112, 112) for frame in frames) or audio is None:
            raise ValueError("研究影音固定8張112平方RGB、2fps及完整4秒音訊")
        if len(audio) != 64000:
            raise ValueError("影音音訊須完整4秒共同時間")
        content = [{"type": "video", "video": "project-authored-in-memory-frames"}, {"type": "text", "text": text}]
    elif audio is not None:
        content = [{"type": "audio", "audio": "project-authored-in-memory-waveform"}]
    elif text:
        content = [{"type": "text", "text": text}]
    else:
        raise ValueError("研究輸入不能為空")
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
    if audio is not None:
        kwargs["audio"] = [audio]
    if frames is not None:
        kwargs["videos"] = [frames]
        # 4.57.6的video processor在呼叫時使用size；舊min/max_pixels不會覆寫預設size。
        kwargs["videos_kwargs"] = {
            "fps": 2.0,
            "use_audio_in_video": True,
            "size": {"shortest_edge": 112 * 112, "longest_edge": 112 * 112},
            "do_sample_frames": False,
        }
    inputs = processor(**kwargs)
    if inputs["input_ids"].shape[-1] > 1024:
        raise ValueError("研究input超過1024 tokens")
    if frames is not None and inputs["video_grid_thw"].tolist() != [[4, 8, 8]]:
        raise ValueError("實際processor沒有產生固定112平方／8frame的grid")
    return inputs


def input_receipt(inputs):
    result = {}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            result[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
            if name in {"input_ids", "video_grid_thw", "video_second_per_grid"}:
                result[name]["values"] = tensor.tolist()
    return result
