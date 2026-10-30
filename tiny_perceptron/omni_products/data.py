"""自製有限世界、家族切分與可見輸入；不下載成熟模型資料。"""

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass

import torch

GENERATOR_REVISION = "omni-products-data-v2"
SPLIT_SEED = 20261029
SPLITS = ("train", "dev", "calibration", "policy", "final", "auto")


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def split_a():
    """254 個非恆定八段圖樣；時長／間距／位置／波形變體不另算家族。"""
    patterns = list(range(1, 255))
    random.Random(SPLIT_SEED).shuffle(patterns)
    return {"train": patterns[:38], "dev": patterns[38:54], "final": patterns[54:]}


@dataclass
class AVInput:
    frames: torch.Tensor
    frame_ms: torch.Tensor
    waveform: torch.Tensor
    audio_start_ms: float
    duration_ms: int
    sample_rate: int = 16000
    image_present: bool = True
    audio_present: bool = True

    def validate(self):
        frames, times, wave = self.frames, self.frame_ms, self.waveform
        if frames.ndim != 4 or frames.shape[1:] != (1, 32, 32) or not 1 <= len(frames) <= 16:
            raise ValueError("影格必須是 1–16 張灰階 32×32")
        if times.shape != (len(frames),) or wave.ndim != 1 or not len(wave):
            raise ValueError("timestamp／waveform 形狀不符")
        if not all(torch.isfinite(x).all().item() for x in (frames, times, wave)):
            raise ValueError("像素、timestamp、聲音必須有限")
        if frames.min().item() < 0 or frames.max().item() > 1 or wave.abs().max().item() > 1:
            raise ValueError("像素／聲音超出固定尺度")
        if not 0 < self.duration_ms <= 4000 or self.sample_rate != 16000:
            raise ValueError("只收 <=4 秒、16kHz 的共同時間軸")
        if not math.isfinite(self.audio_start_ms) or self.audio_start_ms < 0:
            raise ValueError("聲音起點非法")
        if times[0].item() != 0 or (times < 0).any() or not (times[1:] > times[:-1]).all():
            raise ValueError("影格須從 0 開始且嚴格遞增")
        if times[-1].item() >= self.duration_ms or self.audio_start_ms + len(wave) / 16 > self.duration_ms + 1e-6:
            raise ValueError("有效影格／聲音區間越界")
        if type(self.image_present) is not bool or type(self.audio_present) is not bool:
            raise ValueError("presence mask 必須是 bool")
        return self


def render_shape(square, center=(16, 16)):
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    x, y = x - center[0], y - center[1]
    mask = (x.abs() <= 7) & (y.abs() <= 7) if square else x.square() + y.square() <= 64
    return mask.float().unsqueeze(0)


def av_pair(pattern, variant=0, panel=0):
    """一對共用相同 pixels／waveform／序列，只移 audio_start_ms。

    pulse 在波形內的位置會變，起點 metadata 單獨不足以解所有題。
    每個 pulse 的完整 80ms 支持在單一 sample-and-hold 內，中心離切換 >=125ms。
    """
    if not 1 <= pattern <= 254 or panel not in range(4):
        raise ValueError("非法 family／panel")
    rng = random.Random(SPLIT_SEED + 1009 * pattern + 31 * variant + panel)
    duration = (3750, 4000)[panel // 2]
    spacing = (250, 500)[panel % 2]
    times = list(range(0, duration, spacing))
    bits = [(pattern >> (7 - min(7, i * 8 // len(times)))) & 1 for i in range(len(times))]
    ends = times[1:] + [duration]
    centers = [(a + b) / 2 for a, b in zip(times, ends)]
    eligible = []
    for internal_center in (85, 585):
        groups = [
            [
                i
                for i, bit in enumerate(bits)
                if bit == label and centers[i] >= internal_center and centers[i] - internal_center + 700 <= duration
            ]
            for label in (0, 1)
        ]
        if all(groups):
            eligible.append((internal_center, groups))
    internal_center, groups = rng.choice(eligible)
    # 所有 primary 皆為 700ms；不能讓長度洩漏 pulse 的內部位置。
    samples = 11200
    wave = torch.zeros(samples)
    begin, count = int((internal_center - 40) * 16), 1280
    t = torch.arange(count) / 16000
    amplitude, phase = rng.uniform(0.3, 0.8), rng.uniform(0, 2 * math.pi)
    wave[begin : begin + count] = amplitude * torch.sin(2 * math.pi * 440 * t + phase)
    center = (rng.randint(14, 18), rng.randint(14, 18))
    frames = torch.stack([render_shape(bit, center) for bit in bits])
    results = []
    for label in (0, 1):
        index = rng.choice(groups[label])
        event_center = centers[index]
        left, right = times[index], ends[index]
        assert event_center - left >= 125 and right - event_center >= 125
        item = AVInput(
            frames.clone(),
            torch.tensor(times, dtype=torch.float32),
            wave.clone(),
            event_center - internal_center,
            duration,
        )
        results.append((item.validate(), label))
    return results


def final_a_panels():
    """只宣告事前分母；development 不應呼叫 final 的 model prediction。"""
    return [{"family": pattern, "panel": i // 50, "variant": 0} for i, pattern in enumerate(split_a()["final"])]


def av_waveform_swap_control(pattern, panel=0):
    """development 診斷：相同 metadata／pixels／長度，波形位置改變使答案翻轉。"""
    pair = av_pair(pattern, panel=panel)
    first, second = pair[0][0], pair[1][0]
    # 原配對的實際 pulse 中心由各起點加相同內部中心得到。
    active = first.waveform.abs() > 1e-4
    indices = active.nonzero().flatten()
    internal = (indices[0].item() + indices[-1].item() + 1) / 32
    centers = [first.audio_start_ms + internal, second.audio_start_ms + internal]
    start = max(0, min(centers) - 85)
    length = math.ceil((max(centers) - start + 60) * 16)
    controls = []
    for label, center in enumerate(centers):
        wave = torch.zeros(length)
        offset = round((center - start - 40) * 16)
        wave[offset : offset + 1280] = 0.6 * torch.sin(2 * math.pi * 440 * torch.arange(1280) / 16000)
        controls.append(
            (AVInput(first.frames.clone(), first.frame_ms.clone(), wave, start, first.duration_ms).validate(), label)
        )
    return controls


@dataclass(frozen=True)
class Event:
    frequency: int
    duration_ms: int
    rest_ms: int = 0


def program_id(program):
    return fingerprint([asdict(event) for event in program])


def validate_program(program):
    if not 1 <= len(program) <= 4 or any(type(x) is not Event for x in program):
        raise ValueError("聲音程式須為 1–4 個 Event")
    if any(
        x.frequency not in (220, 440, 880) or x.duration_ms not in (100, 200, 300) or x.rest_ms not in (0, 100, 200)
        for x in program
    ):
        raise ValueError("頻率／時值／停頓不在公開有限契約")
    if program[-1].rest_ms != 0 or sum(x.duration_ms + x.rest_ms for x in program) > 2000:
        raise ValueError("片尾無額外停頓且總長不超過 2 秒")
    if any(a.frequency == b.frequency and a.rest_ms == 0 for a, b in zip(program, program[1:])):
        raise ValueError("同頻零停頓的兩事件不可觀察，須合併後另定 family")
    return program


def sound_splits():
    rng, programs, seen = random.Random(SPLIT_SEED + 1), [], set()
    # 兩個 AV 語義音型明示放 train；它們不是 Talker unseen-combination 成績。
    predefined = [(Event(220, 200, 100), Event(440, 200)), (Event(880, 200, 100), Event(440, 200))]
    for program in predefined:
        programs.append(program)
        seen.add(program_id(program))
    while len(programs) < 2560 + 256 + 200:
        count = rng.randint(1, 4)
        program = tuple(
            Event(
                rng.choice((220, 440, 880)),
                rng.choice((100, 200, 300)),
                rng.choice((0, 100, 200)) if i + 1 < count else 0,
            )
            for i in range(count)
        )
        try:
            validate_program(program)
        except ValueError:
            continue
        key = program_id(program)
        if key not in seen:
            programs.append(program)
            seen.add(key)
    return {"train": programs[:2560], "dev": programs[2560:2816], "final": programs[2816:]}


def sound_wave(program, phase=0.0, amplitude=0.6):
    validate_program(program)
    parts = []
    for event in program:
        n = event.duration_ms * 8
        t = torch.arange(n) / 8000
        parts.append(amplitude * torch.sin(2 * math.pi * event.frequency * t + phase))
        parts.append(torch.zeros(event.rest_ms * 8))
    return torch.cat(parts)


def c_splits():
    """三個實質 entity，各 task3×兩個欄位2²，總共 1728 家族。"""
    families = list(itertools.product(range(12), repeat=3))
    random.Random(SPLIT_SEED + 2).shuffle(families)
    sizes = {"dev": 64, "calibration": 80, "policy": 60, "final": 60, "auto": 30}
    # 先選 reliability 的固定分母，再去重實際可見事實。刪欄會隱藏 bit，
    # 不可拿另一個隱藏 bit 值作跨 split 的「新」家族。
    result, used, selected = {}, set(), set()
    for split, size in sizes.items():
        result[split] = []
        for family in families:
            if family in selected:
                continue
            index = len(result[split])
            panel = (
                "missing"
                if split in {"policy", "final"} and index >= 54
                else "removed"
                if split in {"policy", "final"} and index >= 48
                else "complete"
            )
            visible = {fingerprint(visible_c_facts(family, index, panel))}
            if not visible & used:
                result[split].append(family)
                selected.add(family)
                used.update(visible)
                if len(result[split]) == size:
                    break
        if len(result[split]) != size:
            raise ValueError("可見事實去重後不足預定 reliability 家族數")
    result["train"] = []
    for family in families:
        if family in selected:
            continue
        index = len(result["train"])
        panel = ("complete", "removed", "missing")[index % 3]
        visible = {fingerprint(visible_c_facts(family, index + variant, panel)) for variant in range(3)}
        if not visible & used:
            result["train"].append(family)
            used.update(visible)
    return result


def c_facts(family):
    if len(family) != 3 or any(type(x) is not int or x not in range(12) for x in family):
        raise ValueError("C family 需三個可見事實組")
    return [{"task": x // 4, "permission": bool((x // 2) % 2), "ready": bool(x % 2)} for x in family]


def visible_c_facts(family, index, primary_panel="complete"):
    facts = c_facts(family)
    if primary_panel == "missing":
        facts[0]["permission" if index % 2 else "ready"] = None
    elif primary_panel not in {"complete", "removed"}:
        raise ValueError("Choice panel 不合法")
    if index % 2:
        needed = (
            ("permission", "ready")
            if facts[1]["task"] == 0
            else (("permission",) if facts[1]["task"] == 1 else ("ready",))
        )
        facts[1][needed[index % len(needed)]] = None
    return facts


def manifest():
    a, sound, c = split_a(), sound_splits(), c_splits()
    entries = {
        "A": {split: [f"a-pattern-{x:02x}" for x in families] for split, families in a.items()},
        "sound": {split: [program_id(x) for x in programs] for split, programs in sound.items()},
        "C": {split: [fingerprint(x) for x in families] for split, families in c.items()},
    }
    for module, rows in entries.items():
        flat = [x for values in rows.values() for x in values]
        if len(set(flat)) != len(flat):
            raise ValueError(f"{module} 跨 split 家族重疊")
    return {
        "generator": GENERATOR_REVISION,
        "seed": SPLIT_SEED,
        "families": entries,
        "counts": {module: {split: len(rows) for split, rows in splits.items()} for module, splits in entries.items()},
        "a_final_panels": final_a_panels(),
        "lineage": "自製有限生成器；同模板／組合留出，不宣稱真實世界或成熟模型 pretrain-unseen",
    }
