"""與神經模型獨立的固定訊號度量；正式開封前先驗解析正負例。"""

import math

import torch

from .data import Event, validate_program

ANALYZER = {
    "version": "tone-events-v1",
    "sample_rate": 8000,
    "frame": 160,
    "hop": 80,
    "fft": 4096,
    "rms_threshold": 0.05,
    "frequency_relative_tolerance": 0.05,
    "time_tolerance_ms": 30,
    "max_events": 4,
}


def active_reference_snr(reference, reconstructed, program):
    validate_program(program)
    if reference.shape != reconstructed.shape or reference.ndim != 1:
        return None
    if not torch.isfinite(reconstructed).all():
        return None
    selected = torch.zeros_like(reference, dtype=torch.bool)
    position = 0
    for event in program:
        selected[position : position + event.duration_ms * 8] = True
        position += (event.duration_ms + event.rest_ms) * 8
    if not selected.any() or reference[selected].square().sum() == 0:
        return None
    ratio = reference[selected].square().sum() / (
        (reference[selected] - reconstructed[selected]).square().sum() + 1e-12
    )
    return 10 * math.log10(max(float(ratio), 1e-30))


def analyze(waveform):
    cfg = ANALYZER
    if waveform.ndim != 1 or not 0 < len(waveform) <= 16000 or not torch.isfinite(waveform).all():
        return {"legal": False, "events": [], "reason": "shape/length/nonfinite"}
    waveform = waveform.detach().float().cpu()
    frames = torch.nn.functional.pad(waveform, (cfg["frame"] // 2, cfg["frame"] // 2)).unfold(
        0, cfg["frame"], cfg["hop"]
    )
    rms = frames.square().mean(-1).sqrt()
    spectra = torch.fft.rfft(frames * torch.hann_window(cfg["frame"]), n=cfg["fft"]).abs()
    frequency = spectra.argmax(-1).float() * cfg["sample_rate"] / cfg["fft"]
    labels = []
    for power, hz in zip(rms, frequency):
        if power < cfg["rms_threshold"]:
            labels.append(0)
        else:
            nearest = min((220, 440, 880), key=lambda target: abs(float(hz) - target))
            labels.append(nearest if abs(float(hz) - nearest) / nearest <= cfg["frequency_relative_tolerance"] else -1)
    # 20ms 分析窗跨越兩事件時可能混頻。僅處理 <=20ms 的轉換區；
    # 長的未知頻率照實保留 -1，不能把錯頻率的完整事件修成正解。
    index = 0
    while index < len(labels):
        if labels[index] != -1:
            index += 1
            continue
        end = index
        while end < len(labels) and labels[end] == -1:
            end += 1
        if end - index <= 2:
            left = labels[index - 1] if index else None
            right = labels[end] if end < len(labels) else None
            if left is not None and right is not None and left > 0 and right > 0 and left != right:
                for position in range(index, end):
                    labels[position] = left if position - index < (end - index) // 2 else right
        index = end
    events, begin, current = [], 0, labels[0]
    for i, label in enumerate(labels[1:] + [None], start=1):
        if label != current:
            if current:
                events.append(
                    {"frequency": current, "onset_ms": begin * 10, "offset_ms": min(i * 10, len(waveform) / 8)}
                )
            begin, current = i, label
    legal = len(events) <= cfg["max_events"] and all(event["frequency"] > 0 for event in events)
    return {"legal": legal, "events": events, "sample_rate": 8000, "samples": len(waveform)}


def program_success(waveform, program):
    measured = analyze(waveform)
    expected, position = [], 0
    for event in validate_program(program):
        expected.append({"frequency": event.frequency, "onset_ms": position, "offset_ms": position + event.duration_ms})
        position += event.duration_ms + event.rest_ms
    success = measured["legal"] and len(measured["events"]) == len(expected)
    if success:
        for actual, target in zip(measured["events"], expected):
            success &= abs(actual["frequency"] - target["frequency"]) / target["frequency"] <= 0.05
            success &= abs(actual["onset_ms"] - target["onset_ms"]) <= 30
            success &= abs(actual["offset_ms"] - target["offset_ms"]) <= 30
            success &= (
                abs((actual["offset_ms"] - actual["onset_ms"]) - (target["offset_ms"] - target["onset_ms"])) <= 30
            )
        for actual, following, target, next_target in zip(
            measured["events"], measured["events"][1:], expected, expected[1:]
        ):
            success &= (
                abs((following["onset_ms"] - actual["offset_ms"]) - (next_target["onset_ms"] - target["offset_ms"]))
                <= 30
            )
        success &= abs(len(waveform) / 8 - position) <= 30
    return {"success": bool(success), "measured": measured, "expected": expected}


def analyzer_controls():
    from .data import sound_wave

    target = (Event(220, 200, 100), Event(880, 100, 200), Event(440, 300))
    negative = {
        "wrong_frequency": (Event(440, 200, 100), Event(880, 100, 200), Event(440, 300)),
        "wrong_order": tuple(reversed(target)),
        "wrong_duration": (Event(220, 300, 100), Event(880, 100, 200), Event(440, 300)),
        "wrong_rest": (Event(220, 200, 200), Event(880, 100, 200), Event(440, 300)),
        "missing": target[:-1],
        "extra": (*target, Event(880, 100)),
    }
    # reversed／missing 需去掉最後 rest，保持負例本身仍是合法參考程式。
    for key in ("wrong_order", "missing"):
        values = list(negative[key])
        last = values[-1]
        values[-1] = Event(last.frequency, last.duration_ms)
        negative[key] = tuple(values)
    checks = {"positive": program_success(sound_wave(target), target)["success"]}
    for name, program in negative.items():
        checks[name] = not program_success(sound_wave(program), target)["success"]
    checks["all_silence"] = not program_success(torch.zeros_like(sound_wave(target)), target)["success"]
    checks["positive_snr"] = active_reference_snr(sound_wave(target), sound_wave(target), target) > 100
    return {"all_passed": all(checks.values()), "checks": checks, "analyzer": ANALYZER}
