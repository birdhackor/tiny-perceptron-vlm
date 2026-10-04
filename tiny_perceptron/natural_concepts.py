"""Small CPU experiments for information loss and honest modality evaluation.

These examples do not load a pretrained model and do not measure its ability.
"""

import torch
import torch.nn.functional as F

from tiny_perceptron.multimodal import log_mel, tone


def swapped_picture():
    """Return two toy pictures with identical pooled pixels but opposite order."""
    first = torch.zeros(3, 32, 32)
    first[0, 8:16, 0:4] = 1  # red on the left
    first[2, 8:16, 4:8] = 1  # blue immediately to its right
    second = first.clone()
    second[:, :, :8] = first[:, :, :8].flip(-1)
    return first, second


def picture_order_report():
    first, second = swapped_picture()
    pooled = [F.adaptive_avg_pool2d(x[None], (4, 4)) for x in (first, second)]
    return {
        "different_pixels": int((first != second).sum()),
        "same_4_by_4_summary": bool(torch.equal(*pooled)),
        "same_pixel_sequence": bool(torch.equal(first.flatten(), second.flatten())),
    }


def thin_stroke_report():
    """A one-pixel bright stroke is blurred before any model sees the image."""
    image = torch.zeros(1, 1, 32, 32)
    image[:, :, :, 15] = 1
    small = F.interpolate(image, (4, 4), mode="area")
    return {
        "original_shape": list(image.shape[-2:]),
        "small_shape": list(small.shape[-2:]),
        "original_brightest": float(image.max()),
        "small_brightest": float(small.max()),
        "pixels_at_least_half_bright_before": int((image >= 0.5).sum()),
        "pixels_at_least_half_bright_after": int((small >= 0.5).sum()),
    }


def edit_distance(reference, prediction):
    """Levenshtein distance on Python Unicode code points, not byte tokens."""
    previous = list(range(len(prediction) + 1))
    for i, original in enumerate(reference, 1):
        current = [i]
        for j, guessed in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (original != guessed)))
        previous = current
    return previous[-1]


def text_error_report(reference, prediction):
    errors = edit_distance(reference, prediction)
    return {
        "reference": reference,
        "prediction": prediction,
        "exact": reference == prediction,
        "edits": errors,
        "reference_characters": len(reference),
        "cer": errors / len(reference) if reference else None,
    }


def audio_order_report():
    """Permuting already extracted time frames preserves the mean exactly.

    This is an experiment on representations. Reversing a real waveform and
    recomputing the spectrogram is a different operation with boundary effects.
    """
    first = torch.cat((tone(440, seconds=0.1), tone(880, seconds=0.1)))
    sequence = log_mel(first).transpose(-1, -2)
    reordered = sequence.flip(0)
    return {
        "time_frames": sequence.shape[0],
        "bands": sequence.shape[1],
        "same_time_sequence": bool(torch.equal(sequence, reordered)),
        "same_mean_with_roundoff_tolerance": bool(torch.allclose(sequence.mean(0), reordered.mean(0))),
        "mean_shape": list(sequence.mean(0).shape),
    }


def speech_stages(reference, recognized, typed_answer, spoken_answer):
    """Keep transcription accuracy separate from observed chat agreement."""
    return {
        "transcription": text_error_report(reference, recognized),
        "typed_answer": typed_answer,
        "spoken_answer": spoken_answer,
        "answers_identical": typed_answer == spoken_answer,
        "answers_correct": None,
    }
