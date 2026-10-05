"""Independent numeric/API checks and the lesson's correct_index exercise; CPU only."""
import math
import torch

assert torch.version.cuda is None and not torch.cuda.is_available()
scores = torch.tensor([[2.0, 1.0, 0.0]])
assert tuple(scores.shape) == (1, 3) and scores.device.type == "cpu"
for scale in [1.0, 10.0]:
    probabilities = (scores * scale).softmax(-1)
    confidence, prediction = probabilities.max(-1)
    reference = 1 / (1 + math.exp(-scale) + math.exp(-2 * scale))
    assert abs(confidence.item() - reference) < 1e-7
    assert abs(probabilities.sum(-1).item() - 1) < 1e-7
    assert tuple(confidence.shape) == tuple(prediction.shape) == (1,)
    assert isinstance(confidence.item(), float) and isinstance(prediction.item(), int)
    assert prediction.item() == 0 and confidence.item() < 1
    assert not (prediction.item() == 1) and prediction.item() == 0
    print("scale", scale, "probabilities", probabilities.tolist(), "reference", reference,
          "confidence", confidence.item(), "rounded", round(confidence.item(), 4),
          "correct_index_1", prediction.item() == 1, "correct_index_0", prediction.item() == 0)
    # Labels are used only after softmax: switching true index does not enter the computation.
    assert torch.equal(probabilities, (scores * scale).softmax(-1))
for temperature in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probabilities = (scores / temperature).softmax(-1)
    assert (scores / temperature).argmax(-1).item() == scores.argmax(-1).item()
    print("positive_temperature", temperature, "argmax", probabilities.argmax(-1).item(),
          "confidence", probabilities.max(-1).values.item())
absent_true_candidate = torch.tensor([[2.0, 1.0, 0.0]]).softmax(-1)
assert abs(absent_true_candidate.sum().item() - 1) < 1e-7
print("all_candidates_unsuitable_still_normalized", absent_true_candidate.sum().item())
print("PASS bounded numeric/API variants; no training or model inference")
