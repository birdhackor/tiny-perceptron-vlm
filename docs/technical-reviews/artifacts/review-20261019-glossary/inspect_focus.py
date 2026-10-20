"""定位方法與條件，收據可與首次取得的原文指紋比對。"""

import concurrent.futures
import json
import sys

from read_originals import BASE, SOURCES, fetch

FOCUS = {
    "unicode": ["a single user-perceived character", "combining character sequence", "emoji"],
    "utf8": ["one to four", "0000 0800", "U+D800"],
    "softmax": ["Softmax is defined", "sum to 1"],
    "ce": ["does not contribute", "unreduced", "unnormalized logits"],
    "norm": ["The mean and standard-deviation", "The variance is calculated", "eps"],
    "rms": ["Applies Root Mean Square", "RMS(x)"],
    "adam": ["m_t", "v_t", "state_dict"],
    "autograd": ["chain rule", "gradients are accumulated"],
    "checkpoint": ["optimizer’s state_dict", "resuming training", "state_dict"],
    "mlp": ["defines a mapping", "nonlinear", "back-propagation"],
    "attention": ["An attention function can be described", "separately and identically", "subsequent positions"],
    "vit": ["flattened 2D patches", "trainable linear projection"],
    "dpo": ["reference policy", "loss function", "binary cross-entropy"],
    "lora": ["W 0", "B A", "rank r"],
    "mtp": ["unembedding", "Each head", "trunk"],
    "ppo": ["still possible", "early stopping", "clipping"],
    "rlhf": ["Step 2: Collect comparison data, and train a reward model.", "KL", "SFT"],
    "rope": ["2D case", "block diagonal", "cos"],
    "flash": ["without materializing", "Algorithm 1", "exact attention"],
    "distill": ["transfer the knowledge", "soft targets", "softmax"],
    "passk": ["unbiased", "generate n", "at least one"],
    "rag": ["non-parametric memory", "retriever", "parametric"],
    "quant": ["quantized tensor", "scale", "Observer"],
    "audio": ["S**power", "sample rate", "mel_f.dot"],
    "jev": ["three", "Box", "Noul"],
    "jev_confidence": ["p_max", "1/n", "Noul questions"],
    "split": ["GroupKFold", "validation set", "test set"],
}
EXTRA = {
    "jev_box": ("https://docs.typesafe.ai/primitives/box.md", ["experimental", "Box", "Danger"]),
    "jev_choice": ("https://docs.typesafe.ai/primitives/choice.md", ["probabilities", "choice", "confidence"]),
    "jev_noul": ("https://docs.typesafe.ai/primitives/noul.md", ["probability", "yes", "confidence"]),
    "jev_score": ("https://docs.typesafe.ai/primitives/score.md", ["probabilities", "levels", "confidence"]),
    "calibration_full": ("https://arxiv.org/html/1706.04599v2", ["scalar parameter", "validation set", "perfect calibration"]),
    "ppo_paper": ("https://arxiv.org/html/1707.06347v2", ["pessimistic bound", "clip", "r t"]),
    "reinforce": ("https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html", ["baseline", "log-probabilities", "expected return"]),
    "teacher_forcing": ("https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html", ["teacher forcing", "next input", "target"]),
    "quant_source": ("https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/ao/quantization/observer.py", ["scale", "min_val", "max_val"]),
    "tensor": ("https://docs.pytorch.org/docs/2.14/tensors.html", ["multi-dimensional matrix", "shape", "torch.Tensor"]),
    "relu": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html", ["max", "element-wise"]),
    "kl": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html", ["log", "pointwise", "mathematically"]),
}
SOURCES.update(EXTRA)
for name, patterns in FOCUS.items():
    SOURCES[name] = (SOURCES[name][0], patterns)

if __name__ == "__main__":
    names = sys.argv[1:]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for name, (record, reading) in zip(names, pool.map(fetch, names)):
            records.append(record)
            print(json.dumps({"id": name, "reading": reading}, ensure_ascii=False))
    (BASE / ("focused-" + "-".join(names) + ".json")).write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")
