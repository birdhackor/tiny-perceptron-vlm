"""研究用的既有工具結果依賴診斷；不接公開auto／實際工具往返入口。"""

import torch

from tiny_perceptron.selftrained.inference import _public_history


@torch.inference_mode()
def controlled_tool_reply(old_a, prefix, max_new_tokens=128):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("受控續答限1–128個新token")
    history = _public_history(prefix, require_user=False)
    if not history or history[-1]["role"] != "tool":
        raise ValueError("受控續答必須明示已提供的tool結果")
    # 這裡的prefix是實驗條件（含已標明的人為工具值）；沒有當前assistant標準答案。
    record = {"id": "controlled-tool-result-diagnostic", "task": "tool_reply", "messages": history}
    encoded = old_a.encoder.encode(record, generation=True, messages=history)
    if len(encoded["input_ids"]) + max_new_tokens > old_a.encoder.context:
        raise ValueError("完整受控history超過固定context，不截斷")
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=old_a.encoder.device)
    generated = old_a.model.generate(
        ids,
        modalities=[old_a.encoder.to_device(encoded["modalities"])],
        max_new_tokens=max_new_tokens,
        eos_id=old_a.encoder.tokenizer.eos_id,
    )
    tokens = generated[0].detach().cpu().tolist()
    return {
        "answer": old_a.encoder.tokenizer.decode(tokens, skip_special_tokens=True),
        "generated_ids": tokens,
        "actual_executor_used": False,
        "scope": "controlled supplied-tool-result continuation; not a real tool round",
    }
