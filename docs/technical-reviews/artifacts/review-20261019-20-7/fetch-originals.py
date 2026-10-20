"""以 curl 讀上游原文，只保存小段摘錄與內容指紋。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import subprocess


SOURCES = [
    ("sft", "TRL v0.24.0", "https://raw.githubusercontent.com/huggingface/trl/v0.24.0/docs/source/sft_trainer.md", ["training_args = SFTConfig(assistant_only_loss=True)"],
     "已讀 Preprocessing and tokenization、Computing the loss、Label shifting and masking、Train on assistant messages only、Train on completion only 與 Training Vision Language Models；答案監督是可選配置，並非所有 SFT 都必須屏蔽 prompt。模板支援條件及多模態截斷風險亦有列出。"),
    ("processor", "Transformers v4.57.6", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py", ["image_inputs = self.image_processor(images=images", "num_image_tokens = image_grid_thw[index].prod() // merge_length", 'text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)', 'text_inputs = self.tokenizer(text,'],
     "已讀 Qwen3VLProcessor.__call__ 的影像處理、image_grid_thw、merge_size 平方、placeholder 展開、tokenizer 呼叫與合併回傳；圖片佔用的位置由處理後的 grid 決定，不能以原文字數算完整長度。"),
    ("decoder", "Transformers v4.57.6", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py", ["self.is_causal = True", "attention_mask = create_causal_mask(", "loss = self.loss_function(logits=logits, labels=labels"],
     "已讀 Qwen3VLTextAttention、Qwen3VLTextModel.forward 的 create_causal_mask 與 decoder_layers，以及 Qwen3VLForConditionalGeneration.forward 的 loss 呼叫；文字 decoder 有因果遮罩。視覺 encoder 的雙向 attention 不能誤當文字 decoder 的行為。"),
    ("causal", "Transformers v4.57.6", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/masking_utils.py", ["def causal_mask_function(", "return kv_idx <= q_idx"],
     "已讀 causal_mask_function 及 create_causal_mask 對 causal_mask_function 的採用；可見 key 的索引不得大於 query 索引。"),
    ("causal_loss", "Transformers v4.57.6", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/loss/loss_utils.py", ["labels = nn.functional.pad(labels, (0, 1), value=ignore_index)", "shift_labels = labels[..., 1:].contiguous()"],
     "已讀 ForCausalLMLoss、fixed_cross_entropy 與 LOSS_MAPPING；未先 shift 的 labels 在模型 loss 內移位一次，先上轉 logits 到 float，再忽略 -100。專案 toy renderer 則在函式外完成移位，兩套呼叫約定不能混用。"),
    ("torch_loss", "PyTorch v2.14.1", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py", ["ignore_index (int, optional): Specifies a target value that is ignored", "and does not contribute to the input gradient."],
     "已讀 CrossEntropyLoss 的整數類別目標、unreduced loss 公式、非忽略目標的 mean 分母與 ignore_index 說明；省略 weight、label smoothing 時 -100 不參與直接的 logits loss，不代表前文輸入不能影響後續答案。"),
    ("template", "Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/chat_template.json", ["{{- '<|im_start|>' + message.role + '\\n' }}", "{{- '<|im_end|>\\n' }}", "{{- '<|im_start|>assistant\\n' }}"],
     "已讀解析後的完整 chat_template：user 與 assistant 的角色前綴、圖片佔位、im_end 與 generation prompt。模板在 im_end 後有換行，資料包裝會保留這個格式 token；課文的小例子不是完整 Qwen token 清單。"),
]


def inspect(item):
    identifier, version, url, needles, note = item
    command = ["curl", "-fsSL", "--max-time", "45", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    if identifier == "template":
        text = json.loads(text)["chat_template"]
    if identifier == "torch_loss":
        text = text[text.index("class CrossEntropyLoss("):]
    lines = text.splitlines()
    excerpts = []
    for needle in needles:
        matches = [line.strip() for line in lines if needle in line]
        assert matches, (identifier, needle)
        excerpts.append({"contains": needle, "excerpt": matches[0]})
    return {
        "id": identifier, "url": url, "version": version,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command_argv": command, "exit_code": completed.returncode,
        "original_sha256": hashlib.sha256(raw).hexdigest(), "original_bytes": len(raw),
        "checked_original": True, "excerpts": excerpts, "inspection_note": note,
    }


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(inspect, SOURCES))
    print(json.dumps({"retrieval": "curl；第三方全文僅於記憶體讀取，未保存至 repo", "sources": records}, ensure_ascii=False, indent=2))
