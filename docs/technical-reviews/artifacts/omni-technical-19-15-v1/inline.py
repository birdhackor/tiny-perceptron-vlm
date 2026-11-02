import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.lesson_products.model import Config, LoopedSolver as LibraryLoopedSolver, Tokenizer

torch.set_num_threads(1)
torch.manual_seed(0)
tokenizer = Tokenizer()  # 使用固定公開字元表，不從這題fit。
character_ids = {character: i + 3 for i, character in enumerate(tokenizer.characters)}
model = LibraryLoopedSolver(Config(vocab_size=len(tokenizer))).eval()
print("全部參數", sum(p.numel() for p in model.parameters()))
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
ids = [1]  # CLS放在整段state前面。
for character in state:
    if character not in character_ids:
        raise ValueError("超出公開有限字元表：" + repr(character))
    ids.append(character_ids[character])
assert ids == tokenizer.encode(state)  # 函式庫只核對這份手寫編碼。
tokens = torch.tensor([ids])
print("字表／輸入形狀／編號", len(tokenizer), tuple(tokens.shape), ids)
with torch.no_grad():
    logits = model(tokens, loops=5, all_depths=True)
print(state, "→ 各輪答案分數的形狀", tuple(logits.shape))
losses = F.cross_entropy(logits[0], torch.tensor([2] * 5), reduction="none")
print("1–5 輪各自的 CE", [round(loss, 3) for loss in losses.tolist()], "平均", round(losses.mean().item(), 3))


# 節錄自 tiny_perceptron/lesson_products/model.py：Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff1 = nn.Linear(config.width, config.ff)
        self.ff2 = nn.Linear(config.ff, config.width)

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.norm1(hidden)).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0)
        hidden = hidden + self.out(values.transpose(1, 2).reshape(batch, length, width))
        return hidden + self.ff2(F.gelu(self.ff1(self.norm2(hidden))))


# 節錄自 tiny_perceptron/lesson_products/model.py：LoopedSolver
class LoopedSolver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.width)
        self.shared = Block(config)
        self.norm = nn.LayerNorm(config.width)
        self.answer = nn.Linear(config.width, 19)

    def forward(self, tokens, loops=3, all_depths=False, diagnostics=False):
        if type(loops) is not int or not 1 <= loops <= 6:
            raise ValueError("公開loop範圍是1–6")
        if tokens.shape[1] > self.config.max_length:
            raise ValueError("輸入超過max_length")
        original = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        hidden = original
        logits, norms = [], []
        for _ in range(loops):
            hidden = self.shared(hidden + self.config.input_injection * original, tokens != 0)
            logits.append(self.answer(self.norm(hidden[:, 0])))
            if diagnostics:
                norms.append(float(hidden.detach().float()[tokens != 0].square().mean().sqrt().item()))
        result = torch.stack(logits, dim=1) if all_depths else logits[-1]
        return (result, norms) if diagnostics else result


# 在上面兩個完整類別定義之後執行，載入同一組隨機權重作核對。
inline_model = LoopedSolver(model.config).eval()
inline_model.load_state_dict(model.state_dict())
with torch.no_grad():
    inline_logits = inline_model(tokens, loops=5, all_depths=True)
assert torch.equal(inline_logits, logits)
print("正文完整定義／函式庫最大分數差", (inline_logits - logits).abs().max().item())
