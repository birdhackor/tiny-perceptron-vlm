class MoEFFN(nn.Module):
    """可讀的 dropless dispatch；小矩陣與 Python loop 未必比 Dense 快。"""

    def __init__(self, width, experts=4, top_k=2, hidden=None):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k 必須介於 1 與 expert 數量之間")
        self.top_k = top_k
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([DenseFFN(width, hidden) for _ in range(experts)])

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probabilities = self.router(flat).softmax(-1)
        weights, chosen = probabilities.topk(self.top_k, dim=-1)
        # top-1 不重新除以自己：否則 gate=1，任務 loss 無法給 router 梯度。
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            token_rows, slots = torch.where(chosen == i)
            if len(token_rows):
                contribution = expert(flat[token_rows]) * weights[token_rows, slots, None]
                output.index_add_(0, token_rows, contribution)
        load = F.one_hot(chosen, len(self.experts)).float().mean((0, 1))
        importance = probabilities.mean(0)
        auxiliary = len(self.experts) * (load.detach() * importance).sum()
        return output.view(shape), auxiliary, chosen
