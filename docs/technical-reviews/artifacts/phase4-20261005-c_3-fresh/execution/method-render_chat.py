def render_chat(messages, tokenizer=None):
    """邊界由角色欄位建立；使用者文字不會被解析為特殊 token。"""
    tok = tokenizer or ByteTokenizer()
    ids, targets = [tok.bos_id], [IGNORE]
    roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
    for message in messages:
        role = message["role"]
        if role not in roles or not isinstance(message["content"], str):
            raise ValueError("role 必須是 user/assistant/system，content 必須是文字")
        content = tok.encode(message["content"]) + [tok.eos_id]
        ids += [roles[role]] + content
        targets += [IGNORE] + (content if role == "assistant" else [IGNORE] * len(content))
    if not any(t != IGNORE for t in targets):
        raise ValueError("這筆對話没有可監督的 assistant 答案")
    return torch.tensor(ids[:-1]), torch.tensor(targets[1:])
